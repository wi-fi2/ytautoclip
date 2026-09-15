"""
Persistent multi-GPU clip render farm — global work-queue scheduling.

Two long-lived worker processes (one pinned per physical GPU) drain clip
render tasks from ONE shared queue fed continuously by every video in a
batch, instead of being torn down and rebuilt per video. This removes the
per-video synchronization barrier that the old per-video GPU pool approach
imposed (wait for both GPUs to finish video N's clips before starting video
N+1's) — see clipping/phase1/pipeline_orchestrator.py for how batch mode
wires prep -> this farm -> finalize.

Scheduling rationale: with many small, similar-duration tasks (clip
renders) and a small number of workers, a single shared queue drained
greedily converges to near-optimal makespan (Graham's list-scheduling
bound: within (1 + (m-1)/m) of optimal for m workers, tighter still when
task durations are similar). That beats both:
  - splitting per video with a barrier between videos (idle bubbles from
    per-video clip-count imbalance + no overlap across the video boundary)
  - pinning whole videos to GPUs (coarse-grained imbalance — one long video
    stalls a GPU while the other races through short ones, and a
    single-video pin loses all intra-video parallelism)
"""

import multiprocessing

from . import gpu as gpu_mod

_STOP = None  # sentinel enqueued once per worker to shut it down


def _gpu_worker_loop(gpu_index, task_queue, result_queue):
    """Entry point for one persistent GPU worker process. Pins to one
    physical GPU once at startup, then loops pulling clip tasks off the
    shared queue until told to stop."""
    gpu_mod.pin_process_to_gpu(gpu_index)
    from . import studio  # imported post-pin, same as the old per-video pool workers

    while True:
        task = task_queue.get()
        if task is _STOP:
            break

        video_id, rank, klip, rasio, glitch_ts, data_segmen, cfg, video_encoder, diarization_data = task
        try:
            hasil_render = studio.proses_klip(
                rank,
                klip,
                rasio,
                glitch_ts,
                data_segmen,
                cfg,
                video_encoder,
                diarization_data=diarization_data,
            )
            result_queue.put((video_id, rank, klip, hasil_render, None))
        except Exception as e:  # noqa: BLE001 — must reach the collector, not crash the worker
            result_queue.put((video_id, rank, klip, None, e))


class RenderFarm:
    """
    Owns `num_gpus * workers_per_gpu` persistent GPU worker processes for
    the lifetime of a batch run (or a single video, for the non-batch path).

    `workers_per_gpu` defaults to 1 deliberately: more render processes
    sharing one physical GPU is NOT guaranteed to raise throughput — NVENC
    has a limited number of hardware encode sessions, and concurrent
    processes on the same T4 also contend for NVDEC, VRAM, memory
    bandwidth, and host-side CPU decode/filtering. Bump this only after
    benchmarking clips/hour at 1 vs 2+ workers/GPU on the actual box —
    optimize for that metric, not for raw process count or %GPU-utilization.

    Usage:
        farm = RenderFarm(num_gpus)
        farm.start()
        for clip in clips:
            farm.submit_clip(video_id, rank, klip, ...)
        for _ in range(len(clips)):
            video_id, rank, klip, hasil_render, err = farm.get_result()
        farm.shutdown()
    """

    def __init__(self, num_gpus: int, workers_per_gpu: int = 1):
        self.num_gpus = max(1, num_gpus)
        self.workers_per_gpu = max(1, workers_per_gpu)
        self._ctx = multiprocessing.get_context("spawn")
        self.task_queue = self._ctx.Queue()
        self.result_queue = self._ctx.Queue()
        self._workers = []

    def start(self) -> None:
        for gpu_index in range(self.num_gpus):
            for worker_slot in range(self.workers_per_gpu):
                proc = self._ctx.Process(
                    target=_gpu_worker_loop,
                    args=(gpu_index, self.task_queue, self.result_queue),
                    daemon=True,
                    name=f"gpu-render-worker-{gpu_index}-{worker_slot}",
                )
                proc.start()
                self._workers.append(proc)

    def submit_clip(
        self, video_id, rank, klip, rasio, glitch_ts, data_segmen, cfg, video_encoder, diarization_data
    ) -> None:
        """Thread-safe: multiple prep threads may call this concurrently —
        multiprocessing.Queue.put() is internally synchronized."""
        self.task_queue.put(
            (video_id, rank, klip, rasio, glitch_ts, data_segmen, cfg, video_encoder, diarization_data)
        )

    def get_result(self):
        """Blocking get of the next completed clip result, from whichever
        GPU finished first: (video_id, rank, klip, hasil_render, error)."""
        return self.result_queue.get()

    def shutdown(self) -> None:
        for _ in self._workers:
            self.task_queue.put(_STOP)
        for proc in self._workers:
            proc.join()
