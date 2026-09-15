"""
Pipeline Orchestrator — global clip-queue batch execution (Model C).

The point of this module: on a multi-GPU cloud box (e.g. Kaggle's dual T4),
GPU rendering should never sit idle waiting on the non-GPU half of the
pipeline (download, Whisper fallback, Gemini, metadata/monetization scoring,
TTS voice-over) — for the current video OR the next one — and neither GPU
should ever stall waiting for the *other* GPU to finish the current video's
last clip before either can start the next video's clips.

Architecture: ONE clipping.render_farm.RenderFarm (num_gpus persistent
worker processes, one pinned per physical GPU) lives for the entire batch
run. N prep-worker threads keep preparing upcoming sources concurrently
(I/O/network/API bound — real threads are enough, these release the GIL);
as soon as a source's clip list is ready, every one of its clips is pushed
onto the farm's single shared task queue, tagged with that source's index.
A collector loop drains completed clip results from the farm — whichever
GPU finishes next, from whichever video — and finalizes a video (manifest,
QC, dedup) the moment all of *its* clips have come back, regardless of what
order videos finish in.

This removes the two barriers a naive per-video pipeline has:
  1. no barrier between a video's own clips finishing on both GPUs before
     the next video can start (there's no "next video" boundary at the GPU
     level anymore — it's one continuous queue for the whole batch)
  2. no barrier between prep and render (prep for source N+1..N+k runs
     concurrently with render for whichever sources are already queued)

See clipping.render_farm for the scheduling rationale (greedy list
scheduling over many small tasks ≈ optimal makespan for a fixed worker
count).
"""

import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List

from ..render_farm import RenderFarm
from ..runner import _resolve_render_gpu_count
from .batch_runner import build_item_config

_PREP_SENTINEL = object()


def run_pipelined_batch(base_cfg, sources: List, prep_workers: int = 2, queue_depth: int = 2) -> List[dict]:
    """
    Run the full pipeline over multiple sources on one shared, batch-lifetime
    RenderFarm, so both GPUs stay continuously busy across video boundaries.

    Args:
        base_cfg: Base SimpleNamespace config (from config.build_config()).
        sources: List of InputSource to process.
        prep_workers: How many sources to prepare concurrently (download/
            transcribe-fallback/Gemini/TTS). 2-3 is usually enough to keep
            the render farm continuously fed.
        queue_depth: Max number of prepared-but-not-yet-submitted sources
            held at once, bounding local disk usage from downloaded source
            videos sitting unrendered. Keep small (1-3).

    Returns:
        List of per-item result dicts, in the same shape as
        clipping.phase1.batch_runner.run_batch(): {source, title, status,
        outputs_dir, clips_rendered | error}.
    """
    from ..runner import finalize_video_render, prepare_pipeline, record_clip_result, setup_video_render

    total = len(sources)
    if total == 0:
        return []

    num_gpus = _resolve_render_gpu_count(getattr(base_cfg, "render_gpus", "auto"))
    workers_per_gpu = max(1, int(getattr(base_cfg, "render_workers_per_gpu", 1)))
    print(
        f"🖥️  Starting shared render farm: {num_gpus} GPU(s) x {workers_per_gpu} worker(s) "
        f"for the whole batch of {total} source(s)."
    )
    farm = RenderFarm(num_gpus, workers_per_gpu)
    farm.start()

    prep_ready_queue: "queue.Queue" = queue.Queue(maxsize=max(1, queue_depth))
    results: List[dict] = [None] * total
    video_ctxs: dict = {}
    pending_clip_count: dict = {}
    # Indices that no longer need any more farm results, i.e. failed during
    # prep/setup, had zero clips to render, or all clips are in (dispatched
    # for finalize — not necessarily finalized YET, see finalize_executor
    # below). Drives loop termination instead of scanning `results`, since
    # finalize (manifest/QC/dedup) now runs asynchronously and may still be
    # in flight when a video's last clip result arrives.
    accounted_for: set = set()
    # Finalize (manifest write, quality control, dedup record) is CPU-bound
    # local work — it must never delay draining the next completed clip off
    # the farm's result queue, so it runs in its own small thread pool
    # instead of inline in the collector loop below.
    finalize_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="finalize")

    def _prep_one(index: int, source):
        item_cfg = build_item_config(base_cfg, source, index + 1)
        try:
            prepared = prepare_pipeline(item_cfg)
            prep_ready_queue.put((index, source, item_cfg, prepared, None))
        except Exception as e:  # noqa: BLE001 — surfaced via results, batch must continue
            prep_ready_queue.put((index, source, item_cfg, None, e))

    def _prep_feeder():
        with ThreadPoolExecutor(max_workers=max(1, prep_workers), thread_name_prefix="prep") as ex:
            futures = [ex.submit(_prep_one, i, s) for i, s in enumerate(sources)]
            for f in futures:
                f.result()  # propagate unexpected errors from _prep_one itself (shouldn't happen — it catches)
        prep_ready_queue.put(_PREP_SENTINEL)

    feeder_thread = threading.Thread(target=_prep_feeder, name="prep-feeder", daemon=True)
    feeder_thread.start()

    def _consume_prep_ready(block: bool):
        """Pull everything currently available from prep_ready_queue (or
        block for the first item if `block`), submitting each source's clips
        to the shared farm as soon as it's ready. Returns False once the
        feeder sentinel has been seen (no more sources coming)."""
        first = True
        while True:
            try:
                item = prep_ready_queue.get(block=(block and first), timeout=None if (block and first) else 0)
            except queue.Empty:
                return True
            first = False

            if item is _PREP_SENTINEL:
                return False

            index, source, item_cfg, prepared, err = item
            if err is not None:
                print(f"❌ Batch item {index + 1}/{total} failed during prep: {err}")
                results[index] = {
                    "source": source.source,
                    "title": source.title,
                    "status": "failed",
                    "outputs_dir": item_cfg.outputs_dir,
                    "error": str(err),
                }
                accounted_for.add(index)
                continue

            try:
                video_ctx = setup_video_render(prepared)
            except Exception as e:  # noqa: BLE001 — one item's setup failure must not abort the batch
                print(f"❌ Batch item {index + 1}/{total} failed during render setup: {e}")
                results[index] = {
                    "source": source.source,
                    "title": source.title,
                    "status": "failed",
                    "outputs_dir": item_cfg.outputs_dir,
                    "error": str(e),
                }
                accounted_for.add(index)
                continue

            video_ctx["_source"] = source
            video_ctx["_item_cfg"] = item_cfg
            video_ctxs[index] = video_ctx

            clips = video_ctx["clips_to_render"]
            if not clips:
                # Everything was already checkpointed — nothing to submit,
                # finalize immediately (async, same as the normal path).
                finalize_executor.submit(_finish_video, index)
                accounted_for.add(index)
                continue

            pending_clip_count[index] = len(clips)
            print(
                f"📤 Batch {index + 1}/{total}: queuing {len(clips)} klip dari '{source.source}' "
                f"ke render farm bersama."
            )
            for klip in clips:
                farm.submit_clip(
                    index,
                    klip["rank"],
                    klip,
                    item_cfg.pilihan_rasio,
                    video_ctx["file_glitch_ts"],
                    prepared["data_segmen"],
                    item_cfg,
                    video_ctx["video_encoder"],
                    prepared["diarization_data"],
                )

    def _finish_video(index: int):
        video_ctx = video_ctxs[index]
        source = video_ctx["_source"]
        item_cfg = video_ctx["_item_cfg"]
        try:
            manifest = finalize_video_render(video_ctx)
            results[index] = {
                "source": source.source,
                "title": source.title,
                "status": "success",
                "outputs_dir": item_cfg.outputs_dir,
                "clips_rendered": len(manifest),
            }
        except Exception as e:  # noqa: BLE001 — one item's finalize failure must not abort the batch
            print(f"❌ Batch item {index + 1}/{total} failed during finalize: {e}")
            results[index] = {
                "source": source.source,
                "title": source.title,
                "status": "failed",
                "outputs_dir": item_cfg.outputs_dir,
                "error": str(e),
            }

    # Prime the farm with the first source(s) before entering the
    # results-driven collector loop below.
    prep_still_coming = _consume_prep_ready(block=True)

    while len(accounted_for) < total:
        # Drain any newly-prepared sources without blocking, so their clips
        # join the shared queue as soon as they're ready rather than waiting
        # for the current round of results.
        if prep_still_coming:
            prep_still_coming = _consume_prep_ready(block=False)

        if not any(pending_clip_count.get(i, 0) > 0 for i in video_ctxs if i not in accounted_for):
            # Nothing in flight on the farm — must be waiting on prep.
            if prep_still_coming:
                prep_still_coming = _consume_prep_ready(block=True)
                continue
            break  # nothing pending, nothing coming, and not yet `total` accounted for — bail rather than hang

        video_id, rank, klip, hasil_render, err = farm.get_result()
        if err is not None:
            hasil_render = {"status": "failed", "error": str(err), "rank": rank}
        record_clip_result(video_ctxs[video_id], klip, hasil_render)
        pending_clip_count[video_id] -= 1

        if pending_clip_count[video_id] == 0:
            # Dispatch finalize (manifest/QC/dedup) in the background and
            # immediately go back to draining the farm's result queue —
            # this video's finalize must never delay the next clip result.
            finalize_executor.submit(_finish_video, video_id)
            accounted_for.add(video_id)

    feeder_thread.join()
    farm.shutdown()
    finalize_executor.shutdown(wait=True)  # block only here, once, until every finalize has landed in `results`

    return results
