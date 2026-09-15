"""
GPU/accelerator resolution shared across the pipeline.

CUDA and Apple Silicon Metal (via PyTorch's MPS backend) are both
"GPU available" cases; this module centralizes the priority order
(CUDA > MPS > CPU) so every PyTorch-based component (YOLO face detection,
Pyannote diarization) picks the same device instead of each reimplementing
its own detection — and, importantly, Ultralytics/YOLO does NOT auto-select
MPS on its own even when available (it must be passed explicitly), which is
why callers need this helper rather than relying on library auto-detection.
"""

import functools
import os


@functools.lru_cache(maxsize=1)
def resolve_torch_device() -> str:
    """
    Best GPU device string for PyTorch-based components: 'cuda', 'mps', or
    'cpu'. Cached — device availability doesn't change during a run.

    Respects CUDA_VISIBLE_DEVICES: in a worker process that has been pinned
    to a single physical GPU (e.g. by cuda_device_count()-based render
    parallelism), 'cuda' here correctly refers to that process's cuda:0,
    which is whatever physical device CUDA_VISIBLE_DEVICES restricted it to.
    """
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


@functools.lru_cache(maxsize=1)
def cuda_device_count() -> int:
    """
    Number of physical CUDA GPUs visible to this process (0 if none/no
    CUDA). Used to size per-GPU render worker pools (e.g. dual T4 on Kaggle).
    """
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.device_count()
    except Exception:
        pass
    return 0


def pin_process_to_gpu(gpu_index: int) -> None:
    """
    Restrict this process to a single physical GPU by index, via
    CUDA_VISIBLE_DEVICES. Must be called before any CUDA context is created
    in the process (e.g. at the very start of a multiprocessing worker),
    since torch/CUDA read this env var once at init.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    resolve_torch_device.cache_clear()
    cuda_device_count.cache_clear()


def device_label(device: str) -> str:
    return {
        "cuda": "GPU (CUDA)",
        "mps": "GPU (Apple Metal / MPS)",
        "cpu": "CPU",
    }.get(device, device)
