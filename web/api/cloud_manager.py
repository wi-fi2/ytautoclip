"""
web.api.cloud_manager — supervises the PC-side cloud prep loop
(clipping.cloud.pc_prep_loop) as a background subprocess, and reads its
state (clipping.cloud.job_store.JobStore) for the dashboard's Cloud Batch
page.

Runs the prep loop as a real OS subprocess rather than an in-process
asyncio task: it's a long-lived, multi-threaded loop (thread pool +
periodic Kaggle API pushes/pulls) not designed for cooperative
cancellation, so a subprocess gives clean start/stop via process
termination, and its own stdout/stderr become the log the dashboard tails.

Note: this tracker is in-memory only (mirrors the existing
web.api.worker._settings_env pattern) — if the API server restarts while a
prep loop is running, the subprocess keeps running but the "running" state
shown here resets. Restarting the API server does not kill the prep loop.
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timezone

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CLOUD_DIR = os.path.join(BASE_DIR, "cloud_batch")
BACKLOG_FILE = os.path.join(CLOUD_DIR, "backlog.txt")
LOG_FILE = os.path.join(CLOUD_DIR, "prep_loop.log")
DB_PATH = os.path.join(CLOUD_DIR, "jobs.sqlite")
INBOX_DIR = os.path.join(BASE_DIR, "kaggle_inbox")
OUTBOX_PULL_DIR = os.path.join(CLOUD_DIR, "outbox_pull")
FINAL_OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "cloud_batches")

os.makedirs(CLOUD_DIR, exist_ok=True)

_state: dict = {
    "process": None,  # subprocess.Popen | None
    "log_fh": None,
    "started_at": None,
    "inbox_dataset": None,
    "outbox_dataset": None,
    "threshold_clips": None,
    "prep_workers": None,
}


def is_running() -> bool:
    proc = _state["process"]
    return proc is not None and proc.poll() is None


def start(inbox_dataset: str, outbox_dataset: str, threshold_clips: int, prep_workers: int) -> None:
    if is_running():
        raise RuntimeError("Prep loop is already running.")
    if not os.path.exists(BACKLOG_FILE) or os.path.getsize(BACKLOG_FILE) == 0:
        raise RuntimeError("Backlog is empty — add at least one URL first.")

    log_fh = open(LOG_FILE, "a", encoding="utf-8")
    log_fh.write(f"\n\n===== Prep loop started {datetime.now(timezone.utc).isoformat()} =====\n")
    log_fh.flush()

    cmd = [
        sys.executable, "-m", "clipping.cloud.pc_prep_loop",
        "--backlog-file", BACKLOG_FILE,
        "--inbox-dir", INBOX_DIR,
        "--inbox-dataset", inbox_dataset,
        "--outbox-dataset", outbox_dataset,
        "--outbox-pull-dir", OUTBOX_PULL_DIR,
        "--final-output-dir", FINAL_OUTPUT_DIR,
        "--db-path", DB_PATH,
        "--threshold-clips", str(threshold_clips),
        "--prep-workers", str(prep_workers),
    ]
    proc = subprocess.Popen(cmd, cwd=BASE_DIR, stdout=log_fh, stderr=subprocess.STDOUT)
    _state.update(
        {
            "process": proc,
            "log_fh": log_fh,
            "started_at": datetime.now(timezone.utc),
            "inbox_dataset": inbox_dataset,
            "outbox_dataset": outbox_dataset,
            "threshold_clips": threshold_clips,
            "prep_workers": prep_workers,
        }
    )


def stop() -> None:
    proc = _state["process"]
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    if _state["log_fh"]:
        _state["log_fh"].close()
    _state["process"] = None
    _state["log_fh"] = None


def add_urls(urls: list[str]) -> int:
    """Append new URLs to the backlog (dedup against what's already
    there). Returns the total backlog size after adding."""
    existing = set()
    if os.path.exists(BACKLOG_FILE):
        with open(BACKLOG_FILE, "r", encoding="utf-8") as f:
            existing = {line.strip() for line in f if line.strip()}
    new_urls = [u.strip() for u in urls if u.strip() and u.strip() not in existing]
    if new_urls:
        with open(BACKLOG_FILE, "a", encoding="utf-8") as f:
            for u in new_urls:
                f.write(u + "\n")
    return len(get_backlog())


def get_backlog() -> list[str]:
    if not os.path.exists(BACKLOG_FILE):
        return []
    with open(BACKLOG_FILE, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def clear_backlog() -> None:
    open(BACKLOG_FILE, "w", encoding="utf-8").close()


def job_counts() -> dict:
    if not os.path.exists(DB_PATH):
        return {}
    from clipping.cloud.job_store import JobStore

    return JobStore(DB_PATH).counts()


def recent_log(lines: int = 100) -> list[str]:
    if not os.path.exists(LOG_FILE):
        return []
    with open(LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
        all_lines = f.readlines()
    return [line.rstrip("\n") for line in all_lines[-lines:]]


def status() -> dict:
    proc = _state["process"]
    running = is_running()
    return {
        "running": running,
        "pid": proc.pid if (proc is not None and running) else None,
        "started_at": _state["started_at"],
        "inbox_dataset": _state["inbox_dataset"],
        "outbox_dataset": _state["outbox_dataset"],
        "threshold_clips": _state["threshold_clips"],
        "backlog_count": len(get_backlog()),
        "job_counts": job_counts(),
        "recent_log": recent_log(60),
    }
