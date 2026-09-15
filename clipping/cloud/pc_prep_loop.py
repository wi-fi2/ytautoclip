"""
clipping.cloud.pc_prep_loop — PC entry point.

Runs clipping.runner.prepare_pipeline() (download, captions/Whisper,
Gemini, metadata/monetization scoring, diarization, voice-over TTS) over a
backlog of sources, writes a portable manifest per video into the local
Kaggle "inbox" working dir, and pushes a batched Kaggle Dataset version
once enough clips have accumulated — so the Kaggle GPU session is only
started once there's real backlog, and runs dry as little as possible.

This process never touches the GPU render half of the pipeline at all —
that only happens inside clipping.cloud.kaggle_consumer, on Kaggle.

Usage:
    python -m clipping.cloud.pc_prep_loop --backlog-file urls.txt \\
        --inbox-dir kaggle_inbox --inbox-dataset yourname/ytclipper-inbox \\
        --outbox-dir kaggle_outbox_pull --outbox-dataset yourname/ytclipper-outbox \\
        --threshold-clips 60
"""

import argparse
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor

from . import kaggle_transport
from .job_store import JobStore
from .manifest import to_manifest
from ..config import build_config
from ..phase1.batch_runner import build_item_config
from ..phase1.input_handler import InputManager
from ..runner import prepare_pipeline


def _prep_one(base_cfg, source, index, jobs_dir, store: JobStore):
    item_cfg = build_item_config(base_cfg, source, index)
    video_id = os.path.basename(item_cfg.outputs_dir.rstrip(os.sep))
    try:
        prepared = prepare_pipeline(item_cfg)
    except Exception as e:  # noqa: BLE001 — one source's prep failure must not abort the loop
        print(f"❌ Prep failed for {source.source}: {e}")
        store.mark_failed(video_id, str(e))
        return

    manifest = to_manifest(prepared, video_id)
    manifest_path = os.path.join(jobs_dir, f"{video_id}.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False)

    clip_count = len(manifest["hasil_json"])
    store.add_ready(video_id, source.source, clip_count, manifest_path)
    print(f"✅ Prepared '{source.source}' -> {clip_count} clips queued as {video_id}")

    # PC-local prep working dir is no longer needed — everything that
    # matters is in the manifest (or embedded/re-fetchable on Kaggle).
    # NEVER touch item_cfg.file_video_asli directly here: for local_file
    # sources, build_item_config() points that path AT the user's original
    # file (outside outputs_dir) — only outputs_dir is the isolated,
    # safe-to-delete per-item scratch directory batch_runner created.
    shutil.rmtree(item_cfg.outputs_dir, ignore_errors=True)


def _maybe_push_inbox(inbox_dir: str, inbox_dataset: str, store: JobStore, threshold_clips: int) -> None:
    ready = store.get_by_status("READY")
    ready_clips = sum(r["clip_count"] for r in ready)
    if ready_clips < threshold_clips or not ready:
        return

    print(f"🚀 Threshold reached: {ready_clips} clips ready ({len(ready)} videos) — pushing inbox batch.")
    kaggle_transport.push_dataset_version(
        inbox_dir, f"batch: {len(ready)} videos, {ready_clips} clips"
    )
    store.mark_sent([r["video_id"] for r in ready])
    print("🟢 Inbox pushed. Start the Kaggle notebook now if it isn't already running.")


def _reconcile_outbox(outbox_dataset: str, pull_dir: str, store: JobStore, final_output_dir: str, jobs_dir: str) -> None:
    kaggle_transport.pull_dataset(outbox_dataset, pull_dir, quiet=True)
    if not os.path.isdir(pull_dir):
        return

    for video_id in os.listdir(pull_dir):
        video_dir = os.path.join(pull_dir, video_id)
        if not os.path.isdir(video_dir):
            continue
        # A "DONE" marker means the Kaggle consumer finished this video's
        # finalize_video_render() — only then is it safe to reconcile.
        if not os.path.exists(os.path.join(video_dir, "DONE")):
            continue

        dest = os.path.join(final_output_dir, video_id)
        os.makedirs(dest, exist_ok=True)
        for name in os.listdir(video_dir):
            if name == "DONE":
                continue
            shutil.move(os.path.join(video_dir, name), os.path.join(dest, name))

        store.mark_completed(video_id)
        manifest_path = os.path.join(jobs_dir, f"{video_id}.json")
        if os.path.exists(manifest_path):
            os.remove(manifest_path)  # prune from the next inbox snapshot
        print(f"📥 Reconciled completed video {video_id} -> {dest}")


def main():
    p = argparse.ArgumentParser(description="PC prep loop: prepare videos, push batches to Kaggle.")
    p.add_argument("--backlog-file", required=True, help="Text/CSV file of source URLs, one per line.")
    p.add_argument("--inbox-dir", default="kaggle_inbox")
    p.add_argument("--inbox-dataset", required=True, help="e.g. yourname/ytclipper-inbox")
    p.add_argument("--outbox-dataset", required=True, help="e.g. yourname/ytclipper-outbox")
    p.add_argument("--outbox-pull-dir", default="kaggle_outbox_pull")
    p.add_argument("--final-output-dir", default="outputs/cloud_batches")
    p.add_argument("--db-path", default="kaggle_inbox/../jobs.sqlite")
    p.add_argument("--threshold-clips", type=int, default=60)
    p.add_argument("--outbox-poll-seconds", type=int, default=120)
    p.add_argument("--prep-workers", type=int, default=2)
    args, base_argv = p.parse_known_args()

    base_cfg = build_config(base_argv + ["--batch-file", args.backlog_file])
    manager = InputManager(batch_file=args.backlog_file)
    valid_sources, invalid_sources = manager.validate_all()
    for src, err in invalid_sources:
        print(f"⚠️  Skipping invalid source {src.source}: {err}")

    jobs_dir = os.path.join(args.inbox_dir, "jobs")
    os.makedirs(jobs_dir, exist_ok=True)
    store = JobStore(args.db_path)

    print(f"📋 {len(valid_sources)} valid source(s) in backlog. Preparing with {args.prep_workers} worker(s)...")

    last_outbox_poll = 0.0
    with ThreadPoolExecutor(max_workers=max(1, args.prep_workers), thread_name_prefix="prep") as ex:
        futures = [
            ex.submit(_prep_one, base_cfg, source, i + 1, jobs_dir, store)
            for i, source in enumerate(valid_sources)
        ]
        for f in futures:
            f.result()
            _maybe_push_inbox(args.inbox_dir, args.inbox_dataset, store, args.threshold_clips)
            if time.time() - last_outbox_poll > args.outbox_poll_seconds:
                _reconcile_outbox(
                    args.outbox_dataset, args.outbox_pull_dir,
                    store, args.final_output_dir, jobs_dir,
                )
                last_outbox_poll = time.time()

    # Flush any remainder below threshold, and do a final outbox reconcile.
    ready = store.get_by_status("READY")
    if ready:
        print(f"🚀 Flushing remaining {sum(r['clip_count'] for r in ready)} clips as final batch.")
        kaggle_transport.push_dataset_version(args.inbox_dir, "final batch flush")
        store.mark_sent([r["video_id"] for r in ready])

    _reconcile_outbox(
        args.outbox_dataset, args.outbox_pull_dir,
        store, args.final_output_dir, jobs_dir,
    )
    print("✅ Prep loop finished.", store.counts())


if __name__ == "__main__":
    main()
