"""
clipping.cloud.kaggle_consumer — Kaggle notebook entry point.

Run this inside the Kaggle GPU notebook (T4 x2, internet ON, KAGGLE_USERNAME/
KAGGLE_KEY set as attached Secrets, the inbox dataset attached as input).

For each video manifest found in the inbox dataset: downloads the source
video directly from its source_url (never from the PC — see
clipping.cloud.manifest), reconstructs the same `prepared`-shaped dict
clipping.runner.prepare_pipeline() would have produced, then feeds the
EXISTING setup_video_render() -> RenderFarm -> record_clip_result() ->
finalize_video_render() pipeline exactly as clipping.phase1.pipeline_orchestrator
does for a live batch — the only thing that changed is where `prepared`
comes from.

While the session runs, it keeps re-pulling the inbox dataset for newly
uploaded manifests, and periodically pushes a batched outbox dataset
version with everything finished so far, so a single Kaggle session can
keep consuming the PC's continuous prep output for its whole duration
instead of processing one fixed snapshot and idling.
"""

import argparse
import json
import os
import shutil
import time

from . import kaggle_transport
from .manifest import from_manifest
from .. import engine
from ..render_farm import RenderFarm
from ..runner import _resolve_render_gpu_count, finalize_video_render, record_clip_result, setup_video_render


def _ensure_dataset_metadata(local_dir: str, dataset_ref: str) -> None:
    """dataset_create_version() requires local_dir to already contain a
    dataset-metadata.json pointing at the target dataset id — write one if
    it's missing (fresh ephemeral Kaggle working dir every session)."""
    meta_path = os.path.join(local_dir, "dataset-metadata.json")
    if os.path.exists(meta_path):
        return
    title = dataset_ref.split("/", 1)[-1]
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"title": title, "id": dataset_ref, "licenses": [{"name": "CC0-1.0"}]}, f)


def _load_secrets() -> dict:
    """Read this machine's own hf_token/pexels_api_key from Kaggle Secrets
    (falling back to plain env vars if run outside a Kaggle notebook, e.g.
    for local testing). Never trust the manifest for these."""
    secrets = {"hf_token": "", "pexels_api_key": ""}
    try:
        from kaggle_secrets import UserSecretsClient

        client = UserSecretsClient()
        for key, secret_name in [("hf_token", "HF_TOKEN"), ("pexels_api_key", "PEXELS_API_KEY")]:
            try:
                secrets[key] = client.get_secret(secret_name)
            except Exception:
                pass
    except ImportError:
        secrets["hf_token"] = os.environ.get("HF_TOKEN", "")
        secrets["pexels_api_key"] = os.environ.get("PEXELS_API_KEY", "")
    return secrets


def _hydrate_video(manifest_path: str, work_root: str, secrets: dict) -> dict:
    """manifest.json -> prepared dict, with the source video actually
    downloaded to disk (the one step manifest.from_manifest() doesn't do,
    since that's I/O rather than data reconstruction)."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    workdir = os.path.join(work_root, manifest["video_id"])
    prepared = from_manifest(manifest, workdir, secrets=secrets)
    cfg = prepared["cfg"]

    engine.download_video(
        cfg.url_youtube,
        cfg.file_video_asli,
        getattr(cfg, "use_dlp_subs", False),
        getattr(cfg, "download_source_height", "max"),
        source_platform=getattr(cfg, "source_platform", "youtube"),
    )
    return prepared


def run_session(
    inbox_dataset: str,
    outbox_dataset: str,
    outbox_dir: str,
    work_root: str = "/kaggle/working/cloud_render",
    inbox_pull_dir: str = "/kaggle/working/cloud_inbox_pull",
    poll_seconds: int = 90,
    idle_timeout_seconds: int = 900,
    outbox_push_every_seconds: int = 300,
) -> None:
    secrets = _load_secrets()
    num_gpus = _resolve_render_gpu_count("auto")
    print(f"🖥️  Starting Kaggle render farm: {num_gpus} GPU(s).")
    farm = RenderFarm(num_gpus)
    farm.start()

    os.makedirs(outbox_dir, exist_ok=True)
    os.makedirs(work_root, exist_ok=True)
    _ensure_dataset_metadata(outbox_dir, outbox_dataset)

    already_submitted: set[str] = set()
    video_ctxs: dict = {}
    pending_clip_count: dict = {}
    last_outbox_push = time.time()
    last_new_work_seen = time.time()

    def _pull_and_submit_new_jobs():
        kaggle_transport.pull_dataset(inbox_dataset, inbox_pull_dir, quiet=True)
        jobs_dir = os.path.join(inbox_pull_dir, "jobs")
        if not os.path.isdir(jobs_dir):
            return 0
        new_count = 0
        for fname in sorted(os.listdir(jobs_dir)):
            if not fname.endswith(".json"):
                continue
            video_id = fname[: -len(".json")]
            if video_id in already_submitted:
                continue
            manifest_path = os.path.join(jobs_dir, fname)
            try:
                prepared = _hydrate_video(manifest_path, work_root, secrets)
                video_ctx = setup_video_render(prepared)
            except Exception as e:  # noqa: BLE001 — one bad job must not kill the session
                print(f"❌ Failed to hydrate/setup {video_id}: {e}")
                already_submitted.add(video_id)
                continue

            already_submitted.add(video_id)
            video_ctxs[video_id] = video_ctx
            clips = video_ctx["clips_to_render"]
            if not clips:
                _finalize_and_stage(video_id, video_ctx)
                continue

            pending_clip_count[video_id] = len(clips)
            for klip in clips:
                farm.submit_clip(
                    video_id,
                    klip["rank"],
                    klip,
                    prepared["cfg"].pilihan_rasio,
                    video_ctx["file_glitch_ts"],
                    prepared["data_segmen"],
                    prepared["cfg"],
                    video_ctx["video_encoder"],
                    prepared["diarization_data"],
                )
            new_count += 1
            print(f"📤 Submitted {video_id}: {len(clips)} clips to the render farm.")
        return new_count

    def _finalize_and_stage(video_id: str, video_ctx: dict) -> None:
        finalize_video_render(video_ctx)
        cfg = video_ctx["prepared"]["cfg"]
        dest = os.path.join(outbox_dir, video_id)
        os.makedirs(dest, exist_ok=True)
        for name in os.listdir(cfg.outputs_dir):
            src = os.path.join(cfg.outputs_dir, name)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(dest, name))
        with open(os.path.join(dest, "DONE"), "w") as f:
            f.write("1")
        shutil.rmtree(os.path.join(work_root, video_id), ignore_errors=True)
        print(f"✅ Finalized {video_id} -> outbox staged.")

    # Initial fill before entering the collector loop.
    _pull_and_submit_new_jobs()

    while True:
        if any(pending_clip_count.get(v, 0) > 0 for v in video_ctxs):
            video_id, rank, klip, hasil_render, err = farm.get_result()
            if err is not None:
                hasil_render = {"status": "failed", "error": str(err), "rank": rank}
            record_clip_result(video_ctxs[video_id], klip, hasil_render)
            pending_clip_count[video_id] -= 1
            last_new_work_seen = time.time()

            if pending_clip_count[video_id] == 0:
                _finalize_and_stage(video_id, video_ctxs[video_id])
        else:
            # Nothing in flight — poll the inbox for more work rather than
            # blocking forever, so a still-running PC prep loop can keep
            # feeding this session for its whole duration.
            time.sleep(poll_seconds)
            found = _pull_and_submit_new_jobs()
            if found:
                last_new_work_seen = time.time()
            elif time.time() - last_new_work_seen > idle_timeout_seconds:
                print("💤 No new work and nothing in flight for a while — ending session.")
                break

        if time.time() - last_outbox_push > outbox_push_every_seconds:
            staged = [d for d in os.listdir(outbox_dir) if os.path.exists(os.path.join(outbox_dir, d, "DONE"))]
            if staged:
                print(f"📮 Pushing outbox batch: {len(staged)} finished video(s).")
                kaggle_transport.push_dataset_version(outbox_dir, f"{len(staged)} videos finished")
            last_outbox_push = time.time()

    # Final flush.
    staged = [d for d in os.listdir(outbox_dir) if os.path.exists(os.path.join(outbox_dir, d, "DONE"))]
    if staged:
        print(f"📮 Final outbox push: {len(staged)} finished video(s).")
        kaggle_transport.push_dataset_version(outbox_dir, f"final: {len(staged)} videos")

    farm.shutdown()
    print("🏁 Session complete.")


def main():
    p = argparse.ArgumentParser(description="Kaggle GPU render consumer.")
    p.add_argument("--inbox-dataset", required=True, help="e.g. yourname/ytclipper-inbox")
    p.add_argument("--outbox-dataset", required=True, help="e.g. yourname/ytclipper-outbox")
    p.add_argument("--outbox-dir", default="/kaggle/working/cloud_outbox")
    p.add_argument("--poll-seconds", type=int, default=90)
    p.add_argument("--idle-timeout-seconds", type=int, default=900)
    p.add_argument("--outbox-push-every-seconds", type=int, default=300)
    args = p.parse_args()

    run_session(
        inbox_dataset=args.inbox_dataset,
        outbox_dataset=args.outbox_dataset,
        outbox_dir=args.outbox_dir,
        poll_seconds=args.poll_seconds,
        idle_timeout_seconds=args.idle_timeout_seconds,
        outbox_push_every_seconds=args.outbox_push_every_seconds,
    )


if __name__ == "__main__":
    main()
