"""
clipping.cloud.manifest — portable job manifest for a prepared video.

Bridges clipping.runner.prepare_pipeline()'s output (a `prepared` dict
tied to this machine's local paths/cfg) into something JSON-serializable
that can travel through a Kaggle Dataset, and back into an equivalent
`prepared` dict on the Kaggle side.

Design choice: the manifest carries metadata only, NEVER the source
video — the Kaggle consumer re-downloads it directly via source_url with
clipping.engine.download_video(), exactly like a normal local run would.
That keeps every dataset push/pull to kilobytes instead of gigabytes.
Small assets (voiceover TTS clips) are embedded as base64 since they're
KB-scale. Secrets (API keys) are never included — the Kaggle side
re-supplies its own via Kaggle Secrets, never trusting the manifest for
those.
"""

import base64
import json
import os

# cfg fields that are either secrets or tied to this machine's local
# filesystem — never serialize these into the manifest.
_EXCLUDED_CFG_FIELDS = {
    "api_key_gemini", "hf_token", "pexels_api_key", "api_key_nvidia",
    "outputs_dir", "file_video_asli", "url_youtube", "dedup_db_dir",
    "batch_file",
}


def _json_safe_cfg(cfg) -> dict:
    """vars(cfg) filtered to secret-free, JSON-serializable fields."""
    safe = {}
    for key, value in vars(cfg).items():
        if key in _EXCLUDED_CFG_FIELDS:
            continue
        try:
            json.dumps(value)
        except TypeError:
            continue
        safe[key] = value
    return safe


def _embed_voiceover_audio(hasil_json: list[dict]) -> list[dict]:
    """Copy of hasil_json with each clip's local voiceover audio file (if
    any) replaced by base64-embedded bytes, so it survives the trip
    through the manifest without a separate file transfer."""
    out = []
    for klip in hasil_json:
        klip = dict(klip)
        vo = klip.get("voiceover")
        if vo and vo.get("audio_path") and os.path.exists(vo["audio_path"]):
            with open(vo["audio_path"], "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode("ascii")
            vo = dict(vo)
            vo.pop("audio_path", None)
            vo["audio_b64"] = audio_b64
            klip["voiceover"] = vo
        out.append(klip)
    return out


def to_manifest(prepared: dict, video_id: str) -> dict:
    """Build a portable manifest dict from prepare_pipeline()'s output."""
    cfg = prepared["cfg"]
    return {
        "video_id": video_id,
        "source_url": getattr(cfg, "url_youtube", None),
        "source_platform": getattr(cfg, "source_platform", "youtube"),
        "use_checkpoint": prepared["use_checkpoint"],
        "cfg_overrides": _json_safe_cfg(cfg),
        "hasil_json": _embed_voiceover_audio(prepared["hasil_json"]),
        "data_segmen": prepared["data_segmen"],
        "diarization_data": prepared["diarization_data"],
    }


def _restore_voiceover_audio(hasil_json: list[dict], workdir: str) -> list[dict]:
    out = []
    vo_dir = os.path.join(workdir, "voiceover")
    for klip in hasil_json:
        klip = dict(klip)
        vo = klip.get("voiceover")
        if vo and vo.get("audio_b64"):
            os.makedirs(vo_dir, exist_ok=True)
            audio_path = os.path.join(vo_dir, f"vo_rank_{klip['rank']}.mp3")
            with open(audio_path, "wb") as f:
                f.write(base64.b64decode(vo["audio_b64"]))
            vo = dict(vo)
            vo.pop("audio_b64", None)
            vo["audio_path"] = audio_path
            klip["voiceover"] = vo
        out.append(klip)
    return out


def from_manifest(manifest: dict, workdir: str, secrets: dict | None = None) -> dict:
    """
    Reconstruct a `prepared`-shaped dict (as prepare_pipeline() would
    return) from a manifest downloaded on the Kaggle side, rooted at
    `workdir` for this video's local files.

    Does NOT download the source video — the caller (kaggle_consumer.py)
    does that separately via engine.download_video() before rendering,
    since that's I/O, not data reconstruction.

    `secrets` supplies THIS machine's own hf_token/pexels_api_key/etc —
    never trust the manifest for those (it never carries them anyway).
    """
    from types import SimpleNamespace

    from ..phase1.checkpoint import CheckpointManager

    os.makedirs(workdir, exist_ok=True)
    outputs_dir = os.path.join(workdir, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    cfg_dict = dict(manifest["cfg_overrides"])
    cfg_dict.update(
        {
            "outputs_dir": outputs_dir,
            "file_video_asli": os.path.join(workdir, "source.mp4"),
            "url_youtube": manifest["source_url"],
            "source_platform": manifest.get("source_platform", "youtube"),
        }
    )
    cfg_dict.update(secrets or {})
    cfg = SimpleNamespace(**cfg_dict)

    hasil_json = _restore_voiceover_audio(manifest["hasil_json"], workdir)

    custom_hook_path = None
    if getattr(cfg, "hook_source", None):
        from .. import hook_manager

        custom_hook_path = hook_manager.download_custom_hook(cfg)

    use_checkpoint = manifest["use_checkpoint"]
    checkpoint = CheckpointManager(cfg.outputs_dir, source_key=cfg.url_youtube)

    return {
        "cfg": cfg,
        "checkpoint": checkpoint,
        "use_checkpoint": use_checkpoint,
        "dedup": None,  # dedup is a PC-side prep-time concern, not relevant on the render side
        "dedup_source": None,
        "hasil_json": hasil_json,
        "data_segmen": manifest["data_segmen"],
        "diarization_data": manifest["diarization_data"],
        "custom_hook_path": custom_hook_path,
    }
