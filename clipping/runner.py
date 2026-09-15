"""
clipping.runner — Pipeline Orchestrator

Maps to Cell 4 (Execute) of the notebook.
Orchestrates the full clip generation pipeline.
"""

import json
import os

from . import diarization as diarization_mod
from . import engine, metadata, studio, hook_manager, voiceover
from . import gpu as gpu_mod
from .phase1.checkpoint import CheckpointManager
from .render_farm import RenderFarm


def _resolve_render_gpu_count(render_gpus) -> int:
    """
    Resolve the --render-gpus setting ('auto' or an int) against the number
    of physical CUDA GPUs actually visible, so per-clip rendering can be
    spread across e.g. both T4s on a Kaggle dual-GPU notebook.
    """
    visible = gpu_mod.cuda_device_count()
    if visible <= 1:
        return 1
    if render_gpus in (None, "auto"):
        return visible
    try:
        requested = int(render_gpus)
    except (TypeError, ValueError):
        return visible
    return max(1, min(requested, visible))


def prepare_pipeline(cfg) -> dict:
    """
    Run the non-GPU-render half of the pipeline: dedup check, download,
    transcribe, Gemini analysis, metadata/monetization scoring, diarization,
    and voice-over generation.

    This is the network/API/CPU-bound half — it deliberately does NOT touch
    per-clip GPU rendering, so it can run concurrently (in a thread pool) with
    another video's render_pipeline() call to keep the GPUs continuously fed
    instead of idling while this stage waits on downloads/Whisper/Gemini/TTS.
    See clipping.phase1.pipeline_orchestrator for the producer/consumer
    wiring used in batch mode.

    Parameters
    ----------
    cfg : SimpleNamespace
        Configuration object from ``config.build_config()``.

    Returns
    -------
    dict
        Prepared-job state to pass into render_pipeline().
    """
    use_checkpoint = getattr(cfg, "enable_checkpoint", True)
    checkpoint = CheckpointManager(cfg.outputs_dir, source_key=getattr(cfg, "url_youtube", None))
    if getattr(cfg, "reset_checkpoint", False):
        checkpoint.reset()

    # Step 0 — Deduplication check (warns only; never silently blocks a run —
    # use --force-reprocess to suppress the warning once you've decided to proceed)
    dedup = None
    dedup_source = None
    if getattr(cfg, "url_youtube", None):
        from .phase1.deduplication import DeduplicationManager
        from .phase1.input_handler import InputSource

        dedup_dir = getattr(cfg, "dedup_db_dir", None) or os.path.join(
            os.path.dirname(cfg.outputs_dir.rstrip(os.sep)), "data"
        )
        dedup = DeduplicationManager(dedup_dir)
        try:
            dedup_source = InputSource(source=cfg.url_youtube)
            existing = dedup.check_video_duplicate(dedup_source)
        except ValueError:
            existing = None

        if existing and not getattr(cfg, "force_reprocess", False):
            print(
                f"⚠️  Video ini sudah pernah diproses pada {existing['processed_date']} "
                f"(output: {existing['output_dir']}). Melanjutkan proses ulang — "
                f"gunakan --force-reprocess untuk menghilangkan peringatan ini."
            )

    # Step 1 — Download
    source_platform = getattr(cfg, "source_platform", "youtube")
    if (
        use_checkpoint
        and checkpoint.is_step_complete("download")
        and os.path.exists(cfg.file_video_asli)
    ):
        print(f"⏭️  [1/3] Download dilewati (checkpoint: sudah selesai) — {cfg.file_video_asli}")
    else:
        engine.download_video(
            cfg.url_youtube,
            cfg.file_video_asli,
            getattr(cfg, "use_dlp_subs", False),
            getattr(cfg, "download_source_height", "max"),
            source_platform=source_platform,
        )
        if use_checkpoint:
            checkpoint.mark_step_complete("download", {"file": cfg.file_video_asli})

    # Step 2 — Transcribe
    transkrip_lengkap = ""
    data_segmen = []

    if use_checkpoint and checkpoint.is_step_complete("transcribe"):
        cached = checkpoint.get_step_data("transcribe")
        transkrip_lengkap = cached.get("transkrip_lengkap", "")
        data_segmen = cached.get("data_segmen", [])
        print("⏭️  [2/3] Transkripsi dilewati (checkpoint: sudah selesai)")

    if not transkrip_lengkap or not data_segmen:
        import glob

        # Mencari file json3 apapun (karena bahasanya bisa .id.json3 atau .en.json3)
        json3_files = glob.glob(cfg.file_video_asli.replace(".mp4", ".*.json3"))
        file_json3 = json3_files[0] if json3_files else None

        # Only run YouTube JSON3 subtitle search for YouTube sources
        if source_platform == "youtube":
            if (
                getattr(cfg, "use_dlp_subs", False)
                and file_json3
                and os.path.exists(file_json3)
            ):
                transkrip_lengkap, data_segmen = engine.parse_youtube_json3_subs(
                    file_json3, max_words_per_subtitle=cfg.max_kata_per_subtitle
                )
                if transkrip_lengkap and data_segmen:
                    print(
                        f"✅ Berhasil memparsing subtitle dari YouTube ({os.path.basename(file_json3)}), melewati proses Whisper."
                    )

        if not transkrip_lengkap or not data_segmen:
            transkrip_lengkap, data_segmen = engine.transcribe_video(
                cfg.file_video_asli,
                max_words_per_subtitle=cfg.max_kata_per_subtitle,
                model_size=cfg.whisper_model,
                device=cfg.whisper_device,
                compute_type=cfg.whisper_compute_type,
            )

        if use_checkpoint:
            checkpoint.mark_step_complete(
                "transcribe",
                {"transkrip_lengkap": transkrip_lengkap, "data_segmen": data_segmen},
            )

    # Step 3 — Gemini AI analysis
    gemini_output_path = os.path.join(cfg.outputs_dir, "gemini_response.json")
    
    if getattr(cfg, "load_gemini_json", False) and os.path.exists(gemini_output_path):
        print(f"\n🔄 [3/3] Memuat data AI ({cfg.ai_provider}) dari file lokal: {gemini_output_path}")
        with open(gemini_output_path, "r", encoding="utf-8") as f:
            hasil_json = json.load(f)
    else:
        hasil_json = engine.analyze_with_ai(transkrip_lengkap, cfg)
        
        # Save raw gemini json for future loading/reproduction
        with open(gemini_output_path, "w", encoding="utf-8") as f:
            json.dump(hasil_json, f, indent=4, ensure_ascii=False)
        print(f"💾 Raw AI response tersimpan di: {gemini_output_path}")

    if use_checkpoint:
        checkpoint.mark_step_complete("ai_analysis", {"path": gemini_output_path})

    # Step 4 — Metadata normalisation
    hasil_json = metadata.normalize_and_validate(hasil_json)
    metadata.print_preview(hasil_json)

    metadata_path = os.path.join(cfg.outputs_dir, "metadata_preview.json")
    metadata.save_metadata_preview(hasil_json, path=metadata_path)

    # Step 4.5 — Monetization scoring (optional, feature-flagged)
    from .phase1.candidate_scoring import score_candidates, write_candidates_artifact

    if getattr(cfg, "enable_monetization_scoring", True):
        hasil_json = score_candidates(
            hasil_json,
            data_segmen,
            quality_weight=getattr(cfg, "monetization_quality_weight", 0.7),
            monetization_weight=getattr(cfg, "monetization_weight", 0.3),
            cfg=cfg,
        )
        # Re-rank by combined_score, reassign sequential rank (mirrors
        # metadata.py's own sort-then-reassign-rank pattern).
        hasil_json = sorted(hasil_json, key=lambda x: x["combined_score"], reverse=True)
        for idx, item in enumerate(hasil_json):
            item["rank"] = idx + 1
        print(
            f"💰 Monetization scoring applied "
            f"(quality={cfg.monetization_quality_weight}, monetization={cfg.monetization_weight}) "
            f"— clips re-ranked by combined_score."
        )

    candidates_path = os.path.join(cfg.outputs_dir, "clip_candidates.json")
    write_candidates_artifact(hasil_json, candidates_path)
    print(f"💾 Clip candidates saved to {candidates_path}")

    # Step 5 — Diarization (split-screen / camera-switch)
    diarization_data = None
    if (
        (getattr(cfg, "use_split_screen", False) and cfg.split_trigger == "diarization")
        or getattr(cfg, "use_camera_switch", False)
    ) and studio._is_vertical_ratio(cfg.pilihan_rasio):
        try:
            mode_label = (
                "Split-Screen"
                if getattr(cfg, "use_split_screen", False)
                else "Camera-Switch"
            )
            print(f"\n🎙️ [{mode_label}] Menjalankan speaker diarization...")
            audio_path = cfg.file_video_asli.replace(".mp4", "_audio.wav")
            diarization_mod.extract_audio(cfg.file_video_asli, audio_path)
            num_speakers_arg = getattr(cfg, "diarization_num_speakers", 2)
            min_spk = None
            max_spk = None

            if str(num_speakers_arg).lower() == "auto":
                max_faces = studio.estimate_speaker_count_from_video(
                    cfg.file_video_asli, cfg
                )
                num_speakers_arg = "auto"
                min_spk = max(1, max_faces)
                max_spk = min_spk + 2
                print(f"   ℹ️ Instruksi Pyannote: {min_spk} hingga {max_spk} speaker.")

            diarization_data = diarization_mod.run_diarization(
                audio_path,
                hf_token=cfg.hf_token,
                num_speakers=num_speakers_arg,
                min_speakers=min_spk,
                max_speakers=max_spk,
            )
            # Clean up temp audio
            if os.path.exists(audio_path):
                os.remove(audio_path)
        except Exception as e:
            print(f"⚠️ Diarization gagal: {e}")
            print("   Fallback ke mode render biasa (tanpa split-screen).")
            diarization_data = None

    custom_hook_path = None
    if getattr(cfg, "hook_source", None):
        print("\n🎣 Mengunduh sumber klip Hook kustom...")
        custom_hook_path = hook_manager.download_custom_hook(cfg)

    # Step 5.5 — Generate Voice-Over (if enabled)
    if getattr(cfg, "voiceover", False):
        print(f"\n🎙️ Meng-generate Voice-Over untuk {len(hasil_json)} klip...")
        for klip in hasil_json:
            try:
                # 1. Generate commentary script from snippet
                start = float(klip["start_time"])
                end = float(klip["end_time"])
                # Extract transcript snippet for this time range
                snippet_lines = []
                for seg in data_segmen:
                    if float(seg["end"]) > start and float(seg["start"]) < end:
                        # Support both Whisper format (has 'text') and YouTube JSON3 (only 'words')
                        seg_text = seg.get("text") or " ".join(w["word"] for w in seg.get("words", []))
                        if seg_text:
                            snippet_lines.append(seg_text)
                snippet_text = " ".join(snippet_lines)

                script = voiceover.generate_commentary_script(
                    snippet_text,
                    cfg,
                    style=cfg.voiceover_style,
                    language=cfg.voiceover_lang,
                    length=cfg.voiceover_length,
                )

                if script:
                    # 2. Synthesize TTS
                    audio_path, vo_segments = voiceover.synthesize_voice(
                        script,
                        cfg.voiceover_voice,
                        cfg.outputs_dir,
                        str(klip["rank"])
                    )
                    
                    if os.path.exists(audio_path):
                        klip["voiceover"] = {
                            "script": script,
                            "audio_path": audio_path,
                            "segments": vo_segments,
                            "voice": cfg.voiceover_voice
                        }

            except Exception as e:
                print(f"   ⚠️ Gagal generate voice-over untuk Rank {klip['rank']}: {e}")

    return {
        "cfg": cfg,
        "checkpoint": checkpoint,
        "use_checkpoint": use_checkpoint,
        "dedup": dedup,
        "dedup_source": dedup_source,
        "hasil_json": hasil_json,
        "data_segmen": data_segmen,
        "diarization_data": diarization_data,
        "custom_hook_path": custom_hook_path,
    }


def setup_video_render(prepared: dict) -> dict:
    """
    Lightweight per-video GPU setup that must happen once before this
    video's clip tasks can be submitted to a RenderFarm: encoder detection
    (brief NVENC test encodes), glitch-transition preparation, and figuring
    out which clips still need rendering vs. are already checkpointed.

    Does NOT render any clips itself — callers submit video_ctx["clips_to_render"]
    to a RenderFarm (shared across the whole batch in pipelined-batch mode,
    or a throwaway one-video farm in the single-URL path) and later call
    finalize_video_render() once every clip's result has come back.

    Returns
    -------
    dict
        video_ctx — carries everything finalize_video_render() and the
        RenderFarm submission loop need.
    """
    cfg = prepared["cfg"]
    checkpoint = prepared["checkpoint"]
    use_checkpoint = prepared["use_checkpoint"]
    hasil_json = prepared["hasil_json"]
    custom_hook_path = prepared["custom_hook_path"]

    os.environ["OSC_VIDEO_SCALE_ALGO"] = str(
        getattr(cfg, "video_scale_algo", "lanczos")
    )

    # Get target dimensions for auto-bitrate calculation
    import cv2
    cap_e = cv2.VideoCapture(cfg.file_video_asli)
    src_h_e = int(cap_e.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_e.release()

    target_w_e, target_h_e = studio._get_render_dims(cfg, cfg.pilihan_rasio, source_h=src_h_e)
    video_encoder = studio.detect_video_encoder(cfg, target_h=target_h_e)

    file_glitch_ts = None
    if cfg.use_hook_glitch:
        print(f"⚙️ [{os.path.basename(cfg.outputs_dir)}] Menyiapkan Video Glitch Transisi...")

        cap_g = cv2.VideoCapture(cfg.file_video_asli)
        source_h_g = int(cap_g.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_g.release()

        file_glitch_ts = studio.siapkan_glitch_video(
            cfg.pilihan_rasio, cfg, video_encoder, source_h=source_h_g
        )

    render_manifest: list[dict] = []
    clips_to_render = []
    for klip in sorted(hasil_json, key=lambda x: x["rank"]):
        if custom_hook_path:
            klip["custom_hook_info"] = {"file_path": custom_hook_path}

        rank = klip["rank"]
        step_name = f"render_clip_{rank}"

        if use_checkpoint and checkpoint.is_step_complete(step_name):
            cached = checkpoint.get_step_data(step_name) or {}
            cached_entry = cached.get("manifest_entry")
            cached_video_path = cached_entry.get("video_path") if cached_entry else None
            if (
                cached_entry
                and cached_entry.get("status") == "success"
                and cached_video_path
                and os.path.exists(cached_video_path)
            ):
                print(f"⏭️  Rank {rank} dilewati (checkpoint: sudah dirender) — {cached_video_path}")
                render_manifest.append(cached_entry)
                continue
            # Cached but not a usable success (missing file / prior failure) — re-render.

        clips_to_render.append(klip)

    return {
        "prepared": prepared,
        "video_encoder": video_encoder,
        "file_glitch_ts": file_glitch_ts,
        "clips_to_render": clips_to_render,
        "render_manifest": render_manifest,
    }


def finalize_video_render(video_ctx: dict) -> list[dict]:
    """
    Once every clip in video_ctx["clips_to_render"] has a result recorded in
    video_ctx["render_manifest"] (via record_clip_result()), write the
    manifest, run quality control, and record dedup — mirrors the tail of
    the old single-pool render_pipeline().
    """
    prepared = video_ctx["prepared"]
    cfg = prepared["cfg"]
    dedup = prepared["dedup"]
    dedup_source = prepared["dedup_source"]
    render_manifest = video_ctx["render_manifest"]

    # Inject source metadata for attribution & safety tracking
    for row in render_manifest:
        if not row.get("source_url"):
            row["source_url"] = getattr(cfg, "url_youtube", None)

    manifest_path = os.path.join(cfg.outputs_dir, "render_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(render_manifest, f, ensure_ascii=False, indent=2)

    print(
        f"\n💾 Render manifest disimpan ke {manifest_path} ({len(render_manifest)} item)"
    )

    if getattr(cfg, "enable_quality_control", True):
        from .phase1.quality_control import QualityControlManager

        qc = QualityControlManager(strict_mode=getattr(cfg, "qc_strict", False))
        qc_summary = qc.validate_all_clips(cfg.outputs_dir)
        qc.print_summary(qc_summary)

        qc_path = os.path.join(cfg.outputs_dir, "quality_report.json")
        with open(qc_path, "w", encoding="utf-8") as f:
            json.dump(qc_summary, f, ensure_ascii=False, indent=2)
        print(f"💾 Quality report saved to {qc_path}")

    if dedup is not None and dedup_source is not None and render_manifest:
        dedup.record_video(dedup_source, cfg.file_video_asli, cfg.outputs_dir)

    return render_manifest


def record_clip_result(video_ctx: dict, klip: dict, hasil_render: dict | None) -> None:
    """Append one clip's render result to video_ctx and checkpoint it —
    shared by the single-video RenderFarm loop and the batch orchestrator's
    global-queue collector."""
    prepared = video_ctx["prepared"]
    checkpoint = prepared["checkpoint"]
    use_checkpoint = prepared["use_checkpoint"]

    if hasil_render:
        video_ctx["render_manifest"].append(hasil_render)
    if use_checkpoint and hasil_render:
        step_name = f"render_clip_{klip['rank']}"
        if hasil_render.get("status") == "success":
            checkpoint.mark_step_complete(step_name, {"manifest_entry": hasil_render})
        else:
            checkpoint.mark_step_failed(step_name, hasil_render.get("error", "unknown render error"))


def render_pipeline(prepared: dict) -> list[dict]:
    """
    Run the GPU-bound half of the pipeline for a single video: encoder/glitch
    setup, per-clip rendering spread across all visible GPUs via a
    (throwaway, single-video) RenderFarm, manifest save, quality control, and
    dedup recording.

    For batch runs use clipping.phase1.pipeline_orchestrator instead — it
    keeps ONE RenderFarm alive across the whole batch and feeds it a shared
    clip queue from every video concurrently, which is what actually keeps
    both GPUs continuously busy across video boundaries (this function's
    farm is torn down and rebuilt per call, so calling it in a loop
    reintroduces the per-video barrier).

    Parameters
    ----------
    prepared : dict
        Return value of prepare_pipeline(cfg).

    Returns
    -------
    list[dict]
        Render manifest (one dict per clip).
    """
    cfg = prepared["cfg"]
    video_ctx = setup_video_render(prepared)
    clips_to_render = video_ctx["clips_to_render"]

    if not clips_to_render:
        return finalize_video_render(video_ctx)

    num_gpus = _resolve_render_gpu_count(getattr(cfg, "render_gpus", "auto"))
    workers_per_gpu = max(1, int(getattr(cfg, "render_workers_per_gpu", 1)))
    print(
        f"🖥️  Rendering {len(clips_to_render)} klip di {num_gpus} GPU "
        f"({workers_per_gpu} worker/GPU, mis. dual T4 di Kaggle)..."
    )

    farm = RenderFarm(num_gpus, workers_per_gpu)
    farm.start()
    try:
        for klip in clips_to_render:
            farm.submit_clip(
                0,
                klip["rank"],
                klip,
                cfg.pilihan_rasio,
                video_ctx["file_glitch_ts"],
                prepared["data_segmen"],
                cfg,
                video_ctx["video_encoder"],
                prepared["diarization_data"],
            )

        for _ in range(len(clips_to_render)):
            _video_id, _rank, klip, hasil_render, err = farm.get_result()
            if err is not None:
                hasil_render = {"status": "failed", "error": str(err), "rank": klip["rank"]}
            record_clip_result(video_ctx, klip, hasil_render)
    finally:
        farm.shutdown()

    return finalize_video_render(video_ctx)


def run_pipeline(cfg) -> list[dict]:
    """
    Run the full clipping pipeline for a single source: prepare_pipeline()
    (download/transcribe/Gemini/diarization/voiceover) followed by
    render_pipeline() (GPU rendering/QC/dedup-record).

    For batch runs, prefer clipping.phase1.pipeline_orchestrator, which
    overlaps prepare_pipeline() for the next source with render_pipeline()
    for the current one so the GPU(s) never idle waiting on
    download/Whisper/Gemini/TTS.
    """
    prepared = prepare_pipeline(cfg)
    return render_pipeline(prepared)

