"""
Candidate Scoring Adapter

Bridges the Gemini/NVIDIA-shaped clip candidates produced by
`clipping.engine.analyze_with_ai` (as normalized by
`clipping.metadata.normalize_and_validate`) to the generic
`{text, start, end}` contract expected by `clipping.phase1.monetization`.

`monetization.py` is intentionally pipeline-agnostic (independently tested
against a plain segment schema); this module is the one integration point
that knows about `runner.py`'s actual candidate shape (`start_time`,
`end_time`, `viral_score`, etc.) and does the text-extraction + weighted
combination + artifact writing specific to that pipeline.

Face/motion detection: when `cfg` is passed to `score_candidates()`, the
opening ~3s of each candidate window is sampled from `cfg.file_video_asli`
via OpenCV + the MediaPipe face detector already used by
`clipping/studio/face_detection.py`, and `has_face` / `has_motion` are
threaded through to `MonetizationScorer.score_clip` as real booleans instead
of `None`. Detection is best-effort: any failure (missing video, decode
error, model download failure) falls back to `has_face=None` /
`has_motion=None` (treated as unknown, not confirmed-absent) rather than
raising, so a broken source video never fails the whole scoring pass.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

from .monetization import MonetizationScorer

logger = logging.getLogger(__name__)


def extract_candidate_text(klip: Dict, data_segmen: List[Dict]) -> str:
    """
    Build transcript text for a candidate's [start_time, end_time] window.

    Reuses the exact overlap + text-join pattern already used for the
    voice-over script step in clipping/runner.py (lines ~190-198): a segment
    is included if it overlaps the window at all (segment end > window start
    AND segment start < window end), not only if fully contained. Supports
    both Whisper-style segments (`text` key) and YouTube JSON3 segments
    (`words: [{word: ...}, ...]`, no top-level `text`).

    Args:
        klip: Candidate dict with 'start_time' / 'end_time' keys.
        data_segmen: Transcript segments from Whisper or YouTube JSON3.

    Returns:
        Joined transcript text for the window (empty string if no overlap).
    """
    start = float(klip.get("start_time", 0.0))
    end = float(klip.get("end_time", 0.0))

    lines = []
    for seg in data_segmen:
        seg_end = float(seg.get("end", 0.0))
        seg_start = float(seg.get("start", 0.0))
        if seg_end > start and seg_start < end:
            seg_text = seg.get("text") or " ".join(
                w["word"] for w in seg.get("words", [])
            )
            if seg_text:
                lines.append(seg_text)

    return " ".join(lines).strip()


def build_monetization_input(klip: Dict, data_segmen: List[Dict]) -> Dict:
    """
    Build the {'text', 'start', 'end'} dict MonetizationScorer expects.

    Args:
        klip: Candidate dict with 'start_time' / 'end_time' keys.
        data_segmen: Transcript segments.

    Returns:
        {'text': str, 'start': float, 'end': float}
    """
    return {
        "text": extract_candidate_text(klip, data_segmen),
        "start": float(klip.get("start_time", 0.0)),
        "end": float(klip.get("end_time", 0.0)),
    }


def detect_opening_face_and_motion(
    video_path: str, cfg, start: float, window: float = 3.0
) -> Tuple[Optional[bool], Optional[bool]]:
    """
    Best-effort face/motion detection for a candidate's opening window.

    Samples two frames (at `start` and `start + window`) from `video_path`,
    runs them through the same MediaPipe face detector used for speaker
    counting, and diffs them (mean absolute pixel difference) to guess
    whether there's visible motion/cuts in the opening seconds.

    Returns (has_face, has_motion), either of which is None if detection
    could not be performed (missing file, decode failure, model load
    failure) — never raises.
    """
    try:
        import cv2
        import numpy as np
        from clipping.studio.face_detection import get_face_detector
        from mediapipe.tasks import python as mp_python
    except Exception as e:
        logger.debug(f"Face/motion detection unavailable (import failed): {e}")
        return None, None

    if not video_path or not os.path.exists(video_path):
        return None, None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None

    try:
        frames = []
        for t in (start, start + window):
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)

        if not frames:
            return None, None

        has_face = None
        try:
            detector = get_face_detector(cfg)
            mp_image = mp_python.Image(
                image_format=mp_python.ImageFormat.SRGB,
                data=cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB),
            )
            result = detector.detect(mp_image)
            has_face = bool(result.detections)
        except Exception as e:
            logger.debug(f"Face detection failed for {video_path} @ {start}s: {e}")

        has_motion = None
        if len(frames) == 2:
            try:
                a = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
                b = cv2.cvtColor(frames[1], cv2.COLOR_BGR2GRAY)
                b = cv2.resize(b, (a.shape[1], a.shape[0]))
                diff = float(np.mean(cv2.absdiff(a, b)))
                has_motion = diff > 12.0  # Empirical threshold for a real scene/cut change
            except Exception as e:
                logger.debug(f"Motion diff failed for {video_path} @ {start}s: {e}")

        return has_face, has_motion
    finally:
        cap.release()


def score_candidates(
    hasil_json: List[Dict],
    data_segmen: List[Dict],
    quality_weight: float = 0.7,
    monetization_weight: float = 0.3,
    cfg=None,
) -> List[Dict]:
    """
    Attach monetization scoring + combined ranking score to each candidate.

    Mutates and returns the same list (in the original order given — this
    function does NOT re-sort or reassign 'rank'; that is the caller's
    responsibility, mirroring how clipping.metadata.normalize_and_validate
    already owns its own sort-then-reassign-rank step). Every pre-existing
    key on each candidate dict is left untouched; only new keys are added:

        quality_score_raw   -- original viral_score (1-100), untouched
        quality_score_norm  -- viral_score / 100.0
        monetization        -- full MonetizationMetrics dict (see
                                monetization.MonetizationScorer.batch_score)
        combined_score      -- quality_weight * quality_norm
                                + monetization_weight * monetization_score
        scoring_weights     -- {'quality': quality_weight,
                                 'monetization': monetization_weight}

    Args:
        hasil_json: Candidate list as produced by
            clipping.metadata.normalize_and_validate (each item must already
            have a numeric 'viral_score').
        data_segmen: Transcript segments used to extract per-candidate text.
        quality_weight: Weight (0-1) for the existing AI viral_score.
        monetization_weight: Weight (0-1) for the monetization score.
        cfg: Optional runtime config. When provided and `cfg.file_video_asli`
            exists, each candidate's opening ~3s is sampled for real
            has_face/has_motion signals (see
            `detect_opening_face_and_motion`). Omitted or on any detection
            failure, has_face/has_motion stay `None` (unknown) exactly as
            before.

    Returns:
        The same list, each item enriched with the fields above.

    Raises:
        ValueError: if quality_weight + monetization_weight does not sum to
            1.0 within a 1e-6 tolerance. This is a second guard in addition
            to the CLI-level validation in clipping.config.build_config,
            since this function can be called directly (tests, notebooks)
            without going through the CLI.
    """
    total_weight = quality_weight + monetization_weight
    if abs(total_weight - 1.0) > 1e-6:
        raise ValueError(
            f"quality_weight + monetization_weight must sum to 1.0, "
            f"got {quality_weight} + {monetization_weight} = {total_weight}"
        )

    video_path = getattr(cfg, "file_video_asli", None) if cfg is not None else None

    for klip in hasil_json:
        mon_input = build_monetization_input(klip, data_segmen)

        has_face, has_motion = None, None
        if video_path:
            has_face, has_motion = detect_opening_face_and_motion(
                video_path, cfg, mon_input["start"]
            )

        metrics = MonetizationScorer.score_clip(
            mon_input, has_face=has_face, has_motion=has_motion
        )

        viral_score = float(klip.get("viral_score", 0))
        quality_norm = viral_score / 100.0

        klip["quality_score_raw"] = viral_score
        klip["quality_score_norm"] = quality_norm
        klip["monetization"] = {
            "hook_strength": metrics.hook_strength,
            "vvsa_score": metrics.vvsa_score,
            "retention_score": metrics.retention_score,
            "optimal_length": metrics.optimal_length,
            "structure_score": metrics.structure_score,
            "monetization_score": metrics.monetization_score,
            "series_potential": metrics.series_potential,
            "has_face": metrics.has_face,
            "has_motion": metrics.has_motion,
            "value_density_score": metrics.value_density_score,
            "loop_score": metrics.loop_score,
            "recommendations": metrics.recommendations,
        }
        klip["combined_score"] = (
            quality_weight * quality_norm
            + monetization_weight * metrics.monetization_score
        )
        klip["scoring_weights"] = {
            "quality": quality_weight,
            "monetization": monetization_weight,
        }

    return hasil_json


def write_candidates_artifact(hasil_json: List[Dict], path: str) -> None:
    """
    Write the (scored or unscored) candidate list to an inspectable JSON
    artifact, following the same convention as gemini_response.json and
    metadata_preview.json in clipping/runner.py.

    Args:
        hasil_json: Candidate list, scored or not.
        path: Destination file path (typically
            os.path.join(cfg.outputs_dir, "clip_candidates.json")).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(hasil_json, f, ensure_ascii=False, indent=2)
