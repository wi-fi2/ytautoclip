"""
Automatic YouTube Source-Video Discovery

Implements optimizeclip.md's "Method to find the best source videos for
high-ranking Shorts clips" as an automatic, keyword-driven search instead of
manual research. Uses yt-dlp's flat-playlist search extraction (no YouTube
Data API key required) to pull lightweight metadata for candidate videos,
then scores each one against an automatic approximation of the doc's
0-25 "source video checklist":

    1. Topic strength & searchability (0-5)  -- title/keyword specificity
    2. Overperformance-for-size (0-5)         -- views-per-day vs. small-
                                                  channel expectations (the
                                                  doc's "smaller channel,
                                                  disproportionate views"
                                                  signal)
    3. Duration fit (0-5)                     -- long enough to hold 5-10
                                                  clippable moments, not a
                                                  multi-hour dump
    4. Freshness (0-5)                        -- last-3-12-months bias from
                                                  the doc's competitor-search
                                                  step
    5. Benefit-driven title (0-5)             -- numbers/specificity in the
                                                  title, correlated in the
                                                  doc with "clip-rich"
                                                  structure

Segment-level signals from the doc (clear chapter breaks, emotional peaks,
standalone value) require actually watching/transcribing the video, so they
are NOT scored here — this module is a triage step to shortlist candidate
*videos* worth feeding into the real pipeline (which then scores individual
*clips* via clipping.phase1.monetization). Every score is a best-effort
proxy computed from yt-dlp metadata only; nothing here calls the YouTube
Data API or requires an API key.
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SourceCandidate:
    """A discovered candidate source video plus its checklist score."""
    video_id: str
    title: str
    url: str
    channel: str
    duration_sec: float
    view_count: int
    upload_date: Optional[str]  # YYYYMMDD or None
    score: float  # 0-25
    breakdown: Dict[str, float] = field(default_factory=dict)
    label: str = "skip"  # "high_priority" | "okay" | "skip"
    reasons: List[str] = field(default_factory=list)


# Benefit-driven / specificity signals in a title (mirrors the doc's
# "specific and benefit-driven" titles and the tech/food/travel hook
# vocabulary already used in clipping.phase1.monetization).
_SPECIFICITY_MARKERS = [
    'how to', 'why', 'mistake', 'tips', 'tricks', 'secrets', 'guide',
    'review', 'vs', 'before and after', 'i tried', 'i tested', 'i built',
    'stop doing', 'never', 'best', 'worst', 'ways to', 'reasons',
]


def _has_digit(text: str) -> bool:
    return any(ch.isdigit() for ch in text)


def score_topic_strength(title: str, query: str) -> float:
    """0-5: keyword overlap with the search query + benefit-driven phrasing."""
    if not title:
        return 0.0

    title_lower = title.lower()
    query_terms = [t for t in query.lower().split() if len(t) > 2]

    overlap = sum(1 for t in query_terms if t in title_lower)
    overlap_ratio = overlap / len(query_terms) if query_terms else 0.0

    score = 2.5 * min(1.0, overlap_ratio * 1.5)

    if any(marker in title_lower for marker in _SPECIFICITY_MARKERS):
        score += 1.5
    if _has_digit(title):
        score += 1.0  # "3 Ways to...", "in 2026", specific numbers

    return min(5.0, score)


def score_overperformance(view_count: int, upload_date: Optional[str]) -> float:
    """
    0-5: views-per-day since upload. This is the automatic stand-in for the
    doc's "smaller channel, disproportionately high views" signal — we can't
    cheaply fetch subscriber counts for every search hit without the
    YouTube Data API, but a video racking up views fast is exactly what that
    heuristic is trying to surface either way.
    """
    if not view_count or not upload_date:
        return 2.0  # Unknown - neutral-low, don't reward absence of data

    try:
        uploaded = datetime.strptime(upload_date, "%Y%m%d").replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return 2.0

    days = max(1.0, (datetime.now(timezone.utc) - uploaded).days)
    views_per_day = view_count / days

    # Log-scaled bands: ~50/day is unremarkable, ~50k/day is a clear hit.
    if views_per_day <= 0:
        return 0.0
    scaled = math.log10(max(views_per_day, 1)) - 1.5  # log10(50) ~= 1.7
    return min(5.0, max(0.0, scaled * 1.8))


def score_duration_fit(duration_sec: float) -> float:
    """0-5: long enough for 5-10 clippable moments (doc's 'clip mining' target)."""
    minutes = duration_sec / 60.0
    if 8 <= minutes <= 40:
        return 5.0
    if 4 <= minutes < 8 or 40 < minutes <= 70:
        return 3.5
    if 1.5 <= minutes < 4 or 70 < minutes <= 120:
        return 2.0
    return 0.5


def score_freshness(upload_date: Optional[str]) -> float:
    """0-5: last 3-12 months preferred, per the doc's competitor-search step."""
    if not upload_date:
        return 1.5

    try:
        uploaded = datetime.strptime(upload_date, "%Y%m%d").replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return 1.5

    days = (datetime.now(timezone.utc) - uploaded).days
    if days <= 365:
        return 5.0
    if days <= 730:
        return 3.0
    if days <= 1460:
        return 1.5
    return 0.5


def score_source_video(meta: Dict, query: str) -> SourceCandidate:
    """
    Score a single yt-dlp search-result entry against the automatic
    approximation of optimizeclip.md's source-video checklist.
    """
    title = meta.get("title") or ""
    duration_sec = float(meta.get("duration") or 0.0)
    view_count = int(meta.get("view_count") or 0)
    upload_date = meta.get("upload_date")
    channel = meta.get("channel") or meta.get("uploader") or "Unknown"
    video_id = meta.get("id") or ""
    url = meta.get("webpage_url") or meta.get("url") or (
        f"https://www.youtube.com/watch?v={video_id}" if video_id else ""
    )

    topic = score_topic_strength(title, query)
    overperf = score_overperformance(view_count, upload_date)
    duration_fit = score_duration_fit(duration_sec)
    freshness = score_freshness(upload_date)
    benefit_title = (2.5 if _has_digit(title) else 0.0) + (
        2.5 if any(m in title.lower() for m in _SPECIFICITY_MARKERS) else 0.0
    )

    total = topic + overperf + duration_fit + freshness + benefit_title

    if total >= 18:
        label = "high_priority"
    elif total >= 12:
        label = "okay"
    else:
        label = "skip"

    reasons = []
    if topic < 2:
        reasons.append("Title doesn't closely match your search topic")
    if overperf < 2:
        reasons.append("Low views-per-day for its age — may not be resonating")
    if duration_fit < 3:
        reasons.append("Length isn't ideal for mining 5-10 clips (best: 8-40 min)")
    if freshness < 2:
        reasons.append("Older upload — competitor-search signal is weaker")
    if not reasons:
        reasons.append("Strong all-around candidate — worth clipping")

    return SourceCandidate(
        video_id=video_id,
        title=title,
        url=url,
        channel=channel,
        duration_sec=duration_sec,
        view_count=view_count,
        upload_date=upload_date,
        score=round(total, 1),
        breakdown={
            "topic_strength": round(topic, 1),
            "overperformance": round(overperf, 1),
            "duration_fit": round(duration_fit, 1),
            "freshness": round(freshness, 1),
            "benefit_driven_title": round(benefit_title, 1),
        },
        label=label,
        reasons=reasons,
    )


def search_youtube_candidates(query: str, limit: int = 15) -> List[Dict]:
    """
    Search YouTube via yt-dlp (no API key needed) and return metadata dicts
    for the top `limit` results.

    Uses full (non-flat) extraction: flat/in_playlist search results don't
    carry `upload_date`, which the overperformance and freshness scores
    depend on, so each result costs one extra metadata request to YouTube
    (still no video download — `skip_download` stays on).
    """
    import yt_dlp

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "extract_flat": False,
        "default_search": "ytsearch",
    }

    search_term = f"ytsearch{max(1, limit)}:{query}"

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(search_term, download=False)

    entries = (info or {}).get("entries") or []
    return [e for e in entries if e]


def discover_source_videos(query: str, limit: int = 10, search_pool: int = 15) -> List[SourceCandidate]:
    """
    Search + score + rank candidate source videos for `query`.

    Args:
        query: Niche/topic keywords, e.g. "AI coding tips" or "budget travel India".
        limit: How many ranked candidates to return.
        search_pool: How many raw search results to fetch before scoring
            (larger pool = better ranking, more yt-dlp overhead).

    Returns:
        SourceCandidate list sorted by score descending, length <= limit.
        Returns an empty list (never raises) if the search itself fails,
        so a flaky network call doesn't take down a caller that's just
        trying to suggest videos.
    """
    try:
        raw_results = search_youtube_candidates(query, limit=search_pool)
    except Exception as e:
        logger.warning(f"YouTube search failed for query={query!r}: {e}")
        return []

    candidates = [score_source_video(meta, query) for meta in raw_results]
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:limit]


def print_discovery_report(candidates: List[SourceCandidate], query: str) -> None:
    """Pretty-print a ranked discovery report to stdout (CLI usage)."""
    print("\n" + "=" * 78)
    print(f"🔎 Source Video Discovery — \"{query}\"")
    print("=" * 78)

    if not candidates:
        print("No candidates found (search failed or returned nothing).")
        print("=" * 78 + "\n")
        return

    label_icons = {"high_priority": "🟢 HIGH", "okay": "🟡 OKAY", "skip": "🔴 SKIP"}

    for i, c in enumerate(candidates, start=1):
        mins = c.duration_sec / 60.0
        print(f"\n{i}. {label_icons.get(c.label, c.label)}  score={c.score}/25")
        print(f"   {c.title}")
        print(f"   {c.url}")
        print(
            f"   Channel: {c.channel}  |  Views: {c.view_count:,}  |  "
            f"Duration: {mins:.1f} min  |  Uploaded: {c.upload_date or 'unknown'}"
        )
        breakdown_str = ", ".join(f"{k}={v}" for k, v in c.breakdown.items())
        print(f"   Breakdown: {breakdown_str}")
        for reason in c.reasons:
            print(f"   - {reason}")

    print("\n" + "=" * 78)
    print("Pick a URL above and re-run with:  python main.py --url \"<url>\" ...")
    print("=" * 78 + "\n")
