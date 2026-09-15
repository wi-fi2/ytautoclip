"""
web.api.routes.discover — Automatic YouTube source-video discovery.

Thin HTTP wrapper around clipping.phase1.source_discovery, so the dashboard
can search for candidate source videos and score them against the
optimizeclip.md "source video checklist" without a user leaving the New Job
screen to hunt for a URL manually.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict

from fastapi import APIRouter, HTTPException, Query

router = APIRouter(tags=["discover"])


@router.get("/api/discover")
async def discover_videos(
    query: str = Query(..., min_length=2, description="Niche/topic keywords to search for"),
    limit: int = Query(10, ge=1, le=25),
):
    """
    Search YouTube for candidate source videos matching `query`, score each
    against the automatic source-video checklist, and return them ranked
    best-first.

    Runs the (blocking, network-bound) yt-dlp search in a worker thread so
    it doesn't stall the event loop.
    """
    from clipping.phase1.source_discovery import discover_source_videos

    try:
        candidates = await asyncio.to_thread(discover_source_videos, query, limit)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Discovery search failed: {e}")

    return {
        "query": query,
        "count": len(candidates),
        "candidates": [asdict(c) for c in candidates],
    }
