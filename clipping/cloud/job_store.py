"""
clipping.cloud.job_store — PC-side durable job state.

Tracks each video-level job (one manifest = one source video + its clip
list) through: READY -> SENT -> COMPLETED / FAILED. This is the actual
queue state; Kaggle Dataset pushes/pulls (kaggle_transport.py) are just
batched transport between this store and the Kaggle GPU consumer, which
owns no durable state of its own — if a Kaggle session dies mid-run, the
PC still knows exactly what was SENT but never confirmed COMPLETED, and
can decide to resend it in a later batch.
"""

import os
import sqlite3
import threading
import time

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    video_id TEXT PRIMARY KEY,
    source_url TEXT,
    status TEXT NOT NULL,
    clip_count INTEGER DEFAULT 0,
    manifest_path TEXT,
    error TEXT,
    created_at REAL,
    updated_at REAL
);
"""


class JobStore:
    def __init__(self, db_path: str):
        parent = os.path.dirname(os.path.abspath(db_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._db_path = db_path
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.execute(_SCHEMA)

    def _connect(self):
        return sqlite3.connect(self._db_path, timeout=30)

    def add_ready(self, video_id: str, source_url: str, clip_count: int, manifest_path: str) -> None:
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs (video_id, source_url, status, clip_count, manifest_path, created_at, updated_at)
                VALUES (?, ?, 'READY', ?, ?, ?, ?)
                ON CONFLICT(video_id) DO UPDATE SET
                    status='READY', clip_count=excluded.clip_count,
                    manifest_path=excluded.manifest_path, updated_at=excluded.updated_at
                """,
                (video_id, source_url, clip_count, manifest_path, now, now),
            )

    def mark_sent(self, video_ids: list[str]) -> None:
        if not video_ids:
            return
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.executemany(
                "UPDATE jobs SET status='SENT', updated_at=? WHERE video_id=?",
                [(now, vid) for vid in video_ids],
            )

    def mark_completed(self, video_id: str) -> None:
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET status='COMPLETED', updated_at=? WHERE video_id=?",
                (now, video_id),
            )

    def mark_failed(self, video_id: str, error: str) -> None:
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET status='FAILED', error=?, updated_at=? WHERE video_id=?",
                (error, now, video_id),
            )

    def get_by_status(self, status: str) -> list[dict]:
        with self._lock, self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM jobs WHERE status=?", (status,)).fetchall()
            return [dict(r) for r in rows]

    def counts(self) -> dict:
        with self._lock, self._connect() as conn:
            rows = conn.execute("SELECT status, COUNT(*), COALESCE(SUM(clip_count), 0) FROM jobs GROUP BY status").fetchall()
            return {status: {"videos": n, "clips": clips} for status, n, clips in rows}
