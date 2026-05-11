#!/usr/bin/env python3
"""db.py -- SQLite access layer for king_deltoids bot."""
from __future__ import annotations

import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "data" / "kd.db"


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(DB_PATH))
    con.row_factory = sqlite3.Row
    return con


def init_db() -> None:
    with _conn() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS posts (
                id              TEXT PRIMARY KEY,
                short_code      TEXT,
                url             TEXT,
                post_type       TEXT,
                caption         TEXT,
                hashtags        TEXT,
                video_url       TEXT,
                display_url     TEXT,
                video_duration  REAL DEFAULT 0,
                likes_count     INTEGER DEFAULT 0,
                comments_count  INTEGER DEFAULT 0,
                video_views     INTEGER DEFAULT 0,
                timestamp       TEXT,
                downloaded      INTEGER DEFAULT 0,
                frames_extracted INTEGER DEFAULT 0,
                ocr_done        INTEGER DEFAULT 0,
                ocr_text        TEXT,
                combined_text   TEXT,
                indexed         INTEGER DEFAULT 0,
                skipped_reason  TEXT
            )
        """)


def upsert_post(p: dict) -> None:
    with _conn() as con:
        con.execute("""
            INSERT INTO posts (id, short_code, url, post_type, caption, hashtags,
                              video_url, display_url, video_duration,
                              likes_count, comments_count, video_views, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                short_code     = excluded.short_code,
                url            = excluded.url,
                post_type      = excluded.post_type,
                caption        = excluded.caption,
                hashtags       = excluded.hashtags,
                video_url      = excluded.video_url,
                display_url    = excluded.display_url,
                video_duration = excluded.video_duration,
                likes_count    = excluded.likes_count,
                comments_count = excluded.comments_count,
                video_views    = excluded.video_views,
                timestamp      = excluded.timestamp
        """, (
            p["id"], p.get("short_code", ""), p.get("url", ""), p.get("post_type", ""),
            p.get("caption", ""), p.get("hashtags", ""),
            p.get("video_url", ""), p.get("display_url", ""), p.get("video_duration", 0),
            p.get("likes_count", 0), p.get("comments_count", 0), p.get("video_views", 0),
            p.get("timestamp", ""),
        ))


def mark_skipped(id: str, reason: str) -> None:
    with _conn() as con:
        con.execute("UPDATE posts SET skipped_reason = ? WHERE id = ?", (reason, id))


def mark_downloaded(id: str) -> None:
    with _conn() as con:
        con.execute("UPDATE posts SET downloaded = 1 WHERE id = ?", (id,))


def mark_frames_extracted(id: str) -> None:
    with _conn() as con:
        con.execute("UPDATE posts SET frames_extracted = 1 WHERE id = ?", (id,))


def save_ocr_text(id: str, text: str) -> None:
    with _conn() as con:
        con.execute("UPDATE posts SET ocr_text = ?, ocr_done = 1 WHERE id = ?", (text, id))


def save_combined_text(id: str, text: str) -> None:
    with _conn() as con:
        con.execute("UPDATE posts SET combined_text = ? WHERE id = ?", (text, id))


def mark_indexed(id: str) -> None:
    with _conn() as con:
        con.execute("UPDATE posts SET indexed = 1 WHERE id = ?", (id,))


def get_videos_to_download() -> list[sqlite3.Row]:
    with _conn() as con:
        return con.execute("""
            SELECT id, short_code, video_url FROM posts
            WHERE video_url != '' AND video_url IS NOT NULL
              AND downloaded = 0 AND skipped_reason IS NULL
        """).fetchall()


def get_videos_to_extract() -> list[sqlite3.Row]:
    with _conn() as con:
        return con.execute("""
            SELECT id, short_code FROM posts
            WHERE downloaded = 1 AND frames_extracted = 0 AND skipped_reason IS NULL
        """).fetchall()


def get_videos_to_ocr() -> list[sqlite3.Row]:
    with _conn() as con:
        return con.execute("""
            SELECT id, short_code FROM posts
            WHERE frames_extracted = 1 AND ocr_done = 0 AND skipped_reason IS NULL
        """).fetchall()


def get_posts_to_combine() -> list[sqlite3.Row]:
    with _conn() as con:
        return con.execute("""
            SELECT id, caption, ocr_text FROM posts
            WHERE (ocr_done = 1 OR video_url = '' OR video_url IS NULL)
              AND skipped_reason IS NULL AND combined_text IS NULL
        """).fetchall()


def get_all_combined() -> list[sqlite3.Row]:
    with _conn() as con:
        return con.execute("""
            SELECT id, short_code, url, caption, combined_text, timestamp
            FROM posts
            WHERE combined_text IS NOT NULL AND combined_text != ''
              AND skipped_reason IS NULL
        """).fetchall()


def post_count() -> int:
    with _conn() as con:
        return con.execute("SELECT COUNT(*) FROM posts").fetchone()[0]


def stats() -> dict:
    with _conn() as con:
        cur = con.cursor()
        s = {}
        s["total"] = cur.execute("SELECT COUNT(*) FROM posts").fetchone()[0]
        s["videos"] = cur.execute("SELECT COUNT(*) FROM posts WHERE video_url != '' AND video_url IS NOT NULL").fetchone()[0]
        s["skipped"] = cur.execute("SELECT COUNT(*) FROM posts WHERE skipped_reason IS NOT NULL").fetchone()[0]
        s["downloaded"] = cur.execute("SELECT COUNT(*) FROM posts WHERE downloaded=1").fetchone()[0]
        s["ocr_done"] = cur.execute("SELECT COUNT(*) FROM posts WHERE ocr_done=1").fetchone()[0]
        s["combined"] = cur.execute("SELECT COUNT(*) FROM posts WHERE combined_text IS NOT NULL").fetchone()[0]
        s["indexed"] = cur.execute("SELECT COUNT(*) FROM posts WHERE indexed=1").fetchone()[0]
        return s
