#!/usr/bin/env python3
"""download_videos.py -- Download all video mp4s from Instagram CDN URLs.

Reads video_url from posts table, downloads to data/videos/<short_code>.mp4.
Idempotent: skips already-downloaded videos.

Note: Instagram CDN URLs can expire. Run this soon after scrape.py.
"""
from __future__ import annotations

import ssl
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

import db

ssl._create_default_https_context = ssl._create_unverified_context  # noqa: SLF001

ROOT = Path(__file__).resolve().parent
VIDEOS_DIR = ROOT / "data" / "videos"


def download(url: str, dest: Path) -> bool:
    req = Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    })
    try:
        with urlopen(req, timeout=120) as resp:
            data = resp.read()
            dest.write_bytes(data)
            return True
    except Exception as e:
        print(f"  ERR: {e}", flush=True)
        return False


def main() -> int:
    db.init_db()
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    rows = db.get_videos_to_download()
    print(f"[download] {len(rows)} videos to download", flush=True)

    ok = 0
    fail = 0
    for i, row in enumerate(rows, 1):
        sc = row["short_code"] or row["id"]
        dest = VIDEOS_DIR / f"{sc}.mp4"
        if dest.exists() and dest.stat().st_size > 1000:
            db.mark_downloaded(row["id"])
            ok += 1
            continue
        print(f"[download] [{i}/{len(rows)}] {sc}...", flush=True)
        if download(row["video_url"], dest):
            db.mark_downloaded(row["id"])
            ok += 1
        else:
            fail += 1
            db.mark_skipped(row["id"], "download_failed")
        time.sleep(0.3)  # be polite

    print(f"[download] done. ok={ok} fail={fail}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
