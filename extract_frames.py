#!/usr/bin/env python3
"""extract_frames.py -- Sample frames from videos every N seconds with ffmpeg.

For each video in data/videos/<sc>.mp4:
  Extract frames to data/frames/<sc>/frame_<NNN>.jpg
  Sample interval: 3 seconds (configurable)
  Resize to max 720p for OCR efficiency.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import db

ROOT = Path(__file__).resolve().parent
VIDEOS_DIR = ROOT / "data" / "videos"
FRAMES_DIR = ROOT / "data" / "frames"
INTERVAL_S = 3  # one frame every 3 seconds
MAX_FRAMES = 30  # cap per video


def extract(video_path: Path, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    # ffmpeg: sample at 1/N fps, scale to max 720 wide
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(video_path),
        "-vf", f"fps=1/{INTERVAL_S},scale='min(720,iw)':-2",
        "-frames:v", str(MAX_FRAMES),
        "-q:v", "3",
        str(out_dir / "frame_%03d.jpg"),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    except subprocess.CalledProcessError as e:
        print(f"  ffmpeg error: {e.stderr.decode()[:200]}", flush=True)
        return 0
    except subprocess.TimeoutExpired:
        print("  ffmpeg timeout", flush=True)
        return 0
    frames = list(out_dir.glob("frame_*.jpg"))
    return len(frames)


def main() -> int:
    if not shutil.which("ffmpeg"):
        print("ERROR: ffmpeg not found in PATH", file=sys.stderr)
        return 1
    db.init_db()
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    rows = db.get_videos_to_extract()
    print(f"[frames] {len(rows)} videos to extract from", flush=True)

    ok = 0
    empty = 0
    for i, row in enumerate(rows, 1):
        sc = row["short_code"] or row["id"]
        video_path = VIDEOS_DIR / f"{sc}.mp4"
        out_dir = FRAMES_DIR / sc
        if not video_path.exists():
            db.mark_skipped(row["id"], "video_missing")
            continue
        # Skip if already has frames
        if out_dir.exists() and list(out_dir.glob("frame_*.jpg")):
            db.mark_frames_extracted(row["id"])
            ok += 1
            continue
        n = extract(video_path, out_dir)
        if n > 0:
            db.mark_frames_extracted(row["id"])
            ok += 1
            print(f"[frames] [{i}/{len(rows)}] {sc} -> {n} frames", flush=True)
        else:
            empty += 1
            db.mark_skipped(row["id"], "no_frames")

    print(f"[frames] done. ok={ok} empty={empty}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
