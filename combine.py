#!/usr/bin/env python3
"""combine.py -- Merge caption + on-screen OCR text into a single transcript per post."""
from __future__ import annotations

import sys

import db


def combine_text(caption: str, ocr_text: str) -> str:
    parts = []
    if caption:
        parts.append(f"CAPTION: {caption.strip()}")
    if ocr_text:
        parts.append(f"ON-SCREEN TEXT: {ocr_text.strip()}")
    return "\n\n".join(parts).strip()


def main() -> int:
    db.init_db()
    rows = db.get_posts_to_combine()
    print(f"[combine] {len(rows)} posts to combine", flush=True)
    saved = 0
    for row in rows:
        merged = combine_text(row["caption"] or "", row["ocr_text"] or "")
        if merged:
            db.save_combined_text(row["id"], merged)
            saved += 1
    print(f"[combine] done. saved={saved}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
