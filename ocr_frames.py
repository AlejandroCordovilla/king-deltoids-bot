#!/usr/bin/env python3
"""ocr_frames.py -- Vision OCR on extracted frames to pull on-screen text.

Tries Anthropic Claude Haiku first (if ANTHROPIC_API_KEY set), else falls back
to OpenAI GPT-4o-mini vision.

Sends all frames per video in a single multi-image request to consolidate
on-screen text and deduplicate naturally.
"""
from __future__ import annotations

import base64
import os
import sys
import time
from pathlib import Path

import db

ROOT = Path(__file__).resolve().parent
FRAMES_DIR = ROOT / "data" / "frames"

ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
OPENAI_MODEL = "gpt-4o-mini"
MAX_FRAMES_PER_REQUEST = 12


def _load_dotenv() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


_load_dotenv()

SYSTEM_PROMPT = (
    "You are an OCR specialist. You will receive frames from a fitness Instagram video. "
    "Extract ONLY the text overlays visible on-screen across all frames.\n\n"
    "RULES:\n"
    "1. Combine text from all frames into one coherent message (text often appears progressively).\n"
    "2. Remove duplicates -- if the same line appears in multiple frames, include it ONCE.\n"
    "3. Preserve order: text that appears earlier in the video comes first.\n"
    "4. IGNORE Instagram UI elements: likes count, username, follow button, music label, comment box.\n"
    "5. IGNORE captions outside the video.\n"
    "6. Focus on the educational/informational text the creator added to the video.\n"
    "7. If no on-screen text is visible, return exactly: NO_ONSCREEN_TEXT\n"
    "8. Output the extracted text as plain prose, no preamble, no quotes, no markdown."
)


def encode_b64(path: Path) -> str:
    return base64.standard_b64encode(path.read_bytes()).decode("utf-8")


def ocr_anthropic(client, frames: list[Path]) -> str:
    content = []
    for f in frames:
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": encode_b64(f)},
        })
    content.append({"type": "text", "text": "Extract all unique on-screen text overlays in order."})
    r = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": content}],
    )
    text = r.content[0].text.strip()
    return "" if text.upper().startswith("NO_ONSCREEN_TEXT") else text


def ocr_openai(client, frames: list[Path]) -> str:
    content = []
    for f in frames:
        b64 = encode_b64(f)
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    content.append({"type": "text", "text": "Extract all unique on-screen text overlays in order."})
    r = client.chat.completions.create(
        model=OPENAI_MODEL,
        max_tokens=1024,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
    )
    text = r.choices[0].message.content.strip()
    return "" if text.upper().startswith("NO_ONSCREEN_TEXT") else text


def get_client():
    """Return (client, ocr_fn, provider_name)."""
    anth_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if anth_key and anth_key != "your_key_here" and anth_key.startswith("sk-ant"):
        from anthropic import Anthropic
        return Anthropic(api_key=anth_key), ocr_anthropic, "anthropic"
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if openai_key:
        from openai import OpenAI
        return OpenAI(api_key=openai_key), ocr_openai, "openai"
    return None, None, None


def main() -> int:
    client, ocr_fn, provider = get_client()
    if not client:
        print("ERROR: Need ANTHROPIC_API_KEY or OPENAI_API_KEY in .env", file=sys.stderr)
        return 1
    print(f"[ocr] using provider: {provider}", flush=True)

    db.init_db()
    rows = db.get_videos_to_ocr()
    print(f"[ocr] {len(rows)} videos to OCR", flush=True)

    ok = 0
    empty = 0
    fail = 0
    for i, row in enumerate(rows, 1):
        sc = row["short_code"] or row["id"]
        frame_dir = FRAMES_DIR / sc
        frames = sorted(frame_dir.glob("frame_*.jpg"))[:MAX_FRAMES_PER_REQUEST]
        if not frames:
            db.save_ocr_text(row["id"], "")
            empty += 1
            continue
        try:
            text = ocr_fn(client, frames)
        except Exception as e:
            print(f"  [ocr] {sc} error: {e}", flush=True)
            fail += 1
            continue
        if text:
            db.save_ocr_text(row["id"], text)
            ok += 1
            if i % 25 == 0 or i <= 5:
                print(f"[ocr] [{i}/{len(rows)}] {sc} -> {len(text)} chars", flush=True)
        else:
            db.save_ocr_text(row["id"], "")
            empty += 1
        time.sleep(0.3)

    print(f"[ocr] done. ok={ok} empty={empty} fail={fail}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
