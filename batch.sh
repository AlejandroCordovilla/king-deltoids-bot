#!/usr/bin/env bash
# Full pipeline: scrape -> download -> extract frames -> OCR -> combine -> ingest
set -e
cd "$(dirname "$0")"

echo "=== [1/6] scrape ==="
python3 scrape.py

echo "=== [2/6] download videos ==="
python3 download_videos.py

echo "=== [3/6] extract frames ==="
python3 extract_frames.py

echo "=== [4/6] OCR frames ==="
python3 ocr_frames.py

echo "=== [5/6] combine ==="
python3 combine.py

echo "=== [6/6] ingest into ChromaDB ==="
python3 ingest.py

echo "=== DONE ==="
python3 -c "import db; print(db.stats())"
