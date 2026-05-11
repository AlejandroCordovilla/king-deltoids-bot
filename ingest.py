#!/usr/bin/env python3
"""ingest.py -- Chunk combined texts and embed into ChromaDB.

Reads from data/kd.db (posts.combined_text). Persists vector store at data/chroma/.
Idempotent.
"""
from __future__ import annotations

import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

import db

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CHROMA_DIR = DATA_DIR / "chroma"

COLLECTION = "kd_videos"
EMBED_MODEL = "BAAI/bge-large-en-v1.5"
CHUNK_WORDS = 120
CHUNK_OVERLAP = 40


def chunk_text(text: str, size: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = (text or "").split()
    if not words:
        return []
    if len(words) <= size:
        return [" ".join(words)]
    chunks: list[str] = []
    step = max(1, size - overlap)
    for i in range(0, len(words), step):
        piece = words[i : i + size]
        if not piece:
            break
        chunks.append(" ".join(piece))
        if i + size >= len(words):
            break
    return chunks


def existing_ids(collection) -> set[str]:
    ids: set[str] = set()
    try:
        offset = 0
        page = 1000
        while True:
            res = collection.get(include=["metadatas"], limit=page, offset=offset)
            metas = res.get("metadatas") or []
            if not metas:
                break
            for m in metas:
                if m and "post_id" in m:
                    ids.add(m["post_id"])
            if len(metas) < page:
                break
            offset += page
    except Exception as e:
        print(f"[ingest] warn: {e}", file=sys.stderr)
    return ids


def main() -> int:
    db.init_db()
    rows = db.get_all_combined()
    if not rows:
        print("[ingest] no combined texts to ingest", flush=True)
        return 0

    print(f"[ingest] loading embedding model {EMBED_MODEL}", flush=True)
    embedder = SentenceTransformer(EMBED_MODEL)

    print(f"[ingest] opening Chroma at {CHROMA_DIR}", flush=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(name=COLLECTION)

    already = existing_ids(collection)
    print(f"[ingest] {len(already)} posts already in collection", flush=True)

    added = 0
    for row in rows:
        pid = row["id"]
        if pid in already:
            continue
        text = row["combined_text"] or ""
        chunks = chunk_text(text)
        if not chunks:
            continue
        embeddings = embedder.encode(chunks, show_progress_bar=False).tolist()
        ids = [f"{pid}_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "post_id": pid,
                "short_code": row["short_code"] or "",
                "url": row["url"] or "",
                "date": row["timestamp"] or "",
                "caption_preview": (row["caption"] or "")[:160],
                "chunk_index": i,
            }
            for i in range(len(chunks))
        ]
        collection.add(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas)
        db.mark_indexed(pid)
        added += 1
        if added % 50 == 0:
            print(f"[ingest] progress: {added} posts ingested", flush=True)

    print(f"[ingest] done. new_posts={added} collection_size={collection.count()}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
