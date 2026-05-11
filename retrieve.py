#!/usr/bin/env python3
"""retrieve.py -- Hybrid retrieval (semantic + BM25) for king_deltoids RAG."""
from __future__ import annotations

import os
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CHROMA_DIR = DATA_DIR / "chroma"
COLLECTION = "kd_videos"
EMBED_MODEL = "BAAI/bge-large-en-v1.5"
TOP_K = 10
DISTANCE_THRESHOLD = 1.5


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

_embedder: SentenceTransformer | None = None
_collection = None
_bm25_cache: tuple[BM25Okapi, list[str], list[dict]] | None = None


def embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_or_create_collection(name=COLLECTION)
    return _collection


def _expand_queries(question: str) -> list[str]:
    q = question.strip().rstrip("?")
    words = q.lower().split()
    variants = [q]
    if len(words) > 3:
        variants.append(" ".join(words[-4:]))
    fillers = {"how", "much", "many", "do", "i", "should", "need", "is", "what", "are", "the", "a", "an"}
    keywords = [w for w in words if w not in fillers]
    if keywords and keywords != words:
        variants.append(" ".join(keywords))
    return list(dict.fromkeys(variants))


def _semantic_retrieve(query: str, k: int) -> list[dict]:
    q_emb = embedder().encode([query]).tolist()
    res = collection().query(
        query_embeddings=q_emb,
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    return [
        {"text": doc, "meta": meta or {}, "distance": dist}
        for doc, meta, dist in zip(docs, metas, dists)
        if dist <= DISTANCE_THRESHOLD
    ]


def _bm25_retrieve(query: str, k: int) -> list[dict]:
    global _bm25_cache
    coll = collection()
    total = coll.count()
    if total == 0:
        return []
    if _bm25_cache is None or len(_bm25_cache[1]) != total:
        all_res = coll.get(include=["documents", "metadatas"], limit=total)
        all_docs = all_res.get("documents") or []
        all_metas = all_res.get("metadatas") or []
        if not all_docs:
            return []
        tokenized = [d.lower().split() for d in all_docs]
        bm25 = BM25Okapi(tokenized)
        _bm25_cache = (bm25, all_docs, all_metas)
    bm25, all_docs, all_metas = _bm25_cache
    scores = bm25.get_scores(query.lower().split())
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [
        {"text": all_docs[i], "meta": all_metas[i] or {},
         "distance": 1.0 - (scores[i] / (scores[top_indices[0]] + 1e-9))}
        for i in top_indices
        if scores[i] > 0
    ]


def _rerank(question: str, hits: list[dict]) -> list[dict]:
    """Re-score every hit against the question using the embedding model.

    BM25 hits arrive with a fake 'distance' derived from BM25 score, which is
    not comparable to cosine distance. Re-embedding gives every hit the same
    distance metric so we can rank them honestly.
    """
    if not hits:
        return hits
    q_emb = embedder().encode([question], normalize_embeddings=True)[0]
    doc_embs = embedder().encode(
        [h["text"] for h in hits], normalize_embeddings=True, show_progress_bar=False
    )
    for h, d_emb in zip(hits, doc_embs):
        # Cosine distance = 1 - cosine_similarity (lower = better)
        sim = float((q_emb * d_emb).sum())
        h["rerank_distance"] = 1.0 - sim
    return sorted(hits, key=lambda h: h["rerank_distance"])


# Hits below this rerank distance are kept; chunks above it are dropped from
# the user-facing source list (still considered for synthesis).
RELEVANCE_THRESHOLD = 0.45  # cosine distance; ~similarity > 0.55


def retrieve(question: str, k: int = TOP_K) -> list[dict]:
    seen: dict[str, dict] = {}
    for q in _expand_queries(question):
        for hit in _semantic_retrieve(q, k):
            key = hit["text"][:60]
            if key not in seen or hit["distance"] < seen[key]["distance"]:
                seen[key] = hit
    for hit in _bm25_retrieve(question, k):
        key = hit["text"][:60]
        if key not in seen:
            seen[key] = hit
    results = list(seen.values())
    # Re-rank everything against the question with the same metric
    return _rerank(question, results)
