#!/usr/bin/env python3
"""scrape.py -- Scrape Instagram @king_deltoids via Apify.

Fetches all posts (up to limit) from the Apify run and writes to SQLite.
If APIFY_RUN_ID is set in .env, uses that run; else starts a new one.

Excludes any post timestamped April 1 (any year).
"""
from __future__ import annotations

import json
import os
import ssl
import sys
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import db

ssl._create_default_https_context = ssl._create_unverified_context  # noqa: SLF001

ROOT = Path(__file__).resolve().parent

ACTOR_ID = "shu8hvrXbJbY3Eb9W"  # apify/instagram-scraper
PROFILE_URL = "https://www.instagram.com/king_deltoids/"
RESULTS_LIMIT = 750
POLL_INTERVAL_S = 15
POLL_TIMEOUT_S = 1800  # 30 min


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


def _http(method: str, url: str, token: str, body: dict | None = None) -> dict:
    data = None
    headers = {"Authorization": f"Bearer {token}"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = Request(url, data=data, method=method, headers=headers)
    with urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def start_run(token: str) -> dict:
    url = f"https://api.apify.com/v2/acts/{ACTOR_ID}/runs"
    payload = {
        "directUrls": [PROFILE_URL],
        "resultsType": "posts",
        "resultsLimit": RESULTS_LIMIT,
        "addParentData": False,
    }
    resp = _http("POST", url, token, payload)
    return resp["data"]


def wait_for_run(run_id: str, token: str) -> dict:
    url = f"https://api.apify.com/v2/actor-runs/{run_id}"
    deadline = time.time() + POLL_TIMEOUT_S
    while True:
        resp = _http("GET", url, token)
        run = resp["data"]
        status = run.get("status")
        print(f"[scrape] run {run_id} status={status}", flush=True)
        if status == "SUCCEEDED":
            return run
        if status in {"FAILED", "ABORTED", "TIMED-OUT"}:
            raise RuntimeError(f"Apify run ended with status={status}")
        if time.time() > deadline:
            raise TimeoutError(f"Apify run {run_id} did not finish within {POLL_TIMEOUT_S}s")
        time.sleep(POLL_INTERVAL_S)


def fetch_dataset_items(dataset_id: str, token: str) -> list[dict]:
    qs = urlencode({"clean": "true", "format": "json"})
    url = f"https://api.apify.com/v2/datasets/{dataset_id}/items?{qs}"
    req = Request(url, headers={"Authorization": f"Bearer {token}"})
    with urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode("utf-8"))


def normalize(item: dict) -> dict:
    hashtags = item.get("hashtags") or []
    if isinstance(hashtags, list):
        hashtags = ",".join(hashtags)
    return {
        "id": item.get("id") or item.get("shortCode"),
        "short_code": item.get("shortCode", ""),
        "url": item.get("url", ""),
        "post_type": item.get("type", ""),
        "caption": item.get("caption", "") or "",
        "hashtags": hashtags or "",
        "video_url": item.get("videoUrl", "") or "",
        "display_url": item.get("displayUrl", "") or "",
        "video_duration": float(item.get("videoDuration", 0) or 0),
        "likes_count": int(item.get("likesCount", 0) or 0),
        "comments_count": int(item.get("commentsCount", 0) or 0),
        "video_views": int(item.get("videoViewCount", 0) or 0),
        "timestamp": item.get("timestamp", "") or "",
    }


def is_april_first(ts: str) -> bool:
    """Match any post on April 1 (any year)."""
    if not ts:
        return False
    # ISO format e.g. "2026-04-01T12:00:00.000Z"
    return "04-01T" in ts or "-04-01 " in ts


def main() -> int:
    _load_dotenv()
    token = os.environ.get("APIFY_API_TOKEN")
    if not token:
        print("ERROR: APIFY_API_TOKEN not set", file=sys.stderr)
        return 1

    run_id = os.environ.get("APIFY_RUN_ID")
    if run_id:
        print(f"[scrape] using existing run {run_id}", flush=True)
        finished = wait_for_run(run_id, token)
    else:
        print(f"[scrape] starting actor {ACTOR_ID} for {PROFILE_URL}", flush=True)
        run = start_run(token)
        finished = wait_for_run(run["id"], token)

    dataset_id = finished["defaultDatasetId"]
    items = fetch_dataset_items(dataset_id, token)
    print(f"[scrape] fetched {len(items)} items from dataset {dataset_id}", flush=True)

    db.init_db()
    saved = 0
    skipped_apr1 = 0
    for it in items:
        if not (it.get("id") or it.get("shortCode")):
            continue
        v = normalize(it)
        db.upsert_post(v)
        if is_april_first(v["timestamp"]):
            db.mark_skipped(v["id"], "april_first_excluded")
            skipped_apr1 += 1
            print(f"[scrape] SKIP April 1 post {v['short_code']} ({v['timestamp']})", flush=True)
        saved += 1

    s = db.stats()
    print(f"[scrape] done. saved={saved} skipped_apr1={skipped_apr1} | total_db={s['total']} videos={s['videos']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
