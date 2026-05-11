#!/usr/bin/env python3
"""api.py -- HTTP API for the king_deltoids chatbot.

Run: uvicorn api:app --port 8001
Or: python3 api.py
"""
from __future__ import annotations

import json
import os
import re

from openai import OpenAI
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from retrieve import retrieve, _load_dotenv
from kd_profile import KD_PROFILE

_load_dotenv()

app = FastAPI(title="King Deltoids Bot API")


@app.on_event("startup")
def _warmup():
    """Pre-load embedder + Chroma + BM25 cache so the first request isn't cold."""
    from retrieve import retrieve
    try:
        retrieve("warmup query about training")
        print("[startup] retrieval warmed", flush=True)
    except Exception as e:
        print(f"[startup] warmup failed: {e}", flush=True)

MAX_QUESTION_LEN = 300

_INJECTION_PATTERNS = re.compile(
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?"
    r"|forget\s+(everything|all|your\s+instructions?)"
    r"|you\s+are\s+now\s+(a\s+)?(?!king|kd|deltoids)"
    r"|act\s+as\s+(a\s+)?(?!king|kd|deltoids)"
    r"|new\s+instructions?"
    r"|disregard\s+(all\s+)?(previous|prior)?"
    r"|system\s*:\s*"
    r"|<\s*/?system\s*>"
    r"|\[INST\]|\[\/INST\]"
    r"|###\s*instruction"
    r"|jailbreak",
    re.IGNORECASE,
)


def _sanitize_input(text: str) -> str:
    text = text.strip()
    if len(text) > MAX_QUESTION_LEN:
        text = text[:MAX_QUESTION_LEN]
    text = _INJECTION_PATTERNS.sub("[removed]", text)
    return text


def _wrap_user_input(text: str) -> str:
    return f'<user_question>{text}</user_question>'


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DISAMBIGUATION_PROMPT = (
    "You are a fitness intent extractor. Given a user's question, identify the CORE fitness concept they want to understand.\n\n"
    "Rules:\n"
    "- Strip situational constraints (at home, no equipment, beginner, etc.) -- those are filters, not the core topic.\n"
    "- Strip framing, keep the actionable need.\n"
    "  e.g. 'how should I train shoulders?' -> 'shoulder hypertrophy training delts'\n"
    "  e.g. 'is creatine worth it?' -> 'creatine effectiveness supplementation'\n"
    "  e.g. 'best rep range for growth?' -> 'rep range hypertrophy muscle growth'\n"
    "- Always frame as what they WANT to know.\n"
    "- Use concrete fitness keywords likely to appear in short-form Instagram fitness content.\n"
    "- Output ONE short line: the core retrieval query. No preamble, no quotes."
)

SYNTHESIS_PROMPT = (
    "You are a fitness assistant that answers questions in the voice of king_deltoids (Instagram strength and hypertrophy coach), "
    "based on his Instagram post captions and on-screen video text. Below is his profile, then several retrieved excerpts, then "
    "the user's question.\n\n"
    "TASK:\n"
    "1. Determine if the excerpts directly address the question.\n"
    "2. If they do, write a 2-4 sentence answer using ONLY claims and numbers in the excerpts.\n"
    "3. If they don't, return NO_ANSWER.\n\n"
    "RULES:\n"
    "- Use ONLY claims/numbers present in the excerpts. Do NOT introduce numbers from the profile.\n"
    "- Each excerpt is labeled as CAPTION or ON-SCREEN TEXT. Both are king_deltoids' words.\n"
    "- Numbers must include unit and context (e.g. '8-15 reps', not just '8-15').\n"
    "- Match his direct, instructive tone -- concise, no fluff, no jargon overkill.\n"
    "- Don't sign off with anything -- he doesn't have a catchphrase.\n\n"
    "OUTPUT FORMAT (strict JSON, no preamble, no code fences):\n"
    "{\"answer\": \"<the answer or NO_ANSWER>\", \"confidence\": <integer 0-10>, \"used_excerpt_indices\": [<list of integer indices>]}\n\n"
    "Confidence rubric:\n"
    "- 9-10: excerpts directly and explicitly answer the question.\n"
    "- 6-8: excerpts contain his view on the topic.\n"
    "- 4-5: excerpts touch the topic but don't fully answer -- still answer with what's there.\n"
    "- 1-3: only a glancing mention -- try to answer briefly, mark low confidence.\n"
    "- 0: truly nothing relevant -- set answer to NO_ANSWER.\n\n"
    "IMPORTANT: Prefer a partial answer over NO_ANSWER. Only use NO_ANSWER if the excerpts are completely "
    "unrelated to the question topic.\n\n"
    + KD_PROFILE
)


class AskRequest(BaseModel):
    question: str


class Chunk(BaseModel):
    url: str
    description: str
    excerpt: str
    key_quote: str


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    chunks: list[Chunk]
    confidence: int


@app.get("/")
def health():
    return {"status": "ok", "service": "king-deltoids-bot"}


OFF_TOPIC_RESPONSE = (
    "I can only answer fitness, training, and nutrition questions. Try asking something like "
    "'how should I train shoulders?' or 'best rep range for hypertrophy?'"
)

TOPIC_GUARD_PROMPT = (
    "You are a topic classifier. Respond with only 'YES' or 'NO'.\n"
    "Is the following question related to fitness, nutrition, health, exercise, body composition, supplements, or wellness?\n"
    "Be generous -- if it's even loosely related, say YES."
)


def _is_fitness_related(client: OpenAI, question: str) -> bool:
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini", max_tokens=5, temperature=0,
            messages=[
                {"role": "system", "content": TOPIC_GUARD_PROMPT},
                {"role": "user", "content": _wrap_user_input(question)},
            ],
        )
        return r.choices[0].message.content.strip().upper().startswith("YES")
    except Exception:
        return True


def _disambiguate(client: OpenAI, question: str) -> str:
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini", max_tokens=80, temperature=0,
            messages=[
                {"role": "system", "content": DISAMBIGUATION_PROMPT},
                {"role": "user", "content": _wrap_user_input(question)},
            ],
        )
        return r.choices[0].message.content.strip().strip('"')
    except Exception:
        return question


def _retrieve_for(question: str, disambig: str) -> list[dict]:
    q = disambig if disambig.strip() and disambig.strip() != question.strip() else question
    return retrieve(q)


def _synthesize(client: OpenAI, question: str, hits: list[dict]) -> dict:
    excerpts_block = "\n\n".join(
        f"[Excerpt {i}]\n{h['text']}" for i, h in enumerate(hits)
    )
    user_msg = (
        f"Question: {_wrap_user_input(question)}\n\n"
        f"Retrieved excerpts:\n\n{excerpts_block}\n\n"
        "Output strict JSON only."
    )
    r = client.chat.completions.create(
        model="gpt-4o-mini", max_tokens=600, temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYNTHESIS_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )
    txt = r.choices[0].message.content.strip()
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", txt, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        return {"answer": "NO_ANSWER", "confidence": 0, "used_excerpt_indices": []}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    question = _sanitize_input(req.question)

    if not _is_fitness_related(client, question):
        return AskResponse(answer=OFF_TOPIC_RESPONSE, confidence=0, chunks=[], sources=[])

    disambig = _disambiguate(client, question)
    hits = _retrieve_for(question, disambig)
    if not hits:
        raise HTTPException(status_code=404, detail="No content indexed yet.")

    seen_urls: set[str] = set()
    unique_hits = []
    for h in hits:
        url = h["meta"].get("url", "")
        if url in seen_urls:
            continue
        seen_urls.add(url)
        unique_hits.append(h)
        if len(unique_hits) >= 8:
            break

    result = _synthesize(client, question, unique_hits)
    answer = (result.get("answer") or "").strip()
    confidence = int(result.get("confidence") or 0)
    used = result.get("used_excerpt_indices") or []

    if not answer or answer.upper().startswith("NO_ANSWER") or confidence <= 1:
        return AskResponse(
            answer=(
                "King Deltoids hasn't covered this specific topic in his indexed posts. "
                "Try asking about hypertrophy, rep ranges, shoulder training, programming, "
                "or progressive overload -- those are his strongest areas."
            ),
            sources=[],
            chunks=[],
            confidence=0,
        )

    chunks = []
    for idx in used:
        if not isinstance(idx, int) or idx < 0 or idx >= len(unique_hits):
            continue
        h = unique_hits[idx]
        url = h["meta"].get("url", "")
        chunks.append(Chunk(
            url=url,
            description=(h["meta"].get("caption_preview") or h["meta"].get("short_code", ""))[:140],
            excerpt=h["text"][:600],
            key_quote="",
        ))

    if not chunks and unique_hits:
        h = unique_hits[0]
        chunks.append(Chunk(
            url=h["meta"].get("url", ""),
            description=(h["meta"].get("caption_preview") or h["meta"].get("short_code", ""))[:140],
            excerpt=h["text"][:600],
            key_quote="",
        ))

    return AskResponse(
        answer=answer,
        sources=[c.url for c in chunks],
        chunks=chunks,
        confidence=confidence,
    )


STREAM_SYNTHESIS_PROMPT = (
    "You are answering in the voice of king_deltoids (Instagram strength and hypertrophy coach), "
    "based on his Instagram captions and on-screen video text below. Then several retrieved "
    "excerpts. Each excerpt has CAPTION and/or ON-SCREEN TEXT -- both are his words.\n\n"
    "TASK:\n"
    "1. If the excerpts address the question (even partially), write a 2-4 sentence answer "
    "using ONLY claims and numbers in the excerpts.\n"
    "2. If excerpts are completely unrelated, say: 'King Deltoids hasn't covered this specific "
    "topic in detail -- try asking about hypertrophy, rep ranges, training intensity, or programming.'\n\n"
    "RULES:\n"
    "- Use ONLY claims/numbers from the excerpts. Do NOT invent.\n"
    "- Numbers must include unit and context.\n"
    "- Match his direct, instructive tone -- concise, no fluff, no sign-off.\n"
    "- Prefer a partial answer over saying 'no info'. If even one excerpt mentions the topic, "
    "extract what's there.\n"
    "- Output ONLY the answer text. No JSON, no preamble, no quotes.\n\n"
    + KD_PROFILE
)


def _stream_synthesis(client: OpenAI, question: str, hits: list[dict]):
    """Yield chunks of {type, ...} as SSE-style JSON lines."""
    excerpts_block = "\n\n".join(
        f"[Excerpt {i}]\n{h['text']}" for i, h in enumerate(hits)
    )
    user_msg = (
        f"Question: {_wrap_user_input(question)}\n\n"
        f"Retrieved excerpts:\n\n{excerpts_block}"
    )
    stream = client.chat.completions.create(
        model="gpt-4o-mini", max_tokens=500, temperature=0,
        messages=[
            {"role": "system", "content": STREAM_SYNTHESIS_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            yield json.dumps({"type": "token", "text": delta}) + "\n"


@app.post("/ask-stream")
def ask_stream(req: AskRequest):
    """SSE-ish line-delimited JSON stream: sources first, then tokens."""
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    question = _sanitize_input(req.question)

    def gen():
        # Skip topic guard + disambiguation entirely for streaming -- the
        # synthesis prompt itself handles off-topic gracefully, and retrieval
        # is fast enough that we don't need a rewritten query.
        hits = _retrieve_for(question, question)

        seen_urls: set[str] = set()
        unique_hits = []
        for h in hits:
            url = h["meta"].get("url", "")
            if url in seen_urls:
                continue
            seen_urls.add(url)
            unique_hits.append(h)
            if len(unique_hits) >= 6:
                break

        # Send source chunks BEFORE streaming the answer -- the UI can render them immediately
        chunks = [
            {
                "url": h["meta"].get("url", ""),
                "description": (h["meta"].get("caption_preview") or h["meta"].get("short_code", ""))[:140],
                "excerpt": h["text"][:600],
            }
            for h in unique_hits
        ]
        yield json.dumps({"type": "sources", "chunks": chunks}) + "\n"

        if not unique_hits:
            yield json.dumps({"type": "token", "text": "No content indexed for this question yet."}) + "\n"
            yield json.dumps({"type": "done"}) + "\n"
            return

        # Stream the synthesis tokens
        for line in _stream_synthesis(client, question, unique_hits):
            yield line
        yield json.dumps({"type": "done"}) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8002, reload=False)
