"""
FastAPI backend for CarnaticSearch.

Endpoints:
  POST /search        — query → top pick + shortlist
  POST /feedback      — submit ground truth (logs to feedback_log.jsonl)
  POST /retrain       — retrain LGBMRanker from feedback log
  GET  /stats         — feedback log accuracy stats
  GET  /              — serves the frontend (static)

Run:
  cd backend && ../.venv/bin/uvicorn app:app --reload --port 8000
"""

import json
import os
import sys
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Make the searcher importable
ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(ROOT, 'llm_searcher'))

import bm_ml_v2 as searcher  # noqa: E402

FEEDBACK_LOG = os.path.join(ROOT, 'feedback_log.jsonl')
FRONTEND_DIR = os.path.join(ROOT, 'frontend')

# Precompute unique (song, composer) list once at startup. searcher.metadata
# has ~70k entries because the corpus is chunked; we dedupe to ~thousands.
_UNIQUE_SONGS = []
_seen = set()
for _rec in searcher.metadata:
    _key = (_rec.get("Song Name", ""), _rec.get("Composer", ""))
    if _key in _seen:
        continue
    _seen.add(_key)
    _UNIQUE_SONGS.append({"song": _key[0], "composer": _key[1]})
del _seen

app = FastAPI(title="CarnaticSearch API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------
# Schemas
# -----------------------
class SearchRequest(BaseModel):
    query: str
    shortlist_size: int = 5


class SongResult(BaseModel):
    song: str
    composer: str
    lyrics: str
    score: float


class SearchResponse(BaseModel):
    query: str
    top_pick: SongResult
    shortlist: list[SongResult]


class FeedbackRequest(BaseModel):
    query: str
    correct_song: str
    shortlist: list[dict]
    ml_pick: str


# -----------------------
# Endpoints
# -----------------------
@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Empty query")

    best, shortlist = searcher.search(req.query, shortlist_size=req.shortlist_size)
    return SearchResponse(
        query=req.query,
        top_pick=SongResult(**best),
        shortlist=[SongResult(**s) for s in shortlist],
    )


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    labels = [
        1 if s.get("song", "").strip().lower() == req.correct_song.strip().lower() else 0
        for s in req.shortlist
    ]
    entry = {
        "query": req.query,
        "correct_song": req.correct_song,
        "candidates": req.shortlist,
        "labels": labels,
        "correct_in_shortlist": 1 in labels,
        "ml_pick": req.ml_pick,
        "ml_correct": req.ml_pick.strip().lower() == req.correct_song.strip().lower(),
    }
    with open(FEEDBACK_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return {"ok": True, **{k: entry[k] for k in ("ml_correct", "correct_in_shortlist")}}


@app.post("/retrain")
def retrain():
    ranker = searcher.retrain(min_samples=1)
    return {"trained": ranker is not None}


@app.get("/songs")
def songs(q: str = "", limit: int = 10):
    """Substring lookup over song name + composer for the typeahead picker."""
    q_lower = q.strip().lower()
    if not q_lower:
        return []
    results = []
    for rec in _UNIQUE_SONGS:
        if q_lower in rec["song"].lower() or q_lower in rec["composer"].lower():
            results.append(rec)
            if len(results) >= limit:
                break
    return results


@app.get("/stats")
def stats():
    if not os.path.exists(FEEDBACK_LOG):
        return {"total": 0, "ml_correct": 0, "in_shortlist": 0}

    total = ml_correct = in_shortlist = 0
    with open(FEEDBACK_LOG) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            total += 1
            if d.get("ml_correct"):
                ml_correct += 1
            if d.get("correct_in_shortlist"):
                in_shortlist += 1
    return {
        "total": total,
        "ml_correct": ml_correct,
        "in_shortlist": in_shortlist,
        "ml_accuracy_pct": round(100 * ml_correct / total, 1) if total else 0,
        "shortlist_recall_pct": round(100 * in_shortlist / total, 1) if total else 0,
    }


# Serve frontend if it exists (mounted last so /search etc. take priority)
if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
