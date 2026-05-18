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

import tempfile

from fastapi import FastAPI, HTTPException, UploadFile, File
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
    _UNIQUE_SONGS.append({
        "song": _key[0],
        "composer": _key[1],
        "raagam": _rec.get("Raagam", ""),
    })
del _seen

# Composer / raagam catalogs for the browse-by-X sections
_COMPOSERS = sorted({s["composer"] for s in _UNIQUE_SONGS if s["composer"]})
_RAAGAMS   = sorted({s["raagam"]   for s in _UNIQUE_SONGS if s["raagam"]})

# Reverse indexes for fast filter
_BY_COMPOSER: dict = {}
_BY_RAAGAM:   dict = {}
for _s in _UNIQUE_SONGS:
    if _s["composer"]:
        _BY_COMPOSER.setdefault(_s["composer"], []).append(_s)
    if _s["raagam"]:
        _BY_RAAGAM.setdefault(_s["raagam"], []).append(_s)

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


def _norm_song(name: str) -> str:
    return name.strip().lower()


def _lookup_song(name: str) -> dict | None:
    """Find the canonical record for a song name in the corpus (case-insensitive)."""
    target = _norm_song(name)
    for rec in searcher.metadata:
        if _norm_song(rec.get("Song Name", "")) == target:
            return {
                "song": rec["Song Name"],
                "composer": rec.get("Composer", ""),
                "lyrics": rec.get("Lyrics", ""),
                "score": 0.0,
            }
    return None


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    correct_norm = _norm_song(req.correct_song)
    candidates = list(req.shortlist)

    # If the user picked from the typeahead (song not in shortlist), inject it
    # into the candidate list so the entry has a positive label and retrain can
    # use it. Look up the song's actual BM25 score by re-running BM25 over the
    # query — gives the ranker a realistic feature value rather than a dummy.
    if not any(_norm_song(s.get("song", "")) == correct_norm for s in candidates):
        bm25_top = searcher.search_bm25(req.query, top_k=40)
        match = next((c for c in bm25_top if _norm_song(c["song"]) == correct_norm), None)
        if match is not None:
            candidates.append(match)
        else:
            # Not in BM25 top 40: still trainable, but pull the real lyrics
            # from the corpus so feature extraction has meaningful values.
            stub = _lookup_song(req.correct_song)
            if stub is None:
                stub = {"song": req.correct_song, "composer": "", "lyrics": "", "score": 0.0}
            candidates.append(stub)

    labels = [
        1 if _norm_song(s.get("song", "")) == correct_norm else 0
        for s in candidates
    ]
    entry = {
        "query": req.query,
        "correct_song": req.correct_song,
        "candidates": candidates,
        "labels": labels,
        "correct_in_shortlist": 1 in labels,
        "ml_pick": req.ml_pick,
        "ml_correct": req.ml_pick.strip().lower() == req.correct_song.strip().lower(),
    }
    with open(FEEDBACK_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return {"ok": True, **{k: entry[k] for k in ("ml_correct", "correct_in_shortlist")}}


class RevokeRequest(BaseModel):
    query: str
    correct_song: str


@app.post("/feedback/revoke")
def revoke_feedback(req: RevokeRequest):
    """Append a revoke entry — _load_query_song_counts decrements the count."""
    entry = {
        "query": req.query,
        "correct_song": req.correct_song,
        "revoke": True,
    }
    with open(FEEDBACK_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return {"ok": True}


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
            results.append({"song": rec["song"], "composer": rec["composer"]})
            if len(results) >= limit:
                break
    return results


@app.get("/composers")
def composers(q: str = "", limit: int = 20):
    """Typeahead over composer names. Empty q returns the first `limit`."""
    q_lower = q.strip().lower()
    if not q_lower:
        return [{"name": c, "song_count": len(_BY_COMPOSER[c])} for c in _COMPOSERS[:limit]]
    out = []
    for c in _COMPOSERS:
        if q_lower in c.lower():
            out.append({"name": c, "song_count": len(_BY_COMPOSER[c])})
            if len(out) >= limit:
                break
    return out


@app.get("/raagams")
def raagams(q: str = "", limit: int = 20):
    """Typeahead over raagam names. Empty q returns the first `limit`."""
    q_lower = q.strip().lower()
    if not q_lower:
        return [{"name": r, "song_count": len(_BY_RAAGAM[r])} for r in _RAAGAMS[:limit]]
    out = []
    for r in _RAAGAMS:
        if q_lower in r.lower():
            out.append({"name": r, "song_count": len(_BY_RAAGAM[r])})
            if len(out) >= limit:
                break
    return out


@app.get("/songs/by-composer")
def songs_by_composer(name: str, limit: int = 200):
    out = _BY_COMPOSER.get(name, [])
    return [{"song": s["song"], "composer": s["composer"], "raagam": s["raagam"]} for s in out[:limit]]


@app.get("/songs/by-raagam")
def songs_by_raagam(name: str, limit: int = 200):
    out = _BY_RAAGAM.get(name, [])
    return [{"song": s["song"], "composer": s["composer"], "raagam": s["raagam"]} for s in out[:limit]]


_whisper_model = None
WHISPER_MODEL_NAME = os.environ.get("WHISPER_MODEL", "small")  # tiny|base|small|medium|large-v3


def _get_whisper():
    """Lazy-load the Whisper model the first time it's needed."""
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        _whisper_model = WhisperModel(WHISPER_MODEL_NAME, device="cpu", compute_type="int8")
    return _whisper_model


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """Transcribe an uploaded audio file (mp3/wav/m4a/etc.) to text via
    faster-whisper, then return the transcript so the frontend can drop
    it into the search box."""
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty upload")

    suffix = os.path.splitext(file.filename or "")[1] or ".mp3"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        model = _get_whisper()
        segments, info = model.transcribe(tmp_path, beam_size=5)
        text = " ".join(s.text.strip() for s in segments).strip()
        return {
            "text": text,
            "language": info.language,
            "language_probability": round(info.language_probability, 3),
            "duration": round(info.duration, 1),
        }
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


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
