"""FastAPI app for the HF Space deploy."""

import os
import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import search

HERE         = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR   = os.path.join(HERE, "static")
FEEDBACK_LOG = search.FEEDBACK_PATH

app = FastAPI(title="CarnaticSearch", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Deduplicated (song, composer) list for the typeahead.
_UNIQUE_SONGS = []
_seen = set()
for _rec in search.metadata:
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

_COMPOSERS = sorted({s["composer"] for s in _UNIQUE_SONGS if s["composer"]})
_RAAGAMS   = sorted({s["raagam"]   for s in _UNIQUE_SONGS if s["raagam"]})
_BY_COMPOSER: dict = {}
_BY_RAAGAM:   dict = {}
for _s in _UNIQUE_SONGS:
    if _s["composer"]:
        _BY_COMPOSER.setdefault(_s["composer"], []).append(_s)
    if _s["raagam"]:
        _BY_RAAGAM.setdefault(_s["raagam"], []).append(_s)


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


@app.post("/search", response_model=SearchResponse)
def do_search(req: SearchRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Empty query")
    best, shortlist = search.search(req.query, shortlist_size=req.shortlist_size)
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
    for rec in search.metadata:
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
    # into the candidate list with its actual BM25 score (or 0 if not even in
    # BM25 top 40, with real lyrics looked up so feature extraction works).
    if not any(_norm_song(s.get("song", "")) == correct_norm for s in candidates):
        bm25_top = search.search_bm25(req.query, top_k=40)
        match = next((c for c in bm25_top if _norm_song(c["song"]) == correct_norm), None)
        if match is not None:
            candidates.append(match)
        else:
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
    return {"ok": True, "ml_correct": entry["ml_correct"], "correct_in_shortlist": entry["correct_in_shortlist"]}


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
    ranker = search.retrain(min_samples=1)
    return {"trained": ranker is not None}


@app.get("/songs")
def songs(q: str = "", limit: int = 10):
    q_lower = q.strip().lower()
    if not q_lower:
        return []
    out = []
    for rec in _UNIQUE_SONGS:
        if q_lower in rec["song"].lower() or q_lower in rec["composer"].lower():
            out.append({"song": rec["song"], "composer": rec["composer"]})
            if len(out) >= limit:
                break
    return out


@app.get("/composers")
def composers(q: str = "", limit: int = 20):
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


@app.get("/stats")
def stats():
    if not os.path.exists(FEEDBACK_LOG):
        return {"total": 0, "ml_correct": 0, "in_shortlist": 0,
                "ml_accuracy_pct": 0, "shortlist_recall_pct": 0}
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


# Mount frontend last so the API routes above take priority
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="frontend")
