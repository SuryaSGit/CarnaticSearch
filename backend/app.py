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
import logging
import os
import sys
import time
from typing import Optional

import tempfile
import subprocess

from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# -----------------------
# Logging
# -----------------------
LOG_LEVEL = os.environ.get("CARNATIC_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("carnatic")

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


@app.middleware("http")
async def access_log(request: Request, call_next):
    """One INFO line per request with method, path, status, and duration.
    On unhandled exceptions, log the traceback with the path so we can tie
    errors back to the offending request."""
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        elapsed_ms = (time.perf_counter() - start) * 1000
        log.exception("%s %s -> 500 (%.0fms) UNHANDLED", request.method, request.url.path, elapsed_ms)
        raise
    elapsed_ms = (time.perf_counter() - start) * 1000
    # Quiet down per-request noise for static frontend / poll endpoints
    quiet = request.url.path in {"/style.css", "/app.js", "/stats", "/"} \
            or request.url.path.startswith("/composers") \
            or request.url.path.startswith("/raagams") \
            or request.url.path.startswith("/songs")
    level = logging.DEBUG if quiet else logging.INFO
    log.log(level, "%s %s -> %d (%.0fms)", request.method, request.url.path, response.status_code, elapsed_ms)
    return response


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
    log.info("search q=%r top=%r shortlist=%d",
             req.query, best["song"], len(shortlist))
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
    inject_path = "shortlist"
    if not any(_norm_song(s.get("song", "")) == correct_norm for s in candidates):
        bm25_top = searcher.search_bm25(req.query, top_k=40)
        match = next((c for c in bm25_top if _norm_song(c["song"]) == correct_norm), None)
        if match is not None:
            candidates.append(match)
            inject_path = "bm25-top-40"
        else:
            # Not in BM25 top 40: still trainable, but pull the real lyrics
            # from the corpus so feature extraction has meaningful values.
            stub = _lookup_song(req.correct_song)
            if stub is None:
                stub = {"song": req.correct_song, "composer": "", "lyrics": "", "score": 0.0}
            candidates.append(stub)
            inject_path = "corpus-stub" if _lookup_song(req.correct_song) else "missing-stub"

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
    log.info("feedback q=%r correct=%r path=%s ml_correct=%s",
             req.query, req.correct_song, inject_path, entry["ml_correct"])
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
    log.info("revoke q=%r song=%r", req.query, req.correct_song)
    return {"ok": True}


@app.post("/retrain")
def retrain():
    start = time.perf_counter()
    ranker = searcher.retrain(min_samples=1)
    log.info("retrain trained=%s (%.0fms)", ranker is not None, (time.perf_counter() - start) * 1000)
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
# Force the transcription to come out in English/Latin letters so it matches
# the English-transliterated lyrics in the corpus.
WHISPER_LANGUAGE = os.environ.get("WHISPER_LANGUAGE", "en")
# Cap audio length so transcription completes inside HF Space's edge timeout
# (~60s for free tier). 30s is more than enough to identify lyrics for search.
MAX_AUDIO_SECONDS = int(os.environ.get("WHISPER_MAX_SECONDS", "30"))
# Biases the model toward Carnatic-style transliteration: deity names,
# composer/kriti title fragments, and common Sanskrit/Telugu lyric words.
# Whisper consumes initial_prompt as conversational context but does NOT
# put it in the output — only the word *shapes* leak into its guesses.
WHISPER_PROMPT = (
    "vAtApi gaNapatim bhajeham, raama nannu brOvarA, manasA sancararE, "
    "jagadAnanda kArakA, ninnE nammitinayyA, nagumOmu ganalEni, "
    "endharO mahaanubhaavulu, jananee ninnuvina, brOva baarama, "
    "shankari shankuru, kanaka na ruchirA, sArasAkSa pari pAlaya, "
    "krishna rama hari govinda gopala madhava narayana shiva ganesha "
    "lakshmi parvati saraswati durga ambika kAli; "
    "tyagaraja dikshitar shyama shastri muttuswamy "
    "namostute namami namaha jaya vande prabho deva swami pAhi"
)


def _get_whisper():
    """Lazy-load the Whisper model the first time it's needed."""
    global _whisper_model
    if _whisper_model is None:
        log.info("whisper loading model=%s (first call, this can take ~30s on cold start)",
                 WHISPER_MODEL_NAME)
        t = time.perf_counter()
        from faster_whisper import WhisperModel
        _whisper_model = WhisperModel(WHISPER_MODEL_NAME, device="cpu", compute_type="int8")
        log.info("whisper loaded model=%s in %.1fs", WHISPER_MODEL_NAME, time.perf_counter() - t)
    return _whisper_model


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """Transcribe an uploaded audio file (mp3/wav/m4a/etc.) to text via
    faster-whisper, then return the transcript so the frontend can drop
    it into the search box."""
    contents = await file.read()
    if not contents:
        log.warning("transcribe rejected: empty upload (filename=%r)", file.filename)
        raise HTTPException(status_code=400, detail="Empty upload")

    suffix = os.path.splitext(file.filename or "")[1] or ".mp3"
    upload_kb = len(contents) // 1024
    log.info("transcribe begin filename=%r suffix=%s size=%dKB", file.filename, suffix, upload_kb)

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in:
        tmp_in.write(contents)
        tmp_in_path = tmp_in.name

    # Truncate via ffmpeg to bound transcription time. Output is 16kHz mono
    # WAV (Whisper's native format) so model.transcribe spends zero time on
    # resampling/decoding either.
    tmp_out_path = tmp_in_path + ".clipped.wav"
    try:
        t_clip = time.perf_counter()
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-i", tmp_in_path,
                    "-t", str(MAX_AUDIO_SECONDS),
                    "-ar", "16000", "-ac", "1",
                    "-c:a", "pcm_s16le",
                    tmp_out_path,
                ],
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as e:
            stderr = (e.stderr or b"").decode(errors="replace")[:300]
            log.error("transcribe ffmpeg failed: rc=%d stderr=%s", e.returncode, stderr)
            raise HTTPException(status_code=400, detail=f"Audio decode failed: {stderr}")
        log.info("transcribe ffmpeg clipped to %ds in %.2fs (out=%dKB)",
                 MAX_AUDIO_SECONDS, time.perf_counter() - t_clip,
                 os.path.getsize(tmp_out_path) // 1024)

        try:
            model = _get_whisper()
        except Exception:
            log.exception("transcribe whisper model load failed")
            raise HTTPException(status_code=500, detail="Whisper model load failed (check server logs)")

        t_transcribe = time.perf_counter()
        try:
            segments, info = model.transcribe(
                tmp_out_path,
                beam_size=5,
                language=WHISPER_LANGUAGE,
                initial_prompt=WHISPER_PROMPT,
            )
            text = " ".join(s.text.strip() for s in segments).strip()
        except Exception:
            log.exception("transcribe whisper.transcribe failed")
            raise HTTPException(status_code=500, detail="Whisper transcription failed (check server logs)")

        elapsed = time.perf_counter() - t_transcribe
        log.info("transcribe done lang=%s prob=%.2f dur=%.1fs took=%.1fs text_len=%d preview=%r",
                 info.language, info.language_probability, info.duration, elapsed,
                 len(text), text[:80])
        return {
            "text": text,
            "language": info.language,
            "language_probability": round(info.language_probability, 3),
            "duration": round(info.duration, 1),
            "clipped_to_seconds": MAX_AUDIO_SECONDS,
        }
    finally:
        for p in (tmp_in_path, tmp_out_path):
            try:
                os.unlink(p)
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
