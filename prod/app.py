"""FastAPI app for the HF Space deploy."""

import os
import json
import logging
import tempfile
import subprocess
import time
import uuid
import threading
import glob
from collections import OrderedDict
from typing import Optional

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


@app.middleware("http")
async def access_log(request: Request, call_next):
    """One INFO line per request with method, path, status, and duration.
    Unhandled exceptions are logged with traceback so they're visible in
    the Space's Logs tab even when the client just sees a 500."""
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        elapsed_ms = (time.perf_counter() - start) * 1000
        log.exception("%s %s -> 500 (%.0fms) UNHANDLED", request.method, request.url.path, elapsed_ms)
        raise
    elapsed_ms = (time.perf_counter() - start) * 1000
    quiet = request.url.path in {"/style.css", "/app.js", "/stats", "/"} \
            or request.url.path.startswith("/composers") \
            or request.url.path.startswith("/raagams") \
            or request.url.path.startswith("/songs")
    level = logging.DEBUG if quiet else logging.INFO
    log.log(level, "%s %s -> %d (%.0fms)", request.method, request.url.path, response.status_code, elapsed_ms)
    return response

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
    inject_path = "shortlist"
    if not any(_norm_song(s.get("song", "")) == correct_norm for s in candidates):
        bm25_top = search.search_bm25(req.query, top_k=40)
        match = next((c for c in bm25_top if _norm_song(c["song"]) == correct_norm), None)
        if match is not None:
            candidates.append(match)
            inject_path = "bm25-top-40"
        else:
            stub = _lookup_song(req.correct_song)
            if stub is None:
                stub = {"song": req.correct_song, "composer": "", "lyrics": "", "score": 0.0}
                inject_path = "missing-stub"
            else:
                inject_path = "corpus-stub"
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
    log.info("feedback q=%r correct=%r path=%s ml_correct=%s",
             req.query, req.correct_song, inject_path, entry["ml_correct"])
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
    log.info("revoke q=%r song=%r", req.query, req.correct_song)
    return {"ok": True}


@app.post("/retrain")
def retrain():
    start = time.perf_counter()
    ranker = search.retrain(min_samples=1)
    log.info("retrain trained=%s (%.0fms)", ranker is not None, (time.perf_counter() - start) * 1000)
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


_whisper_model = None
WHISPER_MODEL_NAME = os.environ.get("WHISPER_MODEL", "small")  # tiny|base|small|medium|large-v3
# Force English/Latin output so transcripts match the English-transliterated
# corpus; without this, Whisper outputs native script (Telugu/Tamil/etc.).
WHISPER_LANGUAGE = os.environ.get("WHISPER_LANGUAGE", "en")
# Cap audio length so transcription completes inside HF Space's edge timeout
# (~60s on free tier). 30s is enough to identify lyrics for search.
MAX_AUDIO_SECONDS = int(os.environ.get("WHISPER_MAX_SECONDS", "30"))
# Carnatic vocabulary primer for Whisper — biases the model toward the
# kind of transliterated word-shapes our corpus uses.
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
    """Transcribe an audio upload (mp3/wav/m4a/etc.) via faster-whisper.
    Returns the text so the frontend can drop it into the search box."""
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

    try:
        # Use faster-whisper's PyAV-backed decoder — no system ffmpeg needed.
        t_decode = time.perf_counter()
        try:
            from faster_whisper.audio import decode_audio
            audio = decode_audio(tmp_in_path, sampling_rate=16000)
        except Exception as e:
            log.exception("transcribe decode_audio failed")
            raise HTTPException(status_code=400, detail=f"Audio decode failed: {e}")

        max_samples = MAX_AUDIO_SECONDS * 16000
        clipped = audio[:max_samples]
        log.info("transcribe decoded %.1fs in %.2fs; truncated to %.1fs",
                 len(audio) / 16000.0, time.perf_counter() - t_decode,
                 len(clipped) / 16000.0)

        try:
            model = _get_whisper()
        except Exception:
            log.exception("transcribe whisper model load failed")
            raise HTTPException(status_code=500, detail="Whisper model load failed (check server logs)")

        t_transcribe = time.perf_counter()
        try:
            segments, info = model.transcribe(
                clipped,
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
        try:
            os.unlink(tmp_in_path)
        except OSError:
            pass


# -----------------------
# /identify — chunked audio → vote across chunks → top match
# -----------------------
IDENTIFY_CHUNK_SECONDS = int(os.environ.get("IDENTIFY_CHUNK_SECONDS", "15"))
IDENTIFY_MAX_JOBS_RETAINED = 50

_jobs_lock = threading.Lock()
_jobs: "OrderedDict[str, dict]" = OrderedDict()


def _jobs_put(job_id: str, job: dict):
    with _jobs_lock:
        _jobs[job_id] = job
        while len(_jobs) > IDENTIFY_MAX_JOBS_RETAINED:
            _jobs.popitem(last=False)


def _jobs_get(job_id: str) -> Optional[dict]:
    with _jobs_lock:
        return _jobs.get(job_id)


def _jobs_update(job_id: str, **kwargs):
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(kwargs)


def _audio_duration(path: str) -> float:
    """Audio duration via PyAV (bundled with faster-whisper). Avoids
    shelling out to ffprobe which isn't always present in the container
    even when ffmpeg is. Returns 0.0 on failure — callers treat as
    informational only."""
    try:
        import av  # bundled by faster-whisper
        with av.open(path) as container:
            if container.duration:
                return container.duration / 1_000_000
    except Exception as e:
        log.warning("audio duration probe failed: %s", e)
    return 0.0


def _aggregate_chunks(per_chunk: list) -> dict:
    """Vote across per-chunk search results: most-chunk-appearances wins,
    ties broken by rank-decayed score sum."""
    votes: dict = {}
    for chunk_idx, ch in enumerate(per_chunk):
        for rank, r in enumerate(ch["shortlist"][:5]):
            sn = r["song"]
            if sn not in votes:
                votes[sn] = {
                    "song": sn,
                    "composer": r["composer"],
                    "lyrics": r["lyrics"],
                    "chunks": [],
                    "weighted_score": 0.0,
                }
            votes[sn]["chunks"].append(chunk_idx)
            votes[sn]["weighted_score"] += r["score"] * (1.0 - 0.2 * rank)
    if not votes:
        return {"top": None, "runners_up": []}
    ranked = sorted(votes.values(),
                    key=lambda v: (-len(v["chunks"]), -v["weighted_score"]))
    top = ranked[0]
    return {
        "top": top,
        "runners_up": ranked[1:4],
        "chunks_total": len(per_chunk),
        "chunks_matched": len(top["chunks"]),
    }


def _run_identify(job_id: str, audio_path: str):
    """Background worker wrapper: outer catch-all + temp file cleanup.
    Without this, an uncaught exception leaves the job stuck at queued."""
    work_dir = audio_path + ".chunks"
    try:
        try:
            _do_identify(job_id, audio_path, work_dir)
        except Exception as e:
            log.exception("identify[%s] UNCAUGHT in worker", job_id)
            _jobs_update(job_id, status="error",
                         error=f"{type(e).__name__}: {e}")
    finally:
        try:
            os.unlink(audio_path)
        except OSError:
            pass
        for f in glob.glob(os.path.join(work_dir, "*")):
            try:
                os.unlink(f)
            except OSError:
                pass
        try:
            os.rmdir(work_dir)
        except OSError:
            pass


def _do_identify(job_id: str, audio_path: str, work_dir: str):
    os.makedirs(work_dir, exist_ok=True)

    _jobs_update(job_id, status="probing")
    log.info("identify[%s] decoding audio", job_id)
    t_decode = time.perf_counter()
    try:
        from faster_whisper.audio import decode_audio
        audio = decode_audio(audio_path, sampling_rate=16000)
    except Exception as e:
        log.exception("identify[%s] decode_audio failed", job_id)
        _jobs_update(job_id, status="error", error=f"Could not decode audio: {e}")
        return
    duration = len(audio) / 16000.0
    log.info("identify[%s] decoded %.1fs of audio in %.1fs",
             job_id, duration, time.perf_counter() - t_decode)

    chunk_samples = IDENTIFY_CHUNK_SECONDS * 16000
    chunks = [audio[i:i + chunk_samples]
              for i in range(0, len(audio), chunk_samples)
              if len(audio[i:i + chunk_samples]) >= 16000]
    chunks_total = len(chunks)
    if chunks_total == 0:
        _jobs_update(job_id, status="error", error="Audio too short to chunk")
        log.error("identify[%s] no chunks (audio too short)", job_id)
        return

    _jobs_update(job_id, status="processing",
                 progress={"chunks_done": 0, "chunks_total": chunks_total},
                 chunks=[])
    log.info("identify[%s] %d chunks to process", job_id, chunks_total)

    try:
        model = _get_whisper()
    except Exception:
        log.exception("identify[%s] whisper load failed", job_id)
        _jobs_update(job_id, status="error", error="Whisper model load failed (see logs)")
        return

    per_chunk = []
    chunks_live = []
    for idx, chunk_audio in enumerate(chunks):
        t = time.perf_counter()
        try:
            segments, info = model.transcribe(
                chunk_audio,
                beam_size=5,
                language=WHISPER_LANGUAGE,
                initial_prompt=WHISPER_PROMPT,
            )
            transcript = " ".join(s.text.strip() for s in segments).strip()
        except Exception:
            log.exception("identify[%s] chunk %d transcribe failed", job_id, idx)
            transcript = ""

        top_pick_name = None
        if transcript:
            best, shortlist = search.search(transcript, shortlist_size=5)
            per_chunk.append({
                "chunk_index": idx, "transcript": transcript,
                "shortlist": shortlist, "top_pick": best,
            })
            top_pick_name = best["song"] if best else None
        else:
            per_chunk.append({
                "chunk_index": idx, "transcript": "",
                "shortlist": [], "top_pick": None,
            })

        elapsed = time.perf_counter() - t
        log.info("identify[%s] chunk %d/%d transcript_len=%d took=%.1fs top=%r preview=%r",
                 job_id, idx + 1, chunks_total, len(transcript), elapsed, top_pick_name, transcript[:60])

        chunks_live.append({
            "i": idx,
            "start_s": idx * IDENTIFY_CHUNK_SECONDS,
            "transcript": transcript,
            "top": top_pick_name,
            "took_s": round(elapsed, 1),
        })
        _jobs_update(
            job_id,
            progress={"chunks_done": idx + 1, "chunks_total": chunks_total},
            chunks=list(chunks_live),
        )

    result = _aggregate_chunks(per_chunk)
    top_song = result["top"]["song"] if result["top"] else None
    log.info("identify[%s] done top=%r matched=%d/%d",
             job_id, top_song, result.get("chunks_matched", 0), chunks_total)
    _jobs_update(job_id, status="done", result=result)


@app.post("/identify")
async def identify(file: UploadFile = File(...)):
    """Start a background identification job; returns job_id immediately."""
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty upload")

    job_id = uuid.uuid4().hex[:12]
    suffix = os.path.splitext(file.filename or "")[1] or ".mp3"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(contents)
        audio_path = tmp.name

    _jobs_put(job_id, {
        "status": "queued",
        "filename": file.filename,
        "size_kb": len(contents) // 1024,
        "created_at": time.time(),
    })
    log.info("identify[%s] queued filename=%r size=%dKB",
             job_id, file.filename, len(contents) // 1024)

    threading.Thread(target=_run_identify, args=(job_id, audio_path), daemon=True).start()
    return {"job_id": job_id, "status": "queued"}


@app.get("/identify/{job_id}")
def identify_status(job_id: str):
    job = _jobs_get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job_id (or already evicted)")
    return job


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
