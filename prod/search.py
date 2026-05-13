"""
Production search module for the HF Space deploy.

Identical search logic to llm_searcher/bm_ml_v2.py — kept self-contained so
the Space can run without the rest of the development repo. Paths default
to file-relative locations and can be overridden via env vars:

  CARNATIC_DATA_PATH      — corpus JSON (default: ./song_data/data.json)
  CARNATIC_STATE_DIR      — writable dir for ranker + feedback log
                            (default: ./state, override to /data on HF
                             Spaces if persistent storage is enabled)
"""

import os
import re
import json
import pickle
import unicodedata

import numpy as np
from rank_bm25 import BM25Okapi
from lightgbm import LGBMRanker


_HERE       = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.environ.get("CARNATIC_DATA_PATH", os.path.join(_HERE, "song_data", "data.json"))
STATE_DIR   = os.environ.get("CARNATIC_STATE_DIR", os.path.join(_HERE, "state"))
os.makedirs(STATE_DIR, exist_ok=True)

RANKER_PATH   = os.path.join(STATE_DIR, "ranker_v3.pkl")
FEEDBACK_PATH = os.path.join(STATE_DIR, "feedback_log.jsonl")


# -----------------------
# Normalization
# -----------------------
def normalize(text: str) -> str:
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))

    replacements = [
        ("ai", "a"), ("au", "a"),
        ("aa", "a"), ("ee", "i"), ("ii", "i"), ("oo", "u"), ("uu", "u"),
        ("bh", "b"), ("dh", "d"), ("gh", "g"), ("jh", "j"),
        ("kh", "k"), ("ph", "p"), ("th", "t"),
        ("sh", "s"), ("ṣ", "s"), ("ś", "s"),
        ("ṅ", "n"), ("ñ", "n"), ("ṇ", "n"),
        ("ṭ", "t"), ("ḍ", "d"),
        ("w", "v"),
    ]
    for src, dst in replacements:
        text = text.replace(src, dst)

    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# -----------------------
# Tokenization
# -----------------------
def tokenize(text: str) -> list:
    words = text.split()
    tokens = list(words)

    for word in words:
        for n in (3, 4):
            for i in range(len(word) - n + 1):
                tokens.append(word[i:i + n])

    for i in range(len(words) - 1):
        tokens.append(f"{words[i]}_{words[i + 1]}")
        tokens.append(f"{words[i]}{words[i + 1]}")

    return tokens


# -----------------------
# Index build (at import)
# -----------------------
with open(DATA_PATH, "r") as f:
    data = json.load(f)

docs = []
metadata = []
WINDOW = 10
STRIDE = 5

for song in data:
    text = f"{song['Song Name']} {song['Composer']} {song['Lyrics']}"
    words = normalize(text).split()
    for i in range(0, max(1, len(words) - WINDOW + 1), STRIDE):
        chunk_words = words[i:i + WINDOW]
        if len(chunk_words) < 5:
            continue
        docs.append(tokenize(" ".join(chunk_words)))
        metadata.append(song)

bm25 = BM25Okapi(docs)


# -----------------------
# BM25 retrieval
# -----------------------
def search_bm25(query: str, top_k: int = 20) -> list:
    norm_q = normalize(query)
    query_tokens = tokenize(norm_q)
    scores = bm25.get_scores(query_tokens)
    top_indices = np.argsort(scores)[::-1][:top_k * 4]

    results = []
    seen = set()
    for idx in top_indices:
        song = metadata[idx]
        key = (song["Song Name"], song["Composer"])
        if key in seen:
            continue
        doc_text = " ".join(w for w in docs[idx] if "_" not in w and len(w) > 4)
        phrase_bonus = 5 if norm_q in doc_text else 0
        results.append({
            "song": song["Song Name"],
            "composer": song["Composer"],
            "lyrics": song["Lyrics"],
            "score": float(scores[idx]) + phrase_bonus,
        })
        seen.add(key)
        if len(results) == top_k:
            break

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


# -----------------------
# Features
# -----------------------
def _char_ngrams(text: str, n: int) -> set:
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def extract_features(query: str, candidate: dict, qs_counts: dict = None) -> list:
    norm_q = normalize(query)
    norm_lyrics = normalize(candidate["lyrics"])
    norm_title = normalize(candidate["song"])

    q_words = set(norm_q.split())
    l_words = set(norm_lyrics.split())
    t_words = set(norm_title.split())

    q3 = _char_ngrams(norm_q, 3)
    l3 = _char_ngrams(norm_lyrics, 3)
    t3 = _char_ngrams(norm_title, 3)
    q4 = _char_ngrams(norm_q, 4)
    l4 = _char_ngrams(norm_lyrics, 4)

    pick_count = 0
    if qs_counts is not None:
        pick_count = qs_counts.get(norm_q, {}).get(candidate["song"], 0)

    return [
        candidate.get("score", 0.0),
        len(q_words & l_words) / (len(q_words) + 1),
        len(q_words & t_words) / (len(q_words) + 1),
        1 if norm_q in norm_lyrics else 0,
        1 if norm_q in norm_title else 0,
        len(q3 & l3) / (len(q3) + 1),
        len(q3 & t3) / (len(q3) + 1),
        len(q4 & l4) / (len(q4) + 1),
        len(candidate["lyrics"]),
        pick_count,
    ]


# -----------------------
# Ranker persistence + load
# -----------------------
def load_ranker():
    if os.path.exists(RANKER_PATH):
        with open(RANKER_PATH, "rb") as f:
            return pickle.load(f)
    return None


def save_ranker(ranker):
    with open(RANKER_PATH, "wb") as f:
        pickle.dump(ranker, f)


def _load_query_song_counts() -> dict:
    counts = {}
    if not os.path.exists(FEEDBACK_PATH):
        return counts
    with open(FEEDBACK_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            q = normalize(entry.get("query", ""))
            song = entry.get("correct_song", "")
            if not q or not song:
                continue
            counts.setdefault(q, {})
            counts[q][song] = counts[q].get(song, 0) + 1
    return counts


def retrain(min_samples: int = 1):
    if not os.path.exists(FEEDBACK_PATH):
        return None

    training = []
    with open(FEEDBACK_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                training.append(json.loads(line))

    usable = [d for d in training if d.get("correct_in_shortlist")]
    if len(usable) < min_samples:
        return None

    qs_counts = _load_query_song_counts()
    X, y, groups = [], [], []
    for entry in usable:
        feats = [extract_features(entry["query"], c, qs_counts) for c in entry["candidates"]]
        X.extend(feats)
        y.extend(entry["labels"])
        groups.append(len(entry["candidates"]))

    ranker = LGBMRanker(objective="lambdarank", n_estimators=100)
    ranker.fit(np.array(X), np.array(y), group=groups)
    save_ranker(ranker)
    return ranker


# -----------------------
# Blend ranker as bounded nudge
# -----------------------
RANKER_WEIGHT = 0.2


def ml_rank(query: str, candidates: list) -> tuple:
    if not candidates:
        return None, []

    bm25 = np.array([c["score"] for c in candidates], dtype=float)

    ranker = load_ranker()
    if ranker is None:
        return candidates[0], candidates

    qs_counts = _load_query_song_counts()
    features = np.array([extract_features(query, c, qs_counts) for c in candidates])
    ranker_scores = ranker.predict(features)

    bm25_spread = max(float(bm25.max() - bm25.min()), 1.0)
    r_spread = float(ranker_scores.max() - ranker_scores.min())
    if r_spread > 0:
        ranker_norm = (ranker_scores - ranker_scores.min()) / r_spread
        nudge = ranker_norm * RANKER_WEIGHT * bm25_spread
    else:
        nudge = np.zeros_like(bm25)

    final = bm25 + nudge
    order = np.argsort(final)[::-1]
    ranked = [candidates[i] for i in order]
    return ranked[0], ranked


def search(query: str, shortlist_size: int = 5):
    candidates = search_bm25(query, top_k=20)
    best, ranked = ml_rank(query, candidates)
    return best, ranked[:shortlist_size]
