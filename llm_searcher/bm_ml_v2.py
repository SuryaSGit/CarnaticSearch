import re
import unicodedata
import json
import numpy as np
import os
import pickle
from rank_bm25 import BM25Okapi
from lightgbm import LGBMRanker


# -----------------------
# Normalization
# -----------------------
def normalize(text: str) -> str:
    text = text.lower()

    text = unicodedata.normalize('NFKD', text)
    text = ''.join(c for c in text if not unicodedata.combining(c))

    # Order matters: longer patterns before shorter ones
    replacements = [
        # Diphthongs (before vowel-length rules to avoid partial collapses)
        ("ai", "a"), ("au", "a"),
        # Vowel length
        ("aa", "a"), ("ee", "i"), ("ii", "i"), ("oo", "u"), ("uu", "u"),
        # Aspirate consonants
        ("bh", "b"), ("dh", "d"), ("gh", "g"), ("jh", "j"),
        ("kh", "k"), ("ph", "p"), ("th", "t"),
        # Sibilants / nasals / retroflex (Unicode + ASCII forms)
        ("sh", "s"), ("ṣ", "s"), ("ś", "s"),
        ("ṅ", "n"), ("ñ", "n"), ("ṇ", "n"),
        ("ṭ", "t"), ("ḍ", "d"),
    ]
    for src, dst in replacements:
        text = text.replace(src, dst)

    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# -----------------------
# Tokenization
# -----------------------
def tokenize(text: str) -> list:
    """
    Word tokens + character n-grams (3–4) + word bigrams.

    Character n-grams let BM25 match phonetic variants that share substrings
    even after normalization (e.g. 'rama' ↔ 'raman' share 'ram', 'ama').
    Word bigrams capture contiguous phrase context.
    """
    words = text.split()
    tokens = list(words)

    for word in words:
        for n in (3, 4):
            for i in range(len(word) - n + 1):
                tokens.append(word[i:i + n])

    for i in range(len(words) - 1):
        tokens.append(f"{words[i]}_{words[i + 1]}")

    return tokens


# -----------------------
# Load Data
# -----------------------
_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'song_data', 'data.json')

with open(_DATA_PATH, 'r') as f:
    data = json.load(f)

docs = []        # token lists fed to BM25
metadata = []    # parallel song records

WINDOW = 10
STRIDE = 5

for song in data:
    text = f"{song['Song Name']} {song['Composer']} {song['Lyrics']}"
    norm = normalize(text)
    words = norm.split()

    for i in range(0, max(1, len(words) - WINDOW + 1), STRIDE):
        chunk_words = words[i:i + WINDOW]
        if len(chunk_words) < 5:
            continue
        docs.append(tokenize(" ".join(chunk_words)))
        metadata.append(song)

bm25 = BM25Okapi(docs)


# -----------------------
# Stage 1: BM25 → top_k
# -----------------------
def search_bm25(query: str, top_k: int = 20) -> list:
    norm_q = normalize(query)
    query_tokens = tokenize(norm_q)

    scores = bm25.get_scores(query_tokens)
    top_indices = np.argsort(scores)[::-1][:top_k * 4]  # oversample then dedupe

    results = []
    seen = set()

    for idx in top_indices:
        song = metadata[idx]
        key = (song["Song Name"], song["Composer"])
        if key in seen:
            continue

        doc_text = " ".join(w for w in docs[idx] if '_' not in w and len(w) > 4)
        phrase_bonus = 5 if norm_q in doc_text else 0
        score = float(scores[idx]) + phrase_bonus

        seen.add(key)
        results.append({
            "song": song["Song Name"],
            "composer": song["Composer"],
            "lyrics": song["Lyrics"],
            "score": score,
        })

        if len(results) == top_k:
            break

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


# -----------------------
# Feature Extraction
# -----------------------
def _char_ngrams(text: str, n: int) -> set:
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def extract_features(query: str, candidate: dict) -> list:
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

    return [
        candidate.get("score", 0.0),                             # BM25 score
        len(q_words & l_words) / (len(q_words) + 1),            # word overlap: lyrics
        len(q_words & t_words) / (len(q_words) + 1),            # word overlap: title
        1 if norm_q in norm_lyrics else 0,                       # exact phrase in lyrics
        1 if norm_q in norm_title else 0,                        # exact phrase in title
        len(q3 & l3) / (len(q3) + 1),                           # trigram overlap: lyrics
        len(q3 & t3) / (len(q3) + 1),                           # trigram overlap: title
        len(q4 & l4) / (len(q4) + 1),                           # quadgram overlap: lyrics
        len(candidate["lyrics"]),                                 # lyrics length
    ]


# -----------------------
# ML Ranker: Load / Save / Train
# -----------------------
RANKER_PATH = "ranker_v2.pkl"


def load_ranker():
    if os.path.exists(RANKER_PATH):
        with open(RANKER_PATH, "rb") as f:
            return pickle.load(f)
    return None


def save_ranker(ranker):
    with open(RANKER_PATH, "wb") as f:
        pickle.dump(ranker, f)


def retrain(min_samples: int = 50):
    if not os.path.exists("feedback_log.jsonl"):
        print("No feedback log found.")
        return None

    training_data = []
    with open("feedback_log.jsonl", "r") as f:
        for line in f:
            line = line.strip()
            if line:
                training_data.append(json.loads(line))

    usable = [d for d in training_data if d.get("correct_in_shortlist")]

    if len(usable) < min_samples:
        print(f"Only {len(usable)} usable samples — need {min_samples} to train. Keep collecting!")
        return None

    X, y, groups = [], [], []
    for entry in usable:
        features = [extract_features(entry["query"], c) for c in entry["candidates"]]
        X.extend(features)
        y.extend(entry["labels"])
        groups.append(len(entry["candidates"]))

    ranker = LGBMRanker(objective="lambdarank", n_estimators=100)
    ranker.fit(np.array(X), np.array(y), group=groups)

    save_ranker(ranker)
    print(f"Trained on {len(usable)} queries — saved to {RANKER_PATH}")
    return ranker


# -----------------------
# Stage 2: ML Ranker
# -----------------------
def ml_rank(query: str, candidates: list) -> tuple:
    """Re-rank candidates; returns (best_candidate, ranked_list)."""
    ranker = load_ranker()

    if ranker is None:
        return candidates[0], candidates

    features = np.array([extract_features(query, c) for c in candidates])
    scores = ranker.predict(features)
    order = np.argsort(scores)[::-1]
    ranked = [candidates[i] for i in order]
    return ranked[0], ranked


# -----------------------
# Full Pipeline
# -----------------------
def search(query: str, shortlist_size: int = 5):
    candidates = search_bm25(query, top_k=20)
    best, ranked = ml_rank(query, candidates)
    return best, ranked[:shortlist_size]


# -----------------------
# Interactive Session
# -----------------------
def interactive_session():
    print("=== Carnatic Search ===\n")
    print("Commands: 'quit' to exit | 'retrain' to train ML model\n")

    while True:
        query = input("Query: ").strip()

        if query == "quit":
            break
        if query == "retrain":
            retrain(min_samples=50)
            continue
        if not query:
            continue

        candidates = search_bm25(query, top_k=20)
        best, ranked = ml_rank(query, candidates)
        shortlist = ranked[:5]

        print("\nShortlist:")
        for i, s in enumerate(shortlist):
            marker = " <-- ML pick" if s["song"] == best["song"] else ""
            print(f"  {i + 1}. {s['song']} — {s['composer']}{marker}")

        print(f"\nTop pick: {best['song']} — {best['composer']}")

        correct = input("\nCorrect song? (Enter if top pick is right, or type the name): ").strip()
        if correct == "":
            correct = best["song"]

        labels = [1 if c["song"].lower() == correct.lower() else 0 for c in shortlist]
        entry = {
            "query": query,
            "correct_song": correct,
            "candidates": shortlist,
            "labels": labels,
            "correct_in_shortlist": 1 in labels,
            "ml_pick": best["song"],
            "ml_correct": best["song"].lower() == correct.lower(),
        }

        with open("feedback_log.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")

        if entry["ml_correct"]:
            print("ML got it right!")
        elif entry["correct_in_shortlist"]:
            print("Correct was in shortlist but ML ranked it wrong — good training signal")
        else:
            print("Correct wasn't in shortlist — BM25 stage needs work")

        _print_stats()
        print()


def _print_stats():
    if not os.path.exists("feedback_log.jsonl"):
        return

    total = ml_correct = in_shortlist = 0
    with open("feedback_log.jsonl", "r") as f:
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

    if total > 0:
        print(f"\nStats ({total} queries): "
              f"ML accuracy {ml_correct}/{total} ({100 * ml_correct // total}%) | "
              f"Shortlist recall {in_shortlist}/{total} ({100 * in_shortlist // total}%)")


# -----------------------
# Entry Point
# -----------------------
if __name__ == "__main__":
    interactive_session()
