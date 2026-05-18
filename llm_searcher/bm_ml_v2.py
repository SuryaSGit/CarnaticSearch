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
        # English 'w' is interchangeable with 'v' in Sanskrit/Telugu transliteration
        # (e.g. 'swami' / 'svami', 'swarupa' / 'svarupa').
        ("w", "v"),
    ]
    for src, dst in replacements:
        text = text.replace(src, dst)

    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# -----------------------
# Tokenization
# -----------------------
def tokenize(text: str, expand_splits: bool = False) -> list:
    """
    Word tokens + character n-grams (3–4) + word bigrams (separated and
    concatenated). Used for both indexing and querying.

    Character n-grams let BM25 match phonetic variants that share substrings
    even after normalization (e.g. 'rama' ↔ 'raman' share 'ram', 'ama').
    Word bigrams capture contiguous phrase context. The CONCATENATED bigram
    (no separator) handles the common Carnatic-lyric case where adjacent
    words are written as one — e.g. 'pranavasvarupa' in the lyric matching
    'pranava svarupa' in the query.

    expand_splits=True (queries only): for each query word >5 chars, also
    emit every 2-piece split. This is the inverse of the concat bigram —
    when the *query* glues two words together (e.g. 'munimanasa') we emit
    all candidate splits ('muni'/'manasa', etc.) so the real ones match
    individual lyric words. Splits that don't form real words just have
    no effect (no doc has them, so BM25 contributes 0).
    """
    words = text.split()
    tokens = list(words)

    for word in words:
        for n in (3, 4):
            for i in range(len(word) - n + 1):
                tokens.append(word[i:i + n])

    for i in range(len(words) - 1):
        tokens.append(f"{words[i]}_{words[i + 1]}")
        tokens.append(f"{words[i]}{words[i + 1]}")

    if expand_splits:
        for word in words:
            if len(word) > 5:
                for i in range(3, len(word) - 2):
                    tokens.append(word[:i])
                    tokens.append(word[i:])

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

    last_chunk_end = 0
    for i in range(0, max(1, len(words) - WINDOW + 1), STRIDE):
        chunk_words = words[i:i + WINDOW]
        if len(chunk_words) < 5:
            continue
        docs.append(tokenize(" ".join(chunk_words)))
        metadata.append(song)
        last_chunk_end = i + len(chunk_words)

    # Tail chunk: striding's upper bound `len(words) - WINDOW + 1` truncates
    # the last few words when len(words) isn't aligned with STRIDE. Without
    # this fixup, the END of a song's lyrics may never enter the index — and
    # if the user's query happens to live there, BM25 returns nothing for it.
    if last_chunk_end < len(words) and len(words) >= 5:
        tail_start = max(0, len(words) - WINDOW)
        docs.append(tokenize(" ".join(words[tail_start:])))
        metadata.append(song)

bm25 = BM25Okapi(docs)


# Song name -> list of chunk indices, so we can compute the best BM25 chunk
# score for any specific song without scanning all 70k chunks.
_SONG_TO_CHUNKS: dict = {}
for _i, _rec in enumerate(metadata):
    _SONG_TO_CHUNKS.setdefault(_rec["Song Name"], []).append(_i)

# Song name -> full normalized lyrics. Used for cross-chunk phrase matching
# (BM25 indexes per chunk, so a phrase that straddles two chunks is invisible
# to BM25 alone). At search time we scan this dict for exact substring matches
# and pin those songs into the candidate set.
_SONG_NORM_LYRICS: dict = {}
for _rec in metadata:
    _sn = _rec["Song Name"]
    if _sn not in _SONG_NORM_LYRICS:
        _SONG_NORM_LYRICS[_sn] = normalize(_rec.get("Lyrics", ""))


# -----------------------
# Stage 1: BM25 → top_k
# -----------------------
def search_bm25(query: str, top_k: int = 20, pin_songs: set = None) -> list:
    """
    Return the top_k unique songs by BM25.

    pin_songs (optional): a set of song names that MUST be present in the
    returned candidates even if they're below the top_k BM25 cutoff. Used to
    keep previously-picked songs eligible for the inference-time pick boost
    in ml_rank — without this, songs that fall outside top_k can never be
    promoted no matter how strong the user-feedback signal is.
    """
    norm_q = normalize(query)
    query_tokens = tokenize(norm_q, expand_splits=True)

    scores = bm25.get_scores(query_tokens)
    top_indices = np.argsort(scores)[::-1][:top_k * 4]  # oversample then dedupe

    results = []
    seen = set()

    for idx in top_indices:
        song = metadata[idx]
        key = (song["Song Name"], song["Composer"])
        if key in seen:
            continue

        # Phrase bonus has two tiers:
        # - chunk-level (+5): query appears verbatim in this chunk's words
        # - full-lyric (+50): query appears verbatim anywhere in the song's
        #   full normalized lyrics. Catches phrases that straddle the 10-word
        #   chunk boundary (which would otherwise hide an exact match).
        doc_text = " ".join(w for w in docs[idx] if '_' not in w and len(w) > 4)
        phrase_bonus = 5 if norm_q in doc_text else 0
        if len(norm_q.split()) >= 2 and norm_q in normalize(song["Lyrics"]):
            phrase_bonus = max(phrase_bonus, 50)
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

    # Inject any pinned songs that didn't make the top_k cutoff. Their score
    # is the BM25 score of their best matching chunk for this query.
    if pin_songs:
        existing_song_names = {r["song"] for r in results}
        for song_name in pin_songs:
            if song_name in existing_song_names:
                continue
            chunk_idxs = _SONG_TO_CHUNKS.get(song_name)
            if not chunk_idxs:
                continue
            best_idx = max(chunk_idxs, key=lambda i: scores[i])
            song_rec = metadata[best_idx]
            score = float(scores[best_idx])
            # Same full-lyric phrase bonus as the top-20 loop — pinned songs
            # from _songs_with_exact_phrase already match by construction, so
            # this lifts them appropriately.
            if len(norm_q.split()) >= 2 and norm_q in normalize(song_rec.get("Lyrics", "")):
                score += 50
            results.append({
                "song": song_rec["Song Name"],
                "composer": song_rec["Composer"],
                "lyrics": song_rec["Lyrics"],
                "score": score,
            })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


# -----------------------
# Feature Extraction
# -----------------------
def _char_ngrams(text: str, n: int) -> set:
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def extract_features(query: str, candidate: dict, qs_counts: dict = None) -> list:
    """
    qs_counts is the dict returned by _load_query_song_counts(): how many
    times each song was confirmed correct for each (normalized) query in
    the feedback log. The picked_for_this_query feature gives the model
    per-(query, song) memory that the other generic features lack — so
    confirming a pick once meaningfully boosts that song the next time the
    same query is run.
    """
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

    # Combined exact-match + similarity-weighted fuzzy match.
    pick_count = _fuzzy_pick_count(norm_q, candidate["song"], qs_counts) if qs_counts else 0.0

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
        pick_count,                                               # picks for this exact query
    ]


# -----------------------
# ML Ranker: Load / Save / Train
# -----------------------
# Resolved relative to this file so cwd doesn't matter (CLI vs API both work).
RANKER_PATH   = os.path.join(os.path.dirname(__file__), 'ranker_v3.pkl')
FEEDBACK_PATH = os.path.join(os.path.dirname(__file__), '..', 'feedback_log.jsonl')


def load_ranker():
    if os.path.exists(RANKER_PATH):
        with open(RANKER_PATH, "rb") as f:
            return pickle.load(f)
    return None


def save_ranker(ranker):
    with open(RANKER_PATH, "wb") as f:
        pickle.dump(ranker, f)


def _load_query_song_counts() -> dict:
    """
    Read feedback log, return:
      {
        normalized_query: {
          "trigrams": set(...),       # char-trigrams of space-stripped query
          "picks": {song_name: count},
        }
      }

    The trigrams are precomputed at load time so per-search fuzzy matching
    (in extract_features) is just an O(N_logged) Jaccard over precomputed sets.
    """
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
            if q not in counts:
                counts[q] = {
                    "trigrams": _char_ngrams(q.replace(" ", ""), 3),
                    "picks": {},
                }
            if entry.get("revoke"):
                # Revocation entry: decrement count (clamped at 0).
                counts[q]["picks"][song] = max(0, counts[q]["picks"].get(song, 0) - 1)
            else:
                counts[q]["picks"][song] = counts[q]["picks"].get(song, 0) + 1
    # Drop entries that have decayed to zero so they don't waste lookups
    for q, info in list(counts.items()):
        info["picks"] = {s: c for s, c in info["picks"].items() if c > 0}
        if not info["picks"]:
            del counts[q]
    return counts


# Above this threshold, a logged query's picks contribute (scaled by similarity)
# to the pick_count feature for the current query.
QUERY_SIM_THRESHOLD = 0.5


def _fuzzy_pick_count(norm_q: str, song: str, qs_counts: dict) -> float:
    """
    Sum of pick counts across logged queries similar enough to the current one.
    Exact-match contributes its full count; fuzzy matches contribute (sim * count).
    """
    if not qs_counts:
        return 0.0
    current_trigrams = _char_ngrams(norm_q.replace(" ", ""), 3)
    if not current_trigrams:
        return 0.0

    total = 0.0
    for logged_q, info in qs_counts.items():
        picks_for_song = info["picks"].get(song, 0)
        if picks_for_song == 0:
            continue
        if logged_q == norm_q:
            total += picks_for_song
            continue
        logged_trigrams = info["trigrams"]
        union = current_trigrams | logged_trigrams
        if not union:
            continue
        sim = len(current_trigrams & logged_trigrams) / len(union)
        if sim >= QUERY_SIM_THRESHOLD:
            total += sim * picks_for_song
    return total


def retrain(min_samples: int = 50):
    if not os.path.exists(FEEDBACK_PATH):
        print("No feedback log found.")
        return None

    training_data = []
    with open(FEEDBACK_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                training_data.append(json.loads(line))

    usable = [d for d in training_data if d.get("correct_in_shortlist")]

    if len(usable) < min_samples:
        print(f"Only {len(usable)} usable samples — need {min_samples} to train. Keep collecting!")
        return None

    qs_counts = _load_query_song_counts()

    X, y, groups = [], [], []
    for entry in usable:
        features = [extract_features(entry["query"], c, qs_counts) for c in entry["candidates"]]
        X.extend(features)
        y.extend(entry["labels"])
        groups.append(len(entry["candidates"]))

    ranker = LGBMRanker(objective="lambdarank", n_estimators=100)
    ranker.fit(np.array(X), np.array(y), group=groups)

    save_ranker(ranker)
    print(f"Trained on {len(usable)} queries — saved to {RANKER_PATH}")
    return ranker


# -----------------------
# Stage 2: ML Ranker (blended with BM25 + inference-time pick boost)
# -----------------------
# How much the ranker can shift the order, expressed as a fraction of the
# top-20 BM25 score range. With 0.2: the most-favored candidate gets an extra
# 20% of the BM25 spread, the least-favored gets 0. Keeps BM25 as the primary
# signal — clear BM25 winners stay on top — while the ranker nudges close calls.
RANKER_WEIGHT = 0.2

# Inference-time deterministic boost applied to songs the user has previously
# confirmed for this query (or a similar one — see _fuzzy_pick_count). Each unit
# of pick_count adds (PICK_BOOST_PER_COUNT * bm25_spread) to the final score.
# 1.0 = a single confirmed pick adds the FULL BM25 spread, enough to overtake
# the current BM25 leader. Tune down for milder behavior.
# Applied AFTER the ranker so it doesn't pollute training (no leakage).
PICK_BOOST_PER_COUNT = 1.0


def ml_rank(query: str, candidates: list) -> tuple:
    """Re-rank candidates; returns (best_candidate, ranked_list).

    Final score = BM25
                + RANKER_WEIGHT * bm25_spread * normalized_ranker_score
                + PICK_BOOST_PER_COUNT * bm25_spread * fuzzy_pick_count
    """
    if not candidates:
        return None, []

    bm25 = np.array([c["score"] for c in candidates], dtype=float)
    bm25_spread = max(float(bm25.max() - bm25.min()), 1.0)
    norm_q = normalize(query)

    qs_counts = _load_query_song_counts()
    pick_counts = np.array([
        _fuzzy_pick_count(norm_q, c["song"], qs_counts) for c in candidates
    ])
    pick_boost = PICK_BOOST_PER_COUNT * bm25_spread * pick_counts

    ranker = load_ranker()
    if ranker is None:
        final = bm25 + pick_boost
    else:
        features = np.array([extract_features(query, c, qs_counts) for c in candidates])
        ranker_scores = ranker.predict(features)
        r_spread = float(ranker_scores.max() - ranker_scores.min())
        if r_spread > 0:
            ranker_norm = (ranker_scores - ranker_scores.min()) / r_spread
            nudge = ranker_norm * RANKER_WEIGHT * bm25_spread
        else:
            nudge = np.zeros_like(bm25)
        final = bm25 + nudge + pick_boost

    order = np.argsort(final)[::-1]
    ranked = [candidates[i] for i in order]
    return ranked[0], ranked


# -----------------------
# Full Pipeline
# -----------------------
def _songs_with_picks(norm_q: str, qs_counts: dict) -> set:
    """Names of songs with nonzero fuzzy pick_count for this query."""
    if not qs_counts:
        return set()
    out = set()
    current_trigrams = _char_ngrams(norm_q.replace(" ", ""), 3)
    for logged_q, info in qs_counts.items():
        if logged_q == norm_q:
            out.update(info["picks"].keys())
            continue
        union = current_trigrams | info["trigrams"]
        if not union:
            continue
        sim = len(current_trigrams & info["trigrams"]) / len(union)
        if sim >= QUERY_SIM_THRESHOLD:
            out.update(info["picks"].keys())
    return out


def _songs_with_exact_phrase(norm_q: str) -> set:
    """Songs whose FULL normalized lyrics contain the query as a substring.
    Empty for single-word queries (too noisy)."""
    if len(norm_q.split()) < 2:
        return set()
    return {sn for sn, nl in _SONG_NORM_LYRICS.items() if norm_q in nl}


def search(query: str, shortlist_size: int = 5):
    norm_q = normalize(query)
    qs_counts = _load_query_song_counts()
    pinned = _songs_with_picks(norm_q, qs_counts) | _songs_with_exact_phrase(norm_q)
    candidates = search_bm25(query, top_k=20, pin_songs=pinned)
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

        with open(FEEDBACK_PATH, "a") as f:
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
    if not os.path.exists(FEEDBACK_PATH):
        return

    total = ml_correct = in_shortlist = 0
    with open(FEEDBACK_PATH, "r") as f:
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
