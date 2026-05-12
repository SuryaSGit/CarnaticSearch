# CarnaticSearch

A search engine for Carnatic music kritis. Given a lyric snippet or song name (in any common English transliteration), it identifies the correct kriti from a corpus of song data.

## How it works

The pipeline runs in two stages:

1. **BM25 retrieval** — lyrics are normalized and tokenized into word tokens, character n-grams (3–4 chars), and word bigrams. This lets the index fuzzy-match phonetic variants even when transliteration is inconsistent (e.g. `naiyya` vs `nayya`, `bhajana` vs `bajana`).
2. **LGBMRanker reranking** — a LightGBM learning-to-rank model re-scores the top 20 BM25 candidates using features like token overlap, character n-gram overlap, and exact phrase match. It is trained from user feedback collected during interactive sessions.

### Normalization rules

Before indexing or querying, text goes through:
- Lowercase + Unicode diacritic stripping
- Diphthong collapse: `ai → a`, `au → a`
- Vowel length collapse: `aa → a`, `ee/ii → i`, `oo/uu → u`
- Aspirate collapse: `bh → b`, `dh → d`, `gh → g`, `th → t`, `sh → s`, etc.

## Project structure

```
song_data/
  data.json              # Corpus: song name, composer, lyrics

llm_searcher/
  bm_ml_v2.py            # Search pipeline: BM25 + LGBMRanker
  bm_test_llm.py         # Earlier experiment with Gemini reranking

backend/
  app.py                 # FastAPI: /search, /feedback, /retrain, /stats
  requirements.txt

frontend/
  index.html             # Search UI (served by FastAPI)
  app.js                 # Search + feedback + retrain logic
  style.css

tests/
  test_cases.json        # Query → correct_song test cases
  run_tests.py           # Benchmark vs DuckDuckGo top-5
  web_cache.json         # Cached DuckDuckGo results (auto-managed)

lexical_searcher/
  search_bm.py           # Baseline BM25 (no normalization)

scraping_tools/
  scrape_test.py         # Tools used to build the corpus
```

## Usage

### Web app (recommended)

```bash
pip install -r backend/requirements.txt
uvicorn backend.app:app --reload --port 8000
# open http://127.0.0.1:8000
```

The FastAPI server serves both the API and the static frontend on the same port. The UI provides:
- Search box → top pick (highlighted) + shortlist of 5
- Feedback form → confirm the pick or supply the correct song
- Live stats and a Retrain button

### CLI (interactive)

```bash
cd llm_searcher
python bm_ml_v2.py
```

Commands at the prompt:
- Type a lyric snippet → get top pick + shortlist of 5
- Press Enter to confirm the top pick is correct, or type the correct song name
- `retrain` — retrain the LGBMRanker from collected feedback
- `quit` — exit

### API endpoints

| Method | Path | Body | Returns |
|---|---|---|---|
| POST | `/search` | `{query, shortlist_size}` | `{top_pick, shortlist}` |
| POST | `/feedback` | `{query, correct_song, shortlist, ml_pick}` | `{ok, ml_correct, correct_in_shortlist}` |
| POST | `/retrain` | — | `{trained: bool}` |
| GET  | `/stats` | — | `{total, ml_correct, in_shortlist, ml_accuracy_pct, shortlist_recall_pct}` |

### Run the test suite

```bash
# From repo root
python tests/run_tests.py           # uses cached web results
python tests/run_tests.py --refetch # re-fetch DuckDuckGo results
python tests/run_tests.py --no-web  # skip web column
```

**Current numbers (20 test cases):**

| Metric | Score |
|---|---|
| Top-pick accuracy | 13/20 (65%) |
| Shortlist recall (top 5) | 18/20 (90%) |
| Web top-5 recall (DuckDuckGo) | 17/20 (85%) |

### Add test cases

Append to `tests/test_cases.json`:
```json
{"query": "your lyric snippet", "correct_song": "exact song name from data.json"}
```

## Dependencies

Backend / search engine:
```bash
pip install -r backend/requirements.txt
```
(installs fastapi, uvicorn, rank_bm25, lightgbm, scikit-learn, numpy, pydantic)

For the test runner's DuckDuckGo comparison column:
```bash
pip install ddgs
```
