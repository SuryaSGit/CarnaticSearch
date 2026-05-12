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

lexical_searcher/
  search_bm.py           # Baseline BM25 (no normalization)

llm_searcher/
  bm_ml_v2.py            # Current pipeline: BM25 + LGBMRanker
  bm_test_llm.py         # Earlier experiment with Gemini reranking

tests/
  test_cases.json        # 20 query → correct_song test cases
  run_tests.py           # Benchmark script
  web_cache.json         # Cached DuckDuckGo results (auto-managed)

scraping_tools/
  scrape_test.py         # Tools used to build the corpus
```

## Usage

### Interactive search

```bash
cd llm_searcher
python bm_ml_v2.py
```

Commands at the prompt:
- Type a lyric snippet → get top pick + shortlist of 5
- Press Enter to confirm the top pick is correct, or type the correct song name
- `retrain` — retrain the LGBMRanker from collected feedback (needs ≥50 samples)
- `quit` — exit

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

```
rank_bm25
lightgbm
numpy
ddgs           # for web search column in tests
```

Install:
```bash
pip install rank_bm25 lightgbm numpy ddgs
```
