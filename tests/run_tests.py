"""
Run test_cases.json against bm_ml_v2 and report:
  - Top-pick accuracy
  - Shortlist recall (top 5)
  - Web top-5 recall (DuckDuckGo, cached in web_cache.json)

Usage (from repo root):
    python tests/run_tests.py           # uses cached web results
    python tests/run_tests.py --refetch # force-refresh web cache
    python tests/run_tests.py --no-web  # skip web column entirely
"""

import json
import os
import sys
import time
import argparse

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(ROOT, 'llm_searcher'))
os.chdir(os.path.join(ROOT, 'llm_searcher'))

import bm_ml_v2 as searcher  # noqa: E402

CASES_PATH     = os.path.join(os.path.dirname(__file__), 'test_cases.json')
WEB_CACHE_PATH = os.path.join(os.path.dirname(__file__), 'web_cache.json')
SHORTLIST_SIZE = 5
WEB_TOP_K      = 5
SEARCH_PAUSE   = 2.0  # seconds between DuckDuckGo requests


# -----------------------
# Normalization (mirror bm_ml_v2 for comparison)
# -----------------------
import re, unicodedata  # noqa: E402

def _norm(text: str) -> str:
    text = text.lower()
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(c for c in text if not unicodedata.combining(c))
    for src, dst in [("ai","a"),("au","a"),("aa","a"),("ee","i"),("ii","i"),
                     ("oo","u"),("uu","u"),("bh","b"),("dh","d"),("gh","g"),
                     ("th","t"),("sh","s"),("ṣ","s"),("ṅ","n"),("ñ","n")]:
        text = text.replace(src, dst)
    text = re.sub(r'[^a-z\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def _song_in_text(correct_song: str, text: str) -> bool:
    """True if ≥60% of the correct song's normalized words appear in the result text."""
    song_words = set(_norm(correct_song).split())
    result_words = set(_norm(text).split())
    if not song_words:
        return False
    overlap = len(song_words & result_words) / len(song_words)
    return overlap >= 0.6


# -----------------------
# Web search (DuckDuckGo, with cache)
# -----------------------
def _load_cache() -> dict:
    if os.path.exists(WEB_CACHE_PATH):
        with open(WEB_CACHE_PATH) as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict):
    with open(WEB_CACHE_PATH, 'w') as f:
        json.dump(cache, f, indent=2)


def web_search(query: str, cache: dict, refetch: bool) -> list:
    """Return list of {title, body} dicts for the top WEB_TOP_K results."""
    if not refetch and query in cache:
        return cache[query]

    try:
        from ddgs import DDGS
    except ImportError:
        print("  [web] ddgs not installed — run: pip install ddgs")
        return []

    try:
        with DDGS() as ddgs:
            raw = list(ddgs.text(f"{query} carnatic", max_results=WEB_TOP_K))
        results = [{"title": r.get("title",""), "body": r.get("body","")} for r in raw]
        cache[query] = results
        time.sleep(SEARCH_PAUSE)
        return results
    except Exception as e:
        print(f"  [web] search failed for '{query}': {e}")
        cache[query] = []
        return []


def song_in_web_results(correct_song: str, results: list) -> bool:
    for r in results:
        combined = r.get("title","") + " " + r.get("body","")
        if _song_in_text(correct_song, combined):
            return True
    return False


# -----------------------
# Main
# -----------------------
def run(use_web: bool, refetch: bool):
    with open(CASES_PATH) as f:
        cases = json.load(f)

    cache = _load_cache() if use_web else {}

    total             = len(cases)
    top_pick_correct  = 0
    in_shortlist      = 0
    in_web            = 0

    col_q = max(len(c["query"]) for c in cases)
    col_s = max(len(c["correct_song"]) for c in cases)

    web_col = "  WEB" if use_web else ""
    header  = f"{'QUERY':<{col_q}}  {'CORRECT SONG':<{col_s}}  TOP  LIST{web_col}  ACTUAL TOP PICK"
    print(header)
    print("-" * len(header))

    for case in cases:
        query   = case["query"]
        correct = case["correct_song"]

        best, shortlist = searcher.search(query, shortlist_size=SHORTLIST_SIZE)

        hit_top  = best["song"].strip().lower() == correct.strip().lower()
        hit_list = any(s["song"].strip().lower() == correct.strip().lower() for s in shortlist)

        top_pick_correct += hit_top
        in_shortlist     += hit_list

        top_sym  = "✓" if hit_top  else "✗"
        list_sym = "✓" if hit_list else "✗"
        actual   = best["song"] if not hit_top else ""

        if use_web:
            results  = web_search(query, cache, refetch)
            hit_web  = song_in_web_results(correct, results)
            in_web  += hit_web
            web_sym  = "✓" if hit_web else "✗"
            print(f"{query:<{col_q}}  {correct:<{col_s}}  {top_sym}    {list_sym}    {web_sym}    {actual}")
        else:
            print(f"{query:<{col_q}}  {correct:<{col_s}}  {top_sym}    {list_sym}    {actual}")

    if use_web:
        _save_cache(cache)

    print()
    print(f"Top-pick accuracy : {top_pick_correct}/{total}  ({100 * top_pick_correct // total}%)")
    print(f"Shortlist recall  : {in_shortlist}/{total}  ({100 * in_shortlist // total}%)")
    if use_web:
        print(f"Web top-{WEB_TOP_K} recall  : {in_web}/{total}  ({100 * in_web // total}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-web",   action="store_true", help="Skip web search column")
    parser.add_argument("--refetch",  action="store_true", help="Ignore cache and re-fetch all web results")
    args = parser.parse_args()
    run(use_web=not args.no_web, refetch=args.refetch)
