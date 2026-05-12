"""
Run test_cases.json against bm_ml_v2 and report shortlist recall + top-pick accuracy.

Usage (from repo root):
    python tests/run_tests.py
"""

import json
import os
import sys

# Make sure the searcher module is importable regardless of cwd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'llm_searcher'))
os.chdir(os.path.join(os.path.dirname(__file__), '..', 'llm_searcher'))

import bm_ml_v2 as searcher  # noqa: E402  (import after chdir)

CASES_PATH = os.path.join(os.path.dirname(__file__), 'test_cases.json')
SHORTLIST_SIZE = 5


def _matches(a: str, b: str) -> bool:
    return a.strip().lower() == b.strip().lower()


def run():
    with open(CASES_PATH) as f:
        cases = json.load(f)

    total = len(cases)
    top_pick_correct = 0
    in_shortlist = 0

    col_q  = max(len(c["query"]) for c in cases)
    col_s  = max(len(c["correct_song"]) for c in cases)

    header = f"{'QUERY':<{col_q}}  {'CORRECT SONG':<{col_s}}  TOP  SHORTLIST  ACTUAL TOP PICK"
    print(header)
    print("-" * len(header))

    for case in cases:
        query   = case["query"]
        correct = case["correct_song"]

        best, shortlist = searcher.search(query, shortlist_size=SHORTLIST_SIZE)

        hit_top  = _matches(best["song"], correct)
        hit_list = any(_matches(s["song"], correct) for s in shortlist)

        top_pick_correct += hit_top
        in_shortlist     += hit_list

        top_sym  = "✓" if hit_top  else "✗"
        list_sym = "✓" if hit_list else "✗"
        actual   = best["song"] if not hit_top else ""

        print(f"{query:<{col_q}}  {correct:<{col_s}}  {top_sym}    {list_sym}          {actual}")

    print()
    print(f"Top-pick accuracy : {top_pick_correct}/{total}  ({100 * top_pick_correct // total}%)")
    print(f"Shortlist recall  : {in_shortlist}/{total}  ({100 * in_shortlist // total}%)")


if __name__ == "__main__":
    run()
