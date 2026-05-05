"""
MapCoder x MASFactory CLI: run the MapCoder graph on a HumanEval JSONL.

Usage examples:

    # Subset of the first 5 problems
    python -m applications.mapcoder.main \
        --dataset path/to/HumanEval.jsonl \
        --limit 5 \
        --k 3 --t 3

Environment:
    OPENAI_API_KEY      required
    OPENAI_BASE_URL     optional (e.g. for custom OpenAI-compatible endpoints)
    MAPCODER_MODEL      optional (default "gpt-4o-mini")
"""

from __future__ import annotations

import argparse
import os
import sys

_APP_ROOT = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_APP_ROOT))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from applications.mapcoder.humaneval.dataset import load_humaneval_jsonl
from applications.mapcoder.humaneval.runner import run_dataset
from applications.mapcoder.workflows.graph import build_graph


def _make_model():
    from masfactory import LegacyOpenAIModel

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        print(
            "Error: OPENAI_API_KEY is not set.\n"
            "  set OPENAI_API_KEY=sk-...      (PowerShell: $env:OPENAI_API_KEY = 'sk-...')",
            file=sys.stderr,
        )
        sys.exit(2)

    base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("BASE_URL") or None
    model_name = os.environ.get("MAPCODER_MODEL", "gpt-4o-mini")
    return LegacyOpenAIModel(api_key=api_key, base_url=base_url, model_name=model_name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MapCoder reproduction on HumanEval (MASFactory)."
    )
    parser.add_argument("--dataset", required=True, help="Path to HumanEval .jsonl(.gz)")
    parser.add_argument("--limit", type=int, default=10, help="0 = all problems")
    parser.add_argument("--k", type=int, default=3, help="number of plans (max outer-loop iterations)")
    parser.add_argument("--t", type=int, default=3, help="number of debug attempts (inner loop)")
    parser.add_argument("--language", type=str, default="Python3")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    items = load_humaneval_jsonl(args.dataset)
    if args.limit > 0:
        items = items[: args.limit]
    print(f"Loaded {len(items)} HumanEval items from {args.dataset}")

    model = _make_model()
    g = build_graph(model, k=args.k, t=args.t)
    print(f"MapCoder graph built (k={args.k}, t={args.t}).")

    summary = run_dataset(g, items, language=args.language, k=args.k, verbose=args.verbose)
    print(
        f"\n=== MapCoder Pass@1: {summary['n_pass']}/{summary['n_total']} = {summary['pass_at_1']} ==="
    )


if __name__ == "__main__":
    main()
