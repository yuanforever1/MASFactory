"""
Per-problem driver: take a HumanEval item, build sample_io from its prompt,
invoke the MapCoder graph, and verify the resulting code against the hidden
HumanEval `check(candidate)` test.

Returned dict (one per problem):

    {
      "task_id": str,
      "entry_point": str,
      "final_code": str,
      "graph_passed": bool,    # what the in-graph Tester said
      "hidden_passed": bool,   # what the HumanEval `test` field says
      "hidden_msg": str,
      "elapsed_s": float,
    }
"""

from __future__ import annotations

import time
from typing import Any

from masfactory import RootGraph

from .dataset import extract_sample_io_from_prompt, verify_humaneval_solution


def run_one_problem(
    g: RootGraph,
    item: dict[str, Any],
    *,
    language: str = "Python3",
    k: int = 3,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run one HumanEval item end-to-end through the MapCoder graph."""
    task_id = item.get("task_id") or item.get("name", "<unknown>")
    entry_point = item.get("entry_point", "")
    prompt = item.get("prompt", "")
    test = item.get("test", "")

    sample_io = extract_sample_io_from_prompt(prompt, entry_point)
    if verbose:
        print(f"[{task_id}] sample_io: {len(sample_io)} pair(s)")

    started = time.time()
    final_code = ""
    graph_passed = False
    err = ""
    try:
        out, _attrs = g.invoke(
            {"problem": prompt, "language": language, "k": k},
            attributes={"sample_io": sample_io},
        )
        final_code = out.get("final_code", "") if isinstance(out, dict) else ""
        graph_passed = bool(out.get("final_passed", False)) if isinstance(out, dict) else False
    except Exception as e:  # noqa: BLE001
        err = repr(e)
        if verbose:
            print(f"[{task_id}] graph error: {err}")
    elapsed = time.time() - started

    hidden_passed, hidden_msg = (False, "no candidate")
    if final_code:
        hidden_passed, hidden_msg = verify_humaneval_solution(final_code, test, entry_point)

    return {
        "task_id": task_id,
        "entry_point": entry_point,
        "final_code": final_code,
        "graph_passed": graph_passed,
        "hidden_passed": hidden_passed,
        "hidden_msg": hidden_msg,
        "elapsed_s": round(elapsed, 2),
        "graph_error": err,
    }


def run_dataset(
    g: RootGraph,
    items: list[dict[str, Any]],
    *,
    language: str = "Python3",
    k: int = 3,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run a list of items and return aggregated Pass@1 stats."""
    results: list[dict[str, Any]] = []
    n_pass = 0
    for idx, item in enumerate(items):
        r = run_one_problem(g, item, language=language, k=k, verbose=verbose)
        if r["hidden_passed"]:
            n_pass += 1
        results.append(r)
        if verbose:
            tag = "PASS" if r["hidden_passed"] else "FAIL"
            acc = round(n_pass / (idx + 1), 3)
            print(
                f"[{idx + 1}/{len(items)}] {tag} {r['task_id']} "
                f"(graph={r['graph_passed']}, t={r['elapsed_s']}s)  acc={acc}"
            )
    n = len(items)
    return {
        "results": results,
        "n_total": n,
        "n_pass": n_pass,
        "pass_at_1": round(n_pass / n, 3) if n else 0.0,
    }
