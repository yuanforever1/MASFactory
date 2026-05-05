"""
HumanEval dataset utilities (self-contained; no LATS reuse).

Two responsibilities:

1. Load the HumanEval JSONL (`.jsonl` or `.jsonl.gz`).
2. Parse the docstring inside each problem's `prompt` to recover sample I/O
   pairs that the MapCoder Tester CustomNode can run.

3. Verify a candidate solution against the HumanEval `test` block — used for
   final Pass@1 scoring (NOT inside the MapCoder graph).

Sample I/O format consumed by `applications/mapcoder/components/tester.py`:
    [{"call": "<expr>", "expected": "<repr>"}]
"""

from __future__ import annotations

import ast
import gzip
import json
import os
import re
import subprocess
import sys
from typing import Any


# ---------------------------------------------------------------------------
# JSONL loader
# ---------------------------------------------------------------------------

def load_humaneval_jsonl(path: str) -> list[dict[str, Any]]:
    """Load HumanEval items from `.jsonl` or `.jsonl.gz`."""
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"HumanEval dataset not found at {path!r}")
    items: list[dict[str, Any]] = []
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:  # type: ignore[arg-type]
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


# ---------------------------------------------------------------------------
# Sample I/O extraction from the docstring inside `prompt`
# ---------------------------------------------------------------------------

# HumanEval prompts come in two common dialects:
#   (a) doctest-style:
#         >>> sum_squares([1.4, 4.2, 0])
#         29
#   (b) "Examples:" prose style:
#         Example:
#         is_palindrome("racecar") -> True
#
# We try (a) first since it's by far the most common in the original dataset,
# and fall back to (b).

_PROMPT_PROMPT_RE = re.compile(r">>>\s*(.+?)(?=\n)", re.DOTALL)


def _extract_doctest_pairs(docstring: str) -> list[tuple[str, str]]:
    """Pull `>>> call\n expected_repr` pairs out of a docstring.

    The expected line is the first non-blank line after `>>>` that does NOT
    itself start with `>>>`.
    """
    pairs: list[tuple[str, str]] = []
    lines = [ln.rstrip() for ln in docstring.splitlines()]
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith(">>>"):
            call = line[3:].strip()
            # Multi-line `>>>` continuations (`...`) are rare in HumanEval; keep simple.
            j = i + 1
            expected_lines: list[str] = []
            while j < len(lines):
                nxt = lines[j].strip()
                if not nxt or nxt.startswith(">>>"):
                    break
                # Some HumanEval doctests use `...` continuation; treat as call.
                if nxt.startswith("..."):
                    call += " " + nxt[3:].strip()
                    j += 1
                    continue
                expected_lines.append(nxt)
                j += 1
            if call and expected_lines:
                expected = " ".join(expected_lines).strip()
                pairs.append((call, expected))
            i = j
        else:
            i += 1
    return pairs


_EXAMPLE_ARROW_RE = re.compile(
    r"^([A-Za-z_][\w\s\(\)\[\]\{\}\.,'\"\:\-\+\*\/=<>%\| ]+?)\s*(?:->|=>|==>)\s*(.+)$"
)


def _extract_arrow_pairs(docstring: str) -> list[tuple[str, str]]:
    """Pull `call -> expected` pairs from prose-style examples."""
    pairs: list[tuple[str, str]] = []
    for line in docstring.splitlines():
        m = _EXAMPLE_ARROW_RE.match(line.strip())
        if not m:
            continue
        call = m.group(1).strip()
        expected = m.group(2).strip()
        # Filter obvious non-call lines (e.g. "input -> output" from headings).
        if "(" not in call:
            continue
        pairs.append((call, expected))
    return pairs


def _module_docstring(prompt: str) -> str:
    """Return the FIRST docstring found in the HumanEval prompt code.

    We use `ast` so multi-line strings inside the function are handled
    correctly. Return empty string if parsing fails or no docstring is found.
    """
    try:
        tree = ast.parse(prompt)
    except SyntaxError:
        return ""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)):
            doc = ast.get_docstring(node)
            if doc:
                return doc
    return ""


def _normalize_expected_repr(raw: str, entry_point: str) -> str:
    """Best-effort normalize the expected RHS so it `eval`s cleanly.

    Strategies:
      1. Try `ast.literal_eval(raw)`. If it works, use `repr()` of the value.
      2. Otherwise return `raw` verbatim and let the sandbox fail loudly.
    """
    try:
        value = ast.literal_eval(raw)
        return repr(value)
    except Exception:
        return raw


def extract_sample_io_from_prompt(prompt: str, entry_point: str) -> list[dict[str, str]]:
    """Build the `[{call, expected}]` list that the Tester CustomNode expects.

    Args:
        prompt: HumanEval `prompt` field (function signature + docstring + body).
        entry_point: The function name HumanEval expects the candidate to define.
    """
    doc = _module_docstring(prompt)
    if not doc:
        return []

    pairs = _extract_doctest_pairs(doc)
    if not pairs:
        pairs = _extract_arrow_pairs(doc)
    if not pairs:
        return []

    sample_io: list[dict[str, str]] = []
    for call, expected in pairs:
        # Defensive: if the doctest sample uses bare function name without
        # qualifying entry_point, accept as-is. HumanEval examples normally
        # already use the entry_point name.
        sample_io.append(
            {
                "call": call,
                "expected": _normalize_expected_repr(expected, entry_point),
            }
        )
    return sample_io


# ---------------------------------------------------------------------------
# Hidden-test verifier (Pass@1 scoring; runs OUTSIDE the MapCoder graph).
# ---------------------------------------------------------------------------

def verify_humaneval_solution(
    candidate_code: str,
    test: str,
    entry_point: str,
    timeout_s: int = 10,
) -> tuple[bool, str]:
    """Run the HumanEval `check(candidate)` against `candidate_code`.

    Returns:
        (passed, message)
    """
    if not candidate_code:
        return False, "empty candidate"

    driver = (
        candidate_code
        + "\n\n"
        + test
        + "\n\n"
        + f"check({entry_point})\n"
        + "print('HUMANEVAL_PASS')\n"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-I", "-c", driver],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT after {timeout_s}s"
    except Exception as e:  # noqa: BLE001
        return False, f"sandbox error: {e!r}"

    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    if result.returncode == 0 and "HUMANEVAL_PASS" in stdout:
        return True, "ok"
    msg = stderr or stdout or f"exit={result.returncode}"
    if len(msg) > 2000:
        msg = msg[:2000] + "...(truncated)"
    return False, msg
