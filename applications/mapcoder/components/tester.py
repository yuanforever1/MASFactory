"""
Tester CustomNode for MapCoder.

Responsibilities (kept fully self-contained, no dependency on other applications):

- Extract the first ```python ... ``` (or generic fenced) code block from an LLM
  response. If no fence is present, fall back to using the raw text as-is.
- Run the candidate code together with the problem's sample I/O assertions inside
  a fresh subprocess (`sys.executable -c <driver>`), with a hard timeout. This
  isolates the candidate from the parent process and contains accidental side
  effects.
- Return a flat dict `{code, observation, full_passed}` consumable by both the
  outer `PassSwitch` (which reads `full_passed`) and the Debugging Agent (which
  reads `code` + `observation`).

Sample I/O contract used here:
- `sample_io` is a `list[dict]` with keys `{call, expected}`. `call` is a
  Python expression string (e.g. `"sum_squares([1.4, 4.2, 0])"`). `expected` is
  the repr of the expected return value (e.g. `"29"`). The driver script tests
  each case as `assert <call> == <expected>`. See `humaneval/dataset.py` for
  how this list is built from the HumanEval prompt docstring.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from typing import Any


# Default timeout (seconds) for one candidate execution. Override via env var.
SANDBOX_TIMEOUT_S = int(os.environ.get("MAPCODER_SANDBOX_TIMEOUT_S", "10"))


_FENCE_RE = re.compile(r"```(?:[a-zA-Z0-9_+\-]*)?\s*(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    """Extract the first fenced code block; fall back to raw text.

    The regex is permissive about the language tag (e.g. ```py / ```python3 /
    ``` ... ```). We pick the *first* fence on the assumption that the LLM
    obeys the prompt and emits a single solution block.
    """
    if not isinstance(text, str):
        return ""
    text = text.strip()
    if not text:
        return ""
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    # No fence: treat the whole response as code (best-effort fallback).
    return text


def _build_driver(candidate_code: str, sample_io: list[dict]) -> str:
    """Produce the Python driver script that asserts each sample I/O case."""
    asserts = []
    for idx, case in enumerate(sample_io):
        call = case.get("call", "")
        expected = case.get("expected", "")
        if not call:
            continue
        # Wrap each assert in a try/except so the driver reports the FIRST
        # failure with a clear marker rather than dying on a SyntaxError half
        # way through.
        asserts.append(
            f"try:\n"
            f"    _got = {call}\n"
            f"    _exp = {expected}\n"
            f"    assert _got == _exp, f'case#{idx + 1} expected {{_exp!r}}, got {{_got!r}}'\n"
            f"except AssertionError as _e:\n"
            f"    print('FAIL_CASE_{idx + 1}: ' + str(_e))\n"
            f"    raise SystemExit(2)\n"
            f"except Exception as _e:\n"
            f"    print('ERROR_CASE_{idx + 1}: ' + repr(_e))\n"
            f"    raise SystemExit(3)\n"
        )

    assert_block = "\n".join(asserts) if asserts else "print('NO_SAMPLE_IO')\nraise SystemExit(4)"

    return (
        candidate_code
        + "\n\n# === auto-generated sample-IO assertions ===\n"
        + assert_block
    )


def run_sample_io_tests(input_dict: dict[str, Any], attrs: dict[str, Any]) -> dict[str, Any]:
    """CustomNode forward callable.

    Args:
        input_dict: must carry `code` (string from upstream Coding/Debugging
            Agent) and (optionally) `sample_io` (list[dict]). When `sample_io`
            is not on the in-edge we fall back to `attrs["sample_io"]` which
            is populated via the node's `pull_keys`.
        attrs: pull-key-filtered attribute store. The pull chain is
            RootGraph -> PlanIteration -> [DebugLoop ->] Tester(2).

    Returns:
        `{code, observation, full_passed}`.
    """
    raw = input_dict.get("code", "") or ""
    code = extract_code(raw)

    sample_io = input_dict.get("sample_io") or attrs.get("sample_io") or []
    if isinstance(sample_io, str) and sample_io.strip() == "(not set yet)":
        sample_io = []
    if not isinstance(sample_io, list):
        sample_io = []

    if os.environ.get("MAPCODER_DEBUG_TESTER"):  # pragma: no cover
        raw_attrs_sample = attrs.get("sample_io")
        print(
            f"[TesterDebug] code_chars={len(code)} sample_io_len={len(sample_io)} "
            f"in_keys={sorted(input_dict.keys())} "
            f"attr_sample_io_type={type(raw_attrs_sample).__name__}"
        )

    if not code:
        return {
            "code": "",
            "observation": "Coding/Debugging agent produced no code.",
            "full_passed": False,
        }

    if not sample_io:
        # No way to evaluate; treat as failure but surface a clear note.
        return {
            "code": code,
            "observation": "No sample I/O available; cannot verify.",
            "full_passed": False,
        }

    driver = _build_driver(code, sample_io)
    try:
        result = subprocess.run(
            [sys.executable, "-I", "-c", driver],
            capture_output=True,
            text=True,
            timeout=SANDBOX_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return {
            "code": code,
            "observation": f"TIMEOUT after {SANDBOX_TIMEOUT_S}s while running sample I/O tests.",
            "full_passed": False,
        }
    except Exception as e:  # noqa: BLE001
        return {
            "code": code,
            "observation": f"Sandbox launch error: {e!r}",
            "full_passed": False,
        }

    if result.returncode == 0:
        return {
            "code": code,
            "observation": "All sample tests passed.",
            "full_passed": True,
        }

    # Failure: surface stdout/stderr to help the Debugging Agent reason about
    # the bug. Trim to a reasonable cap so prompts don't explode.
    obs = ((result.stdout or "") + ("\n" + result.stderr if result.stderr else "")).strip()
    if len(obs) > 4000:
        obs = obs[:4000] + "\n...(truncated)"
    if not obs:
        obs = f"Process exited with code {result.returncode} but produced no output."
    return {
        "code": code,
        "observation": obs,
        "full_passed": False,
    }
