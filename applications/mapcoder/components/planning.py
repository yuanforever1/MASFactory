"""
Planning components: PlanGen Agent + ConfidenceEval Agent + ExemplarPicker
+ PlanSorter + PlanPicker.

Following the paper's Algorithm 1, the Planning stage consists of two LLM
calls per exemplar (Figure 9):

  1. PlanGen        -> produces a per-target-problem plan
  2. ConfidenceEval -> scores that plan with an integer 0-100

These two Agents are orchestrated inside `PlanFanout` (a `Loop`) which iterates
once per retrieved exemplar. After the loop, `PlanSorter` sorts the resulting
buffer by confidence DESC and feeds the sorted list to `PlanIteration`.
"""

from __future__ import annotations

import re
from typing import Any

from masfactory import Agent, NodeTemplate
from masfactory.core.message import (
    ParagraphMessageFormatter,
    TwinsFieldTextFormatter,
)

from ..prompts import (
    CONFIDENCE_SYSTEM,
    CONFIDENCE_TEMPLATE,
    PLAN_GEN_SYSTEM,
    PLAN_GEN_TEMPLATE,
)


# ---------------------------------------------------------------------------
# Agent NodeTemplates
# ---------------------------------------------------------------------------

def make_plan_gen_template(model) -> NodeTemplate:
    return NodeTemplate(
        Agent,
        model=model,
        instructions=PLAN_GEN_SYSTEM,
        prompt_template=PLAN_GEN_TEMPLATE,
        formatters=[ParagraphMessageFormatter(), TwinsFieldTextFormatter()],
    )


def make_confidence_eval_template(model) -> NodeTemplate:
    return NodeTemplate(
        Agent,
        model=model,
        instructions=CONFIDENCE_SYSTEM,
        prompt_template=CONFIDENCE_TEMPLATE,
        formatters=[ParagraphMessageFormatter(), TwinsFieldTextFormatter()],
    )


# ---------------------------------------------------------------------------
# ExemplarPicker  (CustomNode, runs each PlanFanout iteration)
# ---------------------------------------------------------------------------

def exemplar_picker_forward(
    input_dict: dict[str, Any],
    attrs: dict[str, Any],
) -> dict[str, Any]:
    """Pick the i-th exemplar (1-indexed via attrs['current_iteration'])."""
    exemplars = input_dict.get("exemplars") or attrs.get("exemplars") or []
    if not isinstance(exemplars, list):
        exemplars = []

    iteration = int(attrs.get("current_iteration", 1) or 1)
    idx = max(0, iteration - 1)
    if idx >= len(exemplars):
        # PlanFanout's terminate function should already have stopped the
        # loop; this is a defensive placeholder so PlanGen still receives
        # well-typed strings if the Loop is mis-configured.
        ex = {"description": "", "code": "", "plan": ""}
    else:
        ex = exemplars[idx] or {}

    problem = input_dict.get("problem") or attrs.get("problem", "")
    algorithm = input_dict.get("algorithm") or attrs.get("algorithm", "")
    sample_io = input_dict.get("sample_io") if "sample_io" in input_dict else attrs.get("sample_io", [])
    language = input_dict.get("language") or attrs.get("language", "Python3")

    # We emit BOTH `sample_io` (the original list, used by Tester) and
    # `sample_io_str` (a prompt-friendly rendering used by Agents). Mixing
    # types under the same key would either break the Tester or produce
    # noisy prompts.
    return {
        "exemplar_problem": ex.get("description", ""),
        "exemplar_plan": ex.get("plan", "") or ex.get("code", ""),
        "algorithm": algorithm,
        "problem": problem,
        "sample_io": sample_io,
        "sample_io_str": _stringify_sample_io(sample_io),
        "language": language,
    }


# ---------------------------------------------------------------------------
# Confidence parser
# ---------------------------------------------------------------------------

_CONFIDENCE_RE = re.compile(r"<\s*confidence\s*>\s*([0-9]+)\s*<", re.DOTALL | re.IGNORECASE)
_NUMBER_RE = re.compile(r"\b(\d{1,3})\b")


def parse_confidence(content: str) -> int:
    """Extract integer 0-100 from the ConfidenceEval Agent's xml-ish output.

    Tolerant fallbacks:
      1. Try `<confidence>NN</confidence>`.
      2. Otherwise pick the first standalone number 0..100 in the text.
      3. Default to 0 (so the plan sinks to the bottom of the sort).
    """
    if not isinstance(content, str):
        return 0
    m = _CONFIDENCE_RE.search(content)
    if m:
        try:
            return max(0, min(100, int(m.group(1))))
        except ValueError:
            pass
    for cand in _NUMBER_RE.findall(content):
        try:
            v = int(cand)
        except ValueError:
            continue
        if 0 <= v <= 100:
            return v
    return 0


# ---------------------------------------------------------------------------
# PlanSorter  (CustomNode, runs once between PlanFanout and PlanIteration)
# ---------------------------------------------------------------------------

def plan_sorter_forward(
    input_dict: dict[str, Any],
    attrs: dict[str, Any],
) -> dict[str, Any]:
    """Sort `plan_buffer` by confidence DESC and emit `sorted_plans`."""
    plan_buffer = input_dict.get("plan_buffer") or attrs.get("plan_buffer") or []
    if not isinstance(plan_buffer, list):
        plan_buffer = []

    sorted_plans = sorted(
        plan_buffer,
        key=lambda item: int(item.get("confidence", 0) or 0),
        reverse=True,
    )

    problem = input_dict.get("problem") or attrs.get("problem", "")
    algorithm = input_dict.get("algorithm") or attrs.get("algorithm", "")
    sample_io = input_dict.get("sample_io") if "sample_io" in input_dict else attrs.get("sample_io", [])
    language = input_dict.get("language") or attrs.get("language", "Python3")

    return {
        "sorted_plans": sorted_plans,
        "problem": problem,
        "algorithm": algorithm,
        "sample_io": sample_io,
        "language": language,
    }


# ---------------------------------------------------------------------------
# PlanPicker  (CustomNode, runs each PlanIteration outer-loop iteration)
# ---------------------------------------------------------------------------

def plan_picker_forward(
    input_dict: dict[str, Any],
    attrs: dict[str, Any],
) -> dict[str, Any]:
    """Pick the i-th plan from the already-sorted list."""
    sorted_plans = input_dict.get("sorted_plans") or attrs.get("sorted_plans") or []
    if not isinstance(sorted_plans, list):
        sorted_plans = []

    iteration = int(attrs.get("current_iteration", 1) or 1)
    idx = max(0, iteration - 1)
    if idx >= len(sorted_plans):
        plan = ""
    else:
        plan = sorted_plans[idx].get("plan", "") if isinstance(sorted_plans[idx], dict) else ""

    problem = input_dict.get("problem") or attrs.get("problem", "")
    algorithm = input_dict.get("algorithm") or attrs.get("algorithm", "")
    sample_io = input_dict.get("sample_io") if "sample_io" in input_dict else attrs.get("sample_io", [])
    language = input_dict.get("language") or attrs.get("language", "Python3")

    return {
        "current_plan": plan,
        "problem": problem,
        "algorithm": algorithm,
        "sample_io": sample_io,
        "sample_io_str": _stringify_sample_io(sample_io),
        "language": language,
    }


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _stringify_sample_io(sample_io: Any) -> str:
    """Render the structured sample I/O list as a human-readable string for prompts.

    Tester (the CustomNode that actually runs them) consumes the list directly
    via `attrs['sample_io']`, but Agent prompts need a string snippet.
    """
    if isinstance(sample_io, str):
        return sample_io
    if not isinstance(sample_io, list) or not sample_io:
        return "(no sample I/O available)"
    lines = []
    for case in sample_io:
        if not isinstance(case, dict):
            continue
        call = case.get("call", "")
        expected = case.get("expected", "")
        if call and expected:
            lines.append(f"assert {call} == {expected}")
    return "\n".join(lines) if lines else "(no sample I/O available)"
