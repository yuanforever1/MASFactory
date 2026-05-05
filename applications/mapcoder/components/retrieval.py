"""
Retrieval Agent + RetrievalParser CustomNode.

Pipeline shape inside the RootGraph:

    entry --(problem, language, k)--> Retrieval (Agent) --(content)-->
        RetrievalParser (CustomNode) --(exemplars, algorithm, ...)--> PlanFanout

The Retrieval Agent emits raw XML following Figure 8. RetrievalParser turns that
XML into a structured `exemplars: list[dict]` plus an `algorithm: str`. We do
NOT teach the Agent to emit JSON, because (a) we want to stay faithful to the
paper's prompt and (b) the parser is trivial and robust.
"""

from __future__ import annotations

import re
from typing import Any

from masfactory import Agent, NodeTemplate
from masfactory.core.message import (
    ParagraphMessageFormatter,
    TwinsFieldTextFormatter,
)

from ..prompts import RETRIEVAL_SYSTEM, RETRIEVAL_USER_TEMPLATE


def make_retrieval_agent_template(model) -> NodeTemplate:
    """Build the Retrieval Agent NodeTemplate (Figure 8)."""
    return NodeTemplate(
        Agent,
        model=model,
        instructions=RETRIEVAL_SYSTEM,
        prompt_template=RETRIEVAL_USER_TEMPLATE,
        formatters=[ParagraphMessageFormatter(), TwinsFieldTextFormatter()],
    )


# ---------------------------------------------------------------------------
# RetrievalParser  (CustomNode)
# ---------------------------------------------------------------------------

# Tolerant patterns: the LLM occasionally drops the outer <root> tag or adds
# trailing prose; we extract <problem>...</problem> and <algorithm>...</algorithm>
# blocks directly.
_PROBLEM_BLOCK_RE = re.compile(r"<\s*problem\s*>(.*?)</\s*problem\s*>", re.DOTALL | re.IGNORECASE)
_DESC_RE = re.compile(r"<\s*description\s*>(.*?)</\s*description\s*>", re.DOTALL | re.IGNORECASE)
_CODE_RE = re.compile(r"<\s*code\s*>(.*?)</\s*code\s*>", re.DOTALL | re.IGNORECASE)
_PLAN_RE = re.compile(r"<\s*planning\s*>(.*?)</\s*planning\s*>", re.DOTALL | re.IGNORECASE)
_ALGO_RE = re.compile(r"<\s*algorithm\s*>(.*?)</\s*algorithm\s*>", re.DOTALL | re.IGNORECASE)


def _clean(s: str) -> str:
    return s.strip() if isinstance(s, str) else ""


def parse_retrieval_xml(content: str) -> tuple[list[dict[str, str]], str]:
    """Best-effort parse of the Retrieval Agent's XML output.

    Returns a `(exemplars, algorithm)` tuple. `exemplars` is a list of
    `{description, code, plan}` dicts (empty list if the LLM produced nothing
    parseable). `algorithm` is a string (empty if missing).
    """
    if not isinstance(content, str) or not content.strip():
        return [], ""

    exemplars: list[dict[str, str]] = []
    for block in _PROBLEM_BLOCK_RE.findall(content):
        desc_match = _DESC_RE.search(block)
        code_match = _CODE_RE.search(block)
        plan_match = _PLAN_RE.search(block)
        description = _clean(desc_match.group(1)) if desc_match else ""
        code = _clean(code_match.group(1)) if code_match else ""
        plan = _clean(plan_match.group(1)) if plan_match else ""
        # Skip blocks that have nothing useful (defensive against LLM noise).
        if not (description or code or plan):
            continue
        exemplars.append({"description": description, "code": code, "plan": plan})

    algo_match = _ALGO_RE.search(content)
    algorithm = _clean(algo_match.group(1)) if algo_match else ""
    return exemplars, algorithm


def retrieval_parser_forward(
    input_dict: dict[str, Any],
    attrs: dict[str, Any],
) -> dict[str, Any]:
    """RetrievalParser CustomNode forward callable.

    Reads `content` (raw Retrieval Agent output) and re-emits the structured
    fields plus the original `problem / sample_io / language` so downstream
    PlanFanout can use them without re-pulling.
    """
    content = input_dict.get("content", "") or ""
    exemplars, algorithm = parse_retrieval_xml(content)

    # Carry-through context (fall back to attrs in case an upstream edge
    # filtered them out).
    problem = input_dict.get("problem") or attrs.get("problem", "")
    sample_io = input_dict.get("sample_io") if "sample_io" in input_dict else attrs.get("sample_io", [])
    language = input_dict.get("language") or attrs.get("language", "Python3")

    return {
        "exemplars": exemplars,
        "algorithm": algorithm,
        "problem": problem,
        "sample_io": sample_io,
        "language": language,
    }
