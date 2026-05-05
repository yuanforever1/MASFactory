"""Coding Agent NodeTemplate (Figure 10, top half)."""

from __future__ import annotations

from masfactory import Agent, NodeTemplate
from masfactory.core.message import (
    ParagraphMessageFormatter,
    TwinsFieldTextFormatter,
)

from ..prompts import CODING_SYSTEM, CODING_TEMPLATE


def make_coding_agent_template(model) -> NodeTemplate:
    return NodeTemplate(
        Agent,
        model=model,
        instructions=CODING_SYSTEM,
        prompt_template=CODING_TEMPLATE,
        formatters=[ParagraphMessageFormatter(), TwinsFieldTextFormatter()],
    )
