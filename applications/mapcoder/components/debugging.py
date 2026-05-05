"""Debugging Agent NodeTemplate (Figure 10, bottom half)."""

from __future__ import annotations

from masfactory import Agent, NodeTemplate
from masfactory.core.message import (
    ParagraphMessageFormatter,
    TwinsFieldTextFormatter,
)

from ..prompts import DEBUG_SYSTEM, DEBUG_TEMPLATE


def make_debug_agent_template(model) -> NodeTemplate:
    return NodeTemplate(
        Agent,
        model=model,
        instructions=DEBUG_SYSTEM,
        prompt_template=DEBUG_TEMPLATE,
        formatters=[ParagraphMessageFormatter(), TwinsFieldTextFormatter()],
    )
