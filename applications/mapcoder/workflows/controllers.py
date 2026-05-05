"""
Terminate-condition functions for the three Loops in MapCoder.

Important framework gotcha: a `Loop`'s `Controller._message_aggregate_in`
seeds the message cache with the literal string `"(not set yet)"` for any
`input_key` that hasn't appeared on an in-edge yet (see
`masfactory/components/graphs/loop.py`). That placeholder is a non-empty
string, so a naive `bool(message["full_passed"])` check would falsely
report success on the very first iteration. Every helper here filters it
out via `_clean(...)`.
"""

from __future__ import annotations

from typing import Any


_PLACEHOLDER = "(not set yet)"


def _clean(value: Any, default: Any = "") -> Any:
    """Return `value` unless it's the framework's "(not set yet)" sentinel."""
    if isinstance(value, str) and value.strip() == _PLACEHOLDER:
        return default
    return value


def _bool_signal(value: Any) -> bool:
    """Coerce to bool while ignoring the placeholder sentinel."""
    if isinstance(value, str):
        if value.strip() == _PLACEHOLDER or value == "":
            return False
        return value.lower() in {"true", "1", "yes"}
    return bool(value)


# ---------------------------------------------------------------------------
# PlanFanout terminate
# ---------------------------------------------------------------------------

def planfanout_terminate(message: dict[str, Any], attrs: dict[str, Any]) -> bool:
    """Accumulate plans + confidence; terminate after k exemplars are processed."""
    plan_buffer = attrs.setdefault("plan_buffer", [])

    plan_raw = _clean(message.get("plan"), default="")
    confidence_raw = _clean(
        message.get("confidence_raw") or message.get("confidence"),
        default="",
    )

    # Append only when an inner pass actually produced a plan; the very first
    # controller fire has no `plan` yet (placeholder filtered above).
    if isinstance(plan_raw, str) and plan_raw.strip():
        from ..components.planning import parse_confidence
        confidence = parse_confidence(
            confidence_raw if isinstance(confidence_raw, str) else str(confidence_raw or "")
        )
        plan_buffer.append(
            {
                "plan": plan_raw.strip(),
                "confidence": confidence,
                # We're seeing the plan that was produced during the PREVIOUS
                # controller iteration (current_iteration is incremented at
                # the start of `Controller._forward`). So the source exemplar
                # index is current - 2.
                "exemplar_idx": max(0, int(attrs.get("current_iteration", 2)) - 2),
            }
        )

    exemplars = _clean(message.get("exemplars"), default=[]) or _clean(
        attrs.get("exemplars"), default=[]
    )
    if not isinstance(exemplars, list):
        exemplars = []
    n = len(exemplars)
    current = int(attrs.get("current_iteration", 0))

    # Terminate AFTER processing all n exemplars. With current_iteration
    # incremented before this hook, we run exemplar[i-1] during iteration i,
    # so termination should fire at iteration n+1.
    done = n > 0 and current > n

    # Always populate the keys the parent edge declares.
    message["plan_buffer"] = list(plan_buffer)
    for k in ("problem", "algorithm", "sample_io", "language"):
        if k not in message and k in attrs:
            message[k] = attrs[k]

    return done


# ---------------------------------------------------------------------------
# DebugLoop terminate
# ---------------------------------------------------------------------------

def debugloop_terminate(message: dict[str, Any], attrs: dict[str, Any]) -> bool:
    """Stop early on `full_passed`; framework caps at t iterations."""
    full_passed = _bool_signal(message.get("full_passed"))
    code = _clean(message.get("code"), default="")
    if not isinstance(code, str):
        code = str(code)

    message["final_code"] = code
    message["final_passed"] = full_passed

    return full_passed


# ---------------------------------------------------------------------------
# PlanIteration terminate
# ---------------------------------------------------------------------------

def plan_iteration_terminate(message: dict[str, Any], attrs: dict[str, Any]) -> bool:
    """Stop early when any plan passes; always surface final_code/final_passed."""
    final_code = (
        _clean(message.get("final_code"), default="")
        or _clean(message.get("code"), default="")
        or _clean(attrs.get("final_code"), default="")
        or ""
    )
    if not isinstance(final_code, str):
        final_code = str(final_code)

    final_passed = (
        _bool_signal(message.get("final_passed"))
        or _bool_signal(message.get("full_passed"))
    )

    message["final_code"] = final_code
    message["final_passed"] = final_passed

    return final_passed
