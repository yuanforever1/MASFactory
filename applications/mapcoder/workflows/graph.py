"""
Build the MapCoder RootGraph.

Top-level shape:

    entry --(problem, language, k)--> Retrieval (Agent)
        --(content)--> RetrievalParser (CustomNode)
        --(exemplars, algorithm, problem, sample_io, language)--> PlanFanout (Loop)
        --(plan_buffer, problem, algorithm, sample_io, language)--> PlanSorter (CustomNode)
        --(sorted_plans, problem, algorithm, sample_io, language)--> PlanIteration (Loop)
        --(final_code, final_passed)--> exit

Inner Loops:

* PlanFanout (k iterations, one per retrieved exemplar)
    controller -> ExemplarPicker -> {PlanGen, ConfidenceEval (passthrough)}
    PlanGen -> ConfidenceEval (carries `plan` for the prompt)
    PlanGen -> controller (carries `plan`)
    ConfidenceEval -> controller (carries `confidence_raw`)

* PlanIteration (up to k outer iterations)
    controller -> PlanPicker -> {Coding, PassSwitch (passthrough static context)}
    Coding -> Tester (carries `code`); Tester pulls `sample_io` from attrs
    Tester -> PassSwitch (carries `code, observation, full_passed`)
    PassSwitch routes:
        passed -> controller (carries `code, full_passed`)
        failed -> DebugLoop (carries the full debug context)
    DebugLoop -> controller (carries `final_code, final_passed`)

* DebugLoop (up to t inner iterations)
    controller -> Debugging -> Tester2 -> controller

Note on `TwinsFieldTextFormatter`: it copies the raw LLM output into every
declared output key. Therefore each Agent has exactly ONE distinct LLM-output
key across all of its outgoing edges (`plan`, `confidence_raw`, `code`); any
static context must flow on sibling edges, NOT through the Agent.
"""

from __future__ import annotations

import os
from typing import Any

from masfactory import (
    CustomNode,
    Loop,
    LogicSwitch,
    NodeTemplate,
    RootGraph,
    Shared,
)

from ..components.coding import make_coding_agent_template
from ..components.debugging import make_debug_agent_template
from ..components.planning import (
    exemplar_picker_forward,
    make_confidence_eval_template,
    make_plan_gen_template,
    plan_picker_forward,
    plan_sorter_forward,
)
from ..components.retrieval import (
    make_retrieval_agent_template,
    retrieval_parser_forward,
)
from ..components.tester import run_sample_io_tests
from .controllers import (
    debugloop_terminate,
    plan_iteration_terminate,
    planfanout_terminate,
)


def build_graph(model, k: int = 3, t: int = 3) -> RootGraph:
    """Construct and build the MapCoder graph for one model + (k, t) config."""

    # =====================================================================
    # Agent templates
    # =====================================================================
    RetrievalT = make_retrieval_agent_template(model)
    PlanGenT = make_plan_gen_template(model)
    ConfidenceT = make_confidence_eval_template(model)
    CodingT = make_coding_agent_template(model)
    DebugT = make_debug_agent_template(model)

    # =====================================================================
    # CustomNode templates
    # =====================================================================
    RetrievalParserT = NodeTemplate(
        CustomNode,
        forward=retrieval_parser_forward,
        pull_keys={"problem": "", "sample_io": "", "language": ""},
    )

    ExemplarPickerT = NodeTemplate(
        CustomNode,
        forward=exemplar_picker_forward,
        pull_keys={
            "exemplars": "",
            "algorithm": "",
            "problem": "",
            "sample_io": "",
            "language": "",
        },
        # Forward pickers stringify `sample_io` for Agent prompts. We MUST
        # NOT push that stringified form back into the parent Loop's attrs,
        # otherwise the Tester CustomNode (which pulls `sample_io` from
        # attrs) would receive a string instead of the list[dict] it needs.
        push_keys={},
    )

    PlanSorterT = NodeTemplate(
        CustomNode,
        forward=plan_sorter_forward,
        pull_keys={
            "problem": "",
            "algorithm": "",
            "sample_io": "",
            "language": "",
        },
    )

    PlanPickerT = NodeTemplate(
        CustomNode,
        forward=plan_picker_forward,
        pull_keys={
            "sorted_plans": "",
            "problem": "",
            "algorithm": "",
            "sample_io": "",
            "language": "",
        },
        # Same reason as ExemplarPickerT: don't pollute Loop attrs with the
        # stringified `sample_io`.
        push_keys={},
    )

    TesterT = NodeTemplate(
        CustomNode,
        forward=run_sample_io_tests,
        pull_keys={"sample_io": ""},
    )

    # =====================================================================
    # PlanFanout (Loop, max_iter=k)
    # =====================================================================
    plan_fanout_edges = [
        (
            "controller",
            "ExemplarPicker",
            {
                "exemplars": "",
                "algorithm": "",
                "problem": "",
                "sample_io": "",
                "language": "",
            },
        ),
        (
            "ExemplarPicker",
            "PlanGen",
            {
                "exemplar_problem": "",
                "exemplar_plan": "",
                "algorithm": "",
                "problem": "",
                "sample_io_str": "",
                "language": "",
            },
        ),
        # Static-context passthrough so ConfidenceEval's prompt can render
        # {problem} and {language} without going through PlanGen.
        ("ExemplarPicker", "ConfidenceEval", {"problem": "", "language": ""}),
        # PlanGen's single LLM output: the plan text. Goes to BOTH the
        # ConfidenceEval prompt (as `plan`) and back to the controller (so
        # terminate_fn can stash it in `plan_buffer`).
        ("PlanGen", "ConfidenceEval", {"plan": ""}),
        ("PlanGen", "controller", {"plan": ""}),
        # ConfidenceEval's single LLM output: the raw XML reply with the
        # `<confidence>NN</confidence>` block.
        ("ConfidenceEval", "controller", {"confidence_raw": ""}),
    ]

    PlanFanoutT = NodeTemplate(
        Loop,
        max_iterations=max(1, k),
        terminate_condition_function=planfanout_terminate,
        nodes=[
            ("ExemplarPicker", Shared(ExemplarPickerT)),
            ("PlanGen", Shared(PlanGenT)),
            ("ConfidenceEval", Shared(ConfidenceT)),
        ],
        edges=plan_fanout_edges,
        pull_keys={
            "exemplars": "",
            "algorithm": "",
            "problem": "",
            "sample_io": "",
            "language": "",
        },
        push_keys={
            "plan_buffer": "",
            "problem": "",
            "algorithm": "",
            "sample_io": "",
            "language": "",
        },
    )

    # =====================================================================
    # DebugLoop (inner Loop, max_iter=t)
    # =====================================================================
    debug_loop_edges = [
        (
            "controller",
            "Debugging",
            {
                "code": "",
                "observation": "",
                "current_plan": "",
                "algorithm": "",
                "problem": "",
                "sample_io": "",
                "language": "",
            },
        ),
        # Debugging's single LLM output: the new candidate code (raw markdown
        # with code fence). Tester2 will fence-extract it.
        ("Debugging", "Tester2", {"code": ""}),
        # Tester2 emits structured fields back to the controller.
        (
            "Tester2",
            "controller",
            {"code": "", "observation": "", "full_passed": ""},
        ),
    ]

    DebugLoopT = NodeTemplate(
        Loop,
        max_iterations=max(1, t),
        terminate_condition_function=debugloop_terminate,
        nodes=[
            ("Debugging", Shared(DebugT)),
            ("Tester2", Shared(TesterT)),
        ],
        edges=debug_loop_edges,
        pull_keys={"sample_io": ""},
        push_keys={"final_code": "", "final_passed": ""},
    )

    # =====================================================================
    # PlanIteration (outer Loop, max_iter=k)
    # =====================================================================
    def _passed_route(message: dict[str, Any], _attrs: dict[str, Any]) -> bool:
        return bool(message.get("full_passed"))

    def _failed_route(message: dict[str, Any], _attrs: dict[str, Any]) -> bool:
        return not bool(message.get("full_passed"))

    # Inside a Loop named "PlanIteration" the framework names the controller
    # `PlanIteration_controller`; that is what LogicSwitch.routes must match.
    PassSwitchT = NodeTemplate(
        LogicSwitch,
        routes={
            "PlanIteration_controller": _passed_route,
            "DebugLoop": _failed_route,
        },
    )

    plan_iteration_edges = [
        (
            "controller",
            "PlanPicker",
            {
                "sorted_plans": "",
                "problem": "",
                "algorithm": "",
                "sample_io": "",
                "language": "",
            },
        ),
        (
            "PlanPicker",
            "Coding",
            {
                "current_plan": "",
                "problem": "",
                "algorithm": "",
                "sample_io_str": "",
                "language": "",
            },
        ),
        # PlanPicker also seeds PassSwitch with the static debug context so
        # that — when the candidate fails — PassSwitch can forward
        # `{current_plan, algorithm, problem, sample_io, language}` to
        # DebugLoop without going through the Coding Agent.
        (
            "PlanPicker",
            "PassSwitch",
            {
                "current_plan": "",
                "algorithm": "",
                "problem": "",
                "sample_io": "",
                "language": "",
            },
        ),
        ("Coding", "Tester", {"code": ""}),
        (
            "Tester",
            "PassSwitch",
            {"code": "", "observation": "", "full_passed": ""},
        ),
        # passed branch -> outer controller
        ("PassSwitch", "controller", {"code": "", "full_passed": ""}),
        # failed branch -> DebugLoop
        (
            "PassSwitch",
            "DebugLoop",
            {
                "code": "",
                "observation": "",
                "current_plan": "",
                "algorithm": "",
                "problem": "",
                "sample_io": "",
                "language": "",
            },
        ),
        ("DebugLoop", "controller", {"final_code": "", "final_passed": ""}),
    ]

    PlanIterationT = NodeTemplate(
        Loop,
        max_iterations=max(1, k),
        terminate_condition_function=plan_iteration_terminate,
        nodes=[
            ("PlanPicker", Shared(PlanPickerT)),
            ("Coding", Shared(CodingT)),
            ("Tester", Shared(TesterT)),
            ("PassSwitch", Shared(PassSwitchT)),
            ("DebugLoop", Shared(DebugLoopT)),
        ],
        edges=plan_iteration_edges,
        pull_keys={
            "sorted_plans": "",
            "problem": "",
            "algorithm": "",
            "sample_io": "",
            "language": "",
        },
        push_keys={"final_code": "", "final_passed": ""},
    )

    # =====================================================================
    # RootGraph
    # =====================================================================
    # `Shared(...)` is for NESTED NodeTemplate references (inside another
    # template's `nodes=[...]`); at the RootGraph level we pass templates
    # directly so `BaseGraph.create_node` can dispatch on `isinstance(...,
    # NodeTemplate)`.
    g = RootGraph(
        name="MapCoder",
        nodes=[
            ("Retrieval", RetrievalT),
            ("RetrievalParser", RetrievalParserT),
            ("PlanFanout", PlanFanoutT),
            ("PlanSorter", PlanSorterT),
            ("PlanIteration", PlanIterationT),
        ],
        edges=[
            ("entry", "Retrieval", {"problem": "", "language": "", "k": ""}),
            # Retrieval Agent's single LLM output is the raw XML reply.
            ("Retrieval", "RetrievalParser", {"content": ""}),
            (
                "RetrievalParser",
                "PlanFanout",
                {
                    "exemplars": "",
                    "algorithm": "",
                    "problem": "",
                    "sample_io": "",
                    "language": "",
                },
            ),
            (
                "PlanFanout",
                "PlanSorter",
                {
                    "plan_buffer": "",
                    "problem": "",
                    "algorithm": "",
                    "sample_io": "",
                    "language": "",
                },
            ),
            (
                "PlanSorter",
                "PlanIteration",
                {
                    "sorted_plans": "",
                    "problem": "",
                    "algorithm": "",
                    "sample_io": "",
                    "language": "",
                },
            ),
            ("PlanIteration", "exit", {"final_code": "", "final_passed": ""}),
        ],
        attributes={
            "k": k,
            "t": t,
        },
    )
    g.build()
    return g


# Convenience build-only smoke (no LLM calls).
if __name__ == "__main__":  # pragma: no cover
    from masfactory import LegacyOpenAIModel

    api_key = os.environ.get("OPENAI_API_KEY") or "sk-dummy-build-only"
    base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("BASE_URL") or None
    model = LegacyOpenAIModel(
        api_key=api_key,
        base_url=base_url,
        model_name=os.environ.get("OPENAI_MODEL_NAME", "gpt-4o-mini"),
    )
    g = build_graph(model, k=3, t=3)
    print("MapCoder graph built ok:", g.name)
