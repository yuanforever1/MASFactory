# MapCoder × MASFactory

Self-contained reproduction of [MapCoder: Multi-Agent Code Generation for Competitive Problem Solving](http://arxiv.org/abs/2405.11403) (ACL 2024) on top of MASFactory.

## Workflow

```
entry --(problem, language, k)--> Retrieval (Agent)
   --(content)--> RetrievalParser (CustomNode)
   --(exemplars, algorithm, ...)--> PlanFanout (Loop, k iters)
       inner: ExemplarPicker -> PlanGen -> ConfidenceEval -> controller
   --(plan_buffer, ...)--> PlanSorter (CustomNode)
   --(sorted_plans, ...)--> PlanIteration (Loop, up to k iters)
       inner: PlanPicker -> Coding -> Tester -> PassSwitch
              -> [passed] back to controller
              -> [failed] DebugLoop (Loop, t iters)
                       inner: Debugging -> Tester2 -> controller
   --(final_code, final_passed)--> exit
```

See [`docs/workflow_design.md`](docs/workflow_design.md) for the full design contract (node templates, edge field-keys, terminate conditions).

## Files

| Path | Purpose |
| --- | --- |
| `workflows/graph.py`        | `build_graph(model, k, t) -> RootGraph` |
| `workflows/controllers.py`  | terminate-condition functions for the three Loops |
| `components/retrieval.py`   | Retrieval Agent NodeTemplate + RetrievalParser CustomNode |
| `components/planning.py`    | PlanGen / ConfidenceEval Agents + ExemplarPicker / PlanSorter / PlanPicker |
| `components/coding.py`      | Coding Agent NodeTemplate |
| `components/debugging.py`   | Debugging Agent NodeTemplate |
| `components/tester.py`      | Tester CustomNode (code extraction + sandbox + sample-IO assertions) |
| `prompts/mapcoder_prompts.py` | Verbatim Appendix-B prompts (Figures 8 / 9 / 10) |
| `humaneval/dataset.py`      | HumanEval JSONL loader + docstring sample-IO parser + hidden-test verifier |
| `humaneval/runner.py`       | Per-problem driver + Pass@1 aggregator |
| `main.py`                   | CLI entrypoint |
| `tests/`                    | Local tests (no API key required) |

## Running

### Tests (no API key required)

```bash
python -m applications.mapcoder.tests.test_tester_node
python -m applications.mapcoder.tests.test_dataset
python -m applications.mapcoder.tests.test_graph_e2e_mock
```

### Real LLM run on a HumanEval JSONL

```bash
# PowerShell
$env:OPENAI_API_KEY = "sk-..."
$env:OPENAI_BASE_URL = "https://your-openai-compatible-endpoint/v1"  # optional
$env:MAPCODER_MODEL = "gpt-4o-mini"                                 # optional

python -m applications.mapcoder.main `
    --dataset path\to\HumanEval.jsonl `
    --limit 10 `
    --k 3 --t 3
```

## Notes

- The Tester runs each candidate inside an isolated `sys.executable -I -c <driver>` subprocess with a 10s timeout (override with `MAPCODER_SANDBOX_TIMEOUT_S`).
- Per the paper, debug feedback only uses sample I/O recovered from the HumanEval prompt docstring; the hidden `test` field is consulted **only** by `humaneval/runner.py` for Pass@1 scoring (never inside the graph).
- The framework occasionally seeds missing controller-input keys with the literal string `"(not set yet)"`. All terminate functions in `workflows/controllers.py` filter that placeholder.
