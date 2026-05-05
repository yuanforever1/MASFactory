"""
Verbatim prompt templates from MapCoder paper (Appendix B, Figures 8/9/10).

Reference:
  Md. Ashraful Islam, Mohammed Eunus Ali, Md Rizwan Parvez.
  "MapCoder: Multi-Agent Code Generation for Competitive Problem Solving."
  ACL 2024. arXiv:2405.11403.

These strings are consumed by Agent NodeTemplates in `applications/mapcoder/components/`.
Placeholders use Python `str.format` style (e.g. `{language}`); the underlying MASFactory
Agent renders them via `prompt_template_format` against the Agent's context knowledge.
"""

# ---------------------------------------------------------------------------
# Retrieval Agent  (Figure 8)
# ---------------------------------------------------------------------------

RETRIEVAL_SYSTEM = (
    "You are a competitive programming expert. Given a problem you must (1) recall "
    "k relevant past problems together with their solutions and plans, and (2) "
    "identify the underlying algorithm and write a high level tutorial for solving "
    "this kind of problem. Follow the user's response format strictly."
)

RETRIEVAL_USER_TEMPLATE = """\
Given a problem, provide relevant problems then identify the algorithm behind it and also explain the tutorial of the algorithm.

# Problem:
{problem}

# Exemplars:
Recall {k} relevant and distinct problems (different from problem mentioned above). For each problem,
1. describe it
2. generate {language} code step by step to solve that problem
3. finally generate a planning to solve that problem

# Algorithm:

----------------
Important:
Your response must follow the following xml format-

<root>
<problem>
# Recall k relevant and distinct problems (different from problem mentioned above). Write each problem in the following format.
<description> # Describe the problem. </description>
<code> # Let's think step by step to solve this problem in {language} programming language. </code>
<planning> # Planning to solve this problem. </planning>
</problem>

# similarly add more problems here...

<algorithm>
# Identify the algorithm (Brute-force, Dynamic Programming, Divide-and-conquer, Greedy, Backtracking, Recursive, Binary search, and so on) that needs to be used to solve the original problem.
# Write a useful tutorial about the above mentioned algorithms. Provide a high level generic tutorial for solving this types of problem. Do not generate code.
</algorithm>
</root>
"""


# ---------------------------------------------------------------------------
# Planning Agent  (Figure 9, top half) - per-exemplar plan generation
# ---------------------------------------------------------------------------

PLAN_GEN_SYSTEM = (
    "You are a competitive programming expert. Generate a concrete, actionable plan "
    "to solve the target problem, leveraging the example problem/plan as inspiration."
)

PLAN_GEN_TEMPLATE = """\
Given a competitive programming problem generate a concrete planning to solve the problem.

# Problem: {exemplar_problem}
# Planning: {exemplar_plan}

## Relevant Algorithm to solve the next problem:
{algorithm}

## Problem to be solved:
{problem}

## Sample Input/Outputs:
{sample_io_str}

----------------
Important: You should give only the planning to solve the problem. Do not add extra explanation or words.
"""


# ---------------------------------------------------------------------------
# Confidence Agent  (Figure 9, bottom half) - score 0-100
# ---------------------------------------------------------------------------

CONFIDENCE_SYSTEM = (
    "You are a strict evaluator. Given a problem and a plan, decide whether the plan "
    "is correct to solve this problem and emit an integer confidence score 0-100. "
    "Follow the xml output format."
)

CONFIDENCE_TEMPLATE = """\
Given a competitive programming problem and a plan to solve the problem in {language} tell whether the plan is correct to solve this problem.

# Problem: {problem}
# Planning: {plan}

----------------
Important: Your response must follow the following xml format-

<root>
<explanation> Discuss whether the given competitive programming problem is solvable by using the above mentioned planning. </explanation>
<confidence> Confidence score regarding the solvability of the problem. Must be an integer between 0 and 100. </confidence>
</root>
"""


# ---------------------------------------------------------------------------
# Coding Agent  (Figure 10, top half)
# ---------------------------------------------------------------------------

CODING_SYSTEM = (
    "You are a competitive programming expert. Translate the given plan into a "
    "complete, runnable program in the requested language. Output ONLY the code "
    "inside a fenced code block."
)

CODING_TEMPLATE = """\
Given a competitive programming problem generate {language} code to solve the problem.

## Relevant Algorithm to solve the next problem:
{algorithm}

## Problem to be solved:
{problem}

## Planning:
{current_plan}

## Sample Input/Outputs:
{sample_io_str}

## Let's think step by step.

------------
Important:
## Your response must contain only the {language} code to solve this problem inside a ```{language} ... ``` block. Do not add extra explanation or words.
"""


# ---------------------------------------------------------------------------
# Debugging Agent  (Figure 10, bottom half)
# ---------------------------------------------------------------------------

DEBUG_SYSTEM = (
    "You are a competitive programming expert. Improve the previously generated "
    "code so that it passes the failing sample I/O. Cross-check against the plan "
    "and the algorithm tutorial."
)

DEBUG_TEMPLATE = """\
Given a competitive programming problem you have generated {language} code to solve the problem. But the generated code can not pass sample test cases. Improve your code to solve the problem correctly.

## Relevant Algorithm to solve the next problem:
{algorithm}

## Problem to be solved:
{problem}

## Planning:
{current_plan}

## Code:
{code}

## Test Report:
{observation}

## Modified Planning:
## Let's think step by step to modify {language} Code for solving this problem.

----------------
Important:
## Your response must contain the modified planning and then the {language} code inside a ```{language} ... ``` block to solve this problem.
"""
