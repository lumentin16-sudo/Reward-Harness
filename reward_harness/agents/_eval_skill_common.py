"""Eval-Skill pairwise Prompt 与 forced-choice 结果绑定工具。"""

from __future__ import annotations

from ..reward_system import Query, Response, WinnerResult


RUBRIC_GENERATION_PROMPT = """
**Task:** Extract a set of evaluation criteria (a rubric) from the `<Target Request>`. The rubric will be used to assess whether a response fully satisfies the request.

1. **Comprehensiveness:**
   - The rubric must cover all critical aspects implied by the request and examples, including both explicit requirements and implicit quality standards.

2. **Conciseness & Uniqueness:**
   - Each item must represent a distinct evaluation criterion. Merge overlapping or redundant criteria into a single item. Use precise, non-repetitive wording.
   - Do NOT introduce formatting requirements or constraints unless they are explicitly specified in the `<Target Request>`.

3. **Categorization:** Classify each rubric item using one of the following tags:
   - `[Core Requirement]`: Directly derived from explicit requirements in the `<Target Request>` (e.g., format, length, structure, required/forbidden elements).
   - `[Qualitative Nuance]`: Derived by abstracting concrete cues into domain-agnostic quality criteria (e.g., clarity, correctness, reasoning quality, pedagogy).

**[Target Request]:**
{prompt}

**Rubric Format Requirements:**
- Use a numbered list.
- Each item must begin with "The response" (third-person phrasing).
- Append `[Core Requirement]` or `[Qualitative Nuance]` to the end of each item.
- Do NOT include reasoning, explanations, or examples—output only the rubric items.

**Output Rubrics:**
""".strip()


VANILLA_PAIRWISE_JUDGE_PROMPT = """
You are a fair and impartial judge. Your task is to evaluate 'Response A' and 'Response B' based on a given instruction to select the single best response. You will conduct this evaluation in distinct phases as outlined below.

### Phase 1: Analyze Each Response
Analyze each response individually against the user instruction. Think step-by-step about the strengths and weaknesses of each response, considering factors such as helpfulness, clarity, accuracy, formatting, and adherence to the user's explicit and implicit constraints. Provide a concise justification for your findings for each response.

### Phase 2: Final Judgment Instructions
Based on the results from the previous phase, determine the overall winner. Provide a final justification explaining your decision first, and then give your final decision. Consider which response best meets the user's needs with the fewest flaws.
Think step-by-step to aggregate the findings and make the decision; keep the reasoning explicit and concise.
**NOTE**: You must select a winner. Never respond with "None" or "Neither" as the winner.

### REQUIRED OUTPUT FORMAT
You must follow this exact output format below.

--- Analysis ---
**Response A:** Justification: <...>
**Response B:** Justification: <...>

--- Final Judgment ---
Aggregation Summary: <1-3 sentences explaining why the winning response was chosen over the other>
Justification: <...>
Winner: <Response A / Response B>

Task to Evaluate:
Instruction:
{instruction}

{response_block}
""".strip()


RUBRIC_PAIRWISE_JUDGE_PROMPT = """
You are a fair, impartial, and highly capable judge. Your task is to evaluate 'Response A' and 'Response B' based on a given instruction and a rubric.

### Phase 1: Core Intent Alignment
First, identify the fundamental "Core Intent" of the user's instruction. What is the user actually trying to achieve or learn?
If any rubric criteria seem overly specific, arbitrary, or disconnected from this Core Intent, you must prioritize the Core Intent over those specific criteria. Write a 1-2 sentence summary of the Core Intent.

### Phase 2: Holistic Analysis
Evaluate each response against the rubric criteria, keeping the Core Intent in mind.
- Assess **conceptual accuracy** over mere keyword density. A response that lists many details but misses the conceptual truth is worse than a concise, accurate response.
- For each response, cite concrete evidence, but evaluate the *degree* of quality rather than a simple pass/fail.

### Phase 3: Final Judgment
Based on the analysis, determine the winner. Weigh the [Core Requirement] items heavily, but use the [Qualitative Nuance] items to break ties or identify superior reasoning.
Think step-by-step to aggregate the findings and make the decision; keep the reasoning explicit and concise.
**NOTE**: You must select a winner even if all responses fail the requirements. Never respond with "None" or "Neither" as the winner.

### REQUIRED OUTPUT FORMAT
You must follow this exact output format below.

--- Core Intent Alignment ---
Identified Core Intent: <1-2 sentences explaining the fundamental goal of the user's instruction>

--- Analysis ---
**Response A:**
- Criterion 1 [Tag]: Justification: <...>
- Criterion 2 [Tag]: Justification: <...>
- Conceptual Accuracy & Tone: <Briefly assess if the response actually answers the Core Intent accurately>

**Response B:**
- Criterion 1 [Tag]: Justification: <...>
- Criterion 2 [Tag]: Justification: <...>
- Conceptual Accuracy & Tone: <...>

--- Final Judgment ---
Aggregation Summary: <1-3 sentences explaining how the responses aligned with the Core Intent and the rubric to lead to your decision>
Justification: <...>
Winner: <Response A / Response B>

Task to Evaluate:
Instruction:
{instruction}

Rubric:
{rubric}

{response_block}
""".strip()


SKILL_PAIRWISE_JUDGE_PROMPT = """
You are a fair and impartial judge. Your task is to evaluate 'Response A' and 'Response B' based on a given instruction to select the single best response.
**NOTE**: You must select a winner. Never respond with "None" or "Neither" as the winner.

You may use the Skill derived from other validated examples, as references if helpful.

**Skill:**
{skill}
-----------------END OF THE SKILL------------------

### REQUIRED OUTPUT FORMAT
You must follow this exact output format below. Conduct your detailed analysis in the `Analysis` section, following the exact structure, workflow, and instructions defined in the **Skill**, and finally give your decision in the `Final Judgment` section.

--- Analysis ---
<Conduct your detailed analysis following the exact workflow and instructions defined in the Skill>

--- Final Judgment ---
Aggregation Summary: <1-3 sentences explaining why the winning response was chosen over the other>
Justification: <...>
Winner: <Response A / Response B>

Task to Evaluate:
Instruction:
{instruction}

{response_block}
""".strip()


def require_pair(responses: tuple[Response, ...]) -> None:
    if len(responses) != 2:
        raise ValueError("Eval-Skill pairwise judging requires exactly 2 responses")


def response_block(responses: tuple[Response, ...]) -> str:
    require_pair(responses)
    return "".join(
        f"--- Response {label} ---\n"
        f"{response.content.strip()}\n"
        f"--- End Response {label} ---\n"
        for label, response in zip(("A", "B"), responses)
    )


def winner_result(
    query: Query,
    responses: tuple[Response, ...],
    evaluation: str,
    *,
    method: str,
) -> WinnerResult:
    require_pair(responses)
    prediction = (
        evaluation.rsplit("Final Decision", 1)[-1]
        .rsplit("Final Judgment", 1)[-1]
        .split("Winner", 1)[-1]
        .split("Response", 1)[-1]
        .split("Candidate", 1)[-1]
    )
    prediction = (
        prediction.replace("*", "")
        .replace(":", "")
        .replace(".", "")
        .replace(" ", "")
        .strip()
    )
    label = prediction[0].upper() if prediction else ""
    if label not in {"A", "B"}:
        raise ValueError("Eval-Skill judge must declare Winner: Response A/B")
    return WinnerResult(
        query_id=query.query_id,
        winner_response_id=responses[ord(label) - ord("A")].response_id,
        metadata={
            "winner_label": label,
            "comparison": "pairwise_forced_choice",
            "method": method,
        },
    )
