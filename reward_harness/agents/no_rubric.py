"""Eval-Skill 官方 vanilla pairwise judging Harness。"""

from __future__ import annotations

from ..reward_system import (
    Query,
    Response,
    RewardSystem,
    RubricSet,
    SkillRegistry,
    WinnerResult,
)


HARNESS_NAME = "no_rubric"


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


def _response_block(responses: tuple[Response, ...]) -> str:
    if len(responses) != 2:
        raise ValueError("Eval-Skill pairwise judging requires exactly 2 responses")
    return "".join(
        f"--- Response {label} ---\n"
        f"{response.content.strip()}\n"
        f"--- End Response {label} ---\n"
        for label, response in zip(("A", "B"), responses)
    )


def _winner_result(
    query: Query,
    responses: tuple[Response, ...],
    evaluation: str,
) -> WinnerResult:
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
            "method": "no_rubric",
        },
    )


class NoRubricHarness(RewardSystem):
    judge_prompt_template = VANILLA_PAIRWISE_JUDGE_PROMPT

    def get_skill_registry(self, task: Query) -> SkillRegistry:
        return SkillRegistry()

    def build_rubrics(
        self, task: Query, responses: tuple[Response, ...]
    ) -> RubricSet:
        return RubricSet(
            query_id=task.query_id,
            rubrics=(),
            metadata={"method": "no_rubric"},
        )

    def judge(
        self,
        task: Query,
        responses: tuple[Response, ...],
        rubrics: RubricSet,
    ) -> WinnerResult:
        prompt = self.judge_prompt_template.format(
            instruction=task.instruction,
            response_block=_response_block(responses),
        )
        return _winner_result(
            task,
            responses,
            self.judge_llm(prompt),
        )


HARNESS_CLASS = NoRubricHarness
