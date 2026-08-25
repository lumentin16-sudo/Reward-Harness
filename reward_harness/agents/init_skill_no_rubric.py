"""只在 Judge 阶段注入 Skill、但不生成 Rubric 的初始 Harness。"""

from __future__ import annotations

from ..reward_system import (
    Query,
    Response,
    RewardSystem,
    RubricSet,
    Skill,
    SkillRegistry,
    WinnerResult,
)
HARNESS_NAME = "init_skill_no_rubric"


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
            "method": "init_skill_no_rubric",
        },
    )


EVAL_SKILL = Skill(
    name="pairwise_evaluation_workflow",
    stage="J",
    description="Compare two responses using an evidence-first evaluation workflow.",
    content="""## Analysis
1. Reconstruct the user's core intent and binding explicit constraints.
2. Evaluate Response A and Response B independently for correctness,
   instruction following, relevance, completeness, safety, and clarity, applying
   only dimensions that matter to the task.
3. Cite concrete evidence and classify defects by consequence. A fatal or major
   answer-changing error outweighs multiple stylistic strengths.
4. Compare the responses directly. Do not reward verbosity, confidence,
   familiar phrasing, or formatting that the user did not request.

## Final Judgment
Aggregate the decisive differences, explain why they matter to task success,
and select exactly one winner. Never output None, Neither, or a tie.
""",
)


class InitSkillNoRubricHarness(RewardSystem):
    """把可由 Harness Optimization 编辑的离线 Skill 注入官方 Judge Prompt。"""

    judge_prompt_template = SKILL_PAIRWISE_JUDGE_PROMPT

    def get_skill_registry(self, task: Query) -> SkillRegistry:
        return SkillRegistry((EVAL_SKILL,))

    def build_rubrics(
        self, task: Query, responses: tuple[Response, ...]
    ) -> RubricSet:
        return RubricSet(
            query_id=task.query_id,
            rubrics=(),
            metadata={
                "method": "init_skill_no_rubric",
                "online_rubric_generation": False,
                "skill": EVAL_SKILL.name,
            },
        )

    def judge(
        self,
        task: Query,
        responses: tuple[Response, ...],
        rubrics: RubricSet,
    ) -> WinnerResult:
        prompt = self.judge_prompt_template.format(
            instruction=task.instruction,
            skill=EVAL_SKILL.content,
            response_block=_response_block(responses),
        )
        return _winner_result(
            task,
            responses,
            self.judge_llm(prompt),
        )


HARNESS_CLASS = InitSkillNoRubricHarness
