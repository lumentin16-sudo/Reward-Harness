"""Eval-Skill 官方 skill pairwise judging Prompt 的初始 Harness。"""

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
from ._eval_skill_common import (
    SKILL_PAIRWISE_JUDGE_PROMPT,
    response_block,
    winner_result,
)


HARNESS_NAME = "eval_skill"


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


class EvalSkillHarness(RewardSystem):
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
                "method": "eval_skill",
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
            response_block=response_block(responses),
        )
        return winner_result(
            task,
            responses,
            self.judge_llm(prompt),
            method="eval_skill",
        )


HARNESS_CLASS = EvalSkillHarness
