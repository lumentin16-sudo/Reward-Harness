"""Eval-Skill 官方 online rubric-based pairwise judging Harness。"""

from __future__ import annotations

from ..reward_system import (
    Query,
    Response,
    RewardSystem,
    Rubric,
    RubricSet,
    SkillRegistry,
    WinnerResult,
)
from ._eval_skill_common import (
    RUBRIC_GENERATION_PROMPT,
    RUBRIC_PAIRWISE_JUDGE_PROMPT,
    response_block,
    winner_result,
)


HARNESS_NAME = "eval_skill_rubric"


class EvalSkillRubricHarness(RewardSystem):
    rubric_prompt_template = RUBRIC_GENERATION_PROMPT
    judge_prompt_template = RUBRIC_PAIRWISE_JUDGE_PROMPT

    def get_skill_registry(self, task: Query) -> SkillRegistry:
        return SkillRegistry()

    def build_rubrics(
        self, task: Query, responses: tuple[Response, ...]
    ) -> RubricSet:
        raw_rubric = self.rubric_llm(
            self.rubric_prompt_template.format(prompt=task.instruction)
        ).strip()
        if len(raw_rubric) < 10:
            raise ValueError("Eval-Skill rubric generation returned an empty rubric")
        return RubricSet(
            query_id=task.query_id,
            rubrics=(
                Rubric(
                    rubric_id="eval_skill_online_rubric",
                    criterion=raw_rubric,
                ),
            ),
            metadata={"method": "eval_skill_rubric_based"},
        )

    def judge(
        self,
        task: Query,
        responses: tuple[Response, ...],
        rubrics: RubricSet,
    ) -> WinnerResult:
        if len(rubrics.rubrics) != 1:
            raise ValueError("Eval-Skill rubric Harness expects one online rubric")
        prompt = self.judge_prompt_template.format(
            instruction=task.instruction,
            rubric=rubrics.rubrics[0].criterion,
            response_block=response_block(responses),
        )
        return winner_result(
            task,
            responses,
            self.judge_llm(prompt),
            method="rubric_based",
        )


HARNESS_CLASS = EvalSkillRubricHarness
