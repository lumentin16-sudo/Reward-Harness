"""Eval-Skill 官方 vanilla pairwise judging Harness。"""

from __future__ import annotations

from ..reward_system import Query, Response, RewardSystem, RubricSet, SkillRegistry, WinnerResult
from ._eval_skill_common import (
    VANILLA_PAIRWISE_JUDGE_PROMPT,
    response_block,
    winner_result,
)


HARNESS_NAME = "eval_skill_vanilla"


class EvalSkillVanillaHarness(RewardSystem):
    judge_prompt_template = VANILLA_PAIRWISE_JUDGE_PROMPT

    def get_skill_registry(self, task: Query) -> SkillRegistry:
        return SkillRegistry()

    def build_rubrics(
        self, task: Query, responses: tuple[Response, ...]
    ) -> RubricSet:
        return RubricSet(
            query_id=task.query_id,
            rubrics=(),
            metadata={"method": "eval_skill_vanilla"},
        )

    def judge(
        self,
        task: Query,
        responses: tuple[Response, ...],
        rubrics: RubricSet,
    ) -> WinnerResult:
        prompt = self.judge_prompt_template.format(
            instruction=task.instruction,
            response_block=response_block(responses),
        )
        return winner_result(
            task,
            responses,
            self.judge_llm(prompt),
            method="vanilla",
        )


HARNESS_CLASS = EvalSkillVanillaHarness
