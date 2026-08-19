"""不生成 Rubric、由 Judge 直接输出标量奖励的 baseline。"""

from __future__ import annotations

import json

from ..reward_system import (
    Response,
    JudgmentResult,
    RewardResult,
    RewardSystem,
    Query,
    RubricSet,
    SkillRegistry,
    _extract_json_object,
)


DIRECT_SCORE_PROMPT = """You are a fair and impartial pointwise reward model.

Evaluate one candidate response for one public task. Judge only the response
shown below. You cannot see any competing response, preference label, or pair
position, and must not guess them.

[Public Task]
{task_json}

[Response]
{candidate_json}

Evaluation protocol:
1. Identify the user's core intent and every explicit constraint in the task.
2. Check the response's correctness, instruction following, relevance,
   completeness, reasoning quality, safety, and clarity where applicable.
3. Distinguish major errors that change the answer from minor omissions or
   presentation flaws. Do not reward verbosity, confidence, polished style, or
   repeated claims unless they improve task fulfillment.
4. Ground the reason in concrete response content, then assign a calibrated
   score. Use the anchors as guidance, not as the only allowed values:
   - 1.00: fully correct and satisfies all important requirements;
   - 0.75: mostly correct and useful, with only minor limitations;
   - 0.50: mixed quality, partially useful but with substantive defects;
   - 0.25: major errors or omissions, with little usable value;
   - 0.00: wholly incorrect, irrelevant, unsafe, or unusable.

Return JSON only. Put the reason before the score so the judgment is based on
an explicit assessment rather than an unsupported number:
{{
  "reason": "brief evidence-based explanation",
  "score": 0.0
}}

Requirements:
- score must be a finite number between 0.0 and 1.0 inclusive.
- Use sufficient decimal resolution to express meaningful quality differences;
  do not mechanically round every response to an anchor.
- Base the score only on the public task and candidate content.
"""


class NoRubricHarness(RewardSystem):
    """跳过 G 阶段模型调用，直接执行单候选标量评分。"""

    judge_prompt_template = DIRECT_SCORE_PROMPT

    def get_skill_registry(self, task: Query) -> SkillRegistry:
        return SkillRegistry()

    def build_rubrics(self, task: Query) -> RubricSet:
        """保留统一接口，但不调用 Rubric Model。"""

        return RubricSet(
            query_id=task.query_id,
            rubrics=(),
            metadata={"baseline": "no_rubric"},
        )

    def score(
        self,
        task: Query,
        candidate: Response,
        rubrics: RubricSet,
    ) -> JudgmentResult:
        task_payload = self._task_payload(task)
        candidate_payload = self._candidate_payload(candidate)
        prompt = self.judge_prompt_template.format(
            task_json=json.dumps(task_payload, ensure_ascii=False, indent=2),
            candidate_json=json.dumps(candidate_payload, ensure_ascii=False, indent=2),
        )
        raw_response = self.judge_llm(prompt)
        payload = _extract_json_object(raw_response)
        raw_score = payload.get("score")
        if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
            raise ValueError("direct judge response must contain a numeric score")
        direct_score = float(raw_score)
        reason = payload.get("reason", "")
        if reason is not None and not isinstance(reason, str):
            raise ValueError("direct judge reason must be a string when provided")

        return JudgmentResult(
            query_id=task.query_id,
            response_id=candidate.response_id,
            judgments=(),
            metadata={
                "direct_score": direct_score,
                "reason": reason or "",
            },
        )

    def aggregate(
        self,
        task: Query,
        candidate: Response,
        rubrics: RubricSet,
        judgment_result: JudgmentResult,
    ) -> RewardResult:
        """A：把 Judge 已给出的直接标量作为最终 reward。"""

        direct_score = judgment_result.metadata.get("direct_score")
        if isinstance(direct_score, bool) or not isinstance(
            direct_score, (int, float)
        ):
            raise ValueError("no_rubric JudgmentResult is missing direct_score")
        return RewardResult(
            query_id=task.query_id,
            response_id=candidate.response_id,
            reward=float(direct_score),
            metadata={
                "aggregation": "direct_scalar",
                "reason": str(judgment_result.metadata.get("reason", "")),
            },
        )
