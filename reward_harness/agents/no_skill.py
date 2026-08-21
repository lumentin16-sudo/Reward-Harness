"""不注入 Skill 的独立 Reward Harness baseline。"""

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
)


RUBRIC_GENERATION_PROMPT = """You are an expert evaluation-rubric designer.

Generate one shared set of task-specific evaluation rubrics from the public
query and an anonymized, unlabeled response set. Preference labels are
intentionally unavailable; do not infer them from response order.

[Public Task]
{task_json}

[Anonymized Response Set]
{responses_json}

Requirements:
- First infer the task's core intent, explicit constraints, and necessary
  implicit quality standards.
- Compare the responses only to discover substantive quality differences and
  common omissions. The response order is arbitrary and carries no meaning.
- Return 2 to 6 task-specific, non-overlapping rubrics that jointly cover the
  critical requirements. Avoid generic criteria unless the task requires them.
- Each rubric must be one atomic, binary-verifiable requirement observable from
  a single response. It must be possible to answer only PASS or FAIL.
- Make the pass condition concrete and self-contained. State the required fact,
  reasoning step, constraint, or behavior instead of using broad labels such as
  "correct", "clear", "helpful", or "high quality" without an explicit test.
- A response receives PASS only when it fully satisfies the stated condition;
  partial satisfaction must be scored FAIL.
- Every rubric must remain independently applicable to any single response.
  Never mention response positions, identifiers, comparisons, winners, or the
  observed response set in the criterion.
- Cover indispensable query requirements even when every observed response
  misses them. Ignore incidental wording, verbosity, and formatting differences.
- Give explicit/core requirements more weight than optional qualitative nuance.
- weight must be between 0.5 and 2.0 and represent relative importance.
- Do not invent formatting constraints or candidate-specific requirements.
- Return JSON only, with this schema:
{{
  "rubrics": [
    {{
      "rubric_id": "short_unique_id",
      "criterion": "PASS if the response ...; otherwise FAIL",
      "weight": 1.0
    }}
  ]
}}
"""


RUBRIC_EVALUATION_PROMPT = """You are a fair and impartial pointwise reward judge.

Evaluate the single response against the one supplied rubric. Do not compare it
with another answer, infer its pair position, or assume access to preference
labels.

[Public Task]
{task_json}

[Single Rubric]
{rubrics_json}

[Response]
{candidate_json}

Requirements:
- Return exactly one judgment for the supplied rubric_id.
- score must be the integer 1 when the response fully satisfies the binary pass
  condition, otherwise 0. Do not award partial credit.
- evidence must quote or briefly identify observable response content.
- Prioritize correctness, core intent, and explicit instruction following.
  Do not reward verbosity, polished style, or repeated claims by themselves.
- Return JSON only, with this schema:
{{
  "judgments": [
    {{
      "rubric_id": "matching rubric id",
      "score": 0,
      "evidence": ["short evidence"],
      "confidence": 0.0
    }}
  ]
}}
"""


class NoSkillHarness(RewardSystem):
    """直接实现 RewardSystem 接口的独立 no-skill baseline。"""

    rubric_prompt_template = RUBRIC_GENERATION_PROMPT
    judge_prompt_template = RUBRIC_EVALUATION_PROMPT

    def get_skill_registry(self, task: Query) -> SkillRegistry:
        return SkillRegistry()

    def aggregate(
        self,
        task: Query,
        candidate: Response,
        rubrics: RubricSet,
        judgment_result: JudgmentResult,
    ) -> RewardResult:
        """A：按 Rubric 权重聚合二值 Judgment。"""

        judgments = judgment_result.judgments
        if not judgments:
            raise ValueError("cannot aggregate an empty judgment set")
        for judgment in judgments:
            if not isinstance(judgment.score, int) or judgment.score not in {0, 1}:
                raise ValueError("no_skill expects binary integer scores 0 or 1")

        judgment_by_id = {judgment.rubric_id: judgment for judgment in judgments}
        total_weight = sum(rubric.weight for rubric in rubrics.rubrics)
        weighted_score = sum(
            rubric.weight * judgment_by_id[rubric.rubric_id].score
            for rubric in rubrics.rubrics
        )
        reward = weighted_score / total_weight
        return RewardResult(
            query_id=task.query_id,
            response_id=candidate.response_id,
            reward=reward,
            metadata={"aggregation": "weighted_binary_mean"},
        )

    def build_rubrics(
        self,
        task: Query,
        responses: tuple[Response, ...],
    ) -> RubricSet:
        task_payload = self._task_payload(task)
        prompt = self.rubric_prompt_template.format(
            task_json=json.dumps(task_payload, ensure_ascii=False, indent=2),
            responses_json=json.dumps(
                self._responses_payload(responses),
                ensure_ascii=False,
                indent=2,
            ),
        )
        raw_response = self.rubric_llm(prompt)
        rubrics = self._parse_rubrics(raw_response)

        return RubricSet(
            query_id=task.query_id,
            rubrics=rubrics,
            metadata={"selected_skills": []},
        )

    def score(
        self,
        task: Query,
        candidate: Response,
        rubrics: RubricSet,
    ) -> JudgmentResult:
        task_payload = self._task_payload(task)
        candidate_payload = self._candidate_payload(candidate)
        judgments = []
        for rubric in rubrics.rubrics:
            single_rubric_set = RubricSet(
                query_id=rubrics.query_id,
                rubrics=(rubric,),
            )
            prompt = self.judge_prompt_template.format(
                task_json=json.dumps(task_payload, ensure_ascii=False, indent=2),
                rubrics_json=json.dumps(
                    self._rubrics_payload(single_rubric_set),
                    ensure_ascii=False,
                    indent=2,
                ),
                candidate_json=json.dumps(
                    candidate_payload, ensure_ascii=False, indent=2
                ),
            )
            raw_response = self.judge_llm(prompt)
            judgments.extend(
                self._parse_judgments(raw_response, single_rubric_set)
            )

        return JudgmentResult(
            query_id=task.query_id,
            response_id=candidate.response_id,
            judgments=tuple(judgments),
            metadata={
                "selected_skills": [],
                "grading": "independent_per_rubric",
                "score_scale": "binary",
            },
        )
