"""使用可自由选择 workflow Skill 的第一版 Reward Harness 候选。"""

from __future__ import annotations

import json

from ..reward_system import (
    Response,
    JudgmentResult,
    RewardResult,
    RewardSystem,
    Query,
    RubricSet,
    Skill,
    SkillRegistry,
)


RUBRIC_SKILL_SELECTION_PROMPT = """Select skills for rubric generation.

[Public Task]
{task_json}

[Available Skills]
{skill_catalog_json}

Select zero or more useful skills. Return JSON only:
{{"skill_calls": ["skill_name"]}}
"""


RUBRIC_GENERATION_PROMPT = """You are a rubric-generation model.

Generate task-specific evaluation rubrics using only the public task and the
selected workflow skills. Response answers are intentionally unavailable.

[Public Task]
{task_json}

[Selected Workflow Skills]
{skills_json}

Requirements:
- Return 2 to 6 non-overlapping rubrics.
- Each rubric must be one atomic, binary-verifiable requirement observable from
  a single response and answerable only as PASS or FAIL.
- Make the pass condition concrete and self-contained. State the required fact,
  reasoning step, constraint, or behavior; avoid vague criteria such as
  "correct", "clear", or "high quality" without an explicit test.
- PASS requires full satisfaction of the condition; partial satisfaction is FAIL.
- weight must be a positive number representing the rubric's relative importance.
- Do not include candidate-specific wording or predicted answers.
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


JUDGE_SKILL_SELECTION_PROMPT = """Select skills for pointwise rubric scoring.

[Public Task]
{task_json}

[Shared Rubrics]
{rubrics_json}

[Current Response]
{candidate_json}

[Available Skills]
{skill_catalog_json}

Select zero or more useful skills. Return JSON only:
{{"skill_calls": ["skill_name"]}}
"""


RUBRIC_EVALUATION_PROMPT = """You are a pointwise reward judge.

Evaluate the single response against the one supplied rubric. Do not compare it
with another answer and do not infer its position in a preference pair.

[Public Task]
{task_json}

[Single Rubric]
{rubrics_json}

[Selected Workflow Skills]
{skills_json}

[Response]
{candidate_json}

Requirements:
- Return exactly one judgment for the supplied rubric_id.
- score must be the integer 1 when the response fully satisfies the binary pass
  condition, otherwise 0. Do not award partial credit.
- Workflow skills are guidance; scores must still be based on the rubrics.
- evidence must quote or briefly identify observable response content.
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


TASK_OBJECTIVE_SKILL = Skill(
    name="task_objective",
    stage="G",
    description="Clarify the user's objective and observable task success.",
    content=(
        "Focus on the user's actual objective, observable correctness, and task "
        "completion rather than surface style or verbosity."
    ),
)

CONSTRAINT_ANALYSIS_SKILL = Skill(
    name="constraint_analysis",
    stage="G",
    description="Identify explicit format, content, and prohibited-behavior constraints.",
    content=(
        "Check every explicit requirement independently. Treat requested format and "
        "prohibited behavior as constraints, without inventing new ones."
    ),
)

POINTWISE_EVIDENCE_SKILL = Skill(
    name="pointwise_evidence",
    stage="J",
    description="Ground rubric scoring in observable evidence from one candidate.",
    content=(
        "Use criteria that can be checked from one candidate at a time. For each "
        "criterion, identify observable supporting or contradicting evidence before "
        "deciding PASS or FAIL."
    ),
)


class InitSkillHarness(RewardSystem):
    """模型可从 Skill Registry 中独立选择零个或多个 Skill。"""

    rubric_skill_selection_prompt = RUBRIC_SKILL_SELECTION_PROMPT
    rubric_prompt_template = RUBRIC_GENERATION_PROMPT
    judge_skill_selection_prompt = JUDGE_SKILL_SELECTION_PROMPT
    judge_prompt_template = RUBRIC_EVALUATION_PROMPT

    def get_skill_registry(self, task: Query) -> SkillRegistry:
        """候选可以自由增加、删除或重命名这里注册的 workflow Skill。"""

        return SkillRegistry(
            (
                TASK_OBJECTIVE_SKILL,
                CONSTRAINT_ANALYSIS_SKILL,
                POINTWISE_EVIDENCE_SKILL,
            )
        )

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
                raise ValueError("init_skill expects binary integer scores 0 or 1")

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

    def build_rubrics(self, task: Query) -> RubricSet:
        """由 Rubric Model 选择 Skill，再生成共享 RubricSet。"""

        registry = self.get_skill_registry(task).for_stage("G")
        task_payload = self._task_payload(task)
        selection_prompt = self.rubric_skill_selection_prompt.format(
            task_json=json.dumps(task_payload, ensure_ascii=False, indent=2),
            skill_catalog_json=json.dumps(
                registry.catalog, ensure_ascii=False, indent=2
            ),
        )
        raw_selection = self.rubric_llm(selection_prompt)
        skill_calls = self._parse_skill_calls(raw_selection, registry)
        selected_skills = registry.select(skill_calls)

        prompt = self.rubric_prompt_template.format(
            task_json=json.dumps(task_payload, ensure_ascii=False, indent=2),
            skills_json=json.dumps(
                self._skills_payload(selected_skills),
                ensure_ascii=False,
                indent=2,
            ),
        )
        raw_response = self.rubric_llm(prompt)
        rubrics = self._parse_rubrics(raw_response)

        return RubricSet(
            query_id=task.query_id,
            rubrics=rubrics,
            metadata={
                "selected_skills": list(skill_calls),
                "skills": self._skills_payload(selected_skills),
            },
        )

    def score(
        self,
        task: Query,
        candidate: Response,
        rubrics: RubricSet,
    ) -> JudgmentResult:
        """由 Reward Model 选择 Skill，再依据共享 Rubric 为单个候选评分。"""

        registry = self.get_skill_registry(task).for_stage("J")
        task_payload = self._task_payload(task)
        rubrics_payload = self._rubrics_payload(rubrics)
        candidate_payload = self._candidate_payload(candidate)
        selection_prompt = self.judge_skill_selection_prompt.format(
            task_json=json.dumps(task_payload, ensure_ascii=False, indent=2),
            rubrics_json=json.dumps(rubrics_payload, ensure_ascii=False, indent=2),
            candidate_json=json.dumps(candidate_payload, ensure_ascii=False, indent=2),
            skill_catalog_json=json.dumps(
                registry.catalog, ensure_ascii=False, indent=2
            ),
        )
        raw_selection = self.judge_llm(selection_prompt)
        skill_calls = self._parse_skill_calls(raw_selection, registry)
        selected_skills = registry.select(skill_calls)

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
                skills_json=json.dumps(
                    self._skills_payload(selected_skills),
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
                "selected_skills": list(skill_calls),
                "skills": self._skills_payload(selected_skills),
                "grading": "independent_per_rubric",
                "score_scale": "binary",
            },
        )
