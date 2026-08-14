"""使用可自由选择原子 Skill 的第一版 Reward Harness 候选。"""

from __future__ import annotations

import json

from ..reward_system import (
    Candidate,
    RewardResult,
    RewardSystem,
    RewardTask,
    RubricSet,
    Skill,
    SkillInput,
    SkillRegistry,
    SkillResult,
    TraceEvent,
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
selected skill results. Candidate answers are intentionally unavailable.

[Public Task]
{task_json}

[Selected Skill Results]
{skill_results_json}

Requirements:
- Return 2 to 6 non-overlapping rubrics.
- Each rubric must test one observable property of task success.
- weight must be a positive number representing the rubric's relative importance.
- Do not include candidate-specific wording or predicted answers.
- Return JSON only, with this schema:
{{
  "rubrics": [
    {{
      "rubric_id": "short_unique_id",
      "criterion": "what should be evaluated",
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

[Current Candidate]
{candidate_json}

[Available Skills]
{skill_catalog_json}

Select zero or more useful skills. Return JSON only:
{{"skill_calls": ["skill_name"]}}
"""


RUBRIC_EVALUATION_PROMPT = """You are a pointwise reward judge.

Evaluate the single candidate against every supplied rubric. Do not compare it
with another answer and do not infer its position in a preference pair.

[Public Task]
{task_json}

[Shared Rubrics]
{rubrics_json}

[Selected Skill Results]
{skill_results_json}

[Candidate]
{candidate_json}

Requirements:
- Return exactly one judgment for every rubric_id.
- score must be an integer from 0 to 5.
- Skill results are supporting knowledge; scores must still be based on the rubrics.
- evidence must quote or briefly identify observable candidate content.
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


class TaskObjectiveSkill(Skill):
    name = "task_objective"
    description = "Clarify the user's objective and observable task success."

    def invoke(self, context: SkillInput) -> SkillResult:
        return SkillResult(
            skill_name=self.name,
            content=(
                "Focus on the user's actual objective, observable correctness, and "
                f"task completion. Task: {context.task.instruction}"
            ),
        )


class ConstraintAnalysisSkill(Skill):
    name = "constraint_analysis"
    description = "Identify explicit format, content, and prohibited-behavior constraints."

    def invoke(self, context: SkillInput) -> SkillResult:
        return SkillResult(
            skill_name=self.name,
            content=(
                "Check every explicit requirement independently. Treat requested format "
                "and prohibited behavior as constraints, without inventing new ones."
            ),
        )


class PointwiseEvidenceSkill(Skill):
    name = "pointwise_evidence"
    description = "Ground rubric scoring in observable evidence from one candidate."

    def invoke(self, context: SkillInput) -> SkillResult:
        if context.candidate is None:
            content = "Write criteria that can later be checked from one candidate at a time."
        else:
            content = (
                "For each rubric, identify observable supporting or contradicting content "
                "in the current candidate before assigning an integer score."
            )
        return SkillResult(skill_name=self.name, content=content)


class InitSkillHarness(RewardSystem):
    """模型可从 Skill Registry 中独立选择零个或多个 Skill。"""

    rubric_skill_selection_prompt = RUBRIC_SKILL_SELECTION_PROMPT
    rubric_prompt_template = RUBRIC_GENERATION_PROMPT
    judge_skill_selection_prompt = JUDGE_SKILL_SELECTION_PROMPT
    judge_prompt_template = RUBRIC_EVALUATION_PROMPT

    def get_skill_registry(self, task: RewardTask) -> SkillRegistry:
        """候选可以自由增加、删除或重命名这里注册的原子 Skill。"""

        return SkillRegistry(
            (
                TaskObjectiveSkill(),
                ConstraintAnalysisSkill(),
                PointwiseEvidenceSkill(),
            )
        )

    def build_rubrics(self, task: RewardTask) -> RubricSet:
        """由 Rubric Model 选择 Skill，再生成共享 RubricSet。"""

        registry = self.get_skill_registry(task)
        task_payload = self._task_payload(task)
        selection_prompt = self.rubric_skill_selection_prompt.format(
            task_json=json.dumps(task_payload, ensure_ascii=False, indent=2),
            skill_catalog_json=json.dumps(
                registry.catalog, ensure_ascii=False, indent=2
            ),
        )
        raw_selection = self.rubric_llm(selection_prompt)
        skill_calls = self._parse_skill_calls(raw_selection, registry)
        skill_results = registry.invoke(
            skill_calls,
            SkillInput(component="G", task=task),
        )

        prompt = self.rubric_prompt_template.format(
            task_json=json.dumps(task_payload, ensure_ascii=False, indent=2),
            skill_results_json=json.dumps(
                self._skill_results_payload(skill_results),
                ensure_ascii=False,
                indent=2,
            ),
        )
        raw_response = self.rubric_llm(prompt)
        rubrics = self._parse_rubrics(raw_response)

        skill_trace = tuple(
            TraceEvent(
                component="G",
                name="skill_called",
                payload={"skill_name": result.skill_name},
            )
            for result in skill_results
        )
        return RubricSet(
            task_id=task.task_id,
            rubrics=rubrics,
            trace=(
                TraceEvent(
                    component="G",
                    name="skills_selected",
                    payload={"skill_names": list(skill_calls)},
                ),
            )
            + skill_trace
            + (
                TraceEvent(
                    component="G",
                    name="rubrics_generated",
                    payload={"rubric_count": len(rubrics)},
                ),
            ),
            metadata={
                "selected_skills": list(skill_calls),
                "raw_skill_selection": raw_selection,
                "raw_rubric_response": raw_response,
            },
        )

    def score(
        self,
        task: RewardTask,
        candidate: Candidate,
        rubrics: RubricSet,
    ) -> RewardResult:
        """由 Reward Model 选择 Skill，再依据共享 Rubric 为单个候选评分。"""

        registry = self.get_skill_registry(task)
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
        skill_results = registry.invoke(
            skill_calls,
            SkillInput(
                component="J",
                task=task,
                rubrics=rubrics,
                candidate=candidate,
            ),
        )

        prompt = self.judge_prompt_template.format(
            task_json=json.dumps(task_payload, ensure_ascii=False, indent=2),
            rubrics_json=json.dumps(rubrics_payload, ensure_ascii=False, indent=2),
            skill_results_json=json.dumps(
                self._skill_results_payload(skill_results),
                ensure_ascii=False,
                indent=2,
            ),
            candidate_json=json.dumps(candidate_payload, ensure_ascii=False, indent=2),
        )
        raw_response = self.judge_llm(prompt)
        judgments = self._parse_judgments(raw_response, rubrics)
        reward = self.aggregate(judgments, rubrics)

        skill_trace = tuple(
            TraceEvent(
                component="J",
                name="skill_called",
                payload={"skill_name": result.skill_name},
            )
            for result in skill_results
        )
        return RewardResult(
            task_id=task.task_id,
            candidate_id=candidate.candidate_id,
            reward=reward,
            judgments=judgments,
            trace=(
                TraceEvent(
                    component="J",
                    name="skills_selected",
                    payload={"skill_names": list(skill_calls)},
                ),
            )
            + skill_trace
            + (
                TraceEvent(
                    component="J",
                    name="candidate_scored",
                    payload={"judgment_count": len(judgments)},
                ),
                TraceEvent(
                    component="A",
                    name="weighted_mean_aggregated",
                    payload={"reward": reward},
                ),
            ),
            metadata={
                "aggregation": "weighted_mean",
                "selected_skills": list(skill_calls),
                "raw_skill_selection": raw_selection,
                "raw_judge_response": raw_response,
            },
        )
