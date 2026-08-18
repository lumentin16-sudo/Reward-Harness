"""不注入 Skill 的独立 Reward Harness baseline。"""

from __future__ import annotations

import json

from ..reward_system import (
    Candidate,
    RewardResult,
    RewardSystem,
    RewardTask,
    RubricSet,
    SkillRegistry,
    TraceEvent,
)


RUBRIC_GENERATION_PROMPT = """You are an expert evaluation-rubric designer.

Generate task-specific evaluation rubrics using only the public task. Candidate
answers and preference labels are intentionally unavailable; do not infer them.

[Public Task]
{task_json}

Requirements:
- First infer the task's core intent, explicit constraints, and necessary
  implicit quality standards.
- Return 2 to 6 task-specific, non-overlapping rubrics that jointly cover the
  critical requirements. Avoid generic criteria unless the task requires them.
- Each rubric must test one distinct property observable from a response.
- The criterion must include calibrated anchors for scores 0, 1, 3, and 5;
  scores 2 and 4 represent intermediate quality. Score 3 is an acceptable but
  imperfect response, while score 5 is genuinely complete and excellent.
- Give explicit/core requirements more weight than optional qualitative nuance.
- weight must be between 0.5 and 2.0 and represent relative importance.
- Do not invent formatting constraints or candidate-specific requirements.
- Return JSON only, with this schema:
{{
  "rubrics": [
    {{
      "rubric_id": "short_unique_id",
      "criterion": "distinct observable criterion; Score 0: ...; Score 1: ...; Score 3: ...; Score 5: ...",
      "weight": 1.0
    }}
  ]
}}
"""


RUBRIC_EVALUATION_PROMPT = """You are a fair and impartial pointwise reward judge.

Evaluate the single candidate against every supplied rubric. Do not compare it
with another answer, infer its pair position, or assume access to preference
labels.

[Public Task]
{task_json}

[Shared Rubrics]
{rubrics_json}

[Candidate]
{candidate_json}

Requirements:
- Return exactly one judgment for every rubric_id.
- For each rubric, first identify concrete strengths, errors, or omissions in
  the candidate, then select the calibrated score defined by that rubric.
- score must be an integer from 0 to 5.
- evidence must quote or briefly identify observable candidate content.
- Prioritize correctness, core intent, and explicit instruction following.
  Do not reward verbosity, polished style, or repeated claims by themselves.
- Distinguish major correctness failures from minor presentation defects and
  keep scores calibrated: 3 is acceptable, 5 is exceptional, and 0 is unusable.
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

    def get_skill_registry(self, task: RewardTask) -> SkillRegistry:
        return SkillRegistry()

    def build_rubrics(self, task: RewardTask) -> RubricSet:
        task_payload = self._task_payload(task)
        prompt = self.rubric_prompt_template.format(
            task_json=json.dumps(task_payload, ensure_ascii=False, indent=2)
        )
        raw_response = self.rubric_llm(prompt)
        rubrics = self._parse_rubrics(raw_response)

        return RubricSet(
            task_id=task.task_id,
            rubrics=rubrics,
            trace=(
                TraceEvent(
                    component="G",
                    name="rubrics_generated",
                    payload={"rubric_count": len(rubrics)},
                ),
            ),
            metadata={
                "selected_skills": [],
                "raw_rubric_response": raw_response,
            },
        )

    def score(
        self,
        task: RewardTask,
        candidate: Candidate,
        rubrics: RubricSet,
    ) -> RewardResult:
        task_payload = self._task_payload(task)
        rubrics_payload = self._rubrics_payload(rubrics)
        candidate_payload = self._candidate_payload(candidate)
        prompt = self.judge_prompt_template.format(
            task_json=json.dumps(task_payload, ensure_ascii=False, indent=2),
            rubrics_json=json.dumps(rubrics_payload, ensure_ascii=False, indent=2),
            candidate_json=json.dumps(candidate_payload, ensure_ascii=False, indent=2),
        )
        raw_response = self.judge_llm(prompt)
        judgments = self._parse_judgments(raw_response, rubrics)
        reward = self.aggregate(judgments, rubrics)

        return RewardResult(
            task_id=task.task_id,
            candidate_id=candidate.candidate_id,
            reward=reward,
            judgments=judgments,
            trace=(
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
                "selected_skills": [],
                "raw_judge_response": raw_response,
            },
        )
