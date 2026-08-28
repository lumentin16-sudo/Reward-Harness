"""使用 G-stage Rubric Skill 生成判别性标准的初始 Harness。"""

from __future__ import annotations

import json

from ..reward_system import (
    Query,
    Response,
    RewardSystem,
    RubricSet,
    SkillRegistry,
    WinnerResult,
)
from ..skill_store import load_skill_registry, render_skill_block


HARNESS_NAME = "init_skill"
INITIAL_SKILLS = (
    "response_aware_rubric_workflow",
    "rubric_guided_pairwise_judge",
)


RUBRIC_SKILL_PROMPT = """
You are generating a shared evaluation rubric for a comparative reward model.
Follow the selected Rubric Skills exactly.

Selected Rubric Skills:
{skills}
-----------------END OF THE SKILLS------------------

Public Query:
{query}

Anonymous Responses:
{responses}

Return one JSON object only:
{{
  "rubrics": [
    {{
      "rubric_id": "r1",
      "criterion": "A single atomic and observable requirement"
    }}
  ]
}}

The rubric may use the complete response set to discover discriminative criteria,
but every criterion must remain position-independent and must not reveal or encode
which response should win.
""".strip()


RUBRIC_JUDGE_PROMPT = """
You are a fair and impartial judge. Compare Response A and Response B under the
shared response-aware rubric and select the single response that best satisfies
the user's instruction. You must select exactly one winner.

Follow the selected Judge Skills exactly.

Selected Judge Skills:
{skills}
-----------------END OF THE SKILLS------------------

Evaluate decisive hard failures before softer quality differences. Do not let a
minor criterion override the user's core intent. Cite concrete evidence from both
responses and do not reward verbosity.

--- Analysis ---
<Compare both responses against the relevant rubric criteria>

--- Final Judgment ---
Aggregation Summary: <the decisive differences>
Justification: <why those differences determine the result>
Winner: <Response A / Response B>

Task to Evaluate:
Instruction:
{instruction}

Rubric:
{rubric}

{response_block}
""".strip()


def _response_block(responses: tuple[Response, ...]) -> str:
    """把双回答渲染成稳定的 A/B 比较格式。"""

    if len(responses) != 2:
        raise ValueError("init_skill pairwise judging requires exactly 2 responses")
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
    """从固定 Final Judgment 区域提取 A/B winner。"""

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
        raise ValueError("init_skill judge must declare Winner: Response A/B")
    return WinnerResult(
        query_id=query.query_id,
        winner_response_id=responses[ord(label) - ord("A")].response_id,
        metadata={
            "winner_label": label,
            "comparison": "pairwise_forced_choice",
            "method": "init_skill",
        },
    )


class InitSkillHarness(RewardSystem):
    """用固定 G-stage Skill 生成 Rubrics，再执行 rubric-guided Judge。"""

    rubric_prompt_template = RUBRIC_SKILL_PROMPT
    judge_prompt_template = RUBRIC_JUDGE_PROMPT

    def get_skill_registry(self, task: Query) -> SkillRegistry:
        del task
        return load_skill_registry(INITIAL_SKILLS)

    def build_rubrics(
        self,
        task: Query,
        responses: tuple[Response, ...],
    ) -> RubricSet:
        selected_skills = self.retrieve_skills(task, responses, "G")
        prompt = self.rubric_prompt_template.format(
            skills=render_skill_block(selected_skills),
            query=json.dumps(
                self._task_payload(task), ensure_ascii=False, indent=2
            ),
            responses=json.dumps(
                self._responses_payload(responses), ensure_ascii=False, indent=2
            ),
        )
        rubrics = self._parse_rubrics(self.rubric_llm(prompt))
        return RubricSet(
            query_id=task.query_id,
            rubrics=rubrics,
            metadata={
                "method": "init_skill",
                "online_rubric_generation": True,
                "skills": [skill.name for skill in selected_skills],
            },
        )

    def judge(
        self,
        task: Query,
        responses: tuple[Response, ...],
        rubrics: RubricSet,
    ) -> WinnerResult:
        selected_skills = self.retrieve_skills(task, responses, "J")
        prompt = self.judge_prompt_template.format(
            skills=render_skill_block(selected_skills),
            instruction=task.instruction,
            rubric=json.dumps(
                self._rubrics_payload(rubrics), ensure_ascii=False, indent=2
            ),
            response_block=_response_block(responses),
        )
        return _winner_result(task, responses, self.judge_llm(prompt))


HARNESS_CLASS = InitSkillHarness
