"""只在 Judge 阶段注入 Skill、但不生成 Rubric 的初始 Harness。"""

from __future__ import annotations

from ..reward_system import (
    Query,
    Response,
    RewardSystem,
    RubricSet,
    SkillRegistry,
    WinnerResult,
)
from ..skill_store import load_skill_registry, render_skill_block


HARNESS_NAME = "init_skill_no_rubric"
INITIAL_SKILLS: tuple[str, ...] = ()


SKILL_PAIRWISE_JUDGE_PROMPT = """
You are a fair and impartial judge. Your task is to evaluate 'Response A' and 'Response B' based on a given instruction to select the single best response.
**NOTE**: You must select a winner. Never respond with "None" or "Neither" as the winner.

You may use the selected Judge Skills derived from other successful examples, as references if helpful.

Selected Judge Skills:
{skills}
-----------------END OF THE SKILLS------------------

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


class InitSkillNoRubricHarness(RewardSystem):
    """把可由 Harness Optimization 编辑的离线 Skill 注入官方 Judge Prompt。"""

    judge_prompt_template = SKILL_PAIRWISE_JUDGE_PROMPT

    def get_skill_registry(self, task: Query) -> SkillRegistry:
        del task
        return load_skill_registry(INITIAL_SKILLS)

    def build_rubrics(
        self, task: Query, responses: tuple[Response, ...]
    ) -> RubricSet:
        return RubricSet(
            query_id=task.query_id,
            rubrics=(),
            metadata={
                "method": "init_skill_no_rubric",
                "online_rubric_generation": False,
                "skills": [
                    skill.name
                    for skill in self.retrieve_skills(task, responses, "J")
                ],
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
            instruction=task.instruction,
            skills=render_skill_block(selected_skills),
            response_block=_response_block(responses),
        )
        return _winner_result(
            task,
            responses,
            self.judge_llm(prompt),
        )


HARNESS_CLASS = InitSkillNoRubricHarness
