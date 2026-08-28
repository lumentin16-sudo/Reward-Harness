"""Filesystem-backed Skill loading and lightweight selection helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable

from .reward_system import JSONValue, Skill, SkillRegistry, SkillStage


SKILL_ROOT = Path(__file__).resolve().parent / "skills"


def load_skill_registry(
    names: Iterable[str] | None = None,
    *,
    root: Path = SKILL_ROOT,
) -> SkillRegistry:
    """Load independent JSON Skill files from ``reward_harness/skills/*.json``."""

    if not root.is_dir():
        return SkillRegistry()

    wanted = set(names) if names is not None else None
    skills: list[Skill] = []
    loaded: set[str] = set()
    for path in sorted(root.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        name = str(payload.get("name", ""))
        if wanted is not None and name not in wanted:
            continue
        loaded.add(name)
        skills.append(
            Skill(
                name=name,
                stage=str(payload.get("stage", "")),  # type: ignore[arg-type]
                description=str(payload.get("description", "")),
                content=str(payload.get("content", "")),
            )
        )
    if wanted is not None:
        missing = wanted - loaded
        if missing:
            raise FileNotFoundError(f"missing Skill file(s): {sorted(missing)}")
    return SkillRegistry(tuple(skills))


def render_skill_block(skills: tuple[Skill, ...]) -> str:
    """Render selected Skills for prompt injection."""

    return "\n\n".join(
        f"## Skill: {skill.name}\n{skill.content.strip()}" for skill in skills
    )


def select_stage_skills(
    *,
    registry: SkillRegistry,
    stage: SkillStage,
    query_payload: dict[str, JSONValue],
    responses_payload: list[dict[str, JSONValue]],
    llm: Callable[[str], str],
    parse_skill_calls: Callable[[str, SkillRegistry], tuple[str, ...]],
    max_skills: int = 2,
) -> tuple[Skill, ...]:
    """Select relevant Skills for one G/J stage.

    When the stage has at most ``max_skills`` Skills, return all of them without
    an extra model call. Larger banks use the stage LLM to retrieve by name from
    the catalog, while parser failures fall back to the first available Skills so
    the harness remains usable from cold start.
    """

    stage_registry = registry.for_stage(stage)
    if not stage_registry.skills:
        return ()
    if len(stage_registry.skills) <= max_skills:
        return stage_registry.skills

    prompt = f"""Select up to {max_skills} workflow Skills for stage {stage}.
Return exactly one JSON object: {{"skill_calls": ["skill_name"]}}

Skill catalog:
{json.dumps(stage_registry.catalog, ensure_ascii=False, indent=2)}

Public query:
{json.dumps(query_payload, ensure_ascii=False, indent=2)}

Anonymous responses:
{json.dumps(responses_payload, ensure_ascii=False, indent=2)}
""".strip()
    try:
        names = parse_skill_calls(llm(prompt), stage_registry)
    except ValueError:
        names = ()
    selected = [skill for skill in stage_registry.skills if skill.name in names]
    return tuple(selected[:max_skills] or stage_registry.skills[:max_skills])
