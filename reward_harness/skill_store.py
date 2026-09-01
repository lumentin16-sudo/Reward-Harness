"""Filesystem-backed Skill loading and prompt-injection helpers.

This is a stable, mechanical layer over ``reward_system``: it only loads Skills
from ``reward_harness/skills/*.json`` and renders them into prompt text. Skill
retrieval/selection policy belongs in the candidate Harness (see
``agents/init_skill.py`` for a reference implementation), not here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping

from .reward_system import JSONValue, Query, Response, Skill, SkillRegistry, SkillStage


SKILL_ROOT = Path(__file__).resolve().parent / "skills"


def _string_tuple(payload: dict, key: str) -> tuple[str, ...]:
    """Read an optional string-list field from a Skill JSON file."""

    value = payload.get(key, ())
    if value in (None, "", ()):
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return tuple(value)
    raise ValueError(f"Skill field {key!r} must be a string or string list")


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
                intended_use=str(payload.get("intended_use", "")),
                failure_modes=_string_tuple(payload, "failure_modes"),
                positive_triggers=_string_tuple(payload, "positive_triggers"),
                negative_triggers=_string_tuple(payload, "negative_triggers"),
                parent_skill=payload.get("parent_skill"),
                status=str(payload.get("status", "seed")),  # type: ignore[arg-type]
                source_evidence=_string_tuple(payload, "source_evidence"),
            )
        )
    if wanted is not None:
        missing = wanted - loaded
        if missing:
            raise FileNotFoundError(f"missing Skill file(s): {sorted(missing)}")
    return SkillRegistry(tuple(skills))


def build_retrieval_text(
    task: Query,
    responses: tuple[Response, ...] = (),
) -> str:
    """Build the public text surface a candidate may use for trigger matching."""

    parts = [
        task.instruction,
        task.context or "",
        task.domain or "",
        _metadata_text(task.metadata),
    ]
    parts.extend(response.content for response in responses)
    return "\n".join(part for part in parts if part).lower()


def select_by_triggers(
    registry: SkillRegistry,
    stage: SkillStage,
    retrieval_text: str,
    *,
    fallback_names: tuple[str, ...] = (),
    max_skills: int | None = None,
) -> tuple[Skill, ...]:
    """Select stage Skills with positive trigger hits and no negative trigger hits.

    This is a conservative helper, not the global retrieval policy. Candidate
    Harnesses can use it as a building block and still own the final strategy.
    """

    stage_registry = registry.for_stage(stage)
    selected = tuple(
        skill
        for skill in stage_registry.skills
        if skill.status != "deprecated"
        and _has_trigger(skill.positive_triggers, retrieval_text)
        and not _has_trigger(skill.negative_triggers, retrieval_text)
    )
    selected = apply_successor_overrides(selected)
    if not selected:
        selected = _fallback_skills(stage_registry, fallback_names)
    if max_skills is not None:
        return selected[:max_skills]
    return selected


def apply_successor_overrides(skills: tuple[Skill, ...]) -> tuple[Skill, ...]:
    """Drop a selected parent Skill when a selected successor names it."""

    replaced = {
        skill.parent_skill
        for skill in skills
        if skill.parent_skill and skill.status != "deprecated"
    }
    return tuple(skill for skill in skills if skill.name not in replaced)


def default_stage_skills(
    registry: SkillRegistry,
    stage: SkillStage,
    *,
    preferred_names: tuple[str, ...] = (),
    max_skills: int | None = None,
) -> tuple[Skill, ...]:
    """Return a deterministic non-deprecated fallback set for a stage."""

    stage_registry = registry.for_stage(stage)
    selected = _fallback_skills(stage_registry, preferred_names)
    if max_skills is not None:
        return selected[:max_skills]
    return selected


def render_skill_block(skills: tuple[Skill, ...]) -> str:
    """Render selected Skills for prompt injection."""

    return "\n\n".join(
        f"## Skill: {skill.name}\n{skill.content.strip()}" for skill in skills
    )


def _metadata_text(metadata: Mapping[str, JSONValue]) -> str:
    if not metadata:
        return ""
    return json.dumps(metadata, ensure_ascii=False, sort_keys=True)


def _has_trigger(triggers: tuple[str, ...], text: str) -> bool:
    return any(trigger.lower() in text for trigger in triggers if trigger.strip())


def _fallback_skills(
    registry: SkillRegistry,
    preferred_names: tuple[str, ...],
) -> tuple[Skill, ...]:
    usable = tuple(skill for skill in registry.skills if skill.status != "deprecated")
    if preferred_names:
        preferred = tuple(skill for skill in usable if skill.name in preferred_names)
        if preferred:
            return preferred
    seed = tuple(skill for skill in usable if skill.status == "seed")
    return seed or usable[:1]
