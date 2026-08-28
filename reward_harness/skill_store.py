"""Filesystem-backed Skill loading and prompt-injection helpers.

This is a stable, mechanical layer over ``reward_system``: it only loads Skills
from ``reward_harness/skills/*.json`` and renders them into prompt text. Skill
retrieval/selection policy belongs in the candidate Harness (see
``agents/init_skill.py`` for a reference implementation), not here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .reward_system import Skill, SkillRegistry


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
