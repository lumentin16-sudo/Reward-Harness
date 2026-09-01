Filesystem-backed Skill bank for Reward Harnesses.

Each Skill is an independent JSON file under:

```text
reward_harness/skills/*.json
```

Each JSON file must contain the prompt-visible core:

```json
{
  "name": "skill_name",
  "stage": "G",
  "description": "Short catalog text used for retrieval.",
  "content": "Prompt instructions injected after selection."
}
```

Use stage `"G"` for rubric-generation Skills and `"J"` for judge Skills.
Harnesses decide which Skills to use through `get_skill_registry()` and
`retrieve_skills(...)`.

Newer Skill files may also include optimizer-facing applicability fields:

```json
{
  "intended_use": "When this skill should be considered.",
  "failure_modes": ["Reusable failure pattern this skill addresses."],
  "positive_triggers": ["Signals that should retrieve this skill."],
  "negative_triggers": ["Signals that should avoid or narrow this skill."],
  "parent_skill": "older_skill_name",
  "status": "experimental",
  "source_evidence": ["Short trajectory/result evidence for the lesson."]
}
```

These fields are for selection, provenance, and review. Keep the actual
workflow instruction in `content`.

Optimization treats this bank as append-only. Reuse existing Skills when they
already encode the lesson, change retrieval when a Skill was misapplied, and add
new or successor Skill JSON files only for reusable trajectory-backed lessons.
Do not modify, rename, or delete existing Skill files that may be referenced by
evaluated Harnesses. Retrieval and selection logic belongs in the candidate
Harness.

For trigger-based retrieval, candidate Harnesses can use the helpers in
`reward_harness.skill_store`: `build_retrieval_text`, `select_by_triggers`,
`apply_successor_overrides`, and `default_stage_skills`. These helpers provide a
safe default pattern, but the candidate still owns the actual retrieval policy.
