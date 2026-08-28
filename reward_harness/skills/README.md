Filesystem-backed Skill bank for Reward Harnesses.

Each Skill is an independent JSON file under:

```text
reward_harness/skills/*.json
```

Each JSON file must contain:

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

Optimization treats this bank as append-only. Add new Skill JSON files for new
ideas, but do not modify, rename, or delete existing Skill files that may be
referenced by evaluated Harnesses. Retrieval and selection logic belongs in the
candidate Harness.
