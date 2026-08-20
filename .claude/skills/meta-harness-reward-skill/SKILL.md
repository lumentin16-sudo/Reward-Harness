---
name: meta-harness-reward-skill
description: Run one iteration of reward skill harness evolution. Called by meta_harness.py or interactively via /meta-harness.
---

# Meta-Harness (Reward Skill Harness Evolution)

Run ONE iteration of reward skill harness evolution. Do all work in the main session — do NOT delegate to subagents. Constraints get lost when you delegate, leading to parameter-only changes and skipped prototyping.

**You do NOT run benchmarks.** You analyze results + reward trajectories, prototype changes, and implement new systems. The outer loop (`meta_harness.py`) handles benchmarking separately.

## Critical Constraints

- You MUST implement 3 new reward skill harnesses every iteration.
- Do NOT write "the frontier is optimal", "stop iterating", or abort early.
- ALWAYS complete all steps including prototyping.
- Design exactly 3 candidates per iteration: at least 1 exploitation of current frontier, at least 1 exploration.
- Do NOT modify benchmark code, data files, model clients, evaluator logic, or `reward_system.py`.
- Do NOT modify existing baseline harnesses such as `no_skill.py`, `init_skill.py`, or `no_rubric.py`.
- Each candidate MUST be a new Python file under `reward_harness/agents/`.

## Anti-Parameter-Tuning Rules

The most common failure mode is creating systems that are just parameter variants of existing ones. Check `evolution_summary.jsonl` for what's been tried — parameter sweeps (rubric counts, skill counts, context budgets, score ranges, weighting constants) almost always regress or tie.

Bad candidates only tune surface constants:

- Changing "2 to 6 rubrics" to "3 to 7 rubrics" without a mechanism change.
- Changing score ranges without a new scoring rationale.
- Renaming skills without changing their function.
- Reordering prompt bullets.
- Adding generic phrases such as "be careful" or "think deeply".

Good candidates change a mechanism:

- A new skill bank organization.
- A new rubric-generation workflow.
- A new judge-skill-selection strategy.
- Binary-checkable weighted rubrics.
- Global rubric vs hard-constraint separation.
- Better uncertainty, tie, or hard-failure handling.
- Stronger evidence-first scoring.
- Aggregation that handles hard constraints or severe failures.
- Stage-specific skills for rubric generation vs rubric evaluation.

Each candidate should test one primary hypothesis. Avoid adding unrelated "and also" changes in the same file.

## Anti-Overfitting Rules

- No dataset-specific hints. Do not hardcode knowledge about specific datasets. Reward skill harnesses must be general-purpose.
- Never mention dataset names in system code, prompts, or comments.
- General patterns are OK. Rules like "prioritize severe failures" or "balance rubric coverage" are fine — they apply broadly.

## RewardSystem Interface

Every candidate must define exactly one `RewardSystem` subclass or set `HARNESS_CLASS` to the intended subclass.

```python
class RewardSystem(ABC):
    def get_skill_registry(self, task: Query) -> SkillRegistry: ...

    def build_rubrics(self, task: Query) -> RubricSet: ...

    def score(
        self,
        task: Query,
        candidate: Response,
        rubrics: RubricSet,
    ) -> JudgmentResult: ...

    def aggregate(
        self,
        task: Query,
        candidate: Response,
        rubrics: RubricSet,
        judgment_result: JudgmentResult,
    ) -> RewardResult: ...
```

Extend `RewardSystem` from `..reward_system`
Import `Query`, `Response`, `Rubric`, `RubricSet`, `Skill`, `SkillRegistry`, `RubricJudgment`, `JudgmentResult`, and `RewardResult` from `..reward_system`
Use `self._parse_skill_calls(response, registry)` for skill selection parsing (NOT custom regex)
Use `self._parse_rubrics(response)` for rubric extraction (NOT custom regex)
Use `self._parse_judgments(response, rubrics)` for judgment extraction (NOT custom regex)
Use `self.rubric_llm(prompt)` for rubric generation calls (NOT `self._rubric_llm` directly)
Use `self.judge_llm(prompt)` for response scoring calls (NOT `self._judge_llm` directly)
`build_rubrics` and `score` must work without any prior learning (cold start)

## Workflow

**Do ALL steps yourself in the main session.**

### Step 0: Post-eval reports (write if missing)

Check the reports directory (path in the task prompt's "Run directories" section). For each past iteration that has results in `evolution_summary.jsonl` but NO report, write one. Each report should be <=30 lines covering: what changed, which benchmarks improved/regressed and why, and a takeaway for future iterations.

### Step 1: Analyze

1. Read all state files:

   * `evolution_summary.jsonl` — what's been tried (one JSON per candidate)
   * `frontier_val.json` — current best per benchmark (validation metric)
   * task prompt benchmark command/config for current benchmarks and baselines
   * recent `results/<run_tag>/<benchmark>/<harness>/<model>/trajectories.jsonl` traces if they exist
2. Formulate 3 hypotheses — each must be falsifiable and target a different mechanism.

### Step 2: Prototype — MANDATORY

You MUST prototype your mechanism before writing the final system. Do NOT skip this step. Candidates that skip prototyping tend to have bugs or produce no improvement.

For each candidate:

1. Write a test script in `/tmp/` that exercises the core skill/rubric/judging logic in isolation.
2. Pull real examples from `results/<run_tag>/<benchmark>/<harness>/<model>/trajectories.jsonl` to test against.
3. Try 2-3 variants and compare before picking the best one.
4. Delete scripts when done.

### Step 3: Implement

For each of the 3 candidates:

1. Copy a top-performing base harness to `reward_harness/agents/<name>.py`, then make targeted modifications. This copy-then-edit approach ensures correct imports and proven patterns.
2. Implement the new mechanism according to your hypothesis.
3. Self-critique (mandatory): After implementing, re-read the file and check: does this harness introduce a genuinely NEW mechanism, or is it just a parameter variant? If the logic in `get_skill_registry()`, `build_rubrics()`, `score()`, and `aggregate()` is identical to the base except for constants, REWRITE with a truly novel mechanism.
4. Validate:

```bash
python3 -c "from reward_harness.agents.<name> import *; print('OK')"
```

Do not edit config files just to register candidates. The benchmark auto-discovers files in `reward_harness/agents/`.

### Step 4: Write pending_eval.json

Write to the path specified in the task prompt (NOT hardcoded — it may be in a run-specific subdirectory):

```json
{
  "iteration": "<N>",
  "candidates": [
    {
      "name": "<snake_case_name>",
      "file": "reward_harness/agents/<name>.py",
      "hypothesis": "<falsifiable claim>",
      "axis": "exploitation|exploration",
      "base_harness": "<what it builds on>",
      "components": ["tag1", "tag2", "..."]
    }
  ]
}
```

Output:

```text
CANDIDATES: <name1>, <name2>, <name3>
```

## Current Baselines

- `no_rubric.py`: no rubric generation; direct scalar reward.
- `no_skill.py`: generates shared rubrics and scores responses without workflow skill selection.
- `init_skill.py`: selects workflow skills for rubric generation and pointwise scoring.

Usually build candidates from `init_skill.py` unless the task prompt gives a different base.

## Result Files

Typical run outputs are:

- `results/<run_tag>/<benchmark>/<harness>/<model>/trajectories.jsonl`
- `results/<run_tag>/<benchmark>/<harness>/<model>/summary.json`
- `results/<run_tag>/<benchmark>/<harness>/<model>/config.json`

Trajectory rows are the main source for failure analysis. Summaries are the source for aggregate frontier decisions.

Held-in/train-search trajectories may be used for analysis. Regression held-out trajectories should not be used for detailed optimization unless the outer-loop task prompt explicitly allows it.

## evolution_summary.jsonl Format

One JSON object per line, one line per evaluated candidate:

```json
{"iteration": 1, "system": "example_harness", "avg_val": 45.0, "axis": "exploitation", "hypothesis": "...", "delta": +2.1, "outcome": "45.0% (+2.1)", "components": ["tag1", "tag2", "tag3"]}
```

## Component Analysis

Treat `evolution_summary.jsonl`, `frontier_val.json`, and recent training traces as the only shipped history sources in this trimmed repo.
