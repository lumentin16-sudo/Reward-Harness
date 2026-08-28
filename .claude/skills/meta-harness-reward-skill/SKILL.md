---
name: meta-harness-reward-skill
description: Run one iteration of reward skill harness evolution. Called by meta_harness.py or interactively via /meta-harness.
---

# Meta-Harness (Reward Skill Harness Evolution)

Run ONE iteration of reward skill harness evolution. Do all work in the main session — do NOT delegate to subagents. Constraints get lost when you delegate, leading to parameter-only changes and skipped prototyping.

**You do NOT run benchmarks.** You analyze results + reward trajectories, prototype changes, and implement new systems. The outer loop (`meta_harness.py`) handles benchmarking separately.

## Context

You are evolving comparative Reward Harnesses under `reward_harness/agents/`.
The default Reward Harness workflow is: first generate shared, response-aware
rubrics for the Query and the two anonymous Responses, then judge the Responses
based on the generated rubrics and return one `WinnerResult`. Evaluator-only
gold labels must never enter model prompts.

The searchable pipeline has two stages:

- G: Generate rubrics first. In `build_rubrics(query, responses)`, inspect the
  Query and both anonymous Responses, optionally select G-stage Skills, call
  `rubric_llm`, and return a shared, response-aware `RubricSet` that captures
  criteria useful for distinguishing the two Responses.
- J: Judge based on the generated rubrics. In
  `judge(query, responses, rubrics)`, inspect the Query, both anonymous
  Responses, and the generated `RubricSet`, optionally select J-stage Skills,
  call `judge_llm`, compare rubric satisfaction and decisive evidence, and
  return one winner.

Available baseline/reference Harnesses:

- `no_rubric.py`: direct pairwise Judge without Rubrics or Skills.
- `no_skill.py`: online Rubric generation without Skills, then rubric-based pairwise judging.
- `init_skill_no_rubric.py`: J-stage Skill without Rubric generation.
- `init_skill.py`: G-stage Skill for response-aware Rubric generation, then rubric-based pairwise judging.

The active baselines and benchmarks for the current run are defined by the
task prompt and `config.yaml`; do not assume every reference Harness is active
in the current search.
`no_skill.py` is a comparison baseline, not the preferred base for skill
evolution. If `no_skill.py` is the global frontier, build candidates from the
best skill-using Harness reported in the task prompt or evolution history.

The default search should build from the current frontier or another
top-performing rubric-based Harness unless the task prompt explicitly asks for
no-rubric exploration.

You may add new Skills, redesign Skill retrieval/selection, prompts,
Rubric/Judge workflows, and in-Harness control flow inside new candidate
Harnesses. Do not modify benchmark data, model clients, fixed payload and
parsing helpers, validators, `reward_system.py`, config files, existing
baseline Harnesses, or existing Skill files.

Read the baseline Harnesses, `frontier_val.json`, `evolution_summary.jsonl`,
and recent held-in trajectories before proposing candidates. `frontier_val.json`
tracks the best Harness per reported domain plus the best aggregate `avg_val`
for the current held-in/search benchmark configuration.

## Critical Constraints

- You MUST implement 3 new reward skill harnesses every iteration.
- Do NOT write "the frontier is optimal", "stop iterating", or abort early.
- ALWAYS complete all steps including prototyping.
- Design exactly 3 candidates per iteration: at least 1 exploitation of current frontier, at least 1 exploration.
- Every candidate MUST use the Skill bank. Do not submit candidates whose `get_skill_registry()` returns an empty `SkillRegistry`.
- Every candidate MUST add at least one new G-stage or J-stage Skill JSON file, then use retrieval to decide when to inject it.

## Candidate Design

Each candidate has one Python file at `reward_harness/agents/<name>.py`
containing the full Reward Harness control flow. Skills are independent JSON
files in the shared bank under `reward_harness/skills/*.json`; the candidate
decides which Skills to retrieve before G and J. Define exactly one
`RewardSystem` subclass or set `HARNESS_CLASS` to the intended subclass.
The Skill bank is append-only for optimization: preserve existing Skill files
so evaluated Harnesses remain reproducible. Add new Skills when needed, and
put selection responsibility in `retrieve_skills(...)`.

What you can and cannot modify:

- CAN: edit your new `reward_harness/agents/<name>.py` file freely.
- CAN: copy a top-performing baseline or frontier Harness into a new file, then make targeted changes.
- CAN: add new G-stage and J-stage Skill JSON files under `reward_harness/skills/`.
- CAN: redesign Skill retrieval/selection inside the new Harness file.
- CAN: redesign rubric-generation prompts, judge prompts, helper functions, and in-Harness control flow inside the new file.
- CANNOT: modify, overwrite, rename, or delete existing Skill JSON files.
- CANNOT: hide new reusable judging or rubric-generation experience only as inline prompt text. If the idea is a reusable workflow instruction, add it as a new Skill and retrieve it.
- CANNOT: modify benchmark code, data files, model clients, evaluator logic, fixed payload/parsing helpers, validators, `reward_system.py`, config files, or existing baseline Harnesses.

Design principles:

- Mechanism-first. Identify a specific failure mode from trajectories, then design a change that targets it. Never add changes speculatively.
- One mechanism per candidate. Each candidate tests exactly one hypothesis. If you are tempted to add "and also...", that is a second candidate.
- Evidence-driven hypotheses. Each hypothesis must state: observed failure pattern, change, expected mechanism, and risk.
- Error attribution before editing. For each candidate, first decide whether the target failures come from Skill retrieval, rubric generation, rubric quality, rubric-guided judging, aggregation/comparison logic, output parsing, or prompt/control-flow issues.
- Skill-first. Treat Skills as an append-only bank of reusable workflow instructions. New trajectory-derived experience should enter the Skill bank, then the candidate should retrieve/select relevant G-stage Skills before rubric generation and relevant J-stage Skills before judging.
- Prefer minimal targeted changes to the current frontier or another top-performing base Harness. Do not add multi-stage gates, appeals, or verification passes unless the observed failures repeatedly require that extra stage.
- Do not exploit quirks of the held-in set, benchmark ordering, response labels, parser behavior, prompt formatting, or known answer patterns.
- Do not hardcode dataset-specific hints. Never mention dataset names in system code, prompts, or comments. General patterns such as "prioritize severe failures" or "balance rubric coverage" are fine.
- Avoid parameter-only variants. Changing rubric counts, skill counts, context budgets, score ranges, weighting constants, winner parsing wording, prompt bullet order, or generic caution phrases is not a new mechanism by itself.

Good candidates change a mechanism, such as:

- A new trajectory-derived G-stage or J-stage Skill plus a retrieval rule that decides when to use it.
- A new Skill retrieval or selection mechanism over existing and newly added Skills.
- A new rubric-generation workflow.
- A new judge-skill-selection strategy.
- Response-set-aware rubric discovery that improves head-to-head discrimination.
- Global rubric vs hard-constraint separation.
- Better uncertainty, tie, or hard-failure handling.
- Stronger evidence-first scoring.
- Winner selection that handles hard constraints, severe failures, or near ties.
- Stage-specific skills for rubric generation vs pairwise judging.

## RewardSystem Interface

Every candidate must define exactly one `RewardSystem` subclass or set `HARNESS_CLASS` to the intended subclass.

```python
class RewardSystem(ABC):
    def __init__(
        self,
        rubric_llm: LLMCallable,
        judge_llm: LLMCallable,
    ) -> None: ...

    @property
    def rubric_llm(self) -> LLMCallable: ...

    @property
    def judge_llm(self) -> LLMCallable: ...

    def get_skill_registry(self, task: Query) -> SkillRegistry: ...

    def retrieve_skills(
        self,
        task: Query,
        responses: tuple[Response, ...],
        stage: SkillStage,
    ) -> tuple[Skill, ...]: ...

    def build_rubrics(
        self,
        task: Query,
        responses: tuple[Response, ...],
    ) -> RubricSet: ...

    def judge(
        self,
        task: Query,
        responses: tuple[Response, ...],
        rubrics: RubricSet,
    ) -> WinnerResult: ...

    def _task_payload(task: Query) -> dict[str, JSONValue]: ...
    def _candidate_payload(candidate: Response) -> dict[str, JSONValue]: ...
    def _responses_payload(responses: tuple[Response, ...]) -> list[dict[str, JSONValue]]: ...
    def _judge_responses_payload(responses: tuple[Response, ...]) -> list[dict[str, JSONValue]]: ...
    def _rubrics_payload(rubrics: RubricSet) -> list[dict[str, JSONValue]]: ...
    def _skills_payload(skills: tuple[Skill, ...]) -> list[dict[str, JSONValue]]: ...
    def _parse_skill_calls(raw_response: str, registry: SkillRegistry) -> tuple[str, ...]: ...
    def _parse_rubrics(raw_response: str) -> tuple[Rubric, ...]: ...
    def _parse_judgments(raw_response: str, rubrics: RubricSet) -> tuple[RubricJudgment, ...]: ...
    def _validate_rubric_set(self, task: Query, rubrics: RubricSet) -> None: ...
    def _validate_winner_result(task: Query, responses: tuple[Response, ...], result: WinnerResult) -> None: ...
```

Extend `RewardSystem` from `..reward_system`
Import `LLMCallable`, `Query`, `Response`, `Rubric`, `RubricJudgment`, `RubricSet`, `Skill`, `SkillStage`, `SkillRegistry`, and `WinnerResult` from `..reward_system`
`Skill` requires `name`, `stage`, `description`, and `content`; stage must be `"G"` or `"J"`
Use `load_skill_registry(...)` from `..skill_store` to load independent Skill files from `reward_harness/skills/*.json`
Use `retrieve_skills(task, responses, "G")` before rubric generation and `retrieve_skills(task, responses, "J")` before judging when Skills are used
Use `registry.for_stage("G")` or `registry.for_stage("J")` before skill selection when a stage-specific skill pool is needed
Use `self._task_payload(task)` for public task payloads
Use `self._candidate_payload(candidate)` for single-response judge payloads
Use `self._responses_payload(responses)` for anonymized response-set payloads in rubric generation (NOT custom payloads with IDs or labels)
Use `self._judge_responses_payload(responses)` when a structured comparative Judge payload is needed
Use `self._rubrics_payload(rubrics)` for judge-visible rubric payloads
Use `self._skills_payload(selected_skills)` for selected skill payloads
Use `self._parse_skill_calls(response, registry)` for skill selection parsing (NOT custom regex)
Use `self._parse_rubrics(response)` for rubric extraction (NOT custom regex)
Use `self._parse_judgments(response, rubrics)` for judgment extraction (NOT custom regex)
Do NOT override `_task_payload`, `_candidate_payload`, `_responses_payload`, `_judge_responses_payload`, `_rubrics_payload`, `_skills_payload`, `_parse_skill_calls`, `_parse_rubrics`, `_parse_judgments`, `_validate_rubric_set`, or `_validate_winner_result`
The benchmark/evaluator calls `_validate_rubric_set` and `_validate_winner_result`; candidates should satisfy these checks rather than bypass them
Use `self.rubric_llm(prompt)` for rubric generation calls (NOT `self._rubric_llm` directly)
Use `self.judge_llm(prompt)` for comparative judging calls (NOT `self._judge_llm` directly)
Use `select_stage_skills(...)` and `render_skill_block(...)` from `..skill_store` when a candidate needs LLM-based Skill selection and prompt injection
`build_rubrics` and `judge` must work without any prior learning (cold start)

## Workflow

**Do ALL steps yourself in the main session.**

### Step 0: Post-eval reports (write if missing)

Check the reports directory (path in the task prompt's "Optimization state" section). For each past iteration that has results in `evolution_summary.jsonl` but NO report, write one. Each report should be <=30 lines covering: what changed, which benchmarks improved/regressed and why, and a takeaway for future iterations.

### Step 1: Analyze

1. Read all state files:

   * `evolution_summary.jsonl` — what's been tried (one JSON per candidate)
   * `frontier_val.json` — current best on the held-in search metric
   * task prompt benchmark command/config for current benchmarks and baselines
   * recent `results/<run_tag>/<benchmark>/<harness>/<model>/trajectories.jsonl` traces if they exist
2. Identify the current frontier/base Harness from `frontier_val.json` and inspect its source file before editing.
3. Inspect incorrect or fragile trajectories first, then compare against nearby systems when useful. For each failure pattern, attribute the likely failure source: Skill retrieval, rubric generation, rubric quality, rubric-guided judging, aggregation/comparison logic, output parsing, or prompt/control-flow.
4. Formulate 3 hypotheses — each must be falsifiable, target a different mechanism, and include: observed failure pattern, attributed failure source, change, expected mechanism, and risk.

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
2. Add at least one new independent Skill JSON file under `reward_harness/skills/`. Do not edit or delete existing Skill files.
3. Implement the new mechanism according to your hypothesis.
4. Self-critique (mandatory): After implementing, re-read the file and check: does this harness introduce a genuinely NEW mechanism, or is it just a parameter variant? If the logic in `get_skill_registry()`, `build_rubrics()`, and `judge()` is identical to the base except for constants, REWRITE with a truly novel mechanism.
5. Import/interface check:

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
      "hypothesis": "<observed failure pattern + attributed failure source + change + expected mechanism + risk>",
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
