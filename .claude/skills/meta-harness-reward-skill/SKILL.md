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

You may reuse existing Skills, create successor Skills, add genuinely new
Skills, redesign Skill retrieval/selection, prompts, Rubric/Judge workflows,
and in-Harness control flow inside new candidate Harnesses. Do not modify
benchmark data, model clients, fixed payload and parsing helpers, validators,
`reward_system.py`, `skill_store.py`, config files, existing baseline
Harnesses, or existing Skill files.

Read the baseline Harnesses, `frontier_val.json`, `evolution_summary.jsonl`,
and recent held-in trajectories before proposing candidates. `frontier_val.json`
tracks the best Harness per reported domain plus the best aggregate `avg_val`
for the current held-in/search benchmark configuration.

Skill evolution is not arbitrary prompt tweaking. Before proposing candidates,
first learn reusable experience from the current evidence: identify what
top-performing Harnesses and Skills do well, identify unresolved failure modes,
then decide whether each lesson belongs in an existing Skill selection policy,
a successor/new Skill, or candidate-local control flow. Do not create a new
Skill unless the evidence shows a reusable workflow lesson that is not already
covered by the frozen Skill pool.

## Critical Constraints

- You MUST implement 3 new reward skill harnesses every iteration.
- Do NOT write "the frontier is optimal", "stop iterating", or abort early.
- ALWAYS complete all steps including prototyping.
- Design exactly 3 candidates per iteration: at least 1 exploitation of current frontier, at least 1 exploration.
- Every candidate MUST use the Skill bank. Do not submit candidates whose `get_skill_registry()` returns an empty `SkillRegistry`.
- Every candidate MUST retrieve at least one G-stage Skill before rubric generation and at least one J-stage Skill before judging.
- Every candidate MUST make an evidence-backed Skill decision: reuse existing Skills, change retrieval/selection, create a successor Skill, or add a new Skill. Adding a Skill is optional and should happen only when there is a trajectory-backed reusable lesson not already represented.
- Every candidate MUST justify its Skill pool: which existing Skills are reused because they worked, which Skills are avoided or narrowed because they failed, which successor/new Skills are added if any, and how retrieval decides among them.

## Candidate Design

Each candidate has one Python file at `reward_harness/agents/<name>.py`
containing the full Reward Harness control flow. Skills are independent JSON
files in the shared bank under `reward_harness/skills/*.json`; the candidate
decides which Skills to retrieve before G and J. Define exactly one
`RewardSystem` subclass or set `HARNESS_CLASS` to the intended subclass.
The Skill bank is append-only for optimization: preserve existing Skill files
so evaluated Harnesses remain reproducible. Add new Skills only when needed,
and put selection responsibility in `retrieve_skills(...)`.

Do not edit an existing Skill in place, even if it failed. If an existing Skill
has useful ideas but caused regressions, over-triggered, missed a boundary, or
needs a narrower/broader version, create a new successor Skill with a new name.
The successor should explicitly preserve the useful lesson and correct the
observed failure mode. The candidate may include both the original Skill and the
successor Skill in its frozen pool, then use retrieval to decide when the
successor should replace or complement the original.

If a candidate declares a `SKILLS = (...)` tuple, treat it as that candidate's
frozen retrievable Skill-bank snapshot, not as a hardcoded prompt snippet list.
Before defining it, inspect existing Skills and recent trajectories. Include
reusable effective Skills from top-performing Harnesses plus any successor/new
Skills needed for the candidate hypothesis. It is valid for `new_skills` to be
empty when the hypothesis is specifically about better reuse, trigger narrowing,
negative triggers, or stage-specific selection over existing Skills.

What you can and cannot modify:

- CAN: edit your new `reward_harness/agents/<name>.py` file freely.
- CAN: copy a top-performing baseline or frontier Harness into a new file, then make targeted changes.
- CAN: add new G-stage and J-stage Skill JSON files under `reward_harness/skills/` when evidence supports a reusable missing lesson.
- CAN: redesign Skill retrieval/selection inside the new Harness file.
- CAN: redesign rubric-generation prompts, judge prompts, helper functions, and in-Harness control flow inside the new file.
- CANNOT: modify, overwrite, rename, or delete existing Skill JSON files.
- CANNOT: "fix" a failed existing Skill by editing it in place. Derive a new successor Skill instead.
- CANNOT: hide new reusable judging or rubric-generation experience only as inline prompt text. If the idea is a reusable workflow instruction not already represented, add it as a successor/new Skill and retrieve it. If the idea is routing, parsing, orchestration, or tie-breaking control flow, implement it in the candidate.
- CANNOT: move reusable workflow instructions out of Skill JSON into candidate-only inline strings. Candidate code may route, select, and compose Skills, but reusable arbitration frames, checklists, and evaluation workflows belong in Skills.
- CANNOT: modify benchmark code, data files, model clients, evaluator logic, fixed payload/parsing helpers, validators, `reward_system.py`, `skill_store.py`, config files, or existing baseline Harnesses.

Design principles:

- Mechanism-first. Identify a specific failure mode from trajectories, then design a change that targets it. Never add changes speculatively.
- One mechanism per candidate. Each candidate tests exactly one hypothesis. If you are tempted to add "and also...", that is a second candidate.
- Evidence-driven hypotheses. Each hypothesis must state: observed failure pattern, change, expected mechanism, and risk.
- Error attribution before editing. For each candidate, first decide whether the target failures come from Skill retrieval, rubric generation, rubric quality, rubric-guided judging, aggregation/comparison logic, output parsing, or prompt/control-flow issues.
- Skill-first, not Skill-only. Treat Skills as an append-only bank of reusable workflow instructions. New trajectory-derived reusable experience may enter the Skill bank as a successor/new Skill, while misapplication evidence should usually change retrieval/selection, and non-reusable orchestration should stay in candidate code.
- Learn from both wins and failures. First summarize what current top-performing Harnesses do correctly, including selected Skills, generated rubrics, decisive evidence, and winner decisions. Then inspect failures to identify missing or misapplied Skills.
- Failed Skills can still be useful evidence. When an existing Skill partly helps but also causes errors, derive a successor Skill that keeps the useful mechanism and fixes the observed failure mode; do not overwrite the original.
- Keep mechanism in Skills when it is reusable. Candidate Python may implement routing, parsing-safe prompt assembly, and minimal control flow, but reusable rubrics, judge workflows, arbitration frames, evidence ledgers, veto rules, and checklists should be stored as Skill JSON and injected through retrieval.
- Do not assume each stage should use only one Skill. G and J can each retrieve multiple complementary Skills when the task needs them.
- Skill categories may include general workflow Skills, domain/task-type Skills, failure-mode Skills, response-contrast Skills, calibration Skills, and arbitration Skills.
- Domain/task-type Skills are allowed when they are general-purpose, such as code, math/reasoning, safety/refusal, factuality, instruction following, multilingual handling, concise deliverables, or tool/API correctness. Dataset-specific Skills are forbidden.
- Prefer minimal targeted changes to the current frontier or another top-performing base Harness. Do not add multi-stage gates, appeals, or verification passes unless the observed failures repeatedly require that extra stage.
- Do not exploit quirks of the held-in set, benchmark ordering, response labels, parser behavior, prompt formatting, or known answer patterns.
- Do not hardcode dataset-specific hints. Never mention dataset names in system code, prompts, or comments. General patterns such as "prioritize severe failures" or "balance rubric coverage" are fine.
- Avoid parameter-only variants. Changing rubric counts, skill counts, context budgets, score ranges, weighting constants, winner parsing wording, prompt bullet order, or generic caution phrases is not a new mechanism by itself.

Good candidates change a mechanism, such as:

- A new trajectory-derived G-stage or J-stage Skill plus a retrieval rule that decides when to use it.
- A successor Skill that preserves the useful part of an older Skill and narrows or corrects the failure mode that caused regressions.
- A new Skill retrieval or selection mechanism over a frozen pool of reused effective Skills, with negative triggers for Skills that over-fired.
- A reuse-only candidate that changes when existing Skills are injected, if trajectory evidence shows the Skill content is good but misapplied.
- A new rubric-generation workflow.
- A new judge-skill-selection strategy.
- Response-set-aware rubric discovery that improves head-to-head discrimination.
- Global rubric vs hard-constraint separation.
- Better uncertainty, tie, or hard-failure handling.
- Stronger evidence-first scoring.
- Winner selection that handles hard constraints, severe failures, or near ties.
- Stage-specific skills for rubric generation vs pairwise judging.
- Domain/task-type Skill combinations, such as code-correctness G Skills plus execution-trace J Skills, or safety-boundary G Skills plus refusal-calibration J Skills.

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
`Skill` requires `name`, `stage`, `description`, and `content`; stage must be `"G"` or `"J"`. Skill JSON may also include optional optimizer-facing fields: `intended_use`, `failure_modes`, `positive_triggers`, `negative_triggers`, `parent_skill`, `status`, and `source_evidence`. Keep reusable workflow instructions in `content`; use optional fields for selection, provenance, and applicability.
Use `load_skill_registry(...)` from `..skill_store` to load independent Skill files from `reward_harness/skills/*.json`
Use `retrieve_skills(task, responses, "G")` before rubric generation and `retrieve_skills(task, responses, "J")` before judging; both calls must return at least one selected Skill
`retrieve_skills(...)` may return multiple Skills per stage when they address complementary needs; keep the selected set compact and relevant
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
Use `build_retrieval_text`, `select_by_triggers`, `apply_successor_overrides`, and `default_stage_skills` from `..skill_store` when implementing trigger-based retrieval. Candidate-specific retrieval policy still belongs in `retrieve_skills(...)`.
Do NOT override `_task_payload`, `_candidate_payload`, `_responses_payload`, `_judge_responses_payload`, `_rubrics_payload`, `_skills_payload`, `_parse_skill_calls`, `_parse_rubrics`, `_parse_judgments`, `_validate_rubric_set`, or `_validate_winner_result`
The benchmark/evaluator calls `_validate_rubric_set` and `_validate_winner_result`; candidates should satisfy these checks rather than bypass them
Use `self.rubric_llm(prompt)` for rubric generation calls (NOT `self._rubric_llm` directly)
Use `self.judge_llm(prompt)` for comparative judging calls (NOT `self._judge_llm` directly)
Use `render_skill_block(...)` from `..skill_store` for prompt injection; implement Skill retrieval/selection policy directly in `retrieve_skills(...)` (see `agents/init_skill.py` for a reference)
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
2. Read the existing Skill bank under `reward_harness/skills/*.json`. Identify reusable G-stage and J-stage Skills, their intended triggers, and whether recent trajectories show them helping or failing.
3. Identify the current frontier/base Harness from `frontier_val.json` and inspect its source file before editing.
4. Summarize top-performing behavior first: what selected Skills, rubric patterns, evidence handling, and winner decisions appear to explain correct predictions.
5. Inspect incorrect or fragile trajectories next, then compare against nearby systems when useful. For each failure pattern, attribute the likely failure source: Skill retrieval, rubric generation, rubric quality, rubric-guided judging, aggregation/comparison logic, output parsing, or prompt/control-flow.
6. Convert the analysis into reusable Skill lessons:
   * G-stage lessons for rubric discovery, criteria specificity, domain/task-type handling, or response contrast.
   * J-stage lessons for evidence checking, hard-failure dominance, calibration, arbitration, or final winner selection.
   * Successor lessons for existing Skills that partly worked but need narrower triggers, broader coverage, clearer evidence discipline, or better stage separation.
7. Formulate 3 hypotheses — each must be falsifiable, target a different mechanism, and include: observed success pattern, observed failure pattern, attributed failure source, Skill action (`reuse`, `retrieval_update`, `successor_skill`, or `new_skill`), reused Skills, successor/new Skills if any, retrieval policy, expected mechanism, and risk.

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
2. Decide the Skill action from evidence. Reuse existing Skill files when they already encode the needed lesson; add a successor/new Skill JSON file only when a reusable missing or corrected lesson is needed. Do not edit or delete existing Skill files. If the new Skill revises an old Skill, name it as a successor and describe which failure mode it corrects.
3. Define the candidate Skill pool as a frozen retrievable set containing reused effective Skills plus any successor/new Skill(s). The pool must include at least one G-stage Skill and at least one J-stage Skill.
4. Implement the new mechanism according to your hypothesis. The candidate must inject selected G Skills in `build_rubrics(...)` and selected J Skills in `judge(...)`. Retrieval may select multiple Skills per stage when the task has multiple relevant signals.
5. Self-critique (mandatory): After implementing, re-read the file and check: does this harness introduce a genuinely NEW mechanism, or is it just a parameter variant? If the logic in `get_skill_registry()`, `retrieve_skills()`, `build_rubrics()`, and `judge()` is identical to the base except for constants, REWRITE with a truly novel mechanism.
6. Import/interface check:

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
      "skill_action": "reuse|retrieval_update|successor_skill|new_skill",
      "reused_skills": ["<existing_skill_name>"],
      "new_skills": ["<new_or_successor_skill_name_if_any>"],
      "skill_selection_rationale": "<why this frozen pool and retrieval policy are appropriate>",
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
