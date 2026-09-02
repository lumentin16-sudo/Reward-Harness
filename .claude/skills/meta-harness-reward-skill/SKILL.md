---
name: meta-harness-reward-skill
description: Run one iteration of reward skill harness evolution. Called by meta_harness.py or interactively via /meta-harness.
---

# Meta-Harness (Reward Skill Harness Evolution)

Run ONE iteration of reward skill harness evolution. Do all work in the main session; do NOT delegate to subagents.

**You do NOT run benchmarks.** Analyze result summaries and reward trajectories, prototype changes, implement candidates, and write `pending_eval.json`. The outer loop (`meta_harness.py`) runs benchmarks.

## Context

You are evolving comparative Reward Harnesses under `reward_harness/agents/`. Each Harness receives a public `Query` and two anonymous `Response`s. Gold labels are evaluator-only and must never enter model prompts.

The default searchable workflow is two-stage:

1. **G: Rubric generation.** `build_rubrics(query, responses)` selects G-stage Skills, calls `rubric_llm`, and returns a shared, response-aware `RubricSet`. The rubric may inspect both responses to discover discriminative criteria, but criteria must remain position-independent and must not encode the winner.
2. **J: Rubric-based judging.** `judge(query, responses, rubrics)` selects J-stage Skills, calls `judge_llm`, compares both responses based on the generated rubrics and decisive evidence, and returns exactly one `WinnerResult`.

Reference Harnesses:

- `no_rubric.py`: direct pairwise Judge without Rubrics or Skills.
- `no_skill.py`: online Rubric generation without Skills, then rubric-based pairwise judging.
- `init_skill_no_rubric.py`: no-rubric reference Harness.
- `init_skill.py`: seed G/J Skills, response-aware Rubric generation, then rubric-based pairwise judging.

The active baselines and benchmarks are defined by the task prompt and `config.yaml`. Do not assume every reference Harness is active.

Default search should build from the current frontier or another top-performing rubric-based, skill-using Harness. If `no_skill.py` is the global frontier, build candidates from the best skill-using Harness reported in the task prompt or evolution history. Explore no-rubric designs only when the task prompt explicitly asks for them.

## Critical Rules

- Implement exactly 3 new reward skill Harnesses per iteration.
- Include at least 1 exploitation candidate and at least 1 exploration candidate.
- Do not write that the frontier is optimal, stop iterating, or abort early.
- Complete analysis, prototyping, implementation, import checks, and `pending_eval.json`.
- Every candidate must use the Skill bank: non-empty `get_skill_registry()`, at least one selected G-stage Skill before rubric generation, and at least one selected J-stage Skill before judging.
- Every candidate must make an evidence-backed Skill decision: reuse existing Skills, update retrieval/selection, create a successor Skill, or add a new Skill.
- Add a Skill only when trajectories show a reusable workflow lesson not already represented by the frozen Skill pool.
- Do not exploit held-in quirks, benchmark order, response labels, parser behavior, prompt formatting, known answer patterns, or dataset names.
- Avoid parameter-only variants: changing counts, budgets, thresholds, score ranges, wording order, or generic caution phrases is not a new mechanism by itself.

## Candidate Design

Each candidate is one Python file:

```text
reward_harness/agents/<name>.py
```

It must define exactly one `RewardSystem` subclass or set `HARNESS_CLASS` to the intended subclass.

Skills are independent JSON files:

```text
reward_harness/skills/*.json
```

Treat the Skill bank as append-only. Do not modify, rename, overwrite, or delete existing Skill JSON files. If an existing Skill partly helps but causes failures, create a successor Skill with a new name and retrieve it only when appropriate.

Candidate design starts from learning, not from writing code. First inspect the top-performing Harness, its trajectories, and the selected Skills. Summarize reusable success lessons and recurring failure lessons. Then decide which existing Skills to keep, which failed Skills need successor versions, which new Skills are missing, and whether the main change should be retrieval/selection rather than Skill content.

Each candidate should declare its Skill pool before the rest of the logic. If it uses `SKILLS = (...)`, that tuple is the candidate's frozen retrievable Skill-bank snapshot: reused effective Skills, successor Skills, and new Skills needed for the hypothesis. Domain/task-type Skills are allowed, and a candidate may include multiple domain-specific Skills, as long as they encode general-purpose lessons rather than dataset-specific hints. It is valid to add no new Skills only when the hypothesis is specifically about better reuse, trigger narrowing, negative triggers, or stage-specific retrieval over existing Skills.

What you can modify:

- CAN edit the new candidate file freely.
- CAN copy a frontier/top-performing Harness into a new file, then make targeted changes.
- CAN add new G-stage or J-stage Skill JSON files when evidence supports a reusable missing lesson.
- CAN redesign Skill retrieval/selection, prompt assembly, helper functions, and in-Harness control flow inside the new file.

What you cannot modify:

- CANNOT modify benchmark code, benchmark data, model clients, evaluator logic, fixed payload/parsing helpers, validators, `reward_system.py`, `skill_store.py`, config files, existing baseline Harnesses, or existing Skill files.
- CANNOT hide reusable rubric-generation or judging workflows only as inline prompt text. Reusable rubrics, judge workflows, arbitration frames, evidence ledgers, veto rules, and checklists belong in Skill JSON and must be injected through retrieval. Candidate code should handle routing, selection, parsing-safe prompt assembly, and local orchestration.

Design principles:

- Mechanism-first: identify a trajectory-backed failure mode, then target it.
- One mechanism per candidate; no unrelated "and also" changes.
- Evidence-driven hypotheses must state observed pattern, attributed failure source, change, expected mechanism, and risk.
- First learn from top-performing behavior, then inspect failures; candidate code should reflect what was learned.
- Attribute failures before editing: Skill retrieval, rubric generation, rubric quality, rubric-guided judging, aggregation/comparison, output parsing, or prompt/control flow.
- G and J may each retrieve multiple complementary Skills.
- Skill categories may include general workflow, domain/task-type, failure-mode, response-contrast, calibration, and arbitration Skills.

Good candidate mechanisms include:

- A trajectory-derived G/J Skill plus retrieval rule.
- A successor Skill that preserves a useful old mechanism and fixes its observed failure mode.
- A retrieval/selection policy over reused effective Skills with negative triggers for over-firing.
- A reuse-only candidate that changes when existing Skills are injected.
- Better response-set-aware rubric discovery, hard-constraint separation, evidence-first judging, uncertainty/near-tie handling, or winner arbitration.
- Domain/task-type Skill combinations such as code-correctness G Skills plus execution-trace J Skills.

## RewardSystem Interface

```python
class RewardSystem(ABC):
    def __init__(self, rubric_llm: LLMCallable, judge_llm: LLMCallable) -> None: ...

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

    def build_rubrics(self, task: Query, responses: tuple[Response, ...]) -> RubricSet: ...

    def judge(
        self,
        task: Query,
        responses: tuple[Response, ...],
        rubrics: RubricSet,
    ) -> WinnerResult: ...
```

Use the existing protocol helpers from `RewardSystem`:

- `self._task_payload(task)`
- `self._candidate_payload(candidate)`
- `self._responses_payload(responses)`
- `self._judge_responses_payload(responses)`
- `self._rubrics_payload(rubrics)`
- `self._skills_payload(selected_skills)`
- `self._parse_skill_calls(raw_response, registry)`
- `self._parse_rubrics(raw_response)`
- `self._parse_judgments(raw_response, rubrics)`

Do not override fixed payload, parser, or validator helpers. The evaluator calls `_validate_rubric_set` and `_validate_winner_result`; satisfy those checks rather than bypassing them.

Use:

- `self.rubric_llm(prompt)` for rubric-generation and G-stage skill-selection calls.
- `self.judge_llm(prompt)` for response judging and J-stage skill-selection calls.
- `load_skill_registry(...)` from `..skill_store` to load Skill JSON files.
- `render_skill_block(...)` from `..skill_store` for prompt injection.
- `registry.for_stage("G")` / `registry.for_stage("J")` before stage-specific selection.
- `build_retrieval_text`, `select_by_triggers`, `apply_successor_overrides`, and `default_stage_skills` from `..skill_store` if implementing trigger-based retrieval.

`Skill` requires `name`, `stage`, `description`, and `content`; stage is `"G"` or `"J"`. Optional optimizer-facing fields are `intended_use`, `failure_modes`, `positive_triggers`, `negative_triggers`, `parent_skill`, `status`, and `source_evidence`. Keep prompt-visible workflow instructions in `content`.

`build_rubrics` and `judge` must work cold start.

## Workflow

Do all steps yourself in the main session.

### Step 0: Post-eval Reports

Check the reports directory named in the task prompt's "Optimization state" section. For each past iteration with results in `evolution_summary.jsonl` but no report, write one report of at most 30 lines covering: what changed, which benchmarks improved/regressed and why, and one takeaway.

### Step 1: Analyze

Read:

- `evolution_summary.jsonl`
- `frontier_val.json`
- current task prompt/config for active benchmarks and baselines
- recent `results/<run_tag>/<benchmark>/<harness>/<model>/trajectories.jsonl`
- baseline/frontier Harness source files
- existing Skill JSON files under `reward_harness/skills/`

Then follow this learning chain:

1. Identify the current frontier or another top-performing skill-using Harness and inspect its source before editing.
2. Read its trajectories first. Summarize what worked: selected Skills, generated rubric patterns, evidence handling, domain/task-type behavior, and winner decisions.
3. Read failed or fragile trajectories. Summarize what failed and attribute each pattern to Skill retrieval, rubric generation, rubric quality, rubric-guided judging, aggregation/comparison, output parsing, or prompt/control flow.
4. Inspect the existing Skill bank after the trajectory analysis. Decide which Skills should be reused, which Skills should be avoided or narrowed by retrieval, which failed-but-useful Skills need successor versions, and which missing lessons require new Skills.
5. Separate lessons by stage:
   - G-stage lessons: rubric discovery, criterion specificity, response contrast, domain/task-type requirements.
   - J-stage lessons: evidence checking, hard-failure dominance, calibration, arbitration, final winner selection.
6. Formulate 3 falsifiable hypotheses. Each must include observed success pattern, observed failure pattern, attributed failure source, Skill action (`reuse`, `retrieval_update`, `successor_skill`, or `new_skill`), reused Skills, new/successor Skills if any, retrieval policy, expected mechanism, and risk.

### Step 2: Prototype

Prototype before writing final candidate files. For each candidate:

1. Write a temporary test script in `/tmp/`.
2. Pull real examples from recent trajectories.
3. Try 2-3 variants when useful.
4. Delete temporary scripts when done.

### Step 3: Implement

For each candidate:

1. For exploitation candidates, copy the current frontier or another top-performing skill-using Harness. For exploration candidates, copy the closest working reference Harness as a scaffold, then implement a genuinely different mechanism.
2. Add new/successor Skill JSON files only when needed.
3. Declare the frozen Skill pool near the top of the file, including reused Skills plus any new/successor Skills. The pool must include at least one G Skill and one J Skill.
4. Implement `get_skill_registry(...)` and `retrieve_skills(...)` so retrieval explains how learned Skills are selected for each task/domain/stage.
5. Implement prompt/control-flow changes according to the hypothesis, injecting selected G Skills in `build_rubrics(...)` and selected J Skills in `judge(...)`.
6. Self-critique: if the candidate differs only by constants or wording, rewrite it with a real mechanism.
7. Run:

```bash
python3 -c "from reward_harness.agents.<name> import *; print('OK')"
```

Do not edit config files just to register candidates. The benchmark auto-discovers files in `reward_harness/agents/`.

### Step 4: Write pending_eval.json

Write to the path specified in the task prompt, not a hardcoded path:

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

Typical outputs:

- `results/<run_tag>/<benchmark>/<harness>/<model>/trajectories.jsonl`
- `results/<run_tag>/<benchmark>/<harness>/<model>/summary.json`
- `results/<run_tag>/<benchmark>/<harness>/<model>/config.json`

Use held-in/search trajectories for optimization analysis. Do not use regression held-out trajectories for detailed optimization unless the outer-loop task prompt explicitly allows it.

## evolution_summary.jsonl

One JSON object per evaluated candidate:

```json
{"iteration": 1, "system": "example_harness", "avg_val": 45.0, "axis": "exploitation", "hypothesis": "...", "delta": 2.1, "outcome": "45.0% (+2.1)", "components": ["tag1", "tag2", "tag3"]}
```

Treat `evolution_summary.jsonl`, `frontier_val.json`, and recent training traces as the shipped history sources in this trimmed repo.
