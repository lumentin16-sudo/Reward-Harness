"""本地 vLLM 上的 RewardBench、RewardBench 2 与 RM-Bench 评测 CLI。"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .agent_loader import DEFAULT_AGENTS_DIR, HarnessSpec, discover_harnesses
from .benchmarks import ADAPTERS, BenchmarkAdapter, BenchmarkCase
from .benchmarks.base import DEFAULT_DATA_ROOT
from .model_client import RecordingLLM, VLLMBackend
from .reward_system import Response, RewardSystem, RubricSet, WinnerResult


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _path_name(value: str) -> str:
    """把模型名称转换成安全且可读的目录名。"""

    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-.") or "unnamed"


def _run_directory(
    *,
    run_root: Path,
    benchmark: str,
    harness: str,
    model: str,
) -> Path:
    return run_root / benchmark / harness / _path_name(model.split("/")[-1])


def _usable_summary(
    path: Path,
    *,
    expected_cases: int,
    expected_trials: int,
) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        isinstance(value, dict)
        and value.get("status") == "complete"
        and isinstance(value.get("counts"), dict)
        and value["counts"].get("cases") == expected_cases
        and isinstance(value.get("trials"), list)
        and len(value["trials"]) == expected_trials
        and isinstance(value.get("voting"), dict)
        and value["voting"].get("n") == expected_trials
    ):
        return value
    return None


def _attempt(
    call: Callable[[], Any],
    retries: int,
    *,
    on_retry: Callable[[], None] | None = None,
) -> tuple[Any | None, str | None]:
    last_error: str | None = None
    for attempt in range(retries + 1):
        try:
            return call(), None
        except Exception as exc:  # 单条错误必须被记录并允许 benchmark 继续。
            last_error = f"{type(exc).__name__}: {exc}"
            # vLLM 客户端已经完成请求重试；这里只重试 JSON 解析和接口校验错误。
            if isinstance(exc, RuntimeError):
                break
            if attempt < retries and on_retry is not None:
                on_retry()
    return None, last_error


def evaluate_case(
    case: BenchmarkCase,
    harness_type: type[RewardSystem],
    rubric_llm: Callable[[str], str],
    judge_llm: Callable[[str], str],
    *,
    stage_retries: int = 2,
) -> dict[str, Any]:
    """可信 evaluator：完整匿名 Responses 上一次 G、一次 forced-choice J。"""

    harness = harness_type(rubric_llm, judge_llm)
    outcome: dict[str, Any] = {
        "case_id": case.case_id,
        "group": case.group,
        "query": _jsonable(case.task),
        "responses": _jsonable(case.candidates),
        "gold": _jsonable(case.gold),
        "selected_skills": {"G": [], "J": []},
        "skill_calls": [],
        "rubrics": None,
        "winner_result": None,
        "error": None,
    }
    judging_responses, original_id_by_anonymous_id = _anonymous_responses(case)
    original_retrieve_skills = harness.retrieve_skills

    def recording_retrieve_skills(
        task: Any,
        responses: tuple[Response, ...],
        stage: Any,
    ) -> Any:
        skills = original_retrieve_skills(task, responses, stage)
        payload = _jsonable(RewardSystem._skills_payload(tuple(skills)))
        stage_key = str(stage)
        if stage_key in outcome["selected_skills"]:
            outcome["selected_skills"][stage_key] = payload
        outcome["skill_calls"].append({"stage": stage_key, "skills": payload})
        return skills

    harness.retrieve_skills = recording_retrieve_skills  # type: ignore[method-assign]

    def build() -> RubricSet:
        rubrics = harness.build_rubrics(case.task, judging_responses)
        # 显式调用基类校验器，候选实现无法通过覆盖方法绕过 evaluator。
        RewardSystem._validate_rubric_set(harness, case.task, rubrics)
        return rubrics

    rubrics, error = _attempt(
        build,
        stage_retries,
        on_retry=getattr(rubric_llm, "invalidate_last", None),
    )
    if error or rubrics is None:
        outcome["error"] = {"stage": "build_rubrics", "message": error}
        return outcome
    outcome["rubrics"] = _jsonable(rubrics)

    def judge() -> WinnerResult:
        result = harness.judge(case.task, judging_responses, rubrics)
        RewardSystem._validate_winner_result(case.task, judging_responses, result)
        return result

    winner, error = _attempt(
        judge,
        stage_retries,
        on_retry=getattr(judge_llm, "invalidate_last", None),
    )
    if error or winner is None:
        outcome["error"] = {"stage": "judge", "message": error}
        return outcome

    original_winner_id = original_id_by_anonymous_id[winner.winner_response_id]
    outcome["winner_result"] = {
        "query_id": winner.query_id,
        "winner_response_id": original_winner_id,
        "metadata": {
            **dict(winner.metadata),
            "anonymous_winner_response_id": winner.winner_response_id,
        },
    }
    return outcome


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    # ``utf-8-sig`` 同时兼容普通 UTF-8 和 Windows 工具写出的 UTF-8 BOM。
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            # 进程中断会留下不完整记录；如果之后曾继续追加，它不一定仍是最后一行。
            # 跳过该行后，其 case_id 不会进入 completed，resume 会自动重新评测该题。
            print(
                f"Warning: skipping malformed JSONL record "
                f"{path}:{line_number}: {exc}",
                file=sys.stderr,
                flush=True,
            )
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _sample_cases(
    cases: list[BenchmarkCase], sample_size: int, seed: int
) -> list[BenchmarkCase]:
    """对 adapter 产出的全部样本做可复现的全局随机抽样。"""

    # RM-Bench 的一个原始 prompt 被展开成9个 pair；抽样时必须整组保留。
    if cases and all("original_case_id" in case.gold for case in cases):
        grouped: dict[str, list[BenchmarkCase]] = {}
        for case in cases:
            grouped.setdefault(str(case.gold["original_case_id"]), []).append(case)
        if sample_size <= 0 or sample_size >= len(grouped):
            return cases
        selected_ids = set(
            random.Random(seed).sample(list(grouped), sample_size)
        )
        return [
            case
            for original_id, group in grouped.items()
            if original_id in selected_ids
            for case in group
        ]

    if sample_size <= 0 or sample_size >= len(cases):
        return cases
    return random.Random(seed).sample(cases, sample_size)


def _anonymous_responses(
    case: BenchmarkCase,
) -> tuple[tuple[Response, ...], dict[str, str]]:
    """稳定重排并匿名化 Responses，同时保留 evaluator 侧 ID 映射。"""

    ordered = sorted(
        case.candidates,
        key=lambda response: hashlib.sha256(
            (case.case_id + "\0" + response.content).encode("utf-8")
        ).hexdigest(),
    )
    anonymous = tuple(
        Response(response_id=f"anonymous_{index:03d}", content=response.content)
        for index, response in enumerate(ordered)
    )
    original_id_by_anonymous_id = {
        anonymous_response.response_id: original_response.response_id
        for anonymous_response, original_response in zip(anonymous, ordered)
    }
    return anonymous, original_id_by_anonymous_id


def _mean_summary_values(values: list[Any]) -> Any:
    """递归平均多轮 summary 中同构的数值指标。"""

    if values and all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in values
    ):
        return sum(float(value) for value in values) / len(values)
    if values and all(isinstance(value, dict) for value in values):
        first = values[0]
        return {
            key: _mean_summary_values([value[key] for value in values])
            for key in first
            if all(key in value for value in values)
        }
    return values[0] if values else None


def _build_voting_outcomes(
    outcomes: list[dict[str, Any]],
    trial_num: int,
) -> list[dict[str, Any]]:
    """对每轮 forced-choice winner 投票，平票保持 unresolved。"""

    by_case: dict[str, list[dict[str, Any]]] = {}
    for row in outcomes:
        by_case.setdefault(str(row["case_id"]), []).append(row)

    voting_outcomes = []
    for case_id, rows in by_case.items():
        rows.sort(key=lambda row: int(row.get("trial_index", 0)))
        prototype = rows[0]
        response_ids = [
            str(response["response_id"])
            for response in prototype.get("responses", [])
        ]
        votes = {response_id: 0.0 for response_id in response_ids}
        abstentions = 0

        for row in rows:
            winner = row.get("winner_result")
            winner_id = (
                str(winner.get("winner_response_id"))
                if isinstance(winner, dict)
                else ""
            )
            if row.get("error") or winner_id not in votes:
                abstentions += 1
                continue
            votes[winner_id] += 1.0

        vote_rates = {
            response_id: votes[response_id] / trial_num
            for response_id in response_ids
        }
        best_vote = max(votes.values()) if votes else 0.0
        voting_winners = [
            response_id
            for response_id in response_ids
            if votes[response_id] == best_vote and best_vote > 0
        ]
        resolved_winner = voting_winners[0] if len(voting_winners) == 1 else None
        voting_outcomes.append(
            {
                "case_id": case_id,
                "group": prototype["group"],
                "query": prototype["query"],
                "responses": prototype["responses"],
                "gold": prototype["gold"],
                "rubrics": None,
                "winner_result": (
                    {
                        "query_id": prototype["query"]["query_id"],
                        "winner_response_id": resolved_winner,
                        "metadata": {"aggregation": "voting"},
                    }
                    if resolved_winner is not None
                    else None
                ),
                "model_calls": [],
                "usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "latency_ms": 0.0,
                },
                "error": (
                    {
                        "stage": "voting",
                        "message": "all trials failed or were incomplete",
                    }
                    if abstentions == trial_num
                    else None
                ),
                "voting": {
                    "n": trial_num,
                    "votes": votes,
                    "vote_rates": vote_rates,
                    "abstentions": abstentions,
                    "tied": len(voting_winners) > 1,
                },
            }
        )
    return voting_outcomes


def _summarize_trials(
    adapter: BenchmarkAdapter,
    outcomes: list[dict[str, Any]],
    trial_num: int,
) -> dict[str, Any]:
    """每轮独立计算官方指标，再对 N 轮指标做算术平均。"""

    raw_trials = []
    for trial_index in range(trial_num):
        rows = [
            row
            for row in outcomes
            if int(row.get("trial_index", 0)) == trial_index
        ]
        raw_trials.append(
            {"trial_index": trial_index, **adapter.summarize(rows)}
        )

    averaged = _mean_summary_values(
        [
            {key: value for key, value in summary.items() if key != "trial_index"}
            for summary in raw_trials
        ]
    )
    if not isinstance(averaged, dict):
        raise TypeError("benchmark adapter summary must be a dictionary")
    voting_outcomes = [
        adapter.score_outcome(outcome)
        for outcome in _build_voting_outcomes(outcomes, trial_num)
    ]
    raw_voting = adapter.summarize(voting_outcomes)

    def organize(raw: dict[str, Any]) -> dict[str, Any]:
        """把 adapter 输出拆成关键指标、其他指标和计数。"""

        primary_metrics = {
            key: raw[key]
            for key in ("beyond_rubric", "auto_rubric")
            if key in raw
        }
        ignored = {
            "benchmark",
            "beyond_rubric",
            "auto_rubric",
            "num_cases",
            "num_errors",
            "trial_index",
        }
        return {
            "primary_metrics": primary_metrics,
            "metrics": {
                key: value for key, value in raw.items() if key not in ignored
            },
            "counts": {
                "cases": int(raw.get("num_cases", 0) or 0),
                "errors": int(raw.get("num_errors", 0) or 0),
            },
        }

    trials = []
    for raw in raw_trials:
        organized = organize(raw)
        trials.append({"trial_index": raw["trial_index"], **organized})

    return {
        "benchmark": adapter.name,
        **organize(averaged),
        "trials": trials,
        "voting": {"n": trial_num, **organize(raw_voting)},
        "counts": {
            "cases": len({str(row["case_id"]) for row in outcomes}),
            "evaluations": len(outcomes),
            "errors": sum(bool(row.get("error")) for row in outcomes),
        },
    }


def _run_one_case(
    case: BenchmarkCase,
    adapter: BenchmarkAdapter,
    harness_type: type[RewardSystem],
    backend: VLLMBackend,
    trial_index: int,
    use_cache: bool,
    stage_retries: int,
) -> dict[str, Any]:
    rubric = backend.recorder("rubric", use_cache=use_cache)
    judge = backend.recorder("judge", use_cache=use_cache)
    outcome = evaluate_case(
        case,
        harness_type,
        rubric,
        judge,
        stage_retries=stage_retries,
    )
    outcome["trial_index"] = trial_index
    outcome["model_calls"] = [record.to_dict() for record in (*rubric.records, *judge.records)]
    outcome["usage"] = {
        "input_tokens": sum(record.input_tokens for record in (*rubric.records, *judge.records)),
        "output_tokens": sum(record.output_tokens for record in (*rubric.records, *judge.records)),
        "latency_ms": sum(record.latency_ms for record in (*rubric.records, *judge.records)),
    }
    return adapter.score_outcome(outcome)


def run_configuration(
    *,
    adapter: BenchmarkAdapter,
    harness: HarnessSpec,
    cases: list[BenchmarkCase],
    backend: VLLMBackend,
    run_root: Path,
    workers: int,
    force: bool,
    trial_num: int,
    stage_retries: int,
    smoke_per_group: int,
    seed: int,
    request_workers: int = 16,
    sample_size: int = 0,
    data_dir: Path | None = None,
) -> dict[str, Any]:
    run_directory = _run_directory(
        run_root=run_root,
        benchmark=adapter.name,
        harness=harness.name,
        model=backend.model,
    )
    summary_path = run_directory / "summary.json"
    if not force:
        completed_summary = _usable_summary(
            summary_path,
            expected_cases=len(cases),
            expected_trials=trial_num,
        )
        if completed_summary is not None:
            print(
                f"[{adapter.name}/{harness.name}] SKIP already evaluated: "
                f"{summary_path}",
                flush=True,
            )
            return completed_summary

    run_directory.mkdir(parents=True, exist_ok=True)
    trajectories_path = run_directory / "trajectories.jsonl"
    raw_existing = [] if force else _read_jsonl(trajectories_path)
    target_ids = {case.case_id for case in cases}
    target_keys = {
        (trial_index, case.case_id)
        for trial_index in range(trial_num)
        for case in cases
    }
    # 同一轮的同一个 case 若留下重复行，以最后一条完整记录为准。
    existing_by_key = {
        (int(row.get("trial_index", 0)), str(row["case_id"])): row
        for row in raw_existing
        if (
            str(row.get("case_id")) in target_ids
            and (int(row.get("trial_index", 0)), str(row.get("case_id")))
            in target_keys
        )
    }
    existing = list(existing_by_key.values())
    if force or not trajectories_path.exists():
        trajectories_path.write_text("", encoding="utf-8")
    completed = set(existing_by_key)
    pending = [
        (trial_index, case)
        for trial_index in range(trial_num)
        for case in cases
        if (trial_index, case.case_id) not in completed
    ]

    config = {
        "status": "running",
        "benchmark": adapter.name,
        "dataset_id": adapter.dataset_id,
        "split": adapter.split,
        "agent": harness.name,
        "agent_file": str(harness.source_path),
        "agent_source_sha256": harness.source_sha256,
        "result_protocol": "winner",
        "backend": "vllm",
        "base_url": getattr(backend, "base_url", None),
        "model": backend.model,
        "temperature": backend.temperature,
        "max_tokens": backend.max_tokens,
        "enable_thinking": False,
        "request_retries": backend.request_retries,
        "stage_retries": stage_retries,
        "trial_num": trial_num,
        "workers": workers,
        "request_workers": request_workers,
        "prompt_cache": (
            str(backend.cache_dir) if getattr(backend, "cache_dir", None) else None
        ),
        "smoke_per_group": smoke_per_group,
        "sample_size": sample_size,
        "data_dir": str(data_dir.resolve()) if data_dir is not None else None,
        "seed": seed,
        "automatic_resume": True,
        "trajectories": str(trajectories_path.resolve()),
        "summary": str(summary_path.resolve()),
    }
    _write_json(run_directory / "config.json", config)

    with trajectories_path.open("a", encoding="utf-8", buffering=1) as stream:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _run_one_case,
                    case,
                    adapter,
                    harness.harness_type,
                    backend,
                    trial_index,
                    trial_num == 1,
                    stage_retries,
                ): (trial_index, case.case_id)
                for trial_index, case in pending
            }
            for future in as_completed(futures):
                trial_index, case_id = futures[future]
                try:
                    row = future.result()
                except Exception as exc:  # evaluator 自身异常也不得终止其余样本。
                    case = next(
                        item
                        for index, item in pending
                        if index == trial_index and item.case_id == case_id
                    )
                    row = adapter.score_outcome({
                        "trial_index": trial_index,
                        "case_id": case_id,
                        "group": case.group,
                        "query": _jsonable(case.task),
                        "responses": _jsonable(case.candidates),
                        "gold": _jsonable(case.gold),
                        "rubrics": None,
                        "winner_result": None,
                        "model_calls": [],
                        "usage": {"input_tokens": 0, "output_tokens": 0, "latency_ms": 0.0},
                        "error": {"stage": "evaluator", "message": f"{type(exc).__name__}: {exc}"},
                    })
                # 每行就是一条可独立交给 Harness Optimizer 的完整轨迹。
                row = {
                    "benchmark": adapter.name,
                    "harness": harness.name,
                    **row,
                }
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                existing.append(row)
                status = "ERROR" if row.get("error") else "OK"
                message = ""
                if row.get("error"):
                    error = row["error"]
                    if isinstance(error, dict):
                        stage = error.get("stage", "unknown")
                        detail = str(error.get("message", "unknown error")).replace("\n", " ")
                        message = f" | stage={stage} | {detail[:300]}"
                    else:
                        message = f" | {str(error)[:300]}"
                print(
                    f"[{adapter.name}/{harness.name}] "
                    f"trial={trial_index + 1}/{trial_num} "
                    f"{status} {case_id}{message}",
                    flush=True,
                )

    aggregated = _summarize_trials(adapter, existing, trial_num)
    summary = {
        "status": "complete",
        "benchmark": aggregated["benchmark"],
        "harness": harness.name,
        "primary_metrics": aggregated["primary_metrics"],
        "metrics": aggregated["metrics"],
        "trials": aggregated["trials"],
        "voting": aggregated["voting"],
        "counts": aggregated["counts"],
        "usage": {
            key: sum(float(row.get("usage", {}).get(key, 0)) for row in existing)
            for key in ("input_tokens", "output_tokens", "latency_ms")
        },
        "artifacts": {
            "trajectories": str(trajectories_path.resolve()),
        },
    }
    config["status"] = "complete"
    _write_json(run_directory / "config.json", config)
    _write_json(summary_path, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    # 保持帮助文本为 ASCII，兼容 Windows 上仍使用 cp1252 的终端。
    parser = argparse.ArgumentParser(
        description="Run Qwen3-8B reward harness benchmarks."
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=sorted(ADAPTERS),
        default=["rewardbench"],
    )
    parser.add_argument(
        "--agents",
        "--harnesses",
        dest="harnesses",
        nargs="+",
        default=None,
        help=(
            "Agent file stems to evaluate; default selects concrete winner-only agents. "
            "--harnesses is kept as a compatibility alias."
        ),
    )
    parser.add_argument("--agents-dir", type=Path, default=DEFAULT_AGENTS_DIR)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--request-workers", type=int, default=16)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument(
        "--trial-num",
        type=int,
        default=1,
        help=(
            "Run the complete benchmark N independent times (G and J), then "
            "report both Average@N and Voting@N from those runs."
        ),
    )
    parser.add_argument("--smoke-per-group", type=int, default=2)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="Globally sample N cases after adapter loading; 0 keeps all cases.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--run-tag",
        default=None,
        help="Top-level run directory name; default is local time YYYYMMDD_HHMMSS.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Directory containing pre-generated normalized benchmark data.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-evaluate and replace existing files under the selected run tag.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage-retries", type=int, default=2)
    parser.add_argument("--skip-preflight", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_tag = args.run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_tag):
        raise SystemExit("run-tag may contain only letters, digits, dot, underscore and dash")
    run_root = args.output_dir.resolve() / run_tag
    if (
        args.workers < 1
        or args.request_workers < 1
        or args.trial_num < 1
        or args.max_tokens < 1
        or args.temperature < 0
        or args.smoke_per_group < 0
        or args.sample_size < 0
        or args.stage_retries < 0
    ):
        raise SystemExit(
            "workers, request-workers, trial-num and max-tokens must be >= 1; "
            "temperature must be >= 0; "
            "smoke-per-group, sample-size and stage-retries must be >= 0"
        )
    if args.trial_num > 1 and args.temperature == 0:
        print(
            "Warning: trial-num > 1 repeats complete evaluations, but "
            "temperature=0 may make the independent runs identical.",
            file=sys.stderr,
            flush=True,
        )
    try:
        requested_names = set(args.harnesses) if args.harnesses else None
        discovered = discover_harnesses(args.agents_dir, include_names=requested_names)
    except (FileNotFoundError, ImportError, TypeError, ValueError) as exc:
        print(f"Failed to discover agents: {exc}", file=sys.stderr)
        return 5
    by_name = {item.name: item for item in discovered}
    requested_harnesses = args.harnesses or sorted(by_name)
    unknown = sorted(set(requested_harnesses) - set(by_name))
    if unknown:
        print(
            f"Unknown harnesses {unknown}; discovered: {sorted(by_name)}",
            file=sys.stderr,
        )
        return 5
    harnesses = [by_name[name] for name in requested_harnesses]
    print(
        "Discovered agents: " + ", ".join(item.name for item in discovered),
        flush=True,
    )

    supported_pairwise = {
        "held_in",
        "rewardbench",
        "rmbench",
    }
    unsupported_benchmarks = sorted(set(args.benchmarks) - supported_pairwise)
    if unsupported_benchmarks:
        print(
            "The winner-only runner currently supports pairwise datasets only; "
            f"unsupported: {unsupported_benchmarks}",
            file=sys.stderr,
        )
        return 4

    # 在连接 vLLM 之前验证本地数据，缺失或损坏时不产生模型请求。
    prepared: list[tuple[BenchmarkAdapter, list[BenchmarkCase]]] = []
    for benchmark_name in args.benchmarks:
        adapter = ADAPTERS[benchmark_name](data_root=args.data_dir)
        load_started = time.perf_counter()
        try:
            cases = adapter.load_cases(
                smoke_per_group=args.smoke_per_group, seed=args.seed
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"Failed to load {benchmark_name}: {exc}", file=sys.stderr)
            return 4
        cases = _sample_cases(cases, args.sample_size, args.seed)
        load_seconds = time.perf_counter() - load_started
        print(
            f"Loaded {len(cases)} cases for {benchmark_name} in {load_seconds:.1f}s",
            flush=True,
        )
        prepared.append((adapter, cases))

    base_url = args.base_url or "http://127.0.0.1:8000/v1"

    jobs: list[tuple[BenchmarkAdapter, list[BenchmarkCase], HarnessSpec]] = []
    for adapter, cases in prepared:
        for harness in harnesses:
            jobs.append((adapter, cases, harness))

    if not args.force:
        incomplete_jobs = []
        for adapter, cases, harness in jobs:
            run_directory = _run_directory(
                run_root=run_root,
                benchmark=adapter.name,
                harness=harness.name,
                model=args.model,
            )
            if _usable_summary(
                run_directory / "summary.json",
                expected_cases=len(cases),
                expected_trials=args.trial_num,
            ) is None:
                incomplete_jobs.append((adapter, cases, harness))
            else:
                print(
                    f"[{adapter.name}/{harness.name}] SKIP already evaluated",
                    flush=True,
                )
        if not incomplete_jobs:
            print("All requested benchmark/agent combinations are complete.", flush=True)
            return 0
        jobs = incomplete_jobs

    backend = VLLMBackend(
        base_url=base_url,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        request_workers=args.request_workers,
        cache_dir=run_root / ".llm_cache",
    )
    if not args.skip_preflight:
        preflight: RecordingLLM = backend.recorder("rubric", use_cache=False)
        try:
            preflight('Return exactly this JSON object: {"ok": true}')
        except Exception as exc:
            print(f"vllm preflight failed: {exc}", file=sys.stderr)
            return 3
        record = preflight.records[-1]
        print(
            f"vllm preflight OK: {record.input_tokens}+{record.output_tokens} tokens, "
            f"{record.latency_ms:.0f} ms",
            flush=True,
        )

    print(f"Run directory: {run_root}", flush=True)
    for adapter, cases, harness in jobs:
        summary = run_configuration(
            adapter=adapter,
            harness=harness,
            cases=cases,
            backend=backend,
            run_root=run_root,
            workers=args.workers,
            force=args.force,
            trial_num=args.trial_num,
            stage_retries=args.stage_retries,
            smoke_per_group=args.smoke_per_group,
            seed=args.seed,
            request_workers=args.request_workers,
            sample_size=args.sample_size,
            data_dir=args.data_dir,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
