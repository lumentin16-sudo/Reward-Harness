"""本地 vLLM 上的 RewardBench、RewardBench 2 与 RM-Bench 评测 CLI。"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import random
import re
import sys
import time
from concurrent.futures import Executor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable

from .agent_loader import DEFAULT_AGENTS_DIR, HarnessSpec, discover_harnesses
from .benchmarks import ADAPTERS, BenchmarkAdapter, BenchmarkCase
from .benchmarks.base import DEFAULT_DATA_ROOT, processed_paths
from .model_client import RecordingLLM, VLLMBackend
from .reward_system import RewardResult, RewardSystem, RubricSet


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _slug(value: str) -> str:
    """把模型/agent 名称转换成稳定的目录名。"""

    normalized = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-.")
    return normalized.lower() or "unnamed"


def _sha256_files(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted({item.resolve() for item in paths}, key=str):
        digest.update(str(path).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _run_signature(
    *,
    adapter: BenchmarkAdapter,
    cases: list[BenchmarkCase],
    harness: HarnessSpec,
    base_url: str,
    model: str,
    smoke_per_group: int,
    sample_size: int,
    seed: int,
) -> str:
    """绑定数据、agent、evaluator、模型和抽样配置，避免错误复用旧结果。"""

    _, manifest_path = processed_paths(
        adapter.data_root, benchmark=adapter.name, split=adapter.split
    )
    evaluator_sha = _sha256_files(
        [
            Path(__file__),
            Path(inspect.getfile(type(adapter))),
            Path(inspect.getfile(RewardSystem)),
            Path(inspect.getfile(VLLMBackend)),
        ]
    )
    payload = {
        "benchmark": adapter.name,
        "dataset_manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        "case_ids_sha256": hashlib.sha256(
            "\n".join(case.case_id for case in cases).encode("utf-8")
        ).hexdigest(),
        "agent_name": harness.name,
        "agent_source_sha256": harness.source_sha256,
        "evaluator_sha256": evaluator_sha,
        "backend": "vllm",
        "base_url": base_url.rstrip("/"),
        "model": model,
        "temperature": 0.0,
        "max_tokens": 2048,
        "enable_thinking": False,
        "smoke_per_group": smoke_per_group,
        "sample_size": sample_size,
        "seed": seed,
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _run_directories(
    *,
    logs_root: Path,
    results_root: Path,
    benchmark: str,
    harness: str,
    model: str,
    run_signature: str,
) -> tuple[Path, Path]:
    leaf = f"{_slug(model.split('/')[-1])}_{run_signature[:12]}"
    relative = Path(benchmark) / harness / leaf
    return logs_root / relative, results_root / relative


def _usable_summary(
    path: Path,
    *,
    run_signature: str,
    expected_cases: int,
) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        isinstance(value, dict)
        and value.get("status") == "complete"
        and value.get("run_signature") == run_signature
        and value.get("num_cases") == expected_cases
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
    score_executor: Executor | None = None,
) -> dict[str, Any]:
    """可信 evaluator：一题一次 Rubric，并行候选复用同一 RubricSet。"""

    harness = harness_type(rubric_llm, judge_llm)
    outcome: dict[str, Any] = {
        "case_id": case.case_id,
        "group": case.group,
        "task": _jsonable(case.task),
        "candidates": _jsonable(case.candidates),
        "gold": _jsonable(case.gold),
        "rubrics": None,
        "results": [],
        "error": None,
    }

    def build() -> RubricSet:
        rubrics = harness.build_rubrics(case.task)
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

    def score_candidate(candidate) -> tuple[RewardResult | None, str | None]:
        def score() -> RewardResult:
            result = harness.score(case.task, candidate, rubrics)
            RewardSystem._validate_reward_result(case.task, candidate, rubrics, result)
            return result

        return _attempt(
            score,
            stage_retries,
            on_retry=getattr(judge_llm, "invalidate_last", None),
        )

    if score_executor is None:
        scored = [score_candidate(candidate) for candidate in case.candidates]
    else:
        # 同题候选彼此独立，只共享不可变 RubricSet；保持提交顺序以稳定结果顺序。
        futures = [
            score_executor.submit(score_candidate, candidate)
            for candidate in case.candidates
        ]
        scored = [future.result() for future in futures]

    for candidate, (result, error) in zip(case.candidates, scored):
        if error or result is None:
            if outcome["error"] is None:
                outcome["error"] = {
                    "stage": "score",
                    "candidate_id": candidate.candidate_id,
                    "message": error,
                }
        else:
            outcome["results"].append(_jsonable(result))
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

    if sample_size <= 0 or sample_size >= len(cases):
        return cases
    return random.Random(seed).sample(cases, sample_size)


def _run_one_case(
    case: BenchmarkCase,
    adapter: BenchmarkAdapter,
    harness_type: type[RewardSystem],
    backend: VLLMBackend,
    stage_retries: int,
    score_executor: Executor | None = None,
) -> dict[str, Any]:
    rubric = backend.recorder("rubric")
    judge = backend.recorder("judge")
    outcome = evaluate_case(
        case,
        harness_type,
        rubric,
        judge,
        stage_retries=stage_retries,
        score_executor=score_executor,
    )
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
    logs_root: Path,
    results_root: Path,
    run_signature: str,
    workers: int,
    force: bool,
    stage_retries: int,
    smoke_per_group: int,
    seed: int,
    request_workers: int = 16,
    sample_size: int = 0,
    data_dir: Path | None = None,
) -> dict[str, Any]:
    log_directory, result_directory = _run_directories(
        logs_root=logs_root,
        results_root=results_root,
        benchmark=adapter.name,
        harness=harness.name,
        model=backend.model,
        run_signature=run_signature,
    )
    summary_path = result_directory / "summary.json"
    if not force:
        completed_summary = _usable_summary(
            summary_path,
            run_signature=run_signature,
            expected_cases=len(cases),
        )
        if completed_summary is not None:
            print(
                f"[{adapter.name}/{harness.name}] SKIP already evaluated: "
                f"{summary_path}",
                flush=True,
            )
            return completed_summary

    log_directory.mkdir(parents=True, exist_ok=True)
    result_directory.mkdir(parents=True, exist_ok=True)
    trajectories_path = log_directory / "trajectories.jsonl"
    raw_existing = [] if force else _read_jsonl(trajectories_path)
    target_ids = {case.case_id for case in cases}
    # 同一个 case 若因历史中断留下重复行，以最后一条完整记录为准。
    existing_by_id = {
        str(row["case_id"]): row
        for row in raw_existing
        if str(row.get("case_id")) in target_ids
    }
    existing = list(existing_by_id.values())
    if force or not trajectories_path.exists():
        trajectories_path.write_text("", encoding="utf-8")
    completed = set(existing_by_id)
    pending = [case for case in cases if case.case_id not in completed]

    config = {
        "status": "running",
        "run_signature": run_signature,
        "benchmark": adapter.name,
        "dataset_id": adapter.dataset_id,
        "split": adapter.split,
        "agent": harness.name,
        "agent_file": str(harness.source_path),
        "agent_source_sha256": harness.source_sha256,
        "backend": "vllm",
        "base_url": getattr(backend, "base_url", None),
        "model": backend.model,
        "temperature": backend.temperature,
        "max_tokens": backend.max_tokens,
        "enable_thinking": False,
        "request_retries": backend.request_retries,
        "stage_retries": stage_retries,
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
    _write_json(log_directory / "config.json", config)
    _write_json(result_directory / "config.json", config)

    with trajectories_path.open("a", encoding="utf-8", buffering=1) as stream:
        with (
            ThreadPoolExecutor(max_workers=request_workers) as score_executor,
            ThreadPoolExecutor(max_workers=workers) as executor,
        ):
            futures = {
                executor.submit(
                    _run_one_case,
                    case,
                    adapter,
                    harness.harness_type,
                    backend,
                    stage_retries,
                    score_executor,
                ): case.case_id
                for case in pending
            }
            for future in as_completed(futures):
                case_id = futures[future]
                try:
                    row = future.result()
                except Exception as exc:  # evaluator 自身异常也不得终止其余样本。
                    case = next(item for item in pending if item.case_id == case_id)
                    row = adapter.score_outcome({
                        "case_id": case_id,
                        "group": case.group,
                        "task": _jsonable(case.task),
                        "candidates": _jsonable(case.candidates),
                        "gold": _jsonable(case.gold),
                        "rubrics": None,
                        "results": [],
                        "model_calls": [],
                        "usage": {"input_tokens": 0, "output_tokens": 0, "latency_ms": 0.0},
                        "error": {"stage": "evaluator", "message": f"{type(exc).__name__}: {exc}"},
                    })
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
                    f"[{adapter.name}/{harness.name}] {status} {case_id}{message}",
                    flush=True,
                )

    summary = adapter.summarize(existing)
    summary["agent"] = harness.name
    summary["status"] = "complete"
    summary["run_signature"] = run_signature
    summary["trajectory_path"] = str(trajectories_path.resolve())
    summary["usage"] = {
        key: sum(float(row.get("usage", {}).get(key, 0)) for row in existing)
        for key in ("input_tokens", "output_tokens", "latency_ms")
    }
    config["status"] = "complete"
    _write_json(log_directory / "config.json", config)
    _write_json(result_directory / "config.json", config)
    _write_json(summary_path, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    # 保持帮助文本为 ASCII，兼容 Windows 上仍使用 cp1252 的终端。
    parser = argparse.ArgumentParser(
        description="Run Qwen3-8B reward harness benchmarks."
    )
    parser.add_argument("--benchmarks", nargs="+", choices=sorted(ADAPTERS), default=["rewardbench2", "rmbench"])
    parser.add_argument(
        "--agents",
        "--harnesses",
        dest="harnesses",
        nargs="+",
        default=None,
        help=(
            "Agent file stems to evaluate; default discovers every agents/*.py file. "
            "--harnesses is kept as a compatibility alias."
        ),
    )
    parser.add_argument("--agents-dir", type=Path, default=DEFAULT_AGENTS_DIR)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--request-workers", type=int, default=16)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--smoke-per-group", type=int, default=2)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="Globally sample N cases after adapter loading; 0 keeps all cases.",
    )
    parser.add_argument("--logs-dir", type=Path, default=Path("logs/reward_agent"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/reward_agent"))
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Directory containing pre-generated normalized benchmark data.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Deprecated compatibility flag; resume and completed-run skipping are automatic.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-evaluate even when an identical completed run already exists.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage-retries", type=int, default=2)
    parser.add_argument("--skip-preflight", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if (
        args.workers < 1
        or args.request_workers < 1
        or args.smoke_per_group < 0
        or args.sample_size < 0
        or args.stage_retries < 0
    ):
        raise SystemExit(
            "workers and request-workers must be >= 1; "
            "smoke-per-group, sample-size and stage-retries must be >= 0"
        )

    try:
        discovered = discover_harnesses(args.agents_dir)
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

    jobs: list[tuple[BenchmarkAdapter, list[BenchmarkCase], HarnessSpec, str]] = []
    for adapter, cases in prepared:
        for harness in harnesses:
            signature = _run_signature(
                adapter=adapter,
                cases=cases,
                harness=harness,
                base_url=base_url,
                model=args.model,
                smoke_per_group=args.smoke_per_group,
                sample_size=args.sample_size,
                seed=args.seed,
            )
            jobs.append((adapter, cases, harness, signature))

    if not args.force:
        incomplete_jobs = []
        for adapter, cases, harness, signature in jobs:
            _, result_directory = _run_directories(
                logs_root=args.logs_dir,
                results_root=args.results_dir,
                benchmark=adapter.name,
                harness=harness.name,
                model=args.model,
                run_signature=signature,
            )
            if _usable_summary(
                result_directory / "summary.json",
                run_signature=signature,
                expected_cases=len(cases),
            ) is None:
                incomplete_jobs.append((adapter, cases, harness, signature))
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
        request_workers=args.request_workers,
        cache_dir=args.logs_dir / ".llm_cache",
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

    for adapter, cases, harness, signature in jobs:
        summary = run_configuration(
            adapter=adapter,
            harness=harness,
            cases=cases,
            backend=backend,
            logs_root=args.logs_dir,
            results_root=args.results_dir,
            run_signature=signature,
            workers=args.workers,
            force=args.force,
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
