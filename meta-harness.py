#!/usr/bin/env python3
"""使用 Codex CLI 驱动 Reward-Harness 优化外循环。

每轮由 Codex 按 Skill 生成三个新 Harness；本脚本负责检查候选、调用可信
benchmark，并更新追加式演化历史和 held-in 搜索集前沿。

使用示例：
    python meta-harness.py --iterations 1 --run-name reward-search
    python meta-harness.py --iterations 5 --run-name reward-search
    python meta-harness.py --run-name reward-search --status
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import yaml


ROOT = Path(__file__).resolve().parent
SKILL_PATH = ROOT / ".claude" / "skills" / "meta-harness-reward-skill" / "SKILL.md"
DEFAULT_STATE_ROOT = ROOT / "meta_runs"
BASELINES = ("no_skill", "init_skill")
NAME_PATTERN = re.compile(r"[a-z][a-z0-9_]{2,63}")

_interrupted = False


# 每个命名实验都有独立的状态目录，复用 run-name 即可断点续跑。
@dataclass(frozen=True)
class RunPaths:
    root: Path
    pending: Path
    frontier: Path
    evolution: Path
    reports: Path
    codex_sessions: Path
    benchmark_logs: Path

    @classmethod
    def create(cls, state_root: Path, run_name: str) -> "RunPaths":
        root = state_root / run_name
        result = cls(
            root=root,
            pending=root / "pending_eval.json",
            frontier=root / "frontier_val.json",
            evolution=root / "evolution_summary.jsonl",
            reports=root / "reports",
            codex_sessions=root / "codex_sessions",
            benchmark_logs=root / "benchmark_logs",
        )
        for directory in (
            result.root,
            result.reports,
            result.codex_sessions,
            result.benchmark_logs,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        return result


def _handle_signal(signum: int, frame: Any) -> None:
    """收到中断信号后，在当前子进程结束时退出。"""

    del signum, frame
    global _interrupted
    _interrupted = True
    print("\nInterrupt requested; finishing the current subprocess.", flush=True)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _safe_tag(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    return value or "meta-harness"


def _atomic_json(path: Path, value: Any) -> None:
    """通过临时文件原子写入 JSON，避免中断留下半截状态。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", buffering=1) as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _run(
    command: list[str],
    *,
    timeout: int,
    cwd: Path = ROOT,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            cwd=str(cwd),
            input=input_text,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(
            command,
            124,
            stdout=exc.stdout or "",
            stderr=f"Timed out after {timeout}s\n{exc.stderr or ''}",
        )
    except OSError as exc:
        return subprocess.CompletedProcess(
            command,
            127,
            stdout="",
            stderr=f"{type(exc).__name__}: {exc}",
        )


def _iteration(paths: RunPaths) -> int:
    """从演化日志中读取已完成的最大迭代编号。"""

    return max(
        (int(row.get("iteration", 0)) for row in _read_jsonl(paths.evolution)),
        default=0,
    )


def _best_skill_using_harness(paths: RunPaths) -> dict[str, Any] | None:
    """从演化记录里找当前分数最高的 skill-using Harness 作为 proposer base。"""

    comparison_only = {"no_rubric", "no_skill"}
    best: dict[str, Any] | None = None
    for row in _read_jsonl(paths.evolution):
        system = row.get("system")
        if not isinstance(system, str) or system in comparison_only:
            continue
        try:
            score = float(row.get("avg_val", 0.0))
        except (TypeError, ValueError):
            continue
        if best is None or score > float(best.get("avg_val", -1.0)):
            best = {
                "harness": system,
                "avg_val": score,
                "scores": row.get("scores", {}),
                "run_tag": row.get("run_tag"),
            }
    return best


def _task_prompt(
    paths: RunPaths,
    args: argparse.Namespace,
    iteration: int,
) -> str:
    """Build a minimal runtime prompt for one evolution iteration.

    Optimization policy lives in SKILL.md.
    Experiment configuration lives in config/state files.
    This prompt only exposes the current run context and artifact locations.
    """

    held_in_directory = args.data_dir / args.held_in_dataset
    frontier = _read_json(paths.frontier, {})
    global_best = frontier.get("_best", {}) if isinstance(frontier, dict) else {}
    skill_best = _best_skill_using_harness(paths) or {
        "harness": "init_skill",
        "avg_val": None,
        "scores": {},
        "run_tag": None,
    }
    return (
        f"Run iteration {iteration} of the Reward-Harness evolution loop.\n\n"

        f"First read and follow "
        f"`{SKILL_PATH.relative_to(ROOT)}` completely. "
        f"Treat the run state and configuration files below as the source of truth "
        f"for this iteration.\n\n"

        f"## Run context\n"
        f"- Run root: `{paths.root}`\n"
        f"- Configuration: `{args.config}`\n"
        f"- Held-in search dataset: `{args.held_in_dataset}` "
        f"at `{held_in_directory}`\n"
        f"- Held-out benchmarks (do NOT inspect their data or traces during "
        f"search): {', '.join(args.held_out_benchmarks)}.\n\n"

        f"## Optimization state\n"
        f"- Evolution history: `{paths.evolution}`\n"
        f"- Current frontier: `{paths.frontier}`\n"
        f"- Current global best Harness: "
        f"`{global_best.get('harness', 'unknown')}` "
        f"at `{global_best.get('avg_val', 'unknown')}` avg_val\n"
        f"- Current best skill-using base Harness: "
        f"`{skill_best.get('harness')}` "
        f"at `{skill_best.get('avg_val')}` avg_val\n"
        f"- Post-evaluation reports: `{paths.reports}`\n\n"

        f"Use the evolution history to locate evaluation traces or summaries. "
        f"Respect all data-access and optimization constraints defined in the "
        f"skill and configuration.\n\n"

        f"## Output\n"
        f"Write the candidate proposal manifest to: `{paths.pending}`\n"
    )


def _codex_propose(
    paths: RunPaths,
    args: argparse.Namespace,
    iteration: int,
    prompt: str,
) -> subprocess.CompletedProcess[str]:
    """非交互调用 Codex CLI，并保存 JSONL 事件和最终回复。"""

    session = paths.codex_sessions / f"iteration_{iteration:03d}"
    session.mkdir(parents=True, exist_ok=True)
    last_message = session / "last_message.md"
    # codex exec 本身是非交互模式,不会请求审批;部分 CLI 版本的 exec
    # 子命令不识别 --ask-for-approval,因此不再显式传递。
    command = [
        *shlex.split(args.codex_bin, posix=os.name != "nt"),
        "exec",
        "--sandbox",
        "workspace-write",
        "--ephemeral",
        "--json",
        "-C",
        str(ROOT),
        "--output-last-message",
        str(last_message),
    ]
    if args.codex_model:
        command.extend(["--model", args.codex_model])
    if args.codex_reasoning_effort:
        command.extend(
            ["-c", f'model_reasoning_effort="{args.codex_reasoning_effort}"']
        )
    for extra in args.codex_arg:
        command.append(extra)
    command.append(prompt)

    started = time.time()
    result = _run(command, timeout=args.propose_timeout)
    (session / "events.jsonl").write_text(result.stdout or "", encoding="utf-8")
    (session / "stderr.txt").write_text(result.stderr or "", encoding="utf-8")
    _atomic_json(
        session / "meta.json",
        {
            "iteration": iteration,
            "started_at": datetime.fromtimestamp(started).isoformat(),
            "finished_at": _now(),
            "elapsed_seconds": round(time.time() - started, 3),
            "returncode": result.returncode,
            "command": command[:-1] + ["<task-prompt>"],
        },
    )
    return result


def _load_pending(paths: RunPaths) -> list[dict[str, Any]]:
    """读取 Codex 输出的候选列表；具体约束由 Skill 负责。"""

    value = _read_json(paths.pending, {})
    if not isinstance(value, dict):
        raise ValueError("pending_eval.json must contain an object")
    candidates = value.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("pending_eval.json must contain candidate objects")
    return candidates


def _check_one_candidate(candidate: dict[str, Any]) -> str | None:
    """检查候选路径、命名、冷启动导入和 RewardSystem 抽象接口。"""

    name = candidate.get("name")
    if not isinstance(name, str) or not NAME_PATTERN.fullmatch(name):
        return "invalid snake_case candidate name"
    expected = f"reward_harness/agents/{name}.py"
    if candidate.get("file") != expected:
        return f"file must be {expected}"
    path = ROOT / expected
    if not path.is_file():
        return "candidate file is missing"

    import_check_code = f"""
import importlib.util, inspect, sys
from pathlib import Path
from reward_harness.reward_system import Query, Response, RewardSystem
path = Path({str(path)!r})
module_name = 'reward_harness.agents._meta_import_check_{name}'
spec = importlib.util.spec_from_file_location(module_name, path)
module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module
spec.loader.exec_module(module)
explicit = getattr(module, 'HARNESS_CLASS', None)
classes = [explicit] if explicit is not None else [
    value for value in vars(module).values()
    if inspect.isclass(value) and value.__module__ == module_name
    and issubclass(value, RewardSystem) and value is not RewardSystem
]
assert len(classes) == 1
assert not inspect.isabstract(classes[0])
def _dummy_llm(prompt):
    return '{{"rubrics": [{{"rubric_id": "r1", "criterion": "The response should satisfy the request."}}]}}'
harness = classes[0](_dummy_llm, _dummy_llm)
task = Query(query_id="meta_check", instruction="Answer the user request.", context=None, domain="meta_check", metadata={{}})
responses = (
    Response(response_id="a", content="Answer A", metadata={{}}),
    Response(response_id="b", content="Answer B", metadata={{}}),
)
registry = harness.get_skill_registry(task)
assert registry.skills, 'candidate must expose a non-empty Skill bank'
selected = harness.retrieve_skills(task, responses, "G") + harness.retrieve_skills(task, responses, "J")
assert selected, 'candidate must retrieve at least one Skill for G or J'
print('OK')
"""
    result = _run([sys.executable, "-c", import_check_code], timeout=60)
    if result.returncode != 0 or "OK" not in result.stdout:
        detail = (result.stderr or result.stdout).strip().replace("\n", " ")
        return f"import check failed: {detail[:300]}"
    return None


def _check_candidates(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    valid = []
    names: set[str] = set()
    for candidate in candidates:
        name = candidate.get("name") if isinstance(candidate, dict) else None
        if not isinstance(candidate, dict):
            print("  INVALID candidate entry is not an object")
            continue
        error = "duplicate candidate name" if name in names else _check_one_candidate(candidate)
        if isinstance(name, str):
            names.add(name)
        if error:
            print(f"  INVALID {name}: {error}")
            continue
        print(f"  VALID {name}")
        valid.append(candidate)
    return valid


def _benchmark_command(
    args: argparse.Namespace,
    agents: list[str],
    run_tag: str,
    benchmarks: Iterable[str],
) -> list[str]:
    """构造可信 benchmark 子进程命令。"""

    command = [
        sys.executable,
        "-u",
        "-m",
        "reward_harness.benchmark",
        "--base-url",
        args.base_url,
        "--model",
        args.model,
        "--benchmarks",
        *benchmarks,
        "--agents",
        *agents,
        "--workers",
        str(args.workers),
        "--request-workers",
        str(args.request_workers),
        "--smoke-per-group",
        str(args.smoke_per_group),
        "--sample-size",
        str(args.sample_size),
        "--trial-num",
        "1",
        "--temperature",
        str(args.temperature),
        "--max-tokens",
        str(args.max_tokens),
        "--stage-retries",
        str(args.stage_retries),
        "--seed",
        str(args.seed),
        "--data-dir",
        str(args.data_dir),
        "--output-dir",
        str(args.output_dir),
        "--run-tag",
        run_tag,
    ]
    if args.skip_preflight:
        command.append("--skip-preflight")
    return command


def _model_directory(model: str) -> str:
    return _safe_tag(model.split("/")[-1])


def _summary_path(
    args: argparse.Namespace,
    run_tag: str,
    benchmark: str,
    harness: str,
) -> Path:
    expected = (
        args.output_dir
        / run_tag
        / benchmark
        / harness
        / _model_directory(args.model)
        / "summary.json"
    )
    if expected.exists():
        return expected
    candidates = list((args.output_dir / run_tag / benchmark / harness).glob("*/summary.json"))
    return candidates[0] if len(candidates) == 1 else expected


def _evaluate(
    paths: RunPaths,
    args: argparse.Namespace,
    agents: list[str],
    run_tag: str,
    benchmarks: Iterable[str],
) -> dict[str, dict[str, Any]]:
    """批量评测 Harness，并从各自 summary 中读取目标指标。"""

    benchmark_names = tuple(dict.fromkeys(benchmarks))
    if not benchmark_names:
        raise ValueError("at least one benchmark is required")
    command = _benchmark_command(args, agents, run_tag, benchmark_names)
    started = time.time()
    result = _run(command, timeout=args.benchmark_timeout)
    log_prefix = paths.benchmark_logs / run_tag
    log_prefix.with_suffix(".stdout.txt").write_text(result.stdout or "", encoding="utf-8")
    log_prefix.with_suffix(".stderr.txt").write_text(result.stderr or "", encoding="utf-8")
    _atomic_json(
        log_prefix.with_suffix(".meta.json"),
        {
            "command": command,
            "returncode": result.returncode,
            "elapsed_seconds": round(time.time() - started, 3),
        },
    )

    evaluated: dict[str, dict[str, Any]] = {}
    for agent in agents:
        domain_scores: dict[str, float] = {}
        summaries: dict[str, str] = {}
        errors = 0
        for benchmark in benchmark_names:
            path = _summary_path(args, run_tag, benchmark, agent)
            summary = _read_json(path, {})
            summaries[benchmark] = str(path)
            raw_domains = summary.get("metrics", {}).get("domain_scores", {})
            if isinstance(raw_domains, dict):
                domain_scores.update(
                    {
                        str(domain): float(score) * 100.0
                        for domain, score in raw_domains.items()
                        if isinstance(score, (int, float))
                        and not isinstance(score, bool)
                    }
                )
            counts = summary.get("counts", {}) if isinstance(summary, dict) else {}
            errors += int(counts.get("errors", 0) or 0) if isinstance(counts, dict) else 0
        evaluated[agent] = {
            "scores": domain_scores,
            "avg_val": (
                sum(domain_scores.values()) / len(domain_scores)
                if domain_scores
                else 0.0
            ),
            "num_errors": errors,
            "summary_paths": summaries,
            "benchmark_run_tags": {
                benchmark: run_tag for benchmark in benchmark_names
            },
            "run_tag": run_tag,
            "benchmark_returncode": result.returncode,
        }
    return evaluated


def _frontier_best(frontier: dict[str, Any]) -> float:
    best = frontier.get("_best", {})
    return float(best.get("avg_val", 0.0)) if isinstance(best, dict) else 0.0


def _update_frontier(
    paths: RunPaths,
    results: dict[str, dict[str, Any]],
) -> None:
    """更新各领域最优 Harness 和当前 held-in/search 聚合指标最高的 Harness。"""

    frontier = _read_json(paths.frontier, {})
    if not isinstance(frontier, dict):
        frontier = {}
    frontier.pop("benchmarks", None)
    frontier.pop("_best_held_in", None)
    domain_frontier = frontier.setdefault("domains", {})
    for harness, result in results.items():
        summary_path = next(iter(result["summary_paths"].values()), None)
        for domain, score in result["scores"].items():
            current = domain_frontier.get(domain, {})
            if not isinstance(current, dict) or score > float(current.get("score", -1)):
                domain_frontier[domain] = {
                    "harness": harness,
                    "score": score,
                    "summary_path": summary_path,
                    "run_tag": result["run_tag"],
                }
        current_best = frontier.get("_best", {})
        if not isinstance(current_best, dict) or result["avg_val"] > float(current_best.get("avg_val", -1)):
            frontier["_best"] = {
                "harness": harness,
                "avg_val": result["avg_val"],
                "domain_scores": result["scores"],
                "run_tag": result["run_tag"],
            }
    frontier["updated_at"] = _now()
    _atomic_json(paths.frontier, frontier)


def _record_results(
    paths: RunPaths,
    iteration: int,
    candidates: list[dict[str, Any]],
    results: dict[str, dict[str, Any]],
    previous_best: float,
    timing: dict[str, float],
) -> None:
    """把本轮每个候选的结果追加到 evolution_summary.jsonl。"""

    rows = []
    for index, candidate in enumerate(candidates):
        name = candidate["name"]
        result = results.get(name, {"avg_val": 0.0, "scores": {}, "num_errors": 0})
        avg = float(result.get("avg_val", 0.0))
        row = {
            "iteration": iteration,
            "system": name,
            "avg_val": round(avg, 4),
            "scores": result.get("scores", {}),
            "axis": candidate.get("axis", "?"),
            "hypothesis": candidate.get("hypothesis", ""),
            "base_harness": candidate.get("base_harness", ""),
            "components": candidate.get("components", []),
            "delta": round(avg - previous_best, 4),
            "outcome": f"{avg:.2f}% ({avg - previous_best:+.2f})",
            "num_errors": result.get("num_errors", 0),
            "run_tag": result.get("run_tag"),
            "summary_paths": result.get("summary_paths", {}),
            "benchmark_run_tags": result.get("benchmark_run_tags", {}),
        }
        if index == 0:
            row["timing_s"] = {key: round(value, 3) for key, value in timing.items()}
        rows.append(row)
    _append_jsonl(paths.evolution, rows)


def _initialize_baselines(paths: RunPaths, args: argparse.Namespace) -> None:
    """首次运行时在单一 held-in 搜索集上评测全部 baseline。"""

    if paths.frontier.exists():
        return
    if args.skip_baseline:
        raise RuntimeError("--skip-baseline requires an existing frontier_val.json")
    print(f"[{_now()}] evaluating baselines: {', '.join(args.baselines)}", flush=True)
    run_tag = _safe_tag(f"mh-{args.run_name}-baseline-search")
    results = _evaluate(
        paths,
        args,
        list(args.baselines),
        run_tag,
        args.benchmarks,
    )
    failed = [
        name
        for name, result in results.items()
        if result["benchmark_returncode"] != 0
        or any(not Path(path).exists() for path in result["summary_paths"].values())
    ]
    if failed:
        raise RuntimeError(
            f"baseline benchmark failed for {failed}; see {paths.benchmark_logs}"
        )
    _update_frontier(paths, results)
    rows = []
    for name, result in results.items():
        rows.append(
            {
                "iteration": 0,
                "system": name,
                "avg_val": round(result["avg_val"], 4),
                "scores": result["scores"],
                "axis": "baseline",
                "hypothesis": "hand-written baseline",
                "delta": None,
                "outcome": f"{result['avg_val']:.2f}%",
                "num_errors": result["num_errors"],
                "run_tag": run_tag,
                "summary_paths": result["summary_paths"],
                "benchmark_run_tags": result.get("benchmark_run_tags", {}),
            }
        )
        print(f"  {name}: {result['avg_val']:.2f}%")
    _append_jsonl(paths.evolution, rows)


def _archive_fresh_run(state_root: Path, run_name: str) -> None:
    """--fresh 使用归档替代删除，保留旧实验状态。"""

    root = state_root / run_name
    if not root.exists():
        return
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    destination = root.with_name(f"{root.name}.bak_{stamp}")
    counter = 1
    while destination.exists():
        destination = root.with_name(f"{root.name}.bak_{stamp}_{counter}")
        counter += 1
    root.replace(destination)
    print(f"Archived previous run state to {destination}")


def _print_status(paths: RunPaths) -> None:
    frontier = _read_json(paths.frontier, {})
    rows = _read_jsonl(paths.evolution)
    print(json.dumps({"run": paths.root.name, "iterations": _iteration(paths), "frontier": frontier, "history_rows": len(rows)}, ensure_ascii=False, indent=2))


def run_evolution(args: argparse.Namespace) -> int:
    """执行 baseline 初始化和指定轮数的 propose/evaluate/update 循环。"""

    state_root = args.state_root.resolve()
    if args.fresh:
        _archive_fresh_run(state_root, args.run_name)
    paths = RunPaths.create(state_root, args.run_name)
    if args.status:
        _print_status(paths)
        return 0
    if not SKILL_PATH.is_file():
        raise FileNotFoundError(f"missing proposer skill: {SKILL_PATH}")

    _initialize_baselines(paths, args)
    start = _iteration(paths) + 1
    for offset in range(args.iterations):
        if _interrupted:
            break
        iteration = start + offset
        print(f"\n[{_now()}] iteration {iteration}", flush=True)
        previous_best = _frontier_best(_read_json(paths.frontier, {}))
        if paths.pending.exists():
            paths.pending.unlink()
        propose_started = time.time()
        result = _codex_propose(
            paths,
            args,
            iteration,
            _task_prompt(paths, args, iteration),
        )
        propose_time = time.time() - propose_started
        if result.returncode != 0 or not paths.pending.exists():
            print(f"Codex proposer failed (exit={result.returncode}); see {paths.codex_sessions}")
            continue

        try:
            candidates = _load_pending(paths)
        except ValueError as exc:
            print(f"Invalid pending_eval.json: {exc}")
            continue
        valid = _check_candidates(candidates)
        if not valid:
            print("No valid candidates; iteration not benchmarked.")
            continue
        if args.propose_only:
            print("CANDIDATES: " + ", ".join(candidate["name"] for candidate in valid))
            return 0

        benchmark_started = time.time()
        run_tag = _safe_tag(f"mh-{args.run_name}-iter-{iteration:03d}-search")
        results = _evaluate(
            paths,
            args,
            [candidate["name"] for candidate in valid],
            run_tag,
            args.benchmarks,
        )
        benchmark_time = time.time() - benchmark_started
        _record_results(
            paths,
            iteration,
            valid,
            results,
            previous_best,
            {
                "propose": propose_time,
                "benchmark": benchmark_time,
                "wall": time.time() - propose_started,
            },
        )
        _update_frontier(paths, results)
        for name, evaluation in results.items():
            print(f"  {name}: {evaluation['avg_val']:.2f}%")
        best = _read_json(paths.frontier, {}).get("_best", {})
        print(f"  frontier: {best.get('harness')} @ {float(best.get('avg_val', 0)):.2f}%")

    return 0


def _load_config(path: Path) -> dict[str, Any]:
    """读取 YAML，并要求顶层及四个配置分组都是对象。"""

    if not path.is_file():
        raise FileNotFoundError(f"Meta-Harness config does not exist: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError("Meta-Harness config root must be a mapping")
    for section in ("run", "codex", "benchmark", "paths", "datasets"):
        section_value = value.get(section, {})
        if not isinstance(section_value, dict):
            raise ValueError(f"config section {section!r} must be a mapping")
    return value


def build_parser(
    config: dict[str, Any] | None = None,
    config_path: Path = ROOT / "config.yaml",
) -> argparse.ArgumentParser:
    # help 文本保持 ASCII，兼容服务器以外仍使用旧代码页的终端。
    config = config or {}
    run_config = config.get("run", {})
    codex_config = config.get("codex", {})
    benchmark_config = config.get("benchmark", {})
    path_config = config.get("paths", {})
    dataset_config = config.get("datasets", {})
    parser = argparse.ArgumentParser(
        description="Codex CLI outer loop for Reward-Harness optimization."
    )
    parser.add_argument("--config", type=Path, default=config_path)
    parser.add_argument("--iterations", type=int, default=run_config.get("iterations", 1))
    parser.add_argument("--run-name", default=run_config.get("name", "reward-search"))
    parser.add_argument(
        "--state-root",
        type=Path,
        default=Path(run_config.get("state_root", DEFAULT_STATE_ROOT)),
    )
    parser.add_argument("--fresh", action="store_true", help="Archive existing run state and start fresh")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--propose-only", action="store_true")

    parser.add_argument("--codex-bin", default=codex_config.get("bin", "codex"))
    parser.add_argument("--codex-model", default=codex_config.get("model"))
    parser.add_argument(
        "--codex-reasoning-effort",
        choices=("minimal", "low", "medium", "high", "xhigh"),
        default=codex_config.get("reasoning_effort", "medium"),
    )
    parser.add_argument(
        "--codex-arg",
        action="append",
        default=list(codex_config.get("args", [])),
    )
    parser.add_argument(
        "--propose-timeout",
        type=int,
        default=codex_config.get("propose_timeout", 3600),
    )
    parser.add_argument(
        "--benchmark-timeout",
        type=int,
        default=benchmark_config.get("benchmark_timeout", 86400),
    )

    parser.add_argument(
        "--base-url",
        default=os.getenv(
            "VLLM_BASE_URL",
            benchmark_config.get("base_url", "http://127.0.0.1:8000/v1"),
        ),
    )
    parser.add_argument("--model", default=benchmark_config.get("model", "Qwen/Qwen3-8B"))
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=list(benchmark_config.get("benchmarks", ["held_in"])),
    )
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=list(benchmark_config.get("baselines", BASELINES)),
    )
    parser.add_argument("--temperature", type=float, default=benchmark_config.get("temperature", 0.7))
    parser.add_argument("--max-tokens", type=int, default=benchmark_config.get("max_tokens", 10000))
    parser.add_argument("--workers", type=int, default=benchmark_config.get("workers", 32))
    parser.add_argument(
        "--request-workers",
        type=int,
        default=benchmark_config.get("request_workers", 32),
    )
    parser.add_argument(
        "--smoke-per-group",
        type=int,
        default=benchmark_config.get("smoke_per_group", 2),
    )
    parser.add_argument("--sample-size", type=int, default=benchmark_config.get("sample_size", 0))
    parser.add_argument("--seed", type=int, default=benchmark_config.get("seed", 42))
    parser.add_argument(
        "--stage-retries",
        type=int,
        default=benchmark_config.get("stage_retries", 2),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(path_config.get("data_dir", ROOT / "data")),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(path_config.get("output_dir", ROOT / "results")),
    )
    parser.add_argument(
        "--held-in-dataset",
        default=str(dataset_config.get("held_in", "held_in")),
    )
    parser.add_argument(
        "--held-out-benchmarks",
        nargs="+",
        default=list(
            dataset_config.get(
                "held_out", ["rewardbench", "rewardbench2", "rmbench"]
            )
        ),
    )
    parser.add_argument(
        "--skip-preflight",
        action=argparse.BooleanOptionalAction,
        default=bool(benchmark_config.get("skip_preflight", False)),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    # 先只解析配置路径，再用 YAML 中的值构建完整 CLI 默认值。
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    known, _ = bootstrap.parse_known_args(argv)
    config_path = known.config
    if not config_path.is_absolute():
        config_path = (ROOT / config_path).resolve()
    config = _load_config(config_path)
    args = build_parser(config, config_path).parse_args(argv)
    if args.iterations < 1 and not args.status:
        raise SystemExit("iterations must be >= 1")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.run_name):
        raise SystemExit("run-name may contain only letters, digits, dot, underscore and dash")
    for attribute in ("state_root", "data_dir", "output_dir"):
        path = getattr(args, attribute)
        if not path.is_absolute():
            setattr(args, attribute, (ROOT / path).resolve())
    for name, value in (
        ("workers", args.workers),
        ("request-workers", args.request_workers),
        ("max-tokens", args.max_tokens),
    ):
        if value < 1:
            raise SystemExit(f"{name} must be >= 1")
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    return run_evolution(args)


if __name__ == "__main__":
    raise SystemExit(main())
