"""在评测/训练前下载并生成统一格式的本地 benchmark 数据。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Callable

from .benchmarks.base import DEFAULT_DATA_ROOT, BenchmarkCase, write_processed_cases
from .benchmarks.rewardbench import RewardBenchAdapter
from .benchmarks.rewardbench2 import RewardBench2Adapter
from .benchmarks.rmbench import RMBenchAdapter


def _load_huggingface_rows(
    dataset_id: str,
    split: str,
    *,
    offline: bool,
) -> tuple[list[dict[str, Any]], str | None]:
    """显式准备阶段使用 Hugging Face；正式 benchmark 不会调用此函数。"""

    if offline:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "The datasets package is required; install reward_agent/requirements.txt"
        ) from exc
    dataset = load_dataset(dataset_id, split=split)
    fingerprint = getattr(dataset, "_fingerprint", None)
    return [dict(row) for row in dataset], str(fingerprint) if fingerprint else None


def _prepare_rewardbench(
    data_root: Path, *, force: bool, offline: bool
) -> tuple[Path, Path]:
    adapter = RewardBenchAdapter(data_root=data_root)
    rows, fingerprint = _load_huggingface_rows(
        adapter.dataset_id, adapter.split, offline=offline
    )
    cases = [adapter._convert(row, index) for index, row in enumerate(rows)]
    return write_processed_cases(
        data_root,
        benchmark=adapter.name,
        dataset_id=adapter.dataset_id,
        split=adapter.split,
        cases=cases,
        source_fingerprint=fingerprint,
        force=force,
    )


def _prepare_rewardbench2(
    data_root: Path, *, force: bool, offline: bool
) -> tuple[Path, Path]:
    adapter = RewardBench2Adapter(data_root=data_root)
    rows, fingerprint = _load_huggingface_rows(
        adapter.dataset_id, adapter.split, offline=offline
    )
    cases = [adapter._convert(row) for row in rows]
    return write_processed_cases(
        data_root,
        benchmark=adapter.name,
        dataset_id=adapter.dataset_id,
        split=adapter.split,
        cases=cases,
        source_fingerprint=fingerprint,
        force=force,
    )


def _prepare_rmbench(
    data_root: Path, *, force: bool, offline: bool
) -> tuple[Path, Path]:
    adapter = RMBenchAdapter(data_root=data_root)
    rows, fingerprint = _load_huggingface_rows(
        adapter.dataset_id, adapter.split, offline=offline
    )
    cases = [adapter._convert(row, index) for index, row in enumerate(rows)]
    return write_processed_cases(
        data_root,
        benchmark=adapter.name,
        dataset_id=adapter.dataset_id,
        split=adapter.split,
        cases=cases,
        source_fingerprint=fingerprint,
        force=force,
    )


PREPARERS: dict[
    str,
    Callable[..., tuple[Path, Path]],
] = {
    "rewardbench": _prepare_rewardbench,
    "rewardbench2": _prepare_rewardbench2,
    "rmbench": _prepare_rmbench,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare local normalized data for reward-agent benchmarks."
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=sorted(PREPARERS),
        default=sorted(PREPARERS),
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use existing Hugging Face/raw caches only; never access the network.",
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data_root = args.data_dir.resolve()
    for name in args.benchmarks:
        print(f"Preparing {name}...", flush=True)
        data_path, manifest_path = PREPARERS[name](
            data_root,
            force=args.force,
            offline=args.offline,
        )
        print(f"  data: {data_path}", flush=True)
        print(f"  manifest: {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
