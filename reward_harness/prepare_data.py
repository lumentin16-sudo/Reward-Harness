"""在评测/训练前下载并生成统一格式的本地 benchmark 数据。"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import random
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterable

from .benchmarks.base import (
    DEFAULT_DATA_ROOT,
    BenchmarkCase,
    public_text,
    write_processed_cases,
)
from .reward_system import Query, Response
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


HELPSTEER3_DATASET_ID = "nvidia/HelpSteer3"
HELPSTEER3_REVISION = "f6d145777bcbde96137596340fab89793acd1031"
HELPSTEER3_DOMAINS = ("general", "stem", "code", "multilingual")
HELPSTEER3_TRAIN_URL = (
    "https://huggingface.co/datasets/nvidia/HelpSteer3/resolve/"
    f"{HELPSTEER3_REVISION}/"
    "preference/train.jsonl.gz"
)


def _iter_helpsteer3_rows(
    *, offline: bool
) -> tuple[Iterable[dict[str, Any]], str | None]:
    """流式读取 HelpSteer3 Preference train，避免加载约 373MB 解压数据。"""

    if offline:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError(
                "Offline HelpSteer3 preparation requires the datasets package"
            ) from exc
        dataset = load_dataset(
            HELPSTEER3_DATASET_ID,
            "preference",
            split="train",
        )
        fingerprint = getattr(dataset, "_fingerprint", None)
        return (dict(row) for row in dataset), (
            str(fingerprint) if fingerprint else None
        )

    def remote_rows() -> Iterable[dict[str, Any]]:
        request = urllib.request.Request(
            HELPSTEER3_TRAIN_URL,
            headers={"User-Agent": "Reward-Harness/1.0"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            with gzip.GzipFile(fileobj=response) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8") as stream:
                    for line_number, line in enumerate(stream, 1):
                        if not line.strip():
                            continue
                        value = json.loads(line)
                        if not isinstance(value, dict):
                            raise ValueError(
                                f"HelpSteer3 row {line_number} is not an object"
                            )
                        yield value

    return remote_rows(), HELPSTEER3_REVISION


def _helpsteer3_case(row: dict[str, Any], source_index: int) -> BenchmarkCase:
    domain = str(row["domain"]).strip().lower()
    preference = int(row["overall_preference"])
    if preference == 0:
        raise ValueError("HelpSteer3 tie rows cannot produce a unique winner")
    response1 = public_text(row["response1"])
    response2 = public_text(row["response2"])
    chosen, rejected = (
        (response1, response2) if preference < 0 else (response2, response1)
    )
    case_id = f"helpsteer3:{domain}:{source_index:06d}"
    return BenchmarkCase(
        case_id=case_id,
        group=domain,
        task=Query(
            query_id=case_id,
            instruction=public_text(row["context"]),
            domain=domain,
            metadata={"language": str(row.get("language", ""))},
        ),
        candidates=(
            Response("candidate_000", chosen),
            Response("candidate_001", rejected),
        ),
        gold={
            "chosen_index": 0,
            "domain": domain,
            "language": str(row.get("language", "")),
            "source_index": source_index,
            "overall_preference": preference,
            "preference_strength": abs(preference),
            "preferred_source_response": 1 if preference < 0 else 2,
            "individual_preference": list(row.get("individual_preference") or []),
        },
    )


def _prepare_helpsteer3(
    data_root: Path,
    *,
    force: bool,
    offline: bool,
    held_in_size: int = 500,
    helpsteer3_seed: int = 42,
    **_: Any,
) -> tuple[Path, Path]:
    """从四个 domain 均衡抽取单一 held-in 搜索集。"""

    if held_in_size <= 0 or held_in_size % 4:
        raise ValueError("HelpSteer3 held-in size must be positive and divisible by 4")
    held_per_domain = held_in_size // 4

    rows, fingerprint = _iter_helpsteer3_rows(offline=offline)
    reservoirs: dict[str, list[tuple[int, dict[str, Any]]]] = {
        domain: [] for domain in HELPSTEER3_DOMAINS
    }
    seen = {domain: 0 for domain in HELPSTEER3_DOMAINS}
    seen_contents: set[tuple[str, str, str]] = set()
    rngs = {
        domain: random.Random(f"{helpsteer3_seed}:{domain}")
        for domain in HELPSTEER3_DOMAINS
    }
    for source_index, row in enumerate(rows):
        domain = str(row.get("domain", "")).strip().lower()
        if (
            domain not in reservoirs
            or int(row.get("overall_preference", 0)) == 0
            or not public_text(row.get("context", "")).strip()
            or not public_text(row.get("response1", "")).strip()
            or not public_text(row.get("response2", "")).strip()
        ):
            continue
        content_key = (
            public_text(row["context"]),
            public_text(row["response1"]),
            public_text(row["response2"]),
        )
        if content_key in seen_contents:
            continue
        seen_contents.add(content_key)
        seen[domain] += 1
        reservoir = reservoirs[domain]
        item = (source_index, row)
        if len(reservoir) < held_per_domain:
            reservoir.append(item)
            continue
        replacement = rngs[domain].randrange(seen[domain])
        if replacement < held_per_domain:
            reservoir[replacement] = item

    held_in_cases: list[BenchmarkCase] = []
    for domain in HELPSTEER3_DOMAINS:
        selected = reservoirs[domain]
        if len(selected) != held_per_domain:
            raise ValueError(
                f"HelpSteer3 domain {domain!r} has only {len(selected)} usable rows"
            )
        rngs[domain].shuffle(selected)
        held_in_cases.extend(
            _helpsteer3_case(row, index)
            for index, row in selected
        )

    random.Random(helpsteer3_seed).shuffle(held_in_cases)
    return write_processed_cases(
        data_root,
        benchmark="held_in",
        dataset_id=HELPSTEER3_DATASET_ID,
        split="train",
        cases=held_in_cases,
        source_fingerprint=fingerprint,
        force=force,
    )


PREPARERS: dict[str, Callable[..., Any]] = {
    "helpsteer3": _prepare_helpsteer3,
    "rewardbench": _prepare_rewardbench,
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
    parser.add_argument("--held-in-size", type=int, default=500)
    parser.add_argument("--helpsteer3-seed", type=int, default=42)
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
        prepare_kwargs = {
            "force": args.force,
            "offline": args.offline,
        }
        if name == "helpsteer3":
            prepare_kwargs.update(
                held_in_size=args.held_in_size,
                helpsteer3_seed=args.helpsteer3_seed,
            )
        prepared = PREPARERS[name](data_root, **prepare_kwargs)
        outputs = prepared if isinstance(prepared, list) else [prepared]
        for data_path, manifest_path in outputs:
            print(f"  data: {data_path}", flush=True)
            print(f"  manifest: {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
