"""Benchmark adapter 的最小稳定协议。"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ..reward_system import Response, JSONValue, Query


SCHEMA_VERSION = 2
DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[2] / "data"


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    """模型可见输入与 evaluator-only gold labels 的隔离容器。"""

    case_id: str
    group: str
    task: Query
    candidates: tuple[Response, ...]
    gold: Mapping[str, JSONValue] = field(default_factory=dict)


class BenchmarkAdapter(ABC):
    """新数据集只需实现加载、单条计分和汇总三个步骤。"""

    name: str
    dataset_id: str
    split: str

    def __init__(self, *, data_root: Path | None = None) -> None:
        self.data_root = (data_root or DEFAULT_DATA_ROOT).resolve()

    def load_processed_cases(self) -> list[BenchmarkCase]:
        """只读取预先生成的本地标准化数据，评测阶段绝不隐式联网。"""

        return read_processed_cases(
            self.data_root,
            benchmark=self.name,
            split=self.split,
        )

    @abstractmethod
    def load_cases(
        self,
        *,
        smoke_per_group: int,
        seed: int,
    ) -> list[BenchmarkCase]:
        """加载并转换数据；``smoke_per_group=0`` 表示全量。"""

    @abstractmethod
    def score_outcome(self, outcome: dict[str, Any]) -> dict[str, Any]:
        """为单条成功或失败的执行结果添加数据集特定贡献。"""

    @abstractmethod
    def summarize(self, outcomes: list[dict[str, Any]]) -> dict[str, Any]:
        """按官方规则汇总当前结果文件中的全部样本。"""


def processed_paths(
    data_root: Path,
    *,
    benchmark: str,
    split: str,
) -> tuple[Path, Path]:
    """返回标准化 JSONL 和对应 manifest 的稳定路径。"""

    directory = data_root / benchmark
    return directory / f"{split}.jsonl", directory / f"{split}.manifest.json"


def benchmark_case_to_dict(case: BenchmarkCase) -> dict[str, Any]:
    """把统一 case 转换为带 schema 版本的可持久化对象。"""

    return {
        "schema_version": SCHEMA_VERSION,
        "case_id": case.case_id,
        "group": case.group,
        "task": asdict(case.task),
        "candidates": [asdict(candidate) for candidate in case.candidates],
        "gold": dict(case.gold),
    }


def benchmark_case_from_dict(value: Mapping[str, Any]) -> BenchmarkCase:
    """从标准化对象恢复统一 case，并重新执行 RewardSystem 数据校验。"""

    version = value.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported benchmark data schema_version={version!r}; "
            f"expected {SCHEMA_VERSION}"
        )
    task_value = value["task"]
    candidate_values = value["candidates"]
    if not isinstance(task_value, Mapping) or not isinstance(candidate_values, list):
        raise ValueError("processed benchmark case has invalid task or candidates")
    return BenchmarkCase(
        case_id=str(value["case_id"]),
        group=str(value["group"]),
        task=Query(**dict(task_value)),
        candidates=tuple(Response(**dict(item)) for item in candidate_values),
        gold=dict(value.get("gold", {})),
    )


def write_processed_cases(
    data_root: Path,
    *,
    benchmark: str,
    dataset_id: str,
    split: str,
    cases: list[BenchmarkCase],
    source_fingerprint: str | None = None,
    force: bool = False,
) -> tuple[Path, Path]:
    """原子写入标准化 JSONL 和可复现 manifest。"""

    data_path, manifest_path = processed_paths(
        data_root.resolve(), benchmark=benchmark, split=split
    )
    if (data_path.exists() or manifest_path.exists()) and not force:
        raise FileExistsError(
            f"processed data already exists: {data_path}; pass --force to replace it"
        )
    data_path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(benchmark_case_to_dict(case), ensure_ascii=False) + "\n"
        for case in cases
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": benchmark,
        "dataset_id": dataset_id,
        "split": split,
        "num_cases": len(cases),
        "content_sha256": digest,
        "source_fingerprint": source_fingerprint,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    data_tmp = data_path.with_suffix(data_path.suffix + ".tmp")
    manifest_tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    data_tmp.write_bytes(payload)
    manifest_tmp.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    data_tmp.replace(data_path)
    manifest_tmp.replace(manifest_path)
    return data_path, manifest_path


def read_processed_cases(
    data_root: Path,
    *,
    benchmark: str,
    split: str,
) -> list[BenchmarkCase]:
    """读取并校验本地标准化数据及其 manifest。"""

    data_path, manifest_path = processed_paths(
        data_root.resolve(), benchmark=benchmark, split=split
    )
    if not data_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(
            f"processed data is missing for {benchmark}/{split}: {data_path}. "
            f"Run: python -m reward_harness.prepare_data --benchmarks {benchmark}"
        )
    payload = data_path.read_bytes()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(payload).hexdigest()
    if manifest.get("content_sha256") != digest:
        raise ValueError(f"processed benchmark checksum mismatch: {data_path}")

    cases: list[BenchmarkCase] = []
    # JSONL 的记录分隔符只认 LF。str.splitlines() 还会把 U+2028、U+0085 等
    # 合法的 JSON 字符串内容当成换行，RewardBench 2 中确实包含这类字符。
    for line_number, line in enumerate(payload.decode("utf-8").split("\n"), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("record is not an object")
            cases.append(benchmark_case_from_dict(value))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"invalid processed benchmark record {data_path}:{line_number}: {exc}"
            ) from exc
    if manifest.get("num_cases") != len(cases):
        raise ValueError(
            f"processed benchmark count mismatch: manifest={manifest.get('num_cases')}, "
            f"actual={len(cases)}"
        )
    return cases


def public_text(value: Any) -> str:
    """把字符串或 chat messages 规范成 Query/Response 的公开文本。"""

    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict) and "content" in item:
                role = str(item.get("role", "message"))
                parts.append(f"[{role}] {item['content']}")
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(value)
