"""自动发现 ``agents/`` 目录中的 RewardSystem 候选实现。"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import sys
from dataclasses import dataclass
from collections.abc import Collection
from pathlib import Path

from .reward_system import RewardSystem


DEFAULT_AGENTS_DIR = Path(__file__).resolve().parent / "agents"


@dataclass(frozen=True, slots=True)
class HarnessSpec:
    """一个从独立 Python 文件发现的可评测 Reward Harness。"""

    name: str
    harness_type: type[RewardSystem]
    source_path: Path
    source_sha256: str


def discover_harnesses(
    agents_dir: Path = DEFAULT_AGENTS_DIR,
    include_names: Collection[str] | None = None,
) -> list[HarnessSpec]:
    """扫描每个 ``*.py``，要求文件内恰好定义一个 RewardSystem 子类。"""

    directory = agents_dir.resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"agents directory does not exist: {directory}")

    discovered: list[HarnessSpec] = []
    for path in sorted(directory.glob("*.py")):
        if path.stem == "__init__" or path.stem.startswith("_"):
            continue
        if include_names is not None and path.stem not in include_names:
            continue
        source = path.read_bytes()
        digest = hashlib.sha256(source).hexdigest()
        module_name = (
            f"{__package__}.agents._discovered_{path.stem}_{digest[:12]}"
        )
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load agent module: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise

        explicit = getattr(module, "HARNESS_CLASS", None)
        if explicit is not None:
            candidates = [explicit]
        else:
            candidates = [
                value
                for value in vars(module).values()
                if inspect.isclass(value)
                and value.__module__ == module_name
                and issubclass(value, RewardSystem)
                and value is not RewardSystem
            ]
        if len(candidates) != 1:
            raise ValueError(
                f"agent file must define exactly one RewardSystem subclass or "
                f"HARNESS_CLASS: {path}; found {len(candidates)}"
            )
        harness_type = candidates[0]
        if not inspect.isclass(harness_type) or not issubclass(
            harness_type, RewardSystem
        ):
            raise TypeError(f"HARNESS_CLASS is not a RewardSystem subclass: {path}")
        name = str(getattr(module, "HARNESS_NAME", path.stem))
        if not name or any(char in name for char in "/\\"):
            raise ValueError(f"invalid HARNESS_NAME in {path}: {name!r}")
        discovered.append(HarnessSpec(name, harness_type, path, digest))

    names = [item.name for item in discovered]
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate discovered harness names: {names}")
    if not discovered:
        raise ValueError(f"no RewardSystem agents found in {directory}")
    return discovered
