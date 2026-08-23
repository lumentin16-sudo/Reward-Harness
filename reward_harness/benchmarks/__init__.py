"""Reward Harness benchmark adapters。"""

from .base import BenchmarkAdapter, BenchmarkCase
from .helpsteer3 import HeldInAdapter, ValidationAdapter
from .rewardbench import RewardBenchAdapter
from .rewardbench2 import RewardBench2Adapter
from .rmbench import RMBenchAdapter


ADAPTERS: dict[str, type[BenchmarkAdapter]] = {
    HeldInAdapter.name: HeldInAdapter,
    RewardBenchAdapter.name: RewardBenchAdapter,
    RMBenchAdapter.name: RMBenchAdapter,
    ValidationAdapter.name: ValidationAdapter,
}

__all__ = [
    "ADAPTERS",
    "BenchmarkAdapter",
    "BenchmarkCase",
    "HeldInAdapter",
    "RewardBenchAdapter",
    "RewardBench2Adapter",
    "RMBenchAdapter",
    "ValidationAdapter",
]
