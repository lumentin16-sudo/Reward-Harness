"""Reward Harness benchmark adapters。"""

from .base import BenchmarkAdapter, BenchmarkCase
from .rewardbench import RewardBenchAdapter
from .rewardbench2 import RewardBench2Adapter
from .rmbench import RMBenchAdapter


ADAPTERS: dict[str, type[BenchmarkAdapter]] = {
    RewardBenchAdapter.name: RewardBenchAdapter,
    RewardBench2Adapter.name: RewardBench2Adapter,
    RMBenchAdapter.name: RMBenchAdapter,
}

__all__ = [
    "ADAPTERS",
    "BenchmarkAdapter",
    "BenchmarkCase",
    "RewardBenchAdapter",
    "RewardBench2Adapter",
    "RMBenchAdapter",
]
