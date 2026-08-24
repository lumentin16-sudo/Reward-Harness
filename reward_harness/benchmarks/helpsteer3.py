"""HelpSteer3 单一 held-in 搜索集适配器。"""

from __future__ import annotations

import random
from collections import defaultdict
from statistics import mean
from typing import Any

from .base import BenchmarkAdapter, BenchmarkCase


class _HelpSteer3Adapter(BenchmarkAdapter):
    dataset_id = "nvidia/HelpSteer3"

    def load_cases(self, *, smoke_per_group: int, seed: int) -> list[BenchmarkCase]:
        grouped: dict[str, list[BenchmarkCase]] = defaultdict(list)
        for case in self.load_processed_cases():
            grouped[case.group].append(case)
        rng = random.Random(seed)
        selected: list[BenchmarkCase] = []
        for domain in sorted(grouped):
            rows = grouped[domain]
            selected.extend(
                rng.sample(rows, min(smoke_per_group, len(rows)))
                if smoke_per_group > 0
                else rows
            )
        return selected

    def score_outcome(self, outcome: dict[str, Any]) -> dict[str, Any]:
        score = 0.0
        if not outcome.get("error"):
            winner = outcome.get("winner_result")
            if isinstance(winner, dict):
                score = float(
                    winner.get("winner_response_id") == "candidate_000"
                )
        outcome["metric"] = {"score": score}
        return outcome

    def summarize(self, outcomes: list[dict[str, Any]]) -> dict[str, Any]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for outcome in outcomes:
            grouped[str(outcome["group"])].append(outcome)
        domain_scores = {
            domain: mean(
                float(row.get("metric", {}).get("score", 0.0)) for row in rows
            )
            for domain, rows in sorted(grouped.items())
        }
        macro_average = mean(domain_scores.values()) if domain_scores else 0.0
        return {
            "beyond_rubric": macro_average,
            "auto_rubric": macro_average,
            "benchmark": self.name,
            "domain_scores": domain_scores,
            "derived_macro_average": macro_average,
            "num_cases": len(outcomes),
            "num_errors": sum(bool(row.get("error")) for row in outcomes),
        }


class HeldInAdapter(_HelpSteer3Adapter):
    """同时用于 Harness 搜索评分与轨迹分析的 500 条数据。"""

    name = "held_in"
    split = "train"
