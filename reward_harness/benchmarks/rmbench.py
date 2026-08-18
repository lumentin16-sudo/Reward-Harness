"""RM-Bench 数据转换与官方 3×3 严格比较指标。"""

from __future__ import annotations

import random
from collections import defaultdict
from statistics import mean
from typing import Any

from .base import BenchmarkAdapter, BenchmarkCase, public_text
from ..reward_system import Candidate, RewardTask


class RMBenchAdapter(BenchmarkAdapter):
    name = "rmbench"
    dataset_id = "THU-KEG/RM-Bench"
    split = "train"

    def load_cases(self, *, smoke_per_group: int, seed: int) -> list[BenchmarkCase]:
        grouped: dict[str, list[BenchmarkCase]] = defaultdict(list)
        for case in self.load_processed_cases():
            grouped[case.group].append(case)
        rng = random.Random(seed)
        selected: list[BenchmarkCase] = []
        for domain in sorted(grouped):
            rows = grouped[domain]
            selected.extend(rng.sample(rows, min(smoke_per_group, len(rows))) if smoke_per_group > 0 else rows)
        return selected

    @staticmethod
    def _convert(row: dict[str, Any], fallback_index: int) -> BenchmarkCase:
        chosen = list(row["chosen"])
        rejected = list(row["rejected"])
        source_id = str(row.get("id", fallback_index))
        domain = str(row["domain"])
        case_id = f"rmbench:{domain}:{source_id}"
        return BenchmarkCase(
            case_id=case_id,
            group=domain,
            task=RewardTask(task_id=case_id, instruction=public_text(row["prompt"]), domain=domain),
            candidates=tuple(
                Candidate(candidate_id=f"candidate_{i:03d}", content=public_text(text))
                for i, text in enumerate(chosen + rejected)
            ),
            gold={"raw_domain": domain, "chosen_count": len(chosen), "source_id": source_id},
        )

    def score_outcome(self, outcome: dict[str, Any]) -> dict[str, Any]:
        if outcome.get("error"):
            matrix = [[0.0] * 3 for _ in range(3)]
        else:
            scores = [float(item["reward"]) for item in outcome["results"]]
            chosen_count = int(outcome["gold"]["chosen_count"])
            chosen, rejected = scores[:chosen_count], scores[chosen_count:]
            matrix = [[float(chosen[i] > rejected[j]) for j in range(3)] for i in range(3)]
        outcome["metric"] = {
            "matrix": matrix,
            "hard": mean((matrix[0][2], matrix[0][1], matrix[1][2])),
            "normal": mean(matrix[i][i] for i in range(3)),
            "easy": mean((matrix[1][0], matrix[2][0], matrix[2][1])),
            "average": mean(value for row in matrix for value in row),
        }
        return outcome

    @staticmethod
    def _coarse_domain(raw_domain: str) -> str:
        return "safety" if raw_domain.startswith("safety") else raw_domain.lower()

    def summarize(self, outcomes: list[dict[str, Any]]) -> dict[str, Any]:
        by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for outcome in outcomes:
            by_domain[self._coarse_domain(str(outcome["gold"]["raw_domain"]))].append(outcome)
        domains: dict[str, dict[str, float]] = {}
        for domain in ("chat", "code", "math", "safety"):
            rows = by_domain.get(domain, [])
            domains[domain] = {
                key: mean(float(row.get("metric", {}).get(key, 0.0)) for row in rows) if rows else 0.0
                for key in ("hard", "normal", "easy", "average")
            }
        return {
            "benchmark": self.name,
            "domains": domains,
            "hard": mean(domains[d]["hard"] for d in domains),
            "normal": mean(domains[d]["normal"] for d in domains),
            "easy": mean(domains[d]["easy"] for d in domains),
            "overall": mean(domains[d]["average"] for d in domains),
            "num_cases": len(outcomes),
            "num_errors": sum(bool(row.get("error")) for row in outcomes),
        }
