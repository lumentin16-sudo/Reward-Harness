"""RewardBench 2 数据转换、分层抽样与官方风格指标。"""

from __future__ import annotations

import random
from collections import defaultdict
from statistics import mean
from typing import Any

from .base import BenchmarkAdapter, BenchmarkCase, public_text
from ..reward_system import Response, Query


class RewardBench2Adapter(BenchmarkAdapter):
    """``allenai/reward-bench-2`` adapter。"""

    name = "rewardbench2"
    dataset_id = "allenai/reward-bench-2"
    split = "test"

    def load_cases(self, *, smoke_per_group: int, seed: int) -> list[BenchmarkCase]:
        rng = random.Random(seed)
        normal: dict[str, list[BenchmarkCase]] = defaultdict(list)
        for case in self.load_processed_cases():
            subset = case.group
            if subset != "Ties":
                normal[subset].append(case)

        selected: list[BenchmarkCase] = []
        for subset in sorted(normal):
            group = normal[subset]
            if smoke_per_group > 0:
                selected.extend(rng.sample(group, min(smoke_per_group, len(group))))
            else:
                selected.extend(group)

        return selected

    @staticmethod
    def _convert(row: dict[str, Any]) -> BenchmarkCase:
        completions = list(row.get("chosen") or []) + list(row.get("rejected") or [])
        row_id = str(row["id"])
        subset = str(row["subset"])
        return BenchmarkCase(
            case_id=f"rewardbench2:{row_id}",
            group=subset,
            task=Query(
                query_id=f"rewardbench2:{row_id}",
                instruction=public_text(row["prompt"]),
                domain=subset,
            ),
            candidates=tuple(
                Response(response_id=f"candidate_{index:03d}", content=public_text(text))
                for index, text in enumerate(completions)
            ),
            # 这些字段只供 evaluator 使用，永远不进入 Query/Response。
            gold={"num_correct": int(row["num_correct"]), "subset": subset, "source_id": row_id},
        )

    def score_outcome(self, outcome: dict[str, Any]) -> dict[str, Any]:
        contribution: dict[str, Any] = {"score": 0.0}
        if outcome.get("error"):
            outcome["metric"] = contribution
            return outcome
        winner = outcome.get("winner_result")
        num_correct = int(outcome["gold"]["num_correct"])
        if isinstance(winner, dict):
            response_ids = [
                str(response["response_id"])
                for response in outcome.get("responses", [])
            ]
            winner_id = str(winner.get("winner_response_id", ""))
            if winner_id in response_ids:
                contribution["score"] = float(
                    response_ids.index(winner_id) < num_correct
                )
        outcome["metric"] = contribution
        return outcome

    def summarize(self, outcomes: list[dict[str, Any]]) -> dict[str, Any]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for outcome in outcomes:
            grouped[str(outcome["group"])].append(outcome)

        subset_scores: dict[str, float] = {}
        for subset, rows in grouped.items():
            if subset != "Ties":
                subset_scores[subset] = mean(float(row.get("metric", {}).get("score", 0.0)) for row in rows)

        beyond_rubric = mean(subset_scores.values()) if subset_scores else 0.0
        subset_counts = {
            subset: len(rows)
            for subset, rows in grouped.items()
            if subset in subset_scores
        }
        total_count = sum(subset_counts.values())
        auto_rubric = (
            sum(
                subset_scores[subset] * subset_counts[subset]
                for subset in subset_counts
            )
            / total_count
            if total_count
            else 0.0
        )
        return {
            # Ties 已在加载阶段排除；关键指标使用其余五个 subset。
            "beyond_rubric": beyond_rubric,
            # auto_rubric：按五个非 Ties subset 的实际样本数加权。
            "auto_rubric": auto_rubric,
            "benchmark": self.name,
            "subset_scores": subset_scores,
            "excluded_subsets": ["Ties"],
            "derived_macro_average": beyond_rubric,
            "num_cases": len(outcomes),
            "num_errors": sum(bool(row.get("error")) for row in outcomes),
        }
