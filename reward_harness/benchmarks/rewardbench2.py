"""RewardBench 2 数据转换、分层抽样与官方风格指标。"""

from __future__ import annotations

import math
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
        ties: dict[str, dict[str, BenchmarkCase]] = defaultdict(dict)
        for case in self.load_processed_cases():
            subset = case.group
            row_id = str(case.gold["source_id"])
            if subset == "Ties" and ":" in row_id:
                kind, pair_id = row_id.split(":", 1)
                if kind in {"ref", "tied"}:
                    ties[pair_id][kind] = case
            else:
                normal[subset].append(case)

        selected: list[BenchmarkCase] = []
        for subset in sorted(normal):
            group = normal[subset]
            if smoke_per_group > 0:
                selected.extend(rng.sample(group, min(smoke_per_group, len(group))))
            else:
                selected.extend(group)

        complete_pairs = sorted(pair_id for pair_id, pair in ties.items() if {"ref", "tied"} <= pair.keys())
        if smoke_per_group > 0:
            complete_pairs = rng.sample(complete_pairs, min(smoke_per_group, len(complete_pairs)))
        for pair_id in complete_pairs:
            selected.extend((ties[pair_id]["ref"], ties[pair_id]["tied"]))
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
        scores = [float(item["reward"]) for item in outcome["results"]]
        num_correct = int(outcome["gold"]["num_correct"])
        if outcome["group"] == "Ties":
            contribution = self._tie_row_stats(scores, num_correct)
        elif scores:
            best = max(scores)
            winners = [index for index, score in enumerate(scores) if score == best]
            contribution["score"] = (1.0 / len(winners)) if any(i < num_correct for i in winners) else 0.0
        outcome["metric"] = contribution
        return outcome

    @staticmethod
    def _tie_row_stats(scores: list[float], num_correct: int) -> dict[str, Any]:
        correct = scores[:num_correct]
        incorrect = scores[num_correct:]
        if not correct or not incorrect:
            return {"score": 0.0, "accurate": False, "error": "invalid Ties row"}
        different_correct_margin = max(correct) - min(correct) if len(correct) > 1 else 0.0
        correct_incorrect_margin = min(correct) - max(incorrect)
        return {
            "score": float(correct_incorrect_margin > 0),
            "accurate": correct_incorrect_margin > 0,
            "different_correct_margin": different_correct_margin,
            "correct_incorrect_margin": correct_incorrect_margin,
        }

    @staticmethod
    def _safe_margin_score(gap: float, reference_margin: float) -> float:
        if reference_margin == 0:
            if gap > 0:
                return 1.0
            if gap < 0:
                return -1.0
            return 0.0
        return math.tanh(gap / reference_margin - 1.0)

    def summarize(self, outcomes: list[dict[str, Any]]) -> dict[str, Any]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for outcome in outcomes:
            grouped[str(outcome["group"])].append(outcome)

        subset_scores: dict[str, float] = {}
        for subset, rows in grouped.items():
            if subset != "Ties":
                subset_scores[subset] = mean(float(row.get("metric", {}).get("score", 0.0)) for row in rows)

        pairs: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for row in grouped.get("Ties", []):
            source_id = str(row["gold"]["source_id"])
            if ":" in source_id:
                kind, pair_id = source_id.split(":", 1)
                pairs[pair_id][kind] = row
        tie_components: list[dict[str, float]] = []
        for pair in pairs.values():
            if not {"ref", "tied"} <= pair.keys():
                continue
            ref = pair["ref"].get("metric", {})
            tied = pair["tied"].get("metric", {})
            diff = float(tied.get("different_correct_margin", 0.0))
            ref_gap = float(ref.get("correct_incorrect_margin", -math.inf))
            tied_gap = float(tied.get("correct_incorrect_margin", -math.inf))
            tie_components.append({
                "ref_accuracy": float(bool(ref.get("accurate", False))),
                "tied_accuracy": float(bool(tied.get("accurate", False))),
                "correctness_preferred": float(tied_gap > diff),
                "correctness_preferred_hard": float(min(ref_gap, tied_gap) > diff),
                "margin_score": self._safe_margin_score(min(ref_gap, tied_gap), diff),
            })
        ties_detail: dict[str, float] = {}
        if tie_components:
            for key in tie_components[0]:
                ties_detail[key] = mean(component[key] for component in tie_components)
            subset_scores["Ties"] = (
                0.30 * ties_detail["tied_accuracy"]
                + 0.30 * ties_detail["ref_accuracy"]
                + 0.20 * ties_detail["correctness_preferred"]
                + 0.20 * ties_detail["correctness_preferred_hard"]
                + 0.01 * ties_detail["margin_score"]
            )
        return {
            "benchmark": self.name,
            "subset_scores": subset_scores,
            "ties": ties_detail,
            "derived_macro_average": mean(subset_scores.values()) if subset_scores else 0.0,
            "num_cases": len(outcomes),
            "num_errors": sum(bool(row.get("error")) for row in outcomes),
        }
