"""RewardBench v1 数据转换与官方 core section 指标。"""

from __future__ import annotations

import random
from collections import defaultdict
from statistics import mean
from typing import Any

from .base import BenchmarkAdapter, BenchmarkCase, public_text
from ..reward_system import Response, Query


# 来自 allenai/reward-bench/rewardbench/constants.py。Math 按官方规则上采样到
# 与六种 HumanEvalPack 代码题的总权重相同。
EXAMPLE_COUNTS = {
    "alpacaeval-easy": 100,
    "alpacaeval-length": 95,
    "alpacaeval-hard": 95,
    "mt-bench-easy": 28,
    "mt-bench-med": 40,
    "mt-bench-hard": 37,
    "math-prm": 984,
    "refusals-dangerous": 100,
    "refusals-offensive": 100,
    "llmbar-natural": 100,
    "llmbar-adver-neighbor": 134,
    "llmbar-adver-GPTInst": 92,
    "llmbar-adver-GPTOut": 47,
    "llmbar-adver-manual": 46,
    "xstest-should-refuse": 154,
    "xstest-should-respond": 250,
    "donotanswer": 136,
    "hep-cpp": 164,
    "hep-go": 164,
    "hep-java": 164,
    "hep-js": 164,
    "hep-python": 164,
    "hep-rust": 164,
}

SUBSET_MAPPING = {
    "Chat": (
        "alpacaeval-easy",
        "alpacaeval-length",
        "alpacaeval-hard",
        "mt-bench-easy",
        "mt-bench-med",
    ),
    "Chat Hard": (
        "mt-bench-hard",
        "llmbar-natural",
        "llmbar-adver-neighbor",
        "llmbar-adver-GPTInst",
        "llmbar-adver-GPTOut",
        "llmbar-adver-manual",
    ),
    "Safety": (
        "refusals-dangerous",
        "refusals-offensive",
        "xstest-should-refuse",
        "xstest-should-respond",
        "donotanswer",
    ),
    "Reasoning": (
        "math-prm",
        "hep-cpp",
        "hep-go",
        "hep-java",
        "hep-js",
        "hep-python",
        "hep-rust",
    ),
}

SUBSET_ALIASES = {
    "mt-bench-medium": "mt-bench-med",
    "do not answer": "donotanswer",
}


class RewardBenchAdapter(BenchmarkAdapter):
    """``allenai/reward-bench`` 官方 filtered core set adapter。"""

    name = "rewardbench"
    dataset_id = "allenai/reward-bench"
    split = "filtered"

    def load_cases(self, *, smoke_per_group: int, seed: int) -> list[BenchmarkCase]:
        grouped: dict[str, list[BenchmarkCase]] = defaultdict(list)
        for case in self.load_processed_cases():
            grouped[case.group].append(case)
        rng = random.Random(seed)
        selected: list[BenchmarkCase] = []
        for subset in sorted(grouped):
            rows = grouped[subset]
            selected.extend(
                rng.sample(rows, min(smoke_per_group, len(rows)))
                if smoke_per_group > 0
                else rows
            )
        return selected

    @staticmethod
    def _convert(row: dict[str, Any], fallback_index: int) -> BenchmarkCase:
        source_id = str(row.get("id", fallback_index))
        raw_subset = str(row["subset"])
        subset = SUBSET_ALIASES.get(raw_subset, raw_subset)
        # 官方 filtered split 的 id=3692 出现两次（donotanswer 与 hep-python）。
        # 加入稳定行号，避免断点续跑按 case_id 去重时漏掉其中一题。
        case_id = f"rewardbench:{fallback_index:04d}:{source_id}"
        return BenchmarkCase(
            case_id=case_id,
            group=subset,
            task=Query(
                query_id=case_id,
                instruction=public_text(row["prompt"]),
                domain=subset,
            ),
            candidates=(
                Response("candidate_000", public_text(row["chosen"])),
                Response("candidate_001", public_text(row["rejected"])),
            ),
            # chosen 的位置只保存在 evaluator-only gold，绝不进入模型 prompt。
            gold={"chosen_index": 0, "subset": subset, "source_id": source_id},
        )

    def score_outcome(self, outcome: dict[str, Any]) -> dict[str, Any]:
        score = 0.0
        if not outcome.get("error"):
            winner = outcome.get("winner_result")
            if isinstance(winner, dict):
                # candidate_000 是 evaluator-only chosen；Judge 只见随机匿名位置。
                score = float(
                    winner.get("winner_response_id") == "candidate_000"
                )
        outcome["metric"] = {"score": score}
        return outcome

    def summarize(self, outcomes: list[dict[str, Any]]) -> dict[str, Any]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for outcome in outcomes:
            grouped[str(outcome["group"])].append(outcome)
        subset_scores = {
            subset: mean(
                float(row.get("metric", {}).get("score", 0.0)) for row in rows
            )
            for subset, rows in sorted(grouped.items())
        }

        section_scores: dict[str, float] = {}
        for section, subsets in SUBSET_MAPPING.items():
            available = [subset for subset in subsets if subset in subset_scores]
            total_weight = sum(EXAMPLE_COUNTS[subset] for subset in available)
            section_scores[section] = (
                sum(
                    subset_scores[subset] * EXAMPLE_COUNTS[subset]
                    for subset in available
                )
                / total_weight
                if total_weight
                else 0.0
            )

        core_average = mean(section_scores.values())
        return {
            # 两个关键指标都采用四个官方 core section 的宏平均。
            "beyond_rubric": core_average,
            "auto_rubric": core_average,
            "benchmark": self.name,
            "subset_scores": subset_scores,
            "section_scores": section_scores,
            # 官方 leaderboard 还包含单独的 Prior Sets；本 adapter 只加载 core set。
            "derived_core_overall": core_average,
            "num_cases": len(outcomes),
            "num_errors": sum(bool(row.get("error")) for row in outcomes),
        }
