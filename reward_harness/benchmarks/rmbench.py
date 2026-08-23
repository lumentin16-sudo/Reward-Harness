"""RM-Bench 数据转换与官方 3×3 严格比较指标。"""

from __future__ import annotations

import random
from collections import defaultdict
from statistics import mean
from typing import Any

from .base import BenchmarkAdapter, BenchmarkCase, public_text
from ..reward_system import Response, Query


class RMBenchAdapter(BenchmarkAdapter):
    name = "rmbench"
    dataset_id = "THU-KEG/RM-Bench"
    split = "train"

    def load_cases(self, *, smoke_per_group: int, seed: int) -> list[BenchmarkCase]:
        grouped: dict[str, list[BenchmarkCase]] = defaultdict(list)
        for case in self.load_processed_cases():
            grouped[case.group].append(case)
        rng = random.Random(seed)
        selected_prompts: list[BenchmarkCase] = []
        for domain in sorted(grouped):
            rows = grouped[domain]
            selected_prompts.extend(
                rng.sample(rows, min(smoke_per_group, len(rows)))
                if smoke_per_group > 0
                else rows
            )
        return [
            pair
            for case in selected_prompts
            for pair in self._expand_pairs(case)
        ]

    @staticmethod
    def _expand_pairs(case: BenchmarkCase) -> list[BenchmarkCase]:
        """把3 chosen × 3 rejected 展开成9个 forced-choice case。"""

        chosen_count = int(case.gold["chosen_count"])
        chosen = case.candidates[:chosen_count]
        rejected = case.candidates[chosen_count:]
        if len(chosen) != 3 or len(rejected) != 3:
            raise ValueError("RM-Bench requires exactly 3 chosen and 3 rejected")
        pairs = []
        for chosen_style, chosen_response in enumerate(chosen):
            for rejected_style, rejected_response in enumerate(rejected):
                pair_id = (
                    f"{case.case_id}:pair:{chosen_style}:{rejected_style}"
                )
                difficulty = (
                    "hard"
                    if chosen_style < rejected_style
                    else "easy"
                    if chosen_style > rejected_style
                    else "normal"
                )
                pairs.append(
                    BenchmarkCase(
                        case_id=pair_id,
                        group=case.group,
                        task=Query(
                            query_id=pair_id,
                            instruction=case.task.instruction,
                            context=case.task.context,
                            domain=case.task.domain,
                            metadata=case.task.metadata,
                        ),
                        candidates=(
                            Response("candidate_000", chosen_response.content),
                            Response("candidate_001", rejected_response.content),
                        ),
                        gold={
                            "chosen_index": 0,
                            "raw_domain": str(case.gold["raw_domain"]),
                            "source_id": str(case.gold["source_id"]),
                            "original_case_id": case.case_id,
                            "chosen_style": chosen_style,
                            "rejected_style": rejected_style,
                            "difficulty": difficulty,
                        },
                    )
                )
        return pairs

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
            task=Query(query_id=case_id, instruction=public_text(row["prompt"]), domain=domain),
            candidates=tuple(
                Response(response_id=f"candidate_{i:03d}", content=public_text(text))
                for i, text in enumerate(chosen + rejected)
            ),
            gold={"raw_domain": domain, "chosen_count": len(chosen), "source_id": source_id},
        )

    def score_outcome(self, outcome: dict[str, Any]) -> dict[str, Any]:
        score = 0.0
        if not outcome.get("error"):
            winner = outcome.get("winner_result")
            if isinstance(winner, dict):
                score = float(
                    winner.get("winner_response_id") == "candidate_000"
                )
        outcome["metric"] = {
            "score": score,
            "difficulty": str(outcome["gold"]["difficulty"]),
            "chosen_style": int(outcome["gold"]["chosen_style"]),
            "rejected_style": int(outcome["gold"]["rejected_style"]),
        }
        return outcome

    @staticmethod
    def _coarse_domain(raw_domain: str) -> str:
        return "safety" if raw_domain.startswith("safety") else raw_domain.lower()

    def summarize(self, outcomes: list[dict[str, Any]]) -> dict[str, Any]:
        by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for outcome in outcomes:
            by_prompt[str(outcome["gold"]["original_case_id"])].append(outcome)

        by_domain: dict[str, list[dict[str, float]]] = defaultdict(list)
        for rows in by_prompt.values():
            matrix = [[0.0] * 3 for _ in range(3)]
            for row in rows:
                metric = row.get("metric", {})
                i = int(metric.get("chosen_style", row["gold"]["chosen_style"]))
                j = int(metric.get("rejected_style", row["gold"]["rejected_style"]))
                matrix[i][j] = float(metric.get("score", 0.0))
            prompt_metrics = {
                "hard": mean((matrix[0][1], matrix[0][2], matrix[1][2])),
                "normal": mean(matrix[i][i] for i in range(3)),
                "easy": mean((matrix[1][0], matrix[2][0], matrix[2][1])),
                "average": mean(value for matrix_row in matrix for value in matrix_row),
            }
            domain = self._coarse_domain(str(rows[0]["gold"]["raw_domain"]))
            by_domain[domain].append(prompt_metrics)

        domains: dict[str, dict[str, float]] = {}
        for domain in ("chat", "code", "math", "safety"):
            rows = by_domain.get(domain, [])
            domains[domain] = {
                key: mean(row[key] for row in rows) if rows else 0.0
                for key in ("hard", "normal", "easy", "average")
            }
        beyond_rubric = mean(domains[domain]["average"] for domain in domains)
        domain_counts = {
            domain: len(by_domain.get(domain, [])) for domain in domains
        }
        total_count = sum(domain_counts.values())
        auto_rubric = (
            sum(
                domains[domain]["average"] * domain_counts[domain]
                for domain in domains
            )
            / total_count
            if total_count
            else 0.0
        )
        return {
            # beyond_rubric：四个 domain 的 3×3 平均准确率做宏平均。
            "beyond_rubric": beyond_rubric,
            # auto_rubric：四个 domain 的 3×3 平均准确率按样本数加权。
            "auto_rubric": auto_rubric,
            "benchmark": self.name,
            "domains": domains,
            "hard": mean(domains[d]["hard"] for d in domains),
            "normal": mean(domains[d]["normal"] for d in domains),
            "easy": mean(domains[d]["easy"] for d in domains),
            "overall": beyond_rubric,
            "num_prompts": len(by_prompt),
            "num_cases": len(outcomes),
            "num_errors": sum(bool(row.get("error")) for row in outcomes),
        }
