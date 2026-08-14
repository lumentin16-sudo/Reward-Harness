"""
Dynamic-Rubric Reward Agent 的稳定接口与审计数据结构。
"""

from __future__ import annotations

import json
import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, Literal, Mapping, Protocol, final


JSONValue = (
    None
    | bool
    | int
    | float
    | str
    | list["JSONValue"]
    | dict[str, "JSONValue"]
)
Component = Literal["G", "J", "A"]
ModelRole = Literal["rubric", "judge"]


class LLMCallable(Protocol):
    """由 evaluator 注入的冻结模型调用接口。"""

    def __call__(self, prompt: str) -> str: ...


def _require_non_empty(value: str, field_name: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must be non-empty")


def _require_finite(value: float, field_name: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite")


def _json_safe(value: Any, field_name: str) -> None:
    try:
        json.dumps(value, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be JSON-serializable") from exc


def _extract_json_object(text: str) -> dict[str, Any]:
    """从纯 JSON、Markdown 代码块或混合文本中提取第一个 JSON 对象。"""

    stripped = text.strip()
    try:
        value = json.loads(stripped)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass

    for match in re.finditer(r"```(?:json)?\s*([\s\S]*?)\s*```", text):
        try:
            value = json.loads(match.group(1))
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            continue

    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            continue

    raise ValueError("LLM response does not contain a valid JSON object")


@dataclass(frozen=True, slots=True)
class RewardTask:
    """生成 Rubric 时可见的 query，不包含候选回答和真实偏好。"""

    task_id: str
    instruction: str
    context: str | None = None
    domain: str | None = None
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.task_id, "task_id")
        _require_non_empty(self.instruction, "instruction")
        _json_safe(dict(self.metadata), "metadata")


@dataclass(frozen=True, slots=True)
class Candidate:
    """单个待评分回答或可审计的 Agent 轨迹。"""

    candidate_id: str
    content: str
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.candidate_id, "candidate_id")
        _require_non_empty(self.content, "content")
        _json_safe(dict(self.metadata), "metadata")


@dataclass(frozen=True, slots=True)
class TraceEvent:
    """可归因到 Rubric 生成、Judge 或聚合组件的审计事件。"""

    component: Component
    name: str
    payload: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.component not in {"G", "J", "A"}:
            raise ValueError(f"unknown trace component: {self.component!r}")
        _require_non_empty(self.name, "trace event name")
        _json_safe(dict(self.payload), "trace payload")


@dataclass(frozen=True, slots=True)
class ModelCallRecord:
    """模型调用包装器产生的审计记录。"""

    call_id: str
    role: ModelRole
    prompt: str
    response: str
    input_tokens: int
    output_tokens: int
    latency_ms: float

    def __post_init__(self) -> None:
        _require_non_empty(self.call_id, "call_id")
        if self.role not in {"rubric", "judge"}:
            raise ValueError(f"unknown model role: {self.role!r}")
        if self.input_tokens < 0 or self.output_tokens < 0:
            raise ValueError("token counts must be non-negative")
        _require_finite(self.latency_ms, "latency_ms")
        if self.latency_ms < 0:
            raise ValueError("latency_ms must be non-negative")


@dataclass(frozen=True, slots=True)
class Rubric:
    """一条任务相关的评估标准。"""

    rubric_id: str
    criterion: str
    weight: float = 1.0
    hard_constraint: bool = False
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.rubric_id, "rubric_id")
        _require_non_empty(self.criterion, "criterion")
        _require_finite(self.weight, "weight")
        if self.weight <= 0:
            raise ValueError("weight must be greater than zero")
        _json_safe(dict(self.metadata), "metadata")


@dataclass(frozen=True, slots=True)
class RubricSet:
    """A/B 两个回答共同使用的 Rubric 集合。"""

    task_id: str
    rubrics: tuple[Rubric, ...]
    trace: tuple[TraceEvent, ...] = ()
    model_calls: tuple[ModelCallRecord, ...] = ()
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.task_id, "task_id")
        ids = [rubric.rubric_id for rubric in self.rubrics]
        if len(ids) != len(set(ids)):
            raise ValueError("rubric ids must be unique within a RubricSet")
        if any(event.component != "G" for event in self.trace):
            raise ValueError("RubricSet.trace may contain only G events")
        if any(call.role != "rubric" for call in self.model_calls):
            raise ValueError("RubricSet.model_calls may contain only rubric calls")
        _json_safe(dict(self.metadata), "metadata")

    @property
    def rubric_ids(self) -> frozenset[str]:
        return frozenset(rubric.rubric_id for rubric in self.rubrics)


@dataclass(frozen=True, slots=True)
class SkillInput:
    """单次 Skill 调用可见的信息；不包含另一个候选或真实偏好。"""

    component: Literal["G", "J"]
    task: RewardTask
    rubrics: RubricSet | None = None
    candidate: Candidate | None = None

    def __post_init__(self) -> None:
        if self.component == "G":
            if self.rubrics is not None or self.candidate is not None:
                raise ValueError("G-stage skills may see only the task")
        elif self.component == "J":
            if self.rubrics is None or self.candidate is None:
                raise ValueError("J-stage skills require rubrics and one candidate")
        else:
            raise ValueError(f"unknown skill component: {self.component!r}")


@dataclass(frozen=True, slots=True)
class SkillResult:
    """一个原子 Skill 返回的辅助知识。"""

    skill_name: str
    content: str
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.skill_name, "skill_name")
        _require_non_empty(self.content, "skill result content")
        _json_safe(dict(self.metadata), "metadata")


class Skill(ABC):
    """可自由命名、且不与 Rubric/Judge 角色绑定的原子 Skill。"""

    name: str
    description: str

    @abstractmethod
    def invoke(self, context: SkillInput) -> SkillResult:
        """根据调用阶段提供辅助知识。"""


@dataclass(frozen=True, slots=True)
class SkillRegistry:
    """向模型暴露 Skill Catalog，并按名称执行零个或多个 Skill。"""

    skills: tuple[Skill, ...] = ()

    def __post_init__(self) -> None:
        names: list[str] = []
        for skill in self.skills:
            if not isinstance(skill, Skill):
                raise TypeError("SkillRegistry accepts only Skill instances")
            _require_non_empty(skill.name, "skill name")
            _require_non_empty(skill.description, "skill description")
            names.append(skill.name)
        if len(names) != len(set(names)):
            raise ValueError("skill names must be unique within a SkillRegistry")

    @property
    def names(self) -> frozenset[str]:
        return frozenset(skill.name for skill in self.skills)

    @property
    def catalog(self) -> tuple[dict[str, str], ...]:
        return tuple(
            {"name": skill.name, "description": skill.description}
            for skill in self.skills
        )

    def invoke(
        self,
        skill_names: tuple[str, ...],
        context: SkillInput,
    ) -> tuple[SkillResult, ...]:
        if len(skill_names) != len(set(skill_names)):
            raise ValueError("a model may call each skill at most once per stage")
        by_name = {skill.name: skill for skill in self.skills}
        unknown = set(skill_names) - set(by_name)
        if unknown:
            raise ValueError(f"unknown skill names: {sorted(unknown)}")

        results: list[SkillResult] = []
        for name in skill_names:
            result = by_name[name].invoke(context)
            if not isinstance(result, SkillResult):
                raise TypeError(f"skill {name!r} must return SkillResult")
            if result.skill_name != name:
                raise ValueError(f"skill {name!r} returned a mismatched skill_name")
            results.append(result)
        return tuple(results)


@dataclass(frozen=True, slots=True)
class CriterionJudgment:
    """Judge 对单条 Rubric 给出的评分与依据。"""

    rubric_id: str
    score: int
    evidence: tuple[str, ...] = ()
    confidence: float | None = None
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.rubric_id, "rubric_id")
        if isinstance(self.score, bool) or not isinstance(self.score, int):
            raise ValueError("criterion score must be an integer")
        if not 0 <= self.score <= 5:
            raise ValueError("criterion score must be in [0, 5]")
        if self.confidence is not None:
            _require_finite(self.confidence, "confidence")
            if not 0 <= self.confidence <= 1:
                raise ValueError("confidence must be in [0, 1]")
        _json_safe(dict(self.metadata), "metadata")


@dataclass(frozen=True, slots=True)
class RewardResult:
    """单个回答的逐项评分、聚合奖励和审计信息。"""

    task_id: str
    candidate_id: str
    reward: float
    judgments: tuple[CriterionJudgment, ...] = ()
    trace: tuple[TraceEvent, ...] = ()
    model_calls: tuple[ModelCallRecord, ...] = ()
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.task_id, "task_id")
        _require_non_empty(self.candidate_id, "candidate_id")
        _require_finite(self.reward, "reward")
        if not 0 <= self.reward <= 1:
            raise ValueError("reward must be in [0, 1]")
        if any(event.component == "G" for event in self.trace):
            raise ValueError("RewardResult.trace must not duplicate G events")
        if any(call.role == "rubric" for call in self.model_calls):
            raise ValueError("RewardResult.model_calls must not contain rubric calls")
        _json_safe(dict(self.metadata), "metadata")


class Preference(str, Enum):
    """由两个奖励预测的偏好。"""

    A = "a"
    B = "b"
    TIE = "tie"


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    """共享 Rubric 下的回答对比较结果。"""

    task_id: str
    rubric_set: RubricSet
    result_a: RewardResult
    result_b: RewardResult
    preference: Preference
    reward_margin: float

    @property
    def model_calls(self) -> tuple[ModelCallRecord, ...]:
        return (
            self.rubric_set.model_calls
            + self.result_a.model_calls
            + self.result_b.model_calls
        )

    @property
    def trace(self) -> tuple[TraceEvent, ...]:
        return self.rubric_set.trace + self.result_a.trace + self.result_b.trace


class RewardSystem(ABC):
    """所有 Reward Harness 必须遵守的接口。"""

    min_rubrics: ClassVar[int] = 2
    max_rubrics: ClassVar[int] = 6

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        fixed_methods = {
            "compare",
            "aggregate",
            "_parse_skill_calls",
            "_parse_rubrics",
            "_parse_judgments",
        }
        overridden = fixed_methods.intersection(cls.__dict__)
        if overridden:
            names = ", ".join(sorted(overridden))
            raise TypeError(f"RewardSystem subclasses must not override: {names}")

    def __init__(
        self,
        rubric_llm: LLMCallable,
        judge_llm: LLMCallable,
        *,
        tie_tolerance: float = 0.0,
    ) -> None:
        _require_finite(tie_tolerance, "tie_tolerance")
        if tie_tolerance < 0:
            raise ValueError("tie_tolerance must be non-negative")
        self._rubric_llm = rubric_llm
        self._judge_llm = judge_llm
        self._tie_tolerance = tie_tolerance

    @property
    def rubric_llm(self) -> LLMCallable:
        return self._rubric_llm

    @property
    def judge_llm(self) -> LLMCallable:
        return self._judge_llm

    @abstractmethod
    def build_rubrics(self, task: RewardTask) -> RubricSet:
        """只根据 query 生成 Rubric。"""

    @abstractmethod
    def score(
        self,
        task: RewardTask,
        candidate: Candidate,
        rubrics: RubricSet,
    ) -> RewardResult:
        """根据共享 RubricSet 为一个回答评分并聚合奖励。"""

    @final
    def aggregate(
        self,
        judgments: tuple[CriterionJudgment, ...],
        rubrics: RubricSet,
    ) -> float:
        """按 Rubric.weight 加权平均，并将结果归一化到 [0, 1]。"""

        if not judgments:
            raise ValueError("cannot aggregate an empty judgment set")
        judgment_by_id = {judgment.rubric_id: judgment for judgment in judgments}
        total_weight = sum(rubric.weight for rubric in rubrics.rubrics)
        weighted_score = sum(
            rubric.weight * judgment_by_id[rubric.rubric_id].score
            for rubric in rubrics.rubrics
        )
        return weighted_score / (5.0 * total_weight)

    @staticmethod
    @final
    def _parse_skill_calls(
        raw_response: str,
        registry: SkillRegistry,
    ) -> tuple[str, ...]:
        """解析模型选择的 Skill 名称；允许选择零个或多个。"""

        payload = _extract_json_object(raw_response)
        raw_calls = payload.get("skill_calls")
        if not isinstance(raw_calls, list) or any(
            not isinstance(name, str) for name in raw_calls
        ):
            raise ValueError("skill selection must contain a skill_calls string list")
        calls = tuple(raw_calls)
        if len(calls) != len(set(calls)):
            raise ValueError("skill_calls must not contain duplicates")
        unknown = set(calls) - registry.names
        if unknown:
            raise ValueError(f"unknown skill names: {sorted(unknown)}")
        return calls

    @staticmethod
    @final
    def _parse_rubrics(raw_response: str) -> tuple[Rubric, ...]:
        """解析固定 Rubric JSON 协议。"""

        payload = _extract_json_object(raw_response)
        raw_rubrics = payload.get("rubrics")
        if not isinstance(raw_rubrics, list):
            raise ValueError("rubric response must contain a rubrics list")

        rubrics: list[Rubric] = []
        for index, raw in enumerate(raw_rubrics):
            if not isinstance(raw, dict):
                raise ValueError(f"rubric at index {index} must be an object")
            try:
                weight_raw = raw.get("weight", 1.0)
                if isinstance(weight_raw, bool) or not isinstance(
                    weight_raw, (int, float)
                ):
                    raise ValueError("weight must be a positive number")
                rubrics.append(
                    Rubric(
                        rubric_id=str(raw.get("rubric_id", "")),
                        criterion=str(raw.get("criterion", "")),
                        weight=float(weight_raw),
                    )
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid rubric at index {index}: {exc}") from exc
        return tuple(rubrics)

    @staticmethod
    @final
    def _parse_judgments(
        raw_response: str,
        rubrics: RubricSet,
    ) -> tuple[CriterionJudgment, ...]:
        """解析固定 Judge JSON 协议并检查 Rubric 覆盖完整性。"""

        payload = _extract_json_object(raw_response)
        raw_judgments = payload.get("judgments")
        if not isinstance(raw_judgments, list):
            raise ValueError("judge response must contain a judgments list")

        parsed: dict[str, CriterionJudgment] = {}
        for index, raw in enumerate(raw_judgments):
            if not isinstance(raw, dict):
                raise ValueError(f"judgment at index {index} must be an object")
            rubric_id = str(raw.get("rubric_id", ""))
            if rubric_id in parsed:
                raise ValueError(f"duplicate judgment for rubric {rubric_id!r}")
            score = raw.get("score")
            if isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= 5:
                raise ValueError(
                    f"score for rubric {rubric_id!r} must be an integer from 0 to 5"
                )

            evidence_raw = raw.get("evidence", [])
            if isinstance(evidence_raw, str):
                evidence = (evidence_raw,)
            elif isinstance(evidence_raw, list):
                evidence = tuple(str(item) for item in evidence_raw)
            else:
                raise ValueError(f"evidence for rubric {rubric_id!r} must be a list")

            confidence_raw = raw.get("confidence")
            confidence = None if confidence_raw is None else float(confidence_raw)
            parsed[rubric_id] = CriterionJudgment(
                rubric_id=rubric_id,
                score=score,
                evidence=evidence,
                confidence=confidence,
            )

        expected = rubrics.rubric_ids
        actual = frozenset(parsed)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(
                f"judgments must cover every shared rubric; missing={missing}, extra={extra}"
            )
        return tuple(parsed[rubric.rubric_id] for rubric in rubrics.rubrics)

    @final
    def compare(
        self,
        task: RewardTask,
        candidate_a: Candidate,
        candidate_b: Candidate,
    ) -> ComparisonResult:
        """生成一次 Rubric，并用同一对象分别评价 A/B。"""

        if candidate_a.candidate_id == candidate_b.candidate_id:
            raise ValueError("candidate ids must be distinct")

        rubrics = self.build_rubrics(task)
        RewardSystem._validate_rubric_set(self, task, rubrics)

        result_a = self.score(task, candidate_a, rubrics)
        result_b = self.score(task, candidate_b, rubrics)
        RewardSystem._validate_reward_result(task, candidate_a, rubrics, result_a)
        RewardSystem._validate_reward_result(task, candidate_b, rubrics, result_b)

        margin = result_a.reward - result_b.reward
        if abs(margin) <= self._tie_tolerance:
            preference = Preference.TIE
        elif margin > 0:
            preference = Preference.A
        else:
            preference = Preference.B

        return ComparisonResult(
            task_id=task.task_id,
            rubric_set=rubrics,
            result_a=result_a,
            result_b=result_b,
            preference=preference,
            reward_margin=margin,
        )

    def _validate_rubric_set(self, task: RewardTask, rubrics: RubricSet) -> None:
        if rubrics.task_id != task.task_id:
            raise ValueError("RubricSet.task_id does not match the task")
        if not self.min_rubrics <= len(rubrics.rubrics) <= self.max_rubrics:
            raise ValueError(
                f"expected {self.min_rubrics}..{self.max_rubrics} rubrics, "
                f"got {len(rubrics.rubrics)}"
            )

    @staticmethod
    def _validate_reward_result(
        task: RewardTask,
        candidate: Candidate,
        rubrics: RubricSet,
        result: RewardResult,
    ) -> None:
        if result.task_id != task.task_id:
            raise ValueError("RewardResult.task_id does not match the task")
        if result.candidate_id != candidate.candidate_id:
            raise ValueError("RewardResult.candidate_id does not match the candidate")

        judged_ids = [judgment.rubric_id for judgment in result.judgments]
        if len(judged_ids) != len(set(judged_ids)):
            raise ValueError("a RewardResult may judge each rubric only once")
        if frozenset(judged_ids) != rubrics.rubric_ids:
            raise ValueError("judgments must cover every shared rubric")
