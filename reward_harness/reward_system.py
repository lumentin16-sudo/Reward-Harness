"""Dynamic-Rubric Reward Agent 的稳定协议与安全边界。

本模块只负责所有候选 Harness 必须共同遵守的部分：

1. 定义任务、候选回答、Rubric、Skill 和评分结果的数据结构；
2. 限制 Rubric 生成阶段（G）和 Judge 阶段（J）各自可见的信息；
3. 将内部对象转换成稳定的 Prompt payload，并解析模型返回的 JSON；
4. 校验候选输出，固定奖励聚合与 A/B 比较流程；
5. 保存模型调用和 G/J/A 执行轨迹，供复现与错误分析使用。

候选 Harness 的搜索边界只有三个公开接口：
``get_skill_registry``、``build_rubrics`` 和 ``score``。候选可以自由设计
Skill、Prompt 和 G/J 调用流程，但不能修改 payload、解析、聚合与比较协议。
"""

from __future__ import annotations

import json
import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, Literal, Mapping, Protocol, Sequence, final


# ---------------------------------------------------------------------------
# 基础类型与冻结模型接口
# ---------------------------------------------------------------------------

# metadata 和 trace payload 只允许使用可 JSON 序列化的值，保证日志可持久化。
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
    """由 evaluator 注入的冻结模型调用接口。

    候选只拿到这个最小文本接口，无法替换 evaluator 管理的模型参数或调用记录器。
    """

    def __call__(self, prompt: str) -> str: ...


# ---------------------------------------------------------------------------
# 内部校验与 JSON 提取工具
# ---------------------------------------------------------------------------

def _require_non_empty(value: str, field_name: str) -> None:
    """统一检查协议中的必填字符串。"""

    if not value.strip():
        raise ValueError(f"{field_name} must be non-empty")


def _require_finite(value: float, field_name: str) -> None:
    """拒绝 NaN 和正负无穷，避免奖励或统计结果无法序列化。"""

    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite")


def _json_safe(value: Any, field_name: str) -> None:
    """在数据进入审计日志前验证其可被 JSON 稳定保存。"""

    try:
        json.dumps(value, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be JSON-serializable") from exc


def _extract_json_object(text: str) -> dict[str, Any]:
    """从模型文本中提取第一个 JSON 对象。

    模型可能返回纯 JSON、Markdown JSON 代码块，或在 JSON 前后附带少量说明。
    这里按上述顺序做兼容解析；解析失败时显式报错，不猜测缺失字段。
    """

    stripped = text.strip()
    # 最常见也最严格的情况：整个响应就是一个 JSON 对象。
    try:
        value = json.loads(stripped)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass

    # 兼容模型把 JSON 包在 ```json ... ``` 或普通代码块中。
    for match in re.finditer(r"```(?:json)?\s*([\s\S]*?)\s*```", text):
        try:
            value = json.loads(match.group(1))
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            continue

    # 最后尝试从混合文本的第一个左花括号开始解码 JSON 对象。
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


# ---------------------------------------------------------------------------
# 评估输入
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class RewardTask:
    """公开任务信息。

    Rubric Model 在 G 阶段只能看到该对象，不能看到候选回答或真实偏好。
    """

    task_id: str  # benchmark 内稳定且唯一的任务标识
    instruction: str  # 用户要求或待完成的任务正文
    context: str | None = None  # 允许公开给评价器的补充上下文
    domain: str | None = None  # 可选领域标签，例如 math / coding
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)  # 公开扩展字段

    def __post_init__(self) -> None:
        _require_non_empty(self.task_id, "task_id")
        _require_non_empty(self.instruction, "instruction")
        _json_safe(dict(self.metadata), "metadata")


@dataclass(frozen=True, slots=True)
class Candidate:
    """单个待评分回答或可审计的 Agent 轨迹。

    ``candidate_id`` 只用于结果绑定，不会通过固定 payload 暴露给 Judge，
    从而避免模型根据 ``a`` / ``b`` 等位置标识产生偏差。
    """

    candidate_id: str  # evaluator 用于绑定预测结果，不属于 Judge 输入
    content: str  # 当前 Judge 唯一可见的候选正文
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)  # evaluator 侧信息

    def __post_init__(self) -> None:
        _require_non_empty(self.candidate_id, "candidate_id")
        _require_non_empty(self.content, "content")
        _json_safe(dict(self.metadata), "metadata")


# ---------------------------------------------------------------------------
# 审计记录与 Rubric 数据结构
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class TraceEvent:
    """可归因到 G（生成）、J（评分）或 A（聚合）的结构化审计事件。"""

    component: Component  # 事件来自 G、J 或 A 中的哪一阶段
    name: str  # 稳定事件名，例如 rubrics_generated
    payload: Mapping[str, JSONValue] = field(default_factory=dict)  # 事件详情

    def __post_init__(self) -> None:
        if self.component not in {"G", "J", "A"}:
            raise ValueError(f"unknown trace component: {self.component!r}")
        _require_non_empty(self.name, "trace event name")
        _json_safe(dict(self.payload), "trace payload")


@dataclass(frozen=True, slots=True)
class ModelCallRecord:
    """冻结模型包装器记录的一次完整模型调用。"""

    call_id: str  # 单次实验内唯一的调用 ID
    role: ModelRole  # 调用属于 rubric model 还是 judge model
    prompt: str  # 实际送入模型的完整输入
    response: str  # 未解析的模型原始输出
    input_tokens: int  # 输入 token 成本
    output_tokens: int  # 输出 token 成本
    latency_ms: float  # 端到端调用耗时

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
    """一条可独立评分的任务相关评价标准。"""

    rubric_id: str  # RubricSet 内唯一，供 Judgment 精确引用
    criterion: str  # 应检查的单一、可观察属性
    weight: float = 1.0  # 固定聚合器使用的相对权重
    hard_constraint: bool = False  # 为未来硬约束聚合策略保留的显式标记
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
    """同一任务下由 A/B 两个回答共享的 Rubric 集合。

    ``compare`` 只生成一次 RubricSet，并把同一个对象分别传给两次 ``score``，
    防止候选 A 和 B 因评价标准不同而失去可比性。
    """

    task_id: str  # Rubric 所属任务
    rubrics: tuple[Rubric, ...]  # 不可变集合，评分阶段不能原地修改
    trace: tuple[TraceEvent, ...] = ()  # 这里只保存 G 阶段事件
    model_calls: tuple[ModelCallRecord, ...] = ()  # 这里只保存 rubric 模型调用
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
        """返回 Rubric ID 集合，供完整覆盖校验使用。"""

        return frozenset(rubric.rubric_id for rubric in self.rubrics)


# ---------------------------------------------------------------------------
# 原子 Skill 协议
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SkillInput:
    """单次 Skill 调用的受限上下文。

    G 阶段只能看到公开任务；J 阶段可以额外看到共享 Rubric 和当前单个候选。
    任何阶段都看不到另一个候选或 benchmark 的真实偏好标签。
    """

    component: Literal["G", "J"]
    task: RewardTask
    rubrics: RubricSet | None = None
    candidate: Candidate | None = None

    def __post_init__(self) -> None:
        # 在数据结构层强制执行信息隔离，而不是只依赖 Prompt 约定。
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
    """一个原子 Skill 返回、随后可注入模型 Prompt 的辅助知识。"""

    skill_name: str
    content: str
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.skill_name, "skill_name")
        _require_non_empty(self.content, "skill result content")
        _json_safe(dict(self.metadata), "metadata")


class Skill(ABC):
    """候选可自由定义的原子 Skill。

    Skill 不预先绑定 G 或 J；能否在某阶段工作由传入的 ``SkillInput`` 决定。
    """

    name: str
    description: str

    @abstractmethod
    def invoke(self, context: SkillInput) -> SkillResult:
        """根据调用阶段提供辅助知识。"""


@dataclass(frozen=True, slots=True)
class SkillRegistry:
    """向模型暴露 Skill Catalog，并按名称执行零个或多个 Skill。

    Registry 同时负责检查名称唯一性、未知调用和返回值绑定，避免模型或候选
    通过拼写错误静默调用到错误 Skill。
    """

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
        """返回可被模型选择的全部 Skill 名称。"""

        return frozenset(skill.name for skill in self.skills)

    @property
    def catalog(self) -> tuple[dict[str, str], ...]:
        """生成模型选择 Skill 时可见的最小目录。"""

        return tuple(
            {"name": skill.name, "description": skill.description}
            for skill in self.skills
        )

    def invoke(
        self,
        skill_names: tuple[str, ...],
        context: SkillInput,
    ) -> tuple[SkillResult, ...]:
        """按模型给出的顺序执行 Skill，并验证每个结果的来源绑定。"""

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


# ---------------------------------------------------------------------------
# Judge 输出与最终比较结果
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class CriterionJudgment:
    """Judge 对单条 Rubric 给出的离散评分与可审计依据。"""

    rubric_id: str  # 必须引用共享 RubricSet 中存在的 ID
    score: int  # 固定为 0..5 的整数
    evidence: tuple[str, ...] = ()  # 候选正文中的支持或反驳证据
    confidence: float | None = None  # 可选的 0..1 置信度，仅用于审计
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
    """单个候选的逐项评分、归一化奖励和 J/A 审计信息。"""

    task_id: str  # 防止结果被绑定到其他任务
    candidate_id: str  # 防止结果被绑定到另一个候选
    reward: float  # 最终标量奖励，范围固定为 [0, 1]
    judgments: tuple[CriterionJudgment, ...] = ()
    trace: tuple[TraceEvent, ...] = ()  # 只保存 J/A 事件，G 事件属于 RubricSet
    model_calls: tuple[ModelCallRecord, ...] = ()  # 只保存 judge 模型调用
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
    """由 A/B 两个标量奖励推导出的离散偏好。"""

    A = "a"
    B = "b"
    TIE = "tie"


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """一条任务在共享 Rubric 下的完整多候选评分结果。"""

    task_id: str
    rubric_set: RubricSet
    results: tuple[RewardResult, ...]

    @property
    def model_calls(self) -> tuple[ModelCallRecord, ...]:
        """按 Rubric 生成、候选评分的执行顺序合并全部模型调用。"""

        return self.rubric_set.model_calls + tuple(
            call for result in self.results for call in result.model_calls
        )

    @property
    def trace(self) -> tuple[TraceEvent, ...]:
        """按 Rubric 生成、候选评分的执行顺序合并全部审计事件。"""

        return self.rubric_set.trace + tuple(
            event for result in self.results for event in result.trace
        )


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    """共享 Rubric 下的一次完整 A/B 比较结果。"""

    task_id: str
    rubric_set: RubricSet
    result_a: RewardResult
    result_b: RewardResult
    preference: Preference
    reward_margin: float

    @property
    def model_calls(self) -> tuple[ModelCallRecord, ...]:
        """按 G → Judge A → Judge B 的顺序合并全部模型调用。"""

        return (
            self.rubric_set.model_calls
            + self.result_a.model_calls
            + self.result_b.model_calls
        )

    @property
    def trace(self) -> tuple[TraceEvent, ...]:
        """按 G → Judge A → Judge B 的顺序合并全部审计事件。"""

        return self.rubric_set.trace + self.result_a.trace + self.result_b.trace


# ---------------------------------------------------------------------------
# Reward Harness 稳定接口与固定控制流程
# ---------------------------------------------------------------------------

class RewardSystem(ABC):
    """所有候选 Reward Harness 必须实现的稳定接口。

    可搜索部分：
        - ``get_skill_registry``：Skill 的内容、数量和组织方式；
        - ``build_rubrics``：Rubric Prompt、Skill 选择和 G 调用流程；
        - ``score``：Judge Prompt、Skill 选择和 J 调用流程。

    固定部分：
        - 输入 payload 与 JSON 输出协议；
        - 分数范围、Rubric 覆盖和结果绑定校验；
        - 加权平均聚合、共享 Rubric 的 A/B 比较和 tie 判定。
    """

    min_rubrics: ClassVar[int] = 2
    max_rubrics: ClassVar[int] = 6

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """在候选类定义时立即拒绝对固定协议的覆盖。"""

        super().__init_subclass__(**kwargs)
        # @final 主要服务静态检查；这里再做运行时检查，防止生成代码绕过约束。
        fixed_methods = {
            "evaluate",
            "compare",
            "aggregate",
            "_task_payload",
            "_candidate_payload",
            "_rubrics_payload",
            "_skill_results_payload",
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
        """注入冻结的 Rubric/Judge 模型，并配置统一 tie 阈值。"""

        _require_finite(tie_tolerance, "tie_tolerance")
        if tie_tolerance < 0:
            raise ValueError("tie_tolerance must be non-negative")
        self._rubric_llm = rubric_llm
        self._judge_llm = judge_llm
        self._tie_tolerance = tie_tolerance

    @property
    def rubric_llm(self) -> LLMCallable:
        """Rubric 生成阶段使用的冻结模型。"""

        return self._rubric_llm

    @property
    def judge_llm(self) -> LLMCallable:
        """单候选评分阶段使用的冻结模型。"""

        return self._judge_llm

    @abstractmethod
    def get_skill_registry(self, task: RewardTask) -> SkillRegistry:
        """候选接口 1：返回原子 Skill；不使用 Skill 时返回空 Registry。"""

    @abstractmethod
    def build_rubrics(self, task: RewardTask) -> RubricSet:
        """候选接口 2：只根据公开任务生成共享 Rubric。"""

    @abstractmethod
    def score(
        self,
        task: RewardTask,
        candidate: Candidate,
        rubrics: RubricSet,
    ) -> RewardResult:
        """候选接口 3：根据共享 Rubric 为当前单个候选评分。"""

    @staticmethod
    @final
    def _task_payload(task: RewardTask) -> dict[str, JSONValue]:
        """将公开任务转换为所有候选共用的 Prompt payload。

        这里保留任务 metadata，因为它属于 benchmark 明确提供的公开任务信息；
        evaluator 必须保证其中不包含 preference label 等评测答案。
        """

        return {
            "task_id": task.task_id,
            "instruction": task.instruction,
            "context": task.context,
            "domain": task.domain,
            "metadata": dict(task.metadata),
        }

    @staticmethod
    @final
    def _candidate_payload(candidate: Candidate) -> dict[str, JSONValue]:
        """只暴露当前回答内容，避免候选 ID 或 evaluator metadata 泄露。"""

        return {"content": candidate.content}

    @staticmethod
    @final
    def _rubrics_payload(rubrics: RubricSet) -> list[dict[str, JSONValue]]:
        """将共享 Rubric 转换为 Judge 可见的稳定 payload。

        Judge 只需要知道“评什么”，不需要看到 G 阶段日志、模型响应和其他
        evaluator metadata。聚合权重由固定聚合器读取，无需注入 Judge Prompt。
        """

        return [
            {
                "rubric_id": rubric.rubric_id,
                "criterion": rubric.criterion,
            }
            for rubric in rubrics.rubrics
        ]

    @staticmethod
    @final
    def _skill_results_payload(
        results: tuple[SkillResult, ...],
    ) -> list[dict[str, JSONValue]]:
        """将已执行 Skill 的结果转换为稳定 Prompt payload。

        该转换保留 Skill 名称，便于模型区分多个来源，也便于审计最终使用了
        哪些辅助知识。
        """

        return [
            {
                "name": result.skill_name,
                "content": result.content,
                "metadata": dict(result.metadata),
            }
            for result in results
        ]

    @final
    def aggregate(
        self,
        judgments: tuple[CriterionJudgment, ...],
        rubrics: RubricSet,
    ) -> float:
        """按 Rubric.weight 加权平均，并将 0..5 分数归一化到 [0, 1]。

        聚合规则固定在基类中，避免候选通过修改标量映射而非改进评价能力来
        获得更高 benchmark 分数。
        """

        if not judgments:
            raise ValueError("cannot aggregate an empty judgment set")

        # _parse_judgments / _validate_reward_result 已保证每条 Rubric 恰有一项评分。
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
        """解析模型选择的 Skill 名称；允许选择零个或多个。

        固定响应协议为 ``{"skill_calls": ["name", ...]}``。未知 Skill 和重复
        Skill 都会被拒绝，而不是静默忽略。
        """

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
        """解析固定 Rubric JSON 协议，并构造受校验的不可变 Rubric。"""

        payload = _extract_json_object(raw_response)
        raw_rubrics = payload.get("rubrics")
        if not isinstance(raw_rubrics, list):
            raise ValueError("rubric response must contain a rubrics list")

        rubrics: list[Rubric] = []
        for index, raw in enumerate(raw_rubrics):
            if not isinstance(raw, dict):
                raise ValueError(f"rubric at index {index} must be an object")
            try:
                # weight 可以省略，缺省时所有 Rubric 等权。
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
        """解析固定 Judge JSON 协议并检查 Rubric 覆盖完整性。

        Judge 必须对共享 RubricSet 中每条 Rubric 恰好评分一次：遗漏、重复或
        额外生成的 rubric_id 都会使本次评估失败。
        """

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
            if (
                isinstance(score, bool)
                or not isinstance(score, int)
                or not 0 <= score <= 5
            ):
                raise ValueError(
                    f"score for rubric {rubric_id!r} must be an integer from 0 to 5"
                )

            # 为兼容模型偶尔返回单个字符串，内部统一转换成 tuple[str, ...]。
            evidence_raw = raw.get("evidence", [])
            if isinstance(evidence_raw, str):
                evidence = (evidence_raw,)
            elif isinstance(evidence_raw, list):
                evidence = tuple(str(item) for item in evidence_raw)
            else:
                raise ValueError(f"evidence for rubric {rubric_id!r} must be a list")

            # confidence 是可选审计信号，不参与当前固定聚合公式。
            confidence_raw = raw.get("confidence")
            confidence = None if confidence_raw is None else float(confidence_raw)
            parsed[rubric_id] = CriterionJudgment(
                rubric_id=rubric_id,
                score=score,
                evidence=evidence,
                confidence=confidence,
            )

        # 集合相等保证“无遗漏、无额外”；上面的字典检查保证“无重复”。
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
    def evaluate(
        self,
        task: RewardTask,
        candidates: Sequence[Candidate],
    ) -> EvaluationResult:
        """生成一次共享 Rubric，并依次为全部候选评分。

        这是面向普通调用方的完整评估入口。底层 ``build_rubrics`` 和
        ``score`` 仍保持独立，便于 benchmark 使用固定 Rubric、缓存 Rubric，
        或对单个候选执行精细的并发与失败重试。
        """

        candidate_tuple = tuple(candidates)
        if not candidate_tuple:
            raise ValueError("evaluate requires at least one candidate")
        candidate_ids = [candidate.candidate_id for candidate in candidate_tuple]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("candidate ids must be distinct")

        # G：每条任务只生成一次，所有候选严格共享同一个 RubricSet 对象。
        rubrics = self.build_rubrics(task)
        RewardSystem._validate_rubric_set(self, task, rubrics)

        results: list[RewardResult] = []
        for candidate in candidate_tuple:
            result = self.score(task, candidate, rubrics)
            RewardSystem._validate_reward_result(task, candidate, rubrics, result)
            results.append(result)

        return EvaluationResult(
            task_id=task.task_id,
            rubric_set=rubrics,
            results=tuple(results),
        )

    @final
    def compare(
        self,
        task: RewardTask,
        candidate_a: Candidate,
        candidate_b: Candidate,
    ) -> ComparisonResult:
        """执行一次完整、pointwise 且共享 Rubric 的 A/B 比较。

        固定顺序为：生成 Rubric → 评分 A → 评分 B → 校验 → 计算偏好。
        ``score`` 每次只能看到当前候选，不能直接比较 A/B 文本。
        """

        evaluation = self.evaluate(task, (candidate_a, candidate_b))
        rubrics = evaluation.rubric_set
        result_a, result_b = evaluation.results

        # A：tie_tolerance 把足够接近的两个标量奖励映射为平局。
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
        """校验候选生成的 RubricSet 是否属于当前任务且数量合规。"""

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
        """校验单候选结果的绑定关系和 Rubric 覆盖完整性。"""

        if result.task_id != task.task_id:
            raise ValueError("RewardResult.task_id does not match the task")
        if result.candidate_id != candidate.candidate_id:
            raise ValueError("RewardResult.candidate_id does not match the candidate")

        judged_ids = [judgment.rubric_id for judgment in result.judgments]
        if len(judged_ids) != len(set(judged_ids)):
            raise ValueError("a RewardResult may judge each rubric only once")
        if frozenset(judged_ids) != rubrics.rubric_ids:
            raise ValueError("judgments must cover every shared rubric")
