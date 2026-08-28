"""Dynamic-Rubric Reward Agent 的稳定协议与安全边界。

本模块只负责所有候选 Harness 必须共同遵守的部分：

1. 定义任务、候选回答、Rubric、Skill 和评分结果的数据结构；
2. 限制 Rubric 生成阶段（G）和 Judge 阶段（J）各自可见的信息；
3. 将内部对象转换成稳定的 Prompt payload，并解析模型返回的 JSON；
4. 校验 Harness 输出、Rubric 覆盖关系和最终奖励；
5. 保持业务结果可序列化，完整执行轨迹由 benchmark runner 统一记录。

候选 Harness 的搜索边界有四个公开接口：``get_skill_registry``、
``retrieve_skills``、``build_rubrics`` 和 ``judge``。Judge 一次看到完整
Responses 并直接返回唯一 winner，不再暴露独立的逐回答评分和奖励聚合接口。
"""

from __future__ import annotations

import json
import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Protocol, final


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
class Query:
    """公开任务信息。

    Rubric Model 在 G 阶段可以同时看到该对象和匿名 Responses，但看不到真实偏好。
    """

    query_id: str  # benchmark 内稳定且唯一的任务标识
    instruction: str  # 用户要求或待完成的任务正文
    context: str | None = None  # 允许公开给评价器的补充上下文
    domain: str | None = None  # 可选领域标签，例如 math / coding
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)  # 公开扩展字段

    def __post_init__(self) -> None:
        _require_non_empty(self.query_id, "query_id")
        _require_non_empty(self.instruction, "instruction")
        _json_safe(dict(self.metadata), "metadata")


@dataclass(frozen=True, slots=True)
class Response:
    """一个待比较回答或可审计的 Agent 轨迹。

    evaluator 会先稳定重排并替换原始 ID，再把完整匿名 Responses 交给 Judge。
    """

    response_id: str  # evaluator 用于绑定预测结果，不属于 Judge 输入
    content: str  # 当前 Judge 唯一可见的候选正文
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)  # evaluator 侧信息

    def __post_init__(self) -> None:
        _require_non_empty(self.response_id, "response_id")
        _require_non_empty(self.content, "content")
        _json_safe(dict(self.metadata), "metadata")


# ---------------------------------------------------------------------------
# Rubric 数据结构
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Rubric:
    """一条可独立评分的任务相关评价标准。"""

    rubric_id: str  # RubricSet 内唯一，供 Judgment 精确引用
    criterion: str  # 应检查的单一、可观察属性
    weight: float = 1.0  # 供具体 Harness 聚合逻辑使用的相对权重
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
    """同一 Query 下由全部 Response 共享的 Rubric 集合。

    Benchmark 每题只生成一次 RubricSet，并把同一个对象交给完整 Responses 上的
    forced-choice ``judge``。
    """

    query_id: str  # Rubric 所属任务
    rubrics: tuple[Rubric, ...]  # 不可变集合，评分阶段不能原地修改
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.query_id, "query_id")
        ids = [rubric.rubric_id for rubric in self.rubrics]
        if len(ids) != len(set(ids)):
            raise ValueError("rubric ids must be unique within a RubricSet")
        _json_safe(dict(self.metadata), "metadata")

    @property
    def rubric_ids(self) -> frozenset[str]:
        """返回 Rubric ID 集合，供完整覆盖校验使用。"""

        return frozenset(rubric.rubric_id for rubric in self.rubrics)


# ---------------------------------------------------------------------------
# Workflow Skill 协议
# ---------------------------------------------------------------------------

SkillStage = Literal["G", "J"]

@dataclass(frozen=True, slots=True)
class Skill:
    """可由模型选择并注入 Prompt 的静态 workflow 指令。"""

    name: str
    stage: SkillStage
    description: str
    content: str

    def __post_init__(self) -> None:
        _require_non_empty(self.name, "skill name")
        if self.stage not in {"G", "J"}:
            raise ValueError(f"unknown skill stage: {self.stage!r}")
        _require_non_empty(self.description, "skill description")
        _require_non_empty(self.content, "skill content")


@dataclass(frozen=True, slots=True)
class SkillRegistry:
    """向模型暴露 Skill Catalog，并按名称返回选中的 workflow 指令。"""

    skills: tuple[Skill, ...] = ()

    def __post_init__(self) -> None:
        keys: list[tuple[SkillStage, str]] = []
        for skill in self.skills:
            if not isinstance(skill, Skill):
                raise TypeError("SkillRegistry accepts only Skill instances")
            keys.append((skill.stage, skill.name))
        if len(keys) != len(set(keys)):
            raise ValueError("skill names must be unique within each stage")

    def for_stage(self, stage: SkillStage) -> "SkillRegistry":
        """返回只包含指定 G/J 阶段 Skill 的选择池。"""

        if stage not in {"G", "J"}:
            raise ValueError(f"unknown skill stage: {stage!r}")
        return SkillRegistry(tuple(skill for skill in self.skills if skill.stage == stage))

    @property
    def names(self) -> frozenset[str]:
        """返回可被模型选择的全部 Skill 名称。"""

        return frozenset(skill.name for skill in self.skills)

    @property
    def catalog(self) -> tuple[dict[str, str], ...]:
        """生成模型选择 Skill 时可见的最小目录。"""

        return tuple(
            {
                "name": skill.name,
                "stage": skill.stage,
                "description": skill.description,
            }
            for skill in self.skills
        )

    def select(self, skill_names: tuple[str, ...]) -> tuple[Skill, ...]:
        """按模型给出的顺序返回 Skill，并拒绝重复或未知名称。"""

        if len(skill_names) != len(set(skill_names)):
            raise ValueError("a model may call each skill at most once per stage")
        by_name: dict[str, Skill] = {}
        for skill in self.skills:
            if skill.name in by_name:
                raise ValueError("select skills from a stage-specific registry")
            by_name[skill.name] = skill
        unknown = set(skill_names) - set(by_name)
        if unknown:
            raise ValueError(f"unknown skill names: {sorted(unknown)}")

        return tuple(by_name[name] for name in skill_names)


# ---------------------------------------------------------------------------
# Judge 输出与最终比较结果
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class RubricJudgment:
    """Judge 对单条 Rubric 给出的原始数值评分与可审计依据。"""

    rubric_id: str  # 必须引用共享 RubricSet 中存在的 ID
    score: float  # 原始评分尺度由具体 Harness 定义
    evidence: tuple[str, ...] = ()  # 候选正文中的支持或反驳证据
    confidence: float | None = None  # 可选的 0..1 置信度，仅用于审计
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.rubric_id, "rubric_id")
        if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
            raise ValueError("criterion score must be numeric")
        _require_finite(self.score, "criterion score")
        if self.confidence is not None:
            _require_finite(self.confidence, "confidence")
            if not 0 <= self.confidence <= 1:
                raise ValueError("confidence must be in [0, 1]")
        _json_safe(dict(self.metadata), "metadata")


@dataclass(frozen=True, slots=True)
class JudgmentResult:
    """J 阶段对单个 Response 产生的 RubricJudgment 集合。"""

    query_id: str
    response_id: str
    judgments: tuple[RubricJudgment, ...]
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.query_id, "query_id")
        _require_non_empty(self.response_id, "response_id")
        _json_safe(dict(self.metadata), "metadata")


@dataclass(frozen=True, slots=True)
class RewardResult:
    """A 阶段对单个 Response 产生的归一化最终奖励。"""

    query_id: str  # 防止结果被绑定到其他任务
    response_id: str  # 防止结果被绑定到另一个候选
    reward: float  # 最终标量奖励，范围固定为 [0, 1]
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.query_id, "query_id")
        _require_non_empty(self.response_id, "response_id")
        _require_finite(self.reward, "reward")
        if not 0 <= self.reward <= 1:
            raise ValueError("reward must be in [0, 1]")
        _json_safe(dict(self.metadata), "metadata")


@dataclass(frozen=True, slots=True)
class WinnerResult:
    """J 阶段对完整 Responses 做 forced-choice 后产生的唯一 winner。"""

    query_id: str
    winner_response_id: str
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.query_id, "query_id")
        _require_non_empty(self.winner_response_id, "winner_response_id")
        _json_safe(dict(self.metadata), "metadata")


# ---------------------------------------------------------------------------
# Reward Harness 稳定接口与公共控制流程
# ---------------------------------------------------------------------------

class RewardSystem(ABC):
    """所有候选 Reward Harness 必须实现的稳定接口。

    可搜索部分：
        - ``get_skill_registry``：Skill 的内容、数量和组织方式；
        - ``retrieve_skills``：根据 Query、Responses 和 G/J 阶段选择 Skill；
        - ``build_rubrics``：Rubric Prompt、Skill 选择和 G 调用流程；
        - ``judge``：完整 Responses 上的比较 Prompt、Skill 选择和 winner 输出。

    固定部分：
        - 输入 payload 与 JSON 输出协议；
        - Query、Responses、RubricSet 和 WinnerResult 的绑定关系。
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """在候选类定义时立即拒绝对固定协议的覆盖。"""

        super().__init_subclass__(**kwargs)
        # @final 主要服务静态检查；这里再做运行时检查，防止生成代码绕过约束。
        fixed_methods = {
            "_task_payload",
            "_candidate_payload",
            "_responses_payload",
            "_judge_responses_payload",
            "_rubrics_payload",
            "_skills_payload",
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
    ) -> None:
        """注入冻结的 Rubric Model 和 Judge Model。"""

        self._rubric_llm = rubric_llm
        self._judge_llm = judge_llm

    @property
    def rubric_llm(self) -> LLMCallable:
        """Rubric 生成阶段使用的冻结模型。"""

        return self._rubric_llm

    @property
    def judge_llm(self) -> LLMCallable:
        """完整 Responses 比较阶段使用的冻结模型。"""

        return self._judge_llm

    @abstractmethod
    def get_skill_registry(self, task: Query) -> SkillRegistry:
        """候选接口 1：返回当前 Harness 可用的 Skill bank。"""

    def retrieve_skills(
        self,
        task: Query,
        responses: tuple[Response, ...],
        stage: SkillStage,
    ) -> tuple[Skill, ...]:
        """候选接口 2：为一次 G/J 调用选择要注入的 Skills。

        默认实现只按阶段返回全部可用 Skill。需要 query/response-aware 检索时，
        candidate 可以覆盖这个方法。
        """

        del responses
        return self.get_skill_registry(task).for_stage(stage).skills

    @abstractmethod
    def build_rubrics(
        self,
        task: Query,
        responses: tuple[Response, ...],
    ) -> RubricSet:
        """候选接口 3（G）：根据 Query 和匿名 Responses 生成共享 Rubric。"""

    @abstractmethod
    def judge(
        self,
        task: Query,
        responses: tuple[Response, ...],
        rubrics: RubricSet,
    ) -> WinnerResult:
        """候选接口 4（J）：比较完整 Responses 并返回唯一 winner。"""

    @staticmethod
    @final
    def _task_payload(task: Query) -> dict[str, JSONValue]:
        """将公开任务转换为所有候选共用的 Prompt payload。

        这里保留任务 metadata，因为它属于 benchmark 明确提供的公开任务信息；
        evaluator 必须保证其中不包含 preference label 等评测答案。
        """

        return {
            "query_id": task.query_id,
            "instruction": task.instruction,
            "context": task.context,
            "domain": task.domain,
            "metadata": dict(task.metadata),
        }

    @staticmethod
    @final
    def _candidate_payload(candidate: Response) -> dict[str, JSONValue]:
        """只暴露当前回答内容，避免候选 ID 或 evaluator metadata 泄露。"""

        return {"content": candidate.content}

    @staticmethod
    @final
    def _responses_payload(
        responses: tuple[Response, ...],
    ) -> list[dict[str, JSONValue]]:
        """向 G 阶段暴露全部回答正文，但不暴露 ID、metadata、原始位置或 gold。"""

        return [{"content": response.content} for response in responses]

    @staticmethod
    @final
    def _judge_responses_payload(
        responses: tuple[Response, ...],
    ) -> list[dict[str, JSONValue]]:
        """给 Judge 提供匿名 Response ID 与正文，供 winner 结果精确绑定。"""

        return [
            {"response_id": response.response_id, "content": response.content}
            for response in responses
        ]

    @staticmethod
    @final
    def _rubrics_payload(rubrics: RubricSet) -> list[dict[str, JSONValue]]:
        """将共享 Rubric 转换为 Judge 可见的稳定 payload。

        Judge 只需要知道“评什么”，不需要看到 G 阶段日志、模型响应、聚合权重
        和其他 evaluator metadata。Rubric 权重由具体 Harness 的聚合逻辑读取。
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
    def _skills_payload(
        skills: tuple[Skill, ...],
    ) -> list[dict[str, JSONValue]]:
        """将选中的 workflow Skill 转换为稳定 Prompt payload。

        description 只用于选择阶段；选中后只向工作 Prompt 注入名称和内容。
        """

        return [
            {
                "name": skill.name,
                "stage": skill.stage,
                "content": skill.content,
            }
            for skill in skills
        ]

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
    ) -> tuple[RubricJudgment, ...]:
        """解析固定 Judge JSON 协议并检查 Rubric 覆盖完整性。

        Judge 必须对共享 RubricSet 中每条 Rubric 恰好评分一次：遗漏、重复或
        额外生成的 rubric_id 都会使本次评估失败。
        """

        payload = _extract_json_object(raw_response)
        raw_judgments = payload.get("judgments")
        if not isinstance(raw_judgments, list):
            raise ValueError("judge response must contain a judgments list")

        parsed: dict[str, RubricJudgment] = {}
        for index, raw in enumerate(raw_judgments):
            if not isinstance(raw, dict):
                raise ValueError(f"judgment at index {index} must be an object")
            rubric_id = str(raw.get("rubric_id", ""))
            if rubric_id in parsed:
                raise ValueError(f"duplicate judgment for rubric {rubric_id!r}")
            score = raw.get("score")
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise ValueError(f"score for rubric {rubric_id!r} must be numeric")

            # 为兼容模型偶尔返回单个字符串，内部统一转换成 tuple[str, ...]。
            evidence_raw = raw.get("evidence", [])
            if isinstance(evidence_raw, str):
                evidence = (evidence_raw,)
            elif isinstance(evidence_raw, list):
                evidence = tuple(str(item) for item in evidence_raw)
            else:
                raise ValueError(f"evidence for rubric {rubric_id!r} must be a list")

            # confidence 是可选审计信号；是否参与聚合由具体 Harness 决定。
            confidence_raw = raw.get("confidence")
            confidence = None if confidence_raw is None else float(confidence_raw)
            parsed[rubric_id] = RubricJudgment(
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

    def _validate_rubric_set(self, task: Query, rubrics: RubricSet) -> None:
        """校验 Harness 生成的 RubricSet 是否属于当前 Query。"""

        if rubrics.query_id != task.query_id:
            raise ValueError("RubricSet.query_id does not match the task")

    @staticmethod
    def _validate_judgment_result(
        task: Query,
        candidate: Response,
        rubrics: RubricSet,
        result: JudgmentResult,
    ) -> None:
        """校验 J 阶段结果的绑定关系和 Rubric 覆盖完整性。"""

        if result.query_id != task.query_id:
            raise ValueError("JudgmentResult.query_id does not match the task")
        if result.response_id != candidate.response_id:
            raise ValueError("JudgmentResult.response_id does not match the response")

        judged_ids = [judgment.rubric_id for judgment in result.judgments]
        if len(judged_ids) != len(set(judged_ids)):
            raise ValueError("a JudgmentResult may judge each rubric only once")
        if frozenset(judged_ids) != rubrics.rubric_ids:
            raise ValueError("judgments must cover every shared rubric")

    @staticmethod
    def _validate_reward_result(
        task: Query,
        candidate: Response,
        result: RewardResult,
    ) -> None:
        """校验 A 阶段最终奖励与当前 Query/Response 的绑定关系。"""

        if result.query_id != task.query_id:
            raise ValueError("RewardResult.query_id does not match the task")
        if result.response_id != candidate.response_id:
            raise ValueError("RewardResult.response_id does not match the response")

    @staticmethod
    def _validate_winner_result(
        task: Query,
        responses: tuple[Response, ...],
        result: WinnerResult,
    ) -> None:
        """校验 winner 属于当前 Query 的完整匿名 Response 集合。"""

        if result.query_id != task.query_id:
            raise ValueError("WinnerResult.query_id does not match the query")
        response_ids = {response.response_id for response in responses}
        if result.winner_response_id not in response_ids:
            raise ValueError("WinnerResult references an unknown response")
