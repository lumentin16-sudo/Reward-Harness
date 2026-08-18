"""仅连接本地 vLLM 服务的 reward benchmark 模型客户端。"""

from __future__ import annotations

import threading
import time
import uuid
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal


ModelRole = Literal["rubric", "judge"]


@dataclass(frozen=True, slots=True)
class VLLMCallRecord:
    """一次本地 vLLM 推理请求的审计记录。"""

    call_id: str
    role: ModelRole
    prompt: str
    response: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    error: str | None = None
    request_id: str | None = None
    cached: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class VLLMBackend:
    """共享本地 vLLM client，并为每个评测阶段创建记录器。"""

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8000/v1",
        model: str = "Qwen/Qwen3-8B",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        timeout_seconds: float = 120.0,
        request_retries: int = 2,
        request_workers: int = 16,
        cache_dir: Path | None = None,
    ) -> None:
        if request_workers < 1:
            raise ValueError("request_workers must be at least 1")

        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on local environment
            raise RuntimeError(
                "The openai package is required; install reward_agent/requirements.txt"
            ) from exc

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.request_retries = request_retries
        self.request_workers = request_workers
        self.cache_dir = cache_dir
        self._request_slots = threading.BoundedSemaphore(request_workers)
        self._cache_lock = threading.Lock()
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = OpenAI(
            # vLLM 的本地 OpenAI-compatible server 不需要真实密钥。
            api_key="EMPTY",
            base_url=base_url,
            timeout=timeout_seconds,
            max_retries=request_retries,
        )

    def _thinking_extra_body(self) -> dict[str, Any]:
        """通过 vLLM chat template 参数关闭 Qwen3 thinking。"""

        return {"chat_template_kwargs": {"enable_thinking": False}}

    def recorder(self, role: ModelRole, *, use_cache: bool = True) -> "RecordingLLM":
        return RecordingLLM(self, role, use_cache=use_cache)

    def _cache_path(self, role: ModelRole, prompt: str) -> Path | None:
        if self.cache_dir is None:
            return None
        payload = {
            "version": 2,
            "role": role,
            "backend": "vllm",
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "enable_thinking": False,
            "prompt": prompt,
        }
        digest = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def _read_cache(self, path: Path | None) -> dict[str, Any] | None:
        if path is None or not path.exists():
            return None
        try:
            with self._cache_lock:
                return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _write_cache(self, path: Path | None, payload: dict[str, Any]) -> None:
        if path is None:
            return
        temporary = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
        try:
            with self._cache_lock:
                temporary.write_text(
                    json.dumps(payload, ensure_ascii=False), encoding="utf-8"
                )
                temporary.replace(path)
        except OSError:
            # 缓存失败不应影响可信 evaluator 的评分主流程。
            pass

    def invalidate_cache(self, role: ModelRole, prompt: str) -> None:
        """解析或接口校验失败时删除对应响应，确保下一次重试真正访问模型。"""

        path = self._cache_path(role, prompt)
        if path is None:
            return
        try:
            with self._cache_lock:
                path.unlink(missing_ok=True)
        except OSError:
            pass

    def _complete(
        self, role: ModelRole, prompt: str, *, use_cache: bool = True
    ) -> VLLMCallRecord:
        started = time.perf_counter()
        call_id = str(uuid.uuid4())
        cache_path = self._cache_path(role, prompt) if use_cache else None
        cached = self._read_cache(cache_path)
        if cached is not None and isinstance(cached.get("response"), str):
            return VLLMCallRecord(
                call_id=call_id,
                role=role,
                prompt=prompt,
                response=cached["response"],
                input_tokens=0,
                output_tokens=0,
                latency_ms=(time.perf_counter() - started) * 1000,
                request_id=cached.get("request_id"),
                cached=True,
            )
        try:
            # rubric 和候选评分共用一个全局并发上限，避免外层/内层线程池叠加压垮服务端。
            with self._request_slots:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    # 关闭 thinking 以减少输出 token，并让 JSON 更稳定。
                    extra_body=self._thinking_extra_body(),
                )
            content = response.choices[0].message.content or ""
            if not content.strip():
                raise ValueError("model returned an empty response")
            usage = response.usage
            record = VLLMCallRecord(
                call_id=call_id,
                role=role,
                prompt=prompt,
                response=content,
                input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
                latency_ms=(time.perf_counter() - started) * 1000,
                request_id=getattr(response, "id", None),
            )
            self._write_cache(
                cache_path,
                {
                    "response": content,
                    "request_id": record.request_id,
                },
            )
            return record
        except Exception as exc:
            return VLLMCallRecord(
                call_id=call_id,
                role=role,
                prompt=prompt,
                response="",
                input_tokens=0,
                output_tokens=0,
                latency_ms=(time.perf_counter() - started) * 1000,
                error=f"{type(exc).__name__}: {exc}",
            )


class RecordingLLM:
    """实现 ``LLMCallable``，并线程安全地保留本次样本的调用记录。"""

    def __init__(
        self, backend: VLLMBackend, role: ModelRole, *, use_cache: bool = True
    ) -> None:
        self._backend = backend
        self._role = role
        self._use_cache = use_cache
        self._records: list[VLLMCallRecord] = []
        self._lock = threading.Lock()
        self._thread_state = threading.local()

    def __call__(self, prompt: str) -> str:
        self._thread_state.last_prompt = prompt
        record = self._backend._complete(
            self._role, prompt, use_cache=self._use_cache
        )
        with self._lock:
            self._records.append(record)
        if record.error:
            raise RuntimeError(record.error)
        return record.response

    def invalidate_last(self) -> None:
        """只失效当前评分线程最后调用的 Prompt，避免并发候选互相干扰。"""

        prompt = getattr(self._thread_state, "last_prompt", None)
        if isinstance(prompt, str):
            self._backend.invalidate_cache(self._role, prompt)

    @property
    def records(self) -> tuple[VLLMCallRecord, ...]:
        with self._lock:
            return tuple(self._records)
