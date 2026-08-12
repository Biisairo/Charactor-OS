"""LLM 호출 계측 — 턴당 호출 수·토큰을 집계한다.

호출 지점마다 계측 코드를 심으면 한 곳만 빠뜨려도 집계가 조용히 틀린다.
대신 클라이언트를 얇게 감싸 모든 호출을 자동으로 포착한다.
라벨은 감쌀 때 정해지므로 스레드 간 전파 문제가 없다 — Stage 3은 워커 스레드에서
돌기 때문에 컨텍스트 변수 방식은 쓸 수 없다.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from src.validity import provider_error_reason


@dataclass(frozen=True)
class CallRecord:
    """LLM 호출 1회의 기록."""

    label: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    duration_ms: float
    error: str = ""
    # 원문 — 운영 로그에서 "왜 이렇게 답했는가"를 재구성하기 위한 것.
    # 수집 여부는 CallMeter.capture_payload가 정한다.
    messages: list | None = None
    response: str = ""
    # 프로바이더가 usage를 돌려줬는가. False면 위 토큰 수는 "0"이 아니라 "미상"이다.
    # 둘을 뭉뚱그리면 비용이 과소 추정되었는지 판단할 수 없다 (REQ-12-1).
    usage_known: bool = True
    # 응답이 캐릭터 발화가 아니라 프로바이더 거부인가 (REQ-12-5).
    # 거부는 과금되지 않아 토큰 0으로 오므로, 표시하지 않으면 usage 누락과 섞인다.
    refused: bool = False

    @property
    def failed(self) -> bool:
        return bool(self.error)


class MeteredClient:
    """클라이언트를 감싸 호출을 기록한다. 그 외 동작은 원본과 동일하다."""

    def __init__(self, client, meter: CallMeter, label: str):
        self._client = client
        self._meter = meter
        self._label = label

    def _extract_messages(self, args, kwargs) -> list | None:
        """호출 인자에서 messages를 꺼낸다. 위치/키워드 양쪽을 받는다."""
        if not self._meter.capture_payload:
            return None
        messages = kwargs.get("messages")
        if messages is None and args:
            messages = args[0]
        return messages if isinstance(messages, list) else None

    def call_llm(self, *args, **kwargs):
        started = time.perf_counter()
        messages = self._extract_messages(args, kwargs)
        try:
            result = self._client.call_llm(*args, **kwargs)
        except Exception as exc:
            # 실패한 호출이야말로 운영에서 가장 중요한 신호다. 기록하고 그대로 올린다.
            self._meter.record(
                CallRecord(
                    label=self._label,
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    duration_ms=(time.perf_counter() - started) * 1000,
                    error=f"{type(exc).__name__}: {exc}",
                    messages=messages,
                    usage_known=False,
                )
            )
            raise

        duration_ms = (time.perf_counter() - started) * 1000
        usage = getattr(result, "usage", None)
        content = getattr(result, "content", "") or ""
        self._meter.record(
            CallRecord(
                label=self._label,
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                total_tokens=getattr(usage, "total_tokens", 0) or 0,
                duration_ms=duration_ms,
                messages=messages,
                response=content if self._meter.capture_payload else "",
                usage_known=usage is not None,
                # 판별은 여기서 한다. 모든 호출이 이 지점을 지나므로 라벨을 가리지 않고
                # 잡힌다 — 호출 지점마다 검사를 심으면 한 곳만 빠뜨려도 조용히 샌다.
                refused=provider_error_reason(content) is not None,
            )
        )
        return result

    def __getattr__(self, name):
        """계측하지 않는 속성은 원본에 위임한다."""
        return getattr(self._client, name)


@dataclass
class CallMeter:
    """턴 단위로 LLM 호출을 모은다.

    두 소비자가 같은 기록을 쓴다.

    - **턴 스냅샷**: `summary()` — `--trace` 출력과 턴 요약 로그
    - **호출 스트림**: `sink` — 호출마다 즉시 불리는 콜백 (운영 로그)

    기록은 `trace` 같은 디버그 플래그와 무관하게 항상 모은다. 운영 로그의 턴 요약이
    그 기록에 의존하기 때문이다. `reset()`이 턴마다 비우므로 메모리는 한 턴 분량으로 제한된다.
    """

    sink: Callable[[CallRecord], None] | None = None
    # 프롬프트·응답 원문을 함께 수집할지. 운영 로그에서 대화를 재구성하려면 필요하다.
    capture_payload: bool = True
    _records: list[CallRecord] = field(default_factory=list)

    def reset(self) -> None:
        self._records = []

    def record(self, entry: CallRecord) -> None:
        self._records.append(entry)
        if self.sink is not None:
            # 로그 실패가 대화를 막아서는 안 된다
            with contextlib.suppress(Exception):
                self.sink(entry)

    def wrap(self, client, label: str):
        """`label`로 집계되는 클라이언트를 반환한다."""
        return MeteredClient(client, self, label)

    @property
    def records(self) -> list[CallRecord]:
        return list(self._records)

    def summary(self) -> dict:
        """호출 수·토큰을 전체와 라벨별로 집계한다."""
        by_label: dict[str, dict] = {}
        for r in self._records:
            bucket = by_label.setdefault(
                r.label,
                {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "duration_ms": 0.0},
            )
            bucket["calls"] += 1
            bucket["prompt_tokens"] += r.prompt_tokens
            bucket["completion_tokens"] += r.completion_tokens
            bucket["duration_ms"] += r.duration_ms

        for bucket in by_label.values():
            bucket["duration_ms"] = round(bucket["duration_ms"], 1)

        # 비용이 과소 추정되는 경우는 "청구됐을 텐데 usage가 없는" 호출뿐이다.
        # 거부·실패는 청구되지 않으므로 usage가 없어도 하한 경고의 근거가 아니다.
        # 셋을 뭉뚱그리면 정상 턴마다 오경보가 뜬다 — 실측에서 확인했다.
        unaccounted = sum(
            1 for r in self._records if not r.usage_known and not r.refused and not r.failed
        )

        return {
            "calls": len(self._records),
            "failed_calls": sum(1 for r in self._records if r.failed),
            # 성공했고 거부도 아닌데 usage가 없는 호출. 있으면 토큰 합계는 하한이다.
            "unknown_usage_calls": unaccounted,
            "refused_calls": sum(1 for r in self._records if r.refused),
            "tokens_are_lower_bound": unaccounted > 0,
            "prompt_tokens": sum(r.prompt_tokens for r in self._records),
            "completion_tokens": sum(r.completion_tokens for r in self._records),
            "total_tokens": sum(r.total_tokens for r in self._records),
            "by_label": dict(sorted(by_label.items())),
        }
