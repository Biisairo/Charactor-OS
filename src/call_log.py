"""LLM 호출 운영 로그 — 비동기 · 회전 · 항상 켜짐.

계측(`src/metrics.py`)은 턴 단위 스냅샷이라 다음 턴에 덮어쓴다. 운영 중
"어제 비용이 얼마였나", "언제부터 느려졌나", "어떤 호출이 실패했나"에 답하려면
호출이 누적되어야 한다.

설계 요건
    1. **대화 경로를 막지 않는다.** 디스크 쓰기가 응답 지연에 더해지면 안 된다.
       `QueueHandler` + `QueueListener`로 쓰기를 별도 스레드에 넘긴다.
    2. **무한히 커지지 않는다.** `RotatingFileHandler`로 크기 기반 회전.
    3. **로깅 실패가 대화를 막지 않는다.** 핸들러 오류는 삼키고 대화를 계속한다.
    4. **실패한 호출도 남긴다.** 운영에서 가장 중요한 신호다.
    5. **디버그 플래그와 무관하게 동작한다.** 운영 로그는 항상 켜져 있어야 한다.

직접 스레드를 만들지 않고 stdlib 패턴을 쓴 이유는, 회전·큐·종료 처리가
이미 검증되어 있기 때문이다.
"""

from __future__ import annotations

import atexit
import contextlib
import json
import logging
import logging.handlers
import queue
from datetime import datetime
from pathlib import Path

from src.metrics import CallRecord

LOGGER_NAME = "character_os.llm_calls"
DEFAULT_LOG_PATH = Path("logs/llm_calls.jsonl")

# 프롬프트 원문을 담으므로 호출당 최대 ~12KB, 턴당 ~40KB가 된다.
# 파일당 50MB, 10개 보관 → 최대 500MB (약 12,000턴)
DEFAULT_MAX_BYTES = 50 * 1024 * 1024
DEFAULT_BACKUP_COUNT = 10

# 한 필드가 이 길이를 넘으면 자른다. 자른 사실은 반드시 표시한다.
DEFAULT_MAX_FIELD_CHARS = 20_000

# 큐가 가득 차면 기록을 버린다. 로깅 때문에 대화가 멈추는 것보다 낫다.
DEFAULT_QUEUE_SIZE = 10_000


class JsonlFormatter(logging.Formatter):
    """레코드에 실린 dict를 JSON 한 줄로 만든다."""

    def format(self, record: logging.LogRecord) -> str:
        payload = getattr(record, "payload", None)
        if payload is None:
            payload = {"message": record.getMessage()}
        return json.dumps(payload, ensure_ascii=False)


class CallLogger:
    """LLM 호출을 JSONL 파일에 비동기로 누적한다."""

    def __init__(
        self,
        path: Path | str | None = None,
        *,
        max_bytes: int = DEFAULT_MAX_BYTES,
        backup_count: int = DEFAULT_BACKUP_COUNT,
        queue_size: int = DEFAULT_QUEUE_SIZE,
        enabled: bool = True,
        capture_payload: bool = True,
        max_field_chars: int = DEFAULT_MAX_FIELD_CHARS,
    ):
        self.enabled = enabled
        self.capture_payload = capture_payload
        self.max_field_chars = max_field_chars
        self.path = Path(path) if path else DEFAULT_LOG_PATH
        self._listener: logging.handlers.QueueListener | None = None
        self._logger = logging.getLogger(f"{LOGGER_NAME}.{id(self)}")
        self._logger.propagate = False
        self._logger.setLevel(logging.INFO)

        if not enabled:
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            self.path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        file_handler.setFormatter(JsonlFormatter())

        # 큐가 차면 put_nowait이 실패하고 handleError로 넘어간다 — 블로킹하지 않는다
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._logger.addHandler(logging.handlers.QueueHandler(self._queue))

        self._listener = logging.handlers.QueueListener(
            self._queue, file_handler, respect_handler_level=True
        )
        self._listener.start()
        atexit.register(self.shutdown)

    def _truncate(self, text: str) -> str:
        """긴 필드를 자른다. 잘렸다는 사실을 감추지 않는다."""
        if self.max_field_chars <= 0 or len(text) <= self.max_field_chars:
            return text
        dropped = len(text) - self.max_field_chars
        return text[: self.max_field_chars] + f"…[{dropped:,}자 잘림]"

    def _payload_fields(self, record: CallRecord) -> dict:
        """프롬프트·응답 원문. 로그만 보고 대화를 재구성할 수 있게 한다."""
        if not self.capture_payload:
            return {}
        return {
            "messages": [
                {"role": m.get("role", ""), "content": self._truncate(str(m.get("content", "")))}
                for m in (record.messages or [])
                if isinstance(m, dict)
            ],
            "response": self._truncate(record.response),
        }

    def log_turn(
        self,
        *,
        turn_id: str,
        character: str,
        user_input: str,
        response: str,
        metrics: dict,
        duration_ms: float,
        error: str = "",
        extra: dict | None = None,
    ) -> None:
        """턴 1건의 요약. 호출 단위 기록과 turn_id로 이어진다.

        `extra`는 턴마다 달라지는 부가 요약(뇌의 루프 수·사용 도구 등)이다.
        """
        if not self.enabled:
            return

        payload = {
            "ts": datetime.now().isoformat(timespec="milliseconds"),
            "event": "turn",
            "turn_id": turn_id,
            "character": character,
            "user_input": self._truncate(user_input),
            "response": self._truncate(response),
            "duration_ms": round(duration_ms, 1),
            "calls": metrics.get("calls", 0),
            "failed_calls": metrics.get("failed_calls", 0),
            "prompt_tokens": metrics.get("prompt_tokens", 0),
            "completion_tokens": metrics.get("completion_tokens", 0),
            "total_tokens": metrics.get("total_tokens", 0),
            "cost_usd": metrics.get("cost_usd"),
            "model": metrics.get("model", ""),
            "by_label": metrics.get("by_label", {}),
            "error": error,
            **(extra or {}),
        }
        with contextlib.suppress(Exception):
            self._logger.info("", extra={"payload": payload})

    def log_call(
        self,
        record: CallRecord,
        *,
        model: str = "",
        cost_usd: float | None = None,
        turn_id: str = "",
        character: str = "",
    ) -> None:
        """호출 1건을 기록한다. 블로킹하지 않으며 예외를 밖으로 내지 않는다."""
        if not self.enabled:
            return

        payload = {
            "ts": datetime.now().isoformat(timespec="milliseconds"),
            "event": "call",
            "turn_id": turn_id,
            "character": character,
            "label": record.label,
            "model": model,
            "prompt_tokens": record.prompt_tokens,
            "completion_tokens": record.completion_tokens,
            "total_tokens": record.total_tokens,
            "duration_ms": round(record.duration_ms, 1),
            "cost_usd": cost_usd,
            "error": record.error,
            **self._payload_fields(record),
        }
        # 로깅 실패가 대화를 막아서는 안 된다
        with contextlib.suppress(Exception):
            self._logger.info("", extra={"payload": payload})

    def flush(self) -> None:
        """큐에 쌓인 기록을 디스크까지 밀어낸다 (테스트·종료용)."""
        if self._listener is None:
            return
        # QueueListener는 sentinel을 만나야 큐를 비운다. stop→start로 강제 배출한다.
        self._listener.stop()
        self._listener.start()

    def shutdown(self) -> None:
        if self._listener is None:
            return
        self._listener.stop()
        self._listener = None


def from_config(config: dict | None) -> CallLogger:
    """`config.yaml`의 `call_log` 섹션으로 로거를 만든다.

    설정이 없으면 기본값으로 동작한다 — 운영 로그는 설정을 잊어도 남아야 한다.
    """
    section = (config or {}).get("call_log") or {}
    return CallLogger(
        path=section.get("path") or DEFAULT_LOG_PATH,
        max_bytes=int(section.get("max_bytes", DEFAULT_MAX_BYTES)),
        backup_count=int(section.get("backup_count", DEFAULT_BACKUP_COUNT)),
        enabled=bool(section.get("enabled", True)),
        capture_payload=bool(section.get("capture_payload", True)),
        max_field_chars=int(section.get("max_field_chars", DEFAULT_MAX_FIELD_CHARS)),
    )
