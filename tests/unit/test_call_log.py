"""LLM 호출 운영 로그 (TASK-10).

운영 로그의 요건은 디버그 출력과 다르다. 항상 켜져 있고, 누적되며,
파일이 무한히 커지지 않고, 실패해도 대화를 막지 않아야 한다.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.call_log import CallLogger
from src.metrics import CallRecord


def _record(label: str = "response", error: str = "") -> CallRecord:
    return CallRecord(
        label=label,
        prompt_tokens=1000,
        completion_tokens=50,
        total_tokens=1050,
        duration_ms=123.4,
        error=error,
    )


def _read_lines(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class TestWriting:
    def test_writes_jsonl(self, tmp_path: Path):
        path = tmp_path / "calls.jsonl"
        logger = CallLogger(path)
        try:
            logger.log_call(_record(), model="m", cost_usd=0.001, turn_id="t1")
            logger.flush()
        finally:
            logger.shutdown()

        entries = _read_lines(path)
        assert len(entries) == 1
        assert entries[0]["label"] == "response"
        assert entries[0]["model"] == "m"
        assert entries[0]["turn_id"] == "t1"

    def test_accumulates_across_calls(self, tmp_path: Path):
        """운영 로그는 누적되어야 한다. 덮어쓰면 이력이 사라진다."""
        path = tmp_path / "calls.jsonl"
        logger = CallLogger(path)
        try:
            for _ in range(5):
                logger.log_call(_record())
            logger.flush()
        finally:
            logger.shutdown()

        assert len(_read_lines(path)) == 5

    def test_records_all_fields(self, tmp_path: Path):
        path = tmp_path / "calls.jsonl"
        logger = CallLogger(path)
        try:
            logger.log_call(_record(), model="m", cost_usd=0.5, turn_id="t", character="c")
            logger.flush()
        finally:
            logger.shutdown()

        entry = _read_lines(path)[0]
        for key in (
            "ts",
            "turn_id",
            "character",
            "label",
            "model",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "duration_ms",
            "cost_usd",
            "error",
        ):
            assert key in entry, key

    def test_logs_failed_calls(self, tmp_path: Path):
        """실패한 호출이 운영에서 가장 중요한 신호다."""
        path = tmp_path / "calls.jsonl"
        logger = CallLogger(path)
        try:
            logger.log_call(_record(error="RuntimeError: 거부됨"), model="m")
            logger.flush()
        finally:
            logger.shutdown()

        assert "거부됨" in _read_lines(path)[0]["error"]

    def test_turn_id_groups_calls(self, tmp_path: Path):
        """같은 턴의 호출을 묶어 볼 수 있어야 한다."""
        path = tmp_path / "calls.jsonl"
        logger = CallLogger(path)
        try:
            for label in ("response", "emotion", "memory"):
                logger.log_call(_record(label), turn_id="turn-1")
            logger.log_call(_record(), turn_id="turn-2")
            logger.flush()
        finally:
            logger.shutdown()

        entries = _read_lines(path)
        assert sum(1 for e in entries if e["turn_id"] == "turn-1") == 3

    def test_creates_parent_directory(self, tmp_path: Path):
        path = tmp_path / "깊은" / "경로" / "calls.jsonl"
        logger = CallLogger(path)
        try:
            logger.log_call(_record())
            logger.flush()
        finally:
            logger.shutdown()

        assert path.exists()


class TestDisabled:
    def test_writes_nothing_when_disabled(self, tmp_path: Path):
        path = tmp_path / "calls.jsonl"
        logger = CallLogger(path, enabled=False)

        logger.log_call(_record())
        logger.shutdown()

        assert not path.exists()

    def test_disabled_shutdown_is_safe(self, tmp_path: Path):
        CallLogger(tmp_path / "x.jsonl", enabled=False).shutdown()


class TestRotation:
    def test_rotates_when_size_exceeded(self, tmp_path: Path):
        """운영 로그가 무한히 커지면 디스크를 채운다."""
        path = tmp_path / "calls.jsonl"
        logger = CallLogger(path, max_bytes=500, backup_count=2)
        try:
            for _ in range(50):
                logger.log_call(_record(), model="아주긴모델이름" * 5)
            logger.flush()
        finally:
            logger.shutdown()

        rotated = list(tmp_path.glob("calls.jsonl*"))
        assert len(rotated) > 1, "회전 파일이 생성되어야 한다"

    def test_backup_count_is_bounded(self, tmp_path: Path):
        path = tmp_path / "calls.jsonl"
        logger = CallLogger(path, max_bytes=300, backup_count=2)
        try:
            for _ in range(200):
                logger.log_call(_record(), model="x" * 50)
            logger.flush()
        finally:
            logger.shutdown()

        # 본 파일 + 백업 2개 = 최대 3개
        assert len(list(tmp_path.glob("calls.jsonl*"))) <= 3


class TestResilience:
    def test_logging_error_does_not_raise(self, tmp_path: Path, monkeypatch):
        """로깅이 깨져도 대화는 계속되어야 한다."""
        logger = CallLogger(tmp_path / "calls.jsonl")
        try:
            monkeypatch.setattr(
                logger._logger,
                "info",
                lambda *a, **kw: (_ for _ in ()).throw(OSError("디스크 오류")),
            )
            logger.log_call(_record())  # 예외가 밖으로 나오면 안 된다
        finally:
            logger.shutdown()

    def test_shutdown_is_idempotent(self, tmp_path: Path):
        logger = CallLogger(tmp_path / "calls.jsonl")
        logger.shutdown()
        logger.shutdown()


class TestFromConfig:
    """`config.yaml` 연동 — 설정만 있고 코드가 무시하면 의미가 없다."""

    def test_defaults_when_section_missing(self):
        from src.call_log import from_config

        logger = from_config({})
        try:
            assert logger.enabled is True, "설정을 잊어도 운영 로그는 남아야 한다"
            assert logger.capture_payload is True
        finally:
            logger.shutdown()

    def test_reads_settings(self, tmp_path: Path):
        from src.call_log import from_config

        logger = from_config(
            {
                "call_log": {
                    "path": str(tmp_path / "custom.jsonl"),
                    "capture_payload": False,
                    "max_field_chars": 100,
                }
            }
        )
        try:
            assert logger.path == tmp_path / "custom.jsonl"
            assert logger.capture_payload is False
            assert logger.max_field_chars == 100
        finally:
            logger.shutdown()

    def test_can_be_disabled(self, tmp_path: Path):
        from src.call_log import from_config

        logger = from_config({"call_log": {"enabled": False, "path": str(tmp_path / "x.jsonl")}})
        logger.log_call(_record())
        logger.shutdown()

        assert not (tmp_path / "x.jsonl").exists()

    def test_none_config(self):
        from src.call_log import from_config

        logger = from_config(None)
        try:
            assert logger.enabled is True
        finally:
            logger.shutdown()


class TestTruncation:
    def test_long_field_is_truncated_visibly(self, tmp_path: Path):
        """잘렸다는 사실을 감추면 로그를 신뢰할 수 없다."""
        path = tmp_path / "calls.jsonl"
        logger = CallLogger(path, max_field_chars=50)
        try:
            record = CallRecord(
                label="response",
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
                duration_ms=1.0,
                messages=[{"role": "system", "content": "가" * 500}],
                response="나" * 500,
            )
            logger.log_call(record)
            logger.flush()
        finally:
            logger.shutdown()

        entry = _read_lines(path)[0]
        assert "잘림" in entry["messages"][0]["content"]
        assert "잘림" in entry["response"]

    def test_payload_omitted_when_capture_disabled(self, tmp_path: Path):
        path = tmp_path / "calls.jsonl"
        logger = CallLogger(path, capture_payload=False)
        try:
            logger.log_call(
                CallRecord("response", 1, 1, 2, 1.0, messages=[{"role": "user", "content": "비밀"}])
            )
            logger.flush()
        finally:
            logger.shutdown()

        entry = _read_lines(path)[0]
        assert "messages" not in entry
        assert entry["total_tokens"] == 2
