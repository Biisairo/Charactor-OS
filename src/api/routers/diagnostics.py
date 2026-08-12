"""진단 — 헬스체크·디버그 로그·트레이스·성능.

대화 동작에 관여하지 않고 상태를 들여다보기만 한다.
"""

from __future__ import annotations

from fastapi import APIRouter

from src.api import deps

router = APIRouter(prefix="/api", tags=["diagnostics"])


@router.get("/health")
def health():
    """헬스체크."""
    return {"status": "ok"}


@router.get("/debug")
def get_debug_logs():
    """디버그 로그 조회."""
    cos = deps.get_cos()
    return {"logs": cos._debug_logs, "enabled": cos._debug}


@router.post("/debug/toggle")
def toggle_debug():
    """디버그 모드 토글."""
    cos = deps.get_cos()
    cos._debug = not cos._debug
    return {"enabled": cos._debug}


@router.post("/debug/clear")
def clear_debug_logs():
    """디버그 로그 삭제."""
    cos = deps.get_cos()
    cos._debug_logs.clear()
    return {"status": "ok"}


@router.get("/trace/last")
def get_last_trace():
    """마지막 파이프라인 트레이스 조회."""
    cos = deps.get_cos()
    if cos._last_trace is None:
        return {"trace": None}
    return {"trace": cos._last_trace.to_dict()}


@router.get("/logs")
def get_logs(level: str = "all", limit: int = 200):
    """상세 로그 조회. level: all, error, info."""
    cos = deps.get_cos()
    logs = cos._debug_logs
    if level == "error":
        logs = [
            line for line in logs if "실패" in line or "오류" in line or "error" in line.lower()
        ]
    return {"logs": logs[-limit:], "total": len(cos._debug_logs)}


@router.get("/performance")
def get_performance():
    """성능 메트릭 조회 (트레이스 + 카운터)."""
    cos = deps.get_cos()
    trace_data = cos._last_trace.to_dict() if cos._last_trace else None
    return {
        "trace": trace_data,
        "emotion_state": cos.emotion.get_state(),
        "memory_count": cos.memory.snapshot_count(),
        "history_count": cos.history.count(),
    }
