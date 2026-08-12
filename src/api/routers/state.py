"""상태 — 감정·기억·히스토리 조회와 초기화.

캐릭터가 축적한 동적 상태를 다룬다. 정적 자산(페르소나·지식)은
`characters` · `knowledge` 라우터가 맡는다.
"""

from __future__ import annotations

from fastapi import APIRouter

from src.api import deps
from src.api.schemas import ResetRequest

router = APIRouter(prefix="/api", tags=["state"])


@router.get("/emotion")
def get_emotion():
    """현재 감정 상태 조회."""
    cos = deps.get_cos()
    return cos.emotion.get_state()


@router.get("/memory/stats")
def memory_stats():
    """기억 통계."""
    cos = deps.get_cos()
    return {"count": cos.memory.snapshot_count()}


@router.get("/memory")
def list_memory():
    """저장된 기억 목록."""
    cos = deps.get_cos()
    memories = []
    for _key, entry in cos.memory._memories.items():
        memories.append(
            {
                "id": entry.id,
                "content": entry.content,
                "weight": round(entry.weight, 3),
                "emotion_tags": entry.emotion_tags,
                "access_count": entry.access_count,
                "created_at": entry.created_at,
            }
        )
    # 최근순 정렬
    memories.sort(key=lambda x: x["created_at"], reverse=True)
    return {"memories": memories}


@router.get("/history")
def get_history():
    """대화 기록 조회."""
    cos = deps.get_cos()
    turns = []
    for turn in cos.history._turns:
        turns.append(
            {
                "role": "user" if turn.role == "user" else "assistant",
                "content": turn.content,
                "timestamp": turn.timestamp,
            }
        )
    return {"turns": turns}


@router.post("/character/reset")
async def character_reset(req: ResetRequest):
    """캐릭터 초기화 — 워커에서 순차 처리."""
    cos = deps.get_cos()

    def _do_reset():
        import sqlite3 as _sqlite3

        results = {}

        if req.memory:
            cos.memory._memories.clear()
            conn = _sqlite3.connect(str(cos.memory._db_path))
            try:
                conn.execute("DELETE FROM memories")
                conn.commit()
            finally:
                conn.close()
            results["memory"] = "초기화 완료"

        if req.emotion:
            cos.emotion._emotions = {}
            cos.emotion.save()
            results["emotion"] = "초기화 완료"

        if req.history:
            cos.history._turns.clear()
            cos.history.save()
            results["history"] = "초기화 완료"

        return {"status": "ok", "results": results}

    return await deps.run_in_worker(_do_reset)
