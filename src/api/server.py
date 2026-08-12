"""Character OS — FastAPI REST 서버.

이 모듈은 **앱을 조립하기만 한다.** 엔드포인트는 도메인별 라우터에 있다.

    routers/diagnostics.py   헬스체크 · 디버그 로그 · 트레이스 · 성능
    routers/chat.py          대화
    routers/state.py         감정 · 기억 · 히스토리 · 초기화
    routers/characters.py    캐릭터 목록/생성/삭제/전환 · 페르소나
    routers/knowledge.py     세계관 · 관계 · 연표 · 장소 · few-shot
    routers/frontend.py      SPA 정적 파일 (반드시 마지막)

대화 경로는 `POST /api/chat` 하나다. 스트리밍(WebSocket) 경로는 제거했다 —
검토를 거치지 않은 초안이 사용자에게 도달하는 통로였다 (TASK-11).

**등록 순서가 동작을 바꾼다.** FastAPI는 먼저 등록된 라우트를 먼저 매칭하므로
`frontend`의 catch-all이 앞서면 모든 API가 가려진다.
`tests/integration/test_api_surface.py`가 경로 목록과 순서 제약을 지킨다.

실행:
    uv run uvicorn src.api.server:app --reload
"""

from __future__ import annotations

import argparse
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api import deps
from src.api.deps import DEFAULT_CONFIG, load_config
from src.api.paths import SAFE_FILENAME, SAFE_SEGMENT, safe_child
from src.api.routers import characters, chat, diagnostics, frontend, knowledge, state
from src.api.worker import CharacterWorker
from src.call_log import from_config as call_logger_from_config
from src.character_os import CharacterOS

__all__ = ["app", "lifespan", "CharacterWorker", "SAFE_FILENAME", "SAFE_SEGMENT", "safe_child"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작 시 CharacterWorker 초기화, 종료 시 정리."""
    config = dict(DEFAULT_CONFIG)
    config.update(load_config("config.yaml"))
    deps.set_config(config)

    cos = CharacterOS(
        character_dir=config["character_dir"],
        memory_db_path=config["memory_db_path"],
        emotion_save_path=config["emotion_save_path"],
        history_save_path=config["history_save_path"],
        model_type=config["model_type"],
        local_model=config["local_model"],
        adapter_path=config["adapter_path"],
        debug=True,
        trace=True,
        call_logger=call_logger_from_config(config),
    )
    deps.set_worker(CharacterWorker(cos))
    yield
    worker = deps.get_worker()
    if worker is not None:
        worker.shutdown()
    # 종료 시 큐에 남은 호출 로그를 디스크까지 밀어낸다.
    # atexit에도 걸려 있지만, 서버 수명주기에서 명시적으로 비우는 편이 확실하다.
    cos._call_logger.shutdown()
    deps.set_worker(None)


app = FastAPI(
    title="Character OS API",
    description="캐릭터 대화 REST API",
    version="0.2.0",
    lifespan=lifespan,
)

# 순서 주의 — frontend의 catch-all이 마지막이어야 한다.
app.include_router(diagnostics.router)
app.include_router(chat.router)
app.include_router(state.router)
app.include_router(characters.router)
app.include_router(knowledge.router)
app.include_router(frontend.router)

# 정적 파일 마운트 (CSS, JS, 이미지 등)
if frontend.FRONTEND_DIR.exists():
    from fastapi.staticfiles import StaticFiles

    app.mount("/assets", StaticFiles(directory=frontend.FRONTEND_DIR / "assets"), name="assets")


# ---------------------------------------------------------------------------
# CLI 실행: uv run python -m src.api.server [--port 8000]
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Character OS API Server")
    parser.add_argument("--port", type=int, default=8000, help="포트 번호")
    parser.add_argument("--host", default="0.0.0.0", help="호스트")
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)
