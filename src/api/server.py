"""Character OS — FastAPI REST + WebSocket 서버.

엔드포인트:
    POST  /chat                대화 (동기, 전체 응답)
    WS    /ws/chat             대화 (스트리밍, 토큰 단위)
    GET   /health              헬스체크
    GET   /emotion             감정 상태 조회
    GET   /memory/stats        기억 통계
    POST  /character/reset     캐릭터 초기화 (기억/감정)
    GET   /persona             페르소나 조회
    PUT   /persona             페르소나 수정
    GET   /knowledge           지식 파일 목록
    GET   /knowledge/{name}    지식 파일 내용
    PUT   /knowledge/{name}    지식 파일 수정
    GET   /docs                Swagger UI
    GET   /*                   프론트엔드 (정적 파일)

실행:
    uv run uvicorn src.api.server:app --reload
"""

from __future__ import annotations

import argparse
import asyncio
import threading
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from queue import Empty, Queue

import yaml
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.character_os import CharacterOS

# ---------------------------------------------------------------------------
# CharacterWorker — 캐릭터별 전용 스레드 + 큐
# ---------------------------------------------------------------------------


class CharacterWorker:
    """CharacterOS를 전용 스레드에서 순차 처리하는 워커.

    대화 요청은 큐에 들어가고, 전용 스레드가 하나씩 처리한다.
    상태 읽기(emotion, memory 등)는 직접 읽어도 안전하다
    (파이썬 GIL + 워커가 읽기 사이에 yield하지 않음).
    """

    def __init__(self, cos: CharacterOS):
        self.cos = cos
        self._queue: Queue = Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        """전용 스레드: 큐에서 작업을 꺼내 순차 실행."""
        while True:
            try:
                item = self._queue.get(timeout=1.0)
            except Empty:
                continue

            if item is None:  # shutdown signal
                break

            loop, future, fn = item
            try:
                result = fn()
                loop.call_soon_threadsafe(future.set_result, result)
            except Exception as e:
                loop.call_soon_threadsafe(future.set_exception, e)

    def submit(self, fn: Callable) -> tuple[asyncio.AbstractEventLoop, asyncio.Future]:
        """함수를 큐에 제출하고 (loop, future)를 반환한다."""
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._queue.put((loop, future, fn))
        return loop, future

    async def run(self, fn: Callable) -> any:
        """함수를 큐에 제출하고 결과를 기다린다 (async)."""
        _, future = self.submit(fn)
        return await future

    def shutdown(self) -> None:
        """워커 종료."""
        self._queue.put(None)
        self._thread.join(timeout=5)


# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "character_dir": "characters/hong-gil-dong",
    "memory_db_path": "memory/memories.db",
    "emotion_save_path": "memory/emotions.json",
    "history_save_path": "memory/history.json",
    "model_type": "api",
    "local_model": "mlx-community/Qwen3.5-4B-MLX-4bit",
    "adapter_path": None,
}


def _load_config(path: str) -> dict:
    config_file = Path(path)
    if config_file.exists():
        with open(config_file, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


# ---------------------------------------------------------------------------
# CharacterWorker 싱글톤 (lifespan에서 초기화)
# ---------------------------------------------------------------------------

_worker: CharacterWorker | None = None
_config: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작 시 CharacterWorker 초기화, 종료 시 정리."""
    global _worker, _config
    _config = dict(DEFAULT_CONFIG)
    _config.update(_load_config("config.yaml"))
    cos = CharacterOS(
        character_dir=_config["character_dir"],
        memory_db_path=_config["memory_db_path"],
        emotion_save_path=_config["emotion_save_path"],
        history_save_path=_config["history_save_path"],
        model_type=_config["model_type"],
        local_model=_config["local_model"],
        adapter_path=_config["adapter_path"],
        debug=True,
        trace=True,
    )
    _worker = CharacterWorker(cos)
    yield
    _worker.shutdown()
    _worker = None


def _get_cos() -> CharacterOS:
    if _worker is None:
        raise RuntimeError("CharacterOS not initialized")
    return _worker.cos


async def _run_in_worker(fn: Callable) -> any:
    """CharacterWorker에서 함수를 실행하고 결과를 반환한다."""
    if _worker is None:
        raise RuntimeError("CharacterOS not initialized")
    return await _worker.run(fn)


# ---------------------------------------------------------------------------
# FastAPI 앱
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Character OS API",
    description="캐릭터 대화 REST + WebSocket API",
    version="0.2.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# 요청/응답 모델
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str | None
    emotion: dict | None = None


class ResetRequest(BaseModel):
    memory: bool = True
    emotion: bool = True
    history: bool = False


class PersonaUpdate(BaseModel):
    name: str | None = None
    identity: str | None = None
    age: int | str | None = None
    gender: str | None = None
    occupation: str | None = None
    personality: dict | list[str] | None = None
    speaking_style: dict | str | None = None
    values: list[str] | None = None
    backstory: str | None = None
    likes: list[str] | None = None
    dislikes: list[str] | None = None
    fears: list[str] | None = None
    goals: list[str] | None = None
    behavior: dict | None = None
    emotion_triggers: list[dict] | None = None
    relationships: list[dict] | None = None
    inner_world: dict | None = None
    examples: list[dict] | None = None


class KnowledgeUpdate(BaseModel):
    content: str


# ---------------------------------------------------------------------------
# API 엔드포인트 (SPA 라우트보다앞에 정의)
# ---------------------------------------------------------------------------


@app.get("/api/health")
def health():
    """헬스체크."""
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """대화 — 워커 스레드에서 순차 처리."""
    cos = _get_cos()

    def _do_chat():
        response = cos.chat(req.message)
        emotion = cos.emotion.get_state()
        return ChatResponse(response=response, emotion=emotion)

    return await _run_in_worker(_do_chat)


@app.get("/api/emotion")
def get_emotion():
    """현재 감정 상태 조회."""
    cos = _get_cos()
    return cos.emotion.get_state()


@app.get("/api/memory/stats")
def memory_stats():
    """기억 통계."""
    cos = _get_cos()
    return {"count": cos.memory.snapshot_count()}


@app.get("/api/memory")
def list_memory():
    """저장된 기억 목록."""
    cos = _get_cos()
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


@app.get("/api/history")
def get_history():
    """대화 기록 조회."""
    cos = _get_cos()
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


@app.get("/api/debug")
def get_debug_logs():
    """디버그 로그 조회."""
    cos = _get_cos()
    return {"logs": cos._debug_logs, "enabled": cos._debug}


@app.post("/api/debug/toggle")
def toggle_debug():
    """디버그 모드 토글."""
    cos = _get_cos()
    cos._debug = not cos._debug
    return {"enabled": cos._debug}


@app.post("/api/debug/clear")
def clear_debug_logs():
    """디버그 로그 삭제."""
    cos = _get_cos()
    cos._debug_logs.clear()
    return {"status": "ok"}


@app.get("/api/trace/last")
def get_last_trace():
    """마지막 파이프라인 트레이스 조회."""
    cos = _get_cos()
    if cos._last_trace is None:
        return {"trace": None}
    return {"trace": cos._last_trace.to_dict()}


# ---------------------------------------------------------------------------
# 다중 캐릭터 API
# ---------------------------------------------------------------------------


@app.get("/api/characters")
def list_characters():
    """사용 가능한 캐릭터 목록을 반환한다."""
    characters_dir = Path("characters")
    if not characters_dir.exists():
        return {"characters": [], "active": None}

    result = []
    for d in sorted(characters_dir.iterdir()):
        persona_file = d / "persona.yaml"
        if d.is_dir() and persona_file.exists():
            try:
                import yaml as _yaml

                data = _yaml.safe_load(persona_file.read_text(encoding="utf-8")) or {}
                result.append(
                    {
                        "id": d.name,
                        "name": data.get("name", d.name),
                        "identity": data.get("identity", ""),
                    }
                )
            except Exception:
                result.append({"id": d.name, "name": d.name, "identity": ""})

    cos = _get_cos()
    active_id = Path(cos._character_dir).name if cos._character_dir else None
    return {"characters": result, "active": active_id}


class SwitchCharacterRequest(BaseModel):
    character_id: str


@app.post("/api/character/switch")
async def switch_character(req: SwitchCharacterRequest):
    """캐릭터를 전환한다 — CharacterOS를 재생성한다."""
    global _worker

    character_dir = Path("characters") / req.character_id
    if not character_dir.exists():
        raise HTTPException(
            status_code=404, detail=f"캐릭터를 찾을 수 없습니다: {req.character_id}"
        )
    if not (character_dir / "persona.yaml").exists():
        raise HTTPException(status_code=400, detail=f"persona.yaml이 없습니다: {req.character_id}")

    new_cos = CharacterOS(
        character_dir=str(character_dir),
        memory_db_path=_config.get("memory_db_path", "memory/memories.db"),
        emotion_save_path=_config.get("emotion_save_path", "memory/emotions.json"),
        history_save_path=_config.get("history_save_path", "memory/history.json"),
        model_type=_config.get("model_type", "api"),
        local_model=_config.get("local_model", "mlx-community/Qwen3.5-4B-MLX-4bit"),
        adapter_path=_config.get("adapter_path"),
        debug=True,
        trace=True,
    )

    old_worker = _worker
    _worker = CharacterWorker(new_cos)
    if old_worker:
        old_worker.shutdown()

    return {"status": "ok", "character": req.character_id}


class CreateCharacterRequest(BaseModel):
    name: str
    identity: str = ""


@app.post("/api/characters")
def create_character(req: CreateCharacterRequest):
    """새 캐릭터를 생성한다."""
    import re

    import yaml as _yaml

    # 디렉토리 이름: 이름을 kebab-case로 변환
    char_id = re.sub(r"[^a-zA-Z0-9가-힣]", "-", req.name).strip("-").lower()
    if not char_id:
        char_id = f"character-{int(time.time())}"

    char_dir = Path("characters") / char_id
    if char_dir.exists():
        raise HTTPException(status_code=409, detail=f"이미 존재하는 캐릭터입니다: {char_id}")

    # 디렉토리 생성
    char_dir.mkdir(parents=True)
    (char_dir / "examples").mkdir()
    (char_dir / "knowledge").mkdir()

    # persona.yaml 템플릿 생성
    persona_data = {
        "name": req.name,
        "identity": req.identity or f"{req.name}의 정체성",
        "age": "",
        "gender": "",
        "occupation": "",
        "personality": {
            "traits": ["친절한"],
            "big5": {
                "openness": 0.5,
                "conscientiousness": 0.5,
                "extraversion": 0.5,
                "agreeableness": 0.5,
                "neuroticism": 0.5,
            },
        },
        "speaking_style": {
            "summary": "정중한 말투",
            "tone": "차분한",
            "vocabulary": "일상어",
            "sentence_pattern": "보통 문장",
            "fillers": [],
            "emojis": "적게 사용",
            "endings": ["~입니다", "~합니다"],
        },
        "values": [],
        "backstory": "",
        "likes": [],
        "dislikes": [],
        "fears": [],
        "goals": [],
        "behavior": {
            "situations": [],
            "topics": [],
            "rules": [],
        },
        "emotion_triggers": [],
        "relationships": [],
        "inner_world": {
            "current_thought": "",
            "hidden_feelings": "",
            "wants_to_say": "",
        },
        "examples": [
            {"user": "안녕!", "character": "안녕하세요!", "scenario": "인사"},
        ],
    }

    (char_dir / "persona.yaml").write_text(
        _yaml.dump(persona_data, allow_unicode=True, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )

    return {"status": "ok", "character": char_id}


@app.delete("/api/characters/{character_id}")
def delete_character(character_id: str):
    """캐릭터를 삭제한다. 활성 캐릭터는 삭제할 수 없다."""
    import shutil

    cos = _get_cos()
    active_id = Path(cos._character_dir).name if cos._character_dir else None

    if character_id == active_id:
        raise HTTPException(
            status_code=400,
            detail="활성 캐릭터는 삭제할 수 없습니다. 먼저 다른 캐릭터로 전환하세요.",
        )

    char_dir = Path("characters") / character_id
    if not char_dir.exists():
        raise HTTPException(status_code=404, detail=f"캐릭터를 찾을 수 없습니다: {character_id}")

    shutil.rmtree(char_dir)
    return {"status": "ok", "deleted": character_id}


@app.get("/api/logs")
def get_logs(level: str = "all", limit: int = 200):
    """상세 로그 조회. level: all, error, info."""
    cos = _get_cos()
    logs = cos._debug_logs
    if level == "error":
        logs = [
            line for line in logs if "실패" in line or "오류" in line or "error" in line.lower()
        ]
    return {"logs": logs[-limit:], "total": len(cos._debug_logs)}


@app.get("/api/performance")
def get_performance():
    """성능 메트릭 조회 (트레이스 + 카운터)."""
    cos = _get_cos()
    trace_data = cos._last_trace.to_dict() if cos._last_trace else None
    return {
        "trace": trace_data,
        "emotion_state": cos.emotion.get_state(),
        "memory_count": cos.memory.snapshot_count(),
        "history_count": cos.history.count(),
    }


@app.post("/api/character/reset")
async def character_reset(req: ResetRequest):
    """캐릭터 초기화 — 워커에서 순차 처리."""
    cos = _get_cos()

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

    return await _run_in_worker(_do_reset)


@app.get("/api/persona")
def get_persona():
    """페르소나 조회 (파일에서다시로드)."""
    cos = _get_cos()
    cos.persona.load()  # 파일에서다시로드
    return cos.persona._data


@app.put("/api/persona")
def update_persona(req: PersonaUpdate):
    """페르소나 수정 — YAML 파일을 업데이트하고 다시 로드한다."""
    cos = _get_cos()

    # 현재 데이터와 병합
    data = dict(cos.persona._data)
    updates = req.model_dump(exclude_none=True)
    data.update(updates)

    # YAML 파일에 저장
    persona_path = cos.persona._path
    persona_path.parent.mkdir(parents=True, exist_ok=True)
    with open(persona_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False)

    # 다시 로드
    cos.persona.load()

    return {"status": "ok", "persona": cos.persona._data}


@app.get("/api/knowledge")
def list_knowledge():
    """지식 파일 목록."""
    cos = _get_cos()
    knowledge_dir = cos.knowledge._dir

    entries = []
    if not knowledge_dir.exists():
        return {"entries": entries}

    for f in sorted(knowledge_dir.iterdir()):
        if not f.is_file():
            continue
        ext = f.suffix.lower()
        if ext not in cos.knowledge.SUPPORTED_EXTENSIONS:
            continue
        content = f.read_text(encoding="utf-8")
        entries.append(
            {
                "name": f.name,
                "size": len(content),
                "preview": content[:100],
            }
        )

    return {"entries": entries}


@app.get("/api/knowledge/relationships")
def get_relationships():
    """관계 그래프 조회."""
    cos = _get_cos()
    cos.knowledge.load_all()
    return {"relationships": cos.knowledge.get_relationships()}


@app.get("/api/knowledge/relationships/{character}")
def get_relationships_for(character: str):
    """특정 캐릭터의 관계 조회."""
    cos = _get_cos()
    cos.knowledge.load_all()
    return {"relationships": cos.knowledge.get_relationships_for(character)}


@app.get("/api/knowledge/timeline")
def get_timeline():
    """타임라인 조회."""
    cos = _get_cos()
    cos.knowledge.load_all()
    return {"events": cos.knowledge.get_timeline()}


@app.get("/api/knowledge/locations")
def get_locations():
    """장소 목록 조회."""
    cos = _get_cos()
    cos.knowledge.load_all()
    return {"locations": cos.knowledge.get_locations()}


@app.get("/api/knowledge/{name}")
def get_knowledge(name: str):
    """지식 파일 내용 조회 (원본 텍스트)."""
    cos = _get_cos()
    knowledge_dir = cos.knowledge._dir

    # 확자자 포함/미포함 모두 시도
    for candidate in [
        name,
        f"{name}.yaml",
        f"{name}.yml",
        f"{name}.json",
        f"{name}.md",
        f"{name}.txt",
    ]:
        file_path = knowledge_dir / candidate
        if file_path.exists():
            return {"content": file_path.read_text(encoding="utf-8")}

    raise HTTPException(status_code=404, detail=f"지식 파일을 찾을 수 없습니다: {name}")


@app.put("/api/knowledge/{name}")
def update_knowledge(name: str, req: KnowledgeUpdate):
    """지식 파일 수정."""
    cos = _get_cos()

    knowledge_dir = cos.knowledge._dir
    file_path = knowledge_dir / name

    # 파일 쓰기
    knowledge_dir.mkdir(parents=True, exist_ok=True)
    file_path.write_text(req.content, encoding="utf-8")

    # 다시 로드
    cos.knowledge.load_all()

    return {"status": "ok", "name": name, "size": len(req.content)}


# ---------------------------------------------------------------------------
# Few-shot 예시 API
# ---------------------------------------------------------------------------


@app.get("/api/fewshot")
def list_fewshot():
    """Few-shot 예시 태그 목록."""
    cos = _get_cos()
    tags = cos.fewshot.get_all_tags()
    groups = []
    for group in cos.fewshot.get_all_groups():
        groups.append(
            {
                "tag": group.tag,
                "count": len(group.examples),
                "examples": [
                    {"user": e.user, "character": e.character, "emotion_state": e.emotion_state}
                    for e in group.examples
                ],
            }
        )
    return {"tags": tags, "groups": groups}


@app.get("/api/fewshot/search")
def search_fewshot(q: str = ""):
    """Few-shot 예시 검색."""
    cos = _get_cos()
    if not q:
        return {"results": []}
    results = cos.fewshot.search(q, cos.emotion.get_state())
    return {
        "results": [
            {"user": e.user, "character": e.character, "emotion_state": e.emotion_state}
            for e in results
        ]
    }


@app.websocket("/api/ws/chat")
async def ws_chat(ws: WebSocket):
    """스트리밍 대화 — 워커 스레드에서 처리, 토큰 단위 전송.

    프로토콜:
        클라이언트 → 서버: 텍스트 프레임 (메시지)
        서버 → 클라이언트: 텍스트 프레임 (토큰들) + "[DONE]"
    """
    await ws.accept()
    cos = _get_cos()

    try:
        while True:
            user_input = await ws.receive_text()

            # 워커 스레드에서 스트리밍 실행, 토큰을 큐로 전달
            token_queue: Queue = Queue()

            # 기본 인자로 루프 변수를 명시적으로 바인딩한다 (늦은 바인딩 방지)
            def _stream(user_input=user_input, token_queue=token_queue):
                try:
                    for token in cos.chat_stream(user_input):
                        token_queue.put(token)
                except Exception as e:
                    token_queue.put(e)
                finally:
                    token_queue.put(None)  # sentinel

            # 워커에 제출 (순차 보장, fire-and-forget)
            _worker.submit(_stream)

            # 토큰을 비동기적으로 읽어서 전송
            while True:
                try:
                    item = await asyncio.get_event_loop().run_in_executor(
                        None, lambda q=token_queue: q.get(timeout=30)
                    )
                except Empty:
                    break

                if item is None:  # 스트림 종료
                    break
                if isinstance(item, Exception):
                    raise item
                await ws.send_text(item)

            await ws.send_text("[DONE]")
    except WebSocketDisconnect:
        pass


# ---------------------------------------------------------------------------
# 프론트엔드 정적 파일 서빙 (SPA)
# ---------------------------------------------------------------------------

FRONTEND_DIR = Path(__file__).parent.parent.parent / "frontend" / "dist"


# 정적 파일 마운트 (CSS, JS, 이미지 등)
if FRONTEND_DIR.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR / "assets"), name="assets")


@app.get("/{full_path:path}")
async def serve_frontend(full_path: str):
    """프론트엔드 정적 파일 서빙 (SPA fallback)."""
    if FRONTEND_DIR.exists():
        file_path = FRONTEND_DIR / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(FRONTEND_DIR / "index.html")
    return {"error": "Frontend not built. Run: cd frontend && npm run build"}


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
