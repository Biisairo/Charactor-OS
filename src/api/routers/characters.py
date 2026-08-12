"""캐릭터 — 목록·생성·삭제·전환과 페르소나 편집.

캐릭터를 파일시스템 디렉토리로 다루므로, 이름이 경로가 되기 전에
`paths` 모듈의 검증을 반드시 통과시킨다.
"""

from __future__ import annotations

import re
import shutil
import time
from pathlib import Path

import yaml
from fastapi import APIRouter, HTTPException

from src.api import deps
from src.api.paths import CHARACTERS_DIR, SAFE_SEGMENT, safe_child
from src.api.schemas import CreateCharacterRequest, PersonaUpdate, SwitchCharacterRequest
from src.api.worker import CharacterWorker
from src.character_os import CharacterOS

router = APIRouter(prefix="/api", tags=["characters"])


@router.get("/characters")
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

    cos = deps.get_cos()
    active_id = Path(cos._character_dir).name if cos._character_dir else None
    return {"characters": result, "active": active_id}


@router.post("/character/switch")
async def switch_character(req: SwitchCharacterRequest):
    """캐릭터를 전환한다 — CharacterOS를 재생성한다."""
    character_dir = safe_child(CHARACTERS_DIR, req.character_id, SAFE_SEGMENT)
    if not character_dir.exists():
        raise HTTPException(
            status_code=404, detail=f"캐릭터를 찾을 수 없습니다: {req.character_id}"
        )
    if not (character_dir / "persona.yaml").exists():
        raise HTTPException(status_code=400, detail=f"persona.yaml이 없습니다: {req.character_id}")

    new_cos = CharacterOS(
        character_dir=str(character_dir),
        memory_db_path=deps.get_config().get("memory_db_path", "memory/memories.db"),
        emotion_save_path=deps.get_config().get("emotion_save_path", "memory/emotions.json"),
        history_save_path=deps.get_config().get("history_save_path", "memory/history.json"),
        model_type=deps.get_config().get("model_type", "api"),
        local_model=deps.get_config().get("local_model", "mlx-community/Qwen3.5-4B-MLX-4bit"),
        adapter_path=deps.get_config().get("adapter_path"),
        debug=True,
        trace=True,
    )

    old_worker = deps.get_worker()
    deps.set_worker(CharacterWorker(new_cos))
    if old_worker:
        old_worker.shutdown()
        old_worker.cos._call_logger.shutdown()

    return {"status": "ok", "character": req.character_id}


@router.post("/characters")
def create_character(req: CreateCharacterRequest):
    """새 캐릭터를 생성한다."""
    import yaml as _yaml

    # 디렉토리 이름: 이름을 kebab-case로 변환
    char_id = re.sub(r"[^a-zA-Z0-9가-힣]", "-", req.name).strip("-").lower()
    if not char_id:
        char_id = f"character-{int(time.time())}"

    char_dir = safe_child(CHARACTERS_DIR, char_id, SAFE_SEGMENT)
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


@router.delete("/characters/{character_id}")
def delete_character(character_id: str):
    """캐릭터를 삭제한다. 활성 캐릭터는 삭제할 수 없다."""
    cos = deps.get_cos()
    char_dir = safe_child(CHARACTERS_DIR, character_id, SAFE_SEGMENT)

    # 활성 캐릭터 판정은 resolve된 경로로 비교한다 (이름 문자열 비교는 우회 가능)
    active_dir = Path(cos._character_dir).resolve() if cos._character_dir else None
    if active_dir is not None and char_dir == active_dir:
        raise HTTPException(
            status_code=400,
            detail="활성 캐릭터는 삭제할 수 없습니다. 먼저 다른 캐릭터로 전환하세요.",
        )

    if not char_dir.exists():
        raise HTTPException(status_code=404, detail=f"캐릭터를 찾을 수 없습니다: {character_id}")

    shutil.rmtree(char_dir)
    return {"status": "ok", "deleted": character_id}


@router.get("/persona")
def get_persona():
    """페르소나 조회 (파일에서다시로드)."""
    cos = deps.get_cos()
    cos.persona.load()  # 파일에서다시로드
    return cos.persona._data


@router.put("/persona")
def update_persona(req: PersonaUpdate):
    """페르소나 수정 — YAML 파일을 업데이트하고 다시 로드한다."""
    cos = deps.get_cos()

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
