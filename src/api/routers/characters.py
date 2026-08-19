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
from src.api.schemas import (
    CharacterExamplesData,
    CharacterKnowledgeData,
    CharacterStaticData,
    CreateCharacterRequest,
    PersonaUpdate,
    SwitchCharacterRequest,
)
from src.api.worker import CharacterWorker
from src.character_layout import CharacterLayout
from src.character_os import CharacterOS
from src.modules.knowledge import BASE_DIRNAME, GENERAL_DIRNAME, KnowledgeModule

router = APIRouter(prefix="/api", tags=["characters"])


@router.get("/characters")
def list_characters():
    """사용 가능한 캐릭터 목록을 반환한다."""
    characters_dir = Path("characters")
    if not characters_dir.exists():
        return {"characters": [], "active": None}

    result = []
    for d in sorted(characters_dir.iterdir()):
        persona_file = CharacterLayout.of(d).persona_path
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
    if not CharacterLayout.of(character_dir).is_character():
        raise HTTPException(
            status_code=400, detail=f"static/persona.yaml이 없습니다: {req.character_id}"
        )

    new_cos = CharacterOS(
        character_dir=str(character_dir),
        # 기본값을 주지 않는다. 여기에 전역 경로를 넣으면 캐릭터를 전환해도
        # 이전 캐릭터의 기억·감정을 그대로 물려받는다 — TASK-17이 고친 결함이다.
        memory_db_path=deps.get_config().get("memory_db_path"),
        emotion_save_path=deps.get_config().get("emotion_save_path"),
        history_save_path=deps.get_config().get("history_save_path"),
        working_memory_path=deps.get_config().get("working_memory_path"),
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


def _dump_yaml(path: Path, data: dict) -> None:
    """사람이 읽고 편집할 수 있는 YAML로 쓴다 — 편집기 템플릿과 같은 포맷."""
    path.write_text(
        yaml.dump(data, allow_unicode=True, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )


def _write_persona_template(layout: CharacterLayout, name: str, identity: str) -> None:
    """생성 직후 대화 가능한 최소 페르소나를 쓴다."""
    persona_data = {
        "name": name,
        "identity": identity or f"{name}의 정체성",
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
    _dump_yaml(layout.persona_path, persona_data)


def _write_knowledge_assets(
    knowledge_dir: Path,
    knowledge: CharacterKnowledgeData | None,
    delete_empty: bool = False,
) -> None:
    """질문지 응답을 knowledge/ 아래 **마크다운**으로 쓴다 (SPEC-04 v3).

    세계관은 늘 참인 배경이므로 `base/`로, 인물·장소·연표는 찾아 쓰는 자료이므로
    `general/`로 간다. 항목 하나가 `##` 하나가 되어 그대로 검색 단위가 된다.

    비어 있는 섹션은 파일을 만들지 않는다. `world` 같은 dict는 truthy 검사로는
    빈 문자열만 채워진 껍데기도 지나가므로, 값이 하나라도 있는 경우만 쓴다.
    `delete_empty=True`(기존 캐릭터 수정)면 빈 섹션의 이전 파일을 지운다 —
    질문지에서 비운 내용이 파일에 남아 있으면 안 된다.
    """
    base_dir = knowledge_dir / BASE_DIRNAME
    general_dir = knowledge_dir / GENERAL_DIRNAME

    world_filled = (
        knowledge is not None
        and knowledge.world
        and any(len(v) > 0 if isinstance(v, list) else bool(v) for v in knowledge.world.values())
    )
    sections = [
        (general_dir / "people.md", _render_people(knowledge)),
        (general_dir / "places.md", _render_places(knowledge)),
        (general_dir / "history.md", _render_history(knowledge)),
        (
            general_dir / "notes.md",
            (knowledge.freeform or "").strip() if knowledge is not None else "",
        ),
    ]

    if world_filled:
        _write_text(base_dir / "01-world.md", _render_world(knowledge.world))
    elif delete_empty:
        _remove_if_exists(base_dir / "01-world.md")

    for path, body in sections:
        if body:
            _write_text(path, body)
        elif delete_empty:
            _remove_if_exists(path)


def _write_text(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body.rstrip() + "\n", encoding="utf-8")


def _render_world(world: dict) -> str:
    """세계관 → 배경 문서. `era`는 시스템이 읽으므로 앞머리 메타로 남긴다."""
    era = str(world.get("era") or "").strip()
    lines = []
    if era:
        lines += ["---", f"era: {era}", "---", ""]

    lines.append(f"# {world.get('name') or '세계관'}")

    if description := str(world.get("description") or "").strip():
        lines += ["", description]

    for label, key in [("규칙", "rules"), ("기술 수준", "technology_level"), ("사회 구조", "social_structure")]:
        value = world.get(key)
        if not value:
            continue
        lines += ["", f"## {label}", ""]
        if isinstance(value, list):
            lines += [f"- {item}" for item in value]
        else:
            lines.append(str(value))

    return "\n".join(lines)


def _render_people(knowledge: CharacterKnowledgeData | None) -> str:
    if knowledge is None or not knowledge.relationships:
        return ""
    lines = ["# 주변 사람들"]
    for rel in knowledge.relationships:
        target = rel.get("to") or rel.get("target") or "?"
        lines += ["", f"## {target}", ""]
        detail = [str(rel[key]) for key in ("type", "sentiment", "description") if rel.get(key)]
        lines.append(" — ".join(detail) if detail else "(내용 없음)")
    return "\n".join(lines)


def _render_places(knowledge: CharacterKnowledgeData | None) -> str:
    if knowledge is None or not knowledge.locations:
        return ""
    lines = ["# 오가는 곳"]
    for loc in knowledge.locations:
        lines += ["", f"## {loc.get('name') or '?'}", ""]
        detail = [str(loc[key]) for key in ("description", "significance") if loc.get(key)]
        lines.append(" ".join(detail) if detail else "(내용 없음)")
    return "\n".join(lines)


def _render_history(knowledge: CharacterKnowledgeData | None) -> str:
    if knowledge is None or not knowledge.timeline:
        return ""
    lines = ["# 지나온 일"]
    for entry in knowledge.timeline:
        lines += ["", f"## {entry.get('time') or '?'}", ""]
        detail = [str(entry[key]) for key in ("event", "impact") if entry.get(key)]
        lines.append(" ".join(detail) if detail else "(내용 없음)")
    return "\n".join(lines)


def _write_example_assets(
    examples_dir: Path,
    examples: CharacterExamplesData | None,
    delete_empty: bool = False,
) -> None:
    """질문지 응답으로 examples/ 시나리오 파일을 쓴다."""
    if examples is None:
        return
    for key in ("greeting", "comfort", "conflict", "humor", "daily"):
        content = getattr(examples, key)
        if content:
            _dump_yaml(examples_dir / f"{key}.yaml", content)
        elif delete_empty:
            _remove_if_exists(examples_dir / f"{key}.yaml")


def _remove_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


@router.post("/characters")
def create_character(req: CreateCharacterRequest):
    """새 캐릭터를 생성한다.

    `static_data`가 있으면 질문지 응답으로 static/ 전체를 채우고,
    없으면 대화 가능한 최소 템플릿을 만든다 (기존 동작).
    """
    # 디렉토리 이름: 이름을 kebab-case로 변환
    char_id = re.sub(r"[^a-zA-Z0-9가-힣]", "-", req.name).strip("-").lower()
    if not char_id:
        char_id = f"character-{int(time.time())}"

    char_dir = safe_child(CHARACTERS_DIR, char_id, SAFE_SEGMENT)
    if char_dir.exists():
        raise HTTPException(status_code=409, detail=f"이미 존재하는 캐릭터입니다: {char_id}")

    # 디렉토리 생성 — 정적 자산만 만든다. state/는 첫 대화에서 생긴다.
    layout = CharacterLayout.of(char_dir)
    layout.examples_dir.mkdir(parents=True)
    layout.knowledge_dir.mkdir(parents=True)

    if req.static_data is not None:
        if req.static_data.persona is not None:
            _dump_yaml(layout.persona_path, req.static_data.persona.model_dump(exclude_none=True))
        else:
            _write_persona_template(layout, req.name, req.identity)
        _write_knowledge_assets(layout.knowledge_dir, req.static_data.knowledge)
        _write_example_assets(layout.examples_dir, req.static_data.examples)
    else:
        _write_persona_template(layout, req.name, req.identity)

    return {"status": "ok", "character": char_id}


def _load_yaml(path: Path) -> dict | None:
    """YAML 파일이 있으면 dict로, 없으면 None. 손상된 파일은 None으로 간주한다."""
    if not path.exists():
        return None
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return None


def _read_sections(path: Path) -> list[tuple[str, str]]:
    """`## 제목` 단위로 (제목, 본문)을 뽑는다. 없으면 빈 목록."""
    if not path.exists():
        return []

    sections: list[tuple[str, str]] = []
    title = ""
    body: list[str] = []
    for line in path.read_text(encoding="utf-8").split("\n"):
        if line.startswith("## "):
            if title:
                sections.append((title, "\n".join(body).strip()))
            title = line[3:].strip()
            body = []
        elif title:
            body.append(line)
    if title:
        sections.append((title, "\n".join(body).strip()))
    return sections


def _read_knowledge_draft(knowledge_dir: Path) -> dict:
    """마크다운 자산을 질문지 응답 형태로 되읽는다 (SPEC-04 v3).

    저작 형식이 마크다운으로 통일되면서 왕복은 **손실적**이 됐다. 항목의 세부
    필드(관계의 type·sentiment 등)는 설명 한 덩어리로 합쳐진다. 질문지는 처음
    만들 때 쓰는 도구이고, 이후 다듬기는 마크다운을 직접 고치는 편이 자연스럽다.
    """
    knowledge: dict = {}
    base_dir = knowledge_dir / BASE_DIRNAME
    general_dir = knowledge_dir / GENERAL_DIRNAME

    world_path = base_dir / "01-world.md"
    if world_path.exists():
        module = KnowledgeModule(str(knowledge_dir))
        module.load_all()
        raw = world_path.read_text(encoding="utf-8")
        sections = dict(_read_sections(world_path))
        world: dict = {}
        if era := module.era():
            world["era"] = era
        for line in raw.split("\n"):
            if line.startswith("# "):
                world["name"] = line[2:].strip()
                break
        if rules := sections.get("규칙"):
            world["rules"] = [ln.lstrip("- ").strip() for ln in rules.split("\n") if ln.strip()]
        for label, key in [("기술 수준", "technology_level"), ("사회 구조", "social_structure")]:
            if value := sections.get(label):
                world[key] = value
        if world:
            knowledge["world"] = world

    if people := _read_sections(general_dir / "people.md"):
        knowledge["relationships"] = [{"to": t, "description": b} for t, b in people]
    if places := _read_sections(general_dir / "places.md"):
        knowledge["locations"] = [{"name": t, "description": b} for t, b in places]
    if history := _read_sections(general_dir / "history.md"):
        knowledge["timeline"] = [{"time": t, "event": b} for t, b in history]

    notes = general_dir / "notes.md"
    if notes.exists():
        # 저장할 때 끝 개행을 하나로 고르므로, 읽을 때도 걷어내 왕복을 안정시킨다.
        knowledge["freeform"] = notes.read_text(encoding="utf-8").rstrip("\n")

    return knowledge


@router.get("/characters/{character_id}/draft")
def get_character_draft(character_id: str):
    """기존 캐릭터의 static/ 을 질문지 응답 형태로 돌려준다.

    위저지를 "질문지로 다시 열기"로 재사용할 때 쓴다. 표준 5개 knowledge
    마크다운 자산을 되읽으므로 왕복은 손실적이다 — `_read_knowledge_draft` 참조.
    """
    char_dir = safe_child(CHARACTERS_DIR, character_id, SAFE_SEGMENT)
    if not CharacterLayout.of(char_dir).is_character():
        raise HTTPException(status_code=404, detail=f"캐릭터를 찾을 수 없습니다: {character_id}")

    layout = CharacterLayout.of(char_dir)
    persona = _load_yaml(layout.persona_path) or {}

    knowledge = _read_knowledge_draft(layout.knowledge_dir)

    examples: dict = {}
    for key in ("greeting", "comfort", "conflict", "humor", "daily"):
        content = _load_yaml(layout.examples_dir / f"{key}.yaml")
        if content:
            examples[key] = content

    return {"persona": persona, "knowledge": knowledge, "examples": examples}


@router.put("/characters/{character_id}/static")
def update_character_static(character_id: str, req: CharacterStaticData):
    """기존 캐릭터의 static/ 전체를 질문지 응답으로 덮어쓴다.

    `create_character`와 같은 작성 로직을 쓰되, 비운 섹션은 이전 파일을
    지운다 — 질문지가 "현재 상태"의 단일 진실이 되도록.
    활성 캐릭터면 메모리 모듈도 다시 로드해 화면에 바로 반영한다.
    """
    char_dir = safe_child(CHARACTERS_DIR, character_id, SAFE_SEGMENT)
    if not CharacterLayout.of(char_dir).is_character():
        raise HTTPException(status_code=404, detail=f"캐릭터를 찾을 수 없습니다: {character_id}")

    layout = CharacterLayout.of(char_dir)
    if req.persona is not None:
        _dump_yaml(layout.persona_path, req.persona.model_dump(exclude_none=True))
    _write_knowledge_assets(layout.knowledge_dir, req.knowledge, delete_empty=True)
    _write_example_assets(layout.examples_dir, req.examples, delete_empty=True)

    # 활성 캐릭터면 메모리 갱신 — 화면이 낡은 값을 보여주면 안 된다.
    # 서버가 아직 초기화되지 않은 상황(테스트 등)에서는 생략한다.
    worker = deps.get_worker()
    if worker is not None:
        active_dir = Path(worker.cos._character_dir).resolve()
        if char_dir.resolve() == active_dir:
            worker.cos.persona.load()
            worker.cos.knowledge.load_all()
            worker.cos.fewshot.load_all()

    return {"status": "ok", "character": character_id}


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

    # YAML 파일에 저장 — 키 순서를 유지한다. sort_keys 기본값은 파일 전체를
    # 알파벳순으로 뒤집어 편집 diff를 지저분하게 만든다 (다른 작성 경로와 동일).
    persona_path = cos.persona._path
    persona_path.parent.mkdir(parents=True, exist_ok=True)
    with open(persona_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    # 다시 로드
    cos.persona.load()

    return {"status": "ok", "persona": cos.persona._data}
