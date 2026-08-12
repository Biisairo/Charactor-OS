"""지식 — 세계관·관계·연표·장소, 그리고 few-shot 예시.

경로 순서에 주의한다. `/knowledge/relationships` 같은 리터럴 경로는 반드시
`/knowledge/{name}` 앞에 등록되어야 한다. 뒤로 가면 `{name}`이 잡아먹는다.
`tests/integration/test_api_surface.py`가 이 제약을 지킨다.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.api import deps
from src.api.paths import SAFE_FILENAME, safe_child
from src.api.schemas import KnowledgeUpdate

router = APIRouter(prefix="/api", tags=["knowledge"])


@router.get("/knowledge")
def list_knowledge():
    """지식 파일 목록."""
    cos = deps.get_cos()
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


@router.get("/knowledge/relationships")
def get_relationships():
    """관계 그래프 조회."""
    cos = deps.get_cos()
    cos.knowledge.load_all()
    return {"relationships": cos.knowledge.get_relationships()}


@router.get("/knowledge/relationships/{character}")
def get_relationships_for(character: str):
    """특정 캐릭터의 관계 조회."""
    cos = deps.get_cos()
    cos.knowledge.load_all()
    return {"relationships": cos.knowledge.get_relationships_for(character)}


@router.get("/knowledge/timeline")
def get_timeline():
    """타임라인 조회."""
    cos = deps.get_cos()
    cos.knowledge.load_all()
    return {"events": cos.knowledge.get_timeline()}


@router.get("/knowledge/locations")
def get_locations():
    """장소 목록 조회."""
    cos = deps.get_cos()
    cos.knowledge.load_all()
    return {"locations": cos.knowledge.get_locations()}


@router.get("/knowledge/{name}")
def get_knowledge(name: str):
    """지식 파일 내용 조회 (원본 텍스트)."""
    cos = deps.get_cos()
    knowledge_dir = cos.knowledge._dir

    # 확장자 포함/미포함 모두 시도
    for candidate in [
        name,
        f"{name}.yaml",
        f"{name}.yml",
        f"{name}.json",
        f"{name}.md",
        f"{name}.txt",
    ]:
        if not SAFE_FILENAME.fullmatch(candidate):
            continue  # 확장자 없는 원본 name 등 — 다음 후보로
        file_path = safe_child(knowledge_dir, candidate, SAFE_FILENAME)
        if file_path.is_file():
            return {"content": file_path.read_text(encoding="utf-8")}

    raise HTTPException(status_code=404, detail=f"지식 파일을 찾을 수 없습니다: {name}")


@router.put("/knowledge/{name}")
def update_knowledge(name: str, req: KnowledgeUpdate):
    """지식 파일 수정."""
    cos = deps.get_cos()

    knowledge_dir = cos.knowledge._dir

    # 쓰기는 확장자를 포함한 정확한 파일명만 허용한다
    knowledge_dir.mkdir(parents=True, exist_ok=True)
    file_path = safe_child(knowledge_dir, name, SAFE_FILENAME)
    file_path.write_text(req.content, encoding="utf-8")

    # 다시 로드
    cos.knowledge.load_all()

    return {"status": "ok", "name": name, "size": len(req.content)}


@router.get("/fewshot")
def list_fewshot():
    """Few-shot 예시 태그 목록."""
    cos = deps.get_cos()
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


@router.get("/fewshot/search")
def search_fewshot(q: str = ""):
    """Few-shot 예시 검색."""
    cos = deps.get_cos()
    if not q:
        return {"results": []}
    results = cos.fewshot.search(q, cos.emotion.get_state())
    return {
        "results": [
            {"user": e.user, "character": e.character, "emotion_state": e.emotion_state}
            for e in results
        ]
    }
