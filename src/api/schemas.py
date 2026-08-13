"""요청·응답 모델."""

from __future__ import annotations

from pydantic import BaseModel


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
    first_message: str | None = None
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
    secrets: list[str] | None = None
    meta_awareness: str | None = None
    examples: list[dict] | None = None


class KnowledgeUpdate(BaseModel):
    content: str


class SwitchCharacterRequest(BaseModel):
    character_id: str


class CreateCharacterRequest(BaseModel):
    name: str
    identity: str = ""
    static_data: CharacterStaticData | None = None


class CharacterKnowledgeData(BaseModel):
    """질문지 응답으로 채울 knowledge/ 섹션."""

    world: dict | None = None
    locations: list[dict] | None = None
    relationships: list[dict] | None = None
    timeline: list[dict] | None = None
    freeform: str | None = None


class CharacterExamplesData(BaseModel):
    """질문지 응답으로 채울 examples/ 시나리오 파일."""

    greeting: dict | None = None
    comfort: dict | None = None
    conflict: dict | None = None
    humor: dict | None = None
    daily: dict | None = None


class CharacterStaticData(BaseModel):
    """새 캐릭터 생성 시 static/ 전체를 한 번에 채우는 페이로드.

    빠진 섹션은 생성하지 않는다 — 질문지를 건너뛴 부분은 파일을 남기지
    않아 KnowledgeModule·FewShotModule이 빈 지식을 로드하지 않게 한다.
    """

    persona: PersonaUpdate | None = None
    knowledge: CharacterKnowledgeData | None = None
    examples: CharacterExamplesData | None = None
