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


class SwitchCharacterRequest(BaseModel):
    character_id: str


class CreateCharacterRequest(BaseModel):
    name: str
    identity: str = ""
