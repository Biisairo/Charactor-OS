"""대화 — 파이프라인을 실행하는 유일한 경로.

스트리밍(WebSocket) 경로는 제거했다. 검토를 거치지 않은 초안이 사용자에게
도달하는 통로였다 (TASK-11).
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.api import deps
from src.api.schemas import ChatRequest, ChatResponse

router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """대화 — 워커 스레드에서 순차 처리."""
    cos = deps.get_cos()

    def _do_chat():
        # 첫 대화면 캐릭터의 first_message를 히스토리에 심는다.
        # 클라이언트는 이미 그 문장을 첫 버블로 보여줬는데, 여기서 심지 않으면
        # LLM 컨텍스트에 없어 사용자가 오프너를 언급할 때 끊긴다. lazy로 두어
        # 사용자가 아무 말도 하지 않으면 히스토리에 흔적이 남지 않는다.
        if cos.history.count() == 0:
            first = cos.persona._data.get("first_message")
            if first:
                cos.history.add_turn("character", first)
        response = cos.chat(req.message)
        emotion = cos.emotion.get_state()
        return ChatResponse(response=response, emotion=emotion)

    result = await deps.run_in_worker(_do_chat)

    # chat()은 턴이 실패하면 None을 돌려준다 (프로바이더 거부, Stage 실패 등).
    # 캐릭터 발화가 아니므로 200으로 내보내지 않는다 — 실패는 실패로 드러나야 한다.
    if result.response is None:
        raise HTTPException(
            status_code=502, detail="응답 생성에 실패했습니다. 잠시 후 다시 시도해 주세요."
        )

    return result
