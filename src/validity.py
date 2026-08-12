"""응답 유효성 판별.

프로바이더가 콘텐츠 필터·레이트 리밋 등으로 거부하면, 그 거부 메시지가
캐릭터 응답 자리에 그대로 담겨 온다. 이를 캐릭터 발화로 취급하면 두 곳에서
문제가 생긴다.

- **런타임** (TASK-11): 거부 문자열이 사용자에게 캐릭터 응답으로 표시되고,
  히스토리·기억에 저장되어 이후 턴의 프롬프트까지 오염시킨다.
- **평가** (TASK-01): 인프라 문제가 '형편없는 캐릭터 품질'로 집계된다.
  실제로 reflection-off 실행에서 3건이 1점으로 집계되어 Reflection 효과가
  +0.67로 부풀려졌다. 3건을 제외하면 +0.10이었다.

두 곳이 서로 다른 문구 목록을 들고 있으면 반드시 어긋나므로,
`src`에 두고 런타임과 평가 하네스가 함께 쓴다 (REQ-11-5).

이 모듈에는 LLM 호출이 없다.
"""

from __future__ import annotations

# 프로바이더 거부·오류 응답에 나타나는 문구.
# 캐릭터는 한국어로 답하므로 이 영문 문구들이 응답에 섞일 여지가 사실상 없다.
# 잘못 제외하면 실제 품질 문제를 놓치므로, 모호한 문구는 넣지 않는다.
PROVIDER_ERROR_PHRASES = (
    "request was rejected",
    "considered high risk",
    "content management policy",
    "violates our content policy",
    "rate limit exceeded",
    "internal server error",
    "service temporarily unavailable",
    "upstream error",
    "bad gateway",
)


def provider_error_reason(response: str) -> str | None:
    """응답이 캐릭터 발화가 아니라 프로바이더 오류로 보이면 사유를, 아니면 None을 반환한다.

    주의: 캐릭터가 시대에 맞지 않게 답하거나 다른 언어로 답하는 것은 '품질 문제'이지
    프로바이더 오류가 아니다. 그런 응답은 정상 채점되어야 하므로 여기서 걸러내지 않는다.
    """
    if not response or not response.strip():
        return "빈 응답"

    lowered = response.lower()
    for phrase in PROVIDER_ERROR_PHRASES:
        if phrase in lowered:
            return f"프로바이더 오류 응답 ({phrase})"

    return None
