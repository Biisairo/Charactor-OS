"""응답 유효성 판별.

프로바이더가 콘텐츠 필터·레이트 리밋 등으로 거부하면, 그 거부 메시지가
캐릭터 응답 자리에 그대로 담겨 온다. 이를 캐릭터 발화로 취급하면 두 곳에서
문제가 생긴다.

- **런타임** (TASK-11): 거부 문자열이 사용자에게 캐릭터 응답으로 표시되고,
  히스토리·기억에 저장되어 이후 턴의 프롬프트까지 오염시킨다.
- **평가** (TASK-01): 인프라 문제가 '형편없는 캐릭터 품질'로 집계된다.
  실제로 reflection-off 실행에서 3건이 1점으로 집계되어 Reflection 효과가
  +0.67로 부풀려졌다. 3건을 제외하면 +0.10이었다.

**디코딩 폭주**도 같은 성격의 문제다. 한 글자가 수천 번 반복된 응답을 캐릭터
발화로 취급하면 히스토리·기억이 오염되고(런타임), 평가는 그것을 품질로 집계한다
— 실제로 130,755자 응답에 판정자가 만점을 줬다 (SPEC-12 P-14).

두 곳이 서로 다른 판별 기준을 들고 있으면 반드시 어긋나므로,
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


# 같은 문자가 이만큼 연속되면 디코딩이 무너진 것으로 본다.
#
# 실측 근거: 정상 응답 319건의 최대 연속 반복은 **8회**이고(웃음소리·말줄임표),
# 관측된 폭주는 **130,666회**였다. 사이가 4자리수만큼 벌어져 있어 어디를 잡아도
# 되지만, 정상 최대의 3배 이상을 두어 웃음소리 하나에 정상 응답을 버리지 않는다.
MAX_CHAR_RUN = 30


def degenerate_reason(response: str) -> str | None:
    """응답이 디코딩 폭주로 보이면 사유를, 아니면 None을 반환한다.

    출력 상한(`RESPONSE_MAX_OUTPUT_TOKENS`)이 1차 방어선이고 이것이 2차다.
    상한 안에서도 반복은 일어날 수 있다.

    문자 단위 반복만 본다. 구절 반복("안녕 안녕 안녕…")은 잡지 못하지만,
    관측된 사례가 문자 반복이므로 근거 없이 넓히지 않는다.
    """
    if not response:
        return None

    run_char = ""
    run_length = 0
    for char in response:
        if char == run_char:
            run_length += 1
            if run_length > MAX_CHAR_RUN:
                return f"같은 문자({char!r})가 {MAX_CHAR_RUN}회 넘게 반복됨 — 디코딩 폭주"
        else:
            run_char = char
            run_length = 1
    return None


def unusable_response_reason(response: str) -> str | None:
    """캐릭터 발화로 쓸 수 없는 응답인지 판별한다.

    런타임과 평가가 같은 문을 지나게 하는 통합 입구다.
    """
    return provider_error_reason(response) or degenerate_reason(response)


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
