"""응답 출력 상한 (SPEC-12 REQ-21-29).

상한이 없어서 한 글자(`甩`)가 3,266회 반복된 **130,755자·756초** 응답이 나왔고,
판정자는 그것에 만점을 줬다. 프롬프트로 "간결하게"를 지시해도 모델은 지키지
않을 수 있고, 생성 후 잘라내는 것은 지연을 줄이지 못한다.

클라이언트에 상한 인자는 처음부터 있었다. 뇌(800)와 Reflection(400)은 쓰고
있었고, **응답 생성만 쓰지 않았다.**
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.character_os import RESPONSE_MAX_OUTPUT_TOKENS
from tests.conftest import MockClient, make_character_os


@pytest.fixture
def cos(character_dir: Path, tmp_path: Path):
    client = MockClient(response="응, 기억하지.")
    return make_character_os(character_dir, tmp_path, client), client


class TestOutputLimit:
    def test_response_call_specifies_a_limit(self, cos):
        """Stage 2 가 상한 없이 호출되면 폭주를 막을 방법이 없다."""
        os_, client = cos

        os_.chat("안녕")

        assert RESPONSE_MAX_OUTPUT_TOKENS in client.max_tokens_seen

    def test_every_response_call_is_limited(self, cos):
        """초안과 재생성이 모두 같은 통로를 지난다. 어느 경로도 새지 않아야 한다."""
        os_, client = cos

        os_.chat("안녕")

        limited = [t for t in client.max_tokens_seen if t is not None]
        assert limited, "응답 생성 호출에 상한이 하나도 지정되지 않았다"

    def test_limit_covers_the_observed_maximum(self):
        """상한 없이 관측된 최대 사용량(1,354토큰)을 자르지 않아야 한다.

        근거는 운영 로그 1,122건이다 (SPEC-12 4.6). 평가 기록만 보면 597토큰인데,
        그 표본은 짧은 평가 질의에 치우쳐 있어 상한을 과소 설정하게 만든다.
        """
        assert RESPONSE_MAX_OUTPUT_TOKENS >= 1354

    def test_limit_is_not_unbounded(self):
        """상한이 사실상 무한이면 폭주를 막지 못한다.

        폭주 사례는 131,072토큰이었다.
        """
        assert RESPONSE_MAX_OUTPUT_TOKENS <= 3000
