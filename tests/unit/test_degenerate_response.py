"""디코딩 폭주 응답 판별 (SPEC-12 P-14 · TASK-25).

한 글자(`甩`)가 3,266회 반복된 130,755자 응답이 나왔고, **판정자는 만점을
줬다.** 출력 상한이 1차 방어선이라면 이것은 2차 방어선이다 — 상한 안에서도
반복이 일어날 수 있고, 그 응답을 캐릭터 발화로 취급하면 히스토리·기억이
오염되고 평가는 그것을 품질로 집계한다.

`provider_error_reason`과 같은 자리에 두는 이유도 같다. 런타임과 평가가 서로
다른 판별 기준을 들고 있으면 반드시 어긋난다 (REQ-11-5).
"""

from __future__ import annotations

import pytest

from src.validity import MAX_CHAR_RUN, degenerate_reason, unusable_response_reason


class TestDegenerate:
    def test_long_run_is_detected(self):
        assert degenerate_reason("흠... " + "甩" * (MAX_CHAR_RUN + 1)) is not None

    def test_reason_mentions_the_repetition(self):
        reason = degenerate_reason("甩" * (MAX_CHAR_RUN + 10))

        assert "반복" in reason

    def test_observed_degeneration_is_detected(self):
        """실제로 관측된 형태 — 문장 중간에 한 글자가 무너져 이어진다."""
        response = "내가 아는 건 활 쏘는 법, 산길에서 추격자 따" + "甩" * 3266

        assert degenerate_reason(response) is not None

    @pytest.mark.parametrize(
        "response",
        [
            "응, 기억하지. 면접 망했다고 했잖아.",
            "아 당연히 기억하지 ㅋㅋㅋㅋㅋㅋㅋㅋ 면접 봤다고 했잖아",  # 정상 최대 8회
            "그렇구나...... 많이 속상했겠다",
            "",
        ],
    )
    def test_normal_response_is_not_flagged(self, response):
        """관측된 정상 응답의 최대 연속 반복은 8회다 (n=319). 오탐이 없어야 한다."""
        assert degenerate_reason(response) is None

    def test_threshold_has_headroom_over_observed_normal(self):
        """정상 최대(8회)와 임계 사이에 여유가 있어야 한다.

        여유가 없으면 웃음소리 하나에 정상 응답이 버려진다.
        """
        assert MAX_CHAR_RUN >= 16


class TestUnusableReason:
    def test_provider_error_is_still_detected(self):
        """통합 판별이 기존 거부 감지를 잃지 않아야 한다."""
        assert unusable_response_reason("Your request was rejected.") is not None

    def test_degeneration_is_detected(self):
        assert unusable_response_reason("甩" * (MAX_CHAR_RUN + 1)) is not None

    def test_normal_response_passes(self):
        assert unusable_response_reason("응, 기억하지.") is None
