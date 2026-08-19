"""ReflectionReviewer 단위 테스트."""

from __future__ import annotations

from src.modules.reflection import (
    ReflectionReviewer,
    ReviewResult,
    parse_review_response,
)
from tests.conftest import MockClient, MockResponse

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class SequentialMockClient(MockClient):
    """MockClient subclass that makes `tools` optional and returns
    a sequence of responses across successive call_llm invocations.
    """

    def __init__(self, responses: list[MockResponse]):
        super().__init__(response=responses[0].content if responses else "")
        self._responses = list(responses)
        self._call_index = 0

    def call_llm(
        self,
        messages: list,
        tools: list | None = None,
        use_stream: bool = False,
        mute: bool = True,
        response_format: dict | None = None,
        token_callback=None,
        max_tokens=None,
    ):
        resp = self._responses[self._call_index]
        self._call_index += 1
        self.call_count += 1
        self.last_messages = messages
        self.last_kwargs = {
            "tools": tools,
            "use_stream": use_stream,
            "mute": mute,
            "response_format": response_format,
        }
        from src.llm.client import TrimmedMessage

        content = resp.content
        if token_callback and use_stream:
            for char in content:
                token_callback(char)

        return TrimmedMessage(
            content=content,
            role="assistant",
            reasoning_content=resp.reasoning_content,
            tool_calls=[],
            usage=None,
        )


def _make_persona(name: str = "홍길동", rules: list[str] | None = None) -> object:
    """Minimal persona stub with _data dict."""

    class _Persona:
        def __init__(self):
            self._data: dict = {
                "name": name,
                "speaking_style": {
                    "summary": "친근한 말투",
                    "tone": "부드러운",
                },
                "behavior": {
                    "rules": rules if rules is not None else ["욕설 금지", "존댓말 사용"],
                },
            }

    return _Persona()


def _make_emotion() -> object:
    """Minimal emotion stub with get_state()."""

    class _Emotion:
        def get_state(self) -> dict:
            return {"기쁨": 0.8, "흥분": 0.3}

    return _Emotion()


def _make_reviewer(
    client: MockClient,
    name: str = "홍길동",
    rules: list[str] | None = None,
    debug: bool = False,
) -> ReflectionReviewer:
    return ReflectionReviewer(
        client=client,
        persona=_make_persona(name, rules),
        emotion=_make_emotion(),
        debug=debug,
    )


# ---------------------------------------------------------------------------
# 1. ReviewResult dataclass fields
# ---------------------------------------------------------------------------


class TestReviewResultFields:
    def test_approved_only(self):
        result = ReviewResult(approved=True)
        assert result.approved is True
        assert result.feedback == ""

    def test_approved_false_with_feedback(self):
        result = ReviewResult(approved=False, feedback="말투가 부적절합니다")
        assert result.approved is False
        assert result.feedback == "말투가 부적절합니다"

    def test_defaults(self):
        result = ReviewResult(approved=False)
        assert result.approved is False
        assert result.feedback == ""


# ---------------------------------------------------------------------------
# 2. review() with PASS response -> approved=True
# ---------------------------------------------------------------------------


class TestReviewPass:
    def test_pass_approves(self):
        client = SequentialMockClient([MockResponse(content="PASS")])
        reviewer = _make_reviewer(client)

        result = reviewer.review("안녕하세요", "안녕하세요! 반갑습니다.")

        assert result.approved is True
        assert result.feedback == ""
        assert client.call_count == 1


# ---------------------------------------------------------------------------
# 3. review() with FAIL response -> approved=False with feedback
# ---------------------------------------------------------------------------


class TestReviewFail:
    def test_fail_returns_feedback(self):
        client = SequentialMockClient([MockResponse(content="FAIL: 말투가 캐릭터와 다릅니다")])
        reviewer = _make_reviewer(client)

        result = reviewer.review("안녕하세요", "안녕하세요!")

        assert result.approved is False
        assert "말투가 캐릭터와 다릅니다" in result.feedback

    def test_fail_without_colon(self):
        client = SequentialMockClient([MockResponse(content="FAIL 부적절한 표현")])
        reviewer = _make_reviewer(client)

        result = reviewer.review("안녕하세요", "안녕하세요!")

        assert result.approved is False
        assert "부적절한 표현" in result.feedback


# ---------------------------------------------------------------------------
# 4. review_and_improve() with PASS on first try -> returns draft unchanged
# ---------------------------------------------------------------------------


class TestReviewAndImprovePassFirst:
    def test_returns_draft_unchanged(self):
        client = SequentialMockClient([MockResponse(content="PASS")])
        reviewer = _make_reviewer(client)

        draft = "안녕하세요! 반갑습니다."
        result = reviewer.review_and_improve(
            "안녕하세요", draft, regenerate_fn=lambda fb: "should not be called"
        )

        assert result == draft
        assert client.call_count == 1


# ---------------------------------------------------------------------------
# 5. review_and_improve() with FAIL then PASS -> calls regenerate_fn once
# ---------------------------------------------------------------------------


class TestReviewAndImproveFailThenPass:
    def test_calls_regenerate_once(self):
        client = SequentialMockClient(
            [MockResponse(content="FAIL: 말투 수정 필요"), MockResponse(content="PASS")]
        )
        reviewer = _make_reviewer(client)

        call_log: list[str] = []

        def regen(feedback: str) -> str:
            call_log.append(feedback)
            return "수정된 응답입니다."

        result = reviewer.review_and_improve("안녕하세요", "초안", regenerate_fn=regen)

        assert result == "수정된 응답입니다."
        assert len(call_log) == 1
        assert "말투 수정 필요" in call_log[0]
        assert client.call_count == 2


# ---------------------------------------------------------------------------
# 6. review_and_improve() with all FAILs -> returns last response
#    after MAX_REVIEW_ITERATIONS
# ---------------------------------------------------------------------------


class TestReviewAndImproveAllFails:
    def test_returns_last_regenerated_after_max_iterations(self):
        # MAX_REVIEW_ITERATIONS = 2, so loop runs 2 times
        client = SequentialMockClient(
            [
                MockResponse(content="FAIL: 첫 번째 문제"),
                MockResponse(content="FAIL: 두 번째 문제"),
            ]
        )
        reviewer = _make_reviewer(client)

        regen_count = 0

        def regen(feedback: str) -> str:
            nonlocal regen_count
            regen_count += 1
            return f"재생성-{regen_count}"

        result = reviewer.review_and_improve("안녕하세요", "초안", regenerate_fn=regen)

        # Loop: i=0 -> FAIL -> regen -> "재생성-1"
        #        i=1 -> review("재생성-1") -> FAIL -> regen -> "재생성-2"
        # Falls out of loop, returns current = "재생성-2"
        assert result == "재생성-2"
        assert regen_count == 2
        assert client.call_count == 2


# ---------------------------------------------------------------------------
# 7. _build_review_prompt contains persona name and rules
# ---------------------------------------------------------------------------


class TestBuildReviewPrompt:
    def test_contains_persona_name(self):
        client = SequentialMockClient([MockResponse(content="PASS")])
        reviewer = _make_reviewer(client, name="이순신")

        prompt = reviewer._build_review_prompt("안녕", "안녕하세요")

        assert "이순신" in prompt

    def test_contains_rules(self):
        client = SequentialMockClient([MockResponse(content="PASS")])
        rules = ["반말 사용 금지", "주제 이탈 금지"]
        reviewer = _make_reviewer(client, rules=rules)

        prompt = reviewer._build_review_prompt("안녕", "안녕하세요")

        for rule in rules:
            assert rule in prompt

    def test_contains_user_input_and_draft(self):
        client = SequentialMockClient([MockResponse(content="PASS")])
        reviewer = _make_reviewer(client)

        prompt = reviewer._build_review_prompt("오늘 날씨 어때?", "맑습니다!")

        assert "오늘 날씨 어때?" in prompt
        assert "맑습니다!" in prompt


# ---------------------------------------------------------------------------
# 8. 평가 결과로 추가된 검토 기준 (TASK-08)
#
# 아래 세 기준은 평가 하네스가 실제 결함을 관측한 뒤 추가되었다.
# 프롬프트에서 빠지면 그 결함이 다시 통과하므로 회귀 테스트로 고정한다.
# ---------------------------------------------------------------------------


class TestReviewCriteriaFromEvaluation:
    def _prompt(self) -> str:
        client = SequentialMockClient([MockResponse(content="PASS")])
        return _make_reviewer(client)._build_review_prompt("안녕", "안녕하세요")

    def test_checks_response_language(self):
        """관측된 결함: 감사 인사에 중국어로 응답."""
        prompt = self._prompt()

        assert "한국어" in prompt
        assert "중국어" in prompt

    def test_checks_era_consistency(self):
        """관측된 결함: 조선 시대 캐릭터가 '서울'이라는 현대 지명 사용.

        TASK-19 이후 이 기준은 캐릭터의 `era`에서 파생된다. 시대를 아는
        캐릭터에서는 여전히 검사하되, 예시를 조선시대로 박아두지는 않는다.
        """
        reviewer = ReflectionReviewer(
            client=MockClient(),
            persona=_make_persona(),
            emotion=_make_emotion(),
            knowledge=_Knowledge("조선 중기 (16세기)"),
        )

        prompt = reviewer._build_review_prompt("안녕하시오", "안녕하세요")

        assert "시대 정합성" in prompt
        assert "조선 중기 (16세기)" in prompt

    def test_checks_persona_break(self):
        """관측된 결함: '파이썬 코드 짜줘'에 실제 코드를 작성."""
        prompt = self._prompt()

        assert "페르소나 유지" in prompt
        assert "코드" in prompt
        assert "AI" in prompt

    def test_instructs_to_preserve_established_facts(self):
        """관측된 결함: 재생성이 기억 활용 점수를 떨어뜨림 (−0.17)."""
        prompt = self._prompt()

        assert "사실은 반드시 유지" in prompt

    def test_discourages_trivial_rejection(self):
        """재생성은 비용이 든다. 사소한 이유로 FAIL을 남발하면 지연만 늘어난다."""
        prompt = self._prompt()

        assert "사소한 취향 차이로 FAIL" in prompt


# ---------------------------------------------------------------------------
# TASK-19 — 검토 기준을 캐릭터에서 파생하고, 판정을 관대하게 읽는다
#
# 착수 시점 FAIL율: hong-gil-dong 44% · han-so-min 66%.
# 소민 FAIL의 43%가 "시대 정합성"이었다 — 2020년대 스트리머에게 조선시대
# 기준을 적용한 결과다.
# ---------------------------------------------------------------------------


class _Knowledge:
    """세계관 스텁. era가 없는 캐릭터도 표현할 수 있어야 한다."""

    def __init__(self, era: str = ""):
        self._era = era

    def era(self) -> str:
        return self._era


def _reviewer_with_world(era: str, name: str = "소민찌") -> ReflectionReviewer:
    return ReflectionReviewer(
        client=MockClient(),
        persona=_make_persona(name),
        emotion=_make_emotion(),
        knowledge=_Knowledge(era),
    )


class TestEraComesFromTheCharacter:
    """REQ-19-1 · 19-2"""

    def _prompt(self, era: str) -> str:
        return _reviewer_with_world(era)._build_review_prompt("안녕", "안녕하세요")

    def test_modern_character_prompt_states_its_own_era(self):
        prompt = self._prompt("2020년대 후반 대한민국 서울")

        assert "2020년대 후반 대한민국 서울" in prompt

    def test_modern_character_prompt_has_no_joseon_examples(self):
        """'서울 → 한양'을 현대 스트리머에게 들이대면 정상 응답이 FAIL된다."""
        prompt = self._prompt("2020년대 후반 대한민국 서울")

        assert "한양" not in prompt

    def test_historical_character_still_gets_its_era(self):
        prompt = self._prompt("조선 중기 (16세기)")

        assert "조선 중기 (16세기)" in prompt

    def test_missing_era_drops_the_criterion(self):
        """근거 없는 기준으로 FAIL을 주느니 검사하지 않는다 (REQ-19-2)."""
        prompt = self._prompt("")

        assert "시대 정합성" not in prompt

    def test_other_criteria_survive_without_era(self):
        prompt = self._prompt("")

        assert "응답 언어" in prompt
        assert "페르소나 유지" in prompt

    def test_knowledge_is_optional(self):
        """knowledge 없이 만들어도 검토는 동작해야 한다 (기존 호출부 호환)."""
        reviewer = ReflectionReviewer(
            client=MockClient(), persona=_make_persona(), emotion=_make_emotion()
        )

        assert "시대 정합성" not in reviewer._build_review_prompt("안녕", "안녕하세요")


class TestVerdictParsingIsLenient:
    """REQ-19-3 — 모델은 `- PASS:`, `**PASS**`처럼 꾸며서 답한다."""

    def test_list_bullet_pass(self):
        assert parse_review_response("- PASS").approved is True

    def test_list_bullet_pass_with_comment(self):
        assert parse_review_response("- PASS: 응답이 모든 기준을 충족합니다.").approved is True

    def test_bold_pass(self):
        assert parse_review_response("**PASS**").approved is True

    def test_heading_pass(self):
        assert parse_review_response("## PASS").approved is True

    def test_bold_fail_is_still_fail(self):
        result = parse_review_response("**FAIL**: 영어 단어가 섞였습니다")

        assert result.approved is False
        assert "영어" in result.feedback

    def test_bullet_fail_keeps_feedback(self):
        result = parse_review_response("- FAIL: 말투가 어긋납니다")

        assert result.approved is False
        assert "말투" in result.feedback

    def test_plain_text_is_still_fail(self):
        """판정을 못 읽으면 통과시키지 않는다 — 안전한 쪽으로 기운다."""
        assert parse_review_response("음... 애매하네요").approved is False

    def test_empty_is_fail(self):
        assert parse_review_response("").approved is False


class TestReviewStatsAreRecorded:
    """REQ-19-4 — FAIL율을 계속 추적하려면 판정이 값으로 남아야 한다."""

    def test_verdicts_are_collected(self):
        client = SequentialMockClient(
            [
                MockResponse(content='{"verdict": "FAIL", "feedback": "말투"}'),
                MockResponse(content='{"verdict": "PASS", "feedback": ""}'),
            ]
        )
        reviewer = _make_reviewer(client)

        reviewer.review_and_improve("안녕", "초안", lambda _fb: "재생성본")

        assert reviewer.last_verdicts == ["FAIL", "PASS"]

    def test_regeneration_count_is_recorded(self):
        client = SequentialMockClient(
            [
                MockResponse(content='{"verdict": "FAIL", "feedback": "말투"}'),
                MockResponse(content='{"verdict": "PASS", "feedback": ""}'),
            ]
        )
        reviewer = _make_reviewer(client)

        reviewer.review_and_improve("안녕", "초안", lambda _fb: "재생성본")

        assert reviewer.last_regenerations == 1

    def test_clean_pass_records_no_regeneration(self):
        client = SequentialMockClient([MockResponse(content='{"verdict": "PASS"}')])
        reviewer = _make_reviewer(client)

        reviewer.review_and_improve("안녕", "초안", lambda _fb: "재생성본")

        assert reviewer.last_regenerations == 0
        assert reviewer.last_verdicts == ["PASS"]

    def test_stats_reset_between_turns(self):
        client = SequentialMockClient(
            [
                MockResponse(content='{"verdict": "FAIL", "feedback": "말투"}'),
                MockResponse(content='{"verdict": "PASS"}'),
                MockResponse(content='{"verdict": "PASS"}'),
            ]
        )
        reviewer = _make_reviewer(client)

        reviewer.review_and_improve("안녕", "초안", lambda _fb: "재생성본")
        reviewer.review_and_improve("또 안녕", "초안2", lambda _fb: "재생성본2")

        assert reviewer.last_verdicts == ["PASS"]
        assert reviewer.last_regenerations == 0
