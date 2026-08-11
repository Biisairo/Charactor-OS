"""ReflectionReviewer 단위 테스트."""

from __future__ import annotations

from src.modules.reflection import ReflectionReviewer, ReviewResult
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
