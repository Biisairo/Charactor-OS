import concurrent.futures
import json
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from src.analysis import ConflictClassifier, EmotionAnalyzer, MemoryExtractor
from src.call_log import CallLogger
from src.character_layout import CharacterLayout
from src.llm.client import Client
from src.metrics import CallMeter
from src.modules import (
    EmotionModule,
    FewShotModule,
    HistoryModule,
    KnowledgeModule,
    MemoryModule,
    PersonaModule,
    ReflectionReviewer,
)
from src.pricing import estimate_cost, load_price_table
from src.prompts.engine import PromptEngine
from src.trace import PipelineTrace
from src.validity import provider_error_reason

# 프로바이더가 거부를 돌려줬을 때 응답 생성을 다시 시도하는 횟수 (최초 호출 포함).
# 거부는 결정론적이지 않아 재시도로 통과하는 경우가 많다. 평가 하네스도 같은 값을 쓴다.
MAX_RESPONSE_ATTEMPTS = 3


class ProviderRefusalError(RuntimeError):
    """프로바이더가 재시도 후에도 요청을 거부했다.

    캐릭터 발화가 아니므로 사용자에게 응답으로 보여주거나 상태에 저장하지 않는다.
    캐릭터 톤의 대체 문장으로 감추지 않는다 — 인프라 장애를 캐릭터 반응으로
    위장하는 것은 은폐다 (TASK-11 3.11.5).
    """


# ANSI 색상 코드
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"


# 모듈별 헤더 정의
MODULE_HEADERS = {
    "orchestrator": f"{Colors.BLUE}{Colors.BOLD}[Orchestrator]{Colors.RESET}",
    "react": f"{Colors.CYAN}{Colors.BOLD}[ReAct]{Colors.RESET}",
    "response": f"{Colors.YELLOW}{Colors.BOLD}[Response]{Colors.RESET}",
    "postprocess": f"{Colors.GREEN}{Colors.BOLD}[PostProcess]{Colors.RESET}",
}


@dataclass
class _Turn:
    """턴 하나가 지나는 동안 Stage들이 공유하는 상태.

    실패 여부를 값으로 들고 다니면 "앞이 실패했으면 멈춘다"를 호출부에서
    한 줄로 표현할 수 있다.
    """

    user_input: str
    started: float
    trace: PipelineTrace | None = None
    response: str = ""
    failed: bool = False


@dataclass
class ContextBundle:
    persona: str
    knowledge_context: str
    emotion: str
    memory_context: str
    history_context: str


class CharacterOS:
    """3-stage 파이프라인으로 캐릭터 대화를 오케스트레이션한다."""

    def __init__(
        self,
        character_dir: str,
        memory_db_path: str | None = None,
        emotion_save_path: str | None = None,
        history_save_path: str | None = None,
        debug: bool = False,
        output: Callable[[str], None] = print,
        debug_output: Callable[[str], None] | None = None,
        model_type: str = "api",
        local_model: str = "mlx-community/Qwen3.5-4B-MLX-4bit",
        adapter_path: str | None = None,
        no_review: bool = False,
        trace: bool = False,
        client: object | None = None,
        call_logger: CallLogger | None = None,
    ):
        """
        Args:
            client: LLM 클라이언트. 주입하면 `model_type`을 무시하고 그대로 사용한다.
                테스트에서 실제 API 호출 없이 파이프라인을 검증하기 위한 진입점이다.
            call_logger: LLM 호출 운영 로거. 생략하면 기본 경로에 기록한다.
                테스트에서는 비활성 로거를 주입해 파일 쓰기를 막는다.
            memory_db_path, emotion_save_path, history_save_path:
                생략하면 `character_dir/state/` 아래로 파생한다. 캐릭터마다
                상태가 분리되는 것이 기본 동작이다 (TASK-17). 명시하면 그
                경로를 그대로 쓴다 — 평가 하네스와 테스트가 상태를 임시
                디렉토리로 격리할 때 쓰는 경로다.
        """
        self._character_dir = Path(character_dir)
        layout = CharacterLayout.of(self._character_dir)
        self._trace_enabled = trace
        self._last_trace: PipelineTrace | None = None
        persona_path = str(layout.persona_path)
        knowledge_dir = str(layout.knowledge_dir)
        examples_dir = str(layout.examples_dir)
        memory_db_path = memory_db_path or str(layout.memory_db_path)
        emotion_save_path = emotion_save_path or str(layout.emotion_save_path)
        history_save_path = history_save_path or str(layout.history_save_path)
        self._debug = debug
        self._output = output
        self._debug_output = debug_output or (lambda msg: None)
        self._debug_logs: list[str] = []  # 디버그 로그 버퍼

        self._log("=" * 60, module="orchestrator")
        self._log("CharacterOS 초기화 시작", module="orchestrator")
        self._log("=" * 60, module="orchestrator")

        if client is not None:
            self.client = client
            self._is_local = False
            self._log("주입된 클라이언트 사용", module="orchestrator")
        elif model_type == "local":
            from src.llm.local_client import LocalClient

            raw_client = LocalClient(model_name=local_model, adapter_path=adapter_path)
            # 로컬 클라이언트는 캐시 없이 직접 사용
            self.client = raw_client
            self._is_local = True
            self._log(f"로컬 모델 사용: {local_model}", module="orchestrator")
            if adapter_path:
                self._log(f"LoRA 어댑터: {adapter_path}", module="orchestrator")
        else:
            self.client = Client()
            self._is_local = False

        self.persona = PersonaModule(persona_path)
        self.emotion = EmotionModule(
            save_path=emotion_save_path, debug=debug, debug_output=self._debug_output
        )
        self.memory = MemoryModule(
            db_path=memory_db_path,
            embedding_fn=self._embed,
            debug=debug,
            debug_output=self._debug_output,
        )
        self.knowledge = KnowledgeModule(knowledge_dir)
        self.history = HistoryModule(save_path=history_save_path)
        self.fewshot = FewShotModule(examples_dir, embedding_fn=self._embed)
        self.prompt_engine = PromptEngine(max_tokens=3000)

        # 정적 데이터 로드
        self._log("[모듈 로드] persona 로드 중...", module="orchestrator")
        self.persona.load()
        self._log(
            f"[모듈 로드] persona 완료: {self.persona._data.get('name')}", module="orchestrator"
        )

        self._log("[모듈 로드] knowledge 로드 중...", module="orchestrator")
        self.knowledge.load_all()
        char_count = len(self.knowledge.get_characters())
        self._log(
            f"[모듈 로드] knowledge 완료: 캐릭터 {char_count}개",
            module="orchestrator",
        )

        self._log("[모듈 로드] fewshot 로드 중...", module="orchestrator")
        self.fewshot.load_all()
        self._log(
            f"[모듈 로드] fewshot 완료: 태그 {len(self.fewshot.get_all_tags())}개",
            module="orchestrator",
        )

        # 정적 자산의 로드 문제를 드러낸다 (REQ-06-1).
        # 조용히 넘기면 프롬프트 품질이 원인 불명으로 떨어진다.
        self._report_asset_issues()

        # 동적 데이터 로드
        self._log("[모듈 로드] emotion 로드 중...", module="orchestrator")
        self.emotion.load()
        self._log(f"[모듈 로드] emotion 완료: {self.emotion.get_state()}", module="orchestrator")

        self._log("[모듈 로드] memory 로드 중...", module="orchestrator")
        self.memory.load()
        self._log(
            f"[모듈 로드] memory 완료: {len(self.memory._memories)}개 기억", module="orchestrator"
        )

        self._log("[모듈 로드] history 로드 중...", module="orchestrator")
        self.history.load()
        self._log(f"[모듈 로드] history 완료: {len(self.history._turns)}턴", module="orchestrator")

        # 감정 트리거 주입 (persona → emotion)
        triggers = self.persona.get_emotion_triggers()
        if triggers:
            self.emotion.set_triggers(triggers)
            self._log(f"[모듈 로드] 감정 트리거 주입: {len(triggers)}개", module="orchestrator")

        # LLM 호출 관찰 — 라벨별 프록시를 주입해 호출 지점을 자동 분류한다.
        #   · 턴 스냅샷(`--trace`)은 trace 플래그에 따른다
        #   · 운영 로그는 디버그 플래그와 무관하게 항상 남긴다
        self._price_table = load_price_table()
        self._call_logger = call_logger if call_logger is not None else CallLogger()
        self._turn_id = ""
        self._meter = CallMeter(
            sink=self._log_call, capture_payload=self._call_logger.capture_payload
        )

        # Reflection 리뷰어 (no_review=True면 비활성화)
        self._no_review = no_review
        if not no_review:
            self.reviewer = ReflectionReviewer(
                client=self._meter.wrap(self.client, "reflection"),
                persona=self.persona,
                emotion=self.emotion,
                debug=debug,
                debug_output=self._debug_output,
            )
            self._log("[모듈 로드] ReflectionReviewer 활성화", module="orchestrator")
        else:
            self.reviewer = None
            self._log(
                "[모듈 로드] ReflectionReviewer 비활성화 (--no-review)", module="orchestrator"
            )

        self._log("=" * 60, module="orchestrator")
        self._log("CharacterOS 초기화 완료", module="orchestrator")
        self._log("=" * 60, module="orchestrator")

    def _log_prompt(self, module: str, prompt: str) -> None:
        """분석 층이 만든 프롬프트를 디버그 로그로 넘긴다."""
        self._log(f"[{module} 프롬프트]", module="postprocess", data=prompt)

    def _report_asset_issues(self) -> None:
        """정적 자산 모듈이 모은 로드 문제를 로그로 올린다 (REQ-06-1 · 06-3).

        의도된 폴백과 예기치 않은 실패를 다른 말로 적는다. 실패는 사용자에게도
        보인다 — 예시가 조용히 0개가 되는 것이 원래 문제였다.
        """
        issues = [
            *self.knowledge.load_issues,
            *self.fewshot.load_issues,
        ]
        if not issues:
            return

        failures = [i for i in issues if not i.expected]
        for issue in issues:
            self._log(f"[모듈 로드] {issue.describe()}", module="orchestrator")

        if failures:
            self._output(
                f"경고: 정적 자산 {len(failures)}건을 읽지 못했습니다 — "
                + ", ".join(i.filename for i in failures)
            )

    def _log(
        self, message: str, module: str = "orchestrator", data: dict | str | None = None
    ) -> None:
        """디버그 로그를 출력한다."""
        # 항상 버퍼에 저장 (debug 여부无关)
        header = MODULE_HEADERS.get(module, MODULE_HEADERS["orchestrator"])
        if message.startswith("=") or message.startswith("-"):
            log_line = f"[CharacterOS] {message}"
        else:
            log_line = f"{header} {message}"
        # 개행이있으면줄별로저장
        if "\n" in log_line:
            for line in log_line.split("\n"):
                self._debug_logs.append(line)
        else:
            self._debug_logs.append(log_line)
        if data is not None:
            if isinstance(data, dict):
                json_str = json.dumps(data, ensure_ascii=False, indent=2)
                for line in json_str.split("\n"):
                    self._debug_logs.append(line)
            else:
                for line in str(data).split("\n"):
                    self._debug_logs.append(line)
        # 버퍼 크기 제한
        if len(self._debug_logs) > 500:
            self._debug_logs = self._debug_logs[-300:]
        # debug 모드일 때만 출력
        if not self._debug:
            return
        self._debug_output(log_line)
        if data is not None:
            if isinstance(data, dict):
                self._debug_output(json.dumps(data, ensure_ascii=False, indent=2))
            else:
                self._debug_output(str(data))

    _embedding_model = None  # 클래스 레벨 캐시

    def _embed(self, text: str):
        """임베딩 함수 (sentence-transformers 사용)."""
        if CharacterOS._embedding_model is None:
            from sentence_transformers import SentenceTransformer

            CharacterOS._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        return CharacterOS._embedding_model.encode(text, normalize_embeddings=True)

    def _gather_context(self, user_input: str) -> ContextBundle:
        """Stage 1: ReAct — 컨텍스트 번들 생성."""
        self._log("")
        self._log("-" * 40, module="react")
        self._log("Stage 1: 컨텍스트 수집 시작", module="react")
        self._log("-" * 40, module="react")
        self._log(f"사용자 입력: {user_input}", module="react")

        # 항상 포함
        self._log("persona 프롬프트 로드", module="react")
        persona_prompt = self.persona.to_system_prompt()
        self._log(f"persona 프롬프트 ({len(persona_prompt)}자)", module="react")

        self._log("knowledge 프롬프트 로드", module="react")
        knowledge_prompt = self.knowledge.to_prompt()
        self._log(f"knowledge 프롬프트 ({len(knowledge_prompt)}자)", module="react")

        # ReAct loop 시뮬레이션 (도구 호출)
        self._log("get_emotion 호출", module="react")
        emotion_prompt = self.emotion.to_prompt()
        self._log(f"emotion 결과: {self.emotion.get_state()}", module="react")

        self._log("search_memory 호출", module="react")
        memory_prompt = self.memory.to_prompt(user_input)
        self._log(f"memory 프롬프트 ({len(memory_prompt)}자)", module="react")

        self._log("get_recent_history 호출", module="react")
        history_prompt = self.history.to_prompt()
        self._log(f"history 프롬프트 ({len(history_prompt)}자)", module="react")

        self._log("-" * 40, module="react")
        self._log("Stage 1: 컨텍스트 수집 완료", module="react")
        self._log("-" * 40, module="react")

        return ContextBundle(
            persona=persona_prompt,
            knowledge_context=knowledge_prompt,
            emotion=emotion_prompt,
            memory_context=memory_prompt,
            history_context=history_prompt,
        )

    def _generate_response(self, user_input: str, context: ContextBundle) -> str:
        """Stage 2: Response — 응답 생성 (Reflection 포함)."""
        self._log("")
        self._log("-" * 40, module="response")
        self._log("Stage 2: 응답 생성 시작", module="response")
        self._log("-" * 40, module="response")

        system_prompt = self.prompt_engine.assemble_system_prompt(
            user_input=user_input,
            persona=self.persona,
            emotion=self.emotion,
            memory=self.memory,
            knowledge=self.knowledge,
            history=self.history,
            fewshot=self.fewshot,
        )

        self._log("시스템 프롬프트:", module="response")
        self._log(system_prompt, module="response")
        self._log(f"사용자 입력: {user_input}", module="response")

        def _call_llm(extra_user_msg: str | None = None) -> str:
            """응답을 한 번 생성한다. 프로바이더 거부는 재시도하고, 지속되면 예외를 던진다.

            초안과 재생성이 모두 이 함수를 지나므로, 여기 한 곳에서 막으면
            Stage 2를 빠져나가는 모든 경로가 덮인다.
            """
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ]
            if extra_user_msg:
                messages.append({"role": "user", "content": extra_user_msg})

            reason = None
            for attempt in range(MAX_RESPONSE_ATTEMPTS):
                result = self._meter.wrap(self.client, "response").call_llm(
                    messages=messages,
                    tools=[],
                    use_stream=False,
                    mute=True,
                )
                reason = provider_error_reason(result.content)
                if reason is None:
                    return result.content
                self._log(
                    f"프로바이더 거부 ({attempt + 1}/{MAX_RESPONSE_ATTEMPTS}): {reason}",
                    module="response",
                )

            raise ProviderRefusalError(reason or "프로바이더 거부")

        # 초안 생성
        self._log("LLM 호출 시작 (초안)...", module="response")
        draft = _call_llm()
        self._log(f"초안 완료 ({len(draft)}자): {draft}", module="response")

        # Reflection 검토 (활성화된 경우)
        if self.reviewer is not None:
            self._log("Reflection 검토 시작...", module="response")

            def _regenerate(feedback: str) -> str:
                self._log(f"재생성 with 피드백: {feedback}", module="response")
                return _call_llm(
                    extra_user_msg=f"[검토 피드백]\n{feedback}\n\n위 피드백을 반영하여 응답을 개선하세요."
                )

            response = self.reviewer.review_and_improve(user_input, draft, _regenerate)
            self._log(f"Reflection 완료 ({len(response)}자)", module="response")
        else:
            response = draft

        self._log(f"최종 응답: {response}", module="response")
        self._log("-" * 40, module="response")
        self._log("Stage 2: 응답 생성 완료", module="response")
        self._log("-" * 40, module="response")

        return response

    def _post_process(self, user_input: str, response: str) -> None:
        """Stage 3: Post-processing — 상태 업데이트 (병렬, 롤백 지원)."""
        self._log("")
        self._log("-" * 40, module="postprocess")
        self._log("Stage 3: 후처리 시작", module="postprocess")
        self._log("-" * 40, module="postprocess")

        # 롤백용 스냅샷
        emotion_snap = self.emotion.snapshot()
        memory_count = self.memory.snapshot_count()
        history_count = self.history.count()

        # 이전 대화 컨텍스트 생성 (흐름과 맥락을 보기 위함)
        history_context = self.history.to_prompt(n=10)
        self._log(f"이전 대화 컨텍스트 ({len(history_context)}자)", module="postprocess")

        try:
            # 1. Emotion + History 병렬 실행 (서로 독립)
            self._log("emotion + history 병렬 실행", module="postprocess")
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                emotion_future = pool.submit(
                    self.emotion.update,
                    user_input,
                    response,
                    EmotionAnalyzer(
                        self._meter.wrap(self.client, "emotion"), on_prompt=self._log_prompt
                    ),
                    history_context=history_context,
                )
                history_future = pool.submit(self._add_history, user_input, response)
                emotion_future.result()
                history_future.result()

            self._log(f"emotion 상태: {self.emotion.get_state()}", module="postprocess")
            self._log(f"history 턴 수: {self.history.count()}", module="postprocess")

            # 2. Memory 업데이트 (감정 결과 사용 → emotion 완료 후)
            self._log("memory.update() 호출", module="postprocess")
            memory_client = self._meter.wrap(self.client, "memory")
            self.memory.update(
                user_input,
                response,
                self.emotion.get_state(),
                MemoryExtractor(memory_client, on_prompt=self._log_prompt),
                ConflictClassifier(memory_client, on_prompt=self._log_prompt),
                history_context=history_context,
            )
            self._log(f"memory 개수: {self.memory.snapshot_count()}", module="postprocess")

        except Exception as e:
            # 롤백: 모든 상태를 이전으로 복원
            self._log(f"후처리 실패! 롤백 중... ({e})", module="postprocess")
            self.emotion.restore(emotion_snap)
            self.memory.pop_last_n(self.memory.snapshot_count() - memory_count)
            self.history.pop_last_n(self.history.count() - history_count)
            raise

        # 3. 영속화 (실패 시 롤백 불필요 — 아직 저장 안 됨)
        self._log("영속화 시작...", module="postprocess")
        self.emotion.save()
        self.memory.save()
        self.history.save()
        self._log("영속화 완료", module="postprocess")

        self._log("-" * 40, module="postprocess")
        self._log("Stage 3: 후처리 완료", module="postprocess")
        self._log("-" * 40, module="postprocess")

    def _add_history(self, user_input: str, response: str) -> None:
        """대화 기록 추가 (병렬 실행용)."""
        self.history.add_turn("user", user_input)
        self.history.add_turn("character", response)

    def _log_call(self, record) -> None:
        """계측된 호출 1건을 운영 로그로 넘긴다 (비동기)."""
        model = getattr(getattr(self.client, "env", None), "model", "") or ""
        self._call_logger.log_call(
            record,
            model=model,
            cost_usd=estimate_cost(
                model, record.prompt_tokens, record.completion_tokens, self._price_table
            ),
            turn_id=self._turn_id,
            character=self._character_dir.name,
        )

    def _log_turn(self, user_input: str, response: str, started: float, error: str = "") -> None:
        """턴 요약을 운영 로그에 남긴다. 호출 단위 기록과 turn_id로 이어진다."""
        self._call_logger.log_turn(
            turn_id=self._turn_id,
            character=self._character_dir.name,
            user_input=user_input,
            response=response,
            metrics=self._collect_metrics(),
            duration_ms=(time.perf_counter() - started) * 1000,
            error=error,
        )

    def _collect_metrics(self) -> dict:
        """이번 턴의 LLM 호출 집계 + 추정 비용."""
        summary = self._meter.summary()
        model = getattr(getattr(self.client, "env", None), "model", "") or "(미지정)"
        summary["model"] = model
        summary["cost_usd"] = estimate_cost(
            model,
            summary["prompt_tokens"],
            summary["completion_tokens"],
            self._price_table,
        )
        return summary

    def chat(self, user_input: str) -> str | None:
        """전체 3-stage 파이프라인 실행. 어느 단계든 실패하면 None 반환.

        Stage마다 트레이스 구간을 열고 닫고, 실패하면 로그·출력·턴 기록·트레이스
        마감을 똑같이 수행한다. 그 반복을 `_run_stage()`가 맡는다 — 분리 전에는
        세 번 복제되어 있어 오류 처리를 고치려면 세 곳을 함께 고쳐야 했다.
        """
        self._log("")
        self._log("=" * 60, module="orchestrator")
        self._log(f"chat() 호출: {user_input}", module="orchestrator")
        self._log("=" * 60, module="orchestrator")

        self._turn_id = uuid.uuid4().hex[:12]
        self._meter.reset()  # 계측은 턴 단위
        turn = _Turn(user_input=user_input, started=time.perf_counter())

        if self._trace_enabled:
            turn.trace = PipelineTrace()
            turn.trace.start(user_input)

        context = self._run_stage(
            turn,
            "context",
            lambda: self._gather_context(user_input),
            failure_message="컨텍스트 수집 실패",
            details=lambda ctx: {
                "persona_len": len(ctx.persona),
                "knowledge_len": len(ctx.knowledge_context),
                "emotion_len": len(ctx.emotion),
                "memory_len": len(ctx.memory_context),
                "history_len": len(ctx.history_context),
            },
        )
        if turn.failed:
            return None

        response = self._run_stage(
            turn,
            "response",
            lambda: self._generate_response(user_input, context),
            failure_message="응답 생성 실패",
            details=lambda r: {"response_len": len(r)},
        )
        if turn.failed:
            return None

        # 여기부터 응답은 확정이다. 이후 실패는 응답도 함께 버린다 —
        # 상태에 남지 않은 대화를 보여주면 다음 턴의 컨텍스트와 어긋난다.
        turn.response = response

        self._run_stage(
            turn,
            "postprocess",
            lambda: self._post_process(user_input, response),
            failure_message="후처리 실패, 대화가 저장되지 않았습니다",
            details=lambda _: {
                "emotion_state": dict(self.emotion.get_state()),
                "memory_count": self.memory.snapshot_count(),
                "history_count": self.history.count(),
            },
        )
        if turn.failed:
            return None

        self._output(f"캐릭터: {response}")
        self._log_turn(user_input, response, turn.started)
        self._finish_trace(turn, response=response)

        self._log("")
        self._log("=" * 60, module="orchestrator")
        self._log("chat() 완료", module="orchestrator")
        self._log("=" * 60, module="orchestrator")

        return response

    def _run_stage(self, turn, name, action, failure_message, details):
        """Stage 하나를 실행하고 트레이스·실패 처리를 일괄 수행한다.

        실패하면 `turn.failed`를 세우고 None을 반환한다. 예외를 밖으로 올리지 않는
        것은 호출부가 같은 except를 세 번 쓰지 않게 하기 위해서다.
        """
        span = turn.trace.add_stage(name) if turn.trace else None
        try:
            result = action()
        except Exception as e:
            turn.failed = True
            self._log(f"{name} 실패: {e}", module="orchestrator")
            self._output(f"\n오류: {failure_message} — {e}")
            self._log_turn(turn.user_input, turn.response, turn.started, error=f"{name}: {e}")
            self._finish_trace(turn, error=str(e))
            return None

        if span:
            span.finish()
            span.details = details(result)
        return result

    def _finish_trace(self, turn, response: str = "", error: str | None = None) -> None:
        if not turn.trace:
            return
        turn.trace.metrics = self._collect_metrics()
        if error:
            turn.trace.finish(error=error)
        else:
            turn.trace.finish(response=response)
        self._last_trace = turn.trace
