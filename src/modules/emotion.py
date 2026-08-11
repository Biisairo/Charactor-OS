import json
import time
from collections.abc import Callable
from pathlib import Path


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
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"


class EmotionModule:
    """캐릭터의 감정 상태를 추적한다."""

    def __init__(
        self,
        decay_rate: float = 0.1,
        save_path: str | None = None,
        debug: bool = False,
        debug_output: Callable[[str], None] | None = None,
    ):
        self._decay_rate = decay_rate
        self._save_path = Path(save_path) if save_path else None
        self._emotions: dict[str, float] = {}
        self._triggers: list[dict] = []  # persona에서 주입되는 감정 트리거
        self._debug = debug
        self._debug_output = debug_output or (lambda msg: None)

    def _log_debug(self, message: str, data=None) -> None:
        if not self._debug:
            return
        prefix = f"{Colors.YELLOW}{Colors.BOLD}[Emotion]{Colors.RESET}"
        self._debug_output(f"{prefix} {message}")
        if data is not None:
            if isinstance(data, dict):
                self._debug_output(json.dumps(data, ensure_ascii=False, indent=2))
            else:
                self._debug_output(str(data))

    def get_state(self) -> dict[str, float]:
        """현재 감정 상태를 반환한다."""
        return dict(self._emotions)

    def snapshot(self) -> dict[str, float]:
        """롤백용 스냅샷을 반환한다."""
        return dict(self._emotions)

    def restore(self, snap: dict[str, float]) -> None:
        """스냅샷으로 상태를 복원한다."""
        self._emotions = dict(snap)

    def set_triggers(self, triggers: list[dict]) -> None:
        """페르소나에서 감정 트리거를 주입한다.

        Args:
            triggers: [{"keyword": "아버지", "emotion": "분노", "intensity": 0.7}, ...]
        """
        self._triggers = triggers or []
        self._log_debug(f"트리거 설정: {len(self._triggers)}개")

    def _apply_triggers(self, user_input: str) -> None:
        """사용자 입력에서 트리거 키워드를 감지하여 감정을 적용한다."""
        if not self._triggers:
            return

        user_lower = user_input.lower()
        for trigger in self._triggers:
            keyword = trigger.get("keyword", "")
            if keyword and keyword.lower() in user_lower:
                emotion = trigger.get("emotion", "")
                intensity = trigger.get("intensity", 0.5)
                if emotion:
                    current = self._emotions.get(emotion, 0.0)
                    # 높은 쪽 유지
                    self._emotions[emotion] = max(current, intensity)
                    self._log_debug(f"트리거 감지: '{keyword}' → {emotion}={intensity}")

    def to_prompt(self) -> str:
        """감정 상태를 프롬프트 문자열로 변환한다."""
        if not self._emotions:
            return "[현재 감정 상태]\n특별한 감정 상태 없음"

        lines = ["[현재 감정 상태]"]
        for name, value in sorted(self._emotions.items(), key=lambda x: -x[1]):
            lines.append(f"- {name}: {value:.3f}")
        lines.append("\n이 감정 상태에 맞게 응답의 톤을 조절하세요.")
        return "\n".join(lines)

    def apply_decay(self) -> None:
        """모든 감정에 decay를 적용하고, 0.05 이하를 제거한다."""
        self._log_debug(f"apply_decay() 호출 (decay_rate={self._decay_rate})")
        self._log_debug(f"decay 전: {self._emotions}")

        decayed = {}
        for name, value in self._emotions.items():
            new_value = value * (1 - self._decay_rate)
            if new_value > 0.05:
                decayed[name] = new_value
            else:
                self._log_debug(f"  감정 '{name}' 제거 (값 {new_value:.3f} <= 0.05)")
        self._emotions = decayed

        self._log_debug(f"decay 후: {self._emotions}")

    def update(
        self,
        user_input: str,
        character_response: str,
        client,
        history_context: str = "",
        prompt_callback: Callable[[str, str], None] | None = None,
    ) -> None:
        """대화를 분석하여 감정 상태를 업데이트한다. (별도 LLM 호출)

        Args:
            user_input: 사용자 입력
            character_response: 캐릭터 응답
            client: LLM 클라이언트
            history_context: 이전 대화 맥락 (흐름과 맥락을 보기 위함)
            prompt_callback: 프롬프트 로깅 콜백 (module, prompt)
        """
        self._log_debug("")
        self._log_debug("update() 호출")
        self._log_debug(f"사용자 입력: {user_input}")
        self._log_debug(f"캐릭터 응답: {character_response[:50]}...")

        self.apply_decay()
        self._apply_triggers(user_input)

        # 현재 감정 상태를 프롬프트에 포함
        current_emotions_str = (
            json.dumps(self._emotions, ensure_ascii=False) if self._emotions else "{}"
        )

        prompt = f"""캐릭터의 감정 상태를 업데이트하세요. 변화가 미미하면 현재 상태를 유지합니다.

{history_context}

사용자: {user_input}
캐릭터: {character_response}

현재 감정 상태:
{current_emotions_str}

다음 JSON 형식으로 반환하세요:
{{
    "emotions": {{
        "감정이름": 0.0~1.0,
        ...
    }},
    "remove": ["제거할 감정 이름", ...],
    "significant": true/false
}}

규칙:
- 감정이 없는 상태(빈 {{}})가 기본값입니다. 중립=정상 상태입니다
- significant가 true일 때만 emotions/remove를 채우세요
- 일상적 대화(안부, 짧은 대답, 정보 교환)는 significant=false로 반환하세요
- 감정 변화가 명확할 때만 significant=true: 감동, 분노, 슬픔, 큰 기쁨, 충격 등
- 감정 이름은 자유롭게 정하세요 (예: 행복, 슬픔, 분노, 설렘, 피로, 향수 등)
- 값은 0.0에서 1.0 사이
- 현재 감정 중 이 대화로 인해 완전히 사라진 것만 remove에 포함하세요
- 애매하면 유지하세요. 감정은 쉽게 변하지 않습니다
- 대부분의 대화에서 significant=false여야 합니다"""

        self._log_debug("LLM 호출 (감정 분석)")
        self._log_debug(f"현재 감정 상태: {current_emotions_str}")

        if prompt_callback:
            prompt_callback("emotion", prompt)

        result = client.call_llm(
            messages=[
                {"role": "system", "content": "감정 분석기. JSON만 반환하세요."},
                {"role": "user", "content": prompt},
            ],
            tools=[],
            use_stream=False,
            mute=True,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "emotion_update",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "emotions": {
                                "type": "object",
                                "additionalProperties": {
                                    "type": "number",
                                    "maximum": 1.0,
                                    "minimum": 0.0,
                                },
                            },
                            "remove": {"type": "array", "items": {"type": "string"}},
                            "significant": {"type": "boolean"},
                        },
                        "required": ["significant"],
                    },
                },
            },
        ).content

        self._log_debug(f"LLM 응답: {result}")

        try:
            data = json.loads(result)
            significant = data.get("significant", False)
            self._log_debug(f"significant: {significant}")

            if not significant:
                self._log_debug("미미한 변화 — 감정 상태 유지")
                return

            new_emotions = data.get("emotions", {})
            remove_list = data.get("remove", [])
            self._log_debug(f"추출된 감정: {new_emotions}")
            self._log_debug(f"제거 대상: {remove_list}")

            # 제거할 감정 처리
            for name in remove_list:
                if name in self._emotions:
                    old_value = self._emotions.pop(name)
                    self._log_debug(f"  감정 '{name}' 제거 (이전 값: {old_value:.3f})")

            # 새 감정 추가/업데이트 (블렌딩: 가중 평균)
            for name, value in new_emotions.items():
                if isinstance(value, (int, float)) and 0 <= value <= 1:
                    if name in self._emotions:
                        old_value = self._emotions[name]
                        blended = old_value * 0.7 + value * 0.3
                        self._emotions[name] = round(blended, 3)
                        self._log_debug(
                            f"  감정 '{name}' 블렌딩: {old_value:.3f} + {value:.3f} -> {self._emotions[name]:.3f}"
                        )
                    else:
                        self._emotions[name] = value
                        self._log_debug(f"  감정 '{name}' 추가: {value:.3f}")
        except (json.JSONDecodeError, AttributeError) as e:
            self._log_debug(f"JSON 파싱 실패: {e}")

        self._log_debug(f"최종 감정 상태: {self._emotions}")

    def save(self) -> None:
        """감정 상태를 JSON 파일로 저장한다."""
        if not self._save_path:
            return
        self._log_debug(f"save() 호출 -> {self._save_path}")
        self._save_path.parent.mkdir(parents=True, exist_ok=True)
        data = dict(self._emotions)
        data["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._save_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        self._log_debug("저장 완료")

    def load(self) -> None:
        """JSON 파일에서 감정 상태를 로드한다."""
        if not self._save_path or not self._save_path.exists():
            self._log_debug(f"load() 호출 - 파일 없음: {self._save_path}")
            return
        self._log_debug(f"load() 호출 <- {self._save_path}")
        data = json.loads(self._save_path.read_text(encoding="utf-8"))
        self._emotions = {
            k: v for k, v in data.items() if k != "last_updated" and isinstance(v, (int, float))
        }
        self._log_debug(f"로드 완료: {self._emotions}")
