import json
import time
from collections.abc import Callable
from pathlib import Path

# 기존 감정에 두는 무게. 새 값은 (1 - 이 값)만큼 섞인다.
# 대화 한 번으로 감정이 뒤집히지 않게 하려는 장치다.
EXISTING_EMOTION_WEIGHT = 0.7


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
        analyzer,
        history_context: str = "",
    ) -> None:
        """대화를 반영해 감정 상태를 갱신한다.

        LLM 상호작용은 `analyzer`가 맡는다. 이 메서드는 **무엇을 받아들일지**만
        정하므로, 더블 없이 규칙만 바꿔 끼워 검증할 수 있다.

        순서가 의미를 갖는다. decay와 트리거는 분석 **이전**에 적용된다 —
        분석기에 넘기는 "현재 상태"가 이미 시간이 흐른 뒤의 값이어야 하기 때문이다.

        Args:
            analyzer: `analyze(user_input, response, current, history) -> EmotionAnalysis`
        """
        self._log_debug("")
        self._log_debug("update() 호출")
        self._log_debug(f"사용자 입력: {user_input}")
        self._log_debug(f"캐릭터 응답: {character_response[:50]}...")

        self.apply_decay()
        self._apply_triggers(user_input)

        analysis = analyzer.analyze(
            user_input, character_response, dict(self._emotions), history_context
        )
        self._log_debug(f"significant: {analysis.significant}")

        if not analysis.significant:
            self._log_debug("미미한 변화 — 감정 상태 유지")
        else:
            self._apply(analysis)

        self._log_debug(f"최종 감정 상태: {self._emotions}")

    def _apply(self, analysis) -> None:
        """분석 결과를 감정 상태에 반영한다."""
        for name in analysis.remove:
            if name in self._emotions:
                old_value = self._emotions.pop(name)
                self._log_debug(f"  감정 '{name}' 제거 (이전 값: {old_value:.3f})")

        for name, value in analysis.emotions.items():
            if not isinstance(value, (int, float)) or not 0 <= value <= 1:
                continue
            if name in self._emotions:
                # 급변을 막는다 — 기존 값에 무게를 둔 가중 평균으로 섞는다.
                old_value = self._emotions[name]
                self._emotions[name] = round(
                    old_value * EXISTING_EMOTION_WEIGHT + value * (1 - EXISTING_EMOTION_WEIGHT), 3
                )
                self._log_debug(
                    f"  감정 '{name}' 블렌딩: {old_value:.3f} + {value:.3f}"
                    f" -> {self._emotions[name]:.3f}"
                )
            else:
                self._emotions[name] = value
                self._log_debug(f"  감정 '{name}' 추가: {value:.3f}")

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
