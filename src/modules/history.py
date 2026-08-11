import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class ConversationTurn:
    role: str  # "user" | "character"
    content: str
    timestamp: float


class HistoryModule:
    """캐릭터와 사용자 간의 대화 기록을 관리한다."""

    def __init__(self, save_path: str | None = None, max_turns: int = 100):
        self._save_path = Path(save_path) if save_path else None
        self._max_turns = max_turns
        self._turns: list[ConversationTurn] = []

    def add_turn(self, role: str, content: str) -> None:
        """대화 한 턴을 추가한다."""
        self._turns.append(
            ConversationTurn(
                role=role,
                content=content,
                timestamp=time.time(),
            )
        )
        if len(self._turns) > self._max_turns:
            self._turns = self._turns[-self._max_turns :]

    def get_recent(self, n: int = 10) -> list[ConversationTurn]:
        """최근 n개 대화를 반환한다."""
        return self._turns[-n:]

    def to_prompt(self, n: int = 10) -> str:
        """최근 n개 대화를 프롬프트 문자열로 변환한다."""
        recent = self.get_recent(n)
        if not recent:
            return "[최근 대화]\n대화 없음"

        lines = ["[최근 대화]"]
        for turn in recent:
            label = "사용자" if turn.role == "user" else "캐릭터"
            lines.append(f"{label}: {turn.content}")
        return "\n".join(lines)

    def save(self) -> None:
        """대화 기록을 JSON 파일로 저장한다."""
        if not self._save_path:
            return
        self._save_path.parent.mkdir(parents=True, exist_ok=True)
        data = {"turns": [asdict(t) for t in self._turns]}
        self._save_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self) -> None:
        """JSON 파일에서 대화 기록을 로드한다."""
        if not self._save_path or not self._save_path.exists():
            return
        data = json.loads(self._save_path.read_text(encoding="utf-8"))
        self._turns = [ConversationTurn(**t) for t in data.get("turns", [])]

    def count(self) -> int:
        """현재 턴 수를 반환한다 (롤백용)."""
        return len(self._turns)

    def pop_last_n(self, n: int) -> None:
        """가장 최근 n개 턴을 제거한다 (롤백용)."""
        if n <= 0:
            return
        self._turns = self._turns[:-n]
