"""캐릭터 디렉토리 레이아웃 (TASK-17).

캐릭터 하나가 갖는 것은 두 가지다.

    characters/<id>/static/   사람이 쓴 정체성 — persona · knowledge · examples
    characters/<id>/state/    에이전트가 쌓은 경험 — 기억 · 감정 · 히스토리

**에이전트는 정적 파일을 절대 수정하지 않는다**는 것이 이 시스템의 핵심
불변식이고, 이 디렉토리 경계가 그것을 눈에 보이게 만든다. `static/`은 git이
추적하고 `state/`는 추적하지 않는다.

경로 조립을 여기 모은 이유는, `"static"` 같은 문자열이 오케스트레이터·API
라우터·평가 하네스에 흩어지면 한 곳만 빠뜨려도 조용히 어긋나기 때문이다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

STATIC_DIRNAME = "static"
STATE_DIRNAME = "state"

MEMORY_DB_FILENAME = "memories.db"
EMOTION_FILENAME = "emotions.json"
HISTORY_FILENAME = "history.json"
WORKING_MEMORY_FILENAME = "working_memory.json"


@dataclass(frozen=True)
class CharacterLayout:
    """캐릭터 디렉토리 하나의 경로 모음."""

    root: Path

    @classmethod
    def of(cls, character_dir: Path | str) -> CharacterLayout:
        return cls(root=Path(character_dir))

    # ─── 정적 (git 추적) ───

    @property
    def static_dir(self) -> Path:
        return self.root / STATIC_DIRNAME

    @property
    def persona_path(self) -> Path:
        return self.static_dir / "persona.yaml"

    @property
    def knowledge_dir(self) -> Path:
        return self.static_dir / "knowledge"

    @property
    def examples_dir(self) -> Path:
        return self.static_dir / "examples"

    # ─── 동적 (gitignore) ───

    @property
    def state_dir(self) -> Path:
        return self.root / STATE_DIRNAME

    @property
    def memory_db_path(self) -> Path:
        return self.state_dir / MEMORY_DB_FILENAME

    @property
    def emotion_save_path(self) -> Path:
        return self.state_dir / EMOTION_FILENAME

    @property
    def history_save_path(self) -> Path:
        return self.state_dir / HISTORY_FILENAME

    @property
    def working_memory_path(self) -> Path:
        return self.state_dir / WORKING_MEMORY_FILENAME

    def is_character(self) -> bool:
        """캐릭터 디렉토리로 인정할 수 있는가. persona.yaml의 존재로 판단한다."""
        return self.persona_path.exists()
