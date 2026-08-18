"""작업기억 — 턴을 넘어 남는 미해결 사고 (SPEC-09 REQ-RA-50 ~ 57).

Memory가 *확정된 사실*을 쌓는다면, 여기 남는 것은 아직 답이 없는 것들이다.
"왜 갑자기 말수가 줄었을까", "시험 결과를 기다리는 것 같다" 같은 것.
확정 사실을 여기 담으면 검증된 기억과 추측이 섞이므로 `kind`로 막는다.

해소되면 사라지고, 오래 방치되면 스스로 떨어진다. 잘못된 추측을 캐릭터가
영원히 붙들고 있으면 대화가 왜곡되기 때문이다.
"""

from __future__ import annotations

import copy
import json
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from src.agent.schemas import THOUGHT_KINDS, NewThought
from src.modules.asset_issue import AssetLoadIssue

STALE_TURNS = 20
MAX_ITEMS = 10


@dataclass
class WorkingMemoryItem:
    """미해결 사고 하나."""

    id: str
    kind: str
    content: str
    created_turn: int
    last_seen_turn: int
    confidence: float = 0.0


class WorkingMemoryModule:
    """동적 모듈. Stage 3의 스냅샷·롤백 계약을 따른다."""

    def __init__(
        self,
        save_path: str,
        stale_turns: int = STALE_TURNS,
        max_items: int = MAX_ITEMS,
    ):
        self._save_path = Path(save_path)
        self._stale_turns = stale_turns
        self._max_items = max_items
        self._items: list[WorkingMemoryItem] = []
        self._load_issues: list[AssetLoadIssue] = []

    # ─── 조회 ───

    @property
    def items(self) -> list[WorkingMemoryItem]:
        return self._items

    @property
    def load_issues(self) -> list[AssetLoadIssue]:
        return self._load_issues

    def count(self) -> int:
        return len(self._items)

    def to_prompt(self) -> str:
        if not self._items:
            return ""

        lines = ["[미해결 사고]"]
        for item in self._items:
            if item.kind == "hypothesis":
                lines.append(f"- ({item.id}) [추측 {item.confidence:.1f}] {item.content}")
            else:
                lines.append(f"- ({item.id}) {item.content}")
        return "\n".join(lines)

    # ─── 갱신 ───

    def apply(
        self, resolved_ids: list[str], new_thoughts: list[NewThought], turn_index: int
    ) -> None:
        """해소분을 지우고 새 사고를 올린 뒤, 수명이 다한 것을 떨군다."""
        resolved = set(resolved_ids)
        self._items = [item for item in self._items if item.id not in resolved]

        for thought in new_thoughts:
            if thought.kind not in THOUGHT_KINDS:
                continue
            existing = self._find(thought.kind, thought.content)
            if existing is not None:
                existing.last_seen_turn = turn_index
                continue
            self._items.append(
                WorkingMemoryItem(
                    id=uuid.uuid4().hex[:8],
                    kind=thought.kind,
                    content=thought.content,
                    created_turn=turn_index,
                    last_seen_turn=turn_index,
                    confidence=thought.confidence,
                )
            )

        self._drop_stale(turn_index)
        self._enforce_limit()

    def _find(self, kind: str, content: str) -> WorkingMemoryItem | None:
        for item in self._items:
            if item.kind == kind and item.content == content:
                return item
        return None

    def _drop_stale(self, turn_index: int) -> None:
        self._items = [
            item for item in self._items if turn_index - item.last_seen_turn <= self._stale_turns
        ]

    def _enforce_limit(self) -> None:
        if len(self._items) <= self._max_items:
            return
        survivors = {
            item.id
            for item in sorted(self._items, key=lambda i: i.last_seen_turn)[-self._max_items :]
        }
        self._items = [item for item in self._items if item.id in survivors]

    # ─── 영속화 ───

    def load(self) -> None:
        self._items = []
        self._load_issues = []

        if not self._save_path.exists():
            return

        raw = self._save_path.read_text(encoding="utf-8")
        try:
            data = json.loads(raw)
            self._items = [WorkingMemoryItem(**item) for item in data.get("items", [])]
        except (json.JSONDecodeError, TypeError) as e:
            # 조용히 빈 상태로 시작하면 사고가 사라진 이유를 아무도 모른다 (REQ-06-1).
            self._load_issues.append(
                AssetLoadIssue(
                    filename=self._save_path.name,
                    reason=f"{type(e).__name__}: {e} — 빈 작업기억으로 시작",
                    expected=False,
                )
            )
            self._items = []

    def save(self) -> None:
        self._save_path.parent.mkdir(parents=True, exist_ok=True)
        self._save_path.write_text(
            json.dumps({"items": [asdict(i) for i in self._items]}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ─── 롤백 ───

    def snapshot(self) -> list[WorkingMemoryItem]:
        return copy.deepcopy(self._items)

    def restore(self, snap: list[WorkingMemoryItem]) -> None:
        self._items = copy.deepcopy(snap)
