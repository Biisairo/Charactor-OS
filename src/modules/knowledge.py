from pathlib import Path

import yaml


class KnowledgeModule:
    """캐릭터가 속한 세계관, 관계, 타임라인, 장소 등 구조화된 지식을 관리한다.

    YAML 파일은 type 필드에 따라 구조화되고,
    .md/.json/.txt 등은 자유 형식(freeform)으로 처리된다.
    """

    SUPPORTED_EXTENSIONS = {".json", ".md", ".yaml", ".yml", ".txt"}

    def __init__(self, knowledge_dir: str):
        self._dir = Path(knowledge_dir)
        self._world: dict | None = None
        self._characters: list[dict] = []
        self._relationships: list[dict] = []
        self._timeline: list[dict] = []
        self._locations: list[dict] = []
        self._freeform: list[tuple[str, str]] = []  # (filename, content)

    def load_all(self) -> None:
        """지식 디렉토리의 모든 파일을 로드하여 구조화/비구조화 데이터를 저장한다."""
        self._world = None
        self._characters = []
        self._relationships = []
        self._timeline = []
        self._locations = []
        self._freeform = []

        if not self._dir.exists():
            return

        self._scan_directory(self._dir)

    def _scan_directory(self, directory: Path) -> None:
        """디렉토리를 재귀 스캔하여 파일을 분류한다."""
        for item in sorted(directory.iterdir()):
            if item.is_dir():
                # 하위 디렉토리 재귀 스캔 (characters/, freeform/ 등)
                self._scan_directory(item)
                continue

            if not item.is_file() or item.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                continue

            # YAML 파일은 type 필드로 구조화 시도
            if item.suffix.lower() in {".yaml", ".yml"}:
                try:
                    with open(item, encoding="utf-8") as f:
                        data = yaml.safe_load(f)
                    if isinstance(data, dict) and "type" in data:
                        self._classify_structured(data, item.name)
                        continue
                except Exception:
                    pass  # 파싱 실패 시 freeform으로 처리

            # 자유 형식
            content = item.read_text(encoding="utf-8")
            self._freeform.append((item.name, content))

    def _classify_structured(self, data: dict, filename: str) -> None:
        """type 필드에 따라 구조화 데이터를 분류한다."""
        doc_type = data.get("type", "freeform")

        if doc_type == "world":
            self._world = data
        elif doc_type == "character":
            self._characters.append(data)
        elif doc_type == "relationships":
            rels = data.get("relationships", [])
            self._relationships.extend(rels)
        elif doc_type == "timeline":
            events = data.get("events", [])
            self._timeline.extend(events)
        elif doc_type == "locations":
            locs = data.get("locations", [])
            self._locations.extend(locs)
        else:
            # freeform: content 필드가 있으면 그것을, 없으면 YAML 전체를 텍스트로
            content = data.get("content", yaml.dump(data, allow_unicode=True))
            self._freeform.append((filename, content))

    # ─── Getters ───

    def get_world(self) -> dict | None:
        """세계관 정보를 반환한다."""
        return self._world

    def get_characters(self) -> list[dict]:
        """타 캐릭터 목록을 반환한다."""
        return self._characters

    def get_character(self, name: str) -> dict | None:
        """특정 캐릭터 정보를 반환한다."""
        for char in self._characters:
            if char.get("name") == name:
                return char
        return None

    def get_relationships(self) -> list[dict]:
        """관계 그래프를 반환한다."""
        return self._relationships

    def get_relationships_for(self, character: str) -> list[dict]:
        """특정 캐릭터 관련 관계만 반환한다."""
        return [
            r for r in self._relationships if r.get("from") == character or r.get("to") == character
        ]

    def get_timeline(self) -> list[dict]:
        """타임라인 이벤트를 반환한다."""
        return self._timeline

    def get_locations(self) -> list[dict]:
        """장소 목록을 반환한다."""
        return self._locations

    # ─── 프롬프트 변환 ───

    def to_prompt(self) -> str:
        """전체 지식을 프롬프트 문자열로 변환한다. (기존 호환)"""
        parts = ["[캐릭터 지식]"]

        # 세계관
        if self._world:
            parts.append(f"--- 세계관: {self._world.get('name', '')} ---")
            if era := self._world.get("era"):
                parts.append(f"시대: {era}")
            if desc := self._world.get("description"):
                parts.append(desc)
            if rules := self._world.get("rules"):
                parts.append("규칙:")
                for r in rules:
                    parts.append(f"  - {r}")
            parts.append("")

        # 캐릭터
        if self._characters:
            parts.append("--- 등장인물 ---")
            for char in self._characters:
                name = char.get("name", "")
                identity = char.get("identity", "")
                parts.append(f"- {name}: {identity}")
            parts.append("")

        # 관계
        if self._relationships:
            parts.append("--- 관계 ---")
            for r in self._relationships:
                parts.append(
                    f"- {r.get('from', '?')} → {r.get('to', '?')}: "
                    f"{r.get('type', '?')}, {r.get('sentiment', '?')}"
                )
            parts.append("")

        # 타임라인
        if self._timeline:
            parts.append("--- 타임라인 ---")
            for event in self._timeline:
                parts.append(f"- [{event.get('time', '?')}] {event.get('event', '')}")
            parts.append("")

        # 장소
        if self._locations:
            parts.append("--- 장소 ---")
            for loc in self._locations:
                parts.append(f"- {loc.get('name', '?')}: {loc.get('description', '')}")
            parts.append("")

        # 자유 형식
        for filename, content in self._freeform:
            parts.append(f"--- {filename} ---")
            parts.append(content)
            parts.append("")

        if len(parts) == 1:
            return "[캐릭터 지식]\n지식 없음"
        return "\n".join(parts)

    def search_relevant(self, query: str, token_budget: int = 500) -> str:
        """쿼리와 관련된 지식만 선택하여 프롬프트 문자열로 반환한다.

        키워드 기반 매칭으로 관련성을 판단한다.
        """
        if not self._world and not self._characters and not self._freeform:
            return ""

        query_lower = query.lower()
        query_words = set(query_lower.split())

        scored_sections: list[tuple[float, str]] = []

        # 세계관 — 키워드 매칭
        if self._world:
            score = self._keyword_score(query_words, self._world.get("description", ""))
            if score > 0 or self._always_include_world():
                section = self._format_world()
                scored_sections.append((score + 0.1, section))  # 기본 점수 부여

        # 캐릭터 — 이름 직접 매칭
        for char in self._characters:
            name = char.get("name", "")
            if name.lower() in query_lower:
                section = self._format_character(char)
                scored_sections.append((1.0, section))

        # 관계 — 캐릭터 이름 매칭
        for rel in self._relationships:
            if (
                rel.get("from", "").lower() in query_lower
                or rel.get("to", "").lower() in query_lower
            ):
                section = (
                    f"{rel.get('from', '?')} → {rel.get('to', '?')}: "
                    f"{rel.get('type', '?')}, {rel.get('sentiment', '?')}"
                )
                scored_sections.append((0.8, section))

        # 타임라인 — 키워드 매칭
        for event in self._timeline:
            score = self._keyword_score(query_words, event.get("event", ""))
            if score > 0:
                section = f"[{event.get('time', '?')}] {event.get('event', '')}"
                scored_sections.append((score, section))

        # 자유 형식 — 키워드 매칭
        for filename, content in self._freeform:
            score = self._keyword_score(query_words, content)
            if score > 0:
                scored_sections.append((score, f"--- {filename} ---\n{content}"))

        if not scored_sections:
            return ""

        # 점수순 정렬
        scored_sections.sort(key=lambda x: x[0], reverse=True)

        # 예산 내에서 선택
        parts = ["[관련 지식]"]
        used_tokens = self._estimate_tokens(parts[0])

        for _score, section in scored_sections:
            section_tokens = self._estimate_tokens(section)
            if used_tokens + section_tokens > token_budget:
                # 잘라서 포함 시도
                remaining_chars = int((token_budget - used_tokens) / 1.5)
                if remaining_chars > 50:
                    truncated = section[:remaining_chars] + "..."
                    parts.append(truncated)
                    used_tokens += self._estimate_tokens(truncated)
                break
            parts.append(section)
            used_tokens += section_tokens

        if len(parts) == 1:
            return ""
        return "\n\n".join(parts)

    def _keyword_score(self, query_words: set[str], text: str) -> float:
        """쿼리 단어와 텍스트의 매칭 점수를 반환한다.

        부분 문자열 매칭을 사용하여 한국어 조사 문제를 회피한다.
        """
        if not text:
            return 0.0
        text_lower = text.lower()
        # 각 쿼리 단어가 텍스트에 포함되는지 확인 (부분 매칭)
        matches = sum(1 for w in query_words if w in text_lower)
        # 쿼리 단어가 텍스트에 없으면, 텍스트의 단어가 쿼리에 포함되는지 역매칭
        if matches == 0:
            text_words = set(text_lower.split())
            matches = sum(1 for tw in text_words if any(tw in qw for qw in query_words))
        return matches / max(len(query_words), 1)

    def _always_include_world(self) -> bool:
        """세계관을 항상 포함할지 여부."""
        return self._world is not None

    def _format_world(self) -> str:
        """세계관을 프롬프트 문자열로 변환."""
        if not self._world:
            return ""
        parts = [f"[세계관: {self._world.get('name', '')}]"]
        if era := self._world.get("era"):
            parts.append(f"시대: {era}")
        if desc := self._world.get("description"):
            parts.append(desc)
        if rules := self._world.get("rules"):
            parts.append("규칙:")
            for r in rules:
                parts.append(f"  - {r}")
        return "\n".join(parts)

    def _format_character(self, char: dict) -> str:
        """캐릭터를 프롬프트 문자열로 변환."""
        name = char.get("name", "")
        parts = [f"[캐릭터: {name}]"]
        for key, label in [
            ("identity", "정체"),
            ("personality", "성격"),
            ("relationship_to_player", "관계"),
            ("status", "상태"),
        ]:
            if val := char.get(key):
                parts.append(f"{label}: {val}")
        return "\n".join(parts)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """토큰 수 추정 (한글 1자 ≈ 1.5 tokens, 영어 1단어 ≈ 1 token)."""
        korean_chars = sum(1 for c in text if "\uac00" <= c <= "\ud7a3")
        other_chars = len(text) - korean_chars
        return int(korean_chars * 1.5 + other_chars * 0.3)
