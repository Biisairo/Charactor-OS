"""프롬프트 엔진 — 모든 모듈의 데이터를 동적으로 조립한다.

토큰 예산 기반으로 Always/Context 섹션을 분리하고,
관련성 높은 컨텍스트만 선택하여 시스템 프롬프트를 생성한다.
"""

MAX_PROMPT_TOKENS = 3000


class PromptEngine:
    """시스템 프롬프트를 동적으로 조립하는 엔진."""

    # 섹션별 토큰 예산 비율 (Context 섹션 내)
    BUDGET_RATIOS = {
        "fewshot": 0.25,
        "knowledge": 0.35,
        "relationships": 0.10,
        "memory": 0.20,
        "history": 0.10,
    }

    def __init__(self, max_tokens: int = MAX_PROMPT_TOKENS):
        self._max_tokens = max_tokens

    def assemble_system_prompt(
        self,
        user_input: str,
        persona,
        emotion,
        memory,
        knowledge,
        history,
        fewshot,
    ) -> str:
        """모든 모듈의 데이터를 조립하여 시스템 프롬프트를 생성한다.

        Args:
            user_input: 사용자 입력
            persona: PersonaModule 인스턴스
            emotion: EmotionModule 인스턴스
            memory: MemoryModule 인스턴스
            knowledge: KnowledgeModule 인스턴스
            history: HistoryModule 인스턴스
            fewshot: FewShotModule 인스턴스
        """
        sections: list[str] = []

        # ── Always 섹션 ──
        persona_section = persona.to_system_prompt()
        emotion_section = emotion.to_prompt()
        response_guide = self._build_response_guide()

        always_sections = [
            persona_section,
            emotion_section,
        ]
        always_text = "\n\n".join(s for s in always_sections if s)
        always_tokens = self._estimate_tokens(always_text)
        response_tokens = self._estimate_tokens(response_guide)

        # History는 Always이지만 길이 가변
        history_section = history.to_prompt(n=10)
        history_tokens = self._estimate_tokens(history_section)

        # Always 총 비용
        always_total = always_tokens + response_tokens + history_tokens

        # ── Context 예산 계산 ──
        remaining = max(0, self._max_tokens - always_total)

        budgets = {}
        for key, ratio in self.BUDGET_RATIOS.items():
            budgets[key] = int(remaining * ratio)

        # ── Context 섹션 ──
        emotions = emotion.get_state()

        # 행동 지침 + 내면 상태 — 관련 있을 때만 포함
        behavior_section = persona.get_behavior_section()
        inner_world_section = persona.get_inner_world()

        fewshot_section = fewshot.to_prompt(
            query=user_input,
            emotions=emotions,
            top_k=3,
            token_budget=budgets.get("fewshot", 300),
        )

        knowledge_section = knowledge.search_relevant(
            query=user_input,
            token_budget=budgets.get("knowledge", 400),
        )

        # 관계 정보 — 쿼리에 캐릭터 이름이 언급된 경우에만 포함
        relationships_section = self._build_relationships(
            user_input,
            persona,
            knowledge,
            token_budget=budgets.get("relationships", 100),
        )

        memory_section = memory.to_prompt(
            query=user_input,
            top_k=5,
            token_budget=budgets.get("memory", 200),
        )

        # ── 조립 ──
        # Always 섹션
        for s in always_sections:
            if s:
                sections.append(s)

        # Context 섹션 (내용이 있는 것만)
        for s in [
            behavior_section,
            inner_world_section,
            fewshot_section,
            knowledge_section,
            relationships_section,
            memory_section,
        ]:
            if s:
                sections.append(s)

        # History + Response Guide
        if history_section:
            sections.append(history_section)
        sections.append(response_guide)

        return "\n\n".join(sections)

    def _build_relationships(self, query: str, persona, knowledge, token_budget: int) -> str:
        """Relationships 섹션 생성. 쿼리에 관련 인물이 포함된 경우만."""
        query_lower = query.lower()

        # Persona의 관계
        persona_rels = persona.get_relationships()

        # Knowledge의 관계
        knowledge_rels = knowledge.get_relationships()

        all_rels = persona_rels + knowledge_rels

        if not all_rels:
            return ""

        # 쿼리에 관련 인물이 언급된 관계만 필터
        relevant = []
        for rel in all_rels:
            target = rel.get("target", "") or rel.get("to", "")
            if target.lower() in query_lower:
                relevant.append(rel)

        if not relevant:
            return ""

        parts = ["[관계 정보]"]
        used_tokens = self._estimate_tokens(parts[0])

        for rel in relevant:
            target = rel.get("target", "") or rel.get("to", "")
            rel_type = rel.get("type", "")
            desc = rel.get("description", "")
            sentiment = rel.get("sentiment", "")

            line = f"- {target}: {rel_type}"
            if sentiment:
                line += f" ({sentiment})"
            if desc:
                line += f" — {desc}"

            line_tokens = self._estimate_tokens(line)
            if used_tokens + line_tokens > token_budget:
                break

            parts.append(line)
            used_tokens += line_tokens

        if len(parts) == 1:
            return ""
        return "\n".join(parts)

    def _build_response_guide(self) -> str:
        """Response Guide 섹션 생성."""
        return (
            "[응답 규칙]\n"
            "- 위 정보를 바탕으로 자연스럽게 대화하세요.\n"
            "- 캐릭터의 성격과 말투를 유지하세요.\n"
            "- 감정 상태에 맞게 톤을 조절하세요.\n"
            "- 기억된 정보를 활용하여 개인화된 응답을 하세요.\n"
            "- 진짜 사람과의 대화 같아야 합니다. 과한 연기는 금지."
        )

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """토큰 수 추정 (한글 1자 ≈ 1.5 tokens)."""
        if not text:
            return 0
        korean_chars = sum(1 for c in text if "\uac00" <= c <= "\ud7a3")
        other_chars = len(text) - korean_chars
        return int(korean_chars * 1.5 + other_chars * 0.3)
