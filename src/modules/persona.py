from pathlib import Path

import yaml


class PersonaModule:
    """캐릭터의 정체성을 정의하는 정적 데이터를 로드하고 시스템 프롬프트로 변환한다."""

    def __init__(self, persona_path: str):
        self._path = Path(persona_path)
        self._data: dict = {}

    def load(self) -> dict:
        """YAML을 파싱하여 딕셔너리로 반환한다."""
        if not self._path.exists():
            raise FileNotFoundError(f"페르소나 파일을 찾을 수 없습니다: {self._path}")

        with open(self._path, encoding="utf-8") as f:
            self._data = yaml.safe_load(f)

        if not self._data or "name" not in self._data:
            raise ValueError("페르소나 파일에 'name' 필드가 필요합니다.")

        return self._data

    # ─── 프롬프트 변환 ───

    def to_system_prompt(self) -> str:
        """페르소나 데이터를 시스템 프롬프트 문자열로 변환한다."""
        if not self._data:
            self.load()

        d = self._data
        parts = []

        # Identity
        identity = d.get("identity", "")
        if identity:
            parts.append(f'당신은 "{d["name"]}"입니다. {identity}\n')
        else:
            parts.append(f'당신은 "{d["name"]}"입니다.\n')

        # 기본 정보
        meta = []
        if age := d.get("age"):
            meta.append(f"나이: {age}")
        if gender := d.get("gender"):
            meta.append(f"성별: {gender}")
        if occupation := d.get("occupation"):
            meta.append(f"직업: {occupation}")
        if meta:
            parts.append(", ".join(meta))
            parts.append("")

        # 성격
        personality = d.get("personality", {})
        traits = personality.get("traits", [])
        if traits:
            parts.append(f"[성격]\n{', '.join(traits)}\n")

        # 말투
        speaking = d.get("speaking_style", {})
        if speaking:
            if summary := speaking.get("summary"):
                parts.append(f"[말투]\n{summary}")
            if fillers := speaking.get("fillers"):
                parts.append(f"- 말버릇: {', '.join(fillers)}")
            if endings := speaking.get("endings"):
                parts.append(f"- 문미: {', '.join(endings)}")
            parts.append("")

        # 가치관
        if values := d.get("values"):
            parts.append(f"[가치관]\n{', '.join(values)}\n")

        # 배경
        if backstory := d.get("backstory"):
            parts.append(f"[배경]\n{backstory}\n")

        # 좋아하는 것 / 싫어하는 것
        for label, key in [("좋아하는 것", "likes"), ("싫어하는 것", "dislikes")]:
            if items := d.get(key):
                parts.append(f"[{label}]\n{', '.join(items)}\n")

        return "\n".join(parts)

    def get_behavior_section(self) -> str:
        """행동 지침 섹션만 별도 문자열로 반환한다."""
        if not self._data:
            self.load()

        behavior = self._data.get("behavior", {})
        situations = behavior.get("situations", [])
        topics = behavior.get("topics", [])
        rules = behavior.get("rules", [])

        if not situations and not topics and not rules:
            return ""

        parts = ["[행동 지침]"]

        if situations:
            parts.append("상황별:")
            for s in situations:
                parts.append(f"- {s.get('trigger', '')} → {s.get('action', '')}")

        if topics:
            parts.append("주제별:")
            for t in topics:
                parts.append(f"- {t.get('name', '')}: {t.get('stance', '')}")

        if rules:
            parts.append("절대 규칙:")
            for r in rules:
                parts.append(f"- {r}")

        return "\n".join(parts)

    def get_inner_world(self) -> str:
        """내면 상태를 프롬프트 문자열로 변환한다."""
        if not self._data:
            self.load()

        iw = self._data.get("inner_world", {})
        if not iw:
            return ""

        parts = ["[내면 상태]"]
        for key, label in [
            ("current_thought", "현재 생각"),
            ("hidden_feelings", "숨기는 감정"),
            ("wants_to_say", "하고 싶은 말"),
        ]:
            if val := iw.get(key):
                parts.append(f"{label}: {val}")

        if len(parts) == 1:
            return ""
        return "\n".join(parts)

    def get_emotion_triggers(self) -> list[dict]:
        """감정 트리거 목록을 반환한다. EmotionModule에서 참조."""
        if not self._data:
            self.load()
        return self._data.get("emotion_triggers", [])

    def get_examples(self) -> list[dict]:
        """내장 few-shot 예시를 반환한다."""
        if not self._data:
            self.load()
        return self._data.get("examples", [])

    def get_relationships(self) -> list[dict]:
        """관계 설정을 반환한다."""
        if not self._data:
            self.load()
        return self._data.get("relationships", [])
