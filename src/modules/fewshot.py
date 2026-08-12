from dataclasses import dataclass, field
from pathlib import Path

import yaml

from src.modules.asset_issue import AssetLoadIssue


@dataclass
class FewShotExample:
    """Few-shot 예시 하나."""

    user: str
    character: str
    emotion_state: list[str] = field(default_factory=list)


@dataclass
class FewShotGroup:
    """태그별 few-shot 예시 그룹."""

    tag: str
    examples: list[FewShotExample] = field(default_factory=list)


# 태그 → 트리거 키워드 매핑
TAG_KEYWORDS: dict[str, list[str]] = {
    "인사": ["안녕", "하이", "헬로", "반가워", "hello", "hi"],
    "위로": ["힘들어", "슬퍼", "우울", "망했어", "실패", "속상", "아파"],
    "갈등": ["싸웠어", "화나", "짜증", "미워", "싫어"],
    "유머": ["웃겨", "농담", "ㅋㅋ", "재밌어", "웃긴"],
    "일상": ["뭐해", "밥", "오늘", "취미", "날씨"],
    "질문": ["뭐야", "왜", "어떻게", "뭐라고", "설명"],
    "칭찬": ["고마워", "최고", "잘했어", "대단", "좋아"],
}


class FewShotModule:
    """캐릭터의 응답 패턴을 예시 대화로 학습시키는 모듈.

    examples/ 디렉토리의 YAML 파일을 로드하고,
    사용자 입력과 관련성 높은 예시를 동적으로 선택한다.
    """

    def __init__(self, examples_dir: str, embedding_fn=None):
        """예시 디렉토리와 임베딩 함수를 설정한다.

        Args:
            examples_dir: 예시 YAML 파일이 있는 디렉토리 경로
            embedding_fn: 임베딩 함수 (선택, 없으면 키워드 매칭만 사용)
        """
        self._dir = Path(examples_dir)
        self._embedding_fn = embedding_fn
        self._groups: list[FewShotGroup] = []
        self._load_issues: list[AssetLoadIssue] = []
        self._embedding_failed = False

    @property
    def load_issues(self) -> list[AssetLoadIssue]:
        """마지막 `load_all()`에서 생긴 문제들. 오케스트레이터가 읽어 로그로 올린다."""
        return list(self._load_issues)

    def load_all(self) -> None:
        """examples/ 디렉토리의 모든 YAML을 로드한다."""
        self._groups = []
        self._load_issues = []

        if not self._dir.exists():
            return

        for file_path in sorted(self._dir.iterdir()):
            if file_path.is_file() and file_path.suffix.lower() in {".yaml", ".yml"}:
                self._load_file(file_path)

    def _load_file(self, path: Path) -> None:
        """단일 YAML 파일을 로드하여 FewShotGroup으로 변환한다."""
        try:
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f)

            if not isinstance(data, dict):
                return

            tag = data.get("tag", path.stem)
            examples = []

            for ex in data.get("examples", []):
                if isinstance(ex, dict) and "user" in ex and "character" in ex:
                    examples.append(
                        FewShotExample(
                            user=ex["user"],
                            character=ex["character"],
                            emotion_state=ex.get("emotion_state", []),
                        )
                    )

            if examples:
                self._groups.append(FewShotGroup(tag=tag, examples=examples))

        except Exception as e:
            # 조용히 넘기면 예시가 0개가 된 이유를 아무도 알 수 없다 (REQ-06-1).
            # 다른 파일의 로드는 막지 않는다 (REQ-06-2).
            self._load_issues.append(
                AssetLoadIssue(
                    filename=path.name,
                    reason=f"{type(e).__name__}: {e}",
                    expected=False,
                )
            )

    def search(
        self,
        query: str,
        emotions: dict[str, float] | None = None,
        top_k: int = 3,
    ) -> list[FewShotExample]:
        """관련성 기반으로 few-shot 예시를 검색한다.

        점수 = 태그 키워드 매칭(0.4) + 임베딩 유사도(0.4) + 감정 매칭(0.2)
        """
        if not self._groups:
            return []

        emotions = emotions or {}
        query_lower = query.lower()

        # 태그 키워드 매칭 점수
        tag_scores: dict[str, float] = {}
        for tag, keywords in TAG_KEYWORDS.items():
            matches = sum(1 for kw in keywords if kw in query_lower)
            tag_scores[tag] = matches / max(len(keywords), 1)

        # 각 example에 점수 부여
        scored: list[tuple[float, FewShotExample]] = []

        for group in self._groups:
            # 태그 매칭 점수
            tag_score = tag_scores.get(group.tag, 0.0)

            # 태그 이름이 쿼리에 직접 포함되는 경우
            if group.tag.lower() in query_lower:
                tag_score = max(tag_score, 0.8)

            for example in group.examples:
                # 감정 매칭 점수
                emotion_score = 0.0
                if emotions and example.emotion_state:
                    matching = sum(1 for e in example.emotion_state if e in emotions)
                    emotion_score = matching / max(len(example.emotion_state), 1)

                # 임베딩 유사도 (있으면)
                embedding_score = 0.0
                if self._embedding_fn:
                    try:
                        import numpy as np

                        query_vec = self._embedding_fn(query)
                        example_vec = self._embedding_fn(example.user)
                        embedding_score = float(np.dot(query_vec, example_vec))
                        embedding_score = max(0.0, embedding_score)  # 음수 제거
                    except Exception as e:
                        # 임베딩이 죽으면 검색이 태그 매칭으로 조용히 퇴화한다.
                        # 검색마다 기록하면 로그가 넘치므로 처음 1회만 남긴다.
                        if not self._embedding_failed:
                            self._embedding_failed = True
                            self._load_issues.append(
                                AssetLoadIssue(
                                    filename="(임베딩)",
                                    reason=f"{type(e).__name__}: {e} — 태그 매칭으로 퇴화",
                                    expected=False,
                                )
                            )

                # 최종 점수
                if self._embedding_fn:
                    total = tag_score * 0.4 + embedding_score * 0.4 + emotion_score * 0.2
                else:
                    # 임베딩 없으면 태그에 더 높은 가중치
                    total = tag_score * 0.7 + emotion_score * 0.3

                scored.append((total, example))

        # 점수순 정렬 → 상위 top_k
        scored.sort(key=lambda x: x[0], reverse=True)

        # 점수가 0보다 큰 것만 반환
        results = []
        for score, example in scored[: top_k * 2]:  # 여유 있게 가져와서 중복 제거
            if score <= 0:
                break
            # 중복 제거 (같은 user+character)
            if not any(
                e.user == example.user and e.character == example.character for e in results
            ):
                results.append(example)
            if len(results) >= top_k:
                break

        return results

    def to_prompt(
        self,
        query: str,
        emotions: dict[str, float] | None = None,
        top_k: int = 3,
        token_budget: int = 300,
    ) -> str:
        """검색된 예시를 프롬프트 문자열로 변환한다."""
        examples = self.search(query, emotions, top_k)

        if not examples:
            return ""

        parts = ["[예시 대화]"]
        used_tokens = self._estimate_tokens(parts[0])

        for ex in examples:
            block = f"사용자: {ex.user}\n캐릭터: {ex.character}"
            block_tokens = self._estimate_tokens(block)

            if used_tokens + block_tokens > token_budget:
                break

            parts.append(block)
            used_tokens += block_tokens

        if len(parts) == 1:
            return ""
        return "\n\n".join(parts)

    def add_example(
        self,
        tag: str,
        user: str,
        character: str,
        emotion_state: list[str] | None = None,
    ) -> None:
        """새 예시를 추가한다 (런타임)."""
        new_example = FewShotExample(
            user=user,
            character=character,
            emotion_state=emotion_state or [],
        )

        # 기존 그룹에 추가 또는 새 그룹 생성
        for group in self._groups:
            if group.tag == tag:
                group.examples.append(new_example)
                return

        self._groups.append(
            FewShotGroup(
                tag=tag,
                examples=[new_example],
            )
        )

    def get_all_tags(self) -> list[str]:
        """모든 태그 목록을 반환한다."""
        return [g.tag for g in self._groups]

    def get_all_groups(self) -> list[FewShotGroup]:
        """모든 그룹을 반환한다."""
        return self._groups

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """토큰 수 추정."""
        korean_chars = sum(1 for c in text if "\uac00" <= c <= "\ud7a3")
        other_chars = len(text) - korean_chars
        return int(korean_chars * 1.5 + other_chars * 0.3)
