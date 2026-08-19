import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from src.modules.asset_issue import AssetLoadIssue

# 목차에 실을 문서당 소제목 수. 목차가 본문만큼 길어지면 목차의 의미가 없다.
MAX_INDEX_HEADINGS = 8

# 배경지식(base/)과 일반지식(general/)을 가르는 디렉토리 이름 (TASK-20).
BASE_DIRNAME = "base"
GENERAL_DIRNAME = "general"

# 청크 상한. 넘으면 문단 단위로 한 번 더 나눈다.
MAX_CHUNK_CHARS = 600

# 배경지식 총량 경고선. 넘어도 자르지 않는다 — 사람이 넣은 배경을 임의로
# 자르면 캐릭터가 무너진다. 알리기만 한다 (REQ-20-8).
BASE_WARN_TOKENS = 1200

# 검색 점수 하한. 이보다 낮으면 관련 없다고 본다.
MIN_SEARCH_SCORE = 0.05

# 임베딩 유사도를 점수에 반영하는 하한.
#
# 이 프로젝트의 임베딩 모델(`all-MiniLM-L6-v2`)은 영어 전용이라 한국어 질의를
# 사실상 구분하지 못한다. 실측에서 "인삿말"과 "양자역학"의 유사도 분포가
# 완전히 같았다(최고 0.264 / 대상 0.159). 그 구간의 점수를 그대로 더하면
# 무관한 자료가 프롬프트에 실린다.
#
# 임계 위(실측 "방송 시작 인사" 0.729)만 신호로 인정한다. 이 값은 모델 특성에
# 매인 임시방편이며, 한국어 지원 모델로 바꾸면 함께 재검토해야 한다.
MIN_EMBEDDING_SIMILARITY = 0.45

# 임베딩 가중치. 키워드가 주 신호이고 임베딩은 보조다.
#
# 같은 배점으로 더하면 순위가 뒤집힌다. 실측: "방송 시작 인사"에 대해 키워드가
# 3/3 적중한 조각(1.0)이, 1/3만 맞고 임베딩 0.729를 더한 조각(1.06)에 밀렸다.
# 한국어에서 신뢰도가 낮은 신호가 확실한 신호를 이기면 안 된다.
EMBEDDING_WEIGHT = 0.3

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_WORD = re.compile(r"[0-9a-z가-힣]+")


@dataclass(frozen=True)
class KnowledgeChunk:
    """검색 단위 하나. 제목 경로를 달고 다닌다 (REQ-20-6).

    맥락 없이 잘린 문단은 검색되어도 무슨 이야기인지 알 수 없다.
    """

    source: str
    heading_path: str
    text: str
    embedding: object = field(default=None, compare=False)

    def render(self) -> str:
        header = f"--- {self.source}"
        if self.heading_path:
            header += f" > {self.heading_path}"
        return f"{header} ---\n{self.text}"


class KnowledgeModule:
    """캐릭터가 속한 세계관, 관계, 타임라인, 장소 등 구조화된 지식을 관리한다.

    YAML 파일은 type 필드에 따라 구조화되고,
    .md/.json/.txt 등은 자유 형식(freeform)으로 처리된다.
    """

    # 마크다운만 읽는다 (TASK-22). 형식이 하나여야 저작이 단순하고,
    # 저작자가 제목으로 검색 단위를 직접 통제할 수 있다.
    SUPPORTED_EXTENSIONS = {".md"}

    def __init__(self, knowledge_dir: str, embedding_fn=None):
        """
        Args:
            embedding_fn: 임베딩 함수. 없으면 키워드 매칭만으로 검색한다
                (FewShot과 같은 정책). 있으면 두 점수를 합산한다 — 임베딩 모델이
                영어 중심이라 한국어 고유명사는 키워드가 더 정확히 잡는다.
        """
        self._dir = Path(knowledge_dir)
        self._embedding_fn = embedding_fn
        # front matter에서 읽는 시스템 메타. 지금은 era 하나뿐이다 (TASK-22).
        self._meta: dict = {}
        # 배경지식은 원문 그대로 주입되고, 일반지식은 청킹되어 검색된다 (TASK-20).
        self._base_docs: list[tuple[str, str]] = []
        self._general_docs: list[tuple[str, str]] = []
        self._chunks: list[KnowledgeChunk] = []
        self._load_issues: list[AssetLoadIssue] = []
        self._embedding_failed = False

    @property
    def load_issues(self) -> list[AssetLoadIssue]:
        """마지막 `load_all()`에서 생긴 문제들.

        구조화 파싱 실패는 freeform 폴백이라는 **의도된 동작**이므로
        `expected=True`로 표시된다. 예기치 않은 실패와 섞이면 로그를 봐도
        무엇이 문제인지 알 수 없다 (REQ-06-3).
        """
        return list(self._load_issues)

    def load_all(self) -> None:
        """지식 디렉토리의 모든 파일을 로드하여 구조화/비구조화 데이터를 저장한다."""
        self._meta = {}
        self._base_docs = []
        self._general_docs = []
        self._chunks = []
        self._load_issues = []
        self._embedding_failed = False

        if not self._dir.exists():
            return

        self._scan_directory(self._dir)
        self._build_chunks()
        self._warn_if_base_is_large()

    def _is_base(self, path: Path) -> bool:
        """`base/` 아래 있는가. 루트 직속 파일은 general로 본다 (REQ-20-2)."""
        try:
            relative = path.relative_to(self._dir)
        except ValueError:
            return False
        return BASE_DIRNAME in relative.parts[:-1]

    def _scan_directory(self, directory: Path) -> None:
        """디렉토리를 재귀 스캔하여 파일을 분류한다."""
        for item in sorted(directory.iterdir()):
            if item.is_dir():
                # 하위 디렉토리 재귀 스캔 (base/, general/, characters/ 등)
                self._scan_directory(item)
                continue

            if not item.is_file() or item.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                continue

            raw = item.read_text(encoding="utf-8")
            meta, content = self._split_front_matter(raw, item.name)
            for key, value in meta.items():
                # 먼저 발견된 값이 이긴다. 파일명 순으로 훑으므로 결과가 결정적이다.
                self._meta.setdefault(key, value)

            target = self._base_docs if self._is_base(item) else self._general_docs
            target.append((item.name, content))

    def _split_front_matter(self, raw: str, filename: str) -> tuple[dict, str]:
        """앞머리 메타와 본문을 가른다.

        메타는 본문이 아니다. 프롬프트에도 검색 색인에도 들어가면 안 된다.
        """
        if not raw.startswith("---"):
            return {}, raw

        parts = raw.split("---", 2)
        if len(parts) < 3:
            return {}, raw

        _empty, header, body = parts
        try:
            meta = yaml.safe_load(header)
        except Exception as e:
            # 메타가 깨져도 본문은 살린다. 다만 조용히 넘기지 않는다 (REQ-06-1).
            self._load_issues.append(
                AssetLoadIssue(
                    filename=filename,
                    reason=f"{type(e).__name__}: {e} — 앞머리 메타를 읽지 못했다",
                    expected=False,
                )
            )
            return {}, body.lstrip("\n")

        if not isinstance(meta, dict):
            return {}, body.lstrip("\n")

        return meta, body.lstrip("\n")

    # ─── 메타 ───

    def era(self) -> str:
        """캐릭터가 사는 시대. front matter의 `era`이며, 없으면 빈 문자열.

        Reflection의 시대 정합성 기준과 평가 판정자 프로필이 이 값을 쓴다.
        비어 있으면 그 기준은 검사되지 않는다 (TASK-19 REQ-19-2).
        """
        return str(self._meta.get("era") or "").strip()

    # ─── 청킹 (REQ-20-5 · 20-6) ───

    def chunks(self) -> list[KnowledgeChunk]:
        """일반지식의 검색 단위 목록."""
        return list(self._chunks)

    def _build_chunks(self) -> None:
        for filename, content in self._general_docs:
            for heading_path, text in self._split(content):
                self._chunks.append(
                    KnowledgeChunk(
                        source=filename,
                        heading_path=heading_path,
                        text=text,
                        embedding=self._embed(f"{heading_path}\n{text}"),
                    )
                )

    @staticmethod
    def _split(content: str) -> list[tuple[str, str]]:
        """제목으로 나누고, 조각이 크면 문단으로 한 번 더 나눈다.

        제목 경로를 유지하는 것이 핵심이다. "얘들아라고 부른다" 한 줄만 떼어놓으면
        그것이 시청자 호칭 이야기인지 알 수 없다.
        """
        sections: list[tuple[str, str]] = []
        stack: list[tuple[int, str]] = []
        current: list[str] = []

        def flush() -> None:
            body = "\n".join(current).strip()
            current.clear()
            if body:
                sections.append((" > ".join(title for _, title in stack), body))

        for line in content.split("\n"):
            match = _HEADING.match(line.strip())
            if not match:
                current.append(line)
                continue
            flush()
            level = len(match.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, match.group(2).strip()))
        flush()

        # 제목이 하나도 없으면 문서 전체가 조각 하나다.
        if not sections:
            body = content.strip()
            return [("", body)] if body else []

        chunks: list[tuple[str, str]] = []
        for heading_path, body in sections:
            if len(body) <= MAX_CHUNK_CHARS:
                chunks.append((heading_path, body))
                continue
            for paragraph in re.split(r"\n\s*\n", body):
                text = paragraph.strip()
                if text:
                    chunks.append((heading_path, text))
        return chunks

    def _embed(self, text: str):
        """임베딩. 실패는 한 번만 기록하고 키워드 검색으로 계속한다."""
        if self._embedding_fn is None:
            return None
        try:
            return self._embedding_fn(text)
        except Exception as e:
            if not self._embedding_failed:
                self._embedding_failed = True
                self._load_issues.append(
                    AssetLoadIssue(
                        filename="(임베딩)",
                        reason=f"{type(e).__name__}: {e} — 키워드 매칭으로 퇴화",
                        expected=False,
                    )
                )
            return None

    # ─── 배경지식 (REQ-20-1 · 20-4 · 20-8) ───

    def base_text(self) -> str:
        """배경지식 원문. 캐릭터가 항상 아는 것이므로 발화에도 실린다."""
        body = "\n\n".join(
            content.strip() for _name, content in self._base_docs if content.strip()
        )
        return f"[배경 지식]\n{body}" if body else ""

    def to_base_prompt(self) -> str:
        """배경지식 원문 + 일반지식 목차. 뇌가 받는 형태다.

        목차는 "무엇을 더 찾아볼 수 있는가"이므로 검색을 결정하는 뇌에만 필요하다.
        발화 단계에는 원문만 간다 (`base_text`).
        """
        return "\n\n".join(part for part in [self.base_text(), self.to_index()] if part)

    def _warn_if_base_is_large(self) -> None:
        total = sum(self._estimate_tokens(c) for _n, c in self._base_docs)
        if total <= BASE_WARN_TOKENS:
            return
        self._load_issues.append(
            AssetLoadIssue(
                filename=f"{BASE_DIRNAME}/",
                reason=(
                    f"배경 지식이 약 {total} 토큰으로 권장치({BASE_WARN_TOKENS})를 넘는다 "
                    f"— 매 호출 재전송된다. 일부를 {GENERAL_DIRNAME}/로 옮기는 것을 검토하라"
                ),
                expected=False,
            )
        )

    def to_index(self) -> str:
        """**찾아볼 수 있는 것**의 목록. 내용은 담지 않는다 (REQ-20-4).

        배경지식은 이미 원문으로 실려 있으므로 목차에 넣지 않는다. 여기 실리는
        것은 일반지식뿐이고, 뇌는 이것을 보고 검색할지 정한다. 파일명만 노출하면
        그 문서에 무엇이 있는지 알 수 없어 검색을 시도조차 하지 않는다.
        """
        lines = []
        for filename, content in self._general_docs:
            summary = self._summarize(filename, content)
            lines.append(f"- {filename}: {summary}" if summary else f"- 문서: {filename}")

        if not lines:
            return ""

        return (
            "[찾아볼 수 있는 것]\n"
            + "\n".join(lines)
            + "\n(자세한 내용은 search_knowledge로 확인한다)"
        )

    def _summarize(self, filename: str, content: str) -> str:
        """문서 한 줄 요약 — 소제목 나열."""
        return ", ".join(self._headings(content))

    @staticmethod
    def _headings(content: str, limit: int = MAX_INDEX_HEADINGS) -> list[str]:
        """마크다운 소제목만 뽑는다. 본문은 담지 않는다.

        파일명만 노출하면 뇌는 그 문서에 무엇이 있는지 알 수 없어 검색을
        시도조차 하지 않는다 — "쏘하"가 인삿말인 줄 모르는 것이 그 결과였다.
        """
        found = []
        for line in content.split("\n"):
            stripped = line.strip()
            if stripped.startswith("##"):
                title = stripped.lstrip("#").strip()
                if title:
                    found.append(title)
            if len(found) >= limit:
                break
        return found

    def search_relevant(self, query: str, token_budget: int = 500) -> str:
        """일반지식에서 관련 조각을 찾아 프롬프트 문자열로 반환한다.

        키워드와 임베딩을 함께 쓴다 (REQ-20-7). 임베딩 모델이 영어 중심이라
        "쏘하" 같은 캐릭터 고유 어휘는 키워드가 더 정확히 잡고, 표현이 다른
        질의는 임베딩이 잡는다. 하나만으로는 실사용에서 관측된 실패를 못 덮는다.
        """
        if not self._chunks or not query.strip():
            return ""

        query_words = set(_WORD.findall(query.lower()))
        query_vec = self._embed(query)

        scored: list[tuple[float, KnowledgeChunk]] = []
        for chunk in self._chunks:
            score = self._keyword_score(query_words, f"{chunk.heading_path}\n{chunk.text}")
            if query_vec is not None and chunk.embedding is not None:
                similarity = self._cosine(query_vec, chunk.embedding)
                if similarity >= MIN_EMBEDDING_SIMILARITY:
                    score += similarity * EMBEDDING_WEIGHT
            if score >= MIN_SEARCH_SCORE:
                scored.append((score, chunk))

        if not scored:
            return ""

        scored.sort(key=lambda pair: -pair[0])

        parts: list[str] = []
        used = 0
        for _score, chunk in scored:
            block = chunk.render()
            cost = self._estimate_tokens(block)
            if parts and used + cost > token_budget:
                break
            parts.append(block)
            used += cost

        return "\n\n".join(parts)

    @staticmethod
    def _cosine(a, b) -> float:
        import numpy as np

        denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denominator == 0:
            return 0.0
        return max(0.0, float(np.dot(a, b)) / denominator)

    def _keyword_score(self, query_words: set[str], text: str) -> float:
        """쿼리 단어와 텍스트의 매칭 점수를 반환한다.

        양방향으로 부분 매칭한다. 한국어는 조사가 붙고("인사는"), 사용자는 말을
        늘여 쓴다("쏘하쏘하하"). 어느 쪽이든 걸리게 해야 한다.

        비교 전에 구두점을 걷어내는 것이 중요하다 — `"쏘하",` 를 그대로 비교하면
        `쏘하쏘하하` 와 매칭되지 않는다. 실사용에서 이 검색이 실패한 원인이었다.
        """
        if not text or not query_words:
            return 0.0

        text_lower = text.lower()
        matches = sum(1 for w in query_words if w in text_lower)

        if matches == 0:
            text_words = {w for w in _WORD.findall(text_lower) if len(w) >= 2}
            matches = sum(1 for tw in text_words if any(tw in qw for qw in query_words))

        return matches / max(len(query_words), 1)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """토큰 수 추정 (한글 1자 ≈ 1.5 tokens, 영어 1단어 ≈ 1 token)."""
        korean_chars = sum(1 for c in text if "\uac00" <= c <= "\ud7a3")
        other_chars = len(text) - korean_chars
        return int(korean_chars * 1.5 + other_chars * 0.3)
