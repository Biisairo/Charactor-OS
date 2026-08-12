"""대화에서 무언가를 뽑아내는 LLM 상호작용 층.

도메인 모듈(`MemoryModule`·`EmotionModule`)에서 **프롬프트 조립·LLM 호출·응답 파싱**을
떼어낸 자리다. 분리 전에는 한 모듈이 도메인 상태·영속화·LLM 상호작용을 겸했고,
그 결과 `update()`가 163줄까지 자랐다 (TASK-14).

분리의 이득은 길이가 아니다.

- 프롬프트가 한곳에 모여 TASK-08·09 같은 튜닝이 도메인 로직을 건드리지 않는다
- 프로바이더 거부 처리가 파싱 지점마다 흩어지지 않는다
- **도메인 규칙을 LLM 더블 없이 테스트할 수 있다** — 모듈은 이제 결과 타입만 받는다
"""

from src.analysis.emotion_analyzer import EmotionAnalysis, EmotionAnalyzer
from src.analysis.memory_analyzer import (
    Classification,
    ConflictClassifier,
    MemoryCandidate,
    MemoryExtractor,
)

__all__ = [
    "Classification",
    "ConflictClassifier",
    "EmotionAnalysis",
    "EmotionAnalyzer",
    "MemoryCandidate",
    "MemoryExtractor",
]
