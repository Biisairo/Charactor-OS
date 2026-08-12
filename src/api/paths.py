"""요청에서 온 이름을 파일 경로로 쓰기 전의 검증.

`character_id`·지식 파일명은 URL과 요청 본문에서 오는 값이다. 그대로 경로에
붙이면 `../../etc/passwd` 같은 입력으로 디렉토리를 벗어난다.
형식 검증(허용 문자만)과 resolve 후 위치 확인을 함께 적용한다.
"""

from __future__ import annotations

import re
from pathlib import Path

from fastapi import HTTPException

CHARACTERS_DIR = Path("characters")

# 캐릭터 디렉토리명: 영숫자·한글·밑줄·하이픈만 (점·슬래시 불가)
SAFE_SEGMENT = re.compile(r"[A-Za-z0-9가-힣_-]+")
# 지식 파일명: 위 문자 + 허용된 확장자
SAFE_FILENAME = re.compile(r"[A-Za-z0-9가-힣_-]+\.(yaml|yml|json|md|txt)")


def safe_child(base: Path, segment: str, pattern: re.Pattern[str]) -> Path:
    """`base` 바로 아래에 있는 경로를 검증하여 반환한다.

    형식 검증과 위치 확인을 모두 통과해야 한다. 형식 검증만으로도 충분하지만,
    resolve 후 부모가 `base`인지 재확인하여 심볼릭 링크 우회까지 막는다.

    Raises:
        HTTPException: 형식이 맞지 않거나 `base`를 벗어나는 경우 400.
    """
    if not pattern.fullmatch(segment):
        raise HTTPException(status_code=400, detail=f"허용되지 않는 이름입니다: {segment}")

    resolved = (base / segment).resolve()
    if resolved.parent != base.resolve():
        raise HTTPException(status_code=400, detail=f"허용되지 않는 경로입니다: {segment}")
    return resolved
