"""빌드된 프론트엔드 정적 파일 서빙 (SPA).

**이 라우터는 반드시 마지막에 등록되어야 한다.** catch-all이 앞서면
뒤따르는 API 경로가 전부 가려진다. `tests/integration/test_api_surface.py`가
그 순서를 지킨다.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

FRONTEND_DIR = Path(__file__).parent.parent.parent.parent / "frontend" / "dist"

router = APIRouter(tags=["frontend"])


@router.get("/{full_path:path}")
async def serve_frontend(full_path: str):
    """프론트엔드 정적 파일 서빙 (SPA fallback).

    dist 디렉토리를 벗어나는 경로는 파일을 반환하지 않고 index.html로 폴백한다.
    """
    if not FRONTEND_DIR.exists():
        return {"error": "Frontend not built. Run: cd frontend && npm run build"}

    index = FRONTEND_DIR / "index.html"
    base = FRONTEND_DIR.resolve()
    target = (FRONTEND_DIR / full_path).resolve()

    # dist 하위인지 확인 — 벗어나면 SPA 라우트로 간주하고 index.html
    if not str(target).startswith(str(base) + os.sep) or not target.is_file():
        return FileResponse(index)
    return FileResponse(target)
