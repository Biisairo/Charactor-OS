"""경로 안전성 검증 — `_safe_child`가 디렉토리 탈출을 차단하는지 확인한다.

API는 character_id·지식 파일명을 URL/본문에서 그대로 받아 경로로 사용하므로,
검증이 없으면 임의 파일 읽기/쓰기/삭제가 가능하다.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException

from src.api.server import SAFE_FILENAME, SAFE_SEGMENT, _safe_child

# ---------------------------------------------------------------------------
# 정상 입력은 통과한다
# ---------------------------------------------------------------------------


class TestValidInput:
    def test_plain_segment(self, tmp_path: Path):
        result = _safe_child(tmp_path, "hong-gil-dong", SAFE_SEGMENT)
        assert result == (tmp_path / "hong-gil-dong").resolve()

    def test_hangul_segment(self, tmp_path: Path):
        result = _safe_child(tmp_path, "홍길동", SAFE_SEGMENT)
        assert result.name == "홍길동"

    def test_underscore_and_digits(self, tmp_path: Path):
        assert _safe_child(tmp_path, "char_2", SAFE_SEGMENT).name == "char_2"

    @pytest.mark.parametrize("name", ["world.yaml", "world.yml", "a.json", "b.md", "c.txt"])
    def test_allowed_extensions(self, tmp_path: Path, name: str):
        assert _safe_child(tmp_path, name, SAFE_FILENAME).name == name


# ---------------------------------------------------------------------------
# 경로 탈출 시도는 400으로 거부된다
# ---------------------------------------------------------------------------


class TestTraversalRejected:
    @pytest.mark.parametrize(
        "attack",
        [
            "../../etc/passwd",
            "..",
            ".",
            "../sibling",
            "sub/child",
            "sub\\child",
            "/etc/passwd",
            "",
            ".hidden",
            "name\x00.yaml",
        ],
    )
    def test_segment_traversal_rejected(self, tmp_path: Path, attack: str):
        with pytest.raises(HTTPException) as exc:
            _safe_child(tmp_path, attack, SAFE_SEGMENT)
        assert exc.value.status_code == 400

    @pytest.mark.parametrize(
        "attack",
        [
            "../../../etc/passwd",
            "../persona.yaml",
            "sub/world.yaml",
            "world.yaml/../../secret.yaml",
            "/absolute/world.yaml",
        ],
    )
    def test_filename_traversal_rejected(self, tmp_path: Path, attack: str):
        with pytest.raises(HTTPException) as exc:
            _safe_child(tmp_path, attack, SAFE_FILENAME)
        assert exc.value.status_code == 400

    @pytest.mark.parametrize("name", ["script.sh", "config.env", "noextension", "a.exe"])
    def test_disallowed_extension_rejected(self, tmp_path: Path, name: str):
        """지식 파일은 허용된 확장자만 쓸 수 있다."""
        with pytest.raises(HTTPException) as exc:
            _safe_child(tmp_path, name, SAFE_FILENAME)
        assert exc.value.status_code == 400


# ---------------------------------------------------------------------------
# 심볼릭 링크로 base를 벗어나는 경우도 차단된다
# ---------------------------------------------------------------------------


class TestSymlinkEscape:
    def test_symlink_out_of_base_rejected(self, tmp_path: Path):
        base = tmp_path / "base"
        base.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()

        # 형식 검증은 통과하지만 resolve하면 base 바깥을 가리키는 링크
        (base / "escape").symlink_to(outside)

        with pytest.raises(HTTPException) as exc:
            _safe_child(base, "escape", SAFE_SEGMENT)
        assert exc.value.status_code == 400
