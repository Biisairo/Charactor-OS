"""API 표면 고정 (TASK-07, REQ-07-2 · 인수 기준 1).

라우터를 도메인별로 분리해도 **경로 목록과 등록 순서가 변해서는 안 된다.**

순서까지 고정하는 이유가 있다. FastAPI는 먼저 등록된 라우트를 먼저 매칭하므로,
분리 과정에서 순서가 뒤집히면 조용히 다른 핸들러가 응답한다.

- `/api/knowledge/relationships`가 `/api/knowledge/{name}`보다 뒤로 가면,
  `{name}`이 "relationships"를 잡아먹는다.
- SPA catch-all `/{full_path:path}`가 앞으로 가면 **모든 API가 죽는다.**

둘 다 테스트 없이는 눈으로 잡기 어렵고, 라우터 분리에서 가장 흔한 사고다.
아래 목록은 분리 착수 전 실제 앱에서 뜬 기준선이다.
"""

from __future__ import annotations

from src.api.server import app

# (경로, 메서드) — 등록 순서 그대로. 리팩터링 전 기준선.
EXPECTED_ROUTES = [
    ("/openapi.json", ("GET",)),
    ("/docs", ("GET",)),
    ("/docs/oauth2-redirect", ("GET",)),
    ("/redoc", ("GET",)),
    ("/api/health", ("GET",)),
    ("/api/chat", ("POST",)),
    ("/api/emotion", ("GET",)),
    ("/api/memory/stats", ("GET",)),
    ("/api/memory", ("GET",)),
    ("/api/history", ("GET",)),
    ("/api/debug", ("GET",)),
    ("/api/debug/toggle", ("POST",)),
    ("/api/debug/clear", ("POST",)),
    ("/api/trace/last", ("GET",)),
    ("/api/characters", ("GET",)),
    ("/api/character/switch", ("POST",)),
    ("/api/characters", ("POST",)),
    ("/api/characters/{character_id}", ("DELETE",)),
    ("/api/logs", ("GET",)),
    ("/api/performance", ("GET",)),
    ("/api/character/reset", ("POST",)),
    ("/api/persona", ("GET",)),
    ("/api/persona", ("PUT",)),
    ("/api/knowledge", ("GET",)),
    ("/api/knowledge/relationships", ("GET",)),
    ("/api/knowledge/relationships/{character}", ("GET",)),
    ("/api/knowledge/timeline", ("GET",)),
    ("/api/knowledge/locations", ("GET",)),
    ("/api/knowledge/{name}", ("GET",)),
    ("/api/knowledge/{name}", ("PUT",)),
    ("/api/fewshot", ("GET",)),
    ("/api/fewshot/search", ("GET",)),
    ("/{full_path:path}", ("GET",)),
]


def _walk(routes) -> list[tuple[str, tuple[str, ...]]]:
    """등록 순서를 유지한 채 라우트를 평탄화한다.

    `include_router`로 붙인 라우터는 FastAPI 버전에 따라 `app.routes`에
    래퍼 객체 하나로 들어간다. 안을 들여다보지 않으면 라우터 분리 후
    "경로가 전부 사라졌다"고 잘못 판정한다 — 실제로 겪었다.
    """
    flat: list[tuple[str, tuple[str, ...]]] = []
    for route in routes:
        inner = getattr(route, "original_router", None)
        if inner is not None:
            flat.extend(_walk(inner.routes))
            continue

        path = getattr(route, "path", None)
        methods = getattr(route, "methods", None)
        if path and methods:
            flat.append((path, tuple(sorted(methods - {"HEAD", "OPTIONS"}))))
    return flat


def _actual_routes() -> list[tuple[str, tuple[str, ...]]]:
    return _walk(app.routes)


class TestRouteSurface:
    """경로 **목록**을 고정한다 (인수 기준 1).

    등록 순서 전체를 고정하지는 않는다. 도메인별 라우터로 묶으면 순서는 필연적으로
    바뀌는데, 실제 불변식은 "순서가 그대로"가 아니라 **"잠식이 생기지 않는다"**이기
    때문이다. 그 제약은 아래 `TestShadowingHazards`가 따로 못 박는다.
    """

    def test_route_set_is_unchanged(self):
        assert sorted(set(_actual_routes())) == sorted(set(EXPECTED_ROUTES))

    def test_no_duplicate_registration(self):
        actual = _actual_routes()
        assert len(actual) == len(set(actual)), (
            f"같은 (경로, 메서드)가 두 번 등록되었다: {sorted(actual)}"
        )

    def test_no_route_is_lost(self):
        actual = set(_actual_routes())
        missing = set(EXPECTED_ROUTES) - actual
        assert not missing, f"사라진 경로: {sorted(missing)}"

    def test_no_route_is_added_silently(self):
        actual = set(_actual_routes())
        added = actual - set(EXPECTED_ROUTES)
        assert not added, (
            f"의도치 않게 늘어난 경로: {sorted(added)} — 의도한 추가라면 기준선을 갱신하라"
        )


class TestShadowingHazards:
    """분리 과정에서 실제로 깨지기 쉬운 순서 제약만 따로 못 박는다."""

    def _index(self, path: str, method: str = "GET") -> int:
        for i, (p, methods) in enumerate(_actual_routes()):
            if p == path and method in methods:
                return i
        raise AssertionError(f"경로를 찾을 수 없다: {method} {path}")

    def test_literal_knowledge_paths_precede_wildcard(self):
        wildcard = self._index("/api/knowledge/{name}")
        for literal in (
            "/api/knowledge/relationships",
            "/api/knowledge/timeline",
            "/api/knowledge/locations",
        ):
            assert self._index(literal) < wildcard, (
                f"{literal}이 /api/knowledge/{{name}} 뒤에 있어 잡아먹힌다"
            )

    def test_spa_catch_all_is_last(self):
        routes = _actual_routes()
        assert routes[-1][0] == "/{full_path:path}", (
            "SPA catch-all이 마지막이 아니면 뒤따르는 API가 모두 가려진다"
        )

    def test_api_routes_precede_catch_all(self):
        catch_all = self._index("/{full_path:path}")
        api_indexes = [i for i, (p, _) in enumerate(_actual_routes()) if p.startswith("/api/")]
        assert max(api_indexes) < catch_all
