"""CharacterWorker 동시성 검증 (TASK-02 / REQ-02-6 ~ 02-8).

CharacterOS는 스레드 안전하지 않다. API 계층은 락 대신 전용 스레드 + 큐로
모든 작업을 직렬화하여 이 문제를 회피한다. 이 설계가 실제로 직렬성을
보장하는지, 그리고 개별 작업의 실패가 워커 전체를 죽이지 않는지 검증한다.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from src.api.server import CharacterWorker

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _DummyCos:
    """CharacterWorker는 cos를 보관만 하므로 더미로 충분하다."""


@pytest.fixture
def worker():
    w = CharacterWorker(_DummyCos())
    yield w
    w.shutdown()


# ---------------------------------------------------------------------------
# REQ-02-6 — 제출 순서대로 직렬 실행된다
# ---------------------------------------------------------------------------


class TestSerialExecution:
    async def test_tasks_run_in_submission_order(self, worker: CharacterWorker):
        order: list[int] = []

        def make(idx: int):
            def task():
                time.sleep(0.002)
                order.append(idx)
                return idx

            return task

        futures = [worker.submit(make(i))[1] for i in range(8)]
        results = await asyncio.gather(*futures)

        assert order == list(range(8)), "큐는 FIFO이므로 제출 순서가 보존되어야 한다"
        assert results == list(range(8))

    async def test_tasks_never_overlap(self, worker: CharacterWorker):
        """단일 스레드이므로 두 작업이 동시에 실행되어서는 안 된다."""
        state = {"active": 0, "overlap": False}

        def task():
            state["active"] += 1
            if state["active"] > 1:
                state["overlap"] = True
            time.sleep(0.002)
            state["active"] -= 1

        futures = [worker.submit(task)[1] for _ in range(8)]
        await asyncio.gather(*futures)

        assert state["overlap"] is False

    async def test_run_returns_task_result(self, worker: CharacterWorker):
        assert await worker.run(lambda: "결과") == "결과"

    async def test_event_loop_not_blocked(self, worker: CharacterWorker):
        """워커가 블로킹 작업을 처리하는 동안 이벤트 루프는 계속 돌아야 한다."""
        ticks = 0

        async def ticker():
            nonlocal ticks
            for _ in range(5):
                await asyncio.sleep(0.001)
                ticks += 1

        await asyncio.gather(worker.run(lambda: time.sleep(0.02)), ticker())

        assert ticks == 5


# ---------------------------------------------------------------------------
# REQ-02-7 — 작업 실패가 해당 Future에만 전달되고 워커는 살아남는다
# ---------------------------------------------------------------------------


class TestFailureIsolation:
    async def test_exception_delivered_to_future(self, worker: CharacterWorker):
        def boom():
            raise ValueError("작업 실패")

        with pytest.raises(ValueError, match="작업 실패"):
            await worker.run(boom)

    async def test_worker_survives_failed_task(self, worker: CharacterWorker):
        """실패 이후에도 후속 작업이 정상 처리되어야 한다."""

        def boom():
            raise RuntimeError("실패")

        with pytest.raises(RuntimeError):
            await worker.run(boom)

        assert await worker.run(lambda: "정상") == "정상"

    async def test_failure_does_not_affect_other_futures(self, worker: CharacterWorker):
        def boom():
            raise RuntimeError("실패")

        _, bad = worker.submit(boom)
        _, good = worker.submit(lambda: "정상")

        results = await asyncio.gather(bad, good, return_exceptions=True)

        assert isinstance(results[0], RuntimeError)
        assert results[1] == "정상"


# ---------------------------------------------------------------------------
# REQ-02-8 — shutdown이 스레드를 종료시킨다
# ---------------------------------------------------------------------------


class TestShutdown:
    async def test_thread_terminates(self):
        w = CharacterWorker(_DummyCos())
        assert w._thread.is_alive()

        w.shutdown()

        assert not w._thread.is_alive()

    async def test_shutdown_after_work(self):
        w = CharacterWorker(_DummyCos())
        await w.run(lambda: None)

        w.shutdown()

        assert not w._thread.is_alive()

    async def test_worker_is_daemon(self):
        """워커가 남아 프로세스 종료를 막아서는 안 된다."""
        w = CharacterWorker(_DummyCos())
        try:
            assert w._thread.daemon
        finally:
            w.shutdown()
