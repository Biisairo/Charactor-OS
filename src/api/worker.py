"""캐릭터별 전용 스레드 워커.

대화는 상태를 바꾸므로 동시에 실행되면 안 된다. 락을 흩뿌리는 대신
캐릭터당 스레드 하나에 큐를 붙여 **제출 순서대로 직렬 실행**한다.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from queue import Empty, Queue

from src.character_os import CharacterOS


class CharacterWorker:
    """CharacterOS를 전용 스레드에서 순차 처리하는 워커.

    대화 요청은 큐에 들어가고, 전용 스레드가 하나씩 처리한다.
    상태 읽기(emotion, memory 등)는 직접 읽어도 안전하다
    (파이썬 GIL + 워커가 읽기 사이에 yield하지 않음).
    """

    def __init__(self, cos: CharacterOS):
        self.cos = cos
        self._queue: Queue = Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        """전용 스레드: 큐에서 작업을 꺼내 순차 실행."""
        while True:
            try:
                item = self._queue.get(timeout=1.0)
            except Empty:
                continue

            if item is None:  # shutdown signal
                break

            loop, future, fn = item
            try:
                result = fn()
                loop.call_soon_threadsafe(future.set_result, result)
            except Exception as e:
                loop.call_soon_threadsafe(future.set_exception, e)

    def submit(self, fn: Callable) -> tuple[asyncio.AbstractEventLoop, asyncio.Future]:
        """함수를 큐에 제출하고 (loop, future)를 반환한다."""
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._queue.put((loop, future, fn))
        return loop, future

    async def run(self, fn: Callable) -> any:
        """함수를 큐에 제출하고 결과를 기다린다 (async)."""
        _, future = self.submit(fn)
        return await future

    def shutdown(self) -> None:
        """워커 종료."""
        self._queue.put(None)
        self._thread.join(timeout=5)
