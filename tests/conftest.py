from __future__ import annotations

from datetime import datetime

import pytest

from voice_helper.app import Assistant, build
from voice_helper.dialog import CallState, Reply

# Четверг, 15.10.2026 11:00 (GMT+4): операторы работают, слоты в Excel есть
NOW = datetime(2026, 10, 15, 11, 0)


class Clock:
    def __init__(self, now: datetime = NOW):
        self.now = now

    def __call__(self) -> datetime:
        return self.now


class Call:
    """Звонок в тесте: say() — реплика пациента, возвращает ответ ассистента."""

    def __init__(self, assistant: Assistant, state: CallState, first: Reply):
        self.assistant = assistant
        self.state = state
        self.replies = [first]

    @property
    def last(self) -> Reply:
        return self.replies[-1]

    async def say(self, *texts: str) -> Reply:
        for text in texts:
            self.replies.append(await self.assistant.dialog.handle(self.state, text))
        return self.last


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def assistant(clock: Clock) -> Assistant:
    return build(fake=True, clock=clock, db_path=":memory:", use_llm=False)


@pytest.fixture
def call(assistant: Assistant):
    async def _start(phone: str = "+7 900 000-00-01") -> Call:
        state, first = await assistant.dialog.start_inbound("test-call", phone)
        return Call(assistant, state, first)
    return _start


@pytest.fixture
def stand(assistant: Assistant):
    return assistant.registry
