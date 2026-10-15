from __future__ import annotations

import json
from datetime import date, datetime

import httpx
import pytest

from voice_helper.models import VisitStatus
from voice_helper.registry import RateLimiter, RegistryClient, RegistryError

DOCTOR = {"id": 3, "full_name": "Иванова Мария Петровна", "specialty": "терапевт", "is_active": True}
VISIT = {"id": 1, "client_id": 7, "doctor": DOCTOR, "scheduled_at": "16.10.2026 10:20",
         "status": "Запланирована", "created_at": "15.10.2026 11:00", "updated_at": "15.10.2026 11:00"}


class FakeTime:
    def __init__(self):
        self.t = 0.0
        self.slept: list[float] = []

    def clock(self) -> float:
        return self.t

    async def sleep(self, s: float) -> None:
        self.slept.append(s)
        self.t += s


def make_client(handler, ft: FakeTime | None = None, rate: int = 10) -> RegistryClient:
    ft = ft or FakeTime()
    return RegistryClient(
        "https://stand/api/v1", "secret",
        transport=httpx.MockTransport(handler),
        limiter=RateLimiter(rate, 60, clock=ft.clock, sleep=ft.sleep),
        sleep=ft.sleep,
    )


async def test_auth_header_and_formats():
    seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        assert req.headers["Authorization"] == "Bearer secret"
        return httpx.Response(201, json=VISIT)

    client = make_client(handler)
    visit = await client.create_visit(7, 3, datetime(2026, 10, 16, 10, 20))
    assert json.loads(seen[0].content) == {"client_id": 7, "doctor_id": 3, "scheduled_at": "16.10.2026 10:20"}
    assert seen[0].url.path == "/api/v1/visits"
    assert visit.scheduled_at == datetime(2026, 10, 16, 10, 20) and visit.doctor.full_name == DOCTOR["full_name"]


async def test_retry_after_429():
    ft = FakeTime()
    calls = []

    def handler(req):
        calls.append(req)
        if len(calls) == 1:
            return httpx.Response(429, headers={"Retry-After": "7"},
                                  json={"code": "ПРЕВЫШЕН_ЛИМИТ", "message": "Превышен лимит"})
        return httpx.Response(200, json=[DOCTOR])

    client = make_client(handler, ft)
    doctors = await client.list_doctors()
    assert len(calls) == 2 and doctors[0].id == 3
    assert sum(ft.slept) >= 7


async def test_rate_limiter_spaces_requests():
    ft = FakeTime()
    client = make_client(lambda req: httpx.Response(200, json=VISIT), ft, rate=2)
    for _ in range(3):
        await client.get_visit(1)
    assert ft.t >= 60  # третий запрос ждал освобождения окна


async def test_pagination_and_query_params():
    pages = []

    def handler(req):
        pages.append(dict(req.url.params))
        offset = int(req.url.params["offset"])
        n = 50 if offset == 0 else 3
        return httpx.Response(200, json=[{**VISIT, "id": offset + i + 1} for i in range(n)])

    client = make_client(handler)
    visits = await client.list_visits(client_id=7, status=VisitStatus.PLANNED, date_from=date(2026, 10, 15))
    assert len(visits) == 53
    assert pages[0] == {"client_id": "7", "status": "Запланирована", "date_from": "15.10.2026", "limit": "50", "offset": "0"}
    assert pages[1]["offset"] == "50"


async def test_error_mapping():
    def handler(req):
        return httpx.Response(409, json={"code": "КОНФЛИКТ_СТАТУСА", "message": "Запись отменена"})

    client = make_client(handler)
    with pytest.raises(RegistryError) as e:
        await client.update_visit(1, status=VisitStatus.CANCELLED)
    assert e.value.status == 409 and e.value.code == "КОНФЛИКТ_СТАТУСА"


async def test_doctors_are_cached():
    calls = []

    def handler(req):
        calls.append(req)
        return httpx.Response(200, json=[DOCTOR])

    client = make_client(handler)
    await client.list_doctors()
    await client.list_doctors()
    assert len(calls) == 1
