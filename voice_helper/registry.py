"""HTTP-клиент тестового стенда «Регистратура» (/api/v1).

Особенности стенда, которые учитывает клиент:
- лимит 10 запросов в минуту на команду: собственный ограничитель + повтор по 429/Retry-After;
- списки отдаются страницами не больше 50 объектов;
- справочник врачей общий и меняется редко — кэшируется;
- POST /clients идемпотентен: 201 — создан, 200 — карточка уже была.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from datetime import date, datetime
from typing import Any, Awaitable, Callable, Protocol

import httpx

from .models import Client, Doctor, Escalation, Visit, fmt_date, fmt_datetime

log = logging.getLogger(__name__)

PAGE_SIZE = 50


class RegistryError(Exception):
    """Ошибка стенда: HTTP-статус, код («НЕВЕРНЫЙ_ЗАПРОС», «КОНФЛИКТ_СТАТУСА»…) и описание."""

    def __init__(self, status: int, code: str, message: str):
        super().__init__(f"{status} {code}: {message}")
        self.status = status
        self.code = code
        self.message = message


class Registry(Protocol):
    """Операции стенда, которые нужны ассистенту. Реализуют RegistryClient и FakeRegistry."""

    async def list_doctors(self) -> list[Doctor]: ...
    async def find_clients(self, last_name: str, first_name: str, birth_date: date) -> list[Client]: ...
    async def get_client(self, client_id: int) -> Client: ...
    async def ensure_client(
        self, last_name: str, first_name: str, birth_date: date, middle_name: str | None = None
    ) -> Client: ...
    async def list_visits(
        self,
        client_id: int | None = None,
        doctor_id: int | None = None,
        status: str | None = None,
        date_from: date | None = None,
        date_to: date | None = None,
    ) -> list[Visit]: ...
    async def get_visit(self, visit_id: int) -> Visit: ...
    async def create_visit(self, client_id: int, doctor_id: int, at: datetime) -> Visit: ...
    async def update_visit(
        self,
        visit_id: int,
        doctor_id: int | None = None,
        at: datetime | None = None,
        status: str | None = None,
    ) -> Visit: ...
    async def create_escalation(self, client_id: int, description: str) -> Escalation: ...


class RateLimiter:
    """Скользящее окно: не больше max_calls запросов за period секунд."""

    def __init__(
        self,
        max_calls: int = 10,
        period: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ):
        self.max_calls = max_calls
        self.period = period
        self._clock = clock
        self._sleep = sleep
        self._calls: deque[float] = deque()
        self._blocked_until = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            while True:
                now = self._clock()
                while self._calls and now - self._calls[0] >= self.period:
                    self._calls.popleft()
                wait = self._blocked_until - now
                if wait <= 0 and len(self._calls) < self.max_calls:
                    self._calls.append(now)
                    return
                if wait <= 0:
                    wait = self._calls[0] + self.period - now
                await self._sleep(max(wait, 0.05))

    def block_for(self, seconds: float) -> None:
        """Стенд ответил 429: не отправлять ничего ближайшие seconds секунд."""
        self._blocked_until = max(self._blocked_until, self._clock() + seconds)


class RegistryClient:
    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        rate_limit: int = 10,
        doctors_ttl: float = 600.0,
        max_retries: int = 2,
        timeout: float = 15.0,
        transport: httpx.AsyncBaseTransport | None = None,
        limiter: RateLimiter | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ):
        self._http = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {token}"},
            timeout=httpx.Timeout(timeout),
            transport=transport,
        )
        self._limiter = limiter or RateLimiter(rate_limit, 60.0)
        self._sleep = sleep
        self._max_retries = max_retries
        self._doctors_ttl = doctors_ttl
        self._doctors: list[Doctor] | None = None
        self._doctors_at = 0.0

    async def aclose(self) -> None:
        await self._http.aclose()

    # --- транспорт -----------------------------------------------------------

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        for attempt in range(self._max_retries + 1):
            await self._limiter.acquire()
            resp = await self._http.request(method, path, **kwargs)
            if resp.status_code == 429 and attempt < self._max_retries:
                retry_after = float(resp.headers.get("Retry-After", "5"))
                log.warning("Стенд: превышен лимит, ждём %.0f с", retry_after)
                self._limiter.block_for(retry_after)
                continue
            # 500 у PATCH стенд описывает как гонку встречных изменений: повтор проходит
            if resp.status_code == 500 and method == "PATCH" and attempt < self._max_retries:
                await self._sleep(0.5)
                continue
            break
        if resp.status_code >= 400:
            try:
                body = resp.json()
                code, message = body.get("code", ""), body.get("message", resp.text)
            except ValueError:
                code, message = "", resp.text
            raise RegistryError(resp.status_code, code, message)
        return resp

    async def _paginate(self, path: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        offset = 0
        while True:
            page = (await self._request("GET", path, params={**params, "limit": PAGE_SIZE, "offset": offset})).json()
            items.extend(page)
            if len(page) < PAGE_SIZE:
                return items
            offset += PAGE_SIZE

    # --- врачи ---------------------------------------------------------------

    async def list_doctors(self) -> list[Doctor]:
        if self._doctors is None or time.monotonic() - self._doctors_at > self._doctors_ttl:
            self._doctors = [Doctor.from_api(d) for d in await self._paginate("/doctors", {})]
            self._doctors_at = time.monotonic()
        return self._doctors

    # --- клиенты -------------------------------------------------------------

    async def find_clients(self, last_name: str, first_name: str, birth_date: date) -> list[Client]:
        params = {"last_name": last_name, "first_name": first_name, "birth_date": fmt_date(birth_date)}
        return [Client.from_api(c) for c in await self._paginate("/clients", params)]

    async def get_client(self, client_id: int) -> Client:
        return Client.from_api((await self._request("GET", f"/clients/{client_id}")).json())

    async def ensure_client(
        self, last_name: str, first_name: str, birth_date: date, middle_name: str | None = None
    ) -> Client:
        body: dict[str, Any] = {"last_name": last_name, "first_name": first_name, "birth_date": fmt_date(birth_date)}
        if middle_name:
            body["middle_name"] = middle_name
        return Client.from_api((await self._request("POST", "/clients", json=body)).json())

    # --- записи --------------------------------------------------------------

    async def list_visits(
        self,
        client_id: int | None = None,
        doctor_id: int | None = None,
        status: str | None = None,
        date_from: date | None = None,
        date_to: date | None = None,
    ) -> list[Visit]:
        params: dict[str, Any] = {}
        if client_id is not None:
            params["client_id"] = client_id
        if doctor_id is not None:
            params["doctor_id"] = doctor_id
        if status:
            params["status"] = status
        if date_from:
            params["date_from"] = fmt_date(date_from)
        if date_to:
            params["date_to"] = fmt_date(date_to)
        return [Visit.from_api(v) for v in await self._paginate("/visits", params)]

    async def get_visit(self, visit_id: int) -> Visit:
        return Visit.from_api((await self._request("GET", f"/visits/{visit_id}")).json())

    async def create_visit(self, client_id: int, doctor_id: int, at: datetime) -> Visit:
        body = {"client_id": client_id, "doctor_id": doctor_id, "scheduled_at": fmt_datetime(at)}
        return Visit.from_api((await self._request("POST", "/visits", json=body)).json())

    async def update_visit(
        self,
        visit_id: int,
        doctor_id: int | None = None,
        at: datetime | None = None,
        status: str | None = None,
    ) -> Visit:
        body: dict[str, Any] = {}
        if doctor_id is not None:
            body["doctor_id"] = doctor_id
        if at is not None:
            body["scheduled_at"] = fmt_datetime(at)
        if status is not None:
            body["status"] = status
        return Visit.from_api((await self._request("PATCH", f"/visits/{visit_id}", json=body)).json())

    # --- эскалации -----------------------------------------------------------

    async def create_escalation(self, client_id: int, description: str) -> Escalation:
        body = {"client_id": client_id, "description": description[:1000]}
        return Escalation.from_api((await self._request("POST", "/escalations", json=body)).json())
