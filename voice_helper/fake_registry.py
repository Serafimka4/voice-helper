"""Фейковый стенд «Регистратура» в памяти — для тестов и работы без токена.

Повторяет правила настоящего стенда: ключ клиента (фамилия, имя, дата рождения),
POST /clients возвращает существующую карточку, терминальные записи не меняются (409),
к деактивированному врачу нельзя записать или перенести (400), занятость не проверяется.
"""
from __future__ import annotations

from datetime import date, datetime

from .models import Client, Doctor, Escalation, Visit, VisitStatus, stand_now
from .registry import RegistryError
from .schedule import Schedule


class FakeRegistry:
    def __init__(self, doctors: list[Doctor], clock=stand_now):
        self.doctors = {d.id: d for d in doctors}
        self.clients: dict[int, Client] = {}
        self.visits: dict[int, Visit] = {}
        self.escalations: dict[int, Escalation] = {}
        self.clock = clock
        self.requests = 0

    @classmethod
    def from_schedule(cls, schedule: Schedule, clock=stand_now) -> FakeRegistry:
        doctors = [Doctor(i + 1, d.full_name, d.specialty, True) for i, d in enumerate(schedule.doctors)]
        return cls(doctors, clock)

    def deactivate(self, doctor_id: int) -> None:
        d = self.doctors[doctor_id]
        self.doctors[doctor_id] = Doctor(d.id, d.full_name, d.specialty, False)
        for vid, v in self.visits.items():
            if v.doctor.id == doctor_id:
                self.visits[vid] = Visit(v.id, v.client_id, self.doctors[doctor_id], v.scheduled_at, v.status, v.created_at)

    def _tick(self) -> None:
        self.requests += 1

    async def list_doctors(self) -> list[Doctor]:
        self._tick()
        return sorted(self.doctors.values(), key=lambda d: d.id)

    async def find_clients(self, last_name: str, first_name: str, birth_date: date) -> list[Client]:
        self._tick()
        return [c for c in self.clients.values()
                if (c.last_name, c.first_name, c.birth_date) == (last_name.strip(), first_name.strip(), birth_date)]

    async def get_client(self, client_id: int) -> Client:
        self._tick()
        if client_id not in self.clients:
            raise RegistryError(404, "НЕ_НАЙДЕНО", "Клиент не найден.")
        return self.clients[client_id]

    async def ensure_client(self, last_name: str, first_name: str, birth_date: date,
                            middle_name: str | None = None) -> Client:
        found = await self.find_clients(last_name, first_name, birth_date)
        if found:
            return found[0]
        cid = len(self.clients) + 1
        client = Client(cid, last_name.strip(), first_name.strip(), (middle_name or "").strip(), birth_date)
        self.clients[cid] = client
        return client

    async def list_visits(self, client_id: int | None = None, doctor_id: int | None = None, status: str | None = None,
                          date_from: date | None = None, date_to: date | None = None) -> list[Visit]:
        self._tick()
        if client_id is not None and client_id not in self.clients:
            raise RegistryError(404, "НЕ_НАЙДЕНО", "Клиент не найден.")
        out = [
            v for v in self.visits.values()
            if (client_id is None or v.client_id == client_id)
            and (doctor_id is None or v.doctor.id == doctor_id)
            and (status is None or v.status == status)
            and (date_from is None or v.scheduled_at.date() >= date_from)
            and (date_to is None or v.scheduled_at.date() <= date_to)
        ]
        return sorted(out, key=lambda v: (v.scheduled_at, v.id))

    async def get_visit(self, visit_id: int) -> Visit:
        self._tick()
        if visit_id not in self.visits:
            raise RegistryError(404, "НЕ_НАЙДЕНО", "Запись не найдена.")
        return self.visits[visit_id]

    def _active_doctor(self, doctor_id: int) -> Doctor:
        d = self.doctors.get(doctor_id)
        if d is None:
            raise RegistryError(400, "НЕВЕРНЫЙ_ЗАПРОС", "Поле «doctor_id» указывает на врача вне справочника.")
        if not d.is_active:
            raise RegistryError(400, "НЕВЕРНЫЙ_ЗАПРОС",
                                "Поле «doctor_id» указывает на деактивированного врача: запись к нему невозможна.")
        return d

    async def create_visit(self, client_id: int, doctor_id: int, at: datetime) -> Visit:
        self._tick()
        if client_id not in self.clients:
            raise RegistryError(400, "НЕВЕРНЫЙ_ЗАПРОС", "Поле «client_id» указывает на несуществующего клиента.")
        doctor = self._active_doctor(doctor_id)
        vid = len(self.visits) + 1
        visit = Visit(vid, client_id, doctor, at, VisitStatus.PLANNED, self.clock())
        self.visits[vid] = visit
        return visit

    async def update_visit(self, visit_id: int, doctor_id: int | None = None, at: datetime | None = None,
                           status: str | None = None) -> Visit:
        self._tick()
        v = self.visits.get(visit_id)
        if v is None:
            raise RegistryError(404, "НЕ_НАЙДЕНО", "Запись не найдена.")
        if v.status != VisitStatus.PLANNED:
            if status == v.status and doctor_id is None and at is None:
                return v
            raise RegistryError(409, "КОНФЛИКТ_СТАТУСА", "Запись в терминальном статусе изменить нельзя.")
        if not v.doctor.is_active and (doctor_id is not None or at is not None or status not in (None, VisitStatus.CANCELLED)):
            raise RegistryError(409, "КОНФЛИКТ_СТАТУСА", "Врач записи на приём деактивирован: запись можно только отменить.")
        doctor = self._active_doctor(doctor_id) if doctor_id is not None else v.doctor
        new = Visit(v.id, v.client_id, doctor, at or v.scheduled_at, status or v.status, v.created_at)
        self.visits[visit_id] = new
        return new

    async def create_escalation(self, client_id: int, description: str) -> Escalation:
        self._tick()
        if client_id not in self.clients:
            raise RegistryError(400, "НЕВЕРНЫЙ_ЗАПРОС", "Поле «client_id» указывает на несуществующего клиента.")
        if not description.strip() or len(description) > 1000:
            raise RegistryError(400, "НЕВЕРНЫЙ_ЗАПРОС", "Поле «description» недопустимо.")
        eid = len(self.escalations) + 1
        esc = Escalation(eid, client_id, description)
        self.escalations[eid] = esc
        return esc
