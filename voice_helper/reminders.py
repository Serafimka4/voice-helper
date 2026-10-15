"""Отбор записей для исходящих напоминаний (сценарий С-10).

Правила регламента:
- звонить за 1 рабочий день до приёма; для приёмов в понедельник (и субботу) — в пятницу;
- будни 10:00–19:00, суббота 10:00–15:00, в воскресенье не звонить;
- не звонить: запись создана менее 24 часов назад; пациент отказался от обзвона;
  запись уже подтверждена; уже было 2 попытки.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

from .models import Client, Visit, VisitStatus
from .registry import Registry
from .storage import Storage

MAX_ATTEMPTS = 2
FINAL_STATUSES = {"CONFIRMED", "RESCHEDULED", "CANCELLED", "OPT_OUT", "WRONG_PERSON", "UNCONFIRMED"}


@dataclass
class ReminderTask:
    visit: Visit
    client: Client
    phone: str


def call_day_for(visit_day: date) -> date:
    """Рабочий день звонка: накануне, пропуская воскресенье и субботу (Пн → Пт)."""
    d = visit_day - timedelta(days=1)
    while d.weekday() in (5, 6):
        d -= timedelta(days=1)
    return d


def calling_window_open(now: datetime) -> bool:
    if now.weekday() == 6:
        return False
    end = time(15) if now.weekday() == 5 else time(19)
    return time(10) <= now.time() < end


async def due_reminders(registry: Registry, storage: Storage, now: datetime) -> list[ReminderTask]:
    today = now.date()
    horizon = today + timedelta(days=4)  # с пятницы покрываем субботу и понедельник
    visits = await registry.list_visits(status=VisitStatus.PLANNED, date_from=today + timedelta(days=1), date_to=horizon)
    tasks: list[ReminderTask] = []
    for v in visits:
        if call_day_for(v.scheduled_at.date()) != today:
            continue
        if v.created_at and now - v.created_at < timedelta(hours=24):
            continue
        if storage.is_opted_out(v.client_id):
            continue
        rem = storage.reminder(v.id)
        if rem and (rem["status"] in FINAL_STATUSES or rem["attempts"] >= MAX_ATTEMPTS):
            continue
        phone = storage.phone_of(v.client_id)
        if not phone:
            continue
        tasks.append(ReminderTask(v, await registry.get_client(v.client_id), phone))
    return tasks
