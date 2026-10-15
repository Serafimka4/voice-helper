"""Подбор свободного времени.

Свободный слот = «Свободен» в Excel кейса
               − записи нашей команды на стенде в статусе «Запланирована»
               − время раньше чем через 30 минут (правило С-01)
               − дни недоступности врача.
Горизонт записи — 14 календарных дней (С-01).
"""
from __future__ import annotations

import time as _time
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

from .models import Clock, Doctor, Visit, VisitStatus, normalize_name
from .registry import Registry
from .schedule import Schedule

# Части дня по Приложению А.3 регламента
PARTS = {
    "morning": (time(7, 30), time(12, 0)),
    "day": (time(12, 0), time(16, 0)),
    "evening": (time(16, 0), time(23, 59)),
}


@dataclass(frozen=True)
class Slot:
    doctor: Doctor
    at: datetime
    room: int | None = None


@dataclass
class SlotQuery:
    doctor_ids: list[int]
    day: date | None = None          # конкретный день
    date_from: date | None = None    # «на следующей неделе» и т. п.
    date_to: date | None = None
    part: str | None = None          # morning / day / evening
    at: time | None = None           # желаемое время
    late: bool = False               # «попозже» — последние слоты дня
    exclude_days: tuple[int, ...] = ()  # «не в пятницу» → (4,)
    limit: int = 3


class SlotService:
    def __init__(
        self,
        registry: Registry,
        schedule: Schedule,
        clock: Clock,
        *,
        horizon_days: int = 14,
        min_lead: timedelta = timedelta(minutes=30),
        busy_ttl: float = 120.0,
    ):
        self.registry = registry
        self.schedule = schedule
        self.clock = clock
        self.horizon_days = horizon_days
        self.min_lead = min_lead
        self.busy_ttl = busy_ttl
        self._visits: dict[int, Visit] = {}
        self._loaded_at: float | None = None

    # --- врачи ---------------------------------------------------------------

    async def active_doctors(self, specialty: str | None = None) -> list[Doctor]:
        doctors = [d for d in await self.registry.list_doctors() if d.is_active]
        if specialty:
            key = normalize_name(specialty)
            doctors = [d for d in doctors if normalize_name(d.specialty) == key]
        return doctors

    def room_of(self, doctor: Doctor) -> int | None:
        info = self.schedule.doctor(doctor.full_name)
        return info.room if info else None

    # --- занятость (записи команды на стенде) --------------------------------

    async def _refresh_busy(self) -> None:
        if self._loaded_at is not None and _time.monotonic() - self._loaded_at < self.busy_ttl:
            return
        today = self.clock().date()
        visits = await self.registry.list_visits(
            status=VisitStatus.PLANNED,
            date_from=today,
            date_to=today + timedelta(days=self.horizon_days),
        )
        self._visits = {v.id: v for v in visits}
        self._loaded_at = _time.monotonic()

    def note_visit(self, visit: Visit) -> None:
        """Обновить кэш занятости после создания/переноса/отмены, не дожидаясь TTL."""
        self._visits[visit.id] = visit

    def _is_busy(self, doctor: Doctor, at: datetime) -> bool:
        info = self.schedule.doctor(doctor.full_name)
        duration = timedelta(minutes=info.slot_minutes if info else 20)
        return any(
            v.is_planned and v.doctor.id == doctor.id and abs(v.scheduled_at - at) < duration
            for v in self._visits.values()
        )

    async def is_free(self, doctor: Doctor, at: datetime) -> bool:
        self._loaded_at = None  # перед записью проверяем по свежим данным стенда
        await self._refresh_busy()
        return self._slot_ok(doctor, at) and at in self.schedule.free_slots(doctor.full_name, at.date())

    def _slot_ok(self, doctor: Doctor, at: datetime) -> bool:
        if at < self.clock() + self.min_lead:
            return False
        info = self.schedule.doctor(doctor.full_name)
        if info and info.absence_on(at.date()):
            return False
        return not self._is_busy(doctor, at)

    # --- поиск ---------------------------------------------------------------

    async def find(self, q: SlotQuery) -> list[Slot]:
        await self._refresh_busy()
        today = self.clock().date()
        horizon_end = today + timedelta(days=self.horizon_days)
        doctors = {d.id: d for d in await self.registry.list_doctors() if d.id in q.doctor_ids and d.is_active}

        if q.day:
            days = [q.day] if today <= q.day <= horizon_end else []
        else:
            start = max(q.date_from or today, today)
            end = min(q.date_to or horizon_end, horizon_end)
            days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
        days = [d for d in days if d.weekday() not in q.exclude_days]

        candidates: list[Slot] = []
        for day in days:
            for doctor in doctors.values():
                for at in self.schedule.free_slots(doctor.full_name, day):
                    if q.part:
                        lo, hi = PARTS[q.part]
                        if not lo <= at.time() < hi:
                            continue
                    if self._slot_ok(doctor, at):
                        candidates.append(Slot(doctor, at, self.room_of(doctor)))
        if not candidates:
            return []

        if q.at is not None:
            # Ближайшие к желаемому времени — в первый день, где есть время
            first_day = min(s.at.date() for s in candidates)
            same_day = [s for s in candidates if s.at.date() == first_day]
            target = datetime.combine(first_day, q.at)
            same_day.sort(key=lambda s: (abs(s.at - target), s.at))
            return _unique_times(same_day, q.limit)
        if q.late:
            candidates.sort(key=lambda s: (s.at.date(), -s.at.hour * 60 - s.at.minute))
        else:
            candidates.sort(key=lambda s: s.at)
        if q.day:
            return _unique_times(candidates, q.limit)
        # Без конкретного дня — варианты в разные дни, как в эталонных диалогах регламента
        return _spread_by_day(candidates, q.limit)


def _unique_times(slots: list[Slot], limit: int) -> list[Slot]:
    out: list[Slot] = []
    seen: set[datetime] = set()
    for s in slots:
        if s.at not in seen:
            out.append(s)
            seen.add(s.at)
        if len(out) == limit:
            break
    return out


def _spread_by_day(slots: list[Slot], limit: int) -> list[Slot]:
    out: list[Slot] = []
    days: set[date] = set()
    for s in slots:
        if s.at.date() not in days:
            out.append(s)
            days.add(s.at.date())
        if len(out) == limit:
            return out
    for s in _unique_times(slots, len(slots)):
        if s not in out:
            out.append(s)
        if len(out) == limit:
            break
    return sorted(out, key=lambda s: s.at)
