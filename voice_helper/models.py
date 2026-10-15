"""Модели данных стенда «Регистратура» и работа со временем стенда.

Стенд живёт в GMT+4 и передаёт даты без смещения, поэтому внутри ядра все
datetime «наивные» и означают местное время стенда.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable

STAND_TZ = timezone(timedelta(hours=4))
DATE_FMT = "%d.%m.%Y"
DATETIME_FMT = "%d.%m.%Y %H:%M"

Clock = Callable[[], datetime]


def stand_now() -> datetime:
    """Текущее время стенда (GMT+4) без tzinfo, с точностью до минуты."""
    return datetime.now(STAND_TZ).replace(tzinfo=None, second=0, microsecond=0)


def fmt_date(d: date) -> str:
    return d.strftime(DATE_FMT)


def fmt_datetime(dt: datetime) -> str:
    return dt.strftime(DATETIME_FMT)


def parse_date(s: str) -> date:
    return datetime.strptime(s.strip(), DATE_FMT).date()


def parse_datetime(s: str) -> datetime:
    return datetime.strptime(s.strip(), DATETIME_FMT)


def normalize_name(s: str) -> str:
    """Ключ для сопоставления ФИО: регистр, ё/е и лишние пробелы не важны."""
    return " ".join(s.lower().replace("ё", "е").split())


class VisitStatus:
    PLANNED = "Запланирована"
    CANCELLED = "Отменена"
    DONE = "Завершена"


@dataclass(frozen=True)
class Doctor:
    id: int
    full_name: str
    specialty: str
    is_active: bool

    @classmethod
    def from_api(cls, d: dict[str, Any]) -> Doctor:
        return cls(d["id"], d["full_name"], d["specialty"], d["is_active"])

    @property
    def last_name(self) -> str:
        return self.full_name.split()[0]


@dataclass(frozen=True)
class Client:
    id: int
    last_name: str
    first_name: str
    middle_name: str
    birth_date: date

    @classmethod
    def from_api(cls, d: dict[str, Any]) -> Client:
        return cls(
            d["id"],
            d["last_name"],
            d["first_name"],
            d.get("middle_name") or "",
            parse_date(d["birth_date"]),
        )

    @property
    def polite_name(self) -> str:
        """Обращение по имени-отчеству, как требует регламент (3.1)."""
        return f"{self.first_name} {self.middle_name}".strip()


@dataclass(frozen=True)
class Visit:
    id: int
    client_id: int
    doctor: Doctor
    scheduled_at: datetime
    status: str
    created_at: datetime | None = None

    @classmethod
    def from_api(cls, d: dict[str, Any]) -> Visit:
        return cls(
            d["id"],
            d["client_id"],
            Doctor.from_api(d["doctor"]),
            parse_datetime(d["scheduled_at"]),
            d["status"],
            parse_datetime(d["created_at"]) if d.get("created_at") else None,
        )

    @property
    def is_planned(self) -> bool:
        return self.status == VisitStatus.PLANNED


@dataclass(frozen=True)
class Escalation:
    id: int
    client_id: int
    description: str

    @classmethod
    def from_api(cls, d: dict[str, Any]) -> Escalation:
        return cls(d["id"], d["client_id"], d["description"])
