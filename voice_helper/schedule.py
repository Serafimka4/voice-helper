"""Расписание врачей из файла кейса.

Стенд расписание не хранит, поэтому источник слотов — Excel из «Материалов кейса»:
листы «Врачи», «График», «Недоступность» и «Слоты» (статусы «Свободен»/«Записан»…).
Excel читается ~6 секунд, поэтому один раз импортируется в компактный JSON:

    uv run python -m voice_helper.schedule data/case/schedule.xlsx data/schedule.json
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time
from pathlib import Path

from .models import normalize_name

WEEKDAYS_SHORT = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
FREE = "Свободен"


@dataclass
class Absence:
    date_from: date
    date_to: date
    kind: str  # Отпуск / Больничный / Обучение — пациенту причину не называем (С-07)


@dataclass
class DoctorInfo:
    full_name: str
    specialty: str
    department: str
    patients: str  # «Взрослые 18+», «Дети 0–17», …
    room: int | None
    slot_minutes: int
    weekly: dict[int, str] = field(default_factory=dict)  # 0=Пн → «08:00–14:00»
    absences: list[Absence] = field(default_factory=list)

    @property
    def key(self) -> str:
        return normalize_name(self.full_name)

    def absence_on(self, day: date) -> Absence | None:
        return next((a for a in self.absences if a.date_from <= day <= a.date_to), None)


class Schedule:
    def __init__(self, doctors: list[DoctorInfo], free_slots: dict[str, dict[date, list[time]]]):
        self._doctors = {d.key: d for d in doctors}
        self._free = free_slots  # ключ — normalize_name(ФИО)

    @property
    def doctors(self) -> list[DoctorInfo]:
        return list(self._doctors.values())

    def doctor(self, full_name: str) -> DoctorInfo | None:
        return self._doctors.get(normalize_name(full_name))

    def free_slots(self, full_name: str, day: date) -> list[datetime]:
        """Свободные по Excel слоты врача на день (без учёта записей команды на стенде)."""
        times = self._free.get(normalize_name(full_name), {}).get(day, [])
        return [datetime.combine(day, t) for t in times]

    # --- сериализация --------------------------------------------------------

    def to_json(self) -> dict:
        return {
            "doctors": [
                {**asdict(d), "absences": [[a.date_from.isoformat(), a.date_to.isoformat(), a.kind] for a in d.absences]}
                for d in self._doctors.values()
            ],
            "free_slots": {
                key: {day.isoformat(): [t.strftime("%H:%M") for t in times] for day, times in days.items()}
                for key, days in self._free.items()
            },
        }

    @classmethod
    def from_json(cls, data: dict) -> Schedule:
        doctors = []
        for d in data["doctors"]:
            absences = [Absence(date.fromisoformat(a), date.fromisoformat(b), k) for a, b, k in d.pop("absences")]
            weekly = {int(k): v for k, v in d.pop("weekly").items()}
            doctors.append(DoctorInfo(**d, weekly=weekly, absences=absences))
        free = {
            key: {date.fromisoformat(day): [time.fromisoformat(t) for t in times] for day, times in days.items()}
            for key, days in data["free_slots"].items()
        }
        return cls(doctors, free)

    @classmethod
    def load(cls, path: str | Path) -> Schedule:
        return cls.from_json(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), ensure_ascii=False), encoding="utf-8")


def _as_date(v) -> date:
    return v.date() if isinstance(v, datetime) else v


def import_excel(path: str | Path, since: date | None = None) -> Schedule:
    """Читает Excel кейса. Свободные слоты берутся начиная с since (по умолчанию — сегодня)."""
    import openpyxl

    since = since or date.today()
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)

    def rows(sheet: str):
        it = wb[sheet].iter_rows(values_only=True)
        header = [str(h).strip() if h is not None else "" for h in next(it)]
        for r in it:
            if any(v is not None for v in r):
                yield dict(zip(header, r))

    doctors: dict[str, DoctorInfo] = {}
    for r in rows("Врачи"):
        info = DoctorInfo(
            full_name=r["ФИО"],
            specialty=r["Специальность (профиль)"],
            department=r["Отделение"],
            patients=r["Пациенты"],
            room=int(r["Кабинет"]) if r["Кабинет"] else None,
            slot_minutes=int(r["Слот, мин"]),
        )
        doctors[info.key] = info

    for r in rows("График"):
        info = doctors.get(normalize_name(r["ФИО врача"]))
        if info:
            info.weekly = {i: r[d] for i, d in enumerate(WEEKDAYS_SHORT) if r.get(d) and r[d] != "—"}

    for r in rows("Недоступность"):
        info = doctors.get(normalize_name(r["ФИО врача"]))
        if info:
            info.absences.append(Absence(_as_date(r["С"]), _as_date(r["По"]), r["Тип"]))

    free: dict[str, dict[date, list[time]]] = {}
    for r in rows("Слоты"):
        if r["Статус"] != FREE:
            continue
        day = _as_date(r["Дата"])
        if day < since:
            continue
        free.setdefault(normalize_name(r["ФИО врача"]), {}).setdefault(day, []).append(r["Начало"])
    for days in free.values():
        for times in days.values():
            times.sort()

    return Schedule(list(doctors.values()), free)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("usage: python -m voice_helper.schedule <schedule.xlsx> <schedule.json> [since YYYY-MM-DD]")
    since_arg = date.fromisoformat(sys.argv[3]) if len(sys.argv) > 3 else None
    schedule = import_excel(sys.argv[1], since_arg)
    schedule.save(sys.argv[2])
    n_slots = sum(len(t) for days in schedule._free.values() for t in days.values())
    print(f"Врачей: {len(schedule.doctors)}, свободных слотов: {n_slots} → {sys.argv[2]}")
