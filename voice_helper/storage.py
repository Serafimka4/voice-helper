"""Локальное хранилище ассистента (в контуре поликлиники, SQLite).

Стенд не хранит телефоны, подтверждения визитов и результаты звонков — это наша сторона:
- phones: номер пациента для исходящих напоминаний и отказ от обзвона (OPT_OUT);
- calls/turns: журнал звонков с кодами результата (Приложение Г) для метрик раздела 9;
- reminders: статус напоминаний (CONFIRMED/RESCHEDULED/…) — у записи на стенде нет «Подтверждена»;
- transfers: карточки перевода (Приложение В);
- waitlist: лист ожидания (С-11).
Если пациент против записи разговора, реплики в turns не сохраняются (раздел 3.4).
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS phones (
    client_id INTEGER PRIMARY KEY, phone TEXT NOT NULL, opt_out INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS calls (
    call_id TEXT PRIMARY KEY, direction TEXT, phone TEXT, client_id INTEGER, started_at TEXT,
    ended_at TEXT, intent TEXT, result TEXT, rule TEXT, recording INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS turns (
    call_id TEXT, at TEXT, role TEXT, text TEXT
);
CREATE TABLE IF NOT EXISTS reminders (
    visit_id INTEGER PRIMARY KEY, status TEXT, attempts INTEGER NOT NULL DEFAULT 0, updated_at TEXT
);
CREATE TABLE IF NOT EXISTS transfers (
    id INTEGER PRIMARY KEY AUTOINCREMENT, call_id TEXT, at TEXT, card TEXT, escalation_id INTEGER
);
CREATE TABLE IF NOT EXISTS waitlist (
    id INTEGER PRIMARY KEY AUTOINCREMENT, client_id INTEGER, specialty TEXT, preferences TEXT,
    phone TEXT, created_at TEXT
);
"""


class Storage:
    def __init__(self, path: str | Path = ":memory:"):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    def _now(self) -> str:
        return datetime.now().isoformat(timespec="seconds")

    # --- звонки --------------------------------------------------------------

    def start_call(self, call_id: str, direction: str, phone: str | None) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO calls (call_id, direction, phone, started_at) VALUES (?, ?, ?, ?)",
            (call_id, direction, phone, self._now()),
        )
        self.db.commit()

    def recent_call(self, phone: str | None, minutes: int = 30) -> bool:
        """Повторный звонок с того же номера в течение 30 минут (приветствие, раздел 3.3)."""
        if not phone:
            return False
        since = (datetime.now() - timedelta(minutes=minutes)).isoformat(timespec="seconds")
        row = self.db.execute(
            "SELECT 1 FROM calls WHERE phone = ? AND direction = 'inbound' AND started_at >= ?", (phone, since)
        ).fetchone()
        return row is not None

    def log_turn(self, call_id: str, role: str, text: str) -> None:
        row = self.db.execute("SELECT recording FROM calls WHERE call_id = ?", (call_id,)).fetchone()
        if row is not None and not row["recording"]:
            return
        self.db.execute("INSERT INTO turns VALUES (?, ?, ?, ?)", (call_id, self._now(), role, text))
        self.db.commit()

    def disable_recording(self, call_id: str) -> None:
        self.db.execute("UPDATE calls SET recording = 0 WHERE call_id = ?", (call_id,))
        self.db.execute("DELETE FROM turns WHERE call_id = ?", (call_id,))
        self.db.commit()

    def finish_call(self, call_id: str, result: str, intent: str | None = None,
                    rule: str | None = None, client_id: int | None = None) -> None:
        self.db.execute(
            "UPDATE calls SET ended_at = ?, result = ?, intent = COALESCE(?, intent), rule = COALESCE(?, rule), "
            "client_id = COALESCE(?, client_id) WHERE call_id = ?",
            (self._now(), result, intent, rule, client_id, call_id),
        )
        self.db.commit()

    def call(self, call_id: str) -> dict[str, Any] | None:
        row = self.db.execute("SELECT * FROM calls WHERE call_id = ?", (call_id,)).fetchone()
        return dict(row) if row else None

    def turns(self, call_id: str) -> list[dict[str, Any]]:
        return [dict(r) for r in self.db.execute("SELECT * FROM turns WHERE call_id = ? ORDER BY rowid", (call_id,))]

    # --- телефоны ------------------------------------------------------------

    def save_phone(self, client_id: int, phone: str) -> None:
        self.db.execute(
            "INSERT INTO phones (client_id, phone) VALUES (?, ?) "
            "ON CONFLICT(client_id) DO UPDATE SET phone = excluded.phone",
            (client_id, phone),
        )
        self.db.commit()

    def phone_of(self, client_id: int) -> str | None:
        row = self.db.execute("SELECT phone FROM phones WHERE client_id = ?", (client_id,)).fetchone()
        return row["phone"] if row else None

    def set_opt_out(self, client_id: int) -> None:
        self.db.execute("UPDATE phones SET opt_out = 1 WHERE client_id = ?", (client_id,))
        self.db.commit()

    def is_opted_out(self, client_id: int) -> bool:
        row = self.db.execute("SELECT opt_out FROM phones WHERE client_id = ?", (client_id,)).fetchone()
        return bool(row and row["opt_out"])

    # --- напоминания ---------------------------------------------------------

    def reminder(self, visit_id: int) -> dict[str, Any] | None:
        row = self.db.execute("SELECT * FROM reminders WHERE visit_id = ?", (visit_id,)).fetchone()
        return dict(row) if row else None

    def set_reminder(self, visit_id: int, status: str, attempt: bool = False) -> None:
        self.db.execute(
            "INSERT INTO reminders (visit_id, status, attempts, updated_at) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(visit_id) DO UPDATE SET status = excluded.status, "
            "attempts = reminders.attempts + ?, updated_at = excluded.updated_at",
            (visit_id, status, int(attempt), self._now(), int(attempt)),
        )
        self.db.commit()

    # --- переводы и лист ожидания --------------------------------------------

    def save_transfer(self, call_id: str, card: dict[str, Any], escalation_id: int | None) -> None:
        self.db.execute(
            "INSERT INTO transfers (call_id, at, card, escalation_id) VALUES (?, ?, ?, ?)",
            (call_id, self._now(), json.dumps(card, ensure_ascii=False, default=str), escalation_id),
        )
        self.db.commit()

    def transfers(self, call_id: str) -> list[dict[str, Any]]:
        rows = self.db.execute("SELECT * FROM transfers WHERE call_id = ? ORDER BY id", (call_id,))
        return [{**dict(r), "card": json.loads(r["card"])} for r in rows]

    def add_to_waitlist(self, client_id: int, specialty: str, preferences: str, phone: str | None) -> None:
        self.db.execute(
            "INSERT INTO waitlist (client_id, specialty, preferences, phone, created_at) VALUES (?, ?, ?, ?, ?)",
            (client_id, specialty, preferences, phone, self._now()),
        )
        self.db.commit()
