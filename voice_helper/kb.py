"""База знаний для справочных ответов (FAQ регламента, раздел 6).

Сейчас — поиск по регулярным выражениям: ответы короткие и должны звучать дословно.
Для формулировок, которые не ловятся шаблонами, сюда же можно добавить семантический
поиск (multilingual-e5 из старого проекта) — интерфейс search() останется тем же.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .nlu import Utterance


@dataclass(frozen=True)
class KBEntry:
    id: str
    answer: str
    patterns: tuple[re.Pattern, ...]
    offer: str | None = None      # book / reschedule — следующий шаг на «да»
    specialty: str | None = None  # специальность для записи по предложению
    route: str | None = None      # правило маршрутизации вместо ответа


class KnowledgeBase:
    def __init__(self, entries: list[KBEntry]):
        self.entries = entries

    @classmethod
    def load(cls, path: str | Path) -> KnowledgeBase:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        entries = [
            KBEntry(
                id=e["id"],
                answer=e["answer"],
                patterns=tuple(re.compile(p) for p in e["patterns"]),
                offer=e.get("offer"),
                specialty=e.get("specialty"),
                route=e.get("route"),
            )
            for e in data["entries"]
        ]
        return cls(entries)

    def search(self, u: Utterance) -> KBEntry | None:
        for e in self.entries:
            if any(p.search(u.words) or p.search(u.text) for p in e.patterns):
                return e
        return None
