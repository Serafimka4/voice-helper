"""Сборка ассистента из настроек окружения (.env)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from .dialog import DialogManager
from .fake_registry import FakeRegistry
from .kb import KnowledgeBase
from .llm import DEFAULT_BASE_URL, DEFAULT_MODEL, LLMInterpreter, LLMSettings
from .models import Clock, stand_now
from .registry import Registry, RegistryClient
from .schedule import Schedule
from .slots import SlotService
from .storage import Storage

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Settings:
    registry_base_url: str
    registry_token: str
    registry_rate_limit: int
    schedule_path: Path
    kb_path: Path
    db_path: str
    llm: LLMSettings | None = None

    @classmethod
    def from_env(cls) -> Settings:
        load_dotenv(ROOT / ".env")
        return cls(
            registry_base_url=os.getenv("REGISTRY_BASE_URL", "https://crm-med-mock.dbi-testing.ru/api/v1"),
            registry_token=os.getenv("REGISTRY_TOKEN", ""),
            registry_rate_limit=int(os.getenv("REGISTRY_RATE_LIMIT", "10")),
            schedule_path=ROOT / (os.getenv("SCHEDULE_PATH") or "data/schedule.json"),
            kb_path=ROOT / (os.getenv("KNOWLEDGE_BASE_PATH") or "data/knowledge_base.json"),
            db_path=str(ROOT / (os.getenv("DB_PATH") or "data/voice_helper.sqlite")),
            llm=LLMSettings(
                api_key=os.environ["CLAUDEHUB_API_KEY"],
                base_url=os.getenv("CLAUDEHUB_BASE_URL") or DEFAULT_BASE_URL,
                model=os.getenv("LLM_MODEL") or DEFAULT_MODEL,
            ) if os.getenv("CLAUDEHUB_API_KEY") else None,
        )


@dataclass
class Assistant:
    dialog: DialogManager
    registry: Registry
    storage: Storage
    slots: SlotService


def build(settings: Settings | None = None, *, fake: bool | None = None, clock: Clock = stand_now,
          db_path: str | None = None, use_llm: bool = True) -> Assistant:
    settings = settings or Settings.from_env()
    schedule = Schedule.load(settings.schedule_path)
    use_fake = fake if fake is not None else not settings.registry_token
    registry: Registry
    if use_fake:
        registry = FakeRegistry.from_schedule(schedule, clock)
    else:
        registry = RegistryClient(settings.registry_base_url, settings.registry_token,
                                  rate_limit=settings.registry_rate_limit)
    storage = Storage(db_path or settings.db_path)
    slots = SlotService(registry, schedule, clock)
    kb = KnowledgeBase.load(settings.kb_path)
    llm = None
    if use_llm and settings.llm:
        llm = LLMInterpreter(settings.llm, [d.specialty for d in schedule.doctors])
    return Assistant(DialogManager(registry, schedule, slots, kb, storage, clock, llm), registry, storage, slots)
