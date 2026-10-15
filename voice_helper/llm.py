"""Запасной разбор реплик через Claude (шлюз ClaudeHub, Anthropic-совместимый API).

Роль LLM узкая и безопасная:
- вызывается, только когда правила (nlu.py) не поняли реплику;
- не отвечает пациенту и ничего не делает в МИС — лишь переписывает реплику в каноническую
  форму («тот, что попозже» → «последний»), которую затем снова разбирают правила;
- проверка красных флагов уже прошла по исходной реплике, ответ модели её не отменяет;
- персональные данные в модель не уходят: шаги с ФИО, адресом и жалобами пропускаются,
  телефоны и полные даты маскируются (152-ФЗ). ClaudeHub — внешний облачный сервис, поэтому
  в контуре заказчика его заменяет локальная модель за тем же Anthropic-совместимым API.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

import anthropic

log = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.claudehub.fun"
DEFAULT_MODEL = "claude-opus-5-5"

# Шаги, где пациент диктует персональные данные или описывает состояние, — в LLM не отправляем
PII_STEPS = {"identify", "home_address", "home_reason", "rem_who"}

SYSTEM_PROMPT = """Ты — модуль нормализации реплик в голосовом помощнике регистратуры поликлиники.
Тебе дают вопрос, который помощник задал пациенту, шаг диалога и распознанную речью реплику пациента.
Перепиши реплику в короткую каноническую фразу на русском, которую поймёт простой разборщик по правилам.

Канонические формы:
- цель звонка: «записаться к <специальность>», «перенести запись», «отменить запись», «на когда я записан»,
  «вызвать врача на дом», «когда принимает <специальность>», «до скольки вы работаете», «соедините с оператором»;
- согласие и отказ: «да» / «нет»;
- выбор из предложенных вариантов: «первый», «второй», «третий», «последний»;
- дата и время: «завтра», «послезавтра», «в понедельник», «на следующей неделе», «после обеда», «утром»,
  «вечером», «в 10:30», «попозже»; можно сочетать: «в понедельник после обеда»;
- выбор между переносом и отменой: «перенести» / «отменить».
Специальности бери только из списка: {specialties}.

Правила:
- Не отвечай пациенту, не давай советов, не оценивай симптомы. Только переписывай.
- Не выдумывай: если смысл неясен или реплика не про запись в поликлинику, верни understood=false.
- Если в реплике есть просьба соединить с человеком, жалоба или описание плохого самочувствия —
  сохрани это словами пациента в canonical, ничего не смягчая.
Ответ — строго JSON по схеме."""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "understood": {"type": "boolean"},
        "canonical": {"type": "string"},
    },
    "required": ["understood", "canonical"],
    "additionalProperties": False,
}

_PHONE = re.compile(r"\+?\d[\d\s()\-]{8,}\d")
_FULL_DATE = re.compile(r"\b\d{1,2}[./\s-]\d{1,2}[./\s-]\d{2,4}\b")
_YEAR_DATE = re.compile(
    r"\b\d{1,2}\s+(январ|феврал|март|апрел|ма[йя]|июн|июл|август|сентябр|октябр|ноябр|декабр)\w*\s+\d{2,4}\b",
    re.IGNORECASE,
)


def mask_pii(text: str) -> str:
    """Телефоны и полные даты (вероятно, даты рождения) заменяются метками."""
    text = _PHONE.sub("[ТЕЛЕФОН]", text)
    text = _FULL_DATE.sub("[ДАТА]", text)
    return _YEAR_DATE.sub("[ДАТА]", text)


@dataclass
class LLMSettings:
    api_key: str
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    timeout: float = 8.0  # голосовой диалог: дольше ждать нельзя, при таймауте работают правила


class LLMInterpreter:
    def __init__(self, settings: LLMSettings, specialties: list[str]):
        self.settings = settings
        self.client = anthropic.AsyncAnthropic(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=settings.timeout,
            max_retries=0,  # повтор = ещё несколько секунд тишины в трубке
        )
        self.system = SYSTEM_PROMPT.format(specialties=", ".join(sorted(set(specialties))))

    async def rewrite(self, text: str, step: str, last_prompt: str) -> str | None:
        """Каноническая форма реплики или None, если модель не поняла / недоступна."""
        if step in PII_STEPS:
            return None
        user = (f"Шаг диалога: {step}\nВопрос помощника: «{last_prompt}»\n"
                f"Реплика пациента: «{mask_pii(text)}»")
        try:
            response = await self.client.messages.create(
                model=self.settings.model,
                max_tokens=1024,
                system=self.system,
                output_config={"effort": "low", "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
                messages=[{"role": "user", "content": user}],
            )
        except anthropic.APIConnectionError as e:
            log.warning("LLM недоступна: %s", e)
            return None
        except anthropic.RateLimitError:
            log.warning("LLM: превышен лимит запросов")
            return None
        except anthropic.APIStatusError as e:
            log.warning("LLM вернула ошибку %s: %s", e.status_code, e.message)
            return None
        if response.stop_reason in ("refusal", "max_tokens"):
            return None
        raw = next((b.text for b in response.content if b.type == "text"), "")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            log.warning("LLM вернула не JSON: %r", raw[:200])
            return None
        if not data.get("understood") or not str(data.get("canonical", "")).strip():
            return None
        return str(data["canonical"]).strip()
