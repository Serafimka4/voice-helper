"""Диалоговый менеджер голосового ассистента регистратуры.

Реализует сценарии регламента РГ-ТЕЛ-01:
  С-01/С-02 запись (прямая и по направлению), С-03 перенос, С-04 отмена, С-05 проверка записи,
  С-06 вызов на дом (полуавтомат: заявка через /escalations), С-07 график врача, С-10 напоминание,
  С-11 лист ожидания, С-14 экстренная ситуация, С-15 жалоба, FAQ раздела 6.

Принцип: распознавание (nlu) и границы (guard) — правилами; решения о записи принимает код,
а не модель. Каждая реплика пациента сначала проходит проверку красных флагов.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

import httpx

from . import guard
from .guard import Route
from .kb import KnowledgeBase
from .llm import PII_STEPS
from .models import Client, Clock, Doctor, Visit, VisitStatus, fmt_date, fmt_datetime, normalize_name
from .nlu import (
    ADULT_TO_CHILD, CHILD_RE, CHILD_TO_ADULT, DIRECT_SPECIALTIES, FIRST_PERSON_RE, PAID_ONLY_SPECIALTIES,
    RELATIVE_ADULT_RE, DatePref, Person, SpecialtyMatcher, TimePref, Utterance, age_on, choice_index,
    detect_intent, extract_date_pref, extract_person, extract_time_pref, match_doctors, yes_no,
)
from .numbers import hours_range_words, say_date, say_slot_full, say_slot_short, time_words, weekdays_words
from .registry import Registry, RegistryError
from .schedule import Schedule
from .slots import Slot, SlotQuery, SlotService
from .storage import Storage

log = logging.getLogger(__name__)

GREETING = "Здравствуйте! Регистратура поликлиники номер семь, виртуальный помощник. Разговор записывается. Чем могу помочь?"
GREETING_REPEAT = ("Здравствуйте! Регистратура поликлиники номер семь, виртуальный помощник. "
                   "Вы недавно звонили — продолжим тот вопрос или у вас новый?")
ANYTHING_ELSE = "Могу ещё чем-то помочь?"
GOODBYE = "Всего доброго, будьте здоровы!"
ASK_PERSON = "Назовите, пожалуйста, фамилию, имя, отчество и дату рождения{who}."
VISIT_ERRORS = (RegistryError, httpx.HTTPError)


@dataclass
class Reply:
    text: str
    transfer: bool = False       # перевести звонок сотруднику (карточка в card)
    end_call: bool = False
    card: dict[str, Any] | None = None


@dataclass
class CallState:
    call_id: str
    phone: str | None = None
    direction: str = "inbound"
    step: str = "intent"
    intent: str | None = None
    for_child: bool = False
    person: Person = field(default_factory=Person)
    client: Client | None = None
    specialty: str | None = None
    doctor_ids: list[int] = field(default_factory=list)
    referral_ok: bool = False
    date: DatePref = field(default_factory=DatePref)
    time: TimePref = field(default_factory=TimePref)
    visit_hint: DatePref = field(default_factory=DatePref)
    offered: list[Slot] = field(default_factory=list)
    chosen: Slot | None = None
    visits: list[Visit] = field(default_factory=list)
    visit: Visit | None = None
    pending: str | None = None            # book / reschedule / replace / cancel — что подтверждаем
    offer: dict[str, Any] | None = None   # что произойдёт на «да»
    after_identify: str | None = None     # book / visits / home / waitlist
    home: dict[str, str] = field(default_factory=dict)
    id_fails: int = 0
    misses: int = 0
    silence: int = 0
    medical_warned: bool = False
    cancel_offered_reschedule: bool = False
    last_prompt: str = ""
    results: list[str] = field(default_factory=list)
    finished: bool = False
    _missed: bool = False
    _llm_used: bool = False  # LLM вызывается не больше одного раза на реплику

    def reset_task(self) -> None:
        """Новая цель в том же звонке: данные пациента сохраняются, параметры задачи — нет."""
        self.intent = None
        self.specialty = None
        self.doctor_ids = []
        self.referral_ok = False
        self.date, self.time, self.visit_hint = DatePref(), TimePref(), DatePref()
        self.offered, self.chosen, self.visits, self.visit = [], None, [], None
        self.pending = self.offer = self.after_identify = None
        self.home = {}
        self.cancel_offered_reschedule = False


def spoken_specialty(s: str) -> str:
    s = re.sub(r"\s*\(.*\)", "", s).strip().lower()
    return "ЛОР" if s == "лор" else s


def mask_phone(phone: str | None) -> str:
    if not phone:
        return "не определён"
    digits = re.sub(r"\D", "", phone)
    return f"*** {digits[-4:-2]}-{digits[-2:]}" if len(digits) >= 4 else "***"


class DialogManager:
    def __init__(
        self,
        registry: Registry,
        schedule: Schedule,
        slots: SlotService,
        kb: KnowledgeBase,
        storage: Storage,
        clock: Clock,
        llm: Any = None,  # LLMInterpreter или совместимый объект с async rewrite(text, step, last_prompt)
    ):
        self.registry = registry
        self.schedule = schedule
        self.slots = slots
        self.kb = kb
        self.storage = storage
        self.clock = clock
        self.llm = llm
        self._matcher: SpecialtyMatcher | None = None

    @property
    def today(self) -> date:
        return self.clock().date()

    async def matcher(self) -> SpecialtyMatcher:
        if self._matcher is None:
            specialties = [d.specialty for d in await self.registry.list_doctors()]
            specialties += [d.specialty for d in self.schedule.doctors]
            self._matcher = SpecialtyMatcher(specialties)
        return self._matcher

    # =========================================================================
    # Вход
    # =========================================================================

    async def start_inbound(self, call_id: str, phone: str | None = None) -> tuple[CallState, Reply]:
        repeat = self.storage.recent_call(phone)
        self.storage.start_call(call_id, "inbound", phone)
        state = CallState(call_id=call_id, phone=phone)
        return state, self._say(state, GREETING_REPEAT if repeat else GREETING)

    async def handle(self, state: CallState, raw: str) -> Reply:
        self.storage.log_turn(state.call_id, "patient", raw)
        if state.finished:
            return Reply("", end_call=True)
        u = Utterance.parse(raw)
        if not u.words:
            return self._silence(state)
        state.silence = 0
        state._missed = False
        state._llm_used = False
        try:
            reply = await self._dispatch(state, u)
        except VISIT_ERRORS as e:
            log.exception("Ошибка стенда: %s", e)
            reply = await self._route(state, guard.MIS_FAILURE, u)
        if not state._missed:
            state.misses = 0
        self.storage.log_turn(state.call_id, "assistant", reply.text)
        return reply

    async def _dispatch(self, state: CallState, u: Utterance) -> Reply:
        # Красный флаг прерывает любой сценарий (R-01, R-11), в т. ч. исходящий звонок
        if route := guard.check_red_flags(u):
            return await self._route(state, route, u)

        intent = detect_intent(u)
        if intent == "no_recording":
            self.storage.disable_recording(state.call_id)
            return self._say(state, "Хорошо, запись разговора отключена. " + (state.last_prompt or "Чем могу помочь?"))

        if route := guard.check_routing(u):
            if route.transfer:
                return await self._route(state, route, u)
            if route.result:
                state.results.append(route.result)
            if not route.rule:  # «Вы робот?» — честный ответ и предложение соединить (раздел 3.2)
                state.offer = {"kind": "operator", "return_step": state.step, "route": "R-02",
                               "prompt": state.last_prompt}
                state.step = "offer"
                return self._say(state, route.say)
            follow = state.last_prompt if state.step not in ("intent", "anything_else") else ANYTHING_ELSE
            return self._say(state, f"{route.say} {follow}")

        if guard.is_medical_question(u) and state.step not in ("home_reason",):
            if state.medical_warned:
                return await self._route(state, guard.MEDICAL_INSIST, u)
            state.medical_warned = True
            state.offer = {"kind": "book", "specialty": "Терапевт"}
            state.step = "offer"
            return self._say(state, guard.medical_boundary_phrase(u))

        handler = getattr(self, f"_on_{state.step}")
        return await handler(state, u)

    # =========================================================================
    # Общие ответы
    # =========================================================================

    def _say(self, state: CallState, text: str, **kw: Any) -> Reply:
        state.last_prompt = text
        return Reply(text, **kw)

    def _silence(self, state: CallState) -> Reply:
        state.silence += 1
        if state.silence == 1:
            return Reply("Вы меня слышите?")
        return self._finish(state, "Вас не слышно. Перезвоните, пожалуйста.", "DROPPED")

    async def _miss(self, state: CallState, u: Utterance, reprompt: str) -> Reply:
        """Непонимание (3.2): переспросить → предложить оператора → перевести (R-05).

        Перед тем как переспрашивать, реплику один раз пробует переформулировать LLM.
        """
        if self.llm is not None and not state._llm_used and state.step not in PII_STEPS:
            state._llm_used = True
            canonical = await self.llm.rewrite(u.raw, state.step, state.last_prompt)
            if canonical and Utterance.parse(canonical).words != u.words:
                log.info("LLM: «%s» → «%s»", u.raw, canonical)
                return await self._dispatch(state, Utterance.parse(canonical))
        state._missed = True
        state.misses += 1
        if state.misses == 1:
            return self._say(state, f"Извините, не совсем понял. {reprompt}")
        if state.misses == 2:
            state.offer = {"kind": "operator", "return_step": state.step, "prompt": reprompt}
            state.step = "offer"
            return self._say(state, "Похоже, я вас плохо понимаю. Соединить с сотрудником?")
        return await self._route(state, guard.NOT_UNDERSTOOD, u)

    def _finish(self, state: CallState, text: str, result: str | None = None) -> Reply:
        result = result or (state.results[-1] if state.results else "INFO_GIVEN")
        self.storage.finish_call(state.call_id, result, state.intent, client_id=state.client.id if state.client else None)
        state.finished = True
        return Reply(text, end_call=True)

    def _done(self, state: CallState, text: str, result: str) -> Reply:
        """Задача выполнена: зафиксировать результат и спросить, нужно ли что-то ещё."""
        state.results.append(result)
        if state.direction == "outbound":
            return self._finish(state, f"{text} Всего доброго!", result)
        state.step = "anything_else"
        return self._say(state, f"{text} {ANYTHING_ELSE}")

    # =========================================================================
    # Перевод на сотрудника (раздел 7.7, карточка — Приложение В)
    # =========================================================================

    def _operators_available(self) -> bool:
        now = self.clock()
        wd, t = now.weekday(), now.time()
        if wd < 5:
            return datetime.strptime("07:00", "%H:%M").time() <= t < datetime.strptime("20:00", "%H:%M").time()
        if wd == 5:
            return datetime.strptime("08:30", "%H:%M").time() <= t < datetime.strptime("15:00", "%H:%M").time()
        return False

    def _card(self, state: CallState, route: Route, u: Utterance | None) -> dict[str, Any]:
        p = state.client or state.person
        if state.client:
            patient = (f"№ {state.client.id}, {state.client.last_name} {state.client.first_name} "
                       f"{state.client.middle_name}, {fmt_date(state.client.birth_date)}").replace("  ", " ")
        elif state.person.last_name or state.person.birth_date:
            bd = fmt_date(p.birth_date) if p.birth_date else "дата рождения не названа"
            patient = f"не идентифицирован; назвал: {p.last_name or ''} {p.first_name or ''}, {bd}".strip()
        else:
            patient = "не идентифицирован"
        params = {k: v for k, v in {
            "специальность": state.specialty,
            "запись": self._visit_short(state.visit) if state.visit else None,
            "адрес": state.home.get("address"),
            "повод": state.home.get("reason"),
        }.items() if v}
        summary = route.reason + (f": «{u.raw.strip()[:150]}»" if u else "")
        return {
            "id_звонка": state.call_id,
            "время": fmt_datetime(self.clock()),
            "телефон": mask_phone(state.phone),
            "пациент": patient,
            "интент": state.intent or "не определён",
            "параметры": params,
            "причина_перевода": route.rule,
            "приоритет": route.priority,
            "адресат": route.target,
            "резюме": summary[:200],
            "недовольство": route.result == "COMPLAINT",
            "согласие_на_запись": (self.storage.call(state.call_id) or {}).get("recording", 1) == 1,
        }

    async def _escalate(self, state: CallState, description: str) -> int | None:
        """Фиксирует обращение на стенде (POST /escalations). Нужен client_id, поэтому только для
        опознанного пациента; если пациент уже назвал ФИО и дату рождения — карточка заводится."""
        client = state.client
        try:
            if client is None and state.person.complete:
                client = await self._ensure_client(state)
            if client is None:
                return None
            esc = await self.registry.create_escalation(client.id, description[:1000])
            return esc.id
        except VISIT_ERRORS as e:
            log.warning("Не удалось создать заявку на стенде: %s", e)
            return None

    async def _route(self, state: CallState, route: Route, u: Utterance | None) -> Reply:
        card = self._card(state, route, u)
        description = (f"{route.rule} {route.priority} → {route.target}. {card['резюме']}. "
                       f"Интент: {card['интент']}. Параметры: {card['параметры'] or '—'}")
        escalation_id = await self._escalate(state, description)
        self.storage.save_transfer(state.call_id, card, escalation_id)
        state.results.append(route.result)

        if route.rule == "R-01" or route.transfer is False:
            # Экстренная ситуация: рекомендация 103/112 важнее, чем доступность операторов
            if not route.transfer or not self._operators_available():
                text = route.say.split(" Я соединяю")[0]
                return self._finish(state, text, route.result)
            reply = self._finish(state, route.say, route.result)
            return Reply(reply.text, transfer=True, end_call=False, card=card)

        if not self._operators_available():
            if route.rule == "R-11":  # вне часов работы категория Б обрабатывается как А (7.4)
                return self._finish(state, f"Пожалуйста, звоните в скорую помощь: {guard.AMBULANCE}.", "EMERGENCY")
            return self._finish(
                state,
                "Сотрудники регистратуры сейчас не работают. Я записал ваш вопрос — вам перезвонят "
                "на этот номер в рабочее время, после восьми утра.",
                "CALLBACK",
            )
        self.storage.finish_call(state.call_id, route.result, state.intent, route.rule,
                                 state.client.id if state.client else None)
        state.finished = True
        return Reply(route.say, transfer=True, card=card)

    # =========================================================================
    # Шаг: цель обращения
    # =========================================================================

    async def _on_intent(self, state: CallState, u: Utterance) -> Reply:
        intent = detect_intent(u)
        m = await self.matcher()
        spec = m.match(u)
        doctors = match_doctors(u, [d for d in await self.registry.list_doctors()])
        if intent in (None, "book") and (spec or doctors):
            intent = "book"
        if re.search(CHILD_RE, u.words):
            state.for_child = True
        person = extract_person(u, self.today)
        if person.complete:
            state.person.merge(person)

        if intent == "book":
            state.intent = "book"
            self._take_booking_params(state, u, spec, doctors)
            return await self._continue_booking(state, u)
        if intent in ("reschedule", "cancel", "check"):
            state.intent = intent
            if intent == "check" and self._asks_about_other_adult(u):
                return self._done(state, "Сведения о записях я могу сообщить только самому пациенту. "
                                         "Пусть пациент позвонит сам, я подскажу.", "INFO_GIVEN")
            self._take_visit_params(state, u, spec)
            return await self._continue_visits(state, u)
        if intent == "home_visit":
            state.intent = "home_visit"
            return await self._start_home_visit(state, u)
        if intent == "schedule_info":
            state.intent = "schedule_info"
            return await self._schedule_info(state, spec, doctors)
        if intent == "waitlist":
            state.intent = "waitlist"
            state.specialty = spec
            if not spec:
                state.step = "specialty"
                return self._say(state, "К какому врачу поставить вас в лист ожидания?")
            return await self._continue_waitlist(state, u)
        if intent == "gratitude":
            return self._done(state, "Спасибо, обязательно передам.", "GRATITUDE")
        if intent == "goodbye":
            return self._finish(state, GOODBYE)
        if intent == "opt_out" and state.direction == "outbound":
            return await self._on_rem_confirm(state, u)

        if entry := self.kb.search(u):
            return await self._kb_answer(state, entry, u)
        return await self._miss(state, u, "Скажите, пожалуйста, что вы хотите: записаться к врачу, перенести или "
                                          "отменить запись, или узнать что-то о поликлинике?")

    async def _kb_answer(self, state: CallState, entry, u: Utterance) -> Reply:
        if entry.route == "R-10":
            return await self._route(state, guard.Route(
                "R-10", "P3", "М4-П", f"{entry.answer} Соединяю с платным отделением.",
                transfer=True, result="ESCALATED", reason="Платная услуга вне справочника"), u)
        state.intent = state.intent or "info"
        state.results.append("INFO_GIVEN")
        if entry.offer:
            state.offer = {"kind": entry.offer, "specialty": entry.specialty}
            state.step = "offer"
            return self._say(state, entry.answer)
        state.step = "anything_else"
        return self._say(state, f"{entry.answer} {ANYTHING_ELSE}")

    @staticmethod
    def _asks_about_other_adult(u: Utterance) -> bool:
        """Правило «пациент называет первым» и 3.5: о записях взрослого родственника — нельзя."""
        if re.search(FIRST_PERSON_RE, u.words) and not re.search(RELATIVE_ADULT_RE, u.words):
            return False
        return bool(re.search(RELATIVE_ADULT_RE, u.words) or re.search(r"\bзаписан[аы]?\b", u.words))

    # =========================================================================
    # Шаг: «да/нет» на предложение
    # =========================================================================

    async def _on_offer(self, state: CallState, u: Utterance) -> Reply:
        offer = state.offer or {}
        answer = yes_no(u)
        kind = offer.get("kind")
        if kind == "operator":
            if answer is True:
                route = guard.Route("R-02", "P2", "М2", "Соединяю с сотрудником.", transfer=True,
                                    result="ESCALATED", reason="Пациент попросил сотрудника") \
                    if offer.get("route") == "R-02" else guard.NOT_UNDERSTOOD
                return await self._route(state, route, u)
            if answer is False:
                state.step = offer.get("return_step", "intent")
                return self._say(state, "Хорошо. " + (offer.get("prompt") or "Чем могу помочь?"))
            state.step = offer.get("return_step", "intent")
            return await self._dispatch(state, u)

        if answer is True or (kind == "book" and await self._mentions_booking_params(u)):
            state.offer = None
            if kind == "book":
                if offer.get("specialty") and not state.specialty:
                    state.specialty = (await self.matcher()).canonical(offer["specialty"]) or offer["specialty"]
                if offer.get("doctor_ids"):
                    state.doctor_ids = offer["doctor_ids"]
                state.intent = "book"
                m = await self.matcher()
                self._take_booking_params(state, u, m.match(u), [])
                return await self._continue_booking(state, u)
            if kind == "reschedule":
                state.intent = "reschedule"
                if offer.get("visit"):
                    state.visit = offer["visit"]
                return await self._continue_visits(state, u)
            if kind == "waitlist":
                state.intent = "waitlist"
                return await self._continue_waitlist(state, u)
            if kind == "cancel":
                state.intent = "cancel"
                return await self._confirm_cancel(state)
        if answer is False:
            state.offer = None
            if kind == "cancel" and state.direction == "outbound":
                return self._finish(state, "Хорошо, запись остаётся в силе. До свидания!", "UNCONFIRMED")
            state.step = "anything_else"
            return self._say(state, f"Хорошо. {ANYTHING_ELSE}")
        # Не ответ на вопрос — возможно, новая цель
        state.step = "intent"
        return await self._on_intent(state, u)

    async def _mentions_booking_params(self, u: Utterance) -> bool:
        return bool((await self.matcher()).match(u)) or not extract_date_pref(u, self.today).empty

    async def _on_anything_else(self, state: CallState, u: Utterance) -> Reply:
        if yes_no(u) is False or detect_intent(u) == "goodbye":
            return self._finish(state, GOODBYE)
        if yes_no(u) is True and len(u.words.split()) <= 2:
            state.step = "intent"
            return self._say(state, "Слушаю вас.")
        state.reset_task()
        state.step = "intent"
        return await self._on_intent(state, u)

    # =========================================================================
    # Идентификация (раздел 3.5)
    # =========================================================================

    def _ask_person(self, state: CallState, after: str) -> Reply:
        state.after_identify = after
        state.step = "identify"
        who = " ребёнка" if state.for_child else (" пациента" if after == "home" else "")
        return self._say(state, ASK_PERSON.format(who=who))

    async def _on_identify(self, state: CallState, u: Utterance) -> Reply:
        person = extract_person(u, self.today)
        state.person.merge(person)
        p = state.person
        if not person.last_name and not person.birth_date:
            return await self._miss(state, u, ASK_PERSON.format(who=""))
        if not (p.last_name and p.first_name):
            return self._say(state, "Назовите, пожалуйста, фамилию и имя.")
        if not p.birth_date:
            return self._say(state, "Назовите, пожалуйста, дату рождения.")
        after = state.after_identify
        if after == "visits":
            return await self._continue_visits(state, u)
        if after == "home":
            return await self._continue_home_visit(state, u)
        if after == "waitlist":
            return await self._continue_waitlist(state, u)
        return await self._continue_booking(state, u)

    async def _ensure_client(self, state: CallState) -> Client:
        p = state.person
        client = await self.registry.ensure_client(p.last_name, p.first_name, p.birth_date, p.middle_name)
        state.client = client
        if state.phone:  # для ребёнка — телефон законного представителя
            self.storage.save_phone(client.id, state.phone)
        return client

    async def _find_client(self, state: CallState, u: Utterance) -> Client | Reply:
        p = state.person
        found = await self.registry.find_clients(p.last_name, p.first_name, p.birth_date)
        if found:
            state.client = found[0]
            return found[0]
        state.id_fails += 1
        if state.id_fails >= 2:
            return await self._route(state, guard.NOT_IDENTIFIED, u)
        state.person = Person()
        state.step = "identify"
        return self._say(state, "Не нашёл вас в базе. Повторите, пожалуйста, фамилию, имя и дату рождения.")

    def _polite(self, state: CallState) -> str:
        if state.client and not state.for_child:
            return state.client.polite_name
        if state.person.first_name and not state.for_child:
            return f"{state.person.first_name} {state.person.middle_name or ''}".strip()
        return ""

    # =========================================================================
    # С-01/С-02. Запись на приём
    # =========================================================================

    def _take_booking_params(self, state: CallState, u: Utterance, spec: str | None, doctors: list[Doctor]) -> None:
        if spec:
            state.specialty = spec
        if doctors:
            if spec:
                doctors = [d for d in doctors if normalize_name(d.specialty) == normalize_name(spec)] or doctors
            state.doctor_ids = [d.id for d in doctors]
            state.specialty = state.specialty or doctors[0].specialty
        dp = extract_date_pref(u, self.today)
        if not dp.empty:
            state.date = dp
        tp = extract_time_pref(u)
        if not tp.empty:
            state.time = tp

    async def _continue_booking(self, state: CallState, u: Utterance) -> Reply:
        if not state.specialty:
            state.step = "specialty"
            return self._say(state, "К какому врачу вас записать?")
        if normalize_name(state.specialty) in PAID_ONLY_SPECIALTIES:
            return await self._route(state, guard.PAID_ONLY, u)
        if not state.person.complete and not state.client:
            return self._ask_person(state, "book")
        if state.client is None:
            await self._ensure_client(state)
        adapt_note = await self._adapt_specialty_to_age(state)
        if not await self._specialty_doctors(state):
            return await self._route(state, guard.OUT_OF_SCOPE, u)
        if normalize_name(state.specialty) not in DIRECT_SPECIALTIES and not state.referral_ok:
            state.step = "referral"
            return self._say(state, f"{adapt_note}К {self._dative(state.specialty)} запись идёт по направлению "
                                    "лечащего врача. У вас есть электронное направление?")
        # Не более одной активной записи к одной специальности (С-01)
        existing = [v for v in await self.registry.list_visits(client_id=state.client.id, status=VisitStatus.PLANNED)
                    if normalize_name(v.doctor.specialty) == normalize_name(state.specialty)
                    and v.scheduled_at >= self.clock()]
        if existing:
            state.offer = {"kind": "reschedule", "visit": existing[0]}
            state.step = "offer"
            return self._say(state, f"У вас уже есть запись: {self._visit_short(existing[0])}. Хотите её перенести?")
        greet = f"Спасибо, {self._polite(state)}. " if self._polite(state) and state.step == "identify" else ""
        return await self._offer_slots(state, u, prefix=greet + adapt_note)

    async def _adapt_specialty_to_age(self, state: CallState) -> str:
        """Ребёнку — детское отделение, взрослому — взрослое (раздел 2.3)."""
        birth = state.client.birth_date if state.client else state.person.birth_date
        if not birth or not state.specialty:
            return ""
        m = await self.matcher()
        key = normalize_name(state.specialty)
        if age_on(birth, self.today) < 18 and key in ADULT_TO_CHILD:
            child = m.canonical(ADULT_TO_CHILD[key])
            if child:
                state.specialty, state.doctor_ids = child, []
                state.for_child = True
                return f"Для детей приём ведёт {spoken_specialty(child)}. "
        if age_on(birth, self.today) >= 18 and key in CHILD_TO_ADULT:
            adult = m.canonical(CHILD_TO_ADULT[key])
            if adult:
                state.specialty, state.doctor_ids = adult, []
                return f"Взрослых принимает {spoken_specialty(adult)}. "
        return ""

    async def _specialty_doctors(self, state: CallState) -> list[Doctor]:
        doctors = await self.slots.active_doctors(state.specialty)
        if state.doctor_ids:
            chosen = [d for d in doctors if d.id in state.doctor_ids]
            if chosen:
                return chosen
        return doctors

    @staticmethod
    def _dative(specialty: str) -> str:
        """«к терапевту», «к детскому неврологу», «к врачу УЗИ-диагностики», «к ЛОРу»."""
        s = spoken_specialty(specialty)
        if s == "ЛОР":
            return "ЛОРу"
        if s.startswith("врач "):
            return "врачу " + s[5:]
        words = s.split()
        out = []
        for w in words:
            if w.endswith("ий"):
                out.append(w[:-2] + "ому")
            elif re.search(r"[бвгджзклмнпрстфхцчшщ]$", w):
                out.append(re.sub(r"(-[а-я]+)$", lambda m: m.group(1) + "у", w) if "-" in w else w + "у")
            else:
                out.append(w)
        return " ".join(out)

    async def _offer_slots(self, state: CallState, u: Utterance, prefix: str = "") -> Reply:
        doctors = await self._specialty_doctors(state)
        exclude: tuple[int, ...] = state.date.exclude_weekdays
        q = SlotQuery(
            doctor_ids=[d.id for d in doctors],
            day=state.date.day, date_from=state.date.date_from, date_to=state.date.date_to,
            part=state.time.part, at=state.time.at, late=state.time.late, exclude_days=exclude,
        )
        slots = await self.slots.find(q)
        note = ""
        if not slots and (state.date.day or state.date.date_from or state.time.part or state.time.at):
            slots = await self.slots.find(SlotQuery(doctor_ids=q.doctor_ids, exclude_days=exclude))
            note = "На это время свободного нет. "
        if not slots and state.pending in ("reschedule",) and state.visit:
            # Нет времени у того же врача — другие врачи той же специальности (С-03)
            others = await self.slots.active_doctors(state.visit.doctor.specialty)
            slots = await self.slots.find(SlotQuery(doctor_ids=[d.id for d in others]))
        if not slots:
            state.offer = {"kind": "waitlist"}
            state.step = "offer"
            return self._say(state, f"{prefix}Свободного времени к {self._dative(state.specialty)} на ближайшие две "
                                    "недели пока нет. Могу поставить вас в лист ожидания и позвонить, как только время "
                                    "освободится. Поставить?")
        if state.time.at and slots[0].at.time() == state.time.at and (
                not state.date.day or slots[0].at.date() == state.date.day):
            state.chosen = slots[0]
            return self._ask_confirm(state, prefix)
        state.offered = slots
        state.step = "choose_slot"
        return self._say(state, f"{prefix}{note}{self._slots_phrase(state, slots)} Что удобнее?")

    def _slots_phrase(self, state: CallState, slots: list[Slot]) -> str:
        many_doctors = len({s.doctor.id for s in slots}) > 1
        items = []
        for s in slots:
            item = say_slot_short(s.at, self.today)
            if many_doctors:
                item += f", врач {s.doctor.last_name}"  # без склонения фамилии — именительный падеж
            items.append(item)
        opts = items[0] if len(items) == 1 else ", ".join(items[:-1]) + " или " + items[-1]
        head = f"Ближайшее время к {self._dative(state.specialty or slots[0].doctor.specialty)}"
        if not many_doctors:
            head += f", врач {slots[0].doctor.last_name}"
        return f"{head}: {opts}."

    async def _on_specialty(self, state: CallState, u: Utterance) -> Reply:
        m = await self.matcher()
        spec = m.match(u)
        doctors = match_doctors(u, await self.registry.list_doctors())
        if spec or doctors:
            self._take_booking_params(state, u, spec, doctors)
            if state.intent == "waitlist":
                return await self._continue_waitlist(state, u)
            return await self._continue_booking(state, u)
        if u.has(r"не знаю|не уверен|подскажите|к кому"):
            state.offer = {"kind": "book", "specialty": "Терапевт"}
            state.step = "offer"
            return self._say(state, "Начните с участкового терапевта — он осмотрит и при необходимости направит "
                                    "к нужному специалисту. Записать вас?")
        return await self._miss(state, u, "К какому врачу вас записать? Например, к терапевту, хирургу или ЛОРу.")

    async def _on_referral(self, state: CallState, u: Utterance) -> Reply:
        if u.has(r"бумажн|из (больниц|другой)"):
            return await self._route(state, guard.PAPER_REFERRAL, u)
        answer = yes_no(u)
        if answer is True:
            state.referral_ok = True  # стенд не хранит направлений: в боевой МИС здесь проверка направления
            return await self._continue_booking(state, u)
        if answer is False:
            gp = "Педиатр" if state.for_child else "Терапевт"
            state.offer = {"kind": "book", "specialty": gp}
            state.specialty, state.doctor_ids = None, []
            state.step = "offer"
            return self._say(state, f"Сначала нужен приём участкового {'педиатра' if state.for_child else 'терапевта'}"
                                    " — он выдаст направление. Записать вас к нему?")
        return await self._miss(state, u, "У вас есть электронное направление к этому врачу?")

    async def _on_choose_slot(self, state: CallState, u: Utterance) -> Reply:
        offered = state.offered
        idx = choice_index(u, len(offered))
        dp = extract_date_pref(u, self.today)
        tp = extract_time_pref(u, loose=True)

        if idx is not None:
            state.chosen = offered[idx]
            return self._ask_confirm(state)
        if tp.at is not None:
            for s in offered:
                if s.at.time() == tp.at and (dp.day is None or s.at.date() == dp.day):
                    state.chosen = s
                    return self._ask_confirm(state)
        if dp.day and tp.at is None and not tp.part:
            same_day = [s for s in offered if s.at.date() == dp.day]
            if len(same_day) == 1:
                state.chosen = same_day[0]
                return self._ask_confirm(state)

        cancel_req = detect_intent(u) == "cancel" and state.visit is not None
        if cancel_req:
            return await self._offer_cancel_instead(state)
        refused = yes_no(u) is False and dp.empty and tp.empty
        if refused:
            if state.direction == "outbound" and state.visit:
                state.offer = {"kind": "cancel"}
                state.step = "offer"
                return self._say(state, "Тогда отменить запись, чтобы время досталось другому пациенту?")
            state.step = "choose_slot"
            return self._say(state, "Назовите удобный день или время, и я поищу.")
        if yes_no(u) is True and len(offered) == 1:
            state.chosen = offered[0]
            return self._ask_confirm(state)

        # Новые пожелания: «а после обеда есть?», «на следующей неделе», «попозже»
        m = await self.matcher()
        spec = m.match(u)
        if spec and state.pending != "reschedule" and normalize_name(spec) != normalize_name(state.specialty or ""):
            state.specialty, state.doctor_ids = spec, []
            return await self._continue_booking(state, u)
        if not dp.empty or not tp.empty:
            if not dp.empty:
                state.date = dp
            state.time = tp if not tp.empty else TimePref()
            return await self._offer_slots(state, u)
        return await self._miss(state, u, self._slots_phrase(state, offered) + " Какой вариант выбираете?")

    def _ask_confirm(self, state: CallState, prefix: str = "") -> Reply:
        slot = state.chosen
        room = f", кабинет {_room_words(slot.room)}" if slot.room else ""
        what = f"{spoken_specialty(slot.doctor.specialty)} {slot.doctor.last_name}, {say_slot_full(slot.at, self.today)}{room}"
        state.step = "confirm"
        if state.pending in ("reschedule", "replace") and state.visit:
            state.pending = "replace" if not state.visit.doctor.is_active else "reschedule"
            old = say_slot_short(state.visit.scheduled_at, self.today)
            return self._say(state, f"{prefix}Переношу на {what}. Прежнюю запись — {old} — отменяю. Верно?")
        state.pending = "book"
        return self._say(state, f"{prefix}Записываю: {what}. Верно?")

    async def _on_confirm(self, state: CallState, u: Utterance) -> Reply:
        answer = yes_no(u)
        if answer is True:
            if state.pending == "book":
                return await self._do_book(state, u)
            if state.pending in ("reschedule", "replace"):
                return await self._do_reschedule(state, u)
            if state.pending == "cancel":
                return await self._do_cancel(state, u)
        if answer is False:
            if state.pending == "cancel":
                return self._done(state, "Хорошо, запись остаётся в силе.", "INFO_GIVEN")
            state.chosen = None
            if state.offered:
                state.step = "choose_slot"
                return self._say(state, "Хорошо, давайте выберем другое время. Какой день и время вам удобны?")
            return await self._offer_slots(state, u)
        # «нет, давайте в 10:40» и т. п. — разбираем как выбор времени
        if state.offered and (not extract_time_pref(u, loose=True).empty or not extract_date_pref(u, self.today).empty):
            state.step = "choose_slot"
            return await self._on_choose_slot(state, u)
        return await self._miss(state, u, state.last_prompt)

    async def _do_book(self, state: CallState, u: Utterance) -> Reply:
        slot = state.chosen
        if not await self.slots.is_free(slot.doctor, slot.at):
            state.chosen = None
            return await self._offer_slots(state, u, prefix="Извините, это время только что заняли. ")
        visit = await self.registry.create_visit(state.client.id, slot.doctor.id, slot.at)
        self.slots.note_visit(visit)
        if state.phone:
            self.storage.save_phone(state.client.id, state.phone)
        return self._done(state, "Готово, вы записаны. Возьмите паспорт и полис ОМС. Накануне я позвоню и напомню.",
                          "BOOKED")

    # =========================================================================
    # С-03/С-04/С-05. Перенос, отмена, проверка записи
    # =========================================================================

    def _take_visit_params(self, state: CallState, u: Utterance, spec: str | None) -> None:
        dp = extract_date_pref(u, self.today)
        # «перенесите на понедельник» — новая дата; «запись на четверг, хочу перенести» — какая запись
        if re.search(r"(перенес|перезапис|перезапиш|сдвин)\w*\s+(\w+\s+)?(на|во?)\s", u.words):
            state.date = dp
        else:
            state.visit_hint = dp
        state.time = extract_time_pref(u)
        state.specialty = spec

    async def _continue_visits(self, state: CallState, u: Utterance) -> Reply:
        if state.client is None:
            if not state.person.complete:
                return self._ask_person(state, "visits")
            found = await self._find_client(state, u)
            if isinstance(found, Reply):
                return found
        greet = f"{self._polite(state)}, " if self._polite(state) and state.step == "identify" else ""

        if state.visit is None:
            visits = [v for v in await self.registry.list_visits(client_id=state.client.id, status=VisitStatus.PLANNED)
                      if v.scheduled_at >= self.clock()]
            visits.sort(key=lambda v: v.scheduled_at)
            if state.intent == "check":
                if not visits:
                    state.offer = {"kind": "book"}
                    state.step = "offer"
                    return self._say(state, f"{greet}Активных записей не найдено. Хотите записаться?")
                listed = "; ".join(self._visit_full(v) for v in visits[:3])
                more = f" И ещё {len(visits) - 3}." if len(visits) > 3 else ""
                prefix = "вы записаны" if len(visits) == 1 else "ваши записи"
                return self._done(state, _cap1(f"{greet}{prefix}: {listed}.{more}"), "INFO_GIVEN")
            if not visits:
                state.offer = {"kind": "book"}
                state.step = "offer"
                return self._say(state, f"{greet}Активных записей не найдено. Хотите записаться?")
            visits = self._filter_visits(visits, state.visit_hint, state.specialty)
            if len(visits) > 1:
                state.visits = visits
                state.step = "choose_visit"
                action = "перенести" if state.intent == "reschedule" else "отменить"
                if len(visits) > 3:
                    return self._say(state, f"{greet}У вас {len(visits)} записей. К какому врачу запись, которую нужно {action}?")
                listed = "; ".join(f"{i + 1}) {self._visit_short(v)}" for i, v in enumerate(visits))
                return self._say(state, f"{greet}У вас несколько записей: {listed}. Какую {action}?")
            state.visit = visits[0]

        return await self._act_on_visit(state, u, greet)

    def _filter_visits(self, visits: list[Visit], hint: DatePref, spec: str | None) -> list[Visit]:
        out = visits
        if hint.day:
            out = [v for v in out if v.scheduled_at.date() == hint.day] or out
        if spec:
            out = [v for v in out if normalize_name(v.doctor.specialty) == normalize_name(spec)] or out
        return out

    async def _on_choose_visit(self, state: CallState, u: Utterance) -> Reply:
        idx = choice_index(u, len(state.visits))
        if idx is None and (m := re.search(r"\b([1-9])\b", u.text)) and int(m.group(1)) <= len(state.visits):
            idx = int(m.group(1)) - 1
        if idx is not None:
            state.visit = state.visits[idx]
        else:
            spec = (await self.matcher()).match(u)
            filtered = self._filter_visits(state.visits, extract_date_pref(u, self.today), spec)
            doctors = match_doctors(u, [v.doctor for v in state.visits])
            if doctors:
                filtered = [v for v in filtered if v.doctor.id in {d.id for d in doctors}]
            if len(filtered) != 1 or filtered == state.visits:
                return await self._miss(state, u, "Назовите врача или дату записи.")
            state.visit = filtered[0]
        return await self._act_on_visit(state, u, "")

    async def _act_on_visit(self, state: CallState, u: Utterance, greet: str) -> Reply:
        visit = state.visit
        if state.intent == "cancel":
            if not state.cancel_offered_reschedule:
                state.cancel_offered_reschedule = True
                state.step = "cancel_or_reschedule"
                return self._say(state, f"{greet}У вас запись: {self._visit_short(visit)}. Могу перенести её на другой "
                                        "день вместо отмены. Перенести или отменить?")
            return await self._confirm_cancel(state)

        # Перенос: к тому же врачу; если врач деактивирован — к другому той же специальности
        state.pending = "reschedule"
        state.specialty = visit.doctor.specialty
        if visit.doctor.is_active:
            state.doctor_ids = [visit.doctor.id]
            lead = f"{greet}У вас запись: {self._visit_short(visit)}. "
        else:
            state.doctor_ids = []
            lead = (f"{greet}Врач {visit.doctor.last_name} больше не ведёт приём — эту запись я отменю и запишу вас "
                    f"к другому врачу. ")
        return await self._offer_slots(state, u, prefix=lead)

    async def _on_cancel_or_reschedule(self, state: CallState, u: Utterance) -> Reply:
        intent = detect_intent(u)
        if intent == "reschedule" or u.has(r"перенес|перенос|другой день|другое время"):
            state.intent = "reschedule"
            self._take_booking_params(state, u, None, [])
            return await self._act_on_visit(state, u, "")
        if intent == "cancel" or u.has(r"отмен|не нужн|не надо") or yes_no(u) is False:
            return await self._confirm_cancel(state)
        return await self._miss(state, u, "Перенести запись на другой день или отменить?")

    async def _offer_cancel_instead(self, state: CallState) -> Reply:
        state.intent = "cancel"
        if not state.cancel_offered_reschedule:
            state.cancel_offered_reschedule = True
            state.step = "cancel_or_reschedule"
            return self._say(state, "Могу перенести запись на другой день вместо отмены. Перенести или отменить?")
        return await self._confirm_cancel(state)

    async def _confirm_cancel(self, state: CallState) -> Reply:
        state.intent = "cancel"
        state.pending = "cancel"
        state.step = "confirm"
        v = state.visit
        return self._say(state, f"Отменяю запись: {self._visit_short(v)}. Верно?")

    async def _do_cancel(self, state: CallState, u: Utterance) -> Reply:
        visit = await self.registry.update_visit(state.visit.id, status=VisitStatus.CANCELLED)
        self.slots.note_visit(visit)
        if state.direction == "outbound":
            self.storage.set_reminder(visit.id, "CANCELLED")
        return self._done(state, "Готово, запись отменена. Спасибо, что предупредили — это время достанется "
                                 "другому пациенту.", "CANCELLED")

    async def _do_reschedule(self, state: CallState, u: Utterance) -> Reply:
        old, slot = state.visit, state.chosen
        old_when = say_date(old.scheduled_at.date(), self.today).split(",")[0]
        if not await self.slots.is_free(slot.doctor, slot.at):
            state.chosen = None
            return await self._offer_slots(state, u, prefix="Извините, это время только что заняли. ")
        try:
            if old.doctor.is_active:
                visit = await self.registry.update_visit(
                    old.id, at=slot.at, doctor_id=slot.doctor.id if slot.doctor.id != old.doctor.id else None
                )
            else:
                # Запись к деактивированному врачу можно только отменить (стенд, БП-18) — создаём новую
                visit = await self.registry.create_visit(old.client_id, slot.doctor.id, slot.at)
                cancelled = await self.registry.update_visit(old.id, status=VisitStatus.CANCELLED)
                self.slots.note_visit(cancelled)
        except VISIT_ERRORS as e:
            log.warning("Перенос не удался: %s", e)
            route = guard.Route("R-06", "P3", "М2",
                                f"Извините, не получилось перенести. Ваша запись {old_when} остаётся в силе. "
                                "Соединяю с регистратором.",
                                transfer=True, result="ESCALATED", reason="Сбой МИС при переносе")
            return await self._route(state, route, u)
        self.slots.note_visit(visit)
        if state.direction == "outbound":
            self.storage.set_reminder(old.id, "RESCHEDULED")
            return self._finish(state, "Готово, запись перенесена. Спасибо, что предупредили. Всего доброго!",
                                "RESCHEDULED")
        return self._done(state, "Готово, запись перенесена.", "RESCHEDULED")

    def _visit_short(self, v: Visit) -> str:
        return f"{spoken_specialty(v.doctor.specialty)} {v.doctor.last_name}, {say_slot_short(v.scheduled_at, self.today)}"

    def _visit_full(self, v: Visit) -> str:
        room = self.slots.room_of(v.doctor)
        room_s = f", кабинет {_room_words(room)}" if room else ""
        return f"{spoken_specialty(v.doctor.specialty)} {v.doctor.last_name}, {say_slot_full(v.scheduled_at, self.today)}{room_s}"

    # =========================================================================
    # С-07. График врача
    # =========================================================================

    async def _schedule_info(self, state: CallState, spec: str | None, doctors: list[Doctor]) -> Reply:
        if not doctors and spec:
            doctors = await self.slots.active_doctors(spec)
        if not doctors:
            entry = next(e for e in self.kb.entries if e.id == "hours")
            return await self._kb_answer(state, entry, Utterance.parse(""))
        parts = []
        for d in doctors[:2]:
            info = self.schedule.doctor(d.full_name)
            if not info or not d.is_active:
                parts.append(f"Врача {d.last_name} в расписании нет")
                continue
            absence = info.absence_on(self.today)
            if absence:
                parts.append(f"{d.last_name} не принимает до {say_date(absence.date_to + timedelta(days=1), self.today)}")
                continue
            by_hours: dict[str, list[int]] = {}
            for wd, hours in info.weekly.items():
                by_hours.setdefault(hours, []).append(wd)
            sched = ", ".join(f"{weekdays_words(days)} {hours_range_words(h)}" for h, days in by_hours.items())
            room = f", кабинет {_room_words(info.room)}" if info.room else ""
            parts.append(f"{d.last_name} — {sched}{room}")
        more = f" Всего врачей этой специальности: {len(doctors)}." if len(doctors) > 2 else ""
        state.offer = {"kind": "book", "specialty": spec or doctors[0].specialty,
                       "doctor_ids": [d.id for d in doctors] if not spec else []}
        state.step = "offer"
        state.results.append("INFO_GIVEN")
        return self._say(state, ". ".join(parts) + "." + more + " Записать вас?")

    # =========================================================================
    # С-06. Вызов врача на дом (полуавтоматический)
    # =========================================================================

    def _home_visit_day(self) -> tuple[date | None, str]:
        now = self.clock()
        wd, hour = now.weekday(), now.hour
        if wd == 6:
            return None, "В воскресенье заявки на дом не принимаются. При ухудшении звоните сто три."
        cutoff = 12 if wd == 5 else 14
        if hour < cutoff:
            return now.date(), ""
        nxt = now.date() + timedelta(days=1)
        if nxt.weekday() == 6:
            nxt += timedelta(days=1)
        return nxt, (f"Заявки на сегодня принимаются до {'двенадцати' if wd == 5 else 'четырнадцати'} часов, "
                     "оформлю вызов на следующий день. ")

    async def _start_home_visit(self, state: CallState, u: Utterance) -> Reply:
        day, note = self._home_visit_day()
        if day is None:
            return self._done(state, note, "INFO_GIVEN")
        state.home["day"] = fmt_date(day)
        state.after_identify = "home"
        state.step = "identify"
        if state.person.complete:
            return await self._continue_home_visit(state, u)
        return self._say(state, f"Хорошо. Сначала важное: если сейчас сильная боль в груди, трудно дышать или человек "
                                f"теряет сознание — положите трубку и звоните сто три. Если нет, продолжим. {note}"
                                f"Назовите фамилию, имя, отчество и дату рождения пациента.")

    async def _continue_home_visit(self, state: CallState, u: Utterance) -> Reply:
        if state.client is None:
            await self._ensure_client(state)
        if age_on(state.client.birth_date, self.today) < 1:
            return await self._route(state, guard.INFANT_HOME_VISIT, u)
        state.step = "home_address"
        polite = f"{self._polite(state)}, н" if self._polite(state) else "Н"
        return self._say(state, f"{polite}азовите адрес, куда должен прийти врач.")

    async def _on_home_address(self, state: CallState, u: Utterance) -> Reply:
        if len(u.words.split()) < 2:
            return await self._miss(state, u, "Назовите улицу, дом и квартиру.")
        state.home["address"] = u.raw.strip()
        state.step = "home_reason"
        return self._say(state, "Спасибо. Коротко скажите своими словами, что случилось. Я передам врачу.")

    async def _on_home_reason(self, state: CallState, u: Utterance) -> Reply:
        state.home["reason"] = u.raw.strip()  # дословно, без оценки (С-06)
        state.step = "home_self"
        return self._say(state, "Передам врачу. Вы можете сами прийти в поликлинику?")

    async def _on_home_self(self, state: CallState, u: Utterance) -> Reply:
        answer = yes_no(u)
        if answer is True and not state.home.get("insist"):
            state.home["insist"] = "1"
            gp = "Педиатр" if age_on(state.client.birth_date, self.today) < 18 else "Терапевт"
            state.offer = {"kind": "book", "specialty": gp}
            state.step = "home_offer_book"
            return self._say(state, f"Тогда быстрее будет прийти на приём. Записать вас к "
                                    f"{'педиатру' if gp == 'Педиатр' else 'терапевту'} на сегодня или завтра?")
        if answer is None:
            return await self._miss(state, u, "Пациент может сам прийти в поликлинику?")
        return await self._create_home_visit(state, can_come=answer)

    async def _on_home_offer_book(self, state: CallState, u: Utterance) -> Reply:
        if yes_no(u) is False or u.has(r"вызов|на дом|пусть придет|оформите"):
            return await self._create_home_visit(state, can_come=True)
        state.home = {}
        state.step = "offer"
        state.date = DatePref(date_from=self.today, date_to=self.today + timedelta(days=1))
        return await self._on_offer(state, u)

    async def _create_home_visit(self, state: CallState, can_come: bool) -> Reply:
        h = state.home
        description = (
            f"Вызов врача на дом (требует подтверждения). Дата визита: {h['day']}. Адрес: {h.get('address')}. "
            f"Повод (слова пациента): «{h.get('reason')}». "
            f"{'Может прийти сам, настаивает на вызове.' if can_come else 'Прийти сам не может.'} "
            f"Телефон: {state.phone or 'не определён'}."
        )
        esc_id = await self._escalate(state, description)
        self.storage.save_transfer(state.call_id, {"тип": "вызов на дом", **h, "телефон": mask_phone(state.phone)}, esc_id)
        day = "сегодня" if h["day"] == fmt_date(self.today) else "завтра" if h["day"] == fmt_date(
            self.today + timedelta(days=1)) else h["day"]
        state.results.append("HOME_VISIT_REQUESTED")
        return self._finish(state, f"Хорошо, заявка принята на {day}. Врач придёт до восемнадцати часов, точное время "
                                   "назвать не могу. Регистратор проверит заявку и перезвонит, только если понадобится "
                                   "уточнение. Если станет хуже — звоните сто три. Выздоравливайте!",
                            "HOME_VISIT_REQUESTED")

    # =========================================================================
    # С-11. Лист ожидания
    # =========================================================================

    async def _continue_waitlist(self, state: CallState, u: Utterance) -> Reply:
        if not state.specialty:
            state.step = "specialty"
            return self._say(state, "К какому врачу поставить вас в лист ожидания?")
        if state.client is None:
            if not state.person.complete:
                return self._ask_person(state, "waitlist")
            await self._ensure_client(state)
        prefs = ", ".join(x for x in [
            {"morning": "утро", "day": "день", "evening": "вечер"}.get(state.time.part or ""),
            fmt_date(state.date.day) if state.date.day else "",
        ] if x) or "любое время"
        self.storage.add_to_waitlist(state.client.id, state.specialty, prefs, state.phone)
        return self._done(state, f"Готово: {spoken_specialty(state.specialty)}, {prefs}. Как только время освободится, "
                                 "я позвоню на этот номер.", "WAITLISTED")

    # =========================================================================
    # С-10. Исходящий звонок-напоминание
    # =========================================================================

    async def start_reminder(self, call_id: str, visit: Visit, client: Client, phone: str | None) -> tuple[CallState, Reply]:
        self.storage.start_call(call_id, "outbound", phone)
        self.storage.set_reminder(visit.id, "CALLING", attempt=True)
        state = CallState(call_id=call_id, phone=phone, direction="outbound", intent="reminder",
                          client=client, visit=visit, step="rem_who")
        # В исходящем звонке допустимо назвать имя и отчество — без фамилии (раздел 3.3)
        return state, self._say(state, f"Здравствуйте! Это виртуальный помощник поликлиники номер семь. "
                                       f"Я говорю с {client.polite_name}?")

    async def _on_rem_who(self, state: CallState, u: Utterance) -> Reply:
        answer = yes_no(u)
        if u.has(r"ошиблись|не туда|его нет|ее нет|нет дома|это не (он|она|я)|другой человек|вы не туда") or answer is False:
            self.storage.set_reminder(state.visit.id, "WRONG_PERSON")
            return self._finish(state, f"Передайте, пожалуйста, {state.client.polite_name}, чтобы перезвонили "
                                       "в поликлинику номер семь. Всего доброго!", "WRONG_PERSON")
        if answer is True or u.has(r"слушаю|это я|я$"):
            state.step = "rem_confirm"
            return self._say(state, f"Напоминаю: у вас запись — {self._visit_full(state.visit)}. Вы придёте?")
        return await self._miss(state, u, f"Я говорю с {state.client.polite_name}?")

    async def _on_rem_confirm(self, state: CallState, u: Utterance) -> Reply:
        intent = detect_intent(u)
        visit = state.visit
        if intent == "opt_out":
            self.storage.set_opt_out(state.client.id)
            self.storage.set_reminder(visit.id, "OPT_OUT")
            return self._finish(state, "Хорошо, больше не буду звонить с напоминаниями. Всего доброго!", "OPT_OUT")
        if u.has(r"не знаю|может быть|возможно|постараюсь|наверное"):
            self.storage.set_reminder(visit.id, "UNCONFIRMED")
            return self._finish(state, "Если планы изменятся, позвоните, пожалуйста, по номеру регистратуры — время "
                                       "достанется другому пациенту. Всего доброго!", "UNCONFIRMED")
        if u.has(r"умер|скончал"):
            await self._cancel_silently(state)
            await self._escalate(state, "Напоминание: родственник сообщил о смерти пациента. Запись отменена, "
                                        "нужна корректировка данных.")
            return self._finish(state, "Примите наши соболезнования. Запись я отменю. До свидания.", "CANCELLED")
        if u.has(r"в больнице|в стационаре|госпитализ|лежит в"):
            await self._cancel_silently(state)
            return self._finish(state, "Желаем скорейшего выздоровления. Запись я отменю. До свидания.", "CANCELLED")
        if intent == "cancel" and not u.has(r"не (смогу|получится|приду)"):
            state.intent = "cancel"
            return await self._offer_cancel_instead(state)
        answer = yes_no(u)
        if answer is True and intent != "reschedule":
            self.storage.set_reminder(visit.id, "CONFIRMED")
            return self._finish(state, "Отлично, ждём вас. Возьмите паспорт и полис. До свидания!", "CONFIRMED")
        if answer is False or intent in ("reschedule", "cancel"):
            state.intent = "reschedule"
            self._take_booking_params(state, u, None, [])
            state.cancel_offered_reschedule = True
            state.pending = "reschedule"
            state.specialty = visit.doctor.specialty
            state.doctor_ids = [visit.doctor.id] if visit.doctor.is_active else []
            return await self._offer_slots(state, u, prefix="Давайте перенесу. ")
        return await self._miss(state, u, "Вы придёте на приём?")

    async def _cancel_silently(self, state: CallState) -> None:
        try:
            v = await self.registry.update_visit(state.visit.id, status=VisitStatus.CANCELLED)
            self.slots.note_visit(v)
            self.storage.set_reminder(v.id, "CANCELLED")
        except VISIT_ERRORS as e:
            log.warning("Не удалось отменить запись: %s", e)


def _cap1(s: str) -> str:
    return s[:1].upper() + s[1:]


def _room_words(room: int) -> str:
    from .numbers import number_words
    return number_words(room)
