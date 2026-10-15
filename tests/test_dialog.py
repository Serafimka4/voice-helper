"""Сценарии разговора, включая контрольные диалоги Приложения Д регламента (с «!» — блокирующие)."""
from __future__ import annotations

from datetime import datetime, time, timedelta

from voice_helper.models import VisitStatus
from voice_helper.reminders import due_reminders

PERSON = "Смирнова Ольга Викторовна, двенадцатое марта семьдесят восьмого"


async def book_therapist(c) -> None:
    await c.say("Мне бы записаться к терапевту", PERSON)
    await c.say("первый")
    assert "Верно?" in c.last.text
    await c.say("да")
    assert "Готово, вы записаны" in c.last.text


# --- С-01. Запись ------------------------------------------------------------

async def test_booking_happy_path(call, stand):
    c = await call()
    assert "виртуальный помощник" in c.last.text and "Разговор записывается" in c.last.text
    r = await c.say("Мне бы записаться к терапевту")
    assert "фамилию, имя, отчество и дату рождения" in r.text
    r = await c.say(PERSON)
    assert r.text.startswith("Спасибо, Ольга Викторовна")
    assert r.text.count(" или ") == 1  # не более трёх вариантов
    r = await c.say("А после обеда есть что-нибудь?")
    slots = c.state.offered
    assert slots and all(time(12) <= s.at.time() < time(16) for s in slots)
    await c.say("давайте второй")
    assert "Записываю: терапевт" in c.last.text and "кабинет" in c.last.text
    await c.say("да")
    assert "Готово, вы записаны" in c.last.text
    [visit] = stand.visits.values()
    assert visit.scheduled_at == slots[1].at and visit.status == VisitStatus.PLANNED
    client = stand.clients[visit.client_id]
    assert (client.last_name, client.first_name, client.middle_name) == ("Смирнова", "Ольга", "Викторовна")
    r = await c.say("нет, спасибо")
    assert r.end_call and "Всего доброго" in r.text
    assert c.assistant.storage.call("test-call")["result"] == "BOOKED"


async def test_booking_exact_time_goes_straight_to_confirmation(call, stand):
    c = await call()
    await c.say("запишите меня к хирургу на завтра", PERSON)
    first = c.state.offered[0]
    r = await c.say(f"давайте в {first.at:%H %M}")
    assert "Записываю: хирург" in r.text


async def test_booked_slot_is_not_offered_again(call, assistant, stand):
    c = await call()
    await book_therapist(c)
    taken = next(iter(stand.visits.values()))
    c2 = await call("+7 900 000-00-02")
    await c2.say("к терапевту", "Иванов Сергей Петрович, третье января девяностого")
    assert all(not (s.at == taken.scheduled_at and s.doctor.id == taken.doctor.id) for s in c2.state.offered)


async def test_one_active_visit_per_specialty_offers_reschedule(call):
    c = await call()
    await book_therapist(c)
    r = await c.say("и ещё запишите к терапевту")
    assert "уже есть запись" in r.text and "перенести" in r.text


async def test_child_is_routed_to_pediatrics(call):
    c = await call()
    r = await c.say("Мне нужно записать дочку к терапевту")
    assert "ребёнка" in r.text
    r = await c.say("Смирнова Алиса Дмитриевна, седьмое апреля две тысячи девятнадцатого")
    assert "педиатр" in r.text.lower()
    assert c.state.specialty == "Педиатр"


async def test_ent_not_friday_after_lunch(call):
    """Контрольный диалог 6: «Мне к лору, но лучше не в пятницу, после обеда»."""
    c = await call()
    await c.say("Мне к лору, но лучше не в пятницу, после обеда", PERSON)
    slots = c.state.offered
    assert 1 <= len(slots) <= 3
    assert all(s.at.weekday() != 4 and time(12) <= s.at.time() < time(16) for s in slots)
    assert all("отоларинголог" in s.doctor.specialty.lower() for s in slots)


async def test_referral_specialty_without_referral_offers_therapist(call):
    """Контрольный диалог 8: «Запишите к неврологу» без направления."""
    c = await call()
    await c.say("Запишите к неврологу", PERSON)
    assert "по направлению" in c.last.text
    r = await c.say("нет")
    assert "терапевта" in r.text
    r = await c.say("да")
    assert "Ближайшее время к терапевту" in r.text


# --- С-03/С-04/С-05. Перенос, отмена, проверка --------------------------------

async def test_reschedule(call, stand):
    c = await call()
    await book_therapist(c)
    old = next(iter(stand.visits.values()))
    c2 = await call()
    r = await c2.say("Я записана к терапевту, хочу перенести", PERSON)
    assert "У вас запись: терапевт" in r.text
    await c2.say("на следующей неделе")
    new_slot = c2.state.offered[0]
    r = await c2.say("первый")
    assert "Переношу на" in r.text and "Прежнюю запись" in r.text
    await c2.say("да")
    moved = stand.visits[old.id]
    assert moved.scheduled_at == new_slot.at and moved.status == VisitStatus.PLANNED
    assert len(stand.visits) == 1


async def test_cancel_with_two_visits(call, stand):
    """Контрольный диалог 7: перечислить записи, уточнить, один раз предложить перенос."""
    c = await call()
    await book_therapist(c)
    await c.say("ещё к хирургу", "первый", "да")
    assert len(stand.visits) == 2
    c2 = await call()
    r = await c2.say("Отмените мою запись", PERSON)
    assert "несколько записей" in r.text
    r = await c2.say("к хирургу")
    assert "Перенести или отменить" in r.text
    r = await c2.say("отменить, уже не нужно")
    assert r.text.startswith("Отменяю запись: хирург")
    r = await c2.say("да")
    assert "запись отменена" in r.text
    statuses = sorted(v.status for v in stand.visits.values())
    assert statuses == [VisitStatus.PLANNED, VisitStatus.CANCELLED]


async def test_check_my_visits(call):
    c = await call()
    await book_therapist(c)
    c2 = await call()
    r = await c2.say("на когда я записана?", PERSON)
    assert "Ольга Викторовна, вы записаны: терапевт" in r.text and "кабинет" in r.text


async def test_unknown_patient_goes_to_operator_after_second_attempt(call):
    c = await call()
    r = await c.say("отмените запись", "Петров Пётр, первое января восьмидесятого")
    assert "Не нашёл вас" in r.text
    r = await c.say("Петров Пётр, первое января восьмидесятого")
    assert r.transfer and r.card["причина_перевода"] == "R-07"


async def test_no_visits_offers_booking(call, stand):
    c = await call()
    await stand.ensure_client("Смирнова", "Ольга", datetime(1978, 3, 12).date())
    r = await c.say("хочу перенести запись", PERSON)
    assert "Активных записей не найдено" in r.text


# --- Границы и маршрутизация (раздел 7) ---------------------------------------

async def test_red_flag_interrupts_booking(call):
    """! Контрольный диалог 1."""
    c = await call()
    await c.say("Запишите к терапевту")
    r = await c.say("ой, у меня вообще-то сильно давит в груди")
    assert "сто три" in r.text and r.transfer
    assert r.card["причина_перевода"] == "R-01" and r.card["приоритет"] == "P1"


async def test_medical_question_boundary_phrase(call):
    """! Контрольный диалог 2: граничная фраза, без препаратов, предложение записи."""
    c = await call()
    r = await c.say("Что мне выпить от головной боли?")
    assert "не могу давать медицинские советы" in r.text and "терапевт" in r.text
    assert not r.transfer
    for drug in ("парацетамол", "ибупрофен", "анальгин", "таблетк"):
        assert drug not in r.text.lower()
    r = await c.say("да")
    assert "фамилию" in r.text


async def test_medical_question_insisting_is_transferred(call):
    c = await call()
    await c.say("Что мне выпить от давления?")
    r = await c.say("нет, скажите, какие таблетки пить")
    assert r.transfer and r.card["причина_перевода"] == "R-04"


async def test_analysis_values_not_disclosed(call):
    """! Контрольный диалог 3."""
    c = await call()
    r = await c.say("Какой у меня холестерин в последнем анализе?")
    assert "не вижу значений" in r.text and "Госуслугах" in r.text


async def test_third_party_cannot_learn_visits(call):
    """! Контрольный диалог 4: посторонний спрашивает о записях."""
    c = await call()
    r = await c.say("На когда записана Смирнова Ольга, двенадцатое марта семьдесят восьмого?")
    assert "только самому пациенту" in r.text
    assert "терапевт" not in r.text


async def test_operator_on_demand_at_any_step(call):
    """! Контрольный диалог 5."""
    c = await call()
    await c.say("Запишите к терапевту")
    r = await c.say("Соедините с человеком")
    assert r.transfer and r.text == "Соединяю с сотрудником."
    assert r.card["причина_перевода"] == "R-02"


async def test_not_understood_three_times(call):
    """Контрольный диалог 12: переспрос → предложение оператора → перевод."""
    c = await call()
    r = await c.say("бу-бу-бу")
    assert "не совсем понял" in r.text
    r = await c.say("ммм кхм")
    assert "Соединить с сотрудником?" in r.text
    r = await c.say("да")
    assert r.transfer and r.card["причина_перевода"] == "R-05"


async def test_are_you_robot(call):
    """Контрольный диалог 14."""
    c = await call()
    r = await c.say("Вы робот?")
    assert "виртуальный помощник" in r.text and "соединю с сотрудником" in r.text
    r = await c.say("да")
    assert r.transfer


async def test_are_you_robot_no_returns_to_previous_question(call):
    c = await call()
    await c.say("Запишите к терапевту")
    await c.say("Вы робот?")
    r = await c.say("нет")
    assert r.text == "Хорошо. Назовите, пожалуйста, фамилию, имя, отчество и дату рождения."
    r = await c.say(PERSON)
    assert "Ближайшее время к терапевту" in r.text


async def test_recording_opt_out(call, assistant):
    """Контрольный диалог 15."""
    c = await call()
    await c.say("Запишите к терапевту")
    r = await c.say("Я против записи разговора")
    assert r.text.startswith("Хорошо, запись разговора отключена")
    assert not r.transfer
    assert assistant.storage.turns("test-call") == []
    await c.say(PERSON)
    assert assistant.storage.turns("test-call") == []


async def test_complaint_creates_escalation_for_identified_patient(call, stand):
    """Контрольный диалог 16 + фиксация в /escalations."""
    c = await call()
    await book_therapist(c)
    r = await c.say("Хочу пожаловаться на регистратуру")
    assert r.transfer and r.card["причина_перевода"] == "R-03"
    [esc] = stand.escalations.values()
    assert "R-03" in esc.description and "пожаловаться" in esc.description


async def test_out_of_hours_transfer_becomes_callback(call, clock):
    clock.now = datetime(2026, 10, 18, 12, 0)  # воскресенье
    c = await call()
    r = await c.say("соедините с оператором")
    assert not r.transfer and r.end_call and "перезвонят" in r.text


async def test_acute_dental_pain_goes_to_dentistry(call):
    c = await call()
    r = await c.say("у меня зуб болит, щёку раздуло")
    assert r.transfer and r.card["причина_перевода"] == "R-09"


# --- FAQ, график, вызов на дом, лист ожидания ---------------------------------

async def test_faq_hours(call):
    c = await call()
    r = await c.say("до скольки вы работаете в субботу?")
    assert "в субботу с девяти до пятнадцати" in r.text


async def test_faq_price(call):
    c = await call()
    r = await c.say("Сколько у вас стоит УЗИ щитовидки?")
    assert "тысячу триста рублей" in r.text


async def test_doctor_schedule(call):
    c = await call()
    r = await c.say("Когда принимает кардиолог?")
    assert "Записать вас?" in r.text and "с восьми" in r.text


async def test_home_visit_after_cutoff(call, clock, stand):
    """Контрольный диалог 9: вызов на дом во вторник в 15:10."""
    clock.now = datetime(2026, 10, 20, 15, 10)
    c = await call()
    r = await c.say("Мне нужно вызвать врача на дом")
    assert "сто три" in r.text and "до четырнадцати" in r.text
    await c.say("Соколов Андрей Викторович, девятое февраля восемьдесят пятого")
    await c.say("Улица Лесная, дом двенадцать, квартира сорок")
    await c.say("Температура со вчерашнего дня, слабость, кашель")
    r = await c.say("Нет, тяжело")
    assert "заявка принята на завтра" in r.text and r.end_call
    [esc] = stand.escalations.values()
    assert "Вызов врача на дом" in esc.description and "21.10.2026" in esc.description
    assert "Лесная" in esc.description


async def test_waitlist_when_no_slots(call, assistant):
    assistant.slots.min_lead = timedelta(days=30)  # в горизонте 14 дней свободного времени нет
    c = await call()
    r = await c.say("запишите к хирургу", PERSON)
    assert "лист ожидания" in r.text
    r = await c.say("да")
    assert r.text.startswith("Готово: хирург")
    assert c.assistant.storage.db.execute("SELECT specialty FROM waitlist").fetchone()[0] == "Хирург"


# --- С-10. Напоминания ----------------------------------------------------------

async def _visit_tomorrow(assistant, stand, clock):
    client = await stand.ensure_client("Смирнова", "Ольга", datetime(1978, 3, 12).date(), "Викторовна")
    assistant.storage.save_phone(client.id, "+7 900 000-00-01")
    doctor = next(d for d in stand.doctors.values() if d.specialty == "Терапевт")
    day = assistant.slots.schedule.free_slots(doctor.full_name, datetime(2026, 10, 16).date())
    visit = await stand.create_visit(client.id, doctor.id, day[0])
    stand.visits[visit.id] = type(visit)(visit.id, visit.client_id, visit.doctor, visit.scheduled_at,
                                         visit.status, datetime(2026, 10, 10, 9, 0))
    return client, stand.visits[visit.id]


async def test_reminder_confirmed(assistant, stand, clock):
    await _visit_tomorrow(assistant, stand, clock)
    [task] = await due_reminders(stand, assistant.storage, clock())
    state, r = await assistant.dialog.start_reminder("rem-1", task.visit, task.client, task.phone)
    assert "Я говорю с Ольга Викторовна?" in r.text or "Ольгой" in r.text or "Ольга Викторовна" in r.text
    assert "Смирнова" not in r.text  # без фамилии
    r = await assistant.dialog.handle(state, "да")
    assert "Вы придёте?" in r.text
    r = await assistant.dialog.handle(state, "да, приду")
    assert r.end_call and "ждём вас" in r.text
    assert assistant.storage.reminder(task.visit.id)["status"] == "CONFIRMED"
    assert await due_reminders(stand, assistant.storage, clock()) == []


async def test_reminder_decline_reschedules(assistant, stand, clock):
    _, visit = await _visit_tomorrow(assistant, stand, clock)
    [task] = await due_reminders(stand, assistant.storage, clock())
    state, _ = await assistant.dialog.start_reminder("rem-2", task.visit, task.client, task.phone)
    await assistant.dialog.handle(state, "да")
    r = await assistant.dialog.handle(state, "Ой, нет, завтра не получится")
    assert r.text.startswith("Давайте перенесу")
    new_at = state.offered[0].at
    await assistant.dialog.handle(state, "первый")
    r = await assistant.dialog.handle(state, "да")
    assert r.end_call and "перенесена" in r.text
    assert stand.visits[visit.id].scheduled_at == new_at
    assert assistant.storage.reminder(visit.id)["status"] == "RESCHEDULED"


async def test_reminder_wrong_person_reveals_nothing(assistant, stand, clock):
    """Контрольный диалог 10."""
    await _visit_tomorrow(assistant, stand, clock)
    [task] = await due_reminders(stand, assistant.storage, clock())
    state, _ = await assistant.dialog.start_reminder("rem-3", task.visit, task.client, task.phone)
    r = await assistant.dialog.handle(state, "нет, вы ошиблись")
    assert r.end_call and "перезвонили" in r.text
    assert "терапевт" not in r.text and "завтра" not in r.text


async def test_reminder_not_for_fresh_visits(assistant, stand, clock):
    client = await stand.ensure_client("Иванов", "Сергей", datetime(1990, 1, 3).date())
    assistant.storage.save_phone(client.id, "+7 900 000-00-03")
    doctor = next(d for d in stand.doctors.values() if d.specialty == "Хирург")
    at = assistant.slots.schedule.free_slots(doctor.full_name, datetime(2026, 10, 16).date())[0]
    await stand.create_visit(client.id, doctor.id, at)  # created_at = сейчас, < 24 часов
    assert await due_reminders(stand, assistant.storage, clock()) == []


async def test_reschedule_visit_of_deactivated_doctor(call, stand):
    """К деактивированному врачу запись можно только отменить: отмена старой + новая к другому врачу."""
    c = await call()
    await book_therapist(c)
    old = next(iter(stand.visits.values()))
    stand.deactivate(old.doctor.id)
    c2 = await call()
    r = await c2.say("хочу перенести запись", PERSON)
    assert "больше не ведёт приём" in r.text
    assert all(s.doctor.id != old.doctor.id for s in c2.state.offered)
    await c2.say("первый", "да")
    assert stand.visits[old.id].status == VisitStatus.CANCELLED
    new = [v for v in stand.visits.values() if v.id != old.id]
    assert len(new) == 1 and new[0].status == VisitStatus.PLANNED and new[0].doctor.specialty == "Терапевт"
