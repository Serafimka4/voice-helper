from __future__ import annotations

from datetime import date, datetime, time

import pytest

from voice_helper.nlu import (
    SpecialtyMatcher, Utterance, detect_intent, extract_birth_date, extract_date_pref, extract_name,
    extract_time_pref, yes_no,
)
from voice_helper.numbers import say_slot_full, time_words, words_to_digits

TODAY = date(2026, 10, 15)  # четверг


@pytest.mark.parametrize("spoken, digits", [
    ("двенадцатое марта тысяча девятьсот восемьдесят первого", "12 марта 1981"),
    ("седьмое апреля две тысячи девятнадцатого", "7 апреля 2019"),
    ("давайте в десять тридцать", "давайте в 10 30"),
    ("тридцать первого декабря", "31 декабря"),
])
def test_words_to_digits(spoken, digits):
    assert words_to_digits(spoken) == digits


@pytest.mark.parametrize("spoken, expected", [
    ("двенадцатое марта семьдесят восьмого", date(1978, 3, 12)),
    ("12.03.78", date(1978, 3, 12)),
    ("двенадцать ноль три семьдесят восемь", date(1978, 3, 12)),  # Приложение А.3
    ("пятое мая шестьдесят второго", date(1962, 5, 5)),
    ("седьмое апреля две тысячи девятнадцатого", date(2019, 4, 7)),
    ("тридцатое февраля девяностого", None),
])
def test_birth_dates(spoken, expected):
    assert extract_birth_date(Utterance.parse(spoken), TODAY) == expected


def test_names():
    p = extract_name(Utterance.parse("Смирнова Ольга Викторовна, двенадцатое марта семьдесят восьмого"))
    assert (p.last_name, p.first_name, p.middle_name) == ("Смирнова", "Ольга", "Викторовна")
    p = extract_name(Utterance.parse("иван иванович петров"))
    assert (p.last_name, p.first_name, p.middle_name) == ("Петров", "Иван", "Иванович")


@pytest.mark.parametrize("spoken, day", [
    ("завтра", date(2026, 10, 16)),
    ("в понедельник", date(2026, 10, 19)),
    ("в четверг", date(2026, 10, 22)),  # ближайший будущий, не сегодня
    ("на 3 ноября", date(2026, 11, 3)),
    ("двадцатого октября", date(2026, 10, 20)),
])
def test_dates(spoken, day):
    assert extract_date_pref(Utterance.parse(spoken), TODAY).day == day


def test_next_week_and_exclusions():
    pref = extract_date_pref(Utterance.parse("на следующей неделе, только не в пятницу"), TODAY)
    assert pref.date_from == date(2026, 10, 19) and pref.date_to == date(2026, 10, 25)
    assert pref.exclude_weekdays == (4,)


@pytest.mark.parametrize("spoken, at, part", [
    ("в половине десятого", time(9, 30), None),
    ("в пять вечера", time(17, 0), "evening"),
    ("после обеда", None, "day"),
    ("в четыре", time(16, 0), None),
])
def test_times(spoken, at, part):
    pref = extract_time_pref(Utterance.parse(spoken))
    assert pref.at == at and pref.part == part


def test_date_is_not_time():
    assert extract_time_pref(Utterance.parse("на 16 октября"), loose=True).at is None


@pytest.mark.parametrize("phrase, intent", [
    ("Запишите к терапевту", "book"),
    ("нужен талон к лору", "book"),
    ("Не могу прийти в четверг, давайте на другой день", "reschedule"),
    ("перезапишите меня", "reschedule"),
    ("Отмените запись", "cancel"),
    ("я не приду", "cancel"),
    ("На когда я записан?", "check"),
    ("Вызвать врача на дом", "home_visit"),
    ("Готовы ли мои анализы?", "analyses"),
])
def test_intents(phrase, intent):
    assert detect_intent(Utterance.parse(phrase)) == intent


def test_specialty_synonyms():
    m = SpecialtyMatcher(["Терапевт", "Лор (отоларинголог)", "Офтальмолог", "Детский невролог", "Невролог",
                          "Стоматолог-терапевт", "Акушер-гинеколог"])
    cases = {
        "к окулисту": "Офтальмолог", "к ушнику": "Лор (отоларинголог)", "к детскому неврологу": "Детский невролог",
        "к невропатологу": "Невролог", "к зубному": "Стоматолог-терапевт", "в женскую консультацию": "Акушер-гинеколог",
        "к участковому": "Терапевт", "к стоматологу-терапевту": "Стоматолог-терапевт",
    }
    for phrase, spec in cases.items():
        assert m.match(Utterance.parse(phrase)) == spec, phrase


def test_yes_no():
    assert yes_no(Utterance.parse("да, верно")) is True
    assert yes_no(Utterance.parse("нет, не подходит")) is False
    assert yes_no(Utterance.parse("а после обеда")) is None


def test_speech_formatting():
    assert time_words(time(9, 30)) == "девять тридцать"
    assert time_words(time(16, 0)) == "шестнадцать ноль-ноль"
    assert say_slot_full(datetime(2026, 10, 16, 8, 40), TODAY) == "завтра, в пятницу, шестнадцатого октября, в восемь сорок"
