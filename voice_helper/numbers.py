"""Числа в русской речи: слова → цифры (для распознанного текста) и цифры → слова (для синтеза).

Регламент требует произносить время словами («в девять тридцать»), даты — с днём
недели («в четверг, первого октября»), а пациент диктует даты рождения как
«двенадцатое марта семьдесят восьмого».
"""
from __future__ import annotations

import re
from datetime import date, datetime, time

# --- слова → цифры ----------------------------------------------------------

_UNITS = {
    "ноль": 0, "один": 1, "одна": 1, "одно": 1, "два": 2, "две": 2, "три": 3, "четыре": 4,
    "пять": 5, "шесть": 6, "семь": 7, "восемь": 8, "девять": 9, "десять": 10,
    "одиннадцать": 11, "двенадцать": 12, "тринадцать": 13, "четырнадцать": 14,
    "пятнадцать": 15, "шестнадцать": 16, "семнадцать": 17, "восемнадцать": 18,
    "девятнадцать": 19,
}
_TENS = {
    "двадцать": 20, "тридцать": 30, "сорок": 40, "пятьдесят": 50,
    "шестьдесят": 60, "семьдесят": 70, "восемьдесят": 80, "девяносто": 90,
}
_HUNDREDS = {
    "сто": 100, "двести": 200, "триста": 300, "четыреста": 400, "пятьсот": 500,
    "шестьсот": 600, "семьсот": 700, "восемьсот": 800, "девятьсот": 900,
}
_THOUSAND = {"тысяча", "тысячи", "тысяч"}

# Основы порядковых числительных (первого, двенадцатое, восьмидесятого, двухтысячного…)
_ORDINAL_STEMS = [
    ("двухтысячн", 2000, 3), ("тысячн", 1000, 3),
    ("девятисот", 900, 2), ("восьмисот", 800, 2), ("семисот", 700, 2), ("шестисот", 600, 2),
    ("пятисот", 500, 2), ("четырехсот", 400, 2), ("трехсот", 300, 2), ("двухсот", 200, 2), ("сот", 100, 2),
    ("двадцат", 20, 1), ("тридцат", 30, 1), ("сороков", 40, 1), ("пятидесят", 50, 1),
    ("шестидесят", 60, 1), ("семидесят", 70, 1), ("восьмидесят", 80, 1), ("девяност", 90, 1),
    ("одиннадцат", 11, 0), ("двенадцат", 12, 0), ("тринадцат", 13, 0), ("четырнадцат", 14, 0),
    ("пятнадцат", 15, 0), ("шестнадцат", 16, 0), ("семнадцат", 17, 0), ("восемнадцат", 18, 0),
    ("девятнадцат", 19, 0), ("десят", 10, 0),
    ("перв", 1, 0), ("втор", 2, 0), ("трет", 3, 0), ("четверт", 4, 0), ("пят", 5, 0),
    ("шест", 6, 0), ("седьм", 7, 0), ("восьм", 8, 0), ("девят", 9, 0),
]
_ORDINAL_END = re.compile(r"(ый|ой|ий|ое|ее|ая|яя|ого|его|ому|ему|ым|им|ом|ем|ую|юю|ых|их|ье|ьего|ьему|ьи|ья)$")


def _classify(word: str) -> tuple[int, int, bool] | None:
    """(значение, разряд, порядковое ли). Разряд: 0 — единицы/десятки до 19, 1 — десятки, 2 — сотни, 3 — тысячи."""
    if word in _UNITS:
        return _UNITS[word], 0, False
    if word in _TENS:
        return _TENS[word], 1, False
    if word in _HUNDREDS:
        return _HUNDREDS[word], 2, False
    if word in _THOUSAND:
        return 1000, 3, False
    for stem, value, rank in _ORDINAL_STEMS:
        if word.startswith(stem) and _ORDINAL_END.search(word[len(stem):] or "x") and len(word) - len(stem) <= 4:
            return value, rank, True
    return None


def words_to_digits(text: str) -> str:
    """«двенадцатое марта тысяча девятьсот восемьдесят первого» → «12 марта 1981».

    «десять тридцать» остаётся двумя числами (10 30): после «десяти» не может идти «тридцать».
    """
    out: list[str] = []
    total = 0      # накопленное значение (тысячи уже умножены)
    group = 0      # текущая группа до тысяч
    last_rank = 9  # разряд последнего слова в числе
    in_number = False

    def flush() -> None:
        nonlocal total, group, last_rank, in_number
        if in_number:
            out.append(str(total + group))
        total, group, last_rank, in_number = 0, 0, 9, False

    for word in text.split():
        c = _classify(word)
        if c is None:
            flush()
            out.append(word)
            continue
        value, rank, ordinal = c
        if rank == 3 and not ordinal:
            if in_number and total == 0 and group < 1000:
                total, group = (group or 1) * 1000, 0
                last_rank = 3
                continue
            flush()
            total, last_rank, in_number = 1000, 3, True
            continue
        if rank == 3 and ordinal:  # «двухтысячного», «тысячного»
            fits = in_number and group == 0 and total == 0
            if not fits:
                flush()
            total, in_number = value, True
            flush()
            continue
        fits = in_number and rank < last_rank and not (last_rank == 1 and value >= 10)
        if not fits:
            flush()
        group += value
        last_rank, in_number = rank, True
        if ordinal:
            flush()
    flush()
    return " ".join(out)


# --- цифры → слова ----------------------------------------------------------

_CARD = [
    "ноль", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять", "десять",
    "одиннадцать", "двенадцать", "тринадцать", "четырнадцать", "пятнадцать", "шестнадцать",
    "семнадцать", "восемнадцать", "девятнадцать",
]
_CARD_TENS = {20: "двадцать", 30: "тридцать", 40: "сорок", 50: "пятьдесят", 60: "шестьдесят",
              70: "семьдесят", 80: "восемьдесят", 90: "девяносто"}
_CARD_HUNDREDS = {v: k for k, v in _HUNDREDS.items()}

_ORD_GEN = [
    "", "первого", "второго", "третьего", "четвёртого", "пятого", "шестого", "седьмого", "восьмого",
    "девятого", "десятого", "одиннадцатого", "двенадцатого", "тринадцатого", "четырнадцатого",
    "пятнадцатого", "шестнадцатого", "семнадцатого", "восемнадцатого", "девятнадцатого", "двадцатого",
]
MONTHS_GEN = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа",
              "сентября", "октября", "ноября", "декабря"]
WEEKDAYS_ACC = ["в понедельник", "во вторник", "в среду", "в четверг", "в пятницу", "в субботу", "в воскресенье"]


def number_words(n: int) -> str:
    if n < 20:
        return _CARD[n]
    if n < 100:
        tens, unit = divmod(n, 10)
        return _CARD_TENS[tens * 10] + (f" {_CARD[unit]}" if unit else "")
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        return _CARD_HUNDREDS[hundreds * 100] + (f" {number_words(rest)}" if rest else "")
    return str(n)


def day_ordinal_gen(n: int) -> str:
    """15 → «пятнадцатого», 31 → «тридцать первого»."""
    if n <= 20:
        return _ORD_GEN[n]
    if n == 30:
        return "тридцатого"
    tens, unit = divmod(n, 10)
    return f"{_CARD_TENS[tens * 10]} {_ORD_GEN[unit]}"


def time_words(t: time | datetime) -> str:
    """9:30 → «девять тридцать», 16:00 → «шестнадцать ноль-ноль», 8:05 → «восемь ноль пять»."""
    hours = number_words(t.hour)
    if t.minute == 0:
        return f"{hours} ноль-ноль"
    if t.minute < 10:
        return f"{hours} ноль {number_words(t.minute)}"
    return f"{hours} {number_words(t.minute)}"


def _relative(d: date, today: date) -> str | None:
    return {0: "сегодня", 1: "завтра", 2: "послезавтра"}.get((d - today).days)


def say_date(d: date, today: date) -> str:
    """«завтра, в четверг, шестнадцатого октября» / «в понедельник, девятнадцатого октября»."""
    base = f"{WEEKDAYS_ACC[d.weekday()]}, {day_ordinal_gen(d.day)} {MONTHS_GEN[d.month - 1]}"
    rel = _relative(d, today)
    return f"{rel}, {base}" if rel else base


def say_slot_short(dt: datetime, today: date) -> str:
    """Короткий вариант для перечисления: «завтра в восемь сорок», «в пятницу, семнадцатого, в десять тридцать»."""
    rel = _relative(dt.date(), today)
    if rel:
        return f"{rel} в {time_words(dt)}"
    return f"{WEEKDAYS_ACC[dt.weekday()]}, {day_ordinal_gen(dt.day)}, в {time_words(dt)}"


def say_slot_full(dt: datetime, today: date) -> str:
    return f"{say_date(dt.date(), today)}, в {time_words(dt)}"


_HOUR_GEN = {
    0: "нуля", 1: "часа", 2: "двух", 3: "трёх", 4: "четырёх", 5: "пяти", 6: "шести", 7: "семи", 8: "восьми",
    9: "девяти", 10: "десяти", 11: "одиннадцати", 12: "двенадцати", 13: "тринадцати", 14: "четырнадцати",
    15: "пятнадцати", 16: "шестнадцати", 17: "семнадцати", 18: "восемнадцати", 19: "девятнадцати",
    20: "двадцати", 21: "двадцати одного", 22: "двадцати двух", 23: "двадцати трёх",
}
WEEKDAYS_ACC_BARE = ["понедельник", "вторник", "среду", "четверг", "пятницу", "субботу", "воскресенье"]


def _hour_gen(hhmm: str) -> str:
    h, m = (int(x) for x in hhmm.split(":"))
    return _HOUR_GEN[h] + (f" {number_words(m)}" if m else "")


def hours_range_words(spec: str) -> str:
    """«09:00–13:00, 14:00–18:00» → «с девяти до тринадцати и с четырнадцати до восемнадцати»."""
    parts = []
    for rng in spec.split(","):
        start, end = (x.strip() for x in re.split(r"[–-]", rng.strip()))
        parts.append(f"с {_hour_gen(start)} до {_hour_gen(end)}")
    return " и ".join(parts)


def weekdays_words(days: list[int]) -> str:
    """[0, 2, 4] → «в понедельник, среду и пятницу»; Пн–Пт → «по будням»."""
    days = sorted(days)
    if days == [0, 1, 2, 3, 4]:
        return "по будням"
    if days == [0, 1, 2, 3, 4, 5]:
        return "с понедельника по субботу"
    names = [WEEKDAYS_ACC_BARE[d] for d in days]
    joined = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " и " + names[-1]
    return ("во " if days[0] == 1 else "в ") + joined
