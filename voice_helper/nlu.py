"""Понимание реплик пациента без LLM: интенты, специальности, даты, время, ФИО.

Правила взяты из регламента РГ-ТЕЛ-01 (каталог интентов, раздел 4; словарь синонимов,
Приложение А). Детерминированный разбор предсказуем и тестируем; LLM можно подключить
поверх как запасной разборщик для реплик, где правила ничего не нашли.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, time, timedelta

from .models import Doctor, normalize_name
from .numbers import _classify, words_to_digits

MONTHS = {
    "январ": 1, "феврал": 2, "март": 3, "апрел": 4, "ма": 5, "июн": 6,
    "июл": 7, "август": 8, "сентябр": 9, "октябр": 10, "ноябр": 11, "декабр": 12,
}
_MONTH_RE = r"(январ[яеь]|феврал[яеь]|марта?|апрел[яеь]|ма[яй]|июн[яеь]|июл[яеь]|августа?|сентябр[яеь]|октябр[яеь]|ноябр[яеь]|декабр[яеь])"
WEEKDAYS = [
    r"понедельник\w*", r"вторник\w*", r"сред[уаы]", r"четверг\w*", r"пятниц[уаы]", r"суббот[уаы]", r"воскресень[еяю]",
]


@dataclass
class Utterance:
    raw: str
    words: str  # нижний регистр, ё→е, без пунктуации, числа словами
    text: str   # то же, но числа цифрами

    @classmethod
    def parse(cls, raw: str) -> Utterance:
        s = raw.lower().replace("ё", "е")
        s = re.sub(r"(\d)[.:](\d)", r"\1․\2", s)  # сохранить 12.03.78 и 10:30
        s = re.sub(r"[^\w․+]+", " ", s).replace("․", ".")
        words = " ".join(s.split())
        text = words_to_digits(words)
        text = re.sub(r"\b0 0\b", "00", text)              # «шестнадцать ноль ноль»
        text = re.sub(r"\b0 ([1-9])\b", r"0\1", text)       # «двенадцать ноль три …» → 12 03
        return cls(raw, words, text)

    def has(self, pattern: str) -> bool:
        return re.search(pattern, self.words) is not None


def _first(pattern: str, s: str) -> re.Match | None:
    return re.search(pattern, s)


# --- интенты (раздел 4 регламента) ------------------------------------------

INTENTS: list[tuple[str, str]] = [
    ("robot", r"\b(вы|ты)\b.*\b(робот|бот|автомат|машина|живой человек)\b|\bэто робот\b"),
    ("no_recording", r"не записывайте|против записи|без записи разговора|выключите запись|отключите запись"),
    ("opt_out", r"не звоните (мне )?больше|больше не звоните|не надо (мне )?звонить"),
    ("gratitude", r"поблагодарить|выразить благодарность|спасибо (большое )?(врачу|доктору)"),
    ("home_visit", r"на дом\b|вызвать врача|вызов врача|чтобы (врач )?пришл|домой врача"),
    ("reschedule", r"перенес|перенос|перезапис|перезапиш|поменять (время|дату|день)|сдвинуть|на другой день|на другое время"),
    ("cancel", r"отмен|не приду|не смогу прийти|снять запись|снимите|вычеркните|не нужн\w* уже|уже не нужн|уже не надо"),
    ("check", r"(когда|на когда|на какое время|во сколько) (я |мы )?(записан|записана|записаны|мне к врачу|у меня (запись|прием|талон))"
              r"|какой (у меня )?кабинет|мои записи|проверить (мою |свою )?запись|куда я записан"
              r"|на когда записан|когда записан"),
    ("waitlist", r"лист ожидания|в очередь|позвоните когда появится"),
    ("analyses", r"готов\w* (ли )?(мои |мой )?(анализ|результат)|пришл\w* (ли )?(анализ|результат|кровь)|анализы пришли|результаты анализ"),
    ("schedule_info", r"когда (принимает|работает|ведет прием)|(в какие дни|по каким дням|какие часы) (принимает|работает)|график (работы )?(врача|хирурга|терапевта|педиатра)|работает ли (сегодня |завтра )?\w+"),
    ("book", r"запис|запиш|талон|номерок|попасть (к|на)|на прием|на когда есть|когда есть (время|свободн)|хочу к |нужно к |мне к |нужен (врач|доктор)"),
    ("goodbye", r"^(нет )?(спасибо|благодарю)( большое)?( нет)?( все| ничего)?$|до свидания|всего доброго|больше ничего|ничего не нужно|^нет$|^нет спасибо|это все"),
]


def detect_intent(u: Utterance) -> str | None:
    for name, pattern in INTENTS:
        if u.has(pattern):
            return name
    return None


# --- да / нет / выбор --------------------------------------------------------

_NO = r"\b(нет|неа|не надо|не нужно|не верно|неверно|неправильно|не подходит|не устраивает|не хочу|не получится|не смогу|ни то ни другое|не то)\b"
_YES = (r"\b(да|ага|угу|конечно|верно|правильно|подтверждаю|хорошо|давайте|давай|согласен|согласна|точно|"
        r"приду|буду|ладно|окей|ок|можно|записывайте|запишите|подходит|устраивает|годится|отлично|переносим|отменяйте)\b")


def yes_no(u: Utterance) -> bool | None:
    if re.search(_NO, u.words):
        return False
    if re.search(_YES, u.words):
        return True
    return None


def choice_index(u: Utterance, n: int) -> int | None:
    """«первый», «второй вариант», «последний» → индекс в списке из n вариантов."""
    if re.search(_MONTH_RE, u.words):
        return None
    m = re.search(r"\b(перв|втор|трет|последн)\w*", u.words)
    if not m:
        return None
    idx = {"перв": 0, "втор": 1, "трет": 2, "последн": n - 1}[m.group(1)]
    return idx if 0 <= idx < n else None


# --- специальности (Приложение А.1) -----------------------------------------

# Разговорные названия → специальность справочника Excel/стенда
SPECIALTY_SYNONYMS: list[tuple[str, str]] = [
    (r"\bлор\w*|ухо горло нос|ушник|отоларинголог|оториноларинголог", "Лор (отоларинголог)"),
    (r"окулист|глазн(ой|ому|ого|ик\w*)|проверить зрение", "Офтальмолог"),
    (r"гинеколог|женск\w* (врач|консультац)", "Акушер-гинеколог"),
    (r"удалить зуб|удаление зуба", "Стоматолог-хирург"),
    (r"стоматолог|зубн\w*|дантист|запломбир|пломб|зуб\w*", "Стоматолог-терапевт"),
    (r"невропатолог|нервн\w* врач", "Невролог"),
    (r"сердечн\w* врач|по сердцу", "Кардиолог"),
    (r"по щитовидке|по сахару", "Эндокринолог"),
    (r"мужск\w* врач", "Уролог"),
    (r"родинк|вросш\w* ногот|вскрыть", "Хирург"),
    (r"детск\w* врач|врач\w* (с|для) ребенк", "Педиатр"),
    (r"участков\w*|терапевт\w*|сво(й|ему|его) врач\w*|общ(ий|ему|его) врач\w*", "Терапевт"),
]

# Ребёнку — детский профиль (отделения поликлиники, раздел 2.3)
ADULT_TO_CHILD = {
    "терапевт": "Педиатр",
    "лор (отоларинголог)": "Детский отоларинголог",
    "офтальмолог": "Детский офтальмолог",
    "хирург": "Детский хирург",
    "стоматолог-терапевт": "Детский стоматолог",
    "стоматолог-хирург": "Детский стоматолог",
    "невролог": "Детский невролог",
    "кардиолог": "Детский кардиолог",
    "эндокринолог": "Детский эндокринолог",
    "уролог": "Детский уролог-андролог",
    "травматолог": "Детский ортопед-травматолог",
    "аллерголог-иммунолог": "Детский аллерголог-иммунолог",
}
CHILD_TO_ADULT = {normalize_name(v): k for k, v in ADULT_TO_CHILD.items() if k not in ("стоматолог-хирург",)}

# Порядок записи (раздел 2.3): прямая / только платно / по направлению (все остальные)
DIRECT_SPECIALTIES = {
    "терапевт", "хирург", "офтальмолог", "лор (отоларинголог)", "педиатр", "детский хирург",
    "детский офтальмолог", "детский отоларинголог", "акушер-гинеколог", "стоматолог-терапевт",
    "стоматолог-хирург", "детский стоматолог",
}
PAID_ONLY_SPECIALTIES = {"стоматолог-ортопед", "ортодонт"}


def _stem(word: str) -> str:
    return word if len(word) <= 5 else word[: max(4, len(word) - 2)]


class SpecialtyMatcher:
    """Находит специальность из справочника врачей по реплике пациента."""

    def __init__(self, specialties: list[str]):
        self.specialties = sorted(set(specialties), key=lambda s: -len(s.split()))
        self._by_key = {normalize_name(s): s for s in self.specialties}
        self._patterns: list[tuple[re.Pattern, str]] = []
        for s in self.specialties:
            words = [w for w in re.split(r"[\s\-()]+", normalize_name(s)) if w and w != "врач"]
            stems = [_stem(w) for w in words]
            # Все основы слов специальности в любом месте фразы: «детскому неврологу»
            pattern = "".join(rf"(?=.*\b{re.escape(st)})" for st in stems)
            self._patterns.append((re.compile(pattern), s))

    def canonical(self, name: str) -> str | None:
        return self._by_key.get(normalize_name(name))

    def match(self, u: Utterance) -> str | None:
        for pattern, s in self._patterns:
            if pattern.search(u.words):
                # «терапевт» не должен совпадать внутри «стоматолог-терапевт»
                if normalize_name(s) == "терапевт" and re.search(r"стоматолог", u.words):
                    continue
                return s
        for pattern, s in SPECIALTY_SYNONYMS:
            if re.search(pattern, u.words) and (canon := self.canonical(s)):
                return canon
        return None


def match_doctors(u: Utterance, doctors: list[Doctor]) -> list[Doctor]:
    """Врачи, чья фамилия прозвучала в реплике («к Кузнецовой» → Кузнецова)."""
    found = []
    for d in doctors:
        last = normalize_name(d.last_name)
        if re.search(rf"\b{re.escape(_stem(last))}\w*\b", u.words):
            found.append(d)
    return found


# --- даты и время (Приложение А.3) -------------------------------------------


@dataclass
class DatePref:
    day: date | None = None
    date_from: date | None = None
    date_to: date | None = None
    exclude_weekdays: tuple[int, ...] = ()

    @property
    def empty(self) -> bool:
        return not (self.day or self.date_from or self.exclude_weekdays)


def _month_num(word: str) -> int:
    for stem, n in sorted(MONTHS.items(), key=lambda kv: -len(kv[0])):
        if word.startswith(stem):
            return n
    raise ValueError(word)


def _future_date(day: int, month: int, today: date) -> date | None:
    for year in (today.year, today.year + 1):
        try:
            d = date(year, month, day)
        except ValueError:
            return None
        if d >= today:
            return d
    return None


def extract_date_pref(u: Utterance, today: date) -> DatePref:
    t = u.text
    pref = DatePref()

    excl = re.findall(r"\b(?:не|кроме|только не)\s+(?:в\s+|во\s+)?(" + "|".join(WEEKDAYS) + r")", t)
    if excl:
        pref.exclude_weekdays = tuple(
            i for i, p in enumerate(WEEKDAYS) for w in excl if re.fullmatch(p, w)
        )
        t = re.sub(r"\b(?:не|кроме|только не)\s+(?:в\s+|во\s+)?(" + "|".join(WEEKDAYS) + r")", " ", t)

    if re.search(r"\bпослезавтра\b", t):
        pref.day = today + timedelta(days=2)
    elif re.search(r"\bзавтра\b", t):
        pref.day = today + timedelta(days=1)
    elif re.search(r"\bсегодня\b", t):
        pref.day = today
    elif m := re.search(r"\bчерез (\d+) (дн|ден)", t):
        pref.day = today + timedelta(days=int(m.group(1)))
    elif re.search(r"\bчерез неделю\b", t):
        pref.day = today + timedelta(days=7)
    elif re.search(r"следующ\w* недел|после выходных", t):
        monday = today + timedelta(days=7 - today.weekday())
        pref.date_from, pref.date_to = monday, monday + timedelta(days=6)
        if re.search(r"после выходных", t):
            pref.day, pref.date_from, pref.date_to = monday, None, None
    elif re.search(r"(на )?этой неделе", t):
        pref.date_from, pref.date_to = today, today + timedelta(days=6 - today.weekday())

    if m := re.search(r"\b(\d{1,2}) " + _MONTH_RE, t):
        pref.day = _future_date(int(m.group(1)), _month_num(m.group(2)), today) or pref.day
    elif m := re.search(r"\b(\d{1,2})\.(\d{1,2})\b(?!\.\d)", t):
        pref.day = _future_date(int(m.group(1)), int(m.group(2)), today) or pref.day
    elif m := re.search(r"\b(\d{1,2}) числа\b", t):
        d = int(m.group(1))
        pref.day = _future_date(d, today.month, today) or _future_date(d, today.month % 12 + 1, today)
    elif not pref.day:
        for i, p in enumerate(WEEKDAYS):
            if re.search(rf"\b{p}\b", t):
                ahead = (i - today.weekday()) % 7 or 7  # «в понедельник» — ближайший будущий
                pref.day = today + timedelta(days=ahead)
                pref.date_from = pref.date_to = None
                break
    return pref


@dataclass
class TimePref:
    at: time | None = None
    part: str | None = None  # morning / day / evening
    late: bool = False       # «попозже»
    early: bool = False      # «пораньше»

    @property
    def empty(self) -> bool:
        return self.at is None and self.part is None and not self.late and not self.early


_NOT_TIME = r"(?!\s*(?:" + _MONTH_RE + r"|числа|дн|ден|недел|месяц|лет|год|минут|раз|\.\d))"


def _hour_fix(h: int, u: Utterance) -> int:
    if u.has(r"вечера|дня") and h < 12:
        return h + 12
    if h < 7 and not u.has(r"утра|ночи"):
        return h + 12  # «в четыре» в поликлинике — 16:00
    return h


def extract_time_pref(u: Utterance, loose: bool = False) -> TimePref:
    t = u.text
    pref = TimePref()
    if u.has(r"\bутр|с утра"):
        pref.part = "morning"
    if u.has(r"\bдн[её]м\b|после обеда|в обед|середин\w* дня"):
        pref.part = "day"
    if u.has(r"\bвечер"):
        pref.part = "evening"
    pref.late = u.has(r"попозже|позднее|последн\w* (время|прием)")
    pref.early = u.has(r"пораньше|самое ранн|как можно раньше")

    m = re.search(r"\bв половине (\d{1,2})\b", t)
    if m:
        h = _hour_fix(int(m.group(1)) - 1, u)
        pref.at = time(h % 24, 30)
        return pref
    m = re.search(r"\b(?:в|на|к|около|часов в)\s+(\d{1,2})(?:[:.\s](\d{2}))?\b" + _NOT_TIME, t)
    if not m and loose:
        # В ответ на «Во сколько удобно?»: «давайте 8 40», «десять», «в 9 часов»
        m = re.search(r"\b(\d{1,2})[:.\s](\d{2})\b" + _NOT_TIME, t) or re.fullmatch(
            r"(?:давайте |можно |ну )?(\d{1,2})(?: час\w*)?", t
        )
    if m:
        h = int(m.group(1))
        minute = int(m.group(2)) if m.lastindex and m.lastindex >= 2 and m.group(2) else 0
        if 0 <= h <= 23 and 0 <= minute <= 59:
            pref.at = time(_hour_fix(h, u) % 24, minute)
    return pref


# --- ФИО и дата рождения (раздел 3.5) ----------------------------------------

_NAME_STOP = set(
    """меня зовут моя мое мой фамилия имя отчество это я да нет ну вот так пожалуйста дата даты рождения родился
    родилась года год г число мне записаться хочу к на в и а его ее ребенка ребенок дочь дочка дочку сын сына сыночка
    мама маму мамы папа папу жена жену муж мужа значит э эм алло слушаю здравствуйте добрый день вечер утро спасибо
    ой уже тоже записать запишите пациент пациента зовут фио полностью сейчас скажу тысяча давайте вариант хорошо
    секунду минуту подождите щас ага угу""".split()
)
_PATRONYMIC = re.compile(r"(вич|вна|ична|инична|оглы|кызы)$")


@dataclass
class Person:
    last_name: str | None = None
    first_name: str | None = None
    middle_name: str | None = None
    birth_date: date | None = None

    @property
    def complete(self) -> bool:
        return bool(self.last_name and self.first_name and self.birth_date)

    def merge(self, other: Person) -> None:
        for f in ("last_name", "first_name", "middle_name", "birth_date"):
            if getattr(other, f):
                setattr(self, f, getattr(other, f))


def _cap(word: str) -> str:
    return "-".join(p.capitalize() for p in word.split("-"))


def extract_name(u: Utterance) -> Person:
    tokens = []
    for w in u.words.split():
        if w in _NAME_STOP or not re.fullmatch(r"[а-я][а-я\-]+", w) or re.match(_MONTH_RE + r"$", w):
            continue
        if _first(r"^(" + "|".join(WEEKDAYS) + r")$", w) or w.startswith(("терапевт", "педиатр", "врач")):
            continue
        if _classify(w) is not None:  # числительные из даты рождения
            continue
        tokens.append(w)
    tokens = tokens[:3]
    p = Person()
    if len(tokens) >= 3 and _PATRONYMIC.search(tokens[1]) and not _PATRONYMIC.search(tokens[2]):
        p.first_name, p.middle_name, p.last_name = tokens[0], tokens[1], tokens[2]  # «Иван Иванович Петров»
    elif len(tokens) >= 2:
        p.last_name, p.first_name = tokens[0], tokens[1]
        if len(tokens) == 3 and _PATRONYMIC.search(tokens[2]):
            p.middle_name = tokens[2]
    elif len(tokens) == 1:
        p.last_name = tokens[0]
    for f in ("last_name", "first_name", "middle_name"):
        if getattr(p, f):
            setattr(p, f, _cap(getattr(p, f)))
    return p


def _restore_year(y: int, today: date) -> int:
    """Год из двух цифр: пациенту не может быть больше 110 лет (Приложение А.3)."""
    if y >= 100:
        return y
    for century in (2000, 1900):
        if century + y <= today.year and today.year - (century + y) <= 110:
            return century + y
    return 1900 + y


def extract_birth_date(u: Utterance, today: date) -> date | None:
    t = u.text
    m = (re.search(r"\b(\d{1,2})[.\s/-](\d{1,2})[.\s/-](\d{2}|\d{4})\b", t)
         or re.search(r"\b(\d{1,2}) " + _MONTH_RE + r" (\d{2}|\d{4})\b", t))
    if not m:
        return None
    day, month_raw, year = m.group(1), m.group(2), int(m.group(3))
    month = int(month_raw) if month_raw.isdigit() else _month_num(month_raw)
    try:
        d = date(_restore_year(year, today), month, int(day))
    except ValueError:
        return None
    return d if d <= today else None


def extract_person(u: Utterance, today: date) -> Person:
    p = extract_name(u)
    p.birth_date = extract_birth_date(u, today)
    return p


def age_on(birth: date, today: date) -> int:
    return today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))


# --- прочее ------------------------------------------------------------------

CHILD_RE = r"\b(ребен\w*|реб[её]нк\w*|дочь|дочк\w*|дочер\w*|сын\w*|сыночк\w*|малыш\w*|внук\w*|внучк\w*)\b"
RELATIVE_ADULT_RE = r"\b(мам[аыуе]|пап[аыуе]|жен[аыуе]|муж[ауе]?|бабушк\w*|дедушк\w*|сестр\w*|брат\w*|сосед\w*|знаком\w*)\b"
FIRST_PERSON_RE = r"\b(я|мне|меня|мой|моя|мою|мои|у меня|мы|нас)\b"


@dataclass
class Entities:
    specialty: str | None = None
    doctors: list[Doctor] = field(default_factory=list)
    date: DatePref = field(default_factory=DatePref)
    time: TimePref = field(default_factory=TimePref)
    person: Person = field(default_factory=Person)
