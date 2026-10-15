"""Текстовый чат с ассистентом вместо телефона — для отладки сценариев.

    uv run python -m voice_helper.cli                # входящий звонок
    uv run python -m voice_helper.cli --fake         # на фейковом стенде, даже если токен задан
    uv run python -m voice_helper.cli --remind       # исходящие напоминания на сегодня
    uv run python -m voice_helper.cli --now "16.10.2026 11:00"   # подменить «сейчас» (GMT+4)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import uuid

from .app import build
from .models import parse_datetime, stand_now
from .reminders import due_reminders


def _print_reply(reply) -> None:
    print(f"\n🤖 {reply.text}")
    if reply.transfer:
        print("   [ПЕРЕВОД НА СОТРУДНИКА] карточка:")
        print("   " + json.dumps(reply.card, ensure_ascii=False, indent=2).replace("\n", "\n   "))
    if reply.end_call:
        print("   [ЗВОНОК ЗАВЕРШЁН]")


async def _converse(assistant, state, first) -> None:
    _print_reply(first)
    reply = first
    while not (reply.end_call or reply.transfer):
        try:
            text = input("\n👤 ")
        except (EOFError, KeyboardInterrupt):
            print()
            return
        reply = await assistant.dialog.handle(state, text)
        _print_reply(reply)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Текстовый чат с голосовым ассистентом регистратуры")
    parser.add_argument("--fake", action="store_true", help="фейковый стенд в памяти")
    parser.add_argument("--remind", action="store_true", help="обзвон с напоминаниями на сегодня")
    parser.add_argument("--phone", default="+7 900 000-00-01", help="номер звонящего (ANI)")
    parser.add_argument("--now", help="подменить текущее время стенда, ДД.ММ.ГГГГ ЧЧ:ММ")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    clock = (lambda: parse_datetime(args.now)) if args.now else stand_now
    assistant = build(fake=True if args.fake else None, clock=clock)
    kind = "фейковый стенд" if type(assistant.registry).__name__ == "FakeRegistry" else "стенд «Регистратура»"
    print(f"Сейчас {clock():%d.%m.%Y %H:%M} (GMT+4), {kind}. Ctrl+D — завершить.")

    if args.remind:
        tasks = await due_reminders(assistant.registry, assistant.storage, clock())
        if not tasks:
            print("На сегодня напоминаний нет.")
        for task in tasks:
            print(f"\n=== Звонок {task.phone}, запись #{task.visit.id} ===")
            state, first = await assistant.dialog.start_reminder(uuid.uuid4().hex[:12], task.visit, task.client, task.phone)
            await _converse(assistant, state, first)
        return

    state, first = await assistant.dialog.start_inbound(uuid.uuid4().hex[:12], args.phone)
    await _converse(assistant, state, first)


if __name__ == "__main__":
    asyncio.run(main())
