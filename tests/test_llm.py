from __future__ import annotations

from voice_helper.llm import PII_STEPS, mask_pii


class FakeLLM:
    def __init__(self, answers: dict[str, str]):
        self.answers = answers
        self.calls: list[tuple[str, str]] = []

    async def rewrite(self, text: str, step: str, last_prompt: str) -> str | None:
        self.calls.append((text, step))
        if step in PII_STEPS:
            return None
        return self.answers.get(text)


def test_mask_pii():
    masked = mask_pii("мой номер +7 917 123-45-67, родилась 12.03.1978, а ещё 5 мая 1962")
    assert "917" not in masked and "1978" not in masked and "1962" not in masked
    assert masked.count("[ДАТА]") == 2 and "[ТЕЛЕФОН]" in masked


async def test_llm_rescues_unparsed_phrase(call, assistant):
    llm = FakeLLM({"мне бы к доктору, который ухо смотрит": "записаться к ЛОРу"})
    assistant.dialog.llm = llm
    c = await call()
    r = await c.say("мне бы к доктору, который ухо смотрит")
    assert "фамилию" in r.text and c.state.specialty == "Лор (отоларинголог)"
    assert len(llm.calls) == 1


async def test_llm_choice_in_slot_step(call, assistant):
    c = await call()
    await c.say("к хирургу", "Смирнова Ольга Викторовна, двенадцатое марта семьдесят восьмого")
    last = c.state.offered[-1]
    assistant.dialog.llm = FakeLLM({"ну давайте тот, что самый поздний из названных": "последний"})
    r = await c.say("ну давайте тот, что самый поздний из названных")
    assert r.text.startswith("Записываю") and c.state.chosen == last


async def test_llm_not_called_with_personal_data(call, assistant):
    llm = FakeLLM({})
    assistant.dialog.llm = llm
    c = await call()
    await c.say("запишите к терапевту", "ммм секундочку")  # второй ответ — на шаге идентификации
    assert llm.calls == []


async def test_interpreter_parses_structured_answer(monkeypatch):
    from types import SimpleNamespace
    from voice_helper.llm import LLMInterpreter, LLMSettings

    seen = {}

    async def fake_create(**kwargs):
        seen.update(kwargs)
        text = '{"understood": true, "canonical": "последний"}'
        return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text=text)])

    llm = LLMInterpreter(LLMSettings(api_key="sk-hub-test"), ["Терапевт", "Хирург"])
    monkeypatch.setattr(llm.client.messages, "create", fake_create)
    assert await llm.rewrite("тот что позже, 89171234567", "choose_slot", "Что удобнее?") == "последний"
    assert seen["model"] == "claude-opus-5-5"
    assert "89171234567" not in seen["messages"][0]["content"]
    assert "Терапевт, Хирург" in seen["system"]
    assert await llm.rewrite("Иванов Иван", "identify", "") is None


async def test_llm_failure_falls_back_to_rules(call, assistant):
    assistant.dialog.llm = FakeLLM({})  # ничего не поняла
    c = await call()
    r = await c.say("бу-бу-бу")
    assert "не совсем понял" in r.text
