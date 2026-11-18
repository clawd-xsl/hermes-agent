"""Question cards on Aleph (protocol "clarify.request", "clarify.respond"), against Hermes's own
clarify registry; cards in history pages; and every card settling, also across a restart."""

from __future__ import annotations

import pytest
import pytest_asyncio

from gateway.config import PlatformConfig
from tests.gateway.aleph_phone import connect, pair_device


def _config(state):
    return PlatformConfig(enabled=True, extra={"host": "127.0.0.1", "port": 0, "state_dir": str(state)})


@pytest_asyncio.fixture
async def adapter(tmp_path):
    from plugins.platforms.aleph.adapter import AlephAdapter
    instance = AlephAdapter(_config(tmp_path / "aleph"))

    async def handler(event):
        return None
    instance.set_message_handler(handler)
    assert await instance.connect()
    instance.device, instance.keys = pair_device(instance)
    yield instance
    await instance.disconnect()


@pytest.fixture
def clarify():
    import tools.clarify_gateway as clarify
    clarify.asked = []
    yield clarify
    for session_key in clarify.asked:
        clarify.clear_session(session_key)


async def _ask(adapter, phone, clarify, clarify_id, question, choices, *, multi=False, series=None):
    session_key = f"sk-{clarify_id}"  # its own session: Hermes answers a session's oldest pending one
    clarify.asked.append(session_key)
    entry = clarify.register(clarify_id, session_key, question, choices, multi_select=multi)
    metadata = {"_clarify_series": {"index": series[0], "count": series[1]}} if series else None
    await adapter.send_clarify("main", question, choices, clarify_id, session_key, metadata)
    while (frame := (await phone.event())["params"])["type"] != "clarify.request":
        pass
    return entry, frame["payload"]


async def _resolved(phone):
    while (frame := (await phone.event())["params"])["type"] != "clarify.resolved":
        pass
    payload = dict(frame["payload"])
    payload.pop("chat")
    return payload


@pytest.mark.asyncio
async def test_a_question_card_and_its_answers(adapter, clarify):
    from agent.i18n import t
    phone = await connect(adapter, adapter.keys)
    try:
        await phone.hello()
        entry, card = await _ask(adapter, phone, clarify, "q1", f"Which env?\n{t('gateway.clarify.skip_hint')}",
                                 ["staging (Recommended)", "prod"], series=(0, 2))
        created = card.pop("created_at")
        assert created and card == {"chat": "main", "clarify_id": "q1", "question": "Which env?",
                                    "choices": ["staging", "prod"], "multi_select": False, "recommended": 0,
                                    "index": 0, "count": 2}
        assert entry.awaiting_text  # a typed message in the chat answers it
        agent_message = (await adapter.send("main", "An agent message.")).message_id
        for bad in ({"choices": [2]}, {"choices": []}, {"choices": [0, 1]}, {"choices": ["0"]},
                    {"choice": 1}, {"text": "prod"}, {"skip": False}):
            answer = await phone.call("clarify.respond", {"clarify_id": "q1", "message_id": "a-bad", **bad})
            assert answer["error"]["code"] == -32602, bad
        for message_id in (None, "", agent_message):  # required with choices; never another's id
            answer = await phone.call("clarify.respond", {"clarify_id": "q1", "choices": [1], "message_id": message_id})
            assert answer["error"]["code"] == -32602, message_id
        pick = {"clarify_id": "q1", "choices": [1], "message_id": "a-1"}
        assert (await phone.call("clarify.respond", pick))["result"] == {}
        assert entry.response == "prod"
        # The pick is the person's message, logged before the card settles, and read at once.
        logged = []
        while not logged or logged[-1]["type"] != "clarify.resolved":
            logged.append((await phone.logged())["params"])
        said = [p["payload"] for p in logged if p["type"] == "message.new" and p["payload"]["sender"] == "user"]
        assert [(m["id"], m["text"]) for m in said] == [("a-1", "prod")]
        types = [p["type"] for p in logged]
        assert types.index("message.new") < types.index("clarify.resolved")
        resolved = dict(logged[-1]["payload"])
        resolved.pop("chat")
        assert resolved == {"clarify_id": "q1", "answer": "prod", "choices": [1], "by": "user"}
        assert adapter.store.history("main", 10**6, 5)[0][-1]["read"] is True
        latest = adapter.store.latest()
        assert (await phone.call("clarify.respond", pick))["result"] == {}  # a resend is the same message
        assert adapter.store.latest() == latest

        entry, card = await _ask(adapter, phone, clarify, "q2", "Where?", ["a", "b", "c"], multi=True)
        assert card["multi_select"] is True and "recommended" not in card and "index" not in card
        await phone.call("clarify.respond", {"clarify_id": "q2", "choices": [0, 2], "message_id": "a-2"})
        assert entry.response == '["a", "c"]' and adapter.store.message("a-2").text == "a, c"
        assert await _resolved(phone) == {"clarify_id": "q2", "answer": "a, c", "choices": [0, 2], "by": "user"}

        entry, _ = await _ask(adapter, phone, clarify, "q3", "Next one?", ["x", "y"])
        latest = adapter.store.latest()
        await phone.call("clarify.respond", {"clarify_id": "q3", "skip": True})
        assert entry.response == clarify.SKIPPED
        assert await _resolved(phone) == {"clarify_id": "q3", "answer": None, "choices": None, "by": "user"}
        assert [e.type for e in adapter.store.events_after(latest)] == ["clarify.resolved"]  # a skip says nothing
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_a_typed_answer_and_no_answer(adapter, clarify):
    """What the gateway does with a typed message (then retires the card), and a card that ends unanswered."""
    phone = await connect(adapter, adapter.keys)
    try:
        await phone.hello()
        await _ask(adapter, phone, clarify, "t1", "Which env?", ["staging", "prod"])
        assert clarify.resolve_text_response_for_session("sk-t1", "2")
        await adapter.retire_clarify_card("t1", "answered")
        assert await _resolved(phone) == {"clarify_id": "t1", "answer": "prod", "choices": [1], "by": "user"}

        await _ask(adapter, phone, clarify, "t2", "Which env?", ["staging", "prod"])
        assert clarify.resolve_text_response_for_session("sk-t2", "neither, use dev")  # prose answers it too
        await adapter.retire_clarify_card("t2", "answered")
        assert await _resolved(phone) == {"clarify_id": "t2", "answer": "neither, use dev", "choices": None,
                                          "by": "user"}

        await _ask(adapter, phone, clarify, "t3", "Still there?", ["yes", "no"])
        clarify.clear_session("sk-t3")  # a reset: CANCELLED
        await adapter.retire_clarify_card("t3", "expired")
        assert await _resolved(phone) == {"clarify_id": "t3", "answer": None, "choices": None, "by": "timeout"}
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_cards_are_in_history_pages_in_their_place(adapter, clarify):
    phone = await connect(adapter, adapter.keys)
    try:
        await phone.hello()
        await adapter.send("main", "Before.")
        await _ask(adapter, phone, clarify, "h1", "Which env?", ["staging", "prod"])
        await phone.call("clarify.respond", {"clarify_id": "h1", "choices": [0], "message_id": "h-answer"})
        await adapter.send("main", "Between.")
        await _ask(adapter, phone, clarify, "h2", "And now?", None)
        page, older = adapter.store.history("main", adapter.store.latest() + 1, 10)
        assert older is None
        assert [item.get("card") or item["text"] for item in page] == [
            "Before.", "question", "staging", "Between.", "question"]  # the pick is the person's message
        answered, waiting = page[1], page[4]
        assert answered["request"]["clarify_id"] == "h1" and answered["resolved"]["choices"] == [0]
        assert waiting["request"]["question"] == "And now?" and waiting["resolved"] is None
        assert [item["seq"] for item in page] == sorted(item["seq"] for item in page)
        # A page that stops at a message holds only the cards among its messages.
        page, older = adapter.store.history("main", adapter.store.latest() + 1, 1)
        assert [item.get("card") or item["text"] for item in page] == ["Between.", "question"] and older
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_every_card_settles_even_across_a_restart(tmp_path, clarify):
    from plugins.platforms.aleph.adapter import AlephAdapter
    from tests.gateway.test_aleph import _approval_prompt
    from tools import approval
    state = tmp_path / "aleph"
    first = AlephAdapter(_config(state))
    first.set_message_handler(lambda event: None)
    assert await first.connect()
    clarify.asked.append("sk")
    original = approval.observe_gateway_settle
    approval.observe_gateway_settle = lambda *a: True
    try:
        await first._send_exec_approval_prompt(_approval_prompt("a-open"))
        clarify.register("c-open", "sk", "Open?", ["y", "n"])
        await first.send_clarify("main", "Open?", ["y", "n"], "c-open", "sk")
        clarify.register("c-done", "sk", "Done?", ["y", "n"])
        await first.send_clarify("main", "Done?", ["y", "n"], "c-done", "sk")
        await first._respond_clarify({"clarify_id": "c-done", "choices": [0], "message_id": "c-answer"})
    finally:
        approval.observe_gateway_settle = original
    await first.disconnect()  # the process went away with two cards waiting

    second = AlephAdapter(_config(state))
    second.set_message_handler(lambda event: None)
    assert await second.connect()
    try:
        settled = [(e.type, e.payload) for e in second.store.events_after(0)
                   if e.type.endswith(".resolved")]
        assert settled == [
            ("clarify.resolved", {"chat": "main", "clarify_id": "c-done", "answer": "y", "choices": [0], "by": "user"}),
            ("approval.resolved", {"chat": "main", "request_id": "a-open", "choice": None, "by": "cancelled"}),
            ("clarify.resolved", {"chat": "main", "clarify_id": "c-open", "answer": None, "choices": None,
                                  "by": "timeout"})]
        assert second.store.unsettled_cards() == []
    finally:
        await second.disconnect()
