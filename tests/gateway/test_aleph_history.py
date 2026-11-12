"""First sync of a long history on Aleph: a fresh phone's ``hello`` resets to the present (no replay),
and ``history.load`` pages older messages in, as they stand now, as ``history.page`` events placed in
the connection's stream where that state was current."""

from __future__ import annotations

import asyncio

import pytest
import pytest_asyncio

from gateway.config import PlatformConfig
from tests.gateway.aleph_phone import connect, pair_device


@pytest_asyncio.fixture
async def adapter(tmp_path):
    from plugins.platforms.aleph.adapter import AlephAdapter
    config = PlatformConfig(enabled=True, extra={"host": "127.0.0.1", "port": 0,
                                                 "state_dir": str(tmp_path / "aleph")})
    instance = AlephAdapter(config)

    async def handler(event):
        return None
    instance.set_message_handler(handler)
    assert await instance.connect()
    instance.device, instance.keys = pair_device(instance)
    yield instance
    await instance.disconnect()


async def _page(phone, **params):
    answer = await phone.call("history.load", {"chat": "main", **params})
    if "error" in answer:
        return answer
    assert answer["result"] == {}
    while True:
        frame = await phone.event()
        if frame["params"]["type"] == "history.page":
            assert "seq" not in frame["params"]
            return frame["params"]["payload"]


@pytest.mark.asyncio
async def test_a_fresh_phone_starts_from_the_present(adapter):
    for i in range(3):
        await adapter.send("main", f"m{i}")
    await adapter.handle_call("chat.create", {"id": "side-1", "title": "Trip"})
    await adapter.send_typing("main")
    phone = await connect(adapter, adapter.keys)
    try:
        answer, replay = await phone.hello(after=0)
        result = answer["result"]
        assert result["reset"] is True and result["latest"] == 4
        (chat,) = result["chats"]
        assert chat["chat"] == "side-1" and chat["title"] == "Trip" and chat["created_at"]
        assert replay == [{"type": "sync.done", "payload": {"seq": 4}}]
        assert (await phone.event())["params"]["type"] == "typing"  # the present follows, as before
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_an_empty_log_and_a_caught_up_phone_are_no_reset(adapter):
    phone = await connect(adapter, adapter.keys)
    try:
        answer, replay = await phone.hello(after=0)
        assert answer["result"]["reset"] is False and "chats" not in answer["result"]
        assert replay == [{"type": "sync.done", "payload": {"seq": 0}}]
    finally:
        await phone.close()
    await adapter.send("main", "one")
    await adapter.send("main", "two")
    phone = await connect(adapter, adapter.keys)
    try:
        answer, replay = await phone.hello(after=1)
        assert answer["result"]["reset"] is False
        assert [e.get("seq") for e in replay] == [2, None]
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_pages_go_backwards_until_nothing_is_older(adapter):
    for i in range(5):
        await adapter.send("main", f"m{i}")
    phone = await connect(adapter, adapter.keys)
    try:
        await phone.hello()
        first = await _page(phone, before=6, limit=2)
        assert first["chat"] == "main" and first["before"] == 6
        assert [m["text"] for m in first["messages"]] == ["m3", "m4"]
        assert [m["seq"] for m in first["messages"]] == [4, 5] and first["next"] == 4
        second = await _page(phone, before=first["next"], limit=2)
        assert [m["text"] for m in second["messages"]] == ["m1", "m2"] and second["next"] == 2
        last = await _page(phone, before=second["next"], limit=2)
        assert [m["text"] for m in last["messages"]] == ["m0"] and last["next"] is None
        assert (await _page(phone, before=1, limit=2)) == {"chat": "main", "before": 1, "messages": [],
                                                            "next": None}
        stamps = [m["created_at"] for m in last["messages"] + second["messages"] + first["messages"]]
        assert stamps == sorted(stamps)
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_a_page_shows_messages_as_they_stand_now(adapter):
    phone = await connect(adapter, adapter.keys)
    try:
        await phone.hello()
        await phone.call("message.send", {"chat": "main", "id": "u-1", "text": "Is it on?"})
        await phone.call("message.send", {"chat": "main", "id": "u-2", "text": "Hello?"})
        edited = await adapter.send("main", "draft answer")
        gone = await adapter.send("main", "a mistake")
        await adapter.send("main", "Working on it", metadata={"_interim_send": True})
        await adapter.edit_message("main", edited.message_id, "The answer")
        await adapter.delete_message("main", gone.message_id)
        await phone.call("reaction.set", {"message_id": edited.message_id, "emoji": "❤️"})
        await adapter.react("👍", message_id="u-1")
        await adapter.send("main", "elsewhere")  # main; side chats are paged on their own

        page = await _page(phone, before=adapter.store.latest() + 1, limit=50)
        by_text = {m["text"]: m for m in page["messages"]}
        assert list(by_text) == ["Is it on?", "Hello?", "The answer", "Working on it", "elsewhere"]
        assert "a mistake" not in by_text and "draft_id" not in by_text["The answer"]
        assert by_text["The answer"]["reactions"] == {"user": "❤️", "agent": None}
        assert by_text["Is it on?"]["reactions"] == {"user": None, "agent": "👍"}
        assert by_text["Is it on?"]["read"] is True and by_text["Hello?"]["read"] is True  # its turn began
        assert "read" not in by_text["The answer"]
        assert by_text["Working on it"]["kind"] == "system"
        assert by_text["Is it on?"]["sender"] == "user" and by_text["Is it on?"]["id"] == "u-1"
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_a_side_chat_is_paged_on_its_own_and_unknown_chats_are_refused(adapter):
    phone = await connect(adapter, adapter.keys)
    try:
        await phone.hello()
        await phone.call("chat.create", {"id": "side-1"})
        await adapter.send("side-1", "in the side chat")
        await adapter.send("main", "in main")
        side = await _page(phone, chat="side-1", before=100, limit=10)
        assert [m["text"] for m in side["messages"]] == ["in the side chat"]

        assert (await _page(phone, chat="nope", before=10))["error"]["code"] == -32602
        await phone.call("chat.delete", {"id": "side-1"})
        assert (await _page(phone, chat="side-1", before=100))["error"]["code"] == -32602
        for bad in ({"before": 0}, {"before": "9"}, {"before": True}, {"before": 5, "limit": 0},
                    {"before": 5, "limit": 501}):
            assert (await _page(phone, **bad))["error"]["code"] == -32602
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_busy(adapter, monkeypatch):
    from plugins.platforms.aleph import adapter as adapter_module
    monkeypatch.setattr(adapter_module, "MAX_HISTORY_LOADS", 0)
    phone = await connect(adapter, adapter.keys)
    try:
        await phone.hello()
        assert (await _page(phone, before=5))["error"]["code"] == 4002
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_a_page_never_disagrees_with_the_events_around_it(adapter):
    """A delete racing a load: the page holds the message exactly when the delete event follows it."""
    for attempt in range(20):
        sent = await adapter.send("main", f"race {attempt}")
        phone = await connect(adapter, adapter.keys)
        try:
            await phone.hello(after=adapter.store.latest())
            load = asyncio.create_task(phone.call("history.load", {"chat": "main",
                                                                   "before": adapter.store.latest() + 1,
                                                                   "limit": 1}))
            await asyncio.sleep(0)
            await adapter.delete_message("main", sent.message_id)
            await load
            order = []
            while len(order) < 2:
                frame = (await phone.event())["params"]
                if frame["type"] in ("history.page", "message.delete"):
                    order.append(frame)
            page = next(f for f in order if f["type"] == "history.page")["payload"]
            shown = any(m["id"] == sent.message_id for m in page["messages"])
            assert shown == (order[0]["type"] == "history.page"), (attempt, order)
        finally:
            await phone.close()


def test_created_at_never_goes_back_within_a_chat(tmp_path):
    from plugins.platforms.aleph.store import AlephStore
    store = AlephStore(tmp_path)
    try:
        def log(message_id, stamp, chat="main"):
            store.append("message.new", {"chat": chat, "id": message_id, "sender": "agent", "text": message_id,
                                         "attachments": [], "created_at": stamp},
                         message={"id": message_id, "sender": "agent", "text": message_id, "attachments": []})
        log("a", "2026-11-12T10:00:00.000-05:00")
        log("b", "2026-11-12T14:59:00.000+00:00")  # earlier, in another offset
        log("c", "2026-11-12T09:00:00.000-05:00", chat="side")  # another chat keeps its own
        page, _ = store.history("main", 10, 10)
        assert [m["created_at"] for m in page] == ["2026-11-12T10:00:00.000-05:00"] * 2
        assert store.events_after(1)[0].payload["created_at"] == "2026-11-12T10:00:00.000-05:00"
        assert store.history("side", 10, 10)[0][0]["created_at"] == "2026-11-12T09:00:00.000-05:00"
    finally:
        store.close()


def test_a_receipt_covers_the_persons_messages_up_to_it(tmp_path):
    from plugins.platforms.aleph.store import AlephStore
    store = AlephStore(tmp_path)
    try:
        for message_id, chat in (("u1", "main"), ("u2", "main"), ("s1", "side"), ("u3", "main")):
            store.append("message.new", {"chat": chat, "id": message_id, "sender": "user", "text": message_id,
                                         "attachments": [], "created_at": "2026-11-12T10:00:00.000-05:00"},
                         message={"id": message_id, "sender": "user", "text": message_id, "attachments": []})
        store.append("receipt", {"chat": "main", "read_up_to": "u2", "at": "now"})
        store.append("receipt", {"chat": "main", "read_up_to": "s1", "at": "now"})  # another chat's id: nothing
        assert [m["read"] for m in store.history("main", 100, 10)[0]] == [True, True, False]
        assert [m["read"] for m in store.history("side", 100, 10)[0]] == [False]
    finally:
        store.close()


def test_a_store_of_another_layout_starts_the_conversation_over(tmp_path):
    import sqlite3

    from plugins.platforms.aleph import noise
    from plugins.platforms.aleph.store import SCHEMA_VERSION, AlephStore
    store = AlephStore(tmp_path)
    store.append("message.new", {"chat": "main", "id": "m1", "sender": "user", "text": "hi", "attachments": [],
                                 "created_at": "2026-11-12T10:00:00.000-05:00"},
                 message={"id": "m1", "sender": "user", "text": "hi", "attachments": []})
    secret, _ = store.new_pairing()
    device = store.complete_pairing(secret, noise.KeyPair.generate().public, name="Phone", model="iPhone")
    store.set_meta("profile_name", "Iris")
    store.close()

    store = AlephStore(tmp_path)  # the same layout: kept
    assert store.message("m1") is not None
    store.close()

    raw = sqlite3.connect(tmp_path / "aleph.db")
    raw.execute("UPDATE meta SET value = '2' WHERE key = 'schema'")
    raw.commit()
    raw.close()
    store = AlephStore(tmp_path)
    try:
        assert store.message("m1") is None and store.latest() == 0
        assert store.device(device.id) is not None and store.meta("profile_name") == "Iris"
        assert store.meta("schema") == SCHEMA_VERSION
    finally:
        store.close()
