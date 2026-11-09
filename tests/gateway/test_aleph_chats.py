"""Side chats on Aleph: chat.create / chat.delete, every chat event naming its chat, a session of
its own per chat, per-chat receipts, drafts, typing, reactions and steps, session titles as
chat.update, pushes grouped by chat, and the store starting over under a new layout."""

from __future__ import annotations

import asyncio
import sqlite3
import time
from types import SimpleNamespace

import pytest
import pytest_asyncio

from gateway.config import Platform, PlatformConfig
from plugins.platforms.aleph import tools as aleph_tools
from tests.gateway.aleph_phone import connect, pair_device

SIDE = "9B2E4C1A-0000-4000-8000-000000000001"


def _config(tmp_path, **extra):
    return PlatformConfig(enabled=True, extra={"host": "127.0.0.1", "port": 0,
                                               "state_dir": str(tmp_path / "aleph"), **extra})


@pytest_asyncio.fixture
async def adapter(tmp_path, monkeypatch):
    from plugins.platforms.aleph.adapter import AlephAdapter
    instance = AlephAdapter(_config(tmp_path))
    received = []

    async def handler(event):
        received.append(event)
    instance.set_message_handler(handler)
    instance.received = received
    assert await instance.connect()
    instance.device, instance.keys = pair_device(instance)
    monkeypatch.setattr(aleph_tools, "_live", lambda: (None, instance))
    yield instance
    await instance.disconnect()


async def _received(adapter, message_id):
    for _ in range(100):
        hits = [e for e in adapter.received if e.message_id == message_id]
        if hits:
            return hits[-1]
        await asyncio.sleep(0.01)
    raise AssertionError(f"{message_id} never reached the gateway")


async def _settle():
    for _ in range(20):
        await asyncio.sleep(0.005)


def _source(chat):
    return SimpleNamespace(chat_id=chat, platform=Platform("aleph"))


# ── opening ───────────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_side_chat_is_opened_once_and_is_a_session_of_its_own(adapter):
    phone = await connect(adapter, adapter.keys)
    try:
        await phone.hello()
        opened = await phone.call("chat.create", {"id": SIDE, "title": "  Quick   question "})
        event = (await phone.logged())["params"]
        assert event["type"] == "chat.new" and opened["result"] == {"seq": event["seq"]}
        assert event["payload"]["chat"] == SIDE and event["payload"]["title"] == "Quick question"
        assert event["payload"]["created_at"]
        again = await phone.call("chat.create", {"id": SIDE, "title": "other"})
        assert again["result"] == opened["result"] and adapter.store.latest() == event["seq"]  # idempotent
        for bad in ({"id": "main"}, {"id": ""}, {"id": 5}, {}, {"id": "x" * 129}, {"id": "n", "title": 3}):
            assert (await phone.call("chat.create", bad))["error"]["code"] == -32602, bad

        sent = await phone.call("message.send", {"chat": SIDE, "id": "s-1", "text": "Is it raining in Paris?"})
        logged = (await phone.logged())["params"]
        assert sent["result"] == {"seq": logged["seq"]} and logged["payload"]["chat"] == SIDE
        dispatched = await _received(adapter, "s-1")
        assert dispatched.source.chat_id == SIDE and dispatched.source.chat_type == "dm"
        assert dispatched.source.chat_name == "Quick question"
        from gateway.session import build_session_key
        main_source = adapter._source("main")
        assert build_session_key(dispatched.source) != build_session_key(main_source)

        for bad in ({"id": "s-2", "text": "x"}, {"chat": "nope", "id": "s-3", "text": "x"}):
            assert (await phone.call("message.send", bad))["error"]["code"] == -32602, bad
        # An id is the person's in one chat only.
        assert (await phone.call("message.send", {"chat": "main", "id": "s-1", "text": "x"}))["error"]["code"] == -32602
        assert (await phone.call("chat.stop", {"chat": "nope"}))["error"]["code"] == -32602
        assert (await phone.call("chat.stop", {"chat": SIDE}))["result"] == {}
        await _settle()
        assert adapter.received[-1].text == "/stop" and adapter.received[-1].source.chat_id == SIDE
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_side_chats_survive_a_restart(adapter):
    await adapter.handle_call("chat.create", {"id": SIDE, "title": None})
    await adapter.disconnect()
    assert await adapter.connect()
    assert adapter._exists(SIDE)
    assert (await adapter.send(SIDE, "still here")).success
    assert adapter.store.message((await adapter.send(SIDE, "x")).message_id).chat == SIDE


# ── per-chat state ────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_replies_drafts_typing_and_steps_go_to_their_chat(adapter):
    await adapter.handle_call("chat.create", {"id": SIDE})
    phone = await connect(adapter, adapter.keys)
    try:
        await phone.hello()
        await adapter.send_typing(SIDE)
        await adapter.send_draft(SIDE, 3, "Paris is")
        await adapter.send_typing("main")
        await asyncio.to_thread(adapter.on_tool_activity, _source(SIDE), {
            "type": "tool.started", "tool": "web_search", "args": {"query": "Paris weather"}, "live_status": "full"})
        await _settle()
        sent = await adapter.send(SIDE, "Paris is dry today.")
        frames = []
        while not frames or frames[-1]["type"] != "message.new":
            frames.append((await phone.event())["params"])
        by_type = {f["type"]: f["payload"] for f in frames}
        assert by_type["draft"] == {"chat": SIDE, "draft_id": by_type["draft"]["draft_id"], "text": "Paris is"}
        assert by_type["activity"]["chat"] == SIDE
        assert {f["payload"]["chat"] for f in frames if f["type"] == "typing"} == {SIDE, "main"}
        assert by_type["message.new"]["chat"] == SIDE and by_type["message.new"]["draft_id"] == by_type["draft"]["draft_id"]
        assert adapter.store.message(sent.message_id).chat == SIDE
        assert adapter._activity_status[SIDE] == "Searching the web for Paris weather…"
        assert adapter._activity_status.get("main") is None
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_a_phone_connecting_mid_turn_sees_every_chat(adapter):
    await adapter.handle_call("chat.create", {"id": SIDE})
    await adapter.send_typing("main")
    await adapter.send_typing(SIDE)
    await adapter.send_draft(SIDE, 4, "so far")
    phone = await connect(adapter, adapter.keys)
    try:
        await phone.hello()
        frames = [(await phone.event())["params"] for _ in range(3)]
        assert [(f["type"], f["payload"]["chat"]) for f in frames] == [
            ("typing", "main"), ("typing", SIDE), ("draft", SIDE)]
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_receipts_are_per_chat(adapter):
    from gateway.platforms.base import MessageEvent
    await adapter.handle_call("chat.create", {"id": SIDE})
    await adapter.handle_call("message.send", {"chat": "main", "id": "m-1", "text": "main one"})
    await adapter.handle_call("message.send", {"chat": SIDE, "id": "s-1", "text": "side one"})
    await _settle()  # each message's turn starts: a receipt in its chat
    await adapter.on_processing_start(MessageEvent(text="", source=_source(SIDE)))  # nothing new there
    receipts = [e.payload for e in adapter.store.events_after(0) if e.type == "receipt"]
    assert sorted((r["chat"], r["read_up_to"]) for r in receipts) == sorted([("main", "m-1"), (SIDE, "s-1")])


@pytest.mark.asyncio
async def test_reactions_reach_the_chat_they_were_made_in(adapter):
    await adapter.handle_call("chat.create", {"id": SIDE})
    in_main = await adapter.send("main", "Main reply")
    in_side = await adapter.send(SIDE, "Side reply")
    await adapter.handle_call("reaction.set", {"message_id": in_side.message_id, "emoji": "👍"})
    await adapter.handle_call("reaction.set", {"message_id": in_main.message_id, "emoji": "❤️"})
    events = [e.payload for e in adapter.store.events_after(0) if e.type == "reaction"]
    assert [e["chat"] for e in events] == [SIDE, "main"]
    await adapter.handle_call("message.send", {"chat": SIDE, "id": "s-1", "text": "ok"})
    assert (await _received(adapter, "s-1")).text == '[The person reacted 👍 to your message "Side reply"]\n\nok'
    await adapter.handle_call("message.send", {"chat": "main", "id": "m-1", "text": "and"})
    assert (await _received(adapter, "m-1")).text.startswith('[The person reacted ❤️ to your message "Main reply"]')


@pytest.mark.asyncio
async def test_the_agent_reacts_in_the_chat_its_turn_is_in(adapter, monkeypatch):
    import json
    await adapter.handle_call("chat.create", {"id": SIDE})
    await adapter.handle_call("message.send", {"chat": "main", "id": "m-1", "text": "main"})
    await adapter.handle_call("message.send", {"chat": SIDE, "id": "s-1", "text": "side"})
    sessions = {"HERMES_SESSION_PLATFORM": "aleph", "HERMES_SESSION_CHAT_ID": SIDE}
    monkeypatch.setattr("gateway.session_context.get_session_env", lambda name, default="": sessions.get(name, default))
    result = json.loads(await aleph_tools.react({"emoji": "😂"}))
    assert result["structuredContent"]["message_id"] == "s-1"
    assert [e.payload["chat"] for e in adapter.store.events_after(0) if e.type == "reaction"] == [SIDE]
    added = await adapter.add_reaction("main", "🎉")
    assert added["message_id"] == "m-1"
    with pytest.raises(ValueError):
        await adapter.react("👍", chat=SIDE, message_id="m-1")  # not a message of that chat


@pytest.mark.asyncio
async def test_activity_list_filters_by_chat(adapter):
    await adapter.handle_call("chat.create", {"id": SIDE})
    for chat, query in (("main", "one"), (SIDE, "two"), ("main", "three")):
        await asyncio.to_thread(adapter.on_tool_activity, _source(chat), {
            "type": "tool.started", "tool": "web_search", "args": {"query": query}, "live_status": "full"})
        await _settle()
    every = await adapter.handle_call("activity.list", {})
    assert [s["detail"] for s in every["steps"]] == ["three", "two", "one"]
    side = await adapter.handle_call("activity.list", {"chat": SIDE})
    assert [(s["chat"], s["detail"]) for s in side["steps"]] == [(SIDE, "two")]
    main = await adapter.handle_call("activity.list", {"chat": "main", "limit": 1})
    assert [s["detail"] for s in main["steps"]] == ["three"] and main["more"] is True
    from plugins.platforms.aleph.server import RpcError
    with pytest.raises(RpcError):
        await adapter.handle_call("activity.list", {"chat": "nope"})


# ── titles ────────────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_session_titles_name_side_chats(adapter):
    await adapter.handle_call("chat.create", {"id": SIDE, "title": None})
    latest = adapter.store.latest()
    for title, source in (("Is it raining in Paris", "derived"), ("Paris weather today", "llm"),
                          ("Paris weather today", "llm")):
        await asyncio.to_thread(adapter.on_session_title, _source(SIDE), title, source)
        await _settle()
    await asyncio.to_thread(adapter.on_session_title, _source("main"), "Main has no title", "llm")
    await asyncio.to_thread(adapter.on_session_title, _source("gone"), "Unknown chat", "llm")
    await _settle()
    updates = [e for e in adapter.store.events_after(latest)]
    assert [(e.type, e.payload) for e in updates] == [
        ("chat.update", {"chat": SIDE, "title": "Is it raining in Paris"}),
        ("chat.update", {"chat": SIDE, "title": "Paris weather today"})]
    assert adapter.store.chat(SIDE).title == "Paris weather today"
    assert adapter._source(SIDE).chat_name == "Paris weather today"


# ── deleting ──────────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_deleting_ends_the_session_and_drops_the_chat(adapter):
    phone = await connect(adapter, adapter.keys)
    try:
        await phone.hello()
        await adapter.send("main", "before")
        await phone.call("chat.create", {"id": SIDE})
        upload = await phone.upload(adapter, b"\x89PNG side", name="a.bin", mime="application/octet-stream")
        await phone.call("message.send", {"chat": SIDE, "id": "s-1", "text": "look", "attachments": [upload]})
        file_id = adapter.store.message("s-1").attachments[0]["id"]
        file_path = adapter.store.file(file_id).path
        await adapter.send_typing(SIDE)
        await adapter.send(SIDE, "Looks fine")
        await adapter.send("main", "after")
        cursor = adapter.store.latest()

        assert (await phone.call("chat.delete", {"id": "main"}))["error"]["code"] == -32602
        assert (await phone.call("chat.delete", {"id": "nope"}))["error"]["code"] == -32602
        deleted = await phone.call("chat.delete", {"id": SIDE})
        assert deleted["result"]["seq"] == cursor + 1
        await _settle()
        new = [e for e in adapter.received if e.text == "/new"]
        assert len(new) == 1 and new[0].source.chat_id == SIDE  # its Hermes session ends
        assert (await phone.call("chat.delete", {"id": SIDE}))["result"] == deleted["result"]  # a resend
        assert (await phone.call("chat.create", {"id": SIDE}))["error"]["code"] == -32602  # not reused

        events = adapter.store.events_after(0)
        assert all(e.payload.get("chat") != SIDE for e in events[:-1])
        assert (events[-1].type, events[-1].payload) == ("chat.delete", {"chat": SIDE})
        assert adapter.store.message("s-1") is None and adapter.store.file(file_id) is None
        import os
        assert not os.path.exists(file_path)
        assert SIDE not in adapter._typing
        # What the gateway still sends there (the /new banner, a late reply) goes nowhere.
        assert (await adapter.send(SIDE, "late")).success
        assert (await phone.call("message.send", {"chat": SIDE, "id": "s-2", "text": "x"}))["error"]["code"] == -32602
        assert adapter.store.latest() == cursor + 1
    finally:
        await phone.close()

    # A phone catching up: the chat's events are not there, and that is no reset.
    phone = await connect(adapter, adapter.keys)
    try:
        answer, replay = await phone.hello(after=1)
        assert answer["result"]["reset"] is False
        assert [e["payload"].get("chat") for e in replay[:-1]] == ["main", SIDE]
        assert [e["type"] for e in replay[:-1]] == ["message.new", "chat.delete"]
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_deleting_the_newest_events_never_reuses_their_seqs(adapter):
    await adapter.send("main", "first")
    await adapter.handle_call("chat.create", {"id": SIDE})
    await adapter.send(SIDE, "the newest events are this chat's")
    latest = adapter.store.latest()
    deleted = await adapter.handle_call("chat.delete", {"id": SIDE})
    assert deleted == {"seq": latest + 1}
    assert (await adapter.handle_call("chat.delete", {"id": SIDE})) == deleted
    sent = await adapter.send("main", "next")
    assert adapter.store.message(sent.message_id).seq == latest + 2
    await adapter.disconnect()
    assert await adapter.connect()  # the count survives a restart
    assert adapter.store.latest() == latest + 2


@pytest.mark.asyncio
async def test_idle_side_chats_are_deleted(tmp_path):
    from plugins.platforms.aleph.adapter import AlephAdapter
    instance = AlephAdapter(_config(tmp_path, side_chat_idle_days=30))

    async def handler(event):
        return None
    instance.set_message_handler(handler)
    assert await instance.connect()
    try:
        await instance.handle_call("chat.create", {"id": "old"})
        await instance.handle_call("chat.create", {"id": "fresh"})
        instance.store._db.execute("UPDATE chats SET last_active_at = ? WHERE id = 'old'", (time.time() - 31 * 86400,))
        instance._idle_sweep.cancel()
        sweep = asyncio.create_task(instance._sweep_idle_chats())
        await _settle()
        sweep.cancel()
        assert not instance._exists("old") and instance._exists("fresh")
        assert instance.store.events_after(0)[-1].payload == {"chat": "old"}
    finally:
        await instance.disconnect()


# ── pushes ────────────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_pushes_group_by_chat(adapter):
    from tests.gateway.test_aleph_push import FakeSender
    adapter._push = FakeSender()
    adapter.store.set_push_token(adapter.device.id, "ab" * 32, "production")
    await adapter.handle_call("chat.create", {"id": SIDE})
    await adapter.send(SIDE, "side reply")
    await adapter.send("main", "main reply")
    for _ in range(50):
        if len(adapter._push.sent) == 2:
            break
        await asyncio.sleep(0.01)
    payloads = [payload for _token, _env, payload in adapter._push.sent]
    assert [(p["aps"]["thread-id"], p["aleph"]["chat"]) for p in payloads] == [(SIDE, SIDE), ("main", "main")]


# ── the store's layout ────────────────────────────────────────────────────────────────────────


def test_a_store_from_another_layout_starts_the_conversation_over(tmp_path):
    from plugins.platforms.aleph.store import AlephStore
    root = tmp_path / "aleph"
    (root / "files").mkdir(parents=True)
    db = sqlite3.connect(str(root / "aleph.db"))
    db.executescript("""
        CREATE TABLE events (seq INTEGER PRIMARY KEY, type TEXT NOT NULL, payload TEXT NOT NULL);
        CREATE TABLE messages (id TEXT PRIMARY KEY, seq INTEGER NOT NULL, sender TEXT NOT NULL, kind TEXT NOT NULL,
                               text TEXT NOT NULL, attachments TEXT NOT NULL, dispatched INTEGER NOT NULL DEFAULT 0);
        CREATE TABLE files (id TEXT PRIMARY KEY, description TEXT NOT NULL, path TEXT NOT NULL, agent_path TEXT);
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE devices (id TEXT PRIMARY KEY, name TEXT NOT NULL, model TEXT NOT NULL,
                              public_key BLOB NOT NULL UNIQUE, created_at REAL NOT NULL, last_seen_at REAL,
                              push_token TEXT, push_environment TEXT);
        CREATE TABLE activity (rowid INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT NOT NULL UNIQUE, step TEXT NOT NULL);
    """)
    old_file, avatar = root / "files" / "f_old.bin", root / "files" / "f_avatar.bin"
    old_file.write_bytes(b"x")
    avatar.write_bytes(b"y")
    db.executemany("INSERT INTO events VALUES (?, 'message.new', '{}')", [(n,) for n in range(1, 6)])
    db.execute("INSERT INTO files VALUES ('f_old', '{}', ?, NULL)", (str(old_file),))
    db.execute("INSERT INTO files VALUES ('f_avatar', '{}', ?, NULL)", (str(avatar),))
    db.executemany("INSERT INTO meta VALUES (?, ?)", [("profile_name", "Iris"), ("profile_avatar", "f_avatar"),
                                                      ("last_user_message", "m-9"), ("seq_floor", "5")])
    db.execute("INSERT INTO devices (id, name, model, public_key, created_at) VALUES ('d_1', 'Phone', 'iPhone', ?, 1)",
               (b"k" * 32,))
    db.commit()
    db.close()
    store = AlephStore(root)
    try:
        assert store.latest() == 0 and store.events_after(0) == []  # a phone that knew it gets reset: true
        assert [d.id for d in store.devices()] == ["d_1"]
        assert store.meta("profile_name") == "Iris" and store.file("f_avatar") is not None and avatar.exists()
        assert store.file("f_old") is None and not old_file.exists()
        assert store.meta("last_user_message") is None
        store.append("message.new", {"chat": "main", "id": "m"}, message={"id": "m", "sender": "user"})
        assert store.message("m").chat == "main"
    finally:
        store.close()
    reopened = AlephStore(root)  # the current layout is kept as it is
    try:
        assert reopened.latest() == 1
    finally:
        reopened.close()
