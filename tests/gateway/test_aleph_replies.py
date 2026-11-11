"""Replies on Aleph (plugins/platforms/aleph/replies.py): the quote snapshot a message.new carries,
for the person's messages (message.send reply_to) and the agent's (send reply_to)."""

from __future__ import annotations

import asyncio

import pytest
import pytest_asyncio

from gateway.config import PlatformConfig
from plugins.platforms.aleph.replies import excerpt, snapshot
from tests.gateway.aleph_phone import connect, pair_device
from tests.gateway.test_aleph_files import _png_bytes


@pytest_asyncio.fixture
async def adapter(tmp_path):
    from plugins.platforms.aleph.adapter import AlephAdapter
    config = PlatformConfig(enabled=True, extra={"host": "127.0.0.1", "port": 0,
                                                 "state_dir": str(tmp_path / "aleph")})
    instance = AlephAdapter(config)
    received = []

    async def handler(event):
        received.append(event)
    instance.set_message_handler(handler)
    instance.received = received
    assert await instance.connect()
    instance.device, instance.keys = pair_device(instance)
    yield instance
    await instance.disconnect()


def test_excerpt_is_one_line_and_never_splits_a_grapheme():
    assert excerpt("  two\n\nlines  ") == "two lines"
    assert excerpt("x" * 300) == "x" * 200
    family = "👨‍👩‍👧"  # 5 code points
    cut = excerpt("a" * 198 + family)
    assert cut == "a" * 198
    assert excerpt("a" * 199 + "é") == "a" * 199  # an accent stays with its letter
    assert excerpt("a" * 199 + "🇯🇵") == "a" * 199  # a flag is never halved
    assert excerpt("a" * 198 + "🇯🇵") == "a" * 198 + "🇯🇵"


def test_snapshot_names_what_a_media_only_message_carried():
    assert snapshot("m", "user", "", [{"kind": "image"}, {"kind": "file"}], None)["attachment"] == "image"
    assert snapshot("m", "agent", "", [], {"latitude": 1, "longitude": 2})["attachment"] == "location"
    assert snapshot("m", "user", "hi", [], None) == {"id": "m", "sender": "user", "text": "hi", "attachment": None}


async def _send(phone, message_id, chat="main", **params):
    return await phone.call("message.send", {"chat": chat, "id": message_id, "text": "", **params})


async def _message(phone):
    """The next logged message.new (skipping receipts and other logged events)."""
    while True:
        frame = await phone.logged()
        if frame["params"]["type"] == "message.new":
            return frame["params"]["payload"]


@pytest.mark.asyncio
async def test_the_persons_reply_carries_a_snapshot(adapter):
    phone = await connect(adapter, adapter.keys)
    try:
        await phone.hello()
        await _send(phone, "m-1", text="What about Tokyo\nin  May?")
        await _message(phone)
        agent = await adapter.send("main", "Cherry blossoms are over by then.")
        await _message(phone)

        await _send(phone, "m-2", text="Ok", reply_to="m-1")
        assert (await _message(phone))["reply_to"] == {
            "id": "m-1", "sender": "user", "text": "What about Tokyo in May?", "attachment": None}
        await _send(phone, "m-3", text="Really?", reply_to=agent.message_id)
        assert (await _message(phone))["reply_to"] == {
            "id": agent.message_id, "sender": "agent", "text": "Cherry blossoms are over by then.",
            "attachment": None}

        # The agent still gets the quoted text, as before.
        for _ in range(100):
            event = next((e for e in adapter.received if e.message_id == "m-3"), None)
            if event is not None:
                break
            await asyncio.sleep(0.01)
        assert event.reply_to_text == "Cherry blossoms are over by then." and event.reply_to_is_own_message
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_media_only_unknown_and_cross_chat_quotes(adapter):
    phone = await connect(adapter, adapter.keys)
    try:
        await phone.hello()
        image = await phone.upload(adapter, _png_bytes(), name="a.png", mime="image/png")
        await _send(phone, "m-img", attachments=[image])
        await _message(phone)
        await _send(phone, "m-pin", location={"latitude": 31.2, "longitude": 121.4})
        await _message(phone)
        await _send(phone, "m-a", text="photo?", reply_to="m-img")
        assert (await _message(phone))["reply_to"] == {
            "id": "m-img", "sender": "user", "text": "", "attachment": "image"}
        await _send(phone, "m-b", text="there?", reply_to="m-pin")
        assert (await _message(phone))["reply_to"]["attachment"] == "location"

        answer = await _send(phone, "m-c", text="lost", reply_to="never-sent")
        assert "result" in answer
        assert "reply_to" not in (await _message(phone))

        created = await phone.call("chat.create", {"id": "side-1"})
        assert "result" in created
        await _send(phone, "m-side", chat="side-1", text="elsewhere", reply_to="m-a")
        assert "reply_to" not in (await _message(phone))
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_a_resend_keeps_the_first_snapshot(adapter):
    phone = await connect(adapter, adapter.keys)
    try:
        await phone.hello()
        await _send(phone, "m-1", text="first")
        await _message(phone)
        first = await _send(phone, "m-2", text="reply", reply_to="m-1")
        while (logged := (await phone.logged())["params"])["type"] != "message.new":
            pass
        again = await _send(phone, "m-2", text="reply", reply_to="other")
        assert again["result"] == first["result"]
        events = await asyncio.to_thread(adapter.store.events_after, logged["seq"] - 1)
        assert [e.payload.get("reply_to", {}).get("id") for e in events if e.type == "message.new"] == ["m-1"]
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_the_agents_reply_quotes_only_what_is_not_the_latest_message(adapter):
    phone = await connect(adapter, adapter.keys)
    try:
        await phone.hello()
        await _send(phone, "m-1", text="Book the train")
        await _message(phone)
        await _send(phone, "m-2", text="And a hotel")
        await _message(phone)
        for _ in range(100):  # both reached the gateway: m-2 is the latest
            if adapter._last_user_message.get("main") == "m-2":
                break
            await asyncio.sleep(0.01)

        # The gateway anchors every reply to the turn's message: the latest needs no quote.
        await adapter.send("main", "Hotel booked.", reply_to="m-2")
        assert "reply_to" not in (await _message(phone))
        await adapter.send("main", "Train booked.", reply_to="m-1")
        assert (await _message(phone))["reply_to"] == {
            "id": "m-1", "sender": "user", "text": "Book the train", "attachment": None}
        await adapter.send("main", "No anchor.")
        assert "reply_to" not in (await _message(phone))
        await adapter.send("main", "Unknown.", reply_to="never-sent")
        assert "reply_to" not in (await _message(phone))
    finally:
        await phone.close()
