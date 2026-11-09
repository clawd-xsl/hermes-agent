"""Emoji reactions on Aleph (plugins/platforms/aleph/reactions.py): reaction.set, the note that carries
the person's reactions to the agent, the aleph_react tool and send_message's react."""

from __future__ import annotations

import asyncio
import json

import jsonschema
import pytest
import pytest_asyncio

from gateway.config import PlatformConfig
from plugins.platforms.aleph import tools as aleph_tools
from plugins.platforms.aleph.reactions import is_single_emoji, note, quote
from tests.gateway.aleph_phone import connect, pair_device


@pytest_asyncio.fixture
async def adapter(tmp_path, monkeypatch):
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


def _result(text, tool=aleph_tools.REACT):
    result = json.loads(text)
    assert [block["type"] for block in result["content"]] == ["text"]
    if not result.get("isError"):
        jsonschema.validate(result["structuredContent"], tool["outputSchema"])
    return result


# ── emoji ─────────────────────────────────────────────────────────────────────────────────────


def test_one_emoji_grapheme():
    for good in ("❤️", "❤", "👍", "👍🏽", "😂", "🇯🇵", "👨‍👩‍👧‍👦", "🏳️‍🌈", "1️⃣", "#⃣", "🏴󠁧󠁢󠁳󠁣󠁴󠁿", "✅", "🫠", "🧑🏻‍💻"):
        assert is_single_emoji(good), good
    for bad in ("", "a", "ok", "❤️❤️", "👍 ", "😂x", "🇯", "1", "‍", "👍👎", "🇯🇵🇺🇸", None, 5, "x" * 50):
        assert not is_single_emoji(bad), bad


def test_the_note_quotes_the_start_of_each_message():
    assert quote("  short\n reply ") == '"short reply"'
    long = "word " * 30
    assert quote(long) == '"' + long[:60].rstrip() + '…"'
    assert quote("", has_attachments=True) == "with an attachment"
    assert note([("❤️", "Here is the plan for Tokyo", False), ("😂", "", True)]) == (
        '[The person reacted ❤️ to your message "Here is the plan for Tokyo"]\n'
        "[The person reacted 😂 to your message with an attachment]")


# ── the person reacts ─────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_reaction_is_logged_and_reaches_the_agent_with_the_next_message(adapter):
    first = await adapter.send("main", "Here is the plan for Tokyo: fly Tuesday, back Sunday.")
    second = await adapter.send("main", "And a second thought")
    phone = await connect(adapter, adapter.keys)
    try:
        await phone.hello()
        answer = await phone.call("reaction.set", {"message_id": first.message_id, "emoji": "❤️"})
        event = (await phone.logged())["params"]
        assert answer["result"]["seq"] == event["seq"] and event["type"] == "reaction"
        assert event["payload"]["message_id"] == first.message_id and event["payload"]["by"] == "user"
        assert event["payload"]["emoji"] == "❤️" and event["payload"]["at"]
        await phone.call("reaction.set", {"message_id": first.message_id, "emoji": "👍"})   # replaces ❤️
        await phone.call("reaction.set", {"message_id": second.message_id, "emoji": "😂"})
        await phone.call("reaction.set", {"message_id": second.message_id, "emoji": None})  # taken back
        await asyncio.sleep(0.1)
        assert adapter.received == []  # a reaction never starts a turn
        assert "result" in await phone.call("message.send", {"id": "m-1", "text": "Sounds good"})
        dispatched = await _received(adapter, "m-1")
        assert dispatched.text == ('[The person reacted 👍 to your message "Here is the plan for Tokyo: fly Tuesday, '
                                   'back Sunday."]\n\nSounds good')
        assert "result" in await phone.call("message.send", {"id": "m-2", "text": "Next"})
        assert (await _received(adapter, "m-2")).text == "Next"  # notes are for the next message only
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_pending_reactions_survive_a_restart(adapter):
    sent = await adapter.send("main", "Restart-proof")
    phone = await connect(adapter, adapter.keys)
    try:
        await phone.hello()
        await phone.call("reaction.set", {"message_id": sent.message_id, "emoji": "🔥"})
    finally:
        await phone.close()
    await adapter.disconnect()
    assert await adapter.connect()
    phone = await connect(adapter, adapter.keys)
    try:
        await phone.hello()
        await phone.call("message.send", {"id": "m-r", "text": "hi"})
        assert (await _received(adapter, "m-r")).text.startswith('[The person reacted 🔥 to your message "Restart-proof"]')
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_bad_reactions_are_refused(adapter):
    sent = await adapter.send("main", "x")
    phone = await connect(adapter, adapter.keys)
    try:
        await phone.hello()
        latest = adapter.store.latest()
        for params in ({"message_id": sent.message_id, "emoji": "ok"}, {"message_id": sent.message_id, "emoji": "👍👎"},
                       {"message_id": "nope", "emoji": "👍"}, {"emoji": "👍"}, {"message_id": sent.message_id},
                       {"message_id": sent.message_id, "emoji": 5}):
            assert (await phone.call("reaction.set", params))["error"]["code"] == -32602, params
        assert adapter.store.latest() == latest
        # Reacting to one's own message is fine, but it is not news for the agent.
        await phone.call("message.send", {"id": "m-own", "text": "mine"})
        await phone.call("reaction.set", {"message_id": "m-own", "emoji": "👀"})
        await phone.call("message.send", {"id": "m-next", "text": "next"})
        assert (await _received(adapter, "m-next")).text == "next"
    finally:
        await phone.close()


# ── the agent reacts ──────────────────────────────────────────────────────────────────────────


def test_the_tool_is_a_complete_mcp_definition():
    tool = aleph_tools.REACT
    assert set(tool) == {"name", "title", "description", "inputSchema", "outputSchema", "annotations"}
    jsonschema.Draft202012Validator.check_schema(tool["inputSchema"])
    jsonschema.Draft202012Validator.check_schema(tool["outputSchema"])
    assert tool["annotations"] == {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True,
                                   "openWorldHint": False}
    for hint in ("occasionally", "Never narrate", "messages_back", "empty string"):
        assert hint in tool["description"]


@pytest.mark.asyncio
async def test_aleph_react_targets_the_latest_message_or_an_earlier_one(adapter):
    phone = await connect(adapter, adapter.keys)
    try:
        await phone.hello()
        for n in range(3):
            await phone.call("message.send", {"id": f"u{n}", "text": f"message {n}"})
        await asyncio.sleep(0.05)
        latest = adapter.store.latest()
        result = _result(await aleph_tools.react({"emoji": "😂"}))
        assert result["structuredContent"] == {"message_id": "u2", "emoji": "😂"}
        event = None
        while event is None or event["type"] != "reaction":
            event = (await phone.logged())["params"]
        assert event["payload"]["by"] == "agent" and event["payload"]["message_id"] == "u2"
        assert _result(await aleph_tools.react({"emoji": "❤️", "messages_back": 2}))["structuredContent"]["message_id"] == "u0"
        assert _result(await aleph_tools.react({"emoji": "👍", "message_id": "u1"}))["structuredContent"]["message_id"] == "u1"
        removed = _result(await aleph_tools.react({"emoji": ""}))
        assert removed["structuredContent"] == {"message_id": "u2", "emoji": None}
        assert removed["content"][0]["text"] == "Reaction removed."
        assert adapter.store.latest() == latest + 4
        for args, hint in (({"emoji": "nope"}, "single emoji"), ({}, "emoji is required"),
                           ({"emoji": "👍", "messages_back": 5}, "fewer than 6"),
                           ({"emoji": "👍", "messages_back": -1}, "0 or more"),
                           ({"emoji": "👍", "message_id": "missing"}, "no message"),
                           ({"emoji": "👍", "message_id": "u1", "messages_back": 1}, "not both"),
                           ({"emoji": "👍", "target": "x"}, "Unknown argument")):
            result = _result(await aleph_tools.react(args))
            assert result["isError"] is True and hint in result["content"][0]["text"], (args, result)
        assert adapter.store.latest() == latest + 4
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_no_message_yet(adapter):
    result = _result(await aleph_tools.react({"emoji": "👍"}))
    assert result["isError"] is True and "no message" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_send_message_react_goes_through_add_and_remove_reaction(adapter):
    await adapter._accept_user_message({"id": "u-a", "text": "hello"})
    added = await adapter.add_reaction("main", "👋")
    assert added == {"success": True, "message_id": "u-a", "emoji": "👋"}
    assert (await adapter.add_reaction("main", "not one"))["success"] is False
    assert (await adapter.add_reaction("main", ""))["success"] is False
    assert (await adapter.add_reaction("main", "👍", message_id="ghost"))["success"] is False
    removed = await adapter.remove_reaction("main", message_id="u-a")
    assert removed == {"success": True, "message_id": "u-a", "emoji": None}

    # send_message runs it on its own loop, in another thread: the adapter hops to the gateway's.
    def elsewhere():
        return asyncio.run(adapter.add_reaction("main", "🎉"))
    assert (await asyncio.to_thread(elsewhere))["message_id"] == "u-a"


@pytest.mark.asyncio
async def test_reactions_are_not_pushed(adapter):
    from tests.gateway.test_aleph_push import FakeSender
    adapter._push = FakeSender()
    adapter.store.set_push_token(adapter.device.id, "ab" * 32, "production")
    await adapter._accept_user_message({"id": "u-p", "text": "hi"})
    await adapter.react("❤️")
    for _ in range(20):
        await asyncio.sleep(0.01)
    assert adapter._push.sent == []
