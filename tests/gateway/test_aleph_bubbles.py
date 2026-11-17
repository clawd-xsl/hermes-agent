"""Replies on Aleph beyond their text (protocol "Replies"): a reply in bubbles (<message_break/>), each
part a message of its own, streamed part by part; and aleph_options, reply choices on its last part."""

from __future__ import annotations

import json

import jsonschema
import pytest
import pytest_asyncio

from gateway.config import PlatformConfig
from plugins.platforms.aleph import tools as aleph_tools
from plugins.platforms.aleph.bubbles import split, split_streaming
from tests.gateway.aleph_phone import connect, pair_device

BREAK = "<message_break/>"


@pytest_asyncio.fixture
async def adapter(tmp_path, monkeypatch):
    from plugins.platforms.aleph.adapter import AlephAdapter
    instance = AlephAdapter(PlatformConfig(enabled=True, extra={"host": "127.0.0.1", "port": 0,
                                                                "state_dir": str(tmp_path / "aleph")}))

    async def handler(event):
        return None
    instance.set_message_handler(handler)
    assert await instance.connect()
    instance.device, instance.keys = pair_device(instance)
    monkeypatch.setattr(aleph_tools, "_live", lambda: (None, instance))
    yield instance
    await instance.disconnect()


def test_the_marker_breaks_text_but_not_code():
    assert split(f"{BREAK}\n\nFirst.\n\n{BREAK}{BREAK}\n  Second, indented.\n\n{BREAK}") == [
        "First.", "  Second, indented."]
    fenced = f"Run this:\n```\necho '{BREAK}'\n```\n{BREAK}Then this."
    assert split(fenced) == ["Run this:\n```\necho '" + BREAK + "'\n```", "Then this."]
    assert split(f"Write `{BREAK}` to break.{BREAK}Like that.") == [f"Write `{BREAK}` to break.", "Like that."]
    assert split("Just one.") == ["Just one."] and split(f" \n{BREAK}\n ") == []
    # While it is written, the start of a marker is not shown, and the open part is not closed.
    assert split_streaming(f"Hi.{BREAK}How are<message_br") == (["Hi."], "How are")
    assert split_streaming("```\n<message_br") == ([], "```")


async def _messages(phone, count):
    frames = []
    while len(frames) < count:
        params = (await phone.event())["params"]
        if params["type"] in ("message.new", "message.edit", "message.delete", "draft", "draft.clear"):
            frames.append((params["type"], params["payload"]))
    return frames


@pytest.mark.asyncio
async def test_a_streamed_reply_logs_each_part_as_its_marker_passes(adapter):
    phone = await connect(adapter, adapter.keys)
    try:
        await phone.hello()
        await adapter.send_draft("main", 3, "Paris it is.")
        await adapter.send_draft("main", 3, f"Paris it is.{BREAK}\n\nTrains leave")
        (_, draft), (_, first), (_, second_draft) = await _messages(phone, 3)
        assert first["text"] == "Paris it is." and first["draft_id"] == draft["draft_id"]
        assert second_draft["text"] == "Trains leave" and second_draft["draft_id"] != first["draft_id"]
        # The final text reworded the first part a little and ends the second.
        result = await adapter.send("main", f"Paris it is!{BREAK}Trains leave hourly.{BREAK}Want a seat?")
        (_, edit), (_, second), (_, third) = await _messages(phone, 3)
        assert edit == {"chat": "main", "id": first["id"], "text": "Paris it is!"}
        assert second["text"] == "Trains leave hourly." and second["draft_id"] == second_draft["draft_id"]
        assert third["text"] == "Want a seat?" and "draft_id" not in third and result.message_id == third["id"]
        texts = [m["text"] for m in adapter.store.history("main", 10**6, 10)[0]]
        assert texts == ["Paris it is!", "Trains leave hourly.", "Want a seat?"]  # each logged once
        # A late frame of that draft opens nothing.
        await adapter.send_draft("main", 3, "Paris it is.")
        assert "main" not in adapter._open_draft
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_an_unstreamed_reply_logs_its_parts_in_order_with_the_files_last(adapter, tmp_path):
    phone = await connect(adapter, adapter.keys)
    try:
        await phone.hello()
        result = await adapter.send("main", f"One.{BREAK}Two.{BREAK}Three.")
        assert [m["text"] for _, m in await _messages(phone, 3)] == ["One.", "Two.", "Three."]
        assert result.message_id == adapter.store.history("main", 10**6, 1)[0][0]["id"]
        picture = tmp_path / "trip.txt"
        picture.write_text("itinerary")
        await adapter.send_files("main", [str(picture)], caption=f"Here it is.{BREAK}The plan:")
        (_, lead), (_, last) = await _messages(phone, 2)
        assert (lead["text"], lead["attachments"]) == ("Here it is.", [])
        assert last["text"] == "The plan:" and [a["kind"] for a in last["attachments"]] == ["file"]
    finally:
        await phone.close()


def _result(text):
    result = json.loads(text)
    if not result.get("isError"):
        jsonschema.validate(result["structuredContent"], aleph_tools.OPTIONS["outputSchema"])
    return result


@pytest.mark.asyncio
async def test_options_ride_on_the_last_part_of_the_turns_next_reply(adapter):
    phone = await connect(adapter, adapter.keys)
    try:
        await phone.hello()
        assert _result(await aleph_tools.offer_options({"options": ["Yes", "No"]}))["structuredContent"] == {
            "options": ["Yes", "No"]}
        chosen = _result(await aleph_tools.offer_options({"options": [" Tomorrow ", "Friday", "Not now"]}))
        assert chosen["structuredContent"] == {"options": ["Tomorrow", "Friday", "Not now"]}  # replaced
        await adapter.send("main", f"Booked.{BREAK}When do you want to go?")
        (_, plain), (_, asked) = await _messages(phone, 2)
        assert "options" not in plain and asked["options"] == ["Tomorrow", "Friday", "Not now"]
        page = adapter.store.history("main", 10**6, 10)[0]
        assert page[-1]["options"] == ["Tomorrow", "Friday", "Not now"]  # replay and history.page carry them
        await adapter.send("main", "And another thing.")
        assert "options" not in (await _messages(phone, 1))[0][1]  # used once

        # A turn that sends no text afterwards sends none.
        from types import SimpleNamespace
        from gateway.config import Platform
        _result(await aleph_tools.offer_options({"options": ["A", "B"]}))
        await adapter.on_processing_complete(SimpleNamespace(source=SimpleNamespace(
            platform=Platform("aleph"), chat_id="main")), None)
        await adapter.send("main", "Later.")
        assert "options" not in (await _messages(phone, 1))[0][1]

        latest = adapter.store.latest()
        for bad in (["Only one"], ["a", "b", "c", "d", "e"], ["Yes", " "], ["Yes", "two\nlines"],
                    ["Yes", "x" * 41], ["Yes", "Yes "], "Yes", None):
            result = _result(await aleph_tools.offer_options({"options": bad}))
            assert result["isError"] is True and "Nothing was attached" in result["content"][0]["text"], bad
        await adapter.send("main", "No options.")
        assert "options" not in (await _messages(phone, 1))[0][1] and adapter.store.latest() == latest + 1
    finally:
        await phone.close()
