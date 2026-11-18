"""Projects on Aleph, a thread's session: turns and items logged as boundaries while their growth is
live, the person's messages steering, waiting or taken back, a silent top level, stopping, closing
and resuming, a backend that dies or cannot run, and idle processes reaped."""

from __future__ import annotations

import json
import time

import pytest
import pytest_asyncio

from plugins.platforms.aleph.projects.backend import Accepted, Delta, Exit, Item, TurnEnd
from plugins.platforms.aleph.projects.session import REAP_AFTER
from tests.gateway.aleph_backend import project_adapter, unavailable, until
from tests.gateway.aleph_phone import connect

PROJECT = "C41A0000-0000-4000-8000-000000000001"
THREAD = "8D0F0000-0000-4000-8000-000000000002"


@pytest_asyncio.fixture
async def adapter(tmp_path):
    async with project_adapter(tmp_path) as instance:
        await instance.handle_call("project.create", {
            "id": PROJECT, "name": "web", "folder": str(instance.root / "web"), "clone": None, "backend": "claude",
            "model": "opus", "effort": "low", "approval": "auto_review"})
        await instance.handle_call("thread.open", {"id": THREAD, "project": PROJECT})
        instance.backend = instance.projects.backends["claude"]
        yield instance


async def _send(adapter, message_id, text, thread=THREAD, **fields):
    return (await adapter.handle_call("message.send", {"id": message_id, "thread": thread, "text": text,
                                                       **fields}))["seq"]


async def _error(adapter, method, params):
    from plugins.platforms.aleph.server import RpcError
    with pytest.raises(RpcError) as caught:
        await adapter.handle_call(method, params)
    return caught.value


def _since(adapter, seq):
    return adapter.store.events_after(seq)


async def _logged_until(adapter, seq, event_type):
    """The events logged after ``seq``, once one of ``event_type`` is among them."""
    return await until(lambda: (events := _since(adapter, seq)) and any(e.type == event_type for e in events)
                       and events)


async def _frames_until(phone, done):
    """The frames a phone gets until ``done(frame)`` holds for the last one, but the working tree's
    summaries, which come whenever a look at it ends."""
    frames = []
    while not frames or not done(frames[-1]):
        frame = (await phone.event())["params"]
        if frame["type"] != "project.changes":
            frames.append(frame)
    return frames


def _grown(frames, item_id):
    """An item's text as a phone applying its ``item.delta`` frames has it."""
    text = ""
    for frame in frames:
        if frame["type"] == "item.delta" and frame["payload"]["id"] == item_id and "text" in frame["payload"]:
            text = text[:frame["payload"]["from"]] + frame["payload"]["text"]
    return text


# ── turns and items ───────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_turn_logs_its_boundaries_while_its_growth_is_live(adapter):
    await adapter.handle_call("thread.update", {"id": THREAD, "effort": "medium"})
    assert adapter.backend.sessions == []  # nothing runs before the thread's first input
    phone = await connect(adapter, adapter.keys)
    try:
        await phone.hello(after=adapter.store.latest())
        sent = (await phone.call("message.send", {"id": "m-1", "thread": THREAD, "text": "Fix the login"}))
        sent = sent["result"]["seq"]
        [session] = adapter.backend.sessions
        assert session.sent == ["Fix the login"]
        assert (session.spec.model, session.spec.effort, session.spec.approval) == ("opus", "medium", "auto_review")
        assert session.spec.folder == str(adapter.root / "web") and session.spec.resume is None
        session.emit(Item("msg_1", "message"))
        session.emit(Delta("msg_1", text="Looking"))
        session.emit(Delta("msg_1", text="Looking at the form"))
        session.emit(Item("todo_1", "plan", fields={"steps": [{"text": "Read the form", "status": "in_progress"}]}))
        session.emit(Item("todo_2", "plan", fields={"steps": [{"text": "Read the form", "status": "completed"}]}))
        frames = await _frames_until(phone, lambda frame: frame["type"] == "item.delta"
                                     and frame["payload"].get("steps", [{}])[0].get("status") == "completed")
        message_id = next(frame["payload"]["id"] for frame in frames if frame["type"] == "item.started")
        assert _grown(frames, message_id) == "Looking at the form"

        # A phone that catches up mid-turn gets each running item whole, from 0, after sync.done, and
        # pages the transcript in as it stands.
        other = await connect(adapter, adapter.keys)
        try:
            await other.hello(after=sent)
            live = [frame["payload"] for frame in await _frames_until(
                other, lambda frame: frame["type"] == "item.delta" and "steps" in frame["payload"])]
            assert {"from": 0, "text": "Looking at the form"}.items() <= live[0].items()
            assert live[1]["from"] == 0 and live[1]["steps"] == [{"text": "Read the form", "status": "completed"}]
            await other.call("history.load", {"thread": THREAD, "before": adapter.store.latest() + 1})
            page = (await other.events_until("history.page"))[-1]["payload"]
            kinds = [next(key for key in ("message", "turn", "item") if key in entry) for entry in page["entries"]]
            assert kinds == ["message", "turn", "item", "item"]
            turn, message, plan = page["entries"][1]["turn"], page["entries"][2]["item"], page["entries"][3]["item"]
            assert "status" not in turn and turn["cause"] == "user"
            assert message["status"] == "in_progress" and "text" not in message and plan["kind"] == "plan"
        finally:
            await other.close()

        session.emit(Item("msg_1", "message", "completed", {"text": "Looking at the form. Fixed."}))
        session.emit(TurnEnd("completed", usage={"used": 1200, "window": 200000}))
        frames += await _frames_until(phone, lambda frame: frame["type"] == "turn.completed")
    finally:
        await phone.close()
    logged = [frame for frame in frames if "seq" in frame]
    assert [frame["type"] for frame in logged] == ["message.new", "turn.started", "receipt", "item.started",
                                                   "item.started", "item.completed", "item.completed",
                                                   "turn.completed"]
    assert {frame["type"] for frame in frames if "seq" not in frame} == {"item.delta"}
    # One plan item for the turn's every update; wire ids of the server's own, the backend's kept private.
    started, plan_started, completed, plan = (frame["payload"] for frame in logged[3:7])
    assert started["id"].startswith("i_") and "text" not in started and "steps" not in plan_started
    assert completed["id"] == started["id"] and completed["text"] == "Looking at the form. Fixed."
    assert plan["id"] == plan_started["id"] and plan["steps"] == [{"text": "Read the form", "status": "completed"}]
    assert not any(key in json.dumps(logged) for key in ("msg_1", "todo_1", "todo_2"))
    end = logged[-1]["payload"]
    assert end["status"] == "completed" and end["silent"] is False and end["usage"] == {"used": 1200, "window": 200000}
    assert logged[2]["payload"]["thread"] == THREAD and logged[2]["payload"]["read_up_to"] == "m-1"
    entries, _ = adapter.store.thread_history(THREAD, adapter.store.latest() + 1, 10)
    assert entries[1]["turn"]["status"] == "completed" and entries[2]["item"]["text"] == "Looking at the form. Fixed."


# ── input ─────────────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_messages_steer_the_turn_wait_for_its_end_or_are_taken_back(adapter):
    first = await _send(adapter, "m-1", "Go")
    assert await _send(adapter, "m-1", "Go again") == first  # a resend answers as the first time
    [session] = adapter.backend.sessions
    session.accepting = False
    mark = adapter.store.latest()
    steered = await _send(adapter, "m-2", "Also check the API")
    assert session.sent == ["Go", "Also check the API"]
    assert [event.type for event in _since(adapter, mark)] == ["message.new"]  # no receipt until it lands
    session.emit(Accepted(session.keys[-1]))
    receipt = (await _logged_until(adapter, steered, "receipt"))[-1]
    assert receipt.payload["read_up_to"] == "m-2"

    queued = await _send(adapter, "m-3", "Then the docs", delivery="after_turn")
    assert await _send(adapter, "m-3", "Then the docs", delivery="after_turn") == queued
    await _send(adapter, "m-4", "Never mind", delivery="after_turn")
    cancelled = (await adapter.handle_call("message.cancel", {"id": "m-4"}))["seq"]
    assert (await adapter.handle_call("message.cancel", {"id": "m-4"}))["seq"] == cancelled
    await _send(adapter, "m-5", "And the changelog", delivery="after_turn")
    assert [event.type for event in _since(adapter, receipt.seq)] == ["message.queued", "message.queued",
                                                                      "message.cancel", "message.queued"]
    assert len(session.sent) == 2

    # At the turn's end every queued message starts the next turn together, each logged as its own.
    session.accepting = True
    mark = adapter.store.latest()
    session.emit(TurnEnd("completed"))
    events = await _logged_until(adapter, mark, "receipt")
    assert [event.type for event in events] == ["turn.completed", "message.new", "message.new", "turn.started",
                                                "receipt"]
    assert [event.payload["id"] for event in events[1:3]] == ["m-3", "m-5"]
    assert session.sent[-1] == "Then the docs\n\nAnd the changelog" and events[-1].payload["read_up_to"] == "m-5"
    assert (await _error(adapter, "message.cancel", {"id": "m-3"})).code == -32602
    assert await _send(adapter, "m-3", "Then the docs", delivery="after_turn") == queued

    # A steer that misses its turn: the backend runs it as its own next turn.
    session.accepting = False
    await _send(adapter, "m-6", "One more")
    mark = adapter.store.latest()
    session.emit(TurnEnd("completed"))
    session.emit(Accepted(session.keys[-1]))
    events = await _logged_until(adapter, mark, "receipt")
    assert [event.type for event in events] == ["turn.completed", "turn.started", "receipt"]
    assert events[1].payload["cause"] == "user" and events[2].payload["read_up_to"] == "m-6"


@pytest.mark.asyncio
async def test_stop_close_and_reopen_resume_the_same_backend_session(adapter):
    assert await adapter.handle_call("thread.stop", {"thread": THREAD}) == {}  # nothing running is no error
    await _send(adapter, "m-1", "Run the tests")
    [session] = adapter.backend.sessions
    session.emit(Item("toolu_1", "tool", fields={"tool": "execute", "title": "npm test", "input": "npm test"}))
    await adapter.handle_call("thread.stop", {"thread": THREAD})
    assert session.interrupted == 1
    held = await _send(adapter, "m-2", "Run only the unit tests")
    assert session.sent == ["Run the tests"]  # held, not handed to the dying turn
    session.emit(TurnEnd("completed"))
    events = await _logged_until(adapter, held, "receipt")
    assert [event.type for event in events] == ["item.completed", "turn.completed", "turn.started", "receipt"]
    assert events[0].payload["status"] == "failed" and events[1].payload["status"] == "interrupted"
    assert session.sent[-1] == "Run only the unit tests"

    mark = adapter.store.latest()
    await adapter.handle_call("thread.close", {"id": THREAD})
    assert session.closed
    assert [(event.type, event.payload.get("status")) for event in _since(adapter, mark)] == [
        ("turn.completed", "interrupted"), ("thread.close", None)]
    reopened = await _send(adapter, "m-3", "Carry on")
    assert adapter.store.events_after(reopened - 2)[0].payload["closed"] is False
    resumed = adapter.backend.sessions[1]
    assert resumed.spec.resume == session.session_id and resumed.sent == ["Carry on"]


@pytest.mark.asyncio
async def test_a_top_level_reply_of_no_reply_is_never_logged(adapter):
    phone = await connect(adapter, adapter.keys)
    try:
        await phone.hello(after=adapter.store.latest())
        await phone.call("message.send", {"id": "m-1", "thread": PROJECT, "text": "Thanks"})
        [session] = adapter.backend.sessions
        for event in (Item("msg_1", "message"), Delta("msg_1", text="NO_"), Delta("msg_1", text="NO_REPLY"),
                      Item("msg_1", "message", "completed", {"text": "NO_REPLY"}), TurnEnd("completed")):
            session.emit(event)
        frames = await _frames_until(phone, lambda frame: frame["type"] == "turn.completed")
        assert [frame["type"] for frame in frames] == ["message.new", "turn.started", "receipt", "turn.completed"]
        assert frames[-1]["payload"]["silent"] is True

        # A reply that only starts like the token shows once it no longer can be it, whole from 0.
        await phone.call("message.send", {"id": "m-2", "thread": PROJECT, "text": "Should we ship?"})
        for event in (Item("msg_2", "message"), Delta("msg_2", text="NO_REP"),
                      Delta("msg_2", text="NO_REPLY needed, ship it")):
            session.emit(event)
        frames = await _frames_until(phone, lambda frame: frame["type"] == "item.delta")
        assert frames[-1]["payload"]["from"] == 0 and frames[-1]["payload"]["text"] == "NO_REPLY needed, ship it"
        session.emit(Item("msg_2", "message", "completed", {"text": "NO_REPLY needed, ship it."}))
        session.emit(TurnEnd("completed"))
        frames += await _frames_until(phone, lambda frame: frame["type"] == "turn.completed")
    finally:
        await phone.close()
    assert [frame["type"] for frame in frames if "seq" in frame] == ["message.new", "turn.started", "receipt",
                                                                     "item.started", "item.completed",
                                                                     "turn.completed"]
    assert frames[-1]["payload"]["silent"] is False


# ── failures ──────────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_backend_that_dies_or_cannot_run_fails_the_turn(adapter):
    sent = await _send(adapter, "m-1", "Build it")
    [session] = adapter.backend.sessions
    session.emit(Item("toolu_1", "tool", fields={"tool": "execute", "title": "make", "input": "make"}))
    session.emit(Exit("Claude Code exited with status 1."))
    events = await _logged_until(adapter, sent, "turn.completed")
    item, turn = events[-2].payload, events[-1].payload
    assert item["status"] == "failed" and item["exit_code"] is None
    assert turn["status"] == "failed" and turn["error"] == "Claude Code exited with status 1."
    retried = await _send(adapter, "m-2", "Try again")
    assert adapter.backend.sessions[1].spec.resume == session.session_id
    await _logged_until(adapter, retried, "receipt")

    # The message is logged first; the turn fails with the reason, and so does the call.
    adapter.backend.info = unavailable("signed_out", "Claude Code is signed out", "claude", "Claude Code")
    adapter.projects.invalidate("claude")
    mark = adapter.store.latest()
    error = await _error(adapter, "message.send", {"id": "m-3", "thread": PROJECT, "text": "Hello"})
    assert error.code == 4003 and error.data == {"reason": "unavailable", "message": "Claude Code is signed out"}
    events = _since(adapter, mark)
    assert [event.type for event in events] == ["message.new", "turn.started", "turn.completed"]
    assert events[-1].payload["status"] == "failed" and events[-1].payload["error"] == "Claude Code is signed out"


# ── idle processes ────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_an_idle_thread_process_is_reaped_but_never_the_top_levels(adapter):
    for thread, message_id in ((THREAD, "m-1"), (PROJECT, "m-2")):
        mark = await _send(adapter, message_id, "Go", thread=thread)
        adapter.backend.sessions[-1].emit(TurnEnd("completed"))
        await _logged_until(adapter, mark, "turn.completed")
    thread_session, top_session = adapter.backend.sessions
    hub, later = adapter.projects.hub, time.monotonic() + REAP_AFTER + 1
    assert await hub.reap(later - 2) == []
    thread_session.idle = False  # something of it still runs in the background
    assert await hub.reap(later) == []
    thread_session.idle = True
    assert await hub.reap(later) == [THREAD]
    assert thread_session.closed and not top_session.closed
    await _send(adapter, "m-3", "And now")
    assert adapter.backend.sessions[2].spec.resume == thread_session.session_id
