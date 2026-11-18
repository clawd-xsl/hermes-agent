"""Projects on Aleph, requests: approvals and questions as the chat's cards in a thread, how each choice
lands with the backend, session grants across the project, the timeout that declines, a typed message
answering or declining, the server's own plan request, cards that always settle, and what is pushed."""

from __future__ import annotations

import asyncio

import pytest
import pytest_asyncio

from plugins.platforms.aleph.projects.backend import (
    SECRET, Answers, Approve, Ask, Deny, Exit, Item, Question, TurnEnd)
from tests.gateway.aleph_backend import project_adapter, until
from tests.gateway.test_aleph_push import PUSH_TOKEN, FakeSender

PROJECT = "C41A0000-0000-4000-8000-000000000001"
THREAD = "8D0F0000-0000-4000-8000-000000000002"
OTHER = "8D0F0000-0000-4000-8000-000000000003"


@pytest_asyncio.fixture
async def adapter(tmp_path):
    async with project_adapter(tmp_path) as instance:
        await instance.handle_call("project.create", {
            "id": PROJECT, "name": "web", "folder": str(instance.root / "web"), "clone": None, "backend": "claude",
            "model": None, "effort": None, "approval": "ask"})
        for thread in (THREAD, OTHER):
            await instance.handle_call("thread.open", {"id": thread, "project": PROJECT})
        instance.backend = instance.projects.backends["claude"]
        yield instance


async def _session(adapter, thread=THREAD, message_id=None, text="Go"):
    """The thread's backend session, a turn running: started by a message from the person."""
    await adapter.handle_call("message.send", {"id": message_id or f"m-{thread[-1]}", "thread": thread, "text": text})
    return next(session for session in reversed(adapter.backend.sessions) if session.spec.thread == thread)


async def _logged(adapter, event_type, after=0):
    """The payload of the newest ``event_type`` logged after ``after``, once there is one."""
    events = await until(lambda: [e for e in adapter.store.events_after(after) if e.type == event_type])
    return events[-1].payload


async def _asked(adapter, session, ask, kind="approval.request"):
    mark = adapter.store.latest()
    session.emit(ask)
    return await _logged(adapter, kind, mark)


async def _error(adapter, method, params):
    from plugins.platforms.aleph.server import RpcError
    with pytest.raises(RpcError) as caught:
        await adapter.handle_call(method, params)
    return caught.value


def _execute(key, command):
    return Item(key, "tool", fields={"tool": "execute", "title": command, "input": command})


# ── approvals ─────────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_an_approval_lands_with_the_backend_and_a_denied_item_completes_declined(adapter):
    session = await _session(adapter)
    session.emit(_execute("toolu_1", "npm test"))
    item = await _logged(adapter, "item.started")
    request = await _asked(adapter, session, Ask("ask_1", "command", item="toolu_1", text="Runs the suite",
                                                 command="npm test", cwd=session.spec.folder, grant="npm"))
    assert {key: request[key] for key in ("project", "thread", "kind", "item", "command", "cwd", "text")} == {
        "project": PROJECT, "thread": THREAD, "kind": "command", "item": item["id"], "command": "npm test",
        "cwd": session.spec.folder, "text": "Runs the suite"}
    assert [(c["choice"], c["label"]) for c in request["choices"]] == [
        ("once", "Allow Once"), ("session", "For This Session"), ("deny", "Don't Allow")]
    assert (await adapter.projects.snapshot())[0]["threads"][0]["state"] == "waiting"
    error = await _error(adapter, "approval.respond", {"request_id": request["request_id"], "choice": "always"})
    assert error.code == -32602 and error.data == {"choices": ["once", "session", "deny"]}
    await adapter.handle_call("approval.respond", {"request_id": request["request_id"], "choice": "once"})
    assert session.answers == [("ask_1", Approve())]
    resolved = await _logged(adapter, "approval.resolved")
    assert resolved == {"project": PROJECT, "thread": THREAD, "request_id": request["request_id"],
                        "choice": "once", "by": "user"}
    assert (await _error(adapter, "approval.respond", {"request_id": request["request_id"],
                                                       "choice": "once"})).code == -32602

    # Denied: the backend is refused, and the item it held completes declined whatever the backend says.
    session.emit(Item("toolu_2", "tool", fields={"tool": "edit", "title": "src/api.ts", "input": "…"}))
    files = ({"path": "src/api.ts", "added": 2, "removed": 1},)
    request = await _asked(adapter, session, Ask("ask_2", "edit", item="toolu_2", files=files))
    assert request["kind"] == "edit" and request["files"] == list(files) and "command" not in request
    mark = adapter.store.latest()
    await adapter.handle_call("approval.respond", {"request_id": request["request_id"], "choice": "deny"})
    assert session.answers[-1] == ("ask_2", Deny())
    session.emit(Item("toolu_2", "tool", "failed", {"files": []}))
    completed = await _logged(adapter, "item.completed", mark)
    assert completed["status"] == "declined"


@pytest.mark.asyncio
async def test_a_session_grant_covers_the_project_until_its_approval_mode_changes(adapter):
    session = await _session(adapter)
    request = await _asked(adapter, session, Ask("ask_1", "command", command="git status", grant="git"))
    await adapter.handle_call("approval.respond", {"request_id": request["request_id"], "choice": "session"})
    assert session.answers == [("ask_1", Approve(session=True))]
    edit = await _asked(adapter, session, Ask("ask_2", "edit", files=({"path": "src/a.ts", "added": 1, "removed": 0},)))
    await adapter.handle_call("approval.respond", {"request_id": edit["request_id"], "choice": "session"})

    # Another thread's matching asks are allowed with the backend's own session grant, with no card:
    # a command of the same rule, an edit of any file in the folder.
    other = await _session(adapter, OTHER)
    cards = len([e for e in adapter.store.events_after(0) if e.type == "approval.request"])
    other.emit(Ask("ask_3", "command", command="git log --oneline", grant="git"))
    other.emit(Ask("ask_4", "edit", files=({"path": "docs/readme.md", "added": 3, "removed": 0},)))
    await until(lambda: len(other.answers) == 2)
    assert other.answers == [("ask_3", Approve(session=True)), ("ask_4", Approve(session=True))]
    # A file outside the folder, another rule or a permission of another thread still asks.
    for ask in (Ask("ask_5", "edit", files=({"path": "../elsewhere.txt", "added": 1, "removed": 0},)),
                Ask("ask_6", "command", command="npm test", grant="npm")):
        request = await _asked(adapter, other, ask)
        await adapter.handle_call("approval.respond", {"request_id": request["request_id"], "choice": "once"})
    permission = await _asked(adapter, session, Ask("ask_7", "permission", text="Network access", grant="network"))
    await adapter.handle_call("approval.respond", {"request_id": permission["request_id"], "choice": "session"})
    await _asked(adapter, other, Ask("ask_8", "permission", text="Network access", grant="network"))
    assert len([e for e in adapter.store.events_after(0) if e.type == "approval.request"]) == cards + 4

    # A new approval mode ends what the person allowed under the old one.
    await adapter.handle_call("project.update", {"id": PROJECT, "approval": "auto_review"})
    request = await _asked(adapter, other, Ask("ask_9", "command", command="git push", grant="git"))
    assert request["command"] == "git push"


@pytest.mark.asyncio
async def test_an_unanswered_request_is_declined_after_the_timeout(adapter):
    adapter.request_timeout = 0.05
    session = await _session(adapter)
    session.emit(_execute("toolu_1", "rm -rf build"))
    request = await _asked(adapter, session, Ask("ask_1", "command", item="toolu_1", command="rm -rf build"))
    resolved = await _logged(adapter, "approval.resolved")
    assert resolved == {"project": PROJECT, "thread": THREAD, "request_id": request["request_id"],
                        "choice": None, "by": "timeout"}
    assert await until(lambda: session.answers) == [("ask_1", Deny())]
    mark = adapter.store.latest()
    session.emit(TurnEnd("completed"))
    assert (await _logged(adapter, "item.completed", mark))["status"] == "declined"

    await _session(adapter, message_id="m-later")
    question = await _asked(adapter, session, Ask("ask_2", "question", questions=(Question("Which database?"),)),
                            "clarify.request")
    resolved = await _logged(adapter, "clarify.resolved")
    assert resolved == {"project": PROJECT, "thread": THREAD, "clarify_id": question["clarify_id"], "answer": None,
                        "choices": None, "by": "timeout"}
    assert (await until(lambda: session.answers[1:]))[0] == ("ask_2", Deny())


# ── what the person types ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_typed_message_declines_an_approval_or_answers_a_question_of_its_thread(adapter):
    session = await _session(adapter)
    other = await _session(adapter, OTHER)
    session.emit(_execute("toolu_1", "npm publish"))
    request = await _asked(adapter, session, Ask("ask_1", "command", item="toolu_1", command="npm publish"))
    elsewhere = await _asked(adapter, other, Ask("ask_2", "command", command="make"))
    mark = adapter.store.latest()
    seq = (await adapter.handle_call("message.send", {"id": "t-1", "thread": THREAD,
                                                      "text": "Don't publish, bump the version"}))["seq"]
    assert session.answers == [("ask_1", Deny("Don't publish, bump the version"))]
    assert session.sent == ["Go"]  # no turn of its own: the agent reads it with the refusal
    events = adapter.store.events_after(mark)
    assert [event.type for event in events] == ["message.new", "approval.resolved", "receipt"]
    assert events[0].seq == seq and events[1].payload["request_id"] == request["request_id"]
    assert events[1].payload["choice"] == "deny" and events[2].payload["read_up_to"] == "t-1"
    assert adapter.projects.desks[PROJECT].has(elsewhere["request_id"]) and other.answers == []

    question = Question("Which database?", ("Postgres", "SQLite"), recommended=0)
    card = await _asked(adapter, session, Ask("ask_3", "question", questions=(question,)), "clarify.request")
    assert card["choices"] == ["Postgres", "SQLite"] and card["recommended"] == 0 and "index" not in card
    await adapter.handle_call("message.send", {"id": "t-2", "thread": THREAD, "text": "2"})
    assert session.answers[-1] == ("ask_3", Answers((("SQLite",),)))
    resolved = await _logged(adapter, "clarify.resolved")
    assert (resolved["answer"], resolved["choices"], resolved["by"]) == ("SQLite", [1], "user")
    assert session.sent == ["Go"]


@pytest.mark.asyncio
async def test_a_secret_answer_reaches_the_backend_and_is_never_stored(adapter):
    session = await _session(adapter)
    question = Question("The deploy key?", secret=True)
    card = await _asked(adapter, session, Ask("ask_1", "question", questions=(question,)), "clarify.request")
    assert card["secret"] is True
    await adapter.handle_call("message.send", {"id": "s-1", "thread": THREAD, "text": "hunter2-key"})
    assert session.answers == [("ask_1", Answers((("hunter2-key",),)))]
    resolved = await _logged(adapter, "clarify.resolved")
    assert resolved["answer"] == SECRET
    adapter.store._db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    stored = b"".join(path.read_bytes() for path in adapter.store.root.rglob("*") if path.is_file())
    assert b"hunter2-key" not in stored


@pytest.mark.asyncio
async def test_a_series_of_questions_is_answered_together_after_the_last(adapter):
    session = await _session(adapter)
    questions = (Question("Which database?", ("Postgres", "SQLite")),
                 Question("Which features?", ("Auth", "Billing", "Search"), multi_select=True),
                 Question("Anything else?"))
    first = await _asked(adapter, session, Ask("ask_1", "question", questions=questions), "clarify.request")
    assert (first["project"], first["thread"], first["index"], first["count"]) == (PROJECT, THREAD, 0, 3)
    mark = adapter.store.latest()
    await adapter.handle_call("clarify.respond", {"clarify_id": first["clarify_id"], "choices": [0],
                                                  "message_id": "a-1"})
    second = await _logged(adapter, "clarify.request", mark)
    assert second["index"] == 1 and second["multi_select"] is True and second["clarify_id"] != first["clarify_id"]
    error = await _error(adapter, "clarify.respond", {"clarify_id": second["clarify_id"], "choices": [3],
                                                      "message_id": "a-2"})
    assert error.code == -32602
    await adapter.handle_call("clarify.respond", {"clarify_id": second["clarify_id"], "choices": [0, 2],
                                                  "message_id": "a-2"})
    assert session.answers == []  # nothing goes back before the last answer
    third = await _logged(adapter, "clarify.request", mark)
    await adapter.handle_call("clarify.respond", {"clarify_id": third["clarify_id"], "skip": True})
    assert session.answers == [("ask_1", Answers((("Postgres",), ("Auth", "Search"), ())))]
    events = [(e.type, e.payload.get("text") or e.payload.get("answer")) for e in adapter.store.events_after(mark)]
    assert events == [("message.new", "Postgres"), ("clarify.resolved", "Postgres"), ("receipt", None),
                      ("clarify.request", None), ("message.new", "Auth, Search"),
                      ("clarify.resolved", "Auth, Search"), ("receipt", None), ("clarify.request", None),
                      ("clarify.resolved", None)]


# ── plans ─────────────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_plan_is_accepted_through_the_backends_request_or_the_servers_own(adapter):
    await adapter.handle_call("thread.update", {"id": THREAD, "plan": True})
    session = await _session(adapter)
    request = await _asked(adapter, session, Ask("plan_1", "plan", text="1. Add the endpoint"))
    assert [(c["choice"], c["label"]) for c in request["choices"]] == [("once", "Start Coding"),
                                                                       ("deny", "Keep Planning")]
    await adapter.handle_call("approval.respond", {"request_id": request["request_id"], "choice": "once"})
    assert (await _logged(adapter, "thread.update"))["plan"] is False and session.answers == [("plan_1", Approve())]

    # A backend that only ends a plan-mode turn with its plan gets the server's request at its end.
    adapter.backend.asks_to_leave_plan = False
    await adapter.handle_call("thread.update", {"id": OTHER, "plan": True})
    other = await _session(adapter, OTHER)
    mark = adapter.store.latest()
    for event in (Item("msg_1", "message"), Item("msg_1", "message", "completed", {"text": "The plan: …"}),
                  TurnEnd("completed")):
        other.emit(event)
    request = await _logged(adapter, "approval.request", mark)
    assert (request["thread"], request["kind"], request["item"]) == (OTHER, "plan", None)
    mark = adapter.store.latest()
    await adapter.handle_call("approval.respond", {"request_id": request["request_id"], "choice": "once"})
    assert other.sent[-1] == "Implement the plan." and other.spec.plan is False
    events = adapter.store.events_after(mark)
    assert [event.type for event in events] == ["approval.resolved", "thread.update", "turn.started"]
    assert events[2].payload["cause"] == "user"  # with no message.new

    # Keep Planning leaves the thread planning, and nothing starts.
    other.emit(TurnEnd("completed"))
    await adapter.handle_call("thread.update", {"id": OTHER, "plan": True})
    other = await _session(adapter, OTHER, message_id="m-again")
    mark = adapter.store.latest()
    other.emit(Item("msg_2", "message", "completed", {"text": "Another plan"}))
    other.emit(TurnEnd("completed"))
    request = await _logged(adapter, "approval.request", mark)
    await adapter.handle_call("approval.respond", {"request_id": request["request_id"], "choice": "deny"})
    assert other.sent[-1] == "Go" and other.answers == []
    assert adapter.store.events_after(mark)[-1].type == "approval.resolved"


# ── cards always settle ───────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_requests_no_one_can_answer_settle_when_their_thread_closes_or_its_process_exits(adapter):
    session = await _session(adapter)
    approval = await _asked(adapter, session, Ask("ask_1", "command", command="npm test"))
    mark = adapter.store.latest()
    await adapter.handle_call("thread.close", {"id": THREAD})
    assert [(e.type, e.payload.get("by")) for e in adapter.store.events_after(mark)][:2] == [
        ("approval.resolved", "cancelled"), ("turn.completed", None)]
    assert adapter.store.events_after(mark)[0].payload["request_id"] == approval["request_id"]

    top = await _session(adapter, PROJECT)
    question = await _asked(adapter, top, Ask("ask_2", "question", questions=(Question("Ship it?"),)),
                            "clarify.request")
    mark = adapter.store.latest()
    top.emit(Exit("Claude Code exited with status 1."))
    resolved = await _logged(adapter, "clarify.resolved", mark)
    assert (resolved["clarify_id"], resolved["by"], resolved["answer"]) == (question["clarify_id"], "timeout", None)
    assert top.answers == [] and adapter.store.unsettled_cards() == []


# ── push ──────────────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_project_pushes_its_requests_and_top_level_turns_and_settled_cards_leave_the_lock_screen(adapter):
    adapter._push = FakeSender()
    adapter.store.set_push_token(adapter.device.id, PUSH_TOKEN, "development")
    sender = adapter._push

    async def pushed(count):
        await until(lambda: len(sender.sent) >= count)
        return sorted((payload for _, _, payload in sender.sent), key=lambda payload: payload["aleph"]["seq"])

    session = await _session(adapter)
    request = await _asked(adapter, session, Ask("ask_1", "command", command="npm test"))
    [alert] = await pushed(1)
    assert alert["aps"] == {"alert": {"body": "Approval needed"}, "sound": "default", "mutable-content": 1,
                            "thread-id": PROJECT, "interruption-level": "time-sensitive"}
    assert alert["aleph"] == {"seq": adapter.store.latest(), "project": PROJECT, "thread": THREAD}
    await adapter.handle_call("approval.respond", {"request_id": request["request_id"], "choice": "once"})
    await until(lambda: sender.background)
    assert sender.background == [(PUSH_TOKEN, {"aps": {"content-available": 1},
                                               "aleph": {"seq": adapter.store.latest()}})]

    # A thread's turns do not push; the top level's do when it spoke or failed, not when silent.
    session.emit(TurnEnd("completed"))
    top = await _session(adapter, PROJECT)
    await _asked(adapter, top, Ask("ask_2", "question", questions=(Question("Ship it?"),)), "clarify.request")
    top.emit(Item("msg_1", "message", "completed", {"text": "Shipped."}))
    top.emit(TurnEnd("completed"))
    await _logged(adapter, "turn.completed", adapter.store.latest())
    silent = (Item("msg_2", "message", "completed", {"text": "NO_REPLY"}), TurnEnd("completed"))
    for message_id, ends in (("m-a", (TurnEnd("failed", "Out of credits"),)), ("m-b", silent)):
        mark = adapter.store.latest()
        await adapter.handle_call("message.send", {"id": message_id, "thread": PROJECT, "text": "Again"})
        for end in ends:
            top.emit(end)
        await _logged(adapter, "turn.completed", mark)
    await asyncio.sleep(0.05)
    assert len(sender.sent) == 4
    bodies = [(p["aps"]["alert"]["body"], p["aleph"]["thread"]) for p in await pushed(4)]
    assert bodies == [("Approval needed", THREAD), ("Question", PROJECT), ("Finished", PROJECT),
                      ("Failed", PROJECT)]
    # The question settled with its turn: it leaves the Lock Screen too.
    await until(lambda: len(sender.background) == 2)

    # A chat's card, the same.
    await adapter.send_clarify("main", "Which one?", ["a", "b"], "c1", "s")
    await adapter.retire_clarify_card("c1", "")
    await until(lambda: len(sender.background) == 3)
    assert sender.background[-1][1]["aleph"] == {"seq": adapter.store.latest()}
