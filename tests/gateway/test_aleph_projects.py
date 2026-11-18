"""Projects on Aleph: projects and threads idempotent by id, their events in the one log (replayed,
listed on a reset, dropped with the project), the folder browser bounded to the projects root, the
backends on offer, and what a restart settles."""

from __future__ import annotations

import os

import pytest
import pytest_asyncio

from plugins.platforms.aleph.projects.backend import Ask, Delta, Item
from tests.gateway.aleph_backend import FakeBackend, project_adapter, until
from tests.gateway.aleph_phone import connect

PROJECT = "C41A0000-0000-4000-8000-000000000001"
THREAD = "8D0F0000-0000-4000-8000-000000000002"
SIDE = "9B2E0000-0000-4000-8000-000000000003"


@pytest_asyncio.fixture
async def adapter(tmp_path):
    async with project_adapter(tmp_path) as instance:
        yield instance


def _create(adapter, **fields):
    return {"id": PROJECT, "name": "web", "folder": str(adapter.root / "web"), "clone": None, "backend": "claude",
            "model": None, "effort": None, "approval": "auto_review", **fields}


async def _call(adapter, method, params):
    return await adapter.handle_call(method, params)


async def _error(adapter, method, params):
    from plugins.platforms.aleph.server import RpcError
    with pytest.raises(RpcError) as caught:
        await adapter.handle_call(method, params)
    return caught.value


# ── projects and threads ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_project_and_a_thread_are_created_once_by_their_ids(adapter):
    phone = await connect(adapter, adapter.keys)
    try:
        await phone.hello()
        created = await phone.call("project.create", _create(adapter))
        event = (await phone.logged())["params"]
        assert event["type"] == "project.new" and created["result"] == {"seq": event["seq"]}
        project = event["payload"]
        assert project["id"] == PROJECT and project["folder"] == str(adapter.root / "web")
        assert project["host"] == "local" and project["backend"] == "claude" and project["approval"] == "auto_review"
        again = await phone.call("project.create", _create(adapter, name="other"))
        assert again["result"] == created["result"] and adapter.store.latest() == event["seq"]

        opened = await phone.call("thread.open", {"id": THREAD, "project": PROJECT, "name": None})
        thread = (await phone.logged())["params"]
        assert thread["type"] == "thread.new" and opened["result"] == {"seq": thread["seq"]}
        assert thread["payload"]["project"] == PROJECT and thread["payload"]["opened_by"] == "user"
        assert thread["payload"]["closed"] is False and thread["payload"]["plan"] is False
        assert (await phone.call("thread.open", {"id": THREAD, "project": PROJECT}))["result"] == opened["result"]
        assert adapter.store.latest() == thread["seq"]
    finally:
        await phone.close()
    # An id names one thing: the top level is the project's own thread, and a project's id is no thread's.
    for bad in ({"id": PROJECT, "project": PROJECT}, {"id": "t2", "project": "nope"}):
        assert (await _error(adapter, "thread.open", bad)).code == -32602
    assert (await _error(adapter, "project.create", _create(adapter, id=THREAD))).code == -32602


@pytest.mark.asyncio
async def test_a_project_takes_only_what_its_backend_offers(adapter):
    for bad in (_create(adapter, backend="gemini"), _create(adapter, model="gpt"),
                _create(adapter, model="sonnet", effort="high"), _create(adapter, approval="allow_all"),
                _create(adapter, approval=None), _create(adapter, folder="/etc"),
                _create(adapter, folder=str(adapter.root / "missing")), _create(adapter, name="  ")):
        assert (await _error(adapter, "project.create", bad)).code == -32602, bad
    error = await _error(adapter, "project.create", _create(adapter, backend="codex"))
    assert error.code == 4003 and error.data == {"reason": "unavailable", "message": "codex is not on PATH"}
    assert adapter.store.latest() == 0
    assert await _call(adapter, "project.create", _create(adapter, model="opus", effort="low"))


@pytest.mark.asyncio
async def test_an_update_that_changes_nothing_logs_nothing(adapter):
    await _call(adapter, "project.create", _create(adapter))
    await _call(adapter, "thread.open", {"id": THREAD, "project": PROJECT})
    latest = adapter.store.latest()
    assert await _call(adapter, "project.update", {"id": PROJECT, "name": "web", "model": None}) == {"seq": None}
    assert await _call(adapter, "thread.update", {"id": THREAD, "approval": None, "plan": False}) == {"seq": None}
    assert adapter.store.latest() == latest

    renamed = await _call(adapter, "project.update", {"id": PROJECT, "name": "Web app", "approval": "ask"})
    assert renamed == {"seq": latest + 1}
    payload = adapter.store.events_after(latest)[0].payload
    assert payload["name"] == "Web app" and payload["approval"] == "ask" and payload["folder"]
    assert (await _call(adapter, "thread.update", {"id": THREAD, "effort": "medium"}))["seq"] == latest + 2
    # The top level's name and settings are the project's; only its plan mode is its own.
    assert (await _error(adapter, "thread.update", {"id": PROJECT, "name": "x"})).code == -32602
    assert (await _call(adapter, "thread.update", {"id": PROJECT, "plan": True}))["seq"] == latest + 3
    assert (await _error(adapter, "thread.close", {"id": PROJECT})).code == -32602


# ── the one log ───────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_phone_that_was_away_replays_the_projects_events(adapter):
    phone = await connect(adapter, adapter.keys)
    try:
        await phone.hello()
        await phone.call("chat.create", {"id": SIDE})
        seen = (await phone.logged())["params"]["seq"]
    finally:
        await phone.close()
    await _call(adapter, "project.create", _create(adapter))
    await _call(adapter, "thread.open", {"id": THREAD, "project": PROJECT})
    sent = await _call(adapter, "message.send", {"id": "m-1", "thread": THREAD, "text": "Use the new endpoint",
                                                 "comments": [{"path": "src/api.ts", "line": 42, "side": "new",
                                                               "text": "This should retry"}]})
    assert await _call(adapter, "message.send", {"id": "m-1", "thread": THREAD, "text": "again"}) == sent
    assert await _call(adapter, "thread.close", {"id": THREAD})
    phone = await connect(adapter, adapter.keys)
    try:
        answer, replay = await phone.hello(after=seen)
        assert answer["result"]["reset"] is False
        logged = [(event["type"], event["seq"]) for event in replay if "seq" in event]
        # The message started a turn, which closing the thread interrupted.
        assert [kind for kind, _ in logged] == ["project.new", "thread.new", "message.new", "turn.started", "receipt",
                                                "turn.completed", "thread.close"]
        message = replay[2]["payload"]
        assert message["project"] == PROJECT and message["thread"] == THREAD and "chat" not in message
        assert message["comments"][0]["line"] == 42 and sent == {"seq": logged[2][1]}
        # A message reopens a closed thread before it is logged.
        await phone.call("message.send", {"id": "m-2", "thread": THREAD, "text": "One more thing"})
        reopened, again = (await phone.logged())["params"], (await phone.logged())["params"]
        assert reopened["type"] == "thread.update" and reopened["payload"]["closed"] is False
        assert again["type"] == "message.new" and again["payload"]["id"] == "m-2"
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_a_reset_lists_every_project_with_its_threads(adapter):
    await _call(adapter, "chat.create", {"id": SIDE})
    await _call(adapter, "project.create", _create(adapter))
    await _call(adapter, "thread.open", {"id": THREAD, "project": PROJECT, "name": "fix-login"})
    sent = (await _call(adapter, "message.send", {"id": "m-1", "thread": THREAD, "text": "Go"}))["seq"]
    await _call(adapter, "message.send", {"id": "m-2", "thread": THREAD, "text": "Then this", "delivery": "after_turn"})
    await adapter._log("approval.request", {"project": PROJECT, "thread": PROJECT, "request_id": "r-1",
                                            "kind": "command", "command": "npm test", "text": "", "choices": []})
    phone = await connect(adapter, adapter.keys)
    try:
        answer, _ = await phone.hello()
        result = answer["result"]
        assert result["reset"] is True and [chat["chat"] for chat in result["chats"]] == [SIDE]
        [entry] = result["projects"]
        assert entry["project"]["id"] == PROJECT and entry["project"]["name"] == "web"
        [thread] = entry["threads"]  # the top level is the project itself
        assert thread["id"] == THREAD and thread["name"] == "fix-login" and thread["state"] == "running"
        assert [queued["id"] for queued in thread["queued"]] == ["m-2"] and thread["last_active_at"]

        # The thread's transcript pages in as a chat's does.
        await phone.call("history.load", {"thread": THREAD, "before": result["latest"] + 1})
        [page] = [frame["payload"] for frame in await phone.events_until("history.page") if frame["type"] == "history.page"]
        assert page["thread"] == THREAD and page["next"] is None
        assert [next(key for key in ("message", "turn", "item") if key in entry) for entry in page["entries"]] == \
            ["message", "turn"]
        assert page["entries"][0]["seq"] == sent and page["entries"][0]["message"]["read"] is True
        assert (await phone.call("history.load", {"thread": "nope", "before": 5}))["error"]["code"] == -32602
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_deleting_a_project_drops_its_events_and_leaves_its_folder(adapter):
    await _call(adapter, "chat.create", {"id": SIDE})
    before = adapter.store.latest()
    await _call(adapter, "project.create", _create(adapter))
    await _call(adapter, "thread.open", {"id": THREAD, "project": PROJECT})
    await _call(adapter, "message.send", {"id": "m-1", "thread": THREAD, "text": "Go"})
    await _call(adapter, "thread.close", {"id": THREAD})
    after_side = (await adapter.send(SIDE, "still here")).message_id
    bodies = adapter.state_dir / "projects" / PROJECT
    bodies.mkdir(parents=True, exist_ok=True)  # its sessions' memory is there already
    (bodies / "i_1.out").write_text("output")

    deleted = await _call(adapter, "project.delete", {"id": PROJECT})
    assert await _call(adapter, "project.delete", {"id": PROJECT}) == deleted
    assert (adapter.root / "web").is_dir() and not bodies.exists()
    replay = adapter.store.events_after(before)
    assert [event.type for event in replay] == ["message.new", "project.delete"]
    assert replay[0].payload["id"] == after_side and deleted == {"seq": replay[1].seq}
    assert adapter.store.message("m-1") is None
    for method, params in (("thread.open", {"id": "t-2", "project": PROJECT}),
                           ("message.send", {"id": "m-2", "thread": THREAD, "text": "x"}),
                           ("project.create", _create(adapter))):
        assert (await _error(adapter, method, params)).code == -32602, method
    phone = await connect(adapter, adapter.keys)
    try:
        answer, _ = await phone.hello()
        assert answer["result"]["projects"] == []
    finally:
        await phone.close()


# ── the folder browser ────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_folder_browser_stays_inside_the_projects_root(adapter, tmp_path):
    root = adapter.root
    (root / "Api").mkdir()
    (root / "web" / "src").mkdir()
    (root / ".cache").mkdir()
    (root / "notes.txt").write_text("x")
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    os.symlink(outside, root / "out")
    os.symlink(root / "web", root / "site")

    listing = await _call(adapter, "fs.list", {})
    assert listing["root"] == listing["path"] == str(root) and listing["parent"] is None
    assert [folder["name"] for folder in listing["folders"]] == ["Api", "site", "web"] and listing["more"] is False
    nested = await _call(adapter, "fs.list", {"path": str(root / "web")})
    assert nested["parent"] == str(root) and nested["folders"] == [{"name": "src", "path": str(root / "web" / "src")}]
    for path in (str(outside), "web", str(root / "out"), str(root / "web" / ".." / ".."), str(root / "notes.txt"),
                 str(root / "missing")):
        assert (await _error(adapter, "fs.list", {"path": path})).code == -32602, path

    made = await _call(adapter, "fs.mkdir", {"path": str(root / "new-app")})
    assert made == {"path": str(root / "new-app")} and (root / "new-app").is_dir()
    exists = await _error(adapter, "fs.mkdir", {"path": str(root / "new-app")})
    assert exists.code == 4003 and exists.data["reason"] == "exists"
    for path in (str(root / ".hidden"), str(root / ("x" * 256)), str(root / "missing" / "app"),
                 str(root / "out" / "app"), str(outside / "app"), "new-app", str(root) + "/"):
        assert (await _error(adapter, "fs.mkdir", {"path": path})).code == -32602, path
    assert not (outside / "app").exists()


# ── backends ──────────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_backends_list_says_why_a_backend_cannot_run(adapter):
    adapter.projects.backends["broken"] = FakeBackend(RuntimeError("auth status:\n  token=sk-abcdef1234567890abcdef expired"))
    listed = (await _call(adapter, "backends.list", {}))["backends"]
    claude, codex, broken = listed
    assert claude["available"] is True and claude["reason"] is None and claude["default_model"] == "opus"
    assert claude["models"][0] == {"id": "opus", "name": "Opus", "efforts": ["low", "medium", "high"],
                                   "default_effort": "high"}
    assert claude["approvals"] == ["ask", "auto_review"]
    assert codex["available"] is False and codex["reason"] == "not_installed" and codex["message"] == "codex is not on PATH"
    assert broken["available"] is False and broken["reason"] == "error"
    assert "\n" not in broken["message"] and "sk-abcdef1234567890abcdef" not in broken["message"]
    # Asked at most every few minutes, and again at once after a failure to start.
    await _call(adapter, "backends.list", {})
    assert adapter.projects.backends["claude"].probes == 1
    adapter.projects.invalidate("claude")
    await _call(adapter, "backends.list", {})
    assert adapter.projects.backends["claude"].probes == 2


# ── restarts ──────────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_restart_settles_what_was_open_and_starts_what_waited(adapter):
    await _call(adapter, "project.create", _create(adapter))
    await _call(adapter, "thread.open", {"id": THREAD, "project": PROJECT})
    await _call(adapter, "message.send", {"id": "m-1", "thread": THREAD, "text": "Run the tests"})
    backend = adapter.projects.backends["claude"]
    [session] = backend.sessions
    session.emit(Item("toolu_1", "tool", fields={"tool": "execute", "title": "npm test", "input": "npm test"}))
    session.emit(Delta("toolu_1", output="1 passing\n2 fail"))
    session.emit(Ask("ask_1", "command", item="toolu_1", command="npm test"))
    [request] = await until(lambda: [e.payload for e in adapter.store.events_after(0) if e.type == "approval.request"])
    await _call(adapter, "message.send", {"id": "m-2", "thread": THREAD, "text": "Then fix them",
                                          "delivery": "after_turn"})
    # A thread the agent opened, stopped by the restart before its brief started its first turn.
    brief = "Fix the form\n\nIts submit button does nothing."
    await adapter.projects.open_agent_thread(PROJECT, SIDE, "Fix the form", None, brief)
    latest = adapter.store.latest()
    await adapter.disconnect()
    assert session.closed
    assert await adapter.connect()
    # What was open settled before anyone connects; the queued message, kept, starts the next turn,
    # resuming the backend's session; the agent's thread starts with its brief.
    def started(thread):
        return next((one for one in backend.sessions[1:] if one.spec.thread == thread and one.sent), None)
    resumed, briefed = await until(lambda: started(THREAD) and started(SIDE) and (started(THREAD), started(SIDE)))
    assert resumed.spec.resume == session.session_id and resumed.sent == ["Then fix them"]
    assert briefed.sent == [brief]
    [side] = [e.payload for e in adapter.store.events_after(latest) if e.type == "turn.started" and e.payload["thread"] == SIDE]
    assert side["cause"] == "agent"
    await until(lambda: any(e.type == "receipt" for e in adapter.store.events_after(latest)))
    events = [e for e in adapter.store.events_after(latest) if e.payload.get("thread") == THREAD]
    assert [event.type for event in events] == ["approval.resolved", "item.completed", "turn.completed",
                                                "message.new", "turn.started", "receipt"]
    assert events[0].payload == {"project": PROJECT, "thread": THREAD, "request_id": request["request_id"],
                                 "choice": None, "by": "cancelled"}
    item, turn = events[1].payload, events[2].payload
    assert item["status"] == "failed" and item["exit_code"] is None and item["output_lines"] == 2 and item["finished_at"]
    assert turn["status"] == "failed" and turn["error"] == "The agent restarted" and turn["thread"] == THREAD
    assert events[3].payload["id"] == "m-2" and events[5].payload["read_up_to"] == "m-2"
    [entry] = adapter.store.projects_snapshot()
    assert entry["threads"][0]["state"] == "running" and entry["threads"][0]["queued"] == []
