"""Projects on Aleph: projects and threads idempotent by id, their events in the one log (replayed,
listed on a reset, dropped with the project), the folder browser bounded to the projects root, the
backends on offer, and what a restart settles."""

from __future__ import annotations

import asyncio
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
    return {"id": PROJECT, "name": "web", "folder": str(adapter.root / "web"), "backend": "claude",
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
async def test_a_project_comes_with_its_top_level_and_is_created_once_by_its_id(adapter):
    phone = await connect(adapter, adapter.keys)
    try:
        await phone.hello()
        created = await phone.call("project.create", _create(adapter))
        event, top = (await phone.logged())["params"], (await phone.logged())["params"]
        assert event["type"] == "project.new" and created["result"] == {"seq": event["seq"]}
        assert event["payload"] == {"id": PROJECT, "name": "web", "folder": str(adapter.root / "web"), "host": "local",
                                    "backend": "claude", "created_at": event["payload"]["created_at"]}
        # The top level is the project's own thread, and carries the settings it was created with.
        assert top["type"] == "thread.new" and top["payload"]["id"] == PROJECT and top["payload"]["name"] is None
        assert (top["payload"]["approval"], top["payload"]["model"], top["payload"]["tier"]) == ("auto_review", None, None)
        again = await phone.call("project.create", _create(adapter, name="other"))
        assert again["result"] == created["result"] and adapter.store.latest() == top["seq"]
    finally:
        await phone.close()
    # Several projects may share one folder: each is a conversation of its own, its rules its own.
    shared = await _call(adapter, "project.create", _create(adapter, id="P-2", name=None))
    assert adapter.store.events_after(shared["seq"] - 1)[0].payload["folder"] == str(adapter.root / "web")
    # An id names one thing for good: a chat's is no project's.
    await _call(adapter, "chat.create", {"id": SIDE})
    assert (await _error(adapter, "project.create", _create(adapter, id=SIDE, folder=None))).code == -32602


@pytest.mark.asyncio
async def test_a_project_takes_any_folder_it_can_write_and_only_what_its_backend_offers(adapter, tmp_path):
    for bad in (_create(adapter, backend="gemini"), _create(adapter, model="gpt"),
                _create(adapter, model="sonnet", effort="high"), _create(adapter, model="sonnet", tier="fast"),
                _create(adapter, tier="turbo"), _create(adapter, approval="allow_all"), _create(adapter, approval=None),
                _create(adapter, folder="web"), _create(adapter, folder=str(adapter.root / "missing")),
                _create(adapter, host="devbox")):
        assert (await _error(adapter, "project.create", bad)).code == -32602, bad
    error = await _error(adapter, "project.create", _create(adapter, backend="codex"))
    assert error.code == 4003 and error.data == {"reason": "unavailable", "message": "codex is not on PATH"}
    locked = tmp_path / "locked"
    locked.mkdir(mode=0o500)
    error = await _error(adapter, "project.create", _create(adapter, folder=str(locked)))
    assert error.code == 4003 and error.data["reason"] == "access"
    assert adapter.store.latest() == 0
    # Outside the projects root; with no name given, unnamed until its first message.
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    await _call(adapter, "project.create", _create(adapter, name=None, folder=str(elsewhere), model="opus",
                                                   effort="low", tier="fast"))
    assert adapter.store.events_after(0)[0].payload["name"] is None


@pytest.mark.asyncio
async def test_a_project_with_no_folder_works_in_its_own_folder_for_good(adapter):
    await _call(adapter, "project.create", _create(adapter, name=None, folder=None))
    working = adapter.root / ".aleph" / PROJECT
    assert working.is_dir() and adapter.store.events_after(0)[0].payload["folder"] is None
    # Unnamed until the person's first message names it.
    await _call(adapter, "message.send", {"id": "m-1", "thread": PROJECT, "text": "What does   this script do?"})
    [renamed] = [e.payload for e in adapter.store.events_after(0) if e.type == "project.update"]
    assert renamed["name"] == "What does this script do?" and renamed["folder"] is None
    [session] = adapter.projects.backends["claude"].sessions
    assert session.spec.folder == str(working)
    await _call(adapter, "message.send", {"id": "m-2", "thread": PROJECT, "text": "And this one?"})
    assert len([e for e in adapter.store.events_after(0) if e.type == "project.update"]) == 1
    # It has no changes to show, and its folder goes with it.
    assert (await _error(adapter, "changes.get", {"project": PROJECT, "scope": "uncommitted"})).code == -32602
    await _call(adapter, "project.delete", {"id": PROJECT})
    assert not working.exists()


@pytest.mark.asyncio
async def test_each_thread_has_its_own_settings_and_an_update_that_changes_nothing_logs_nothing(adapter):
    await _call(adapter, "project.create", _create(adapter))
    await adapter.projects.open_agent_thread(PROJECT, THREAD, "fix-login", None, None)
    latest = adapter.store.latest()
    assert await _call(adapter, "project.update", {"id": PROJECT, "name": "web"}) == {"seq": None}
    assert await _call(adapter, "thread.update", {"id": THREAD, "approval": "auto_review", "plan": False}) == {"seq": None}
    assert adapter.store.latest() == latest

    assert await _call(adapter, "project.update", {"id": PROJECT, "name": "Web app"}) == {"seq": latest + 1}
    assert adapter.store.events_after(latest)[0].payload["folder"]
    # The top level's settings are its own: any but its name, which is the project's.
    changed = await _call(adapter, "thread.update", {"id": PROJECT, "model": "opus", "effort": "low", "tier": "fast",
                                                     "approval": "ask", "plan": True})
    assert changed == {"seq": latest + 2}
    top = adapter.store.events_after(latest + 1)[0].payload
    assert (top["model"], top["effort"], top["tier"], top["approval"], top["plan"]) == ("opus", "low", "fast", "ask", True)
    assert (await _error(adapter, "thread.update", {"id": PROJECT, "name": "x"})).code == -32602
    # Another thread's stay as they were; what the backend does not list is refused.
    assert (await asyncio.to_thread(adapter.store.projects.thread, THREAD)).payload["approval"] == "auto_review"
    for bad in ({"model": "sonnet", "effort": "low"}, {"tier": "fast", "model": "sonnet"}, {"approval": "allow_all"},
                {"approval": None}):
        assert (await _error(adapter, "thread.update", {"id": THREAD, **bad})).code == -32602, bad
    assert (await _call(adapter, "thread.update", {"id": THREAD, "model": "sonnet"}))["seq"] == latest + 3
    assert (await _error(adapter, "thread.close", {"id": PROJECT})).code == -32602
    assert (await _error(adapter, "project.update", {"id": PROJECT, "model": "opus"})).code == -32602


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
    await adapter.projects.open_agent_thread(PROJECT, THREAD, "fix-login", None, None)
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
        assert [kind for kind, _ in logged] == ["project.new", "thread.new", "thread.new", "message.new", "turn.started",
                                                "receipt", "turn.completed", "thread.close"]
        message = replay[3]["payload"]
        assert message["project"] == PROJECT and message["thread"] == THREAD and "chat" not in message
        assert message["comments"][0]["line"] == 42 and sent == {"seq": logged[3][1]}
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
    await adapter.projects.open_agent_thread(PROJECT, THREAD, "fix-login", None, None)
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
        top, thread = entry["threads"]  # the top level first
        assert top["id"] == PROJECT and top["state"] == "waiting" and top["approval"] == "auto_review"
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
    await adapter.projects.open_agent_thread(PROJECT, THREAD, "fix-login", None, None)
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
    for method, params in (("thread.update", {"id": THREAD, "plan": True}),
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
async def test_the_folder_browser_covers_the_whole_machine(adapter, tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / "Api").mkdir(parents=True)
    (home / "web" / "src").mkdir(parents=True)
    (home / ".cache").mkdir()
    (home / "notes.txt").write_text("x")
    os.symlink(home / "web", home / "site")
    monkeypatch.setenv("HOME", str(home))

    listing = await _call(adapter, "fs.list", {})
    assert listing["path"] == str(home) and listing["parent"] == str(tmp_path) and "root" not in listing
    assert [folder["name"] for folder in listing["folders"]] == ["Api", "site", "web"] and listing["more"] is False
    nested = await _call(adapter, "fs.list", {"host": "local", "path": str(home / "web")})
    assert nested["parent"] == str(home) and nested["folders"] == [{"name": "src", "path": str(home / "web" / "src")}]
    assert (await _call(adapter, "fs.list", {"path": "/"}))["parent"] is None
    for params in ({"path": "web"}, {"path": str(home / "notes.txt")}, {"path": str(home / "missing")},
                   {"host": "devbox"}):
        assert (await _error(adapter, "fs.list", params)).code == -32602, params
    locked = home / "locked"
    locked.mkdir(mode=0o000)
    try:
        error = await _error(adapter, "fs.list", {"path": str(locked)})
        assert error.code == 4003 and error.data["reason"] == "access"
    finally:
        locked.chmod(0o700)
    assert (await _error(adapter, "fs.mkdir", {"path": str(home / "new")})).code == -32601


# ── backends ──────────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_backends_list_says_why_a_backend_cannot_run(adapter):
    adapter.projects.backends["broken"] = FakeBackend(RuntimeError("auth status:\n  token=sk-abcdef1234567890abcdef expired"))
    listed = (await _call(adapter, "backends.list", {}))["backends"]
    claude, codex, broken = listed
    assert claude["available"] is True and claude["reason"] is None and claude["default_model"] == "opus"
    assert claude["models"] == [
        {"id": "opus", "name": "Opus", "efforts": ["low", "medium", "high"], "default_effort": "high",
         "tiers": ["standard", "fast"], "default_tier": "standard"},
        {"id": "sonnet", "name": "Sonnet", "efforts": [], "default_effort": None, "tiers": [], "default_tier": None}]
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
    await adapter.projects.open_agent_thread(PROJECT, THREAD, "fix-login", None, None)
    await _call(adapter, "message.send", {"id": "m-1", "thread": THREAD, "text": "Run the tests"})
    backend = adapter.projects.backends["claude"]
    [session] = backend.sessions
    session.emit(Item("toolu_1", "tool", fields={"tool": "execute", "title": "npm test", "input": "npm test"}))
    session.emit(Delta("toolu_1", output="1 passing\n2 fail"))
    # Queued before the approval (one sent while it waits would answer it instead).
    await _call(adapter, "message.send", {"id": "m-2", "thread": THREAD, "text": "Then fix them",
                                          "delivery": "after_turn"})
    session.emit(Ask("ask_1", "command", item="toolu_1", command="npm test"))
    [request] = await until(lambda: [e.payload for e in adapter.store.events_after(0) if e.type == "approval.request"])
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
    thread = next(thread for thread in entry["threads"] if thread["id"] == THREAD)
    assert thread["state"] == "running" and thread["queued"] == []
