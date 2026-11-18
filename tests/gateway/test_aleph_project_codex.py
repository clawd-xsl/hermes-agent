"""Projects on Aleph, the Codex backend: turns recorded from the real app-server replayed through a
stand-in for it, as the neutral items the runtime logs; approvals, questions and a plan as the
person's cards and back; steering, a steer that missed its turn; what ``backends.list`` says; and an
app-server that dies."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest
import pytest_asyncio

from plugins.platforms.aleph.projects.backend import (
    Accepted, Approve, Ask, Call, Delta, Deny, Exit, Item, McpServer, Patch, SessionSpec, TurnEnd)
from plugins.platforms.aleph.projects.codex import CodexBackend
from tests.gateway.aleph_backend import project_adapter, until

FIXTURES = Path(__file__).parent / "fixtures" / "codex"
PROJECT = "C41A0000-0000-4000-8000-000000000001"
THREAD = "8D0F0000-0000-4000-8000-000000000002"


class FakeCodex:
    """The stand-in, first on PATH: ``play(**scenario)`` sets what it does; ``log`` is what it got."""

    def __init__(self, root: Path) -> None:
        self.scenario, self.log_path = root / "scenario.json", root / "codex.log"
        self.play()

    def play(self, **scenario) -> None:
        self.scenario.write_text(json.dumps(scenario))

    @property
    def log(self) -> list[dict]:
        return [json.loads(line) for line in self.log_path.read_text().splitlines()] if self.log_path.exists() else []

    def sent(self, method: str) -> list[dict]:
        return [message["params"] for message in self.log if message.get("method") == method]

    def answers(self) -> list[dict]:
        return [message["result"] for message in self.log if "result" in message]


@pytest.fixture
def codex(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    shim = bin_dir / "codex"
    shim.write_text(f'#!/bin/sh\nexec "{sys.executable}" -I "{FIXTURES / "fake_app_server.py"}" "$@"\n')
    shim.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    fake = FakeCodex(tmp_path)
    monkeypatch.setenv("FAKE_CODEX_SCENARIO", str(fake.scenario))
    monkeypatch.setenv("FAKE_CODEX_LOG", str(fake.log_path))
    return fake


@pytest_asyncio.fixture
async def backend(tmp_path, codex):
    backend = CodexBackend(tmp_path / "home")
    yield backend
    await backend.stop()


def _spec(folder: Path, **fields) -> SessionSpec:
    state = folder.parent / "state"
    return SessionSpec(**{"thread": THREAD, "project": PROJECT, "folder": str(folder), "top_level": False,
                          "model": None, "effort": None, "approval": "ask", "plan": False,
                          "instructions": "Work in small steps.",
                          "mcp": McpServer("aleph", sys.executable, ("-I", "child.py", "pipe", "bus.sock"),
                                           {"ALEPH_BUS_TOKEN": "token-1"}, ("report",)),
                          "memory": str(state / "memory"), "stop_report": str(state / f"{THREAD}.stop.json"),
                          **fields})


async def _open(backend, tmp_path, **fields):
    """A session of ``backend`` on a folder ``web``, and the events it emits."""
    folder = tmp_path / "web"
    folder.mkdir(exist_ok=True)
    events: list = []
    session = await backend.open(_spec(folder, **fields), events.append)
    return session, events


async def _turn_end(events, count=1):
    return await until(lambda: len([e for e in events if isinstance(e, TurnEnd)]) >= count and events)


# ── the items of a recorded turn ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_recorded_turn_reaches_the_runtime_as_neutral_items(tmp_path, codex, backend):
    codex.play(turns=["turn.json", "edits.json"])
    await backend.probe()
    session, events = await _open(backend, tmp_path)
    await session.send("k1", "List the files, then write hello.txt")
    events = await _turn_end(events)

    assert events[0] == Accepted("k1")
    message = "msg_0eb81d86fb3655c2016afd31fa5b9c87d1963375ef337bfb22"
    assert events[1] == Item(message, "message")
    assert [e.text for e in events if isinstance(e, Delta) and e.key == message][-1] == (
        "I’ll follow the requested sequence exactly.")
    items = {(e.key, e.status): e for e in events if isinstance(e, Item)}
    assert items[message, "completed"].fields == {"text": "I’ll follow the requested sequence exactly."}
    plan = next(e for e in events if isinstance(e, Item) and e.kind == "plan")
    assert plan.fields == {"steps": [{"text": "List files", "status": "in_progress"},
                                     {"text": "Create hello.txt", "status": "pending"}]}
    # The command as the model wrote it, its output (here only on its end) before it completes.
    command = "exec-385be039-72e0-4346-a342-e122623e49c6"
    assert items[command, "in_progress"].fields == {"tool": "execute", "title": "ls", "input": "ls"}
    output = events.index(Delta(command, output="notes.txt\n"))
    assert events.index(items[command, "completed"]) == output + 1
    assert items[command, "completed"].fields["exit_code"] == 0
    edit = items["exec-7827b41a-d869-44cd-b4ab-90a91455cdb8", "in_progress"]
    assert edit.fields["tool"] == "edit" and edit.fields["title"] == "hello.txt"
    assert edit.patches == (Patch("hello.txt", "@@ -0,0 +1,1 @@\n+hi", "added"),)
    tool = items["exec-fd0756d1-f190-48ac-8cff-5b9b937c3018", "in_progress"]
    assert tool.call == Call("aleph", "list_threads", {}) and tool.fields["tool"] == "other"
    # Reasoning with no summary, and the person's own input, are not items.
    assert {e.kind for e in events if isinstance(e, Item)} == {"message", "plan", "tool"}
    assert events[-1] == TurnEnd("completed", None, {"used": 20025, "window": 258400})

    # Output that streams comes once; an edit deleting one file and changing another.
    mark = len(events)
    await session.send("k2", "Change alpha to beta, delete hello.txt")
    events = (await _turn_end(events, 2))[mark:]
    sed = "exec-648b3ee0-25a5-4bbf-a61c-b2299c99a28b"
    assert [e.output for e in events if isinstance(e, Delta) and e.key == sed] == ["alpha\n", "hello.txt-present\n"]
    patch = next(e for e in events if isinstance(e, Item) and e.patches)
    assert patch.fields["tool"] == "edit" and patch.patches == (
        Patch("hello.txt", "@@ -1,1 +0,0 @@\n-hi", "deleted"), Patch("notes.txt", "@@ -1 +1 @@\n-alpha\n+beta\n"))

    # The thread runs with the session's coordination; each turn with its settings.
    [start] = codex.sent("thread/start")
    assert {key: start[key] for key in ("cwd", "approvalPolicy", "approvalsReviewer", "sandbox", "developerInstructions")} == {
        "cwd": str(tmp_path / "web"), "approvalPolicy": "on-request", "approvalsReviewer": "user",
        "sandbox": "workspace-write", "developerInstructions": "Work in small steps."}
    assert start["config"]["mcp_servers"]["aleph"] == {
        "command": sys.executable, "args": ["-I", "child.py", "pipe", "bus.sock"], "env": {"ALEPH_BUS_TOKEN": "token-1"},
        "tools": {"report": {"approval_mode": "approve"}}}
    turn = codex.sent("turn/start")[0]
    assert (turn["model"], turn["effort"]) == ("gpt-5.6-sol", "low")  # backends.list's defaults
    assert turn["collaborationMode"]["mode"] == "default"
    assert turn["sandboxPolicy"]["writableRoots"] == [str(tmp_path / "state" / "memory")]
    assert codex.log[0]["env"] == {"HOME": str(tmp_path / "home"), "CODEX_HOME": str(tmp_path / "home" / ".codex")}


@pytest.mark.asyncio
async def test_settings_apply_from_the_next_turn(tmp_path, codex, backend):
    session, events = await _open(backend, tmp_path)
    await session.apply(_spec(tmp_path / "web", model="gpt-5.6-terra", effort="high", approval="allow_all", plan=True))
    await session.send("k1", "Plan it")
    await _turn_end(events)
    turn = codex.sent("turn/start")[0]
    assert (turn["model"], turn["effort"], turn["approvalPolicy"], turn["sandboxPolicy"]) == (
        "gpt-5.6-terra", "high", "never", {"type": "dangerFullAccess"})
    assert turn["collaborationMode"] == {"mode": "plan", "settings": {
        "model": "gpt-5.6-terra", "reasoning_effort": "high", "developer_instructions": None}}


# ── requests ──────────────────────────────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def adapter(tmp_path, codex):
    async with project_adapter(tmp_path) as instance:
        instance.project_backends["codex"] = instance.codex = CodexBackend(tmp_path / "home")
        try:
            await instance.handle_call("project.create", {
                "id": PROJECT, "name": "web", "folder": str(instance.root / "web"), "clone": None,
                "backend": "codex", "model": None, "effort": None, "approval": "ask"})
            await instance.handle_call("thread.open", {"id": THREAD, "project": PROJECT})
            yield instance
        finally:
            await instance.codex.stop()


async def _logged(adapter, event_type, after=0, count=1):
    events = await until(lambda: len(found := [e for e in adapter.store.events_after(after)
                                               if e.type == event_type]) >= count and found, timeout=10)
    return events[-1].payload


def _recorded(name: str) -> list[dict]:
    return json.loads((FIXTURES / name).read_text())


def _offering(frames: list[dict], decisions: list) -> list[dict]:
    return [{**frame, "params": {**frame["params"], "availableDecisions": decisions}}
            if frame.get("request") == "item/commandExecution/requestApproval" else frame for frame in frames]


async def _send(adapter, message_id, text):
    await adapter.handle_call("message.send", {"id": message_id, "thread": THREAD, "text": text})


@pytest.mark.asyncio
async def test_a_command_approval_round_trip_and_a_steer_into_the_turn(adapter, codex):
    # As Codex asks for a command it lets the person allow for the session.
    codex.play(turns=[_offering(_recorded("approval.json"), ["accept", "acceptForSession", "decline"])])
    await _send(adapter, "m-1", "Fetch example.com")
    request = await _logged(adapter, "approval.request")
    item = await _logged(adapter, "item.started")  # the command (its reasoning had no summary)
    assert (request["kind"], request["command"], request["item"], request["text"]) == (
        "command", "curl -sI https://example.com", item["id"],
        "May I run the requested network command with escalated permissions?")
    await adapter.handle_call("approval.respond", {"request_id": request["request_id"], "choice": "session"})
    assert await until(lambda: codex.answers()) == [{"decision": "acceptForSession"}]

    # The turn takes the person's steer: its receipt, and the turn goes on to its end.
    mark = adapter.store.latest()
    await _send(adapter, "m-2", "Instead, just reply with the word skipped.")
    assert (await _logged(adapter, "receipt", mark))["read_up_to"] == "m-2"
    assert codex.sent("turn/steer")[0]["input"][0]["text"].endswith("Instead, just reply with the word skipped.")
    completed = await _logged(adapter, "turn.completed", mark)
    assert completed["status"] == "completed"
    reply = [e.payload for e in adapter.store.events_after(mark) if e.type == "item.completed"][-1]
    assert (reply["kind"], reply["text"]) == ("message", "skipped")


@pytest.mark.asyncio
async def test_a_typed_message_declines_the_approval_then_steers_the_turn(adapter, codex):
    codex.play(turns=["approval.json"])
    await _send(adapter, "m-1", "Fetch example.com")
    await _logged(adapter, "approval.request")
    mark = adapter.store.latest()
    await _send(adapter, "m-2", "Instead, just reply with the word skipped.")
    resolved = await _logged(adapter, "approval.resolved", mark)
    assert (resolved["choice"], resolved["by"]) == ("deny", "user")
    await _logged(adapter, "turn.completed", mark)
    assert codex.answers() == [{"decision": "decline"}]
    assert codex.sent("turn/steer")[0]["input"][0]["text"].endswith("Instead, just reply with the word skipped.")
    command = next(e.payload for e in adapter.store.events_after(0) if e.type == "item.completed"
                   and e.payload.get("tool") == "execute")
    assert command["status"] == "declined"


@pytest.mark.asyncio
async def test_a_plan_mode_turn_asks_its_question_and_ends_with_its_plan(adapter, codex):
    codex.play(turns=["plan.json", []])
    await adapter.handle_call("thread.update", {"id": THREAD, "plan": True})
    await _send(adapter, "m-1", "Plan a README")
    question = await _logged(adapter, "clarify.request")
    assert (question["question"], question["choices"], question["recommended"]) == (
        "Which tone should the README use?", ["Formal", "Casual"], 0)
    await adapter.handle_call("clarify.respond", {"clarify_id": question["clarify_id"], "choices": [1],
                                                  "message_id": "a-1"})
    assert await until(lambda: codex.answers()) == [{"answers": {"tone": {"answers": ["Casual"]}}}]

    # The proposed plan is the turn's reply; the server asks to start on it.
    plan = await _logged(adapter, "approval.request", count=1)
    assert plan["kind"] == "plan"
    reply = [e.payload for e in adapter.store.events_after(0) if e.type == "item.completed"][-1]
    assert reply["kind"] == "message" and reply["text"].startswith("Create a casual `README.md`")
    assert codex.sent("turn/start")[0]["collaborationMode"]["mode"] == "plan"
    await adapter.handle_call("approval.respond", {"request_id": plan["request_id"], "choice": "once"})
    await until(lambda: len(codex.sent("turn/start")) == 2)
    follow = codex.sent("turn/start")[1]
    assert follow["input"][0]["text"] == "Implement the plan." and follow["collaborationMode"]["mode"] == "default"


@pytest.mark.asyncio
async def test_a_session_grant_codex_does_not_offer_allows_once(tmp_path, codex, backend):
    codex.play(turns=["approval.json"])  # recorded: escalating a command offers no session grant
    session, events = await _open(backend, tmp_path)
    await session.send("k1", "Fetch example.com")
    ask = await until(lambda: next((e for e in events if isinstance(e, Ask)), None))
    assert (ask.command, ask.grant) == ("curl -sI https://example.com", None)
    await session.answer(ask.key, Approve(session=True))
    assert await until(lambda: codex.answers()) == [{"decision": "accept"}]


@pytest.mark.asyncio
async def test_an_mcp_tool_approval_and_other_elicitations(tmp_path, codex, backend):
    elicit = {"threadId": "$THREAD", "turnId": "$TURN", "serverName": "notion", "mode": "url",
              "_meta": None, "message": "Sign in", "url": "https://example.com", "elicitationId": "e-1"}
    recorded = _recorded("tool_approval.json")
    at = next(i for i, frame in enumerate(recorded) if "request" in frame)
    codex.play(turns=[[*recorded[:at], {"request": "mcpServer/elicitation/request", "params": elicit},
                       *recorded[at:]]])
    session, events = await _open(backend, tmp_path)
    await session.send("k1", "Call the danger tool")
    ask = await until(lambda: next((e for e in events if isinstance(e, Ask)), None))
    # A real elicitation is declined without asking; Codex's approval of a tool call is an ask.
    assert codex.answers() == [{"action": "decline", "content": None, "_meta": None}]
    assert (ask.kind, ask.item, ask.text, ask.grant) == (
        "tool", "exec-0f37ddb6-9f8a-4114-8acb-df0b9a4c9698", "danger", "aleph/danger")
    await session.answer(ask.key, Approve(session=True))
    await _turn_end(events)
    assert codex.answers()[1] == {"action": "accept", "content": {}, "_meta": {"persist": "session"}}


@pytest.mark.asyncio
async def test_a_subagent_shows_as_its_call_and_its_approvals_are_its_parents(tmp_path, codex, backend):
    def child_approval(thread):
        return {"request": "item/commandExecution/requestApproval", "params": {
            "threadId": thread, "turnId": "t-child", "itemId": "exec-child", "startedAtMs": 0, "environmentId": "local",
            "reason": None, "command": "/bin/zsh -lc 'npm test'", "cwd": "$CWD"}}
    collab = {"type": "collabAgentToolCall", "id": "call-agent", "tool": "spawnAgent", "status": "inProgress",
              "senderThreadId": "$THREAD", "receiverThreadIds": ["child-1"], "prompt": "Run the tests\nand report",
              "model": None, "reasoningEffort": None, "agentsStates": {}}
    codex.play(parents={"child-2": "child-1"}, turns=[[
        {"notify": "turn/started", "params": {"threadId": "$THREAD", "turn": {"id": "$TURN"}}},
        {"notify": "item/started", "params": {"threadId": "$THREAD", "turnId": "$TURN", "item": collab}},
        {"notify": "item/started", "params": {"threadId": "child-1", "turnId": "t-child", "item": {
            "type": "agentMessage", "id": "msg-child", "text": "", "phase": None}}},
        child_approval("child-1"), child_approval("child-2"),
        {"notify": "item/started", "params": {"threadId": "$THREAD", "turnId": "$TURN", "item": {
            "type": "reasoning", "id": "rs-1", "summary": [], "content": []}}},
        {"notify": "item/reasoning/summaryTextDelta", "params": {
            "threadId": "$THREAD", "turnId": "$TURN", "itemId": "rs-1", "delta": "Checking", "summaryIndex": 0}},
        {"notify": "item/completed", "params": {"threadId": "$THREAD", "turnId": "$TURN", "item": {
            "type": "reasoning", "id": "rs-1", "summary": ["Checking"], "content": []}}},
        {"notify": "item/completed", "params": {"threadId": "$THREAD", "turnId": "$TURN", "item": {
            **collab, "status": "completed"}}},
        {"notify": "model/rerouted", "params": {"threadId": "$THREAD", "turnId": "$TURN", "fromModel": "gpt-5.6-sol",
                                                "toModel": "gpt-5.6-luna", "reason": "highRiskCyberActivity"}},
        {"notify": "item/started", "params": {"threadId": "$THREAD", "turnId": "$TURN", "item": {
            "type": "contextCompaction", "id": "cmp-1"}}},
        {"notify": "item/completed", "params": {"threadId": "$THREAD", "turnId": "$TURN", "item": {
            "type": "contextCompaction", "id": "cmp-1"}}},
        {"notify": "turn/completed", "params": {"threadId": "$THREAD", "turn": {
            "id": "$TURN", "status": "failed", "error": {"message": "The model is overloaded."}}}}]])
    session, events = await _open(backend, tmp_path)
    await session.send("k1", "Test it")
    first = await until(lambda: next((e for e in events if isinstance(e, Ask)), None))
    assert (first.kind, first.item, first.command, first.grant) == ("command", None, "npm test", "/bin/zsh -lc 'npm test'")
    await session.answer(first.key, Deny())
    second = await until(lambda: next((e for e in events if isinstance(e, Ask) and e is not first), None))
    await session.answer(second.key, Deny())
    events = await _turn_end(events)
    items = [e for e in events if isinstance(e, Item)]
    assert items[0] == Item("call-agent", "tool", fields={"tool": "think", "title": "Run the tests",
                                                          "input": "Run the tests\nand report"})
    assert "msg-child" not in {e.key for e in items}
    assert Item("rs-1", "reasoning") in items and Item("rs-1", "reasoning", "completed", {"text": "Checking"}) in items
    assert Item("model-1", "model", "completed", {"model": "gpt-5.6-luna"}) in items
    assert [e.kind for e in items[-2:]] == ["compaction", "compaction"]
    assert events[-1] == TurnEnd("failed", "The model is overloaded.", None)
    assert codex.answers() == [{"decision": "decline"}, {"decision": "decline"}]


# ── steering ──────────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_steer_that_missed_its_turn_starts_the_next(tmp_path, codex, backend):
    codex.play(turns=[[{"notify": "turn/started", "params": {"threadId": "$THREAD", "turn": {"id": "$TURN"}}},
                       {"end": "completed"}]])
    session, events = await _open(backend, tmp_path)
    await session.send("k1", "Start")
    await session.send("k2", "And this")
    events = await _turn_end(events, 2)
    assert events == [Accepted("k1"), TurnEnd("completed"), Accepted("k2"), TurnEnd("completed")]
    assert [turn["input"][0]["text"] for turn in codex.sent("turn/start")] == ["Start", "And this"]


# ── backends.list ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_what_the_probe_says(tmp_path, codex, monkeypatch):
    backend = CodexBackend(tmp_path / "home")
    try:
        codex.play(config={"model": "gpt-5.6-terra"})
        info = await backend.probe()
        assert info.available and info.approvals == ("ask", "auto_review", "allow_all")
        assert [model.id for model in info.models] == ["gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"]
        assert info.model(None).id == "gpt-5.6-terra"
        assert info.models[0].efforts == ("low", "medium", "high", "xhigh", "max", "ultra")
        assert info.models[0].default_effort == "low"
        await backend.stop()

        codex.play(requirements={"allowedApprovalsReviewers": ["user"], "allowedApprovalPolicies": ["on-request"]})
        assert (await backend.probe()).approvals == ("ask",)
        await backend.stop()

        codex.play(account=None)
        info = await backend.probe()
        assert (info.available, info.reason) == (False, "signed_out")
        await backend.stop()

        monkeypatch.setenv("PATH", str(tmp_path / "empty"))
        info = await backend.probe()
        assert (info.available, info.reason, info.message) == (False, "not_installed", "codex is not on PATH")
    finally:
        await backend.stop()


# ── the app-server's end ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_every_session_exits_with_the_app_server_and_the_next_use_starts_another(tmp_path, codex, backend):
    codex.play(turns=[[{"exit": 3}]])
    first, first_events = await _open(backend, tmp_path)
    second, second_events = await _open(backend, tmp_path, thread="other")
    await first.send("k1", "Go")
    error = "Codex's app-server exited (code 3)."
    await until(lambda: Exit(error) in first_events and Exit(error) in second_events)
    with pytest.raises(ConnectionError):
        await second.send("k2", "Go on")

    codex.play()
    third, events = await _open(backend, tmp_path, resume=first.session_id)
    assert third.session_id == first.session_id  # resumed on the new process
    await third.send("k3", "Again")
    await _turn_end(events)
    assert len([entry for entry in codex.log if "env" in entry]) == 2


@pytest.mark.asyncio
async def test_a_thread_codex_lost_starts_anew_and_close_unloads_it(tmp_path, codex, backend):
    codex.play(lost=["thread-gone"])
    session, events = await _open(backend, tmp_path, resume="thread-gone")
    assert session.session_id not in (None, "thread-gone")
    await session.close()
    assert events == [Exit()]
    assert codex.sent("thread/unsubscribe") == [{"threadId": session.session_id}]


@pytest.mark.asyncio
async def test_a_session_closes_even_when_the_app_server_does_not_answer(tmp_path, codex, backend, monkeypatch):
    codex.play()
    session, events = await _open(backend, tmp_path)

    async def wedged(method, params, **kwargs):
        raise TimeoutError
    monkeypatch.setattr(session.server, "request", wedged)
    await session.close()
    assert events == [Exit()]


@pytest.mark.asyncio
async def test_a_subagents_edit_approval_shows_its_files(tmp_path, codex, backend):
    collab = {"type": "collabAgentToolCall", "id": "call-agent", "tool": "spawnAgent", "status": "inProgress",
              "senderThreadId": "$THREAD", "receiverThreadIds": ["child-1"], "prompt": "Fix the typo",
              "model": None, "reasoningEffort": None, "agentsStates": {}}
    change = {"type": "fileChange", "id": "edit-child", "status": "inProgress", "changes": [
        {"path": "$CWD/README.md", "kind": {"type": "update", "move_path": None},
         "diff": "@@ -1 +1 @@\n-teh\n+the\n"}]}
    codex.play(turns=[[
        {"notify": "turn/started", "params": {"threadId": "$THREAD", "turn": {"id": "$TURN"}}},
        {"notify": "item/started", "params": {"threadId": "$THREAD", "turnId": "$TURN", "item": collab}},
        {"notify": "item/started", "params": {"threadId": "child-1", "turnId": "t-child", "item": change}},
        {"request": "item/fileChange/requestApproval", "params": {
            "threadId": "child-1", "turnId": "t-child", "itemId": "edit-child", "reason": None}}]])
    session, events = await _open(backend, tmp_path)
    await session.send("k1", "Fix it")
    ask = await until(lambda: next((e for e in events if isinstance(e, Ask)), None))
    assert (ask.kind, ask.item, ask.files) == ("edit", None, ({"path": "README.md", "added": 1, "removed": 1},))
    assert "edit-child" not in {e.key for e in events if isinstance(e, Item)}
