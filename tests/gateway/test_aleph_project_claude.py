"""Projects on Aleph, the Claude Code backend: frames recorded from the real CLI, replayed by a fake
``claude`` first on PATH (``fixtures/claude``), become the protocol's items, requests and turn ends;
the probe says what the machine runs; the process lives and dies as a thread's session."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Optional

import pytest

from plugins.platforms.aleph.projects.backend import (
    Accepted, Answers, Approve, Ask, Call, Delta, Deny, Exit, Item, McpServer, Question, SessionSpec, TurnEnd)
from plugins.platforms.aleph.projects.claude import ClaudeBackend, ClaudeSession
from tests.gateway.aleph_backend import until

FIXTURES = Path(__file__).parent / "fixtures" / "claude"
TURN_SESSION = "496417e2-84f6-4659-9d33-75e4bc3f8c86"
HELLO_SESSION = "e3706565-d704-477c-955b-964bf9bed848"


def script(name: str) -> list[dict[str, Any]]:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


class Fake:
    """The fake CLI's scenario and what it was given."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def scenario(self, *scripts: list[dict[str, Any]], **fields: Any) -> None:
        models = json.loads((FIXTURES / "models.json").read_text(encoding="utf-8"))
        (self.path / "scenario.json").write_text(json.dumps({"models": models, "scripts": list(scripts), **fields}))

    def log(self) -> list[dict[str, Any]]:
        path = self.path / "log.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def spawns(self) -> list[dict[str, Any]]:
        return [entry for entry in self.log() if "argv" in entry]

    def responses(self) -> dict[str, dict[str, Any]]:
        """The backend's answers to the CLI's requests, by request id."""
        return {entry["in"]["response"]["request_id"]: entry["in"]["response"]["response"]
                for entry in self.log() if entry.get("in", {}).get("type") == "control_response"}

    def controls(self) -> list[dict[str, Any]]:
        return [entry["in"]["request"] for entry in self.log() if entry.get("in", {}).get("type") == "control_request"]


@pytest.fixture
def fake(tmp_path, monkeypatch):
    path = tmp_path / "fake"
    path.mkdir()
    monkeypatch.setenv("FAKE_CLAUDE_DIR", str(path))
    monkeypatch.setenv("PATH", f"{FIXTURES / 'bin'}{os.pathsep}{os.environ['PATH']}")
    return Fake(path)


@pytest.fixture
def folder(tmp_path):
    path = (tmp_path / "web").resolve()
    path.mkdir()
    (path / "notes.md").write_text("a\n")
    return path


def spec(folder: Path, **fields: Any) -> SessionSpec:
    return SessionSpec(**{"thread": "t1", "project": "p1", "folder": str(folder), "top_level": False, "model": None,
                          "effort": None, "approval": "ask", "plan": False, **fields})


class Sink(list):
    def of(self, event_type: type, **match: Any) -> list[Any]:
        return [event for event in self if isinstance(event, event_type)
                and all(getattr(event, key) == value for key, value in match.items())]


async def opened(tmp_path, folder, **fields: Any) -> tuple[ClaudeSession, Sink]:
    events = Sink()
    session = await ClaudeBackend(str(tmp_path / "home")).open(spec(folder, **fields), events.append)
    return session, events


async def turn(session: ClaudeSession, events: Sink, text: str,
               decide: Callable[[Ask], Any] = lambda ask: Approve(), key: str = "k1") -> TurnEnd:
    """Send ``text`` and answer every ask with ``decide`` until the turn ends."""
    answered = 0
    ends = len(events.of(TurnEnd))
    await session.send(key, text)
    while len(events.of(TurnEnd)) == ends:
        await until(lambda: len(events.of(Ask)) > answered or len(events.of(TurnEnd)) > ends, timeout=10)
        for ask in events.of(Ask)[answered:]:
            await session.answer(ask.key, decide(ask))
            answered += 1
    return events.of(TurnEnd)[-1]


def started(events: Sink, kind: str = "tool") -> list[Item]:
    return [event for event in events.of(Item, kind=kind) if event.status == "in_progress"]


def ended(events: Sink, key: str) -> Item:
    return next(event for event in events.of(Item, key=key) if event.status != "in_progress")


# ── a turn ────────────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_recorded_turn_becomes_protocol_items(tmp_path, fake, folder, monkeypatch):
    fake.scenario(script("turn"))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "/elsewhere")
    stop = tmp_path / "state" / "p1" / "t1.stop.json"
    mcp = McpServer("aleph", "/usr/bin/python3", ("-I", "child.py", "pipe", "/tmp/bus.sock"),
                    {"ALEPH_BUS_TOKEN": "secret-token"}, ("list_threads", "report"))
    session, events = await opened(tmp_path, folder, mcp=mcp, instructions="You are a thread.",
                                    memory=str(stop.parent / "memory"), stop_report=str(stop))
    end = await turn(session, events, "Do these steps", lambda ask: Deny() if ask.kind == "command" else Approve())

    # Tools start at their complete call, in protocol shape; ToolSearch is no item.
    tools = started(events)
    assert [(item.fields["tool"], item.fields["title"]) for item in tools] == [
        ("execute", "ls -la"), ("execute", "ls missing-dir"), ("other", "list_threads"), ("edit", "hello.txt"),
        ("edit", "notes.md"), ("execute", "rm notes.md")]
    ls, missing, threads, write, edit, rm = tools
    assert ls.fields["input"] == "ls -la" and json.loads(threads.fields["input"]) == {}
    assert threads.call == Call("aleph", "list_threads", {})
    assert write.arguments == {"file_path": f"{folder}/hello.txt", "content": "hi\n"}
    assert edit.arguments == {"file_path": f"{folder}/notes.md", "old_string": "a", "new_string": "b",
                              "replace_all": False}

    # Claude Code reports no exit status: 0 on success, the code its failure names, else none.
    assert (ended(events, ls.key).status, ended(events, ls.key).fields) == ("completed", {"exit_code": 0})
    assert (ended(events, missing.key).status, ended(events, missing.key).fields) == ("failed", {"exit_code": 2})
    assert [delta.output for delta in events.of(Delta, key=missing.key)] == [
        '"missing-dir": No such file or directory (os error 2)']
    assert ended(events, rm.key).fields == {"exit_code": None} and not events.of(Delta, key=rm.key)
    assert ended(events, write.key).status == "completed"

    # Prose and thinking stream as the whole text so far, and complete with it.
    reply = started(events, "message")[-1]
    texts = [delta.text for delta in events.of(Delta, key=reply.key)]
    final = ended(events, reply.key).fields["text"]
    assert final.startswith("Done") and texts[-1] == final
    assert all(final.startswith(text) for text in texts)
    assert started(events, "reasoning")

    assert events[0] == Accepted("k1")
    assert end.status == "completed" and end.usage["window"] == 1_000_000 and end.usage["used"] > 0
    assert session.session_id == TURN_SESSION

    # What the process ran with: the coordination's MCP server, its token only in the environment.
    [spawn] = fake.spawns()
    argv, env = spawn["argv"], spawn["env"]
    assert "secret-token" not in json.dumps(argv)
    assert env == {"HOME": str(tmp_path / "home"), "ALEPH_BUS_TOKEN": "secret-token",
                   "ALEPH_STOP_REPORT": str(stop), "CLAUDE_CONFIG_DIR": None}
    config = json.loads(argv[argv.index("--mcp-config") + 1])
    assert config == {"mcpServers": {"aleph": {"command": "/usr/bin/python3", "args": list(mcp.args),
                                               "env": {"ALEPH_BUS_TOKEN": "${ALEPH_BUS_TOKEN}"}}}}
    allowed = argv.index("--allowedTools")
    assert argv[allowed + 1:allowed + 3] == ["mcp__aleph__list_threads", "mcp__aleph__report"]
    assert argv[argv.index("--add-dir") + 1] == str(stop.parent / "memory")
    assert argv[argv.index("--append-system-prompt") + 1] == "You are a thread."
    assert [argv[argv.index(flag) + 1] for flag in ("--permission-mode", "--model")] == ["default", "default"]
    assert "--resume" not in argv and "--effort" not in argv
    await session.close()


@pytest.mark.asyncio
async def test_approvals_round_trip_with_session_rules(tmp_path, fake, folder):
    fake.scenario(script("turn"))
    session, events = await opened(tmp_path, folder)
    decisions = {"edit": [Approve(), Approve(session=True)], "command": [Approve(session=True)]}
    await turn(session, events, "Do these steps", lambda ask: decisions[ask.kind].pop(0))

    write, edit, rm = events.of(Ask)
    tools = {item.fields["title"]: item.key for item in started(events)}
    assert (write.kind, write.item, write.files) == ("edit", tools["hello.txt"],
                                                     ({"path": "hello.txt", "added": 1, "removed": 0},))
    assert edit.files == ({"path": "notes.md", "added": 1, "removed": 1},) and edit.grant is None
    assert (rm.kind, rm.item, rm.command, rm.cwd, rm.grant, rm.text) == (
        "command", tools["rm notes.md"], "rm notes.md", str(folder), "rm", "Delete notes.md")

    responses = fake.responses()
    assert responses[write.key] == {"behavior": "allow", "updatedInput": {"file_path": f"{folder}/hello.txt",
                                                                           "content": "hi\n"}}
    # For this session: an edit takes the CLI's own suggestion, a command its first word's prefix rule,
    # each for the session only (never the project's settings file).
    assert responses[edit.key]["updatedPermissions"] == [{"type": "setMode", "mode": "acceptEdits",
                                                          "destination": "session"}]
    assert responses[rm.key]["updatedPermissions"] == [{
        "type": "addRules", "rules": [{"toolName": "Bash", "ruleContent": "rm:*"}], "behavior": "allow",
        "destination": "session"}]
    await session.close()


@pytest.mark.asyncio
async def test_a_refusal_carries_what_the_person_typed_and_a_command_whose_word_runs_anything_grants_nothing(
        tmp_path, fake, folder):
    def asked(request_id: str, command: str) -> list[dict[str, Any]]:
        return [{"out": {"type": "control_request", "request_id": request_id, "request": {
            "subtype": "can_use_tool", "tool_name": "Bash", "input": {"command": command},
            "tool_use_id": f"toolu_{request_id}"}}}, {"await": "control_response"}]
    fake.scenario([{"await": "user"}, {"echo": True}, *asked("req-1", "cd src && rm -rf build"),
                   *asked("req-2", "python3 build.py"),
                   {"out": {"type": "result", "subtype": "success", "is_error": False}}])
    session, events = await opened(tmp_path, folder)
    await turn(session, events, "clean",
               lambda ask: Deny("Use make clean instead") if "rm" in ask.command else Approve(session=True))
    # "cd:*" would cover the rm too, and "python3:*" any program at all: neither is granted.
    assert [ask.grant for ask in events.of(Ask)] == [None, None]
    responses = fake.responses()
    assert responses["req-1"] == {"behavior": "deny", "message": "Use make clean instead"}
    assert responses["req-2"] == {"behavior": "allow", "updatedInput": {"command": "python3 build.py"}}
    await session.close()


@pytest.mark.asyncio
async def test_the_task_tools_fold_into_one_plan(tmp_path, fake, folder):
    fake.scenario(script("plan_tools"))
    session, events = await opened(tmp_path, folder)
    await turn(session, events, "Track it")
    assert [item.fields["steps"] for item in events.of(Item, kind="plan")] == [
        [{"text": "look", "status": "in_progress"}, {"text": "write", "status": "pending"}],
        [{"text": "look", "status": "completed"}, {"text": "write", "status": "in_progress"}],
        [{"text": "ship it", "status": "pending"}],
        # TaskUpdate names the task by the id its TaskCreate's result gave it.
        [{"text": "ship it", "status": "in_progress"}],
        [{"text": "ship it", "status": "completed"}]]
    assert not events.of(Item, kind="tool")
    await session.close()


# ── requests ──────────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ask_user_question_takes_the_answers_by_question(tmp_path, fake, folder):
    fake.scenario(script("question"))
    session, events = await opened(tmp_path, folder)
    await turn(session, events, "Ask me", lambda ask: Answers((("Red",), ("S",))))
    [ask] = events.of(Ask)
    assert ask.kind == "question" and ask.item is None
    assert ask.questions == (Question("What is your favorite color?", ("Red", "Blue"), False, 0),
                             Question("What is your size?", ("S", "M"), False, None))
    assert not events.of(Item, kind="tool")
    # The CLI gets its own labels back, "(Recommended)" and all.
    assert fake.responses()[ask.key]["updatedInput"]["answers"] == {
        "What is your favorite color?": "Red (Recommended)", "What is your size?": "S"}
    assert ended(events, started(events, "message")[-1].key).fields["text"].startswith("Your answers")
    await session.close()


@pytest.mark.asyncio
async def test_exit_plan_mode_shows_the_plan_and_starting_leaves_plan_mode(tmp_path, fake, folder):
    fake.scenario(script("plan"))
    session, events = await opened(tmp_path, folder, plan=True, approval="auto_review")
    await turn(session, events, "Plan it")
    plan, write = events.of(Ask)
    assert plan.kind == "plan"
    # The plan is a message, completed before the request that holds it.
    shown = events.index(plan) - 1
    assert isinstance(events[shown], Item) and events[shown].kind == "message"
    assert events[shown].status == "completed" and events[shown].fields["text"].startswith("1. Create README.md")
    assert fake.spawns()[0]["argv"][fake.spawns()[0]["argv"].index("--permission-mode") + 1] == "plan"
    assert fake.responses()[plan.key]["updatedPermissions"] == [{"type": "setMode", "mode": "auto",
                                                                 "destination": "session"}]
    assert write.kind == "edit" and write.files == ({"path": "README.md", "added": 1, "removed": 0},)

    # The runtime then turns plan mode off: the process is in that mode already, nothing more is sent.
    await session.apply(spec(folder, approval="auto_review"))
    assert not fake.controls()
    await session.close()


# ── input ─────────────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_steer_is_accepted_when_the_cli_echoes_it(tmp_path, fake, folder):
    fake.scenario(script("steer"))
    session, events = await opened(tmp_path, folder)
    await session.send("k1", "Run exactly: sleep 4 ; then say finished.")
    await until(lambda: events.of(Accepted))
    await session.send("k2", "Also end your reply with the word banana.")
    await until(lambda: events.of(TurnEnd), timeout=10)
    sleep = started(events)[0]
    order = [event for event in events if isinstance(event, Accepted) or event == ended(events, sleep.key)]
    # The CLI takes the steer at its next model call, after the running command.
    assert order == [Accepted("k1"), ended(events, sleep.key), Accepted("k2")]
    assert ended(events, started(events, "message")[-1].key).fields["text"].endswith("banana")
    assert len(events.of(TurnEnd)) == 1
    await session.close()


@pytest.mark.asyncio
async def test_an_interrupted_turn_ends_interrupted(tmp_path, fake, folder):
    fake.scenario(script("interrupt"))
    session, events = await opened(tmp_path, folder)
    await session.send("k1", "Run exactly: sleep 30")
    await until(lambda: events.of(Accepted))
    await session.interrupt()
    await until(lambda: events.of(TurnEnd))
    assert [end.status for end in events.of(TurnEnd)] == ["interrupted"]
    assert not started(events)  # its call never came whole
    await session.close()


@pytest.mark.asyncio
async def test_a_stop_as_the_turn_ends_leaves_the_next_turn_alone(tmp_path, fake, folder):
    fake.scenario(script("hello") + script("hello"))  # two turns of one process
    session, events = await opened(tmp_path, folder)
    await turn(session, events, "Reply with just: hi")
    await session.interrupt()  # the result is in: nothing runs to stop
    assert (await turn(session, events, "Reply with just: hi", key="k2")).status == "completed"
    assert not fake.controls()
    await session.close()


@pytest.mark.asyncio
async def test_settings_apply_at_turn_start_and_effort_resumes_in_a_new_process(tmp_path, fake, folder):
    fake.scenario(script("hello"), script("hello"), resolved={"sonnet": "claude-sonnet-5-5"})
    session, events = await opened(tmp_path, folder)
    await turn(session, events, "Reply with just: hi")
    await session.apply(spec(folder, model="sonnet", plan=True))
    assert fake.controls() == [{"subtype": "set_model", "model": "sonnet"},
                               {"subtype": "set_permission_mode", "mode": "plan"}]
    await session.apply(spec(folder, model="sonnet", plan=True, effort="high"))
    await turn(session, events, "Reply with just: hi", key="k2")
    first, second = fake.spawns()
    assert second["argv"][second["argv"].index("--resume") + 1] == HELLO_SESSION
    assert [second["argv"][second["argv"].index(flag) + 1] for flag in ("--effort", "--model", "--permission-mode")] \
        == ["high", "sonnet", "plan"]
    assert "--effort" not in first["argv"] and not events.of(Exit)  # the old process went quietly
    await session.close()


@pytest.mark.asyncio
async def test_a_session_the_cli_no_longer_has_starts_afresh(tmp_path, fake, folder):
    gone = {"type": "result", "subtype": "error_during_execution", "is_error": True, "num_turns": 0,
            "session_id": "gone", "errors": ["No conversation found with session ID: gone"]}
    fake.scenario([{"out": gone}, {"exit": 1, "stderr": "No conversation found with session ID: gone"}],
                  script("hello"))
    session, events = await opened(tmp_path, folder, resume="gone")
    end = await turn(session, events, "Reply with just: hi")
    assert end.status == "completed" and events.of(Accepted) == [Accepted("k1")] and not events.of(Exit)
    assert "--resume" not in fake.spawns()[1]["argv"] and session.session_id == HELLO_SESSION
    await session.close()


# ── the process ───────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_process_that_dies_ends_the_session_with_its_reason(tmp_path, fake, folder):
    fake.scenario([{"await": "user"}, {"echo": True}, {"exit": 3, "stderr": "fatal: the API key was revoked"}])
    session, events = await opened(tmp_path, folder)
    await session.send("k1", "hi")
    await until(lambda: events.of(Exit))
    assert events.of(Exit) == [Exit("Claude Code exited: fatal: the API key was revoked")]
    with pytest.raises(RuntimeError):
        await session.send("k2", "again")

    # Closed on purpose, a session ends with no error.
    fake.scenario(script("hello"))
    (fake.path / "spawns").unlink()
    session, events = await opened(tmp_path, folder)
    await session.close()
    assert events.of(Exit) == [Exit()]


@pytest.mark.asyncio
async def test_idle_ok_follows_the_stop_hooks_report_of_the_last_turn(tmp_path, fake, folder):
    stop = tmp_path / "state" / "t1.stop.json"
    stop.parent.mkdir()
    hello = script("hello")
    background = {"session_id": "s", "background_tasks": [{"id": "b1", "type": "local_bash"}], "session_crons": []}
    fake.scenario([*hello, {"stop": background}, {"await": "user"}, *hello[1:], {"stop": {
        "session_id": "s", "background_tasks": [], "session_crons": []}}])
    session, events = await opened(tmp_path, folder, stop_report=str(stop))
    assert not session.idle_ok()  # no report yet
    await turn(session, events, "Reply with just: hi")
    await until(stop.exists)
    assert not session.idle_ok()  # a shell still runs
    await turn(session, events, "Reply with just: hi", key="k2")
    await until(session.idle_ok)
    await session.close()


# ── probing ───────────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_probe_says_what_runs_here(tmp_path, fake):
    backend = ClaudeBackend(str(tmp_path / "home"))
    assert (await ClaudeBackend(str(tmp_path), command="claude-nowhere").probe()).reason == "not_installed"

    fake.scenario(auth={"loggedIn": False})
    info = await backend.probe()
    assert (info.available, info.reason) == (False, "signed_out")

    fake.scenario(auth={"loggedIn": True})
    info = await backend.probe()
    assert info.available and info.default_model == "opus" and info.approvals == ("ask", "auto_review")
    assert [model.id for model in info.models] == ["opus", "fable", "sonnet", "haiku", "claude-haiku-4-5-20251001"]
    opus, *_, old = info.models
    assert (opus.name, opus.efforts) == ("Opus 5.5", ("low", "medium", "high", "xhigh", "max"))
    assert old.efforts == ()
    # The handshake starts no model turn: it sends nothing but control requests.
    assert all(entry["in"]["type"] == "control_request" for entry in fake.log() if "in" in entry)

    fake.scenario(auth={"loggedIn": True}, auto=False)
    assert (await backend.probe()).approvals == ("ask",)


# ── in the runtime ────────────────────────────────────────────────────────────────────────────


# The session's instructions say "update" and its paths "hermes": the guard would read `hermes update`.
@pytest.mark.live_system_guard_bypass
@pytest.mark.asyncio
async def test_a_thread_runs_on_claude_code_through_the_runtime(tmp_path, fake):
    from tests.gateway.aleph_backend import project_adapter
    fake.scenario(script("hello"))
    project, thread = "C41A0000-0000-4000-8000-000000000001", "8D0F0000-0000-4000-8000-000000000002"
    async with project_adapter(tmp_path) as adapter:
        adapter.project_backends["claude"] = ClaudeBackend(str(tmp_path / "home"))
        [listed] = [entry for entry in (await adapter.handle_call("backends.list", {}))["backends"]
                    if entry["id"] == "claude"]
        assert listed["available"] and listed["default_model"] == "opus"
        await adapter.handle_call("project.create", {
            "id": project, "name": "web", "folder": str(adapter.root / "web"), "clone": None, "backend": "claude",
            "model": None, "effort": None, "approval": "ask"})
        await adapter.handle_call("thread.open", {"id": thread, "project": project})
        mark = adapter.store.latest()
        await adapter.handle_call("message.send", {"id": "m-1", "thread": thread, "text": "Reply with just: hi"})
        [done] = await until(lambda: [e.payload for e in adapter.store.events_after(mark)
                                      if e.type == "turn.completed"], timeout=10)
        logged = adapter.store.events_after(mark)
        [reply] = [e.payload for e in logged if e.type == "item.completed"]
        assert (done["status"], reply["kind"], reply["text"]) == ("completed", "message", "hi")
        assert done["usage"]["window"] == 1_000_000
        assert any(e.type == "receipt" and e.payload["read_up_to"] == "m-1" for e in logged)
        # Kept once the turn has ended, for the next process to resume.
        assert await until(lambda: adapter.projects.tables.thread(thread).session_id) == HELLO_SESSION
