"""Projects on Aleph, coordination: the top level opening threads (once per request_id, a ``thread``
item and ``thread.new``, the brief as the thread's first turn), reports reaching the top level one turn
each after its own turn, messages steering a thread or held while it waits on the person, each role's
tools only, and deliveries a restart caught mid-way left uncertain. All of it through the agent's own
MCP server (``child.py pipe``) and the bus's Unix socket."""

from __future__ import annotations

import asyncio
import json
import os

import pytest
import pytest_asyncio

from plugins.platforms.aleph.projects.backend import Ask, Call, Item, TurnEnd
from plugins.platforms.aleph.projects.tag import SERVER
from tests.gateway.aleph_backend import project_adapter, until

PROJECT = "C41A0000-0000-4000-8000-000000000001"
THREAD = "8D0F0000-0000-4000-8000-000000000002"


@pytest_asyncio.fixture
async def adapter(tmp_path):
    async with project_adapter(tmp_path) as instance:
        await instance.handle_call("project.create", {
            "id": PROJECT, "name": "web", "folder": str(instance.root / "web"), "clone": None, "backend": "claude",
            "model": None, "effort": None, "approval": "ask"})
        await instance.handle_call("thread.open", {"id": THREAD, "project": PROJECT, "name": "fix-login"})
        instance.backend = instance.projects.backends["claude"]
        yield instance


class Mcp:
    """An agent's MCP client, talking to the server its backend session was given to run."""

    def __init__(self, process: asyncio.subprocess.Process) -> None:
        self.process = process
        self._id = 0

    @classmethod
    async def start(cls, session, token=None) -> Mcp:
        server = session.spec.mcp
        env = {**os.environ, **server.env, **({"ALEPH_BUS_TOKEN": token} if token else {})}
        process = await asyncio.create_subprocess_exec(server.command, *server.args, env=env,
                                                       stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE)
        return cls(process)

    async def request(self, method: str, params=None):
        """The answer to one request, or None when the server hung up."""
        self._id += 1
        self.process.stdin.write(json.dumps({"jsonrpc": "2.0", "id": self._id, "method": method,
                                             **({"params": params} if params is not None else {})}).encode() + b"\n")
        await self.process.stdin.drain()
        line = await asyncio.wait_for(self.process.stdout.readline(), 15)
        return json.loads(line) if line else None

    async def call(self, tool: str, **arguments):
        """``(is_error, what it answered)``: the result's JSON, or the error's text."""
        result = (await self.request("tools/call", {"name": tool, "arguments": arguments}))["result"]
        text = result["content"][0]["text"]
        return result["isError"], text if result["isError"] else json.loads(text)

    async def close(self) -> None:
        if self.process.returncode is None:
            self.process.stdin.close()
            await asyncio.wait_for(self.process.wait(), 5)


async def _session(adapter, thread, message_id, text="Go"):
    """The thread's backend session, a turn running: started by a message from the person."""
    await adapter.handle_call("message.send", {"id": message_id, "thread": thread, "text": text})
    return next(session for session in reversed(adapter.backend.sessions) if session.spec.thread == thread)


def _events(adapter, after, event_type, thread=None):
    return [event.payload for event in adapter.store.events_after(after) if event.type == event_type
            and thread in (None, event.payload.get("thread"))]


async def _logged(adapter, after, event_type, thread=None, count=1):
    return await until(lambda: (found := _events(adapter, after, event_type, thread)) and len(found) >= count and found)


def _open_call(key, status="in_progress", **arguments):
    return Item(key, "tool", status, {"tool": "other", "title": "open_thread", "input": json.dumps(arguments)},
                call=Call(SERVER, "open_thread", arguments))


# ── open_thread ───────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_open_thread_opens_one_thread_per_request_id_and_starts_it_with_the_brief(adapter):
    top = await _session(adapter, PROJECT, "m-1", "The login form is broken")
    assert "NO_REPLY" in top.spec.instructions and top.spec.mcp.tools == ("open_thread", "message_thread",
                                                                          "list_threads")
    mcp = await Mcp.start(top)
    try:
        hello = (await mcp.request("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                                  "clientInfo": {"name": "claude-code", "version": "2"}}))["result"]
        assert hello["protocolVersion"] == "2025-06-18" and hello["serverInfo"]["name"] == SERVER
        listed = (await mcp.request("tools/list"))["result"]["tools"]
        assert [tool["name"] for tool in listed] == list(top.spec.mcp.tools)

        brief = "Fix the login form\n\nThe submit button does nothing since the API moved."
        arguments = {"brief": brief, "request_id": "open-1"}
        mark = adapter.store.latest()
        top.emit(_open_call("toolu_1", **arguments))
        [item] = await _logged(adapter, mark, "item.started")
        assert item["kind"] == "thread" and item["thread"] == PROJECT and "title" not in item
        error, opened = await mcp.call("open_thread", **arguments)
        assert not error and opened == {"thread": item["opened"], "name": "Fix the login form"}
        [new] = _events(adapter, mark, "thread.new")
        assert {key: new[key] for key in ("id", "project", "name", "opened_by", "item", "brief", "closed")} == {
            "id": item["opened"], "project": PROJECT, "name": "Fix the login form", "opened_by": "agent",
            "item": item["id"], "brief": brief, "closed": False}

        # The thread's first turn is the brief, the agent's own; its session knows its role.
        child = await until(lambda: next((s for s in adapter.backend.sessions if s.spec.thread == item["opened"]
                                          and s.sent), None))
        assert child.sent == [brief]
        [started] = await _logged(adapter, mark, "turn.started", item["opened"])
        assert started["cause"] == "agent"
        assert "one thread of that project: Fix the login form" in child.spec.instructions
        assert str(adapter.state_dir / "projects" / PROJECT / "memory") in child.spec.instructions
        assert child.spec.mcp.tools == ("report",)
        top.emit(_open_call("toolu_1", "completed", **arguments))
        [done] = await _logged(adapter, mark, "item.completed")
        assert done["id"] == item["id"] and done["status"] == "completed" and done["opened"] == item["opened"]

        # A retry under the same request_id is the same thread; its own item opened none.
        top.emit(_open_call("toolu_2", **arguments))
        assert await mcp.call("open_thread", **arguments) == (False, opened)
        top.emit(_open_call("toolu_2", "completed", **arguments))
        retried = (await _logged(adapter, mark, "item.completed", count=2))[1]
        assert retried["kind"] == "thread" and retried["status"] == "failed"
        assert len(_events(adapter, mark, "thread.new")) == 1
        error, text = await mcp.call("open_thread", brief="Something else", request_id="open-1")
        assert error and "request_id" in text
    finally:
        await mcp.close()


# ── report and message_thread ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reports_wait_for_the_top_levels_turn_and_start_one_turn_each_in_order(adapter):
    thread = await _session(adapter, THREAD, "m-1")
    top = await _session(adapter, PROJECT, "m-2")
    mcp = await Mcp.start(thread)
    try:
        mark = adapter.store.latest()
        for request_id, text in (("r-1", "Fixed the <form> handler"), ("r-2", "Tests pass")):
            error, receipt = await mcp.call("report", text=text, request_id=request_id)
            assert not error and receipt["status"] == "queued" and receipt["request_id"] == request_id
        assert top.sent == ["Go"]  # not steered into the person's turn

        top.emit(TurnEnd("completed"))
        [first] = await _logged(adapter, mark, "turn.started", PROJECT)
        assert first["cause"] == "report" and first["from"] == THREAD
        # The turn is logged as it starts, its input handed over right after.
        await until(lambda: len(top.sent) == 2)
        assert top.sent[-1] == "Report from you in the thread 'fix-login':\nFixed the \\u003cform\\u003e handler"
        top.emit(TurnEnd("completed"))
        second = (await _logged(adapter, mark, "turn.started", PROJECT, count=2))[1]
        await until(lambda: len(top.sent) == 3)
        assert second["cause"] == "report" and top.sent[-1].endswith(":\nTests pass")
        await until(lambda: adapter.store.projects.delivery(THREAD, "r-2").status == "delivered")
        assert adapter.store.projects.delivery(THREAD, "r-1").status == "delivered"
        assert await mcp.call("report", text="Tests pass", request_id="r-2") == (False, {
            "status": "delivered", "delivered": True, "request_id": "r-2"})
    finally:
        await mcp.close()


@pytest.mark.asyncio
async def test_message_thread_steers_the_thread_or_is_held_while_it_waits_on_the_person(adapter):
    thread = await _session(adapter, THREAD, "m-1")
    top = await _session(adapter, PROJECT, "m-2")
    mcp = await Mcp.start(top)
    try:
        mark = adapter.store.latest()
        # The agent talking to itself is no item.
        call = Call(SERVER, "message_thread", {"thread": THREAD, "text": "Use the v2 endpoint"})
        top.emit(Item("toolu_1", "tool", fields={"tool": "other", "title": "message_thread"}, call=call))
        error, receipt = await mcp.call("message_thread", **call.arguments)
        assert not error and receipt["status"] == "delivered"
        top.emit(Item("toolu_1", "tool", "completed", {"tool": "other"}))
        assert thread.sent[-1] == "Message from you at the top level:\nUse the v2 endpoint"
        assert not _events(adapter, mark, "turn.started") and not _events(adapter, mark, "item.started")

        thread.emit(Ask("ask_1", "command", command="npm test", text="Runs the suite"))
        [request] = await _logged(adapter, mark, "approval.request")
        error, receipt = await mcp.call("message_thread", thread=THREAD, text="Skip the e2e suite")
        assert not error and receipt["status"] == "held"
        assert thread.sent[-1] != "Message from you at the top level:\nSkip the e2e suite"
        error, listed = await mcp.call("list_threads")
        assert listed["threads"] == [{"thread": THREAD, "name": "fix-login", "status": "waiting", "opened_by": "user",
                                      "last_active_at": listed["threads"][0]["last_active_at"]}]

        await adapter.handle_call("approval.respond", {"request_id": request["request_id"], "choice": "once"})
        await until(lambda: thread.sent[-1] == "Message from you at the top level:\nSkip the e2e suite")
        assert not _events(adapter, mark, "turn.started")  # still steering the same turn
    finally:
        await mcp.close()


# ── who may call what ─────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_each_role_has_its_own_tools_their_arguments_checked_and_tokens_end_with_the_session(adapter):
    thread = await _session(adapter, THREAD, "m-1")
    top = await _session(adapter, PROJECT, "m-2")
    thread_mcp, top_mcp, stranger = await Mcp.start(thread), await Mcp.start(top), await Mcp.start(top, "forged")
    try:
        assert [tool["name"] for tool in (await thread_mcp.request("tools/list"))["result"]["tools"]] == ["report"]
        error, text = await thread_mcp.call("open_thread", brief="Take over the project")
        assert error and "Unknown tool" in text
        for arguments, refusal in (({"thread": THREAD}, "Missing argument: text"),
                                   ({"thread": THREAD, "text": "x", "urgent": "yes"}, "Unknown argument: urgent"),
                                   ({"thread": THREAD, "text": "x" * 6001}, "Invalid text"),
                                   ({"thread": PROJECT, "text": "x"}, "not a thread of this project")):
            error, text = await top_mcp.call("message_thread", **arguments)
            assert error and refusal in text, arguments
        assert await stranger.request("tools/list") is None  # an unknown token is hung up on

        await adapter.handle_call("thread.close", {"id": THREAD})
        assert await thread_mcp.request("tools/list") is None  # its session ended: so did its token
    finally:
        for mcp in (thread_mcp, top_mcp, stranger):
            await mcp.close()


# ── restarts ──────────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_delivery_a_restart_caught_mid_way_is_uncertain_and_never_replayed(adapter):
    thread = await _session(adapter, THREAD, "m-1")
    await _session(adapter, PROJECT, "m-2")
    mcp = await Mcp.start(thread)
    try:
        for request_id in ("r-1", "r-2"):
            assert (await mcp.call("report", text=f"Report {request_id}", request_id=request_id))[0] is False
    finally:
        await mcp.close()
    tables = adapter.store.projects
    tables.set_delivery(tables.delivery(THREAD, "r-1").seq, "delivering")  # the server stopped handing it over
    sessions = len(adapter.backend.sessions)
    await adapter.disconnect()
    assert await adapter.connect()

    # The one waiting still goes, starting the top level's turn; the one in flight never goes again.
    resumed = await until(lambda: next((s for s in adapter.backend.sessions[sessions:] if s.spec.thread == PROJECT
                                        and s.sent), None))
    assert resumed.sent == ["Report from you in the thread 'fix-login':\nReport r-2"]
    tables = adapter.store.projects
    assert tables.delivery(THREAD, "r-1").status == "uncertain"
    await until(lambda: tables.delivery(THREAD, "r-2").status == "delivered")
