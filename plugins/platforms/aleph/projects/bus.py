"""The coordination bus: how a project's top level and its threads reach each other.

Every session's agent gets one MCP server, ``child.py pipe``, which only carries bytes to a Unix
socket here; the bus answers MCP itself (``initialize``, ``tools/list``, ``tools/call``). The token a
session's process is started with says who calls, and so which role's tools it has (``tag.TOOLS``);
arguments are checked again here against the tool's schema, whatever the agent's side checked.

What one session sends another goes through the journal (``deliveries``) first, and is handed over
by one task per target, in order: ``message_thread`` steers the thread's running turn or starts one,
but is held while the thread waits on the person; a ``report`` waits until the top level's turn ends,
and starts one turn of its own. Delivery is at most once: one the last run was handing over when it
stopped is ``uncertain`` and never replayed. ``open_thread`` is idempotent by its ``request_id``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import shutil
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from plugins.platforms.aleph.projects import child, tag
from plugins.platforms.aleph.projects.backend import McpServer
from plugins.platforms.aleph.projects.store import Delivery, JournalError
from plugins.platforms.aleph.server import RpcError

if TYPE_CHECKING:
    from plugins.platforms.aleph.projects.service import Projects

logger = logging.getLogger(__name__)

CHILD = str(Path(child.__file__).resolve())
MAX_FRAME = 1024 * 1024  # bytes of one JSON-RPC line
MAX_AGE = 24 * 3600.0  # a delivery not handed over by then fails
OPEN_LIMIT = 50  # open threads the agent opened, per project
# How long a tool call waits for its delivery to be handed over (a thread's process may be starting)
# before it answers with the status so far: the agent's side gives up on a tool call after ~20 s.
DELIVERY_WAIT = 10.0
PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
MAX_SOCKET_PATH = 100  # bytes: AF_UNIX takes at most 104 (macOS) or 108 (Linux)
UNCERTAIN = "The server stopped while handing it over; check the target before resending."
_DONE, _BLOCKED, _EMPTY = "done", "blocked", "empty"
_DETAIL = {"held": "Held while the thread waits on the person; delivered after.",
           "queued": "Queued; delivered in order once the target is free.",
           "delivering": "Being handed over now."}


class ToolError(Exception):
    """A tool call refused, its message one the agent reads."""


@dataclass(frozen=True)
class Caller:
    project: str
    thread: str
    top_level: bool

    @property
    def role(self) -> str:
        return tag.role(self.top_level)


def _check(tools: tuple[dict[str, Any], ...], name: Any, arguments: Any) -> dict[str, Any]:
    """The tool's arguments, checked against its schema (all the tools take strings)."""
    schema = next((tool["inputSchema"] for tool in tools if tool["name"] == name), None)
    if schema is None:
        raise ToolError(f"Unknown tool {name!r}")
    if not isinstance(arguments, dict):
        raise ToolError("Expected an arguments object")
    if set(arguments) - set(schema["properties"]):
        raise ToolError("Unknown argument: " + ", ".join(sorted(set(arguments) - set(schema["properties"]))))
    if set(schema.get("required", ())) - set(arguments):
        raise ToolError("Missing argument: " + ", ".join(sorted(set(schema["required"]) - set(arguments))))
    for key, value in arguments.items():
        spec = schema["properties"][key]
        if not isinstance(value, str) or not spec.get("minLength", 0) <= len(value) <= spec.get("maxLength", 200):
            raise ToolError(f"Invalid {key}: a string of {spec.get('minLength', 0)} to "
                            f"{spec.get('maxLength', 200)} characters")
    return arguments


class Bus:
    """The MCP endpoint and the journal's delivery, for every project."""

    def __init__(self, projects: Projects) -> None:
        self.projects = projects
        self.path = projects.state_dir / "bus.sock"
        self._fallback: Optional[str] = None  # the private folder holding the socket instead, if made
        self._server: Optional[asyncio.AbstractServer] = None
        self._tokens: dict[str, Caller] = {}
        self._connections: set[asyncio.StreamWriter] = set()
        self._workers: dict[str, asyncio.Event] = {}  # by target: its delivery task's wake-up
        self._waiting: dict[str, list[asyncio.Future]] = {}  # by target: tool calls awaiting its next pass
        self._tasks: set[asyncio.Task] = set()
        self._open_lock = asyncio.Lock()  # one open_thread at a time

    @property
    def _tables(self):
        return self.projects.tables

    # ── lifecycle ──────────────────────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Before any session starts: settle what the last run was handing over, and listen."""
        recovered = await asyncio.to_thread(self._tables.recover_deliveries, UNCERTAIN)
        if recovered:
            logger.warning("Aleph: %d delivery(ies) between sessions were in flight at the restart: uncertain",
                           recovered)

        def prepare() -> None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            os.chmod(self.path.parent, 0o700)  # the socket's only guard besides the tokens
            if len(os.fsencode(self.path)) > MAX_SOCKET_PATH:
                # A state folder too deep for a socket: one made private to this run (0700) instead.
                self._fallback = tempfile.mkdtemp(prefix="aleph-bus-", dir="/tmp")
                self.path = Path(self._fallback) / "bus.sock"
            self.path.unlink(missing_ok=True)
        await asyncio.to_thread(prepare)
        self._server = await asyncio.start_unix_server(self._serve, path=str(self.path), limit=MAX_FRAME)

    async def resume(self) -> None:
        """After the sessions started: hand over what waited through the restart."""
        for target in await asyncio.to_thread(self._tables.delivery_targets):
            self.wake(target)

    async def stop(self) -> None:
        server, self._server = self._server, None
        self._tokens.clear()
        if server is not None:
            server.close()
        for writer in list(self._connections):
            writer.close()
        for task in list(self._tasks):
            task.cancel()
        self._workers.clear()
        if server is not None:
            await server.wait_closed()
            self.path.unlink(missing_ok=True)
        if self._fallback is not None:
            shutil.rmtree(self._fallback, ignore_errors=True)

    def _spawn(self, coroutine) -> asyncio.Task:
        task = asyncio.create_task(coroutine)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    # ── who calls ──────────────────────────────────────────────────────────────────────────────

    def issue(self, project: str, thread: str, top_level: bool) -> str:
        """A token for a session's process, valid until ``revoke``."""
        token = secrets.token_urlsafe(32)
        self._tokens[token] = Caller(project, thread, top_level)
        return token

    def revoke(self, token: Optional[str]) -> None:
        if token is not None:
            self._tokens.pop(token, None)

    def server(self, token: str) -> McpServer:
        """The MCP server a session's process runs, as its backend starts it."""
        caller = self._tokens[token]
        return McpServer(tag.SERVER, sys.executable, ("-I", CHILD, "pipe", str(self.path)), {child.TOKEN_ENV: token},
                         tuple(tool["name"] for tool in tag.TOOLS[caller.role]))

    # ── MCP ────────────────────────────────────────────────────────────────────────────────────

    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self._connections.add(writer)
        lock = asyncio.Lock()
        calls: set[asyncio.Task] = set()

        async def send(message: dict[str, Any]) -> None:
            async with lock:
                writer.write(json.dumps({"jsonrpc": "2.0", **message}, ensure_ascii=False).encode() + b"\n")
                await writer.drain()

        async def reply(caller: Caller, message: dict[str, Any]) -> None:
            try:
                await send({"id": message["id"], "result": await self._call(caller, message.get("params"))})
            except ConnectionError:
                pass
        try:
            token = (await reader.readline()).decode("utf-8", "replace").strip()
            if token not in self._tokens:
                logger.warning("Aleph: refused a bus connection with an unknown token")
                return
            while line := await reader.readline():
                caller = self._tokens.get(token)
                if caller is None:
                    return  # revoked: its session ended
                try:
                    message = json.loads(line)
                except ValueError:
                    await send({"id": None, "error": {"code": -32700, "message": "Parse error"}})
                    continue
                if not isinstance(message, dict) or not isinstance(message.get("method"), str):
                    await send({"id": None, "error": {"code": -32600, "message": "Invalid request"}})
                    continue
                if "id" not in message:
                    continue  # a notification (initialized, cancelled): nothing to answer
                if message["method"] == "tools/call":
                    # Concurrently: the agent may call several tools at once, and a delivery can wait.
                    task = self._spawn(reply(caller, message))
                    calls.add(task)
                    task.add_done_callback(calls.discard)
                else:
                    await send(self._answer(caller, message))
        except (ConnectionError, ValueError):
            pass  # gone, or a line past MAX_FRAME
        finally:
            for task in calls:
                task.cancel()
            self._connections.discard(writer)
            writer.close()

    def _answer(self, caller: Caller, message: dict[str, Any]) -> dict[str, Any]:
        method, params = message["method"], message.get("params")
        if method == "initialize":
            asked = params.get("protocolVersion") if isinstance(params, dict) else None
            result = {"protocolVersion": asked if asked in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0],
                      "capabilities": {"tools": {"listChanged": False}},
                      "serverInfo": {"name": tag.SERVER, "version": "1"},
                      "instructions": tag.mcp_instructions(caller.role)}
        elif method == "ping":
            result = {}
        elif method == "tools/list":
            result = {"tools": list(tag.TOOLS[caller.role])}
        else:
            return {"id": message["id"], "error": {"code": -32601, "message": f"Method not found: {method}"}}
        return {"id": message["id"], "result": result}

    async def _call(self, caller: Caller, params: Any) -> dict[str, Any]:
        params = params if isinstance(params, dict) else {}
        try:
            name = params.get("name")
            arguments = _check(tag.TOOLS[caller.role], name, params.get("arguments", {}))
            if name == tag.OPEN_THREAD:
                result = await self._open_thread(caller, arguments)
            elif name == "message_thread":
                result = await self._message_thread(caller, arguments)
            elif name == "list_threads":
                result = await self._list_threads(caller)
            else:
                result = await self._report(caller, arguments)
        except (ToolError, JournalError) as exc:
            return {"content": [{"type": "text", "text": str(exc)}], "isError": True}
        except RpcError as exc:
            return {"content": [{"type": "text", "text": exc.message}], "isError": True}
        except Exception:
            logger.exception("Aleph: the %s call of thread %s failed", params.get("name"), caller.thread)
            return {"content": [{"type": "text", "text": "The call failed; do not retry it automatically."}],
                    "isError": True}
        return {"content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}], "isError": False}

    # ── the tools ──────────────────────────────────────────────────────────────────────────────

    async def _open_thread(self, caller: Caller, arguments: dict[str, Any]) -> dict[str, Any]:
        """A thread the top level opens: its ``thread`` item (parked by the top level's session when
        the call's item started) gets the thread, whose first turn is the brief."""
        from plugins.platforms.aleph.projects.service import _name
        brief = arguments["brief"]
        if not brief.strip():
            raise ToolError("brief must not be blank")
        request_id = arguments.get("request_id")
        wanted = json.dumps({key: value for key, value in arguments.items() if key != "request_id"}, sort_keys=True)
        async with self._open_lock:
            if request_id is not None:
                known = await asyncio.to_thread(self._tables.opened, caller.thread, request_id)
                if known is not None:
                    if known[0] != wanted:
                        raise ToolError("request_id already belongs to a different open_thread request")
                    return known[1]
            threads = await asyncio.to_thread(self._tables.threads, caller.project)
            if sum(row.payload.get("opened_by") == "agent" and not row.payload.get("closed") for row in threads) >= OPEN_LIMIT:
                raise ToolError(f"Thread limit reached ({OPEN_LIMIT} open threads opened from the top level)")
            top = self.projects.hub.sessions.get(caller.thread)
            parked = await top.claim(brief) if top is not None else None
            item, thread_id = parked if parked is not None else (None, str(uuid.uuid4()).upper())
            name = _name(arguments.get("name") or brief.strip().splitlines()[0])
            thread = await self.projects.open_agent_thread(caller.project, thread_id, name, item, brief)
            if top is not None:
                top.opened(thread_id)
            result = {"thread": thread_id, "name": name}
            if request_id is not None:
                await asyncio.to_thread(self._tables.record_opened, caller.project, caller.thread, request_id,
                                        wanted, result)
        logger.info("Aleph: project %s opened thread %s (%s)", caller.project, thread_id, name)
        self._spawn(self.projects.hub.first_turn(thread))
        return result

    async def _message_thread(self, caller: Caller, arguments: dict[str, Any]) -> dict[str, Any]:
        target = await asyncio.to_thread(self._tables.thread, arguments["thread"])
        if target is None or target.project != caller.project or target.top_level:
            raise ToolError("thread is not a thread of this project; see list_threads")
        if target.payload.get("closed"):
            raise ToolError("That thread is closed")
        return await self._send(caller, target.id, "message", arguments["text"],
                                arguments.get("request_id") or f"message_{uuid.uuid4().hex}")

    async def _report(self, caller: Caller, arguments: dict[str, Any]) -> dict[str, Any]:
        return await self._send(caller, caller.project, "report", arguments["text"],
                                arguments.get("request_id") or f"report_{uuid.uuid4().hex}")

    async def _list_threads(self, caller: Caller) -> dict[str, Any]:
        desk = self.projects.desks.get(caller.project)
        threads = []
        for row in await asyncio.to_thread(self._tables.threads, caller.project):
            if row.payload.get("closed"):
                continue
            session = self.projects.hub.sessions.get(row.id)
            status = ("waiting" if desk is not None and desk.waits(row.id)
                      else "running" if session is not None and session.turn is not None else "idle")
            threads.append({"thread": row.id, "name": row.payload.get("name"), "status": status,
                            "opened_by": "you" if row.payload.get("opened_by") == "agent" else "user",
                            "last_active_at": row.last_active_at})
        threads.sort(key=lambda entry: entry["last_active_at"], reverse=True)
        return {"threads": threads}

    async def _send(self, caller: Caller, target: str, kind: str, text: str, request_id: str) -> dict[str, Any]:
        """Journal a delivery and answer with its status once its target's task has had a go at it."""
        if not text.strip():
            raise ToolError("text must not be blank")
        await asyncio.to_thread(self._tables.journal, caller.project, caller.thread, target, request_id, kind, text,
                                time.time())
        passed = asyncio.get_running_loop().create_future()
        waiting = self._waiting.setdefault(target, [])
        waiting.append(passed)
        self.wake(target)
        try:
            await asyncio.wait_for(passed, DELIVERY_WAIT)
        except TimeoutError:
            pass
        finally:
            waiting.remove(passed)
            if not waiting and self._waiting.get(target) is waiting:
                del self._waiting[target]
        delivery = await asyncio.to_thread(self._tables.delivery, caller.thread, request_id)
        if delivery.status in ("failed", "uncertain"):
            raise ToolError(delivery.detail or "Not delivered")
        receipt: dict[str, Any] = {"status": delivery.status, "delivered": delivery.status == "delivered",
                                   "request_id": request_id}
        if delivery.status in _DETAIL:
            receipt["detail"] = ("Queued until the top level's turn ends; delivered then." if kind == "report"
                                 and delivery.status == "queued" else _DETAIL[delivery.status])
        return receipt

    # ── delivery ───────────────────────────────────────────────────────────────────────────────

    def wake(self, target: str) -> None:
        """Something may let ``target``'s deliveries go: a new one, its turn's end, a request settled."""
        if self._server is None:
            return
        event = self._workers.get(target)
        if event is None:
            event = self._workers[target] = asyncio.Event()
            self._spawn(self._work(target, event))
        event.set()

    async def _work(self, target: str, event: asyncio.Event) -> None:
        try:
            while True:
                await event.wait()
                event.clear()
                outcome = _DONE
                while outcome == _DONE:
                    outcome = await self._deliver_one(target)
                for passed in self._waiting.get(target, ()):
                    if not passed.done():
                        passed.set_result(None)
                if outcome == _EMPTY and not event.is_set():
                    return
        except Exception:
            # What it was handing over stays marked so, and is uncertain after a restart.
            logger.exception("Aleph: delivering to thread %s failed", target)
        finally:
            if self._workers.get(target) is event:
                del self._workers[target]

    async def _deliver_one(self, target: str) -> str:
        """Hand over ``target``'s oldest waiting delivery, if it can go now."""
        tables = self._tables
        delivery = await asyncio.to_thread(tables.next_delivery, target)
        if delivery is None:
            return _EMPTY
        thread = await asyncio.to_thread(tables.thread, target)
        project = await asyncio.to_thread(tables.project, delivery.project)
        if thread is None or project is None or project.deleted_seq is not None or thread.payload.get("closed"):
            await asyncio.to_thread(tables.set_delivery, delivery.seq, "failed", "That thread is closed.")
            return _DONE
        if time.time() - delivery.created > MAX_AGE:
            await asyncio.to_thread(tables.set_delivery, delivery.seq, "failed", "Expired after 24 hours.")
            return _DONE
        session = self.projects.hub.session(thread)
        report = delivery.kind == "report"
        # Checked again under the session's gate (``submit``); this spares the journal a write.
        if (session.turn is not None) if report else session.waiting:
            await self._block(delivery, delivery.status)
            return _BLOCKED
        await asyncio.to_thread(tables.set_delivery, delivery.seq, "delivering")
        try:
            taken = await session.submit(await self._prompt(delivery), cause="report" if report else "agent",
                                         origin=delivery.source if report else None, when="idle" if report else "free")
        except RpcError as exc:
            await asyncio.to_thread(tables.set_delivery, delivery.seq, "failed", exc.message)
            return _DONE
        except Exception:
            logger.warning("Aleph: handing over delivery %d is uncertain", delivery.seq, exc_info=True)
            await asyncio.to_thread(tables.set_delivery, delivery.seq, "uncertain",
                                    "The target's backend did not confirm it; check before resending.")
            return _DONE
        if not taken:
            await self._block(delivery, "delivering")
            return _BLOCKED
        await asyncio.to_thread(tables.set_delivery, delivery.seq, "delivered")
        return _DONE

    async def _block(self, delivery: Delivery, current: str) -> None:
        """It cannot go yet: a report waits for the top level's turn to end, a message for the thread
        to stop waiting on the person."""
        status = "queued" if delivery.kind == "report" else "held"
        if current != status:
            await asyncio.to_thread(self._tables.set_delivery, delivery.seq, status)

    async def _prompt(self, delivery: Delivery) -> str:
        if delivery.kind != "report":
            return tag.top_message_prompt(delivery.text)
        source = await asyncio.to_thread(self._tables.thread, delivery.source)
        name = (source.payload.get("name") if source is not None else None) or delivery.source
        return tag.report_prompt(name, delivery.text)
