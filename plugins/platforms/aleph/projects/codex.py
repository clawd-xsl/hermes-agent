"""Codex as a projects backend: one shared ``codex app-server`` (JSON-RPC over stdio) serves every
thread, one Codex thread per Aleph thread.

What the app-server sends is routed by its thread id to that thread's session, which turns it into the
neutral events of ``backend.py``. A subagent runs in a Codex thread of its own: its notifications are
dropped (the protocol shows only the call that spawned it) and its requests go to the session of the
thread that spawned it. Settings go with every ``turn/start``, so a change applies from the next turn;
plan mode is the turn's collaboration mode. When the process dies, every session exits with it and the
next use starts a new one.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import logging
import os
import shlex
import shutil
import signal
from pathlib import Path
from typing import Any, Callable, Optional

from plugins.platforms.aleph.projects.backend import (
    Accepted, Answers, Approve, Ask, BackendInfo, Call, Decision, Delta, Deny, Exit, Item, Model, Patch,
    Question, SessionSpec, Sink, TurnEnd, clip, plain_options, title)
from plugins.platforms.aleph.projects import process
from plugins.platforms.aleph.projects.process import STREAM_LIMIT, end, signal_group
from plugins.platforms.aleph.projects.service import one_line

logger = logging.getLogger(__name__)

REQUEST_SECONDS = 60.0
CLIENT = {"name": "aleph", "title": "Aleph", "version": "1"}
# Notifications nothing here reads; the app-server then never sends them.
_UNREAD = ["account/rateLimits/updated", "mcpServer/startupStatus/updated", "turn/moderationMetadata",
           "turn/diff/updated", "thread/status/changed", "remoteControl/status/changed",
           "item/reasoning/textDelta", "rawResponseItem/completed"]
# Each approval mode as the app-server takes it: approval policy, who reviews, sandbox.
_MODES = {"ask": ("on-request", "user", "workspace-write"),
          "auto_review": ("on-request", "auto_review", "workspace-write"),
          "allow_all": ("never", "user", "danger-full-access")}
_STATUS = {"inProgress": "in_progress", "completed": "completed", "failed": "failed", "declined": "declined"}
_SHELLS = frozenset({"bash", "zsh", "sh"})
_APPROVALS = ("item/commandExecution/requestApproval", "item/fileChange/requestApproval")
_QUESTION = "item/tool/requestUserInput"
_PERMISSIONS = "item/permissions/requestApproval"
_ELICITATION = "mcpServer/elicitation/request"


class CodexError(Exception):
    """The app-server answered a request with an error."""

    def __init__(self, method: str, error: dict[str, Any]) -> None:
        super().__init__(str(error.get("message") or f"{method} failed"))


def _shown(command: str) -> str:
    """A command as the model wrote it: Codex runs each through ``<shell> -lc '<command>'``."""
    try:
        words = shlex.split(command)
    except ValueError:
        return command
    if len(words) == 3 and os.path.basename(words[0]) in _SHELLS and words[1] in ("-lc", "-c"):
        return words[2]
    return command


def _whole(text: str, sign: str) -> str:
    """An added (``+``) or deleted (``-``) file's content, which Codex sends bare, as a unified diff."""
    lines = text.splitlines()
    if not lines:
        return ""
    head = f"@@ -0,0 +1,{len(lines)} @@" if sign == "+" else f"@@ -1,{len(lines)} +0,0 @@"
    return "\n".join([head, *(sign + line for line in lines)])


def _counts(diff: str) -> tuple[int, int]:
    lines = [line for line in diff.splitlines() if not line.startswith("@@")]
    return sum(line[:1] == "+" for line in lines), sum(line[:1] == "-" for line in lines)


class _AppServer:
    """One ``codex app-server`` process. Every message with a method (a notification, or a request
    the server makes of us) goes to ``on_message``; ``on_exit`` hears once that the process ended."""

    def __init__(self, executable: str, env: dict[str, str], on_message: Callable[[dict[str, Any]], None],
                 on_exit: Callable[[_AppServer, Optional[str]], None]) -> None:
        self._executable, self._env = executable, env
        self._on_message, self._on_exit = on_message, on_exit
        self._process: Optional[asyncio.subprocess.Process] = None
        self._ids = itertools.count(1)
        # Each request waiting for its answer: its method, and what must run as the answer lands, before
        # any later message is handled.
        self._pending: dict[int, tuple[str, asyncio.Future, Optional[Callable[[Any], None]]]] = {}
        self._lock = asyncio.Lock()
        self._tasks: list[asyncio.Task] = []
        self._closing = False

    async def start(self) -> None:
        self._process = await asyncio.create_subprocess_exec(
            self._executable, "app-server", stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, env=self._env, limit=STREAM_LIMIT, start_new_session=True)
        self._tasks = [asyncio.create_task(self._read(), name="codex-app-server"),
                       asyncio.create_task(self._drain(), name="codex-app-server-stderr")]
        try:
            await self.request("initialize", {"clientInfo": CLIENT, "capabilities": {
                "experimentalApi": True, "requestAttestation": False, "optOutNotificationMethods": _UNREAD}})
            await self._write({"method": "initialized"})
        except BaseException:
            await self.close()
            raise

    async def request(self, method: str, params: dict[str, Any], *,
                      then: Optional[Callable[[Any], None]] = None) -> Any:
        """The result of ``method``; ``then`` runs with it as it arrives, so what it emits comes
        before anything the server sends after its answer."""
        request_id = next(self._ids)
        future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = (method, future, then)
        try:
            await self._write({"id": request_id, "method": method, "params": params})
            return await asyncio.wait_for(future, REQUEST_SECONDS)
        finally:
            self._pending.pop(request_id, None)

    async def respond(self, request_id: Any, result: dict[str, Any]) -> None:
        await self._write({"id": request_id, "result": result})

    async def refuse(self, request_id: Any, message: str) -> None:
        await self._write({"id": request_id, "error": {"code": -32601, "message": message}})

    async def _write(self, message: dict[str, Any]) -> None:
        if self._process is None or self._process.returncode is not None:
            raise ConnectionError("Codex's app-server is not running")
        line = json.dumps({"jsonrpc": "2.0", **message}, ensure_ascii=False, separators=(",", ":")) + "\n"
        async with self._lock:
            self._process.stdin.write(line.encode("utf-8"))
            await self._process.stdin.drain()

    async def _read(self) -> None:
        process = self._process
        try:
            while True:
                try:
                    line = await process.stdout.readline()
                except ValueError:
                    # A line past STREAM_LIMIT, dropped: one message is lost, not every thread's process.
                    logger.warning("Aleph: codex app-server wrote a line past %d bytes", STREAM_LIMIT)
                    continue
                if not line:
                    break
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Aleph: codex app-server wrote a line that is not JSON: %.200s", line)
                    continue
                if not isinstance(message, dict):
                    continue
                if "method" in message:
                    try:
                        self._on_message(message)
                    except Exception:
                        logger.exception("Aleph: handling codex %s failed", message["method"])
                        if "id" in message:  # unanswered, it would hold its thread for good
                            with contextlib.suppress(ConnectionError):
                                await self.refuse(message["id"], "the client could not handle it")
                    continue
                method, future, then = self._pending.get(message.get("id"), ("", None, None))
                if future is None or future.done():
                    continue
                if "error" in message:
                    future.set_exception(CodexError(method, message["error"] or {}))
                    continue
                if then is not None:
                    try:
                        then(message.get("result"))
                    except Exception:
                        logger.exception("Aleph: handling a codex answer failed")
                future.set_result(message.get("result"))
        except ConnectionError as exc:
            logger.warning("Aleph: reading codex app-server failed: %s", exc)
        finally:
            if process.returncode is None:
                signal_group(process, signal.SIGKILL)  # its MCP servers too: they are of its group
            code = await process.wait()
            for _, future, _ in list(self._pending.values()):
                if not future.done():
                    future.set_exception(ConnectionError("Codex's app-server exited"))
            self._on_exit(self, None if self._closing else f"Codex's app-server exited (code {code}).")

    async def _drain(self) -> None:
        async for line in self._process.stderr:
            logger.debug("Aleph: codex app-server: %s", line.decode("utf-8", "replace").rstrip())

    async def close(self) -> None:
        process, self._closing = self._process, True
        if process is None:
            return
        await end(process)
        await asyncio.gather(*self._tasks, return_exceptions=True)


class CodexBackend:
    """Codex through its app-server, run with ``HOME`` as ``home`` and ``CODEX_HOME`` as ``codex_home``
    (``home/.codex`` by default): its sign-in, config and transcripts live there."""

    id = "codex"
    name = "Codex"
    asks_to_leave_plan = False  # a plan-mode turn just ends with its plan: the server asks

    def __init__(self, home: Path, codex_home: Optional[Path] = None) -> None:
        self.home = Path(home)
        self.codex_home = Path(codex_home) if codex_home is not None else self.home / ".codex"
        self._server: Optional[_AppServer] = None
        self._starting = asyncio.Lock()
        self._sessions: dict[str, CodexSession] = {}  # by Codex thread id
        self._parents: dict[str, str] = {}  # a subagent's thread -> the thread that spawned it
        self._info: Optional[BackendInfo] = None
        self._tasks: set[asyncio.Task] = set()

    # ── the shared process ─────────────────────────────────────────────────────────────────────

    async def _ready(self) -> _AppServer:
        async with self._starting:
            if self._server is None:
                executable = shutil.which("codex")
                if executable is None:
                    raise FileNotFoundError("codex is not on PATH")
                self.codex_home.mkdir(parents=True, exist_ok=True)  # Codex refuses a CODEX_HOME that is missing
                env = process.environment(str(self.home), CODEX_HOME=str(self.codex_home))
                server = _AppServer(executable, env, self._dispatch, self._exited)
                await server.start()
                self._server = server
            return self._server

    def _exited(self, server: _AppServer, error: Optional[str]) -> None:
        if server is not self._server:
            return
        if error:
            logger.warning("Aleph: %s", error)
        self._server = None
        sessions, self._sessions = list(self._sessions.values()), {}
        self._parents.clear()
        for session in sessions:
            session.exited(error)

    async def stop(self) -> None:
        """End the app-server (every open session exits); the next use starts it again."""
        if self._server is not None:
            await self._server.close()

    def _dispatch(self, message: dict[str, Any]) -> None:
        params = message.get("params") or {}
        thread_id = str(params.get("threadId") or "")
        if "id" not in message:
            session = self._sessions.get(thread_id)
            if session is not None:
                session.notified(message["method"], params)
            elif (session := self._session_of(thread_id)) is not None:  # a subagent's: not shown
                session.subagent_notified(message["method"], params)
            return
        session = self._session_of(thread_id)
        if session is not None:
            session.requested(message["id"], message["method"], params, child=session.thread_id != thread_id)
        else:
            self.spawn(self._adopt(message["id"], message["method"], params, thread_id))

    def spawn(self, coroutine) -> None:
        task = asyncio.create_task(coroutine)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _session_of(self, thread_id: str) -> Optional[CodexSession]:
        seen = set()
        while thread_id and thread_id not in self._sessions and thread_id not in seen:
            seen.add(thread_id)
            thread_id = self._parents.get(thread_id, "")
        return self._sessions.get(thread_id)

    async def _adopt(self, request_id: Any, method: str, params: dict[str, Any], thread_id: str) -> None:
        """A request from a thread no session knows: a subagent's whose spawning call was not seen
        (spawned by another subagent). Its ancestry says whose it is; nobody's is refused."""
        server = self._server
        try:
            current, seen = thread_id, set()
            while current and current not in self._sessions and current not in self._parents and current not in seen:
                seen.add(current)
                thread = (await server.request("thread/read", {"threadId": current, "includeTurns": False}))["thread"]
                parent = str(thread.get("parentThreadId") or "")
                if parent:
                    self._parents[current] = parent
                current = parent
            session = self._session_of(thread_id)
            if session is not None:
                session.requested(request_id, method, params, child=True)
                return
        except Exception as exc:
            logger.warning("Aleph: codex %s for thread %s: %s", method, thread_id, exc)
        with contextlib.suppress(ConnectionError):
            await server.refuse(request_id, "no session for this thread")

    # ── Backend ────────────────────────────────────────────────────────────────────────────────

    async def probe(self) -> BackendInfo:
        if shutil.which("codex") is None:
            return BackendInfo(self.id, self.name, False, "not_installed", "codex is not on PATH")
        try:
            server = await self._ready()
            account = await server.request("account/read", {})
            if account.get("account") is None and account.get("requiresOpenaiAuth"):
                return BackendInfo(self.id, self.name, False, "signed_out", "Codex is not signed in")
            listed, cursor = [], None
            while True:
                page = await server.request("model/list", {"includeHidden": False, "cursor": cursor})
                listed += page.get("data") or []
                cursor = page.get("nextCursor")
                if not cursor:
                    break
            config = (await server.request("config/read", {})).get("config") or {}
            requirements = (await server.request("configRequirements/read", {})).get("requirements") or {}
        except Exception as exc:
            return BackendInfo(self.id, self.name, False, "error", one_line(exc) or "Codex did not answer")
        models = tuple(Model(entry["model"], entry.get("displayName") or entry["model"],
                             tuple(option["reasoningEffort"] for option in entry.get("supportedReasoningEfforts") or ()),
                             entry.get("defaultReasoningEffort")) for entry in listed)
        ids = [model.id for model in models]
        default = config.get("model") if config.get("model") in ids else next(
            (entry["model"] for entry in listed if entry.get("isDefault")), ids[0] if ids else None)

        def allowed(key: str, value: str) -> bool:
            return requirements.get(key) is None or value in requirements[key]
        approvals = tuple(mode for mode, (policy, reviewer, sandbox) in _MODES.items()
                          if allowed("allowedApprovalPolicies", policy)
                          and allowed("allowedApprovalsReviewers", reviewer)
                          and allowed("allowedSandboxModes", sandbox))
        self._info = BackendInfo(self.id, self.name, True, models=models, default_model=default, approvals=approvals)
        return self._info

    async def open(self, spec: SessionSpec, sink: Sink) -> CodexSession:
        server = await self._ready()
        session = CodexSession(self, server, spec, sink)
        await session.begin()
        return session

    def defaults(self, model: Optional[str]) -> tuple[Optional[str], Optional[str]]:
        """The model and effort a turn runs with when the thread names none: what ``backends.list``
        offered as the default."""
        info = self._info
        model = model or (info.default_model if info else None)
        listed = info.model(model) if info else None
        return model, listed.default_effort if listed else None


class CodexSession:
    """One Aleph thread's Codex thread on the shared app-server."""

    def __init__(self, backend: CodexBackend, server: _AppServer, spec: SessionSpec, sink: Sink) -> None:
        self.backend, self.server, self.spec, self._sink = backend, server, spec, sink
        self.thread_id = ""
        self._model: Optional[str] = None  # the thread's own, as Codex resolved it at start
        self._turn: Optional[str] = None  # the running turn's id
        self._usage: Optional[dict[str, int]] = None
        self._started: set[str] = set()  # items whose start was emitted
        self._texts: dict[str, list[str]] = {}  # a message's or a reasoning's parts so far
        self._streamed: set[str] = set()  # commands whose output came as deltas
        self._edits: dict[str, tuple[Patch, ...]] = {}  # open file changes, for their approval
        self._tools: dict[str, tuple[str, str]] = {}  # open MCP calls: (server, tool)
        # Open requests, by ask key: (request id, method, params, ask, whether a subagent's).
        self._asks: dict[str, tuple[Any, str, dict[str, Any], Ask, bool]] = {}
        self._reroutes = itertools.count(1)
        self._gone = False

    @property
    def session_id(self) -> Optional[str]:
        return self.thread_id or None

    # ── the thread ─────────────────────────────────────────────────────────────────────────────

    def _thread_params(self) -> dict[str, Any]:
        spec = self.spec
        policy, reviewer, sandbox = _MODES[spec.approval]
        params: dict[str, Any] = {"cwd": spec.folder, "approvalPolicy": policy, "approvalsReviewer": reviewer,
                                  "sandbox": sandbox, "developerInstructions": spec.instructions or None}
        if spec.mcp is not None:
            mcp = spec.mcp
            params["config"] = {"mcp_servers": {mcp.name: {
                "command": mcp.command, "args": list(mcp.args), "env": dict(mcp.env),
                "tools": {tool: {"approval_mode": "approve"} for tool in mcp.tools}}}}
        return params

    async def begin(self) -> None:
        """Start the thread, or resume the spec's; a thread Codex no longer has starts anew."""
        params = self._thread_params()

        def bind(result: Any) -> None:
            # As the answer lands: what the app-server sends of the thread right after it is routed here.
            self.thread_id, self._model = result["thread"]["id"], result.get("model")
            self.backend._sessions[self.thread_id] = self
        if self.spec.resume:
            try:
                await self.server.request("thread/resume", {"threadId": self.spec.resume, **params}, then=bind)
                return
            except CodexError as exc:
                logger.warning("Aleph: codex thread %s does not resume (%s): starting a new one", self.spec.resume, exc)
        await self.server.request("thread/start", params, then=bind)

    def _sandbox(self, sandbox: str) -> dict[str, Any]:
        if sandbox == "danger-full-access":
            return {"type": "dangerFullAccess"}
        # The project's shared memory lies outside its folder: writable too, or every note asks.
        roots = [self.spec.memory] if self.spec.memory else []
        return {"type": "workspaceWrite", "writableRoots": roots, "networkAccess": False,
                "excludeTmpdirEnvVar": False, "excludeSlashTmp": False}

    def _turn_params(self, text: str) -> dict[str, Any]:
        spec = self.spec
        policy, reviewer, sandbox = _MODES[spec.approval]
        model, effort = self.backend.defaults(spec.model)
        model, effort = model or self._model, spec.effort or effort
        return {"threadId": self.thread_id, "input": [{"type": "text", "text": text, "text_elements": []}],
                "cwd": spec.folder, "approvalPolicy": policy, "approvalsReviewer": reviewer,
                "sandboxPolicy": self._sandbox(sandbox), "model": model, "effort": effort,
                # Sent on every turn: a mode stays with the thread until another is sent.
                "collaborationMode": {"mode": "plan" if spec.plan else "default", "settings": {
                    "model": model, "reasoning_effort": effort, "developer_instructions": None}}}

    # ── BackendSession ─────────────────────────────────────────────────────────────────────────

    async def send(self, key: str, text: str) -> None:
        await self._give(text, key)

    async def _give(self, text: str, key: Optional[str]) -> None:
        """Steer the running turn with ``text``, or start one with it; ``Accepted(key)`` as it lands."""
        def accepted(_: Any) -> None:
            if key is not None:
                self._sink(Accepted(key))
        if self._turn is not None:
            try:
                await self.server.request("turn/steer", {
                    "threadId": self.thread_id, "expectedTurnId": self._turn,
                    "input": [{"type": "text", "text": text, "text_elements": []}]}, then=accepted)
                return
            except CodexError as exc:  # the turn ended before the steer reached it
                logger.info("Aleph: codex thread %s: steer missed its turn (%s): starting one", self.thread_id, exc)

        def started(result: Any) -> None:
            self._turn = result["turn"]["id"]
            accepted(result)
        await self.server.request("turn/start", self._turn_params(text), then=started)

    async def interrupt(self) -> None:
        if self._turn is not None:
            with contextlib.suppress(CodexError):
                await self.server.request("turn/interrupt", {"threadId": self.thread_id, "turnId": self._turn})

    async def answer(self, ask: str, decision: Decision) -> None:
        pending = self._asks.pop(ask, None)
        if pending is None:
            return
        request_id, method, params, held, _ = pending
        await self.server.respond(request_id, _response(method, params, held, decision))
        if isinstance(decision, Deny) and decision.message:
            await self._give(decision.message, None)

    async def apply(self, spec: SessionSpec) -> None:
        self.spec = spec

    async def close(self) -> None:
        """Unload the thread from the app-server; it resumes on the next input."""
        if self._gone:
            return
        self._gone = True
        self.backend._sessions.pop(self.thread_id, None)
        for child in [child for child, parent in self.backend._parents.items() if parent == self.thread_id]:
            del self.backend._parents[child]
        try:
            await self._decline(self._take_children())
            await self.interrupt()
            await self.server.request("thread/unsubscribe", {"threadId": self.thread_id})
        except Exception as exc:  # a wedged or dead app-server: the session ends all the same
            logger.info("Aleph: closing codex thread %s: %s", self.thread_id, exc)
        finally:
            self._sink(Exit())

    def _take_children(self) -> list[tuple[Any, str, dict[str, Any], Ask]]:
        """The subagents' open requests. The runtime settles their cards with this thread's turn or
        process, and Codex does not: a subagent left unanswered would wait for good."""
        children = [pending[:4] for pending in self._asks.values() if pending[4]]
        self._asks = {key: pending for key, pending in self._asks.items() if not pending[4]}
        return children

    async def _decline(self, requests: list[tuple[Any, str, dict[str, Any], Ask]]) -> None:
        with contextlib.suppress(ConnectionError):  # the app-server is gone, and its subagents with it
            for request_id, method, params, ask in requests:
                await self.server.respond(request_id, _response(method, params, ask, Deny()))

    def idle_ok(self) -> bool:
        return self._turn is None

    def exited(self, error: Optional[str]) -> None:
        """The app-server ended."""
        if not self._gone:
            self._gone = True
            self._sink(Exit(error))

    # ── what the app-server sends ──────────────────────────────────────────────────────────────

    def subagent_notified(self, method: str, params: dict[str, Any]) -> None:
        """A subagent's items are not shown, but its file changes are kept: its edit approvals come to
        this session and show the files they change."""
        if method in ("item/started", "item/completed") and params["item"].get("type") == "fileChange":
            self._edit(params["item"]["id"], params["item"], "", method == "item/completed")

    def notified(self, method: str, params: dict[str, Any]) -> None:
        if method == "item/started" or method == "item/completed":
            self._item(params["item"], method == "item/completed")
        elif method in ("item/agentMessage/delta", "item/plan/delta"):
            self._grow(params["itemId"], 0, params["delta"])
        elif method == "item/reasoning/summaryTextDelta":
            self._grow(params["itemId"], int(params.get("summaryIndex") or 0), params["delta"], kind="reasoning")
        elif method == "item/commandExecution/outputDelta":
            self._streamed.add(params["itemId"])
            self._sink(Delta(params["itemId"], output=params["delta"]))
        elif method == "turn/plan/updated":
            steps = [{"text": step["step"], "status": _STATUS.get(step["status"], step["status"])}
                     for step in params.get("plan") or ()]
            self._sink(Item(f"plan-{params['turnId']}", "plan", fields={"steps": steps}))
        elif method == "thread/tokenUsage/updated":
            usage = params["tokenUsage"]
            if usage.get("modelContextWindow"):
                self._usage = {"used": usage["last"]["totalTokens"], "window": usage["modelContextWindow"]}
        elif method == "model/rerouted":
            self._sink(Item(f"model-{next(self._reroutes)}", "model", "completed", {"model": params["toModel"]}))
        elif method == "turn/started":
            self._turn = params["turn"]["id"]
        elif method == "turn/completed":
            turn = params["turn"]
            if self._turn == turn["id"]:
                self._turn = None
            status = turn.get("status") if turn.get("status") in ("completed", "interrupted", "failed") else "failed"
            error = (one_line((turn.get("error") or {}).get("message")) or None) if status == "failed" else None
            # What the turn left open ends with it; its own requests Codex settled itself.
            for open_items in (self._started, self._texts, self._streamed, self._edits, self._tools):
                open_items.clear()
            children = self._take_children()
            self._asks.clear()
            if children:
                self.backend.spawn(self._decline(children))
            self._sink(TurnEnd(status, error, self._usage))

    def _grow(self, key: str, index: int, delta: str, *, kind: Optional[str] = None) -> None:
        """A message's or reasoning's text grew; a reasoning shows only once it has text."""
        parts = self._texts.setdefault(key, [])
        parts.extend("" for _ in range(index + 1 - len(parts)))
        parts[index] += delta
        if kind is not None and key not in self._started:
            self._started.add(key)
            self._sink(Item(key, kind))
        self._sink(Delta(key, text="\n\n".join(part for part in parts if part)))

    def _relative(self, path: str) -> str:
        for folder in {self.spec.folder, os.path.realpath(self.spec.folder)}:
            if path.startswith(folder.rstrip(os.sep) + os.sep):
                return os.path.relpath(path, folder)
        return path

    def _item(self, raw: dict[str, Any], done: bool) -> None:
        key, kind = raw["id"], raw["type"]
        status = _STATUS.get(raw.get("status"), "completed") if done else "in_progress"
        if kind in ("agentMessage", "plan"):  # plan: the proposed plan a plan-mode turn ends with
            item = Item(key, "message", status, {"text": raw["text"]} if done else {})
        elif kind == "reasoning":
            text = "\n\n".join(part for part in raw.get("summary") or () if part)
            if not done or (not text and key not in self._started):
                return  # its summary deltas start it; one with no summary is not shown
            item = Item(key, "reasoning", status, {"text": text})
        elif kind == "commandExecution":
            command = _shown(raw["command"])
            if done:
                if key not in self._streamed and raw.get("aggregatedOutput"):
                    self._sink(Delta(key, output=raw["aggregatedOutput"]))
                self._streamed.discard(key)
            fields = {"tool": "execute", "title": title(command), "input": clip(command)}
            item = Item(key, "tool", status, {**fields, "exit_code": raw.get("exitCode")} if done else fields)
        elif kind == "fileChange":
            item = self._edit(key, raw, status, done)
        elif kind == "mcpToolCall":
            arguments = raw.get("arguments") if isinstance(raw.get("arguments"), dict) else {}
            if done:
                self._tools.pop(key, None)
            else:
                self._tools[key] = (raw["server"], raw["tool"])
            item = Item(key, "tool", status, _other(raw["tool"], arguments),
                        call=Call(raw["server"], raw["tool"], arguments))
        elif kind == "dynamicToolCall":
            item = Item(key, "tool", status, _other(raw["tool"], raw.get("arguments")))
        elif kind == "webSearch":
            action = raw.get("action") or {}
            query = raw.get("query") or action.get("query") or action.get("url") or ""
            item = Item(key, "tool", status, {"tool": "fetch", "title": title(query or "Web search"),
                                              "input": clip(query)})
        elif kind == "collabAgentToolCall":
            for child in raw.get("receiverThreadIds") or ():
                self.backend._parents[child] = self.thread_id
            prompt = raw.get("prompt") or ""
            item = Item(key, "tool", status, {"tool": "think", "title": title(prompt.strip().splitlines()[0]
                                                                              if prompt.strip() else raw["tool"]),
                                              "input": clip(prompt)})
        elif kind == "imageView":
            path = self._relative(raw["path"])
            item = Item(key, "tool", status, {"tool": "read", "title": title(path), "input": path})
        elif kind == "contextCompaction":
            item = Item(key, "compaction", status)
        else:  # the person's own input, hooks, a subagent's activity, sleeps, images, reviews
            return
        if not done:
            self._started.add(key)
        else:
            self._started.discard(key)
            self._texts.pop(key, None)
        self._sink(item)

    def _edit(self, key: str, raw: dict[str, Any], status: str, done: bool) -> Item:
        patches = []
        for change in raw.get("changes") or ():
            path, change_kind, diff = self._relative(change["path"]), change["kind"], change.get("diff") or ""
            if change_kind["type"] == "add":
                patches.append(Patch(path, _whole(diff, "+"), "added"))
            elif change_kind["type"] == "delete":
                patches.append(Patch(path, _whole(diff, "-"), "deleted"))
            elif change_kind.get("move_path"):
                patches.append(Patch(self._relative(change_kind["move_path"]), diff, "renamed", path))
            else:
                patches.append(Patch(path, diff, "modified"))
        tool = ("delete" if patches and all(patch.status == "deleted" for patch in patches)
                else "move" if any(patch.status == "renamed" for patch in patches) else "edit")
        paths = [patch.path for patch in patches]
        fields = {"tool": tool, "title": title(", ".join(paths)), "input": clip("\n".join(paths))}
        if done:
            self._edits.pop(key, None)
            return Item(key, "tool", status, fields)
        self._edits[key] = tuple(patches)
        return Item(key, "tool", status, fields, patches=tuple(patches))

    def requested(self, request_id: Any, method: str, params: dict[str, Any], *, child: bool) -> None:
        """A request of the app-server for this thread, or for a subagent it spawned (``child``: its
        items are not shown, so its asks hold none)."""
        item = None if child else params.get("itemId")
        key = f"r{request_id}"
        if method == _APPROVALS[0]:
            command = params.get("command") or ""
            offered = params.get("availableDecisions")
            ask = Ask(key, "command", item, params.get("reason") or "", _shown(command), params.get("cwd"),
                      grant=command if offered is None or "acceptForSession" in offered else None)
        elif method == _APPROVALS[1]:
            files = tuple({"path": patch.path, **({"old_path": patch.old_path} if patch.old_path else {}),
                           "added": _counts(patch.diff)[0], "removed": _counts(patch.diff)[1]}
                          for patch in self._edits.get(params.get("itemId"), ()))
            ask = Ask(key, "edit", item, params.get("reason") or "", files=files)
        elif method == _PERMISSIONS:
            ask = Ask(key, "permission", item, params.get("reason") or "",
                      grant=json.dumps(params.get("permissions"), sort_keys=True))
        elif method == _QUESTION:
            ask = Ask(key, "question", item, questions=tuple(_question(entry) for entry in params.get("questions") or ()))
        elif method == _ELICITATION and (params.get("_meta") or {}).get("codex_approval_kind") == "mcp_tool_call":
            # Codex asks for an MCP tool call as an elicitation: an approval, of the server's open call.
            call = None if child else next((tool_key for tool_key, (server, _) in reversed(self._tools.items())
                                            if server == params.get("serverName")), None)
            tool = self._tools[call][1] if call else str(params.get("serverName") or "")
            ask = Ask(key, "tool", call, tool, grant=f"{params.get('serverName')}/{tool}" if call else None)
        elif method == _ELICITATION:
            self.backend.spawn(self.server.respond(request_id, {"action": "decline", "content": None, "_meta": None}))
            return
        else:
            self.backend.spawn(self.server.refuse(request_id, f"{method} is not supported"))
            return
        self._asks[key] = (request_id, method, params, ask, child)
        self._sink(ask)


def _other(tool: str, arguments: Any) -> dict[str, Any]:
    shown = json.dumps(arguments, ensure_ascii=False) if arguments not in (None, {}) else ""
    return {"tool": "other", "title": title(tool), "input": clip(shown)}


def _question(entry: dict[str, Any]) -> Question:
    options, recommended = plain_options(option["label"] for option in entry.get("options") or ())
    return Question(entry.get("question") or entry.get("header") or "", options, False, recommended, entry["id"],
                    bool(entry.get("isSecret")))


def _response(method: str, params: dict[str, Any], ask: Ask, decision: Decision) -> dict[str, Any]:
    """A decision in the app-server's terms for the request it answers."""
    approve = isinstance(decision, Approve)
    session = approve and decision.session
    if method in _APPROVALS:
        if method == _APPROVALS[0] and ask.grant is None:
            session = False  # this command offers no session grant (``availableDecisions``)
        return {"decision": "acceptForSession" if session else "accept" if approve else "decline"}
    if method == _PERMISSIONS:
        return {"permissions": (params.get("permissions") or {}) if approve else {},
                "scope": "session" if session else "turn"}
    if method == _QUESTION:
        answers = decision.answers if isinstance(decision, Answers) else ()
        return {"answers": {question.id: {"answers": list(answer)}
                            for question, answer in zip(ask.questions, answers) if answer}}
    return {"action": "accept" if approve else "decline", "content": {} if approve else None,
            "_meta": {"persist": "session"} if session else None}
