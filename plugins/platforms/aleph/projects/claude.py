"""Claude Code as a projects backend: its ``claude -p`` stream-json CLI, one process per thread.

The process reads user messages and control responses on stdin and writes newline-delimited frames on
stdout: ``stream_event`` (the model's blocks as they grow), ``assistant`` (a message's blocks whole),
``user`` (tool results, and with ``--replay-user-messages`` the echo of each input once a turn takes
it), ``system`` (init, compaction, a model fallback), ``result`` (the turn's end) and the control
protocol both ways: ``can_use_tool`` asks (``--permission-prompt-tool stdio``), ``interrupt``,
``set_model``, ``set_permission_mode``. ``ClaudeSession`` maps them to ``backend.py``'s events.

Everything runs with ``HOME`` set to the plugin's projects home, so the CLI's config, sign-in and
transcripts live there (never ``CLAUDE_CONFIG_DIR``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shlex
import shutil
import signal
import sys
import tempfile
import time
from collections import deque
from typing import Any, Optional

from plugins.platforms.aleph.projects import actions, child, git
from plugins.platforms.aleph.projects.backend import (
    TIERS,
    Accepted, Answers, Approve, Ask, BackendInfo, Call, Decision, Delta, Deny, Exit, Item, Model, Question,
    SessionSpec, Sink, TurnEnd, clip, command_rule, label, mcp_title, plain_options, title)
from plugins.platforms.aleph.projects import process
from plugins.platforms.aleph.projects.process import STREAM_LIMIT, end, kill, signal_group
from plugins.platforms.aleph.projects.service import one_line

logger = logging.getLogger(__name__)

CONTROL_SECONDS = 15.0
PROBE_SECONDS = 20.0
# The person's approval modes as the CLI's permission modes; plan mode overrides either.
_MODES = {"ask": "default", "auto_review": "auto"}
_KINDS = {"Bash": "execute", "Edit": "edit", "MultiEdit": "edit", "Write": "edit", "NotebookEdit": "edit",
          "Read": "read", "Grep": "search", "Glob": "search", "WebFetch": "fetch", "WebSearch": "fetch",
          "Task": "think", "Agent": "think"}
# Tools whose input the runtime applies to the file on disk for the edit's diff (NotebookEdit's is cells).
_DIFFED = frozenset({"Edit", "MultiEdit", "Write"})
_EDIT_TOOLS = frozenset({"Edit", "MultiEdit", "Write", "NotebookEdit"})
# The task list: TodoWrite rewrites it whole; the Task tools create and update one task at a time.
_PLAN_TOOLS = frozenset({"TodoWrite", "TaskCreate", "TaskUpdate"})
# No items: ToolSearch only loads deferred tool schemas, TaskList and TaskGet only read the task list;
# AskUserQuestion and ExitPlanMode are requests.
_HIDDEN = frozenset({"ToolSearch", "TaskList", "TaskGet", "AskUserQuestion", "ExitPlanMode"})
_EXIT_CODE = re.compile(r"\AExit code (\d+)\n?")
# What the CLI says when ``--resume`` names a session it no longer has.
_NO_SESSION = "No conversation found"


def _permission_mode(spec: SessionSpec) -> str:
    return "plan" if spec.plan else _MODES.get(spec.approval, "default")


def _environment(home: str, extra: dict[str, str]) -> dict[str, str]:
    env = process.environment(home, **extra)
    env.pop("CLAUDE_CONFIG_DIR", None)  # it would move the config out of HOME
    return env


def _text_of(content: Any) -> str:
    if isinstance(content, list):
        return "".join(str(block.get("text") or "") for block in content
                       if isinstance(block, dict) and block.get("type") == "text")
    return content if isinstance(content, str) else ""


class ClaudeBackend:
    """Claude Code, found as ``command`` on PATH, run with ``HOME`` = ``home``."""

    id = "claude"
    name = "Claude Code"
    asks_to_leave_plan = True

    def __init__(self, home: str | os.PathLike, command: str = "claude") -> None:
        self.home = str(home)
        self.command = command

    def _info(self, available: bool, reason: Optional[str] = None, message: Optional[str] = None,
              **fields: Any) -> BackendInfo:
        return BackendInfo(self.id, self.name, available, reason, message, **fields)

    async def probe(self) -> BackendInfo:
        path = shutil.which(self.command)
        if path is None:
            return self._info(False, "not_installed", f"{self.command} is not on PATH")
        try:
            if not (await self._auth(path)).get("loggedIn"):
                return self._info(False, "signed_out", "Claude Code is not signed in on the server")
            catalog, auto = await self._catalog(path)
        except Exception as exc:
            logger.warning("Aleph: probing Claude Code failed: %s", exc)
            return self._info(False, "error", one_line(exc))
        listed = [entry for entry in catalog if isinstance(entry, dict) and entry.get("value")]
        models = tuple(Model(str(entry["value"]), str(entry.get("displayName") or entry["value"]),
                             tuple(entry.get("supportedEffortLevels") or ()) if entry.get("supportsEffort") else (),
                             tiers=TIERS if entry.get("supportsFastMode") else (),
                             default_tier="standard" if entry.get("supportsFastMode") else None)
                       for entry in listed if entry["value"] != "default")
        # "default" is the CLI's own selector: what null runs, named here by the model it resolves to.
        resolved = next((entry.get("resolvedModel") for entry in listed if entry["value"] == "default"), None)
        default = next((str(entry["value"]) for entry in listed
                        if entry["value"] != "default" and resolved and entry.get("resolvedModel") == resolved), None)
        return self._info(True, models=models, default_model=default,
                          approvals=("ask", "auto_review") if auto else ("ask",))

    async def _auth(self, path: str) -> dict[str, Any]:
        proc = await asyncio.create_subprocess_exec(
            path, "auth", "status", "--json", stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL, env=_environment(self.home, {}), start_new_session=True)
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), PROBE_SECONDS)
        except TimeoutError:
            await kill(proc)
            raise RuntimeError("claude auth status did not answer") from None
        try:
            status = json.loads(out or b"{}")
        except ValueError:
            raise RuntimeError("claude auth status did not answer in JSON") from None
        return status if isinstance(status, dict) else {}

    async def _catalog(self, path: str) -> tuple[list[Any], bool]:
        """The account's models, from the initialize handshake, and whether it takes the ``auto``
        permission mode: one short process that never starts a model turn, kept apart from project
        hooks, tools and MCP servers, its stdin closed after the two requests."""
        requests = [{"type": "control_request", "request_id": "models",
                     "request": {"subtype": "initialize", "hooks": None}},
                    {"type": "control_request", "request_id": "auto",
                     "request": {"subtype": "set_permission_mode", "mode": "auto"}}]
        with tempfile.TemporaryDirectory(prefix="aleph-claude-probe-") as cwd:
            proc = await asyncio.create_subprocess_exec(
                path, "-p", "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
                "--no-session-persistence", "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
                "--settings", '{"disableAllHooks":true}', "--setting-sources", "user", "--disable-slash-commands",
                "--no-chrome", "--tools", "", cwd=cwd, stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL, limit=STREAM_LIMIT,
                env=_environment(self.home, {}), start_new_session=True)
            stdin = "".join(json.dumps(request) + "\n" for request in requests).encode()
            try:
                out, _ = await asyncio.wait_for(proc.communicate(stdin), PROBE_SECONDS)
            except TimeoutError:
                await kill(proc)
                raise RuntimeError("Claude Code did not answer its initialize request") from None
        responses: dict[str, dict[str, Any]] = {}
        for line in out.splitlines():
            try:
                frame = json.loads(line)
            except ValueError:
                continue
            if isinstance(frame, dict) and frame.get("type") == "control_response":
                response = frame.get("response") or {}
                responses[str(response.get("request_id"))] = response
        models = responses.get("models", {})
        catalog = (models.get("response") or {}).get("models")
        if models.get("subtype") != "success" or not isinstance(catalog, list):
            raise RuntimeError(f"Claude Code did not list its models (exit code {proc.returncode})")
        auto = responses.get("auto", {})
        return catalog, auto.get("subtype") == "success" and (auto.get("response") or {}).get("mode") == "auto"

    async def open(self, spec: SessionSpec, sink: Sink) -> ClaudeSession:
        session = ClaudeSession(self, spec, sink)
        await session.start()
        return session

    async def stop(self) -> None:
        pass  # one process per session: nothing is shared between them


class ClaudeSession:
    """One thread's ``claude`` process. Its events go to ``sink`` from its stdout reader task."""

    def __init__(self, backend: ClaudeBackend, spec: SessionSpec, sink: Sink) -> None:
        self.backend, self.spec, self.sink = backend, spec, sink
        self._session_id = spec.resume
        self._resume = spec.resume  # what the next process resumes
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._reader: Optional[asyncio.Task] = None
        self._write_lock = asyncio.Lock()
        self._stderr: deque[str] = deque(maxlen=5)
        self._closing = False
        self._initialized = False  # this process reported its system init
        self._stale = False  # this process found no conversation to resume
        self._mode = _permission_mode(spec)  # the process's permission mode, as last set
        self._expected: Optional[str] = None  # the model id the selection resolves to
        self._controls: dict[str, asyncio.Future] = {}
        self._control_seq = 0
        # Inputs written, by key, until their echo says a turn took them.
        self._unechoed: list[tuple[str, str]] = []
        self._asks: dict[str, dict[str, Any]] = {}  # open can_use_tool requests, by request id
        self._blocks: dict[int, list[Any]] = {}  # the streaming message's text blocks: [key, kind, text]
        self._message = ""  # the streaming message's id
        self._streamed: set[str] = set()  # this turn's message ids that came as stream events
        self._tools: dict[str, str] = {}  # open tool items: tool_use id -> ACP kind
        self._quiet: set[str] = set()  # tool_use ids that are no items
        # The task list of the Task tools, by the CLI's own ids; a TaskCreate is known by its tool_use id
        # until its result ("Task #N created") names it.
        self._tasks: dict[str, dict[str, str]] = {}
        self._creates: dict[str, str] = {}
        # This turn's.
        self._interrupting = False
        self._fallback = False  # the CLI fell back to another model
        self._model_told = False  # a model item was reported
        self._model = ""  # the model that answered last
        self._used: Optional[int] = None
        self._running = False  # a turn is open
        # When the latest turn started: a Stop-hook report from then on tells of its end.
        self._since = time.time()

    @property
    def session_id(self) -> Optional[str]:
        return self._session_id

    # ── the process ────────────────────────────────────────────────────────────────────────────

    def _command(self) -> tuple[list[str], dict[str, str]]:
        spec = self.spec
        argv = [self.backend.command, "-p", "--input-format", "stream-json", "--output-format", "stream-json",
                "--verbose", "--include-partial-messages", "--replay-user-messages",
                "--permission-prompt-tool", "stdio", "--permission-mode", self._mode,
                # Without --model a resumed session keeps the model it last ran: "default" is the CLI's own.
                "--model", spec.model or "default"]
        if spec.effort:
            argv += ["--effort", spec.effort]
        extra: dict[str, str] = {}
        if spec.mcp is not None:
            server = spec.mcp
            config = {"command": server.command, "args": list(server.args),
                      "env": {key: "${" + key + "}" for key in server.env}}
            extra.update(server.env)
            argv += ["--mcp-config", json.dumps({"mcpServers": {server.name: config}}),
                     "--allowedTools", *(f"mcp__{server.name}__{tool}" for tool in server.tools)]
        # Fast mode is set only at launch, and stated either way: the account's own default is not the thread's.
        settings: dict[str, Any] = {"fastMode": spec.tier == "fast"}
        if spec.stop_report:
            hook = {"type": "command", "timeout": 10,
                    "command": shlex.join([sys.executable, "-I", os.path.abspath(child.__file__), "stop"])}
            settings["hooks"] = {"Stop": [{"hooks": [hook]}]}
            extra[child.STOP_ENV] = spec.stop_report
        argv += ["--settings", json.dumps(settings)]
        if spec.memory:
            # Outside the folder: without the CLI's access to it, every memory write would ask the person.
            argv += ["--add-dir", spec.memory]
        if spec.instructions:
            argv += ["--append-system-prompt", spec.instructions]
        if self._resume:
            argv += ["--resume", self._resume]
        return argv, _environment(self.backend.home, extra)

    async def start(self) -> None:
        argv, env = self._command()
        path = shutil.which(argv[0], path=env.get("PATH"))
        if path is None:
            raise RuntimeError(f"{argv[0]} is not on PATH")
        if self._closing:
            raise RuntimeError("Claude Code is closing")
        proc = await asyncio.create_subprocess_exec(
            path, *argv[1:], cwd=self.spec.folder, env=env, stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, limit=STREAM_LIMIT,
            start_new_session=True)
        if self._closing:  # closed while it spawned (a respawn, a restart without the stale session)
            await end(proc)
            raise RuntimeError("Claude Code is closing")
        self._proc = proc
        self._initialized = self._stale = False
        self._expected = None  # its system init says what the selection resolves to
        self._blocks.clear()
        self._stderr.clear()  # what a process that dies says of itself, not of the one before
        drain = asyncio.create_task(self._read_stderr(proc), name="aleph-claude-stderr")
        self._reader = asyncio.create_task(self._read(proc, drain), name="aleph-claude-stdout")

    async def _read_stderr(self, proc: asyncio.subprocess.Process) -> None:
        while True:
            try:
                line = await proc.stderr.readline()
            except ValueError:  # a line past STREAM_LIMIT, dropped: draining goes on, or the CLI blocks
                continue
            if not line:
                return
            text = line.decode("utf-8", "replace").strip()
            if text:
                self._stderr.append(text)
                logger.debug("Aleph: claude stderr: %s", text)

    async def _read(self, proc: asyncio.subprocess.Process, drain: asyncio.Task) -> None:
        try:
            while line := await proc.stdout.readline():
                try:
                    frame = json.loads(line)
                except ValueError:
                    continue
                if isinstance(frame, dict):
                    try:
                        await self._frame(frame)
                    except Exception:
                        logger.exception("Aleph: handling a Claude Code %s frame failed", frame.get("type"))
        except Exception as exc:
            logger.warning("Aleph: reading Claude Code failed: %s", exc)
            signal_group(proc, signal.SIGKILL)
        await self._exited(proc, drain)

    async def _exited(self, proc: asyncio.subprocess.Process, drain: asyncio.Task) -> None:
        code = await proc.wait()
        await asyncio.wait([drain], timeout=1.0)  # its last stderr line says why it ended
        if proc is not self._proc:
            return  # replaced on purpose (a respawn): the new process carries on
        for future in self._controls.values():
            if not future.done():
                future.set_exception(RuntimeError("Claude Code exited before it answered"))
        self._controls.clear()
        self._asks.clear()
        if self._stale and not self._closing:
            # The session to resume is gone: start afresh and hand the new process what the old never took.
            logger.warning("Aleph: Claude Code no longer has session %s; starting a new one", self._resume)
            self._resume = self._session_id = None
            try:
                await self.start()
                for key, text in list(self._unechoed):
                    await self._write(self._user(text))
                return
            except Exception as exc:
                if self._proc is not None and self._proc is not proc:
                    await end(self._proc)
                self._proc = None
                self.sink(Exit() if self._closing else Exit(one_line(f"Claude Code could not start again: {exc}")))
                return
        self._proc = None
        if self._closing:
            self.sink(Exit())
            return
        why = self._stderr[-1] if self._stderr else None
        self.sink(Exit(one_line(f"Claude Code exited: {why}") if why else f"Claude Code exited with code {code}."))

    async def _write(self, frame: dict[str, Any]) -> None:
        proc = self._proc
        if proc is None or proc.returncode is not None:
            raise RuntimeError("Claude Code is not running")
        async with self._write_lock:
            proc.stdin.write(json.dumps(frame, ensure_ascii=False).encode() + b"\n")
            await proc.stdin.drain()

    def _user(self, text: str) -> dict[str, Any]:
        return {"type": "user", "message": {"role": "user", "content": text}, "parent_tool_use_id": None,
                "session_id": self._session_id or "default"}

    async def _control(self, request: dict[str, Any]) -> dict[str, Any]:
        """A control request of ours, answered by the CLI without a model turn."""
        self._control_seq += 1
        request_id = f"aleph-{self._control_seq}"
        future = self._controls[request_id] = asyncio.get_running_loop().create_future()
        try:
            await self._write({"type": "control_request", "request_id": request_id, "request": request})
            response = await asyncio.wait_for(future, CONTROL_SECONDS)
        except TimeoutError:
            raise RuntimeError(f"Claude Code did not answer {request['subtype']}") from None
        finally:
            self._controls.pop(request_id, None)
        if response.get("subtype") != "success":
            raise RuntimeError(one_line(response.get("error") or f"Claude Code refused {request['subtype']}"))
        return response.get("response") or {}

    async def _reply(self, request_id: str, response: dict[str, Any]) -> None:
        await self._write({"type": "control_response",
                           "response": {"subtype": "success", "request_id": request_id, "response": response}})

    # ── BackendSession ─────────────────────────────────────────────────────────────────────────

    def _turn_started(self) -> None:
        if not self._running:
            self._running, self._interrupting, self._since = True, False, time.time()

    async def send(self, key: str, text: str) -> None:
        self._turn_started()
        self._unechoed.append((key, text))
        try:
            await self._write(self._user(text))
        except (RuntimeError, ConnectionError) as exc:
            # A process that found no session to resume exits at once: its restart writes this again.
            reader = self._reader
            if reader is not None:
                await asyncio.shield(reader)
            if self._proc is None or self._proc.returncode is not None:
                self._unechoed = [(one, said) for one, said in self._unechoed if one != key]
                raise RuntimeError("Claude Code is not running") from exc

    async def interrupt(self) -> None:
        if not self._running:
            return  # its result is in: the turn's end is on its way, and the next turn is not to stop
        self._interrupting = True
        await self._write({"type": "control_request", "request_id": f"aleph-interrupt-{time.monotonic_ns()}",
                           "request": {"subtype": "interrupt"}})

    async def answer(self, ask: str, decision: Decision) -> None:
        request = self._asks.pop(ask, None)
        if request is None:
            return  # the CLI withdrew it, or its process is gone
        name, original = str(request.get("tool_name") or ""), request.get("input") or {}
        if isinstance(decision, Deny):
            await self._reply(ask, {"behavior": "deny", "message": decision.message or (
                "The person kept planning: refine the plan." if name == "ExitPlanMode"
                else "The person declined this.")})
            return
        if isinstance(decision, Answers):
            labels = request["_labels"]
            answers = {question["question"]: ", ".join(labels[index].get(label, label) for label in picked)
                       for index, (question, picked) in enumerate(zip(original.get("questions") or [], decision.answers))
                       if picked}
            await self._reply(ask, {"behavior": "allow", "updatedInput": {**original, "answers": answers}})
            return
        response: dict[str, Any] = {"behavior": "allow", "updatedInput": original}
        if name == "ExitPlanMode":
            # Leave plan mode into the thread's own approval mode, for the rest of the process.
            self._mode = _MODES.get(self.spec.approval, "default")
            response["updatedPermissions"] = [{"type": "setMode", "mode": self._mode, "destination": "session"}]
        elif isinstance(decision, Approve) and decision.session:
            update = _session_grant(request)
            if update is not None:
                response["updatedPermissions"] = [update]
        await self._reply(ask, response)

    async def apply(self, spec: SessionSpec) -> None:
        """``self.spec`` stays what the process runs until each change took: a change refused is tried
        again by the next ``apply``."""
        if spec.effort != self.spec.effort or spec.tier != self.spec.tier:
            # The CLI sets effort and fast mode only at launch: the same session goes on in a new process.
            self.spec, self._mode = spec, _permission_mode(spec)
            await self._respawn()
            return
        if spec.model != self.spec.model:
            resolved = await self._control({"subtype": "set_model", "model": spec.model or "default"})
            self._expected = resolved.get("model") or None
        mode = _permission_mode(spec)
        if mode != self._mode:
            await self._control({"subtype": "set_permission_mode", "mode": mode})
            self._mode = mode
        self.spec = spec

    async def _respawn(self) -> None:
        self._resume = self._session_id
        proc, self._proc = self._proc, None
        if proc is not None:
            await end(proc)
        try:
            await self.start()
        except Exception as exc:
            # The old process's reader stays silent (it was replaced): the session ends here.
            self.sink(Exit() if self._closing else Exit(one_line(f"Claude Code could not start again: {exc}")))
            raise

    async def close(self) -> None:
        self._closing = True  # from here no process starts (``start``)
        proc = self._proc
        if proc is None:
            return
        await end(proc)
        if self._reader is not None:
            await asyncio.shield(self._reader)

    def idle_ok(self) -> bool:
        return bool(self.spec.stop_report) and child.idle(self.spec.stop_report, self._since)

    # ── frames ─────────────────────────────────────────────────────────────────────────────────

    async def _frame(self, frame: dict[str, Any]) -> None:
        kind = frame.get("type")
        if kind == "control_response":
            response = frame.get("response") or {}
            future = self._controls.get(str(response.get("request_id")))
            if future is not None and not future.done():
                future.set_result(response)
        elif kind == "control_request":
            await self._control_request(str(frame.get("request_id") or ""), frame.get("request") or {})
        elif kind == "control_cancel_request":
            self._asks.pop(str(frame.get("request_id")), None)
        elif frame.get("parent_tool_use_id"):
            return  # a subagent's own traffic: its spawning call is the item
        elif kind == "system":
            self._system(frame)
        elif kind == "stream_event":
            self._stream(frame.get("event") or {})
        elif kind == "assistant":
            self._assistant(frame.get("message") or {})
        elif kind == "user":
            self._user_frame(frame)
        elif kind == "result":
            self._result(frame)

    def _system(self, frame: dict[str, Any]) -> None:
        subtype = frame.get("subtype")
        if subtype == "init":
            self._initialized = True
            self._session_id = str(frame.get("session_id") or "") or self._session_id
            self._expected = self._expected or str(frame.get("model") or "") or None
        elif subtype == "compact_boundary":
            self.sink(Item(f"compaction:{frame.get('uuid') or time.monotonic_ns()}", "compaction", "completed"))
        elif subtype == "model_refusal_fallback":
            self._fallback = True

    def _saw_model(self, model: str) -> None:
        """A message came from ``model``: once a turn, a ``model`` item when it is not the one selected
        (the CLI fell back, or redirected without saying)."""
        if not model or model == "<synthetic>":
            return
        self._model = model
        if self._model_told or not (self._fallback or (self._expected and not _same_model(self._expected, model))):
            return
        self._model_told = True
        self.sink(Item(f"model:{model}:{time.monotonic_ns()}", "model", "completed", {"model": model}))

    def _stream(self, event: dict[str, Any]) -> None:
        kind = event.get("type")
        if kind == "message_start":
            self._turn_started()  # the CLI may wake by itself (a background task ended)
            message = event.get("message") or {}
            self._message = str(message.get("id") or time.monotonic_ns())
            self._streamed.add(self._message)
            self._blocks.clear()
            self._saw_model(str(message.get("model") or ""))
        elif kind == "content_block_start":
            block = event.get("content_block") or {}
            item = {"text": "message", "thinking": "reasoning"}.get(block.get("type"))
            if item is not None:
                text = str(block.get(block["type"]) or "")
                key = f"{self._message}:{event.get('index')}"
                self._blocks[event.get("index")] = [key, item, text]
                self.sink(Item(key, item, fields={"text": text}))
        elif kind == "content_block_delta":
            open_block = self._blocks.get(event.get("index"))
            delta = event.get("delta") or {}
            grown = delta.get("text") if delta.get("type") == "text_delta" else (
                delta.get("thinking") if delta.get("type") == "thinking_delta" else None)
            if open_block is not None and grown:
                open_block[2] += grown
                self.sink(Delta(open_block[0], text=open_block[2]))
        elif kind == "content_block_stop":
            open_block = self._blocks.pop(event.get("index"), None)
            if open_block is not None:
                self.sink(Item(open_block[0], open_block[1], "completed", {"text": open_block[2]}))
        elif kind == "message_delta":
            usage = event.get("usage") or {}
            used = sum(int(usage.get(key) or 0) for key in (
                "input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens"))
            self._used = used or self._used

    def _assistant(self, message: dict[str, Any]) -> None:
        """A message's blocks whole: tool calls start here, their input complete (a stream event's
        content_block_start has it empty). A message that never streamed (the CLI's own, as an API
        error) brings its text here."""
        streamed = str(message.get("id") or "") in self._streamed
        if not streamed:
            self._saw_model(str(message.get("model") or ""))
        for index, block in enumerate(message.get("content") or []):
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":
                self._tool_started(block)
            elif block.get("type") == "text" and not streamed and block.get("text"):
                self.sink(Item(f"{message.get('id')}:{index}", "message", "completed", {"text": block["text"]}))

    def _tool_started(self, block: dict[str, Any]) -> None:
        key, name = str(block.get("id") or ""), str(block.get("name") or "tool")
        data = block.get("input") if isinstance(block.get("input"), dict) else {}
        if not key or key in self._tools or key in self._quiet:
            return
        if name in _PLAN_TOOLS:
            self._quiet.add(key)
            self.sink(Item(key, "plan", fields={"steps": self._plan(name, data, key)}))
            return
        if name in _HIDDEN:
            self._quiet.add(key)
            return
        call = None
        if name.startswith("mcp__") and name.count("__") >= 2:
            _, server, tool = name.split("__", 2)
            call = Call(server, tool, data)
        kind = _KINDS.get(name, "other")
        text = str(data.get("command") or "") if kind == "execute" else json.dumps(data, ensure_ascii=False)
        self._tools[key] = kind
        fields = {"tool": kind, "title": mcp_title(call.server, call.tool) if call else title(self._title(name, data)),
                  "input": clip(text)}
        if kind == "execute":
            fields.update(label=label(data.get("description")), actions=actions.classify(text, self._relative))
        self.sink(Item(key, "tool", fields={field: value for field, value in fields.items() if value is not None},
                       call=call, arguments=data if name in _DIFFED else None))

    def _title(self, name: str, data: dict[str, Any]) -> str:
        for field in ("command", "file_path", "notebook_path", "pattern", "url", "query", "description"):
            value = data.get(field)
            if isinstance(value, str) and value:
                return self._relative(value) if field.endswith("_path") else value
        return name

    def _relative(self, path: str) -> str:
        folder, real = os.path.realpath(self.spec.folder), os.path.realpath(path)
        return os.path.relpath(real, folder) if real.startswith(folder.rstrip(os.sep) + os.sep) else path

    def _user_frame(self, frame: dict[str, Any]) -> None:
        message = frame.get("message") or {}
        content = message.get("content")
        if frame.get("isReplay"):
            # The echo of an input: a turn took it. Matched by its text, oldest first.
            text = _text_of(content)
            match = next((index for index, (_, said) in enumerate(self._unechoed) if said == text), None)
            if match is not None:
                key, _ = self._unechoed.pop(match)
                self.sink(Accepted(key))
            return
        if not isinstance(content, list):
            return
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                self._tool_ended(block, frame.get("tool_use_result"))

    def _tool_ended(self, block: dict[str, Any], result: Any) -> None:
        key = str(block.get("tool_use_id") or "")
        if key in self._creates:
            self._bind_task(key, _text_of(block.get("content")))
        self._quiet.discard(key)
        kind = self._tools.pop(key, None)
        if kind is None:
            return
        failed = bool(block.get("is_error"))
        fields: dict[str, Any] = {}
        if kind == "execute":
            text = _text_of(block.get("content"))
            code = _EXIT_CODE.match(text) if failed else None
            # Claude Code reports no exit status: 0 on success, the code its failure text names, else none.
            fields["exit_code"] = int(code.group(1)) if code else (None if failed else 0)
            if isinstance(result, dict) and ("stdout" in result or "stderr" in result):
                out, err = str(result.get("stdout") or ""), str(result.get("stderr") or "")
                output = out + ("\n" if out and err and not out.endswith("\n") else "") + err
            else:
                # A failed command's result is "Exit code N" and its output; a refused one's is no output.
                output = text[code.end():] if code else ""
            if output:
                self.sink(Delta(key, output=output))
        self.sink(Item(key, "tool", "failed" if failed else "completed", fields))

    def _result(self, frame: dict[str, Any]) -> None:
        errors = [str(error) for error in frame.get("errors") or []]
        if self._resume and not self._initialized and any(_NO_SESSION in error for error in errors):
            self._stale = True  # the process exits next; ``_exited`` starts a fresh session
            return
        self._session_id = str(frame.get("session_id") or "") or self._session_id
        windows = {name: int(usage.get("contextWindow") or 0)
                   for name, usage in (frame.get("modelUsage") or {}).items() if isinstance(usage, dict)}
        window = windows.get(self._model) or max(windows.values(), default=0)
        usage = {"used": self._used, "window": window} if self._used and window else None
        if self._interrupting:
            end = TurnEnd("interrupted", usage=usage)
        elif frame.get("is_error"):
            why = frame.get("result") if isinstance(frame.get("result"), str) else None
            end = TurnEnd("failed", one_line(why or (errors[0] if errors else "Claude Code reported an error.")),
                          usage)
        else:
            end = TurnEnd("completed", usage=usage)
        self._running = self._interrupting = self._fallback = self._model_told = False
        self._streamed.clear()
        self._tools.clear()
        self._blocks.clear()
        self._asks.clear()  # a request the turn left open can take no answer now (the runtime cancels its card)
        self.sink(end)

    # ── requests ───────────────────────────────────────────────────────────────────────────────

    async def _control_request(self, request_id: str, request: dict[str, Any]) -> None:
        subtype = request.get("subtype")
        if subtype == "elicitation":
            await self._reply(request_id, {"action": "decline"})
            return
        if subtype != "can_use_tool":
            await self._write({"type": "control_response", "response": {
                "subtype": "error", "request_id": request_id, "error": f"Unsupported control request {subtype}"}})
            return
        name = str(request.get("tool_name") or "")
        data = request.get("input") if isinstance(request.get("input"), dict) else {}
        item = str(request.get("tool_use_id") or "") or None
        self._asks[request_id] = request
        if name == "AskUserQuestion":
            questions, labels = [], []
            for entry in data.get("questions") or []:
                originals = [str(option.get("label") or "") for option in entry.get("options") or []
                             if isinstance(option, dict)]
                options, recommended = plain_options(originals)
                questions.append(Question(str(entry.get("question") or ""), options,
                                          bool(entry.get("multiSelect")), recommended))
                labels.append(dict(zip(options, originals)))
            request["_labels"] = labels
            self.sink(Ask(request_id, "question", questions=tuple(questions)))
            return
        if name == "ExitPlanMode":
            self.sink(Item(f"{item or request_id}:plan", "message", "completed", {"text": str(data.get("plan") or "")}))
            self.sink(Ask(request_id, "plan"))
            return
        reason = str(request.get("decision_reason") or "")
        if name == "Bash":
            command = str(data.get("command") or "")
            self.sink(Ask(request_id, "command", item, reason or str(data.get("description") or ""), command,
                          self.spec.folder, rule=command_rule(command)))
        elif name in _EDIT_TOOLS:
            files = await asyncio.to_thread(git.edit_files, self.spec.folder, data, ()) if name in _DIFFED else []
            self.sink(Ask(request_id, "edit", item, reason, files=tuple(
                {key: entry[key] for key in ("path", "old_path", "added", "removed") if entry.get(key) is not None}
                for entry in files)))
        else:
            self.sink(Ask(request_id, "tool", item, str(request.get("display_name") or name), rule=name))

    # ── the task list ──────────────────────────────────────────────────────────────────────────

    def _plan(self, name: str, data: dict[str, Any], key: str) -> list[dict[str, str]]:
        """The task list after a plan tool call, as plan steps."""
        if name == "TodoWrite":
            return [{"text": " ".join(str(todo.get("content") or todo.get("activeForm") or "").split()),
                     "status": str(todo.get("status") or "pending")}
                    for todo in data.get("todos") or [] if isinstance(todo, dict)]
        status = str(data.get("status") or "").strip()
        subject = " ".join(str(data.get("subject") or "").split())
        if name == "TaskCreate" and subject:
            if self._tasks and all(task["status"] == "completed" for task in self._tasks.values()):
                self._tasks.clear()  # every task done: this one starts a new list
            self._tasks[f"pending:{key}"] = {"text": subject, "status": "pending"}
            self._creates[key] = f"pending:{key}"
        elif name == "TaskUpdate":
            task_id = str(data.get("taskId") or "")
            task = self._tasks.get(task_id)
            if status == "deleted":
                self._tasks.pop(task_id, None)
            elif task is not None:
                task.update({key: value for key, value in (("status", status), ("text", subject)) if value})
            elif task_id and subject:
                # A task from before this process: its update names it well enough.
                self._tasks[task_id] = {"text": subject, "status": status or "pending"}
        return [dict(task) for task in self._tasks.values()]

    def _bind_task(self, key: str, result: str) -> None:
        """A TaskCreate's result names the task's id ("Task #N created"): its entry goes by it, in place."""
        provisional = self._creates.pop(key)
        found = re.search(r"Task #(\d+)", result)
        if found is None or provisional not in self._tasks:
            return
        self._tasks = {(found.group(1) if task_id == provisional else task_id): task
                       for task_id, task in self._tasks.items()}


def _session_grant(request: dict[str, Any]) -> Optional[dict[str, Any]]:
    """The CLI's own rule for the rest of the session, as Always Allow gives it: a Bash command's
    ``command_rule`` (``npm test:*``, or the command exactly); another tool, the CLI's suggestion. Never
    saved to the folder's own settings: the server keeps the lasting rule."""
    if request.get("tool_name") == "Bash":
        kind, _, words = command_rule(str((request.get("input") or {}).get("command") or "")).partition(":")
        return {"type": "addRules", "rules": [{"toolName": "Bash", "ruleContent": f"{words}:*" if kind == "prefix" else words}],
                "behavior": "allow", "destination": "session"}
    suggestions = [one for one in request.get("permission_suggestions") or [] if isinstance(one, dict)]
    preferred = sorted(suggestions, key=lambda one: one.get("type") == "setMode")
    return {**preferred[0], "destination": "session"} if preferred else None


def _same_model(expected: str, actual: str) -> bool:
    """Whether ``actual`` is the model ``expected`` names: the same id, or its dated release."""
    expected, actual = expected.split("[", 1)[0].strip(), actual.split("[", 1)[0].strip()
    return expected == actual or bool(not re.search(r"-\d{8}$", expected)
                                      and re.fullmatch(re.escape(expected) + r"-\d{8}", actual))
