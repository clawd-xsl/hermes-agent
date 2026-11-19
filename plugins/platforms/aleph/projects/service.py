"""Projects on Aleph (protocol "Projects"): the calls on projects and their threads, the folder
browser, the backends on offer, and what a reconnecting phone needs of them.

A project with no folder of its own runs in a working folder the server gives it under the projects
root (``folder``), for its whole life.

``Projects`` is all the adapter sees of them. Entity changes are logged through the adapter's
``_log``, like any event, and their rows follow from the log (``ProjectTables``). They are checked and
logged under one lock, so an id is taken once however calls race.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Optional

from plugins.platforms.aleph.projects import git
from plugins.platforms.aleph.projects.backend import APPROVALS, TIERS, Ask, Backend, BackendInfo, Item
from plugins.platforms.aleph.projects.output import Outputs
from plugins.platforms.aleph.projects.store import ThreadRow
from plugins.platforms.aleph.server import FAILED, INVALID_PARAMS, RpcError, event_frame

if TYPE_CHECKING:
    from plugins.platforms.aleph.adapter import AlephAdapter
    from plugins.platforms.aleph.projects.requests import RequestDesk
    from plugins.platforms.aleph.projects.session import Session
    from plugins.platforms.aleph.server import Connection

logger = logging.getLogger(__name__)

DEFAULT_PROJECTS_ROOT = "/opt/data/projects"
PROBE_SECONDS = 300.0
MAX_ID = 128
MAX_NAME = 100  # code points, as a chat's title
MAX_FOLDERS = 1000
HOSTS = ("local",)
WORKING = ".aleph"  # under the projects root: the working folders of projects with no folder (hidden from fs.list)
MAX_TEXT_CHARS = 64000
MAX_COMMENTS = 200
MAX_LINE_CHARS = 300
RESTARTED = "The agent restarted"
DELIVERIES = ("steer", "after_turn")
_PLAN_KEY = "plan"  # the key of the server's own plan request: no backend knows it
LOOK_SECONDS = 2.0  # how often a running turn's working tree is looked at while a phone is connected
MAX_ITEM_DIFF_BYTES = 2 * 1024 * 1024
SCOPES = ("uncommitted", "base")
# Tools whose end may leave the working tree changed (a subagent's edits included).
_CHANGING = frozenset({"execute", "edit", "delete", "move", "think"})
_EDITS = frozenset({"edit", "delete", "move"})
# What an item's end brings, by kind (or a tool's ACP kind), for one settled without it.
_END_FIELDS: dict[str, dict[str, Any]] = {
    "message": {"text": ""}, "reasoning": {"text": ""}, "plan": {"steps": []},
    "execute": {"exit_code": None, "output_lines": 0},
    "edit": {"files": []}, "delete": {"files": []}, "move": {"files": []}}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def one_line(text: Any) -> str:
    """A cause as the phone shows it: one line, secrets redacted, at most ``MAX_LINE_CHARS``."""
    from agent.redact import redact_sensitive_text
    text = " ".join(redact_sensitive_text(str(text or ""), redact_url_credentials=True).split())
    return text if len(text) <= MAX_LINE_CHARS else text[: MAX_LINE_CHARS - 1].rstrip() + "…"


def failed(reason: str, cause: Any) -> RpcError:
    """``4003``: the call failed on this machine, ``reason`` a code the call lists."""
    message = one_line(cause)
    return RpcError(FAILED, message or reason, {"reason": reason, "message": message})


def _invalid(message: str) -> RpcError:
    return RpcError(INVALID_PARAMS, message)


def _id(params: dict[str, Any], key: str = "id") -> str:
    value = params.get(key)
    if not isinstance(value, str) or not value.strip() or len(value) > MAX_ID:
        raise _invalid(f"{key} must be a non-empty string")
    return value


def _name(value: Any, *, nullable: bool = False) -> Optional[str]:
    """A name under the chats' Titles rule: one line, at most ``MAX_NAME`` code points."""
    if value is None and nullable:
        return None
    name = " ".join(value.split())[:MAX_NAME].rstrip() if isinstance(value, str) else ""
    if not name and not nullable:
        raise _invalid("name must be a non-empty string")
    return name or None


def _optional_str(params: dict[str, Any], key: str) -> Optional[str]:
    value = params.get(key)
    if value is not None and (not isinstance(value, str) or not value):
        raise _invalid(f"{key} must be a string or null")
    return value


def _comments(raw: Any) -> list[dict[str, Any]]:
    """``message.send``'s line comments, ``{path, line, side, text}`` each."""
    if raw is None:
        return []
    if not isinstance(raw, list) or len(raw) > MAX_COMMENTS:
        raise _invalid(f"comments must be a list of at most {MAX_COMMENTS}")
    comments = []
    for comment in raw:
        if not isinstance(comment, dict):
            raise _invalid("a comment must be {path, line, side, text}")
        path, line, side, text = (comment.get(key) for key in ("path", "line", "side", "text"))
        if (not isinstance(path, str) or not path or isinstance(line, bool) or not isinstance(line, int) or line < 1
                or side not in ("new", "old") or not isinstance(text, str) or not text.strip()):
            raise _invalid("a comment must be {path, line (from 1), side (new or old), text}")
        comments.append({"path": path, "line": line, "side": side, "text": text})
    return comments


def settle_item(item: dict[str, Any], status: str, at: str) -> dict[str, Any]:
    """An item completed with ``status``, what its end brings defaulted where it is not known."""
    ends = _END_FIELDS.get(item.get("tool") if item.get("kind") == "tool" else item.get("kind"), {})
    return {**ends, **item, "status": status, "finished_at": at}


class Projects:
    """The adapter's handle on projects: ``calls`` (method -> handler), ``snapshot()`` for a reset,
    ``live_frames()`` after ``sync.done``, ``send`` and ``history`` for a thread's messages and pages."""

    def __init__(self, adapter: AlephAdapter, *, root: Path, backends: dict[str, Backend]) -> None:
        # Imported here: sessions build on this module's helpers.
        from plugins.platforms.aleph.projects.bus import Bus
        from plugins.platforms.aleph.projects.session import SessionHub
        self.adapter = adapter
        # The root as it really is, so a folder is inside it exactly when its real path is.
        self.root = Path(os.path.realpath(root))
        self.state_dir = adapter.state_dir / "projects"
        self.backends = backends  # the adapter's registry, which outlives a reconnect
        self.hub = SessionHub(self)
        self.bus = Bus(self)
        self.desks: dict[str, RequestDesk] = {}  # by project, made at its first request
        self._probes: dict[str, tuple[float, BackendInfo]] = {}
        self.outputs = Outputs(self)
        self._lock = asyncio.Lock()
        self._creating: dict[str, asyncio.Future] = {}
        self._changes: dict[str, dict[str, Any]] = {}  # each project's project.changes as last sent
        self._looking: dict[str, asyncio.Task] = {}  # by project: a look at its working tree under way
        self._look_again: set[str] = set()  # asked for while one was under way
        self._poller: Optional[asyncio.Task] = None
        self._deleted: set[str] = set()  # this run: a look under way at one must not bring it back
        self.calls: dict[str, Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]] = {
            "backends.list": self._backends_list, "fs.list": self._fs_list, "fs.mkdir": self._fs_mkdir,
            "project.create": self._create_project, "project.update": self._update_project,
            "project.delete": self._delete_project, "thread.update": self._update_thread, "thread.close": self._close_thread,
            "thread.stop": self._stop_thread, "message.cancel": self._cancel_message,
            "changes.get": self._changes_get, "diff.get": self._diff_get}
        # Calls that answer to one connection.
        self.connection_calls: dict[str, Callable[[dict[str, Any], Optional[Connection]], Awaitable[dict[str, Any]]]] = {
            "output.get": self.outputs.get, "output.stop": self._output_stop}

    @property
    def tables(self):
        return self.adapter.store.projects

    def folder(self, project) -> str:
        """Where a project's sessions run: its folder, or the working folder the server gave it."""
        return project.payload.get("folder") or str(self.root / WORKING / project.id)

    async def start(self) -> None:
        """Before the server accepts connections: settle what the last run left open. Turns and items
        always end; the backend session itself resumes on the thread's next turn."""
        items, turns = await asyncio.to_thread(self.tables.settle_open)
        at = now_iso()
        for item in items:
            if item.get("tool") == "execute":
                row = await asyncio.to_thread(self.tables.item, item["id"])
                item["output_lines"] = await self.outputs.lines(item["project"], item["id"], row.out_start)
            await self.adapter._log("item.completed", settle_item(item, "failed", at))
        for turn in turns:
            await self.adapter._log("turn.completed", {
                "project": turn["project"], "thread": turn["thread"], "turn": turn["turn"], "status": "failed",
                "error": RESTARTED, "silent": False, "usage": None, "finished_at": at})
        if items or turns:
            logger.info("Aleph: settled %d turn(s) and %d item(s) left open by the last run", len(turns), len(items))
        await self.bus.start()
        self.hub.start(await asyncio.to_thread(self.tables.waiting_threads),
                       await asyncio.to_thread(self.tables.unbriefed))
        await self.bus.resume()
        for project_id in await asyncio.to_thread(self.tables.live_projects):
            self.look(project_id)
        self._poller = asyncio.create_task(self._poll_forever(), name="aleph-projects-changes")

    async def stop(self) -> None:
        """The server stops: every backend process ends; what was open settles at the next start."""
        await self.bus.stop()
        for task in [*self._looking.values(), *([self._poller] if self._poller else [])]:
            task.cancel()
        for desk in self.desks.values():
            desk.stop()
        self.desks.clear()
        await self.hub.stop()
        for backend in self.backends.values():
            await backend.stop()

    # ── what a phone catching up gets ──────────────────────────────────────────────────────────

    async def snapshot(self) -> list[dict[str, Any]]:
        """Under ``append_lock``: ``hello``'s ``projects`` on a reset."""
        return await asyncio.to_thread(self.adapter.store.projects_snapshot)

    def live_frames(self) -> list[str]:
        """After ``sync.done``: the frames that show what is growing now (running items' text so far,
        each project's changes)."""
        return self.hub.live_frames() + self._changes_frames()

    def _changes_frames(self) -> list[str]:
        """One ``project.changes`` per project, its working tree's summary as last seen."""
        return [event_frame("project.changes", payload) for payload in self._changes.values()]

    # ── working trees ──────────────────────────────────────────────────────────────────────────

    def look(self, project_id: str) -> None:
        """Look at a project's working tree soon, and send its ``project.changes`` when the summary
        changed. Nothing watches the folder: this runs after what may have changed it."""
        if project_id in self._looking:
            self._look_again.add(project_id)
            return
        task = self._looking[project_id] = asyncio.create_task(self._look(project_id))
        task.add_done_callback(lambda _: self._looking.pop(project_id, None))

    async def _look(self, project_id: str) -> None:
        while True:
            self._look_again.discard(project_id)
            row = await asyncio.to_thread(self.tables.project, project_id)
            if row is None or row.deleted_seq is not None or row.payload.get("folder") is None:
                return  # a project with no folder has no changes to show
            try:
                payload = {"project": project_id, **await git.summary(row.payload["folder"])}
            except git.GitError as exc:
                logger.warning("Aleph: looking at project %s's working tree failed: %s", project_id, exc)
                return
            if self._changes.get(project_id) != payload and project_id not in self._deleted:
                self._changes[project_id] = payload
                self.adapter._live("project.changes", payload)
            if project_id not in self._look_again:
                return

    async def _poll_forever(self) -> None:
        """While a turn runs and a phone is connected, its project's working tree every ``LOOK_SECONDS``:
        what a command changes shows while it runs."""
        while True:
            await asyncio.sleep(LOOK_SECONDS)
            server = self.adapter._server
            if server is None or not any(conn.ready for conn in server.connections):
                continue
            for project_id in {session.project for session in self.hub.sessions.values() if session.turn is not None}:
                self.look(project_id)

    # ── what sessions hand on ──────────────────────────────────────────────────────────────────

    def desk(self, project_id: str) -> RequestDesk:
        from plugins.platforms.aleph.projects.requests import RequestDesk
        desk = self.desks.get(project_id)
        if desk is None:
            desk = self.desks[project_id] = RequestDesk(self, project_id)
        return desk

    async def on_ask(self, session: Session, ask: Ask) -> None:
        """Under the session's gate: the backend waits on the person."""
        await self.desk(session.project).ask(session, ask)

    async def on_plan(self, session: Session) -> None:
        """Under the session's gate: a turn run in plan mode completed with a reply and nothing followed
        it. A backend that does not ask to leave plan mode itself gets the server's own request."""
        project = await asyncio.to_thread(self.tables.project, session.project)
        if not self.backends[project.payload["backend"]].asks_to_leave_plan:
            await self.desk(session.project).ask(session, Ask(_PLAN_KEY, "plan"), server=True)

    async def cancel_requests(self, session: Session, *, backend_only: bool = False) -> None:
        """Under the session's gate: its thread's requests can no longer be answered."""
        desk = self.desks.get(session.project)
        if desk is not None:
            await desk.cancel(session.thread, backend_only=backend_only)

    def _desk_with(self, card_id: Any) -> Optional[RequestDesk]:
        return next((desk for desk in self.desks.values() if desk.has(card_id)), None)

    async def respond_approval(self, request_id: str, choice: str) -> None:
        """``approval.respond`` for a project's request."""
        desk = self._desk_with(request_id)
        if desk is None:
            raise _invalid("no pending approval with that request_id")
        await desk.respond(request_id, choice)

    def question(self, clarify_id: str) -> Optional[tuple[list[str], bool]]:
        """A project's question on show as ``clarify_id``: its labels and whether it takes several."""
        desk = self._desk_with(clarify_id)
        return desk.question(clarify_id) if desk is not None else None

    async def answer_question(self, clarify_id: str, picks: Optional[list[int]], message_id: Optional[str]) -> None:
        """``clarify.respond`` for a project's question, its picks checked against ``question``."""
        await self._desk_with(clarify_id).pick(clarify_id, picks, message_id)

    async def leave_plan(self, thread_id: str) -> None:
        """A plan accepted: the thread leaves plan mode (``thread.update``)."""
        async with self._lock:
            thread = await asyncio.to_thread(self.tables.thread, thread_id)
            if thread is not None and thread.payload.get("plan"):
                await self.adapter._log("thread.update", {**thread.payload, "plan": False})

    async def on_output(self, session: Session, item: str, chunk: str) -> int:
        """Under the session's gate: a running command's new output (``item`` its wire id), kept and
        sent to who follows it; how many lines it has now."""
        return await self.outputs.append(session.project, item, chunk)

    async def on_edit(self, session: Session, item: str, event: Item) -> Optional[list[dict[str, Any]]]:
        """Under the session's gate, as an edit item starts (before the person is asked to approve it):
        its diff, kept with it for ``diff.get {item}``. The item's ``files``; None when nothing says
        what it changes."""
        if not event.patches and not event.arguments:
            return None
        project = await asyncio.to_thread(self.tables.project, session.project)
        files = await asyncio.to_thread(git.edit_files, self.folder(project), event.arguments, event.patches)
        await asyncio.to_thread(self._keep_diff, session.project, item, files)
        return [{key: entry[key] for key in ("path", "old_path", "added", "removed") if entry[key] is not None}
                for entry in files]

    def _keep_diff(self, project_id: str, item: str, files: list[dict[str, Any]]) -> None:
        body = json.dumps({"files": files}, ensure_ascii=False, separators=(",", ":"))
        if len(body.encode("utf-8")) > MAX_ITEM_DIFF_BYTES:
            body = json.dumps({"files": [{**entry, "too_large": True, "hunks": []} for entry in files]},
                              ensure_ascii=False, separators=(",", ":"))
        folder = self.state_dir / project_id
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{item}.diff").write_text(body, encoding="utf-8")

    def on_completed(self, session: Session, item: dict[str, Any]) -> None:
        """Under the session's gate: an item completed. A command's output is whole; what may have
        changed the working tree has it looked at."""
        if item.get("kind") != "tool":
            return
        if item.get("tool") == "execute":
            self.outputs.finish(item["id"])
        if item.get("tool") in _CHANGING:
            self.look(session.project)

    def on_turn_end(self, session: Session) -> None:
        self.look(session.project)

    def on_idle(self, session: Session) -> None:
        """Under the session's gate: a turn ended and nothing waited to start the next. What the other
        sessions sent it may go now."""
        self.bus.wake(session.thread)

    async def history(self, thread_id: Any, before: int, limit: int) -> dict[str, Any]:
        """Under ``append_lock``: ``history.page`` for ``history.load {thread}``."""
        thread = await self._live_thread(thread_id)
        entries, older = await asyncio.to_thread(self.adapter.store.thread_history, thread.id, before, limit)
        return {"thread": thread.id, "before": before, "entries": entries, "next": older}

    # ── backends ───────────────────────────────────────────────────────────────────────────────

    async def backend_info(self, backend_id: str) -> BackendInfo:
        """What a backend can run here, asked at most every ``PROBE_SECONDS``."""
        cached = self._probes.get(backend_id)
        if cached is not None and time.monotonic() - cached[0] < PROBE_SECONDS:
            return cached[1]
        backend = self.backends[backend_id]
        try:
            info = await backend.probe()
        except Exception as exc:
            logger.warning("Aleph: probing %s failed: %s", backend_id, exc)
            info = BackendInfo(backend.id, backend.name, False, "error", one_line(exc) or "The probe failed")
        self._probes[backend_id] = (time.monotonic(), info)
        return info

    def invalidate(self, backend_id: str) -> None:
        """Ask the backend again next time (it failed to start)."""
        self._probes.pop(backend_id, None)

    async def _backends_list(self, params: dict[str, Any]) -> dict[str, Any]:
        infos = await asyncio.gather(*(self.backend_info(backend_id) for backend_id in self.backends))
        return {"backends": [info.wire() for info in infos]}

    async def _settings(self, backend_id: str, settings: dict[str, Any]) -> None:
        """Check a thread's settings as they will be (``model``, ``effort``, ``tier``, ``approval``)
        against what the backend offers: ``-32602`` for a choice it does not have, ``4003 unavailable``
        when it cannot run. An effort or a tier is the model's, the default model's for a null one."""
        info = await self.backend_info(backend_id)
        if not info.available:
            raise failed("unavailable", info.message or f"{info.name} cannot run here ({info.reason})")
        model = info.model(settings.get("model"))
        if settings.get("model") is not None and model is None:
            raise _invalid(f"{info.name} has no model {settings['model']}")
        effort, tier = settings.get("effort"), settings.get("tier")
        if effort is not None and (model is None or effort not in model.efforts):
            raise _invalid(f"{effort} is no effort of {model.name if model else 'the default model'}")
        if tier is not None and (tier not in TIERS or model is None or tier not in model.tiers):
            raise _invalid(f"{tier} is no speed of {model.name if model else 'the default model'}")
        approval = settings.get("approval")
        if approval not in APPROVALS or approval not in info.approvals:
            raise _invalid(f"{info.name} does not offer approval {approval} here")

    # ── the folder browser ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _host(params: dict[str, Any]) -> str:
        host = params.get("host", "local")
        if host not in HOSTS:
            raise _invalid(f"unknown host {host!r}")
        return host

    @staticmethod
    def _real_folder(path: Any) -> str:
        """``path``, absolute, resolved, and a folder."""
        if not isinstance(path, str) or not os.path.isabs(path) or "\x00" in path:
            raise _invalid("path must be an absolute path")
        real = os.path.realpath(path)
        if not os.path.isdir(real):
            raise _invalid("path is not a folder")
        return real

    def _list(self, path: Optional[str]) -> dict[str, Any]:
        real = self._real_folder(str(self.root) if path is None else path)
        folders = []
        try:
            with os.scandir(real) as entries:
                for entry in entries:
                    if not entry.name.startswith(".") and entry.is_dir():
                        folders.append({"name": entry.name, "path": entry.path})
        except PermissionError as exc:
            raise failed("access", f"{real} cannot be read") from exc
        except OSError as exc:
            raise failed("access", exc) from exc
        folders.sort(key=lambda folder: (folder["name"].casefold(), folder["name"]))
        parent = os.path.dirname(real)
        return {"path": real, "parent": None if parent == real else parent,
                "folders": folders[:MAX_FOLDERS], "more": len(folders) > MAX_FOLDERS}

    async def _fs_list(self, params: dict[str, Any]) -> dict[str, Any]:
        self._host(params)
        return await asyncio.to_thread(self._list, params.get("path"))

    def _mkdir(self, path: Any, name: Any) -> dict[str, Any]:
        name = name.strip() if isinstance(name, str) else ""
        if not name:
            raise _invalid("Give the folder a name.")
        # A leading "." also rules out "." and "..": the folder would be hidden from fs.list.
        if "/" in name or "\x00" in name or name.startswith("."):
            raise _invalid("A folder name can't contain \"/\" or start with \".\".")
        if len(name.encode()) > 255:
            raise _invalid("That folder name is too long.")
        if not isinstance(path, str) or not os.path.isabs(path) or "\x00" in path:
            raise _invalid("path must be an absolute path")
        real = os.path.realpath(path)
        if not os.path.isdir(real):
            raise failed("access", f"{real} is not a folder")
        made = os.path.join(real, name)
        try:
            os.mkdir(made)
        except FileExistsError as exc:
            raise _invalid(f"A folder named \"{name}\" already exists.") from exc
        except PermissionError as exc:
            raise failed("access", f"{real} cannot be written to") from exc
        except OSError as exc:
            raise failed("access", exc) from exc
        return {"path": made}

    async def _fs_mkdir(self, params: dict[str, Any]) -> dict[str, Any]:
        """``fs.mkdir {host?, path, name}``: a new folder in the one the browser shows."""
        self._host(params)
        return await asyncio.to_thread(self._mkdir, params.get("path"), params.get("name"))

    # ── projects ───────────────────────────────────────────────────────────────────────────────

    async def _live_project(self, project_id: Any):
        row = await asyncio.to_thread(self.tables.project, project_id) if isinstance(project_id, str) else None
        if row is None or row.deleted_seq is not None:
            raise _invalid("no project with that id")
        return row

    async def _id_unused(self, entity_id: str) -> None:
        """An id names one thing for good: a project, a thread or a chat, deleted ones included."""
        store = self.adapter.store
        if (await asyncio.to_thread(self.tables.project, entity_id) is not None
                or await asyncio.to_thread(self.tables.thread, entity_id) is not None
                or await asyncio.to_thread(store.chat, entity_id) is not None or entity_id == self.adapter.chat_id):
            raise _invalid("that id is already used")

    async def _create_project(self, params: dict[str, Any]) -> dict[str, Any]:
        """``project.create``. The same id again answers as the first time: a second call while the
        first runs awaits it."""
        project_id = _id(params)
        pending = self._creating.get(project_id)
        if pending is None:
            pending = asyncio.ensure_future(self._create_once(project_id, params))
            self._creating[project_id] = pending

            def done(future: asyncio.Future) -> None:
                self._creating.pop(project_id, None)
                if not future.cancelled():
                    future.exception()  # retrieved: the caller that would have may be gone
            pending.add_done_callback(done)
        return await asyncio.shield(pending)

    async def _create_once(self, project_id: str, params: dict[str, Any]) -> dict[str, Any]:
        known = await asyncio.to_thread(self.tables.project, project_id)
        if known is not None:
            if known.deleted_seq is not None:
                raise _invalid("that project was deleted")
            return {"seq": known.created_seq}
        await self._id_unused(project_id)
        host, backend_id = self._host(params), params.get("backend")
        if backend_id not in self.backends:
            raise _invalid(f"unknown backend {backend_id!r}")
        folder = params.get("folder")
        if folder is not None:
            folder = await asyncio.to_thread(self._real_folder, folder)
            if not await asyncio.to_thread(os.access, folder, os.R_OK | os.W_OK | os.X_OK):
                raise failed("access", f"{folder} cannot be read and written here")
        name = _name(params.get("name"), nullable=True)  # else named from the person's first message
        settings = {key: _optional_str(params, key) for key in ("model", "effort", "tier", "approval")}
        await self._settings(backend_id, settings)
        async with self._lock:
            known = await asyncio.to_thread(self.tables.project, project_id)
            if known is not None:  # created meanwhile by a call that did not wait on this one
                return {"seq": known.created_seq}
            if folder is None:
                await asyncio.to_thread(os.makedirs, self.root / WORKING / project_id, exist_ok=True)
            created_at = now_iso()
            event = await self.adapter._log("project.new", {
                "id": project_id, "name": name, "folder": folder, "host": host, "backend": backend_id,
                "created_at": created_at})
            await self.adapter._log("thread.new", {
                "id": project_id, "project": project_id, "name": None, "created_at": created_at, "item": None,
                "brief": None, **settings, "plan": False, "closed": False})
        logger.info("Aleph: created project %s in %s (%s)", project_id, folder or "its working folder", backend_id)
        self.look(project_id)
        return {"seq": event.seq}

    async def _update_project(self, params: dict[str, Any]) -> dict[str, Any]:
        """``project.update``: its name. ``{seq: null}`` when nothing changed."""
        project_id, name = _id(params), _name(params.get("name"))
        async with self._lock:
            row = await self._live_project(project_id)
            if row.payload.get("name") == name:
                return {"seq": None}
            event = await self.adapter._log("project.update", {**row.payload, "name": name})
        return {"seq": event.seq}

    async def _delete_project(self, params: dict[str, Any]) -> dict[str, Any]:
        """``project.delete``: its sessions end, its events and bodies go; the folder stays, but a working
        folder the server gave goes too."""
        project_id = _id(params)
        async with self._lock:
            row = await asyncio.to_thread(self.tables.project, project_id)
            if row is None:
                raise _invalid("no project with that id")
            if row.deleted_seq is not None:
                return {"seq": row.deleted_seq}  # a resend after a lost answer
            await self._end_sessions(project_id)
            self._deleted.add(project_id)
            self._changes.pop(project_id, None)
            self.outputs.forget(project_id)
            desk = self.desks.pop(project_id, None)
            if desk is not None:
                desk.stop()
            async with self.adapter.append_lock:
                event = await asyncio.to_thread(self.adapter.store.delete_project, project_id)
                self.adapter._broadcast(event)
        await asyncio.to_thread(shutil.rmtree, self.state_dir / project_id, ignore_errors=True)
        if row.payload.get("folder") is None:
            await asyncio.to_thread(shutil.rmtree, self.root / WORKING / project_id, ignore_errors=True)
        logger.info("Aleph: deleted project %s", project_id)
        return {"seq": event.seq}

    async def _end_sessions(self, project_id: str, thread_id: Optional[str] = None) -> None:
        """Stop and end the sessions of a project, or of one of its threads."""
        await self.hub.end(project_id, thread_id)

    # ── fetches ────────────────────────────────────────────────────────────────────────────────

    async def _folder_of(self, params: dict[str, Any]) -> tuple[str, str]:
        """``(folder, scope)`` of a call on a project's changes."""
        project = await self._live_project(params.get("project"))
        if project.payload.get("folder") is None:
            raise _invalid("a project with no folder has no changes")
        scope = params.get("scope")
        if scope not in SCOPES:
            raise _invalid("scope must be uncommitted or base")
        return project.payload["folder"], scope

    async def _changes_get(self, params: dict[str, Any]) -> dict[str, Any]:
        folder, scope = await self._folder_of(params)
        try:
            return await git.changes(folder, scope)
        except git.NotGit as exc:
            raise failed("not_git", f"{folder} is not a git repository") from exc
        except git.GitError as exc:
            raise failed("io", exc) from exc

    async def _diff_get(self, params: dict[str, Any]) -> dict[str, Any]:
        """``diff.get``: a changed file's diff by ``{project, scope, path}``, or the one an edit item kept."""
        if "item" in params:
            return await self._item_diff(params["item"])
        folder, scope = await self._folder_of(params)
        path = params.get("path")
        if not isinstance(path, str) or not path:
            raise _invalid("path must be a path changes.get listed")
        try:
            entry = await git.file_diff(folder, scope, path, params.get("full") is True)
        except git.NotGit as exc:
            raise failed("not_git", f"{folder} is not a git repository") from exc
        except git.GitError as exc:
            raise failed("io", exc) from exc
        if entry is None:
            raise _invalid("that file has no changes in this scope")
        return {"files": [entry]}

    async def _item_diff(self, item_id: Any) -> dict[str, Any]:
        """An edit item's diff as it was kept when the item started. ``full`` does not apply: what the
        file was beside the change is not kept, only the change."""
        row = await asyncio.to_thread(self.tables.item, item_id) if isinstance(item_id, str) else None
        if row is None or row.payload.get("tool") not in _EDITS:
            raise _invalid("no edit with that item id")
        try:
            body = await asyncio.to_thread((self.state_dir / row.project / f"{row.id}.diff").read_text, encoding="utf-8")
        except FileNotFoundError:
            return {"files": []}
        return json.loads(body)

    async def _output_stop(self, params: dict[str, Any], conn: Optional[Connection]) -> dict[str, Any]:
        if conn is not None:
            self.outputs.drop(conn)
        return {}

    def connection_closed(self, conn: Connection) -> None:
        """What a connection followed ends with it."""
        self.outputs.drop(conn)

    # ── threads ────────────────────────────────────────────────────────────────────────────────

    async def _live_thread(self, thread_id: Any) -> ThreadRow:
        """A thread of a project that exists, closed or not."""
        row = await asyncio.to_thread(self.tables.thread, thread_id) if isinstance(thread_id, str) else None
        if row is None:
            raise _invalid("no thread with that id")
        await self._live_project(row.project)
        return row

    async def open_agent_thread(self, project_id: str, thread_id: str, name: str, item: Optional[str],
                                brief: Optional[str]) -> ThreadRow:
        """A thread the top level opens (``open_thread``), named ``name``, its row in the main transcript
        the ``thread`` item ``item``. It starts with the top level's settings as they are now."""
        async with self._lock:
            await self._live_project(project_id)
            top = await asyncio.to_thread(self.tables.thread, project_id)
            settings = {key: top.payload.get(key) for key in ("model", "effort", "tier", "approval")}
            await self.adapter._log("thread.new", {
                "id": thread_id, "project": project_id, "name": name, "created_at": now_iso(), "item": item,
                "brief": brief, **settings, "plan": False, "closed": False})
        return await asyncio.to_thread(self.tables.thread, thread_id)

    async def _update_thread(self, params: dict[str, Any]) -> dict[str, Any]:
        """``thread.update``, at any time, from the next turn on; the top level takes all but ``name``
        (its name is the project's). ``{seq: null}`` when nothing changed."""
        thread_id = _id(params)
        changes: dict[str, Any] = {}
        if "name" in params:
            changes["name"] = _name(params["name"])
        for key in ("model", "effort", "tier", "approval"):
            if key in params:
                changes[key] = _optional_str(params, key)
        if "plan" in params:
            if not isinstance(params["plan"], bool):
                raise _invalid("plan must be true or false")
            changes["plan"] = params["plan"]
        async with self._lock:
            thread = await self._live_thread(thread_id)
            if thread.top_level and "name" in changes:
                raise _invalid("the top level's name is the project's")
            current = thread.payload
            changed = {key: value for key, value in changes.items() if current.get(key) != value}
            if not changed:
                return {"seq": None}
            if changed.keys() & {"model", "effort", "tier", "approval"}:
                project = await self._live_project(thread.project)
                await self._settings(project.payload["backend"], {**current, **changed})
            event = await self.adapter._log("thread.update", {**current, **changed})
        return {"seq": event.seq}

    async def _stop_thread(self, params: dict[str, Any]) -> dict[str, Any]:
        """``thread.stop``: its turn ends ``interrupted``; nothing running is no error."""
        thread = await self._live_thread(params.get("thread"))
        session = self.hub.sessions.get(thread.id)
        if session is not None:
            await session.stop()
        return {}

    async def _close_thread(self, params: dict[str, Any]) -> dict[str, Any]:
        """``thread.close``: its session ends, kept to resume; its events stay. Closing a closed
        thread changes nothing (``{seq: null}``)."""
        thread_id = _id(params)
        async with self._lock:
            thread = await self._live_thread(thread_id)
            if thread.top_level:
                raise _invalid("the top level cannot be closed")
            if thread.payload.get("closed"):
                return {"seq": None}
            await self._end_sessions(thread.project, thread.id)
            event = await self.adapter._log("thread.close", {"project": thread.project, "thread": thread.id})
        return {"seq": event.seq}

    # ── messages ───────────────────────────────────────────────────────────────────────────────

    async def send(self, params: dict[str, Any]) -> int:
        """``message.send {thread}``: log the person's message in the thread (reopening a closed one)
        and hand it to the thread's session. Idempotent by id, as in a chat; returns its seq."""
        if "chat" in params:
            raise _invalid("a message names its chat or its thread, not both")
        adapter = self.adapter
        message_id = _id(params)
        text = params.get("text") or ""
        if not isinstance(text, str) or len(text) > MAX_TEXT_CHARS:
            raise _invalid("text must be a string")
        uploads = params.get("attachments") or []
        if not isinstance(uploads, list) or not all(isinstance(item, dict) for item in uploads):
            raise _invalid("attachments must be a list of {id, ticket, key, name, mime}")
        comments = _comments(params.get("comments"))
        delivery = params.get("delivery", "steer")
        if delivery not in DELIVERIES:
            raise _invalid("delivery must be steer or after_turn")
        thread = await self._live_thread(params.get("thread"))
        async with adapter._intake_lock:
            known = await self.sent_seq(message_id, thread.id)
            if known is not None:
                return known
            if not text.strip() and not uploads and not comments:
                raise _invalid("a message needs text, attachments or comments")
            attachments = await adapter._take_uploads(uploads)
            if thread.payload.get("closed"):
                async with self._lock:
                    thread = await self._live_thread(thread.id)
                    if thread.payload.get("closed"):
                        await adapter._log("thread.update", {**thread.payload, "closed": False})
            await self._name_from(thread.project, text)
        payload: dict[str, Any] = {"project": thread.project, "thread": thread.id, "id": message_id,
                                   "sender": "user", "text": text, "attachments": adapter._with_links(attachments)}
        if comments:
            payload["comments"] = comments
        payload["created_at"] = now_iso()
        # Logged by the session, under its gate: whether it waits for the turn is decided there.
        return await self.hub.session(thread).accept(payload, attachments, delivery)

    async def _name_from(self, project_id: str, text: str) -> None:
        """A project still unnamed is named from the person's first message."""
        name = _name(text, nullable=True)
        if name is None:
            return
        async with self._lock:
            row = await self._live_project(project_id)
            if row.payload.get("name") is None:
                await self.adapter._log("project.update", {**row.payload, "name": name})

    async def sent_seq(self, message_id: str, thread_id: str) -> Optional[int]:
        """The seq ``message.send`` first answered for ``message_id`` in this thread (its
        ``message.queued``, or its ``message.new``), or None for an id not used yet."""
        queued = await asyncio.to_thread(self.tables.queued, message_id)
        if queued is not None:
            if queued[0] != thread_id:
                raise _invalid("id belongs to another message")
            return queued[1]
        existing = await asyncio.to_thread(self.adapter.store.message, message_id)
        if existing is not None:
            if existing.sender != "user" or existing.chat != thread_id:
                raise _invalid("id belongs to another message")
            return existing.seq
        return None

    async def _cancel_message(self, params: dict[str, Any]) -> dict[str, Any]:
        """``message.cancel``: take back a queued message; one already sent is ``-32602``."""
        message_id = _id(params)
        queued = await asyncio.to_thread(self.tables.queued, message_id)
        if queued is None:
            raise _invalid("no queued message with that id")
        thread = await self._live_thread(queued[0])
        return {"seq": await self.hub.session(thread).cancel(message_id)}

    async def prompt(self, message: dict[str, Any]) -> str:
        """A logged message as its backend reads it: its line comments first, as quote blocks under
        ``path:line``, then its text, then the paths of its files on this machine."""
        parts = []
        for comment in message.get("comments") or ():
            where = f"{comment['path']}:{comment['line']}" + (" (before the change)" if comment["side"] == "old" else "")
            parts.append("\n".join([f"> {where}", *(f"> {line}" for line in comment["text"].splitlines())]))
        if message.get("text", "").strip():
            parts.append(message["text"])
        files = []
        for description in message.get("attachments") or ():
            stored = await asyncio.to_thread(self.adapter.store.file, str(description.get("id")))
            if stored is not None and stored.agent_path:
                files.append(f"- {description.get('name') or 'file'}: {stored.agent_path}")
        if files:
            parts.append("Attached files:\n" + "\n".join(files))
        return "\n\n".join(parts)
