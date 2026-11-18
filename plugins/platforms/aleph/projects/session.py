"""Each project thread's live session with its backend, and the hub that holds them.

A ``Session`` turns what its ``BackendSession`` reports into the log. A turn's and its items' starts
and ends are logged (``turn.*``, ``item.*``); what grows in between (an item's text, a plan's steps)
goes live as ``item.delta``, coalesced per item, and is resent from 0 to a phone that catches up. The
backend's own ids for items stay here: the wire knows items by ``i_`` ids the session gives them.

It owns the person's input too. A message steers the running turn, waits for its end
(``after_turn``), or starts one; when a turn ends, everything that waited starts the next one
together. One lock per session, its gate, serialises what changes the session (a submission, a
backend event, a stop or a close), so a decision and the log it leads to are never split by another;
it is held for a submission or an event, never for a turn. No gate is taken while another is held.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import secrets
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from plugins.platforms.aleph.projects.backend import (
    SECRET, Accepted, Ask, BackendSession, Delta, Event, Exit, Item, SessionSpec, Sink, TurnEnd)
from plugins.platforms.aleph.projects import tag
from plugins.platforms.aleph.projects.tag import reply_state
from plugins.platforms.aleph.projects.service import failed, now_iso, one_line, settle_item
from plugins.platforms.aleph.projects.store import ThreadRow
from plugins.platforms.aleph.server import INVALID_PARAMS, RpcError, event_frame

if TYPE_CHECKING:
    from plugins.platforms.aleph.projects.service import Projects

logger = logging.getLogger(__name__)

DELTA_SECONDS = 0.05  # an item's growth goes live at most this often
REAP_AFTER = 3600.0  # an idle thread's process is closed after this long
REAP_EVERY = 300.0
PROCESS_EXITED = "The agent's process exited."
CLAIM_SECONDS = 5.0  # how long an open_thread call waits for its item to be seen
_TEXT_KINDS = frozenset({"message", "reasoning"})
# What an item's end brings: never on its item.started.
_END_KEYS = frozenset({"text", "steps", "exit_code", "output_lines", "files"})
_EDITS = frozenset({"edit", "delete", "move"})


def _wire_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(8)}"


def _common_prefix(a: str, b: str) -> int:
    limit = min(len(a), len(b))
    index = 0
    while index < limit and a[index] == b[index]:
        index += 1
    return index


@dataclass
class _Input:
    """What a turn is given: the person's messages (several when they waited together), or the
    agent's own input."""
    key: str
    text: str
    order: int = 0  # when it was sent: the seq of its message.new, or of its message.queued
    seq: int = 0  # the newest message.new it carries
    read_up_to: Optional[str] = None  # that message's id: what its receipt names
    steer: bool = False  # handed to a turn already running (so lost with it, it goes again)
    cause: str = "user"
    origin: Optional[str] = None  # the thread a report is from

    @staticmethod
    def joined(inputs: list[_Input]) -> _Input:
        inputs = sorted(inputs, key=lambda one: one.order)
        newest = max(inputs, key=lambda one: one.seq)
        return _Input(_wire_id("in"), "\n\n".join(one.text for one in inputs if one.text), inputs[-1].order,
                      newest.seq, newest.read_up_to)


@dataclass(eq=False)
class _Live:
    """An item of the open turn."""
    id: str
    kind: str
    fields: dict[str, Any]
    started_at: str
    text: str = ""
    steps: Optional[list[dict[str, Any]]] = None
    shown: str = ""  # the text connected phones have
    logged: bool = False  # its item.started is in the log
    held: bool = False  # a top-level message not shown yet (it may be NO_REPLY)
    declined: bool = False  # the person refused its approval: it completes declined, whatever the backend says


@dataclass
class _Turn:
    id: str
    cause: str
    items: dict[str, _Live] = field(default_factory=dict)  # open, by the backend's key
    plan: Optional[_Live] = None  # the turn's one plan item
    quiet: set[str] = field(default_factory=set)  # coordination tool calls: no items (backend keys)
    interrupting: bool = False
    silenced: bool = False  # a top-level reply was NO_REPLY
    spoke: bool = False  # a top-level reply showed


class Session:
    """One thread's session: its backend session (open from its first input until closed or reaped),
    the open turn, and the input waiting for it."""

    def __init__(self, hub: SessionHub, thread: ThreadRow) -> None:
        self.hub = hub
        self.thread, self.project, self.top_level = thread.id, thread.project, thread.top_level
        self.gate = asyncio.Lock()
        self.backend: Optional[BackendSession] = None
        self.turn: Optional[_Turn] = None
        self.idle_since = time.monotonic()
        self._spec: Optional[SessionSpec] = None  # what the open backend session runs with
        self._session_id = thread.session_id
        self._pending: list[_Input] = []  # logged, not yet given to the backend
        self._unaccepted: dict[str, _Input] = {}  # given, not yet taken into a turn
        self._read_seq = 0
        self._gone = False  # disposed: its project is deleted
        self._token: Optional[str] = None  # the bus's token for the open backend session's process
        # open_thread items seen, by brief, each (item id, thread id) until its call claims it.
        self._parked: dict[str, list[tuple[str, str]]] = {}
        self._parking = asyncio.Condition()
        self._opened: set[str] = set()  # threads opened by an open_thread item still open
        # Bumped whenever the backend session is dropped: what the old one still reports is stale.
        self._generation = 0
        self._events: asyncio.Queue[tuple[int, Event]] = asyncio.Queue()
        self._dirty: dict[_Live, None] = {}  # grown since the last item.delta, in order
        self._flush: Optional[asyncio.TimerHandle] = None
        self._consumer = asyncio.create_task(self._consume(), name=f"aleph-thread-{thread.id}")

    @property
    def _projects(self) -> Projects:
        return self.hub.projects

    async def _log(self, event_type: str, payload: dict[str, Any], **kwargs: Any):
        return await self._projects.adapter._log(event_type, payload, **kwargs)

    def _scoped(self, **payload: Any) -> dict[str, Any]:
        return {"project": self.project, "thread": self.thread, **payload}

    # ── input ──────────────────────────────────────────────────────────────────────────────────

    async def accept(self, payload: dict[str, Any], attachments: list[dict[str, Any]], delivery: str) -> int:
        """The person's ``message.send``: logged as ``message.new`` and given to the backend, or, sent
        ``after_turn`` while a turn runs, logged as ``message.queued`` to wait for its end. Returns the
        seq of what was logged. ``4003 unavailable`` when the backend cannot start: the message is
        logged and its turn fails first."""
        async with self.gate:
            if self._gone:
                raise RpcError(INVALID_PARAMS, "no thread with that id")
            # Checked again here: a resend racing the first attempt meets it at the gate.
            known = await self._projects.sent_seq(payload["id"], self.thread)
            if known is not None:
                return known
            if delivery == "after_turn" and self.turn is not None:
                queued = self._scoped(id=payload["id"], text=payload["text"], attachments=payload["attachments"],
                                      comments=payload.get("comments", []), created_at=payload["created_at"])
                return (await self._log("message.queued", queued)).seq
            desk = self._projects.desks.get(self.project)
            text = await self._projects.prompt(payload)
            if desk is not None and desk.asks_secret(self.thread):
                # The answer to a secret question goes to the backend from memory; the log keeps a mask.
                payload, attachments = {**payload, "text": SECRET, "comments": []}, []
            event = await self._log("message.new", payload, message={
                "id": payload["id"], "sender": "user", "kind": "text", "text": payload["text"],
                "attachments": attachments})
            # Sent to the turn while it waits on the person: the message is their answer.
            if desk is not None and desk.waits(self.thread) and await desk.typed(self, text, (payload["id"], event.seq)):
                return event.seq
            self._pending.append(_Input(_wire_id("in"), text, event.seq, event.seq, payload["id"],
                                        steer=self.turn is not None))
            error = await self._submit()
        if isinstance(error, RpcError):
            raise error
        return event.seq

    @property
    def waiting(self) -> bool:
        """Whether a request of the thread waits on the person."""
        desk = self._projects.desks.get(self.project)
        return desk is not None and desk.waits(self.thread)

    async def submit(self, text: str, *, cause: str, origin: Optional[str] = None, when: str = "now") -> bool:
        """Input that is not the person's message (a brief, a report, a message from the top level, the
        server's own follow-up): steers the running turn, or starts one with ``cause`` and ``origin``.
        ``when`` ``free``: not while the thread waits on the person; ``idle``: only as a turn of its
        own. False when it was not taken for that; the error when its turn could not start."""
        async with self.gate:
            if self._gone or (when == "free" and self.waiting) or (when == "idle" and self.turn is not None):
                return False
            if self.turn is None:
                error = await self._start_turn([_Input(_wire_id("in"), text, cause=cause, origin=origin)])
            else:
                self._pending.append(_Input(_wire_id("in"), text, steer=True, cause=cause, origin=origin))
                error = await self._submit()
        if error is not None:
            raise error
        return True

    async def read(self, message_id: str, seq: int) -> None:
        """A ``receipt``: the agent has the person's message ``message_id`` (logged at ``seq``)."""
        if seq > self._read_seq:
            self._read_seq = seq
            await self._log("receipt", self._scoped(read_up_to=message_id, at=now_iso()))

    async def resume(self) -> None:
        """Start what waited for this thread through a restart (its queued messages), if anything."""
        async with self.gate:
            if not self._gone:
                await self._submit()

    async def cancel(self, message_id: str) -> int:
        """``message.cancel``: take back a queued message; the seq of its ``message.cancel``."""
        async with self.gate:
            state = await asyncio.to_thread(self._projects.tables.queued_state, message_id)
            if state == "cancelled":
                return await asyncio.to_thread(self._projects.tables.cancel_seq, message_id)
            if state != "waiting":
                raise RpcError(INVALID_PARAMS, "that message was already sent")
            return (await self._log("message.cancel", self._scoped(id=message_id))).seq

    async def _submit(self) -> Optional[Exception]:
        """Under the gate: give the backend what is pending. A running turn takes it as steers; one
        ending (or being interrupted) leaves it to its end, which starts the next turn with it;
        otherwise it starts a turn, together with every queued message."""
        turn = self.turn
        if turn is not None:
            if turn.interrupting or not self._pending or self.backend is None:
                return None
            inputs, self._pending = self._pending, []
            for index, one in enumerate(inputs):
                self._unaccepted[one.key] = one
                try:
                    await self.backend.send(one.key, one.text)
                except Exception as exc:
                    # The process is going; its exit fails the turn, and what it did not take goes again.
                    logger.warning("Aleph: thread %s: steering failed: %s", self.thread, exc)
                    self._unaccepted.pop(one.key, None)
                    self._pending[:0] = inputs[index:]
                    break
            return None
        inputs = await self._take_queued() + self._pending
        self._pending = []
        if not inputs:
            return None
        return await self._start_turn(inputs)

    async def _take_queued(self) -> list[_Input]:
        """The thread's waiting queued messages, each logged now as its ``message.new`` under its id,
        so its bubble stands where the agent read it."""
        adapter = self._projects.adapter
        inputs = []
        for order, queued in await asyncio.to_thread(self._projects.tables.waiting, self.thread):
            attachments = [{key: value for key, value in item.items() if key != "url"}
                           for item in queued.get("attachments") or []]
            payload = self._scoped(id=queued["id"], sender="user", text=queued.get("text") or "",
                                   attachments=adapter._with_links(attachments))
            if queued.get("comments"):
                payload["comments"] = queued["comments"]
            payload["created_at"] = now_iso()
            event = await self._log("message.new", payload, message={
                "id": queued["id"], "sender": "user", "kind": "text", "text": payload["text"],
                "attachments": attachments})
            inputs.append(_Input(_wire_id("in"), await self._projects.prompt(payload), order, event.seq,
                                 queued["id"]))
        return inputs

    async def _start_turn(self, inputs: list[_Input]) -> Optional[Exception]:
        """Under the gate, with no turn open: log ``turn.started`` and give the backend its input,
        opening the backend session first if it is not open. A failure to start fails the turn, and
        is returned."""
        one = inputs[0] if len(inputs) == 1 else _Input.joined(inputs)
        one.steer = False  # its turn's own: lost with that turn, it fails with it rather than going again
        cause, origin = inputs[0].cause, inputs[0].origin
        self.turn = _Turn(_wire_id("t"), cause)
        started = self._scoped(turn=self.turn.id, cause=cause)
        if origin is not None:
            started["from"] = origin
        started["started_at"] = now_iso()
        await self._log("turn.started", started)
        try:
            backend = await self._ready()
            self._unaccepted[one.key] = one
            await backend.send(one.key, one.text)
        except Exception as exc:
            self._unaccepted.pop(one.key, None)
            logger.warning("Aleph: thread %s: the turn could not start: %s", self.thread, exc)
            await self._end_turn(TurnEnd("failed", exc.message if isinstance(exc, RpcError) else one_line(exc)))
            return exc
        return None

    async def _ready(self) -> BackendSession:
        """The backend session, opened (resuming the thread's last one) or brought to the thread's
        settings as they are now: its own, else the project's."""
        tables = self._projects.tables
        thread = await asyncio.to_thread(tables.thread, self.thread)
        project = await asyncio.to_thread(tables.project, self.project)
        own, base = thread.payload, project.payload

        def setting(key: str) -> Any:
            return own.get(key) if own.get(key) is not None else base.get(key)
        spec = SessionSpec(self.thread, self.project, base["folder"], self.top_level, setting("model"),
                           setting("effort"), setting("approval"), bool(own.get("plan")), self._session_id)
        if self.backend is not None:
            # The coordination is fixed for the process: a memory index that changed reaches the next one.
            spec = dataclasses.replace(spec, instructions=self._spec.instructions, mcp=self._spec.mcp,
                                       memory=self._spec.memory, stop_report=self._spec.stop_report)
            if dataclasses.replace(spec, resume=None) != dataclasses.replace(self._spec, resume=None):
                await self.backend.apply(spec)
                self._spec = spec
            return self.backend
        projects, backend_id = self._projects, base["backend"]
        if backend_id not in projects.backends:
            raise failed("unavailable", f"{backend_id} is not available on this server")
        info = await projects.backend_info(backend_id)
        if not info.available:
            raise failed("unavailable", info.message or f"{info.name} cannot run here ({info.reason})")
        self._generation += 1
        state = projects.state_dir / self.project
        self._token = projects.bus.issue(self.project, self.thread, self.top_level)
        instructions = await asyncio.to_thread(
            tag.instructions, top_level=self.top_level, project=base["name"], folder=base["folder"],
            memory=state / "memory", thread=own.get("name"), opened_by_agent=own.get("opened_by") == "agent")
        spec = dataclasses.replace(spec, instructions=instructions, mcp=projects.bus.server(self._token),
                                   memory=str(state / "memory"), stop_report=str(state / f"{self.thread}.stop.json"))
        try:
            self.backend = await projects.backends[backend_id].open(spec, self._sink(self._generation))
        except Exception:
            projects.bus.revoke(self._token)
            self._token = None
            projects.invalidate(backend_id)
            raise
        self._spec = spec
        await self._remember(self.backend)
        return self.backend

    def item_id(self, key: Optional[str]) -> Optional[str]:
        """The wire id of the open turn's item the backend knows as ``key``."""
        live = self.turn.items.get(key) if self.turn is not None and key is not None else None
        return live.id if live is not None else None

    def decline(self, key: Optional[str]) -> None:
        """The person refused the approval holding the item ``key``: it completes ``declined``."""
        live = self.turn.items.get(key) if self.turn is not None and key is not None else None
        if live is not None:
            live.declined = True

    def _sink(self, generation: int) -> Sink:
        def sink(event: Event) -> None:
            self._events.put_nowait((generation, event))
        return sink

    async def _remember(self, backend: Optional[BackendSession]) -> None:
        """Keep the backend's session id, so the next process resumes it."""
        session_id = backend.session_id if backend is not None else None
        if not session_id or session_id == self._session_id:
            return
        self._session_id = session_id
        adapter = self._projects.adapter
        async with adapter.append_lock:
            await asyncio.to_thread(self._projects.tables.set_session_id, self.thread, session_id)

    # ── stopping and closing ───────────────────────────────────────────────────────────────────

    async def stop(self) -> None:
        """``thread.stop``: interrupt the running turn, which ends ``interrupted``. What the person
        steers meanwhile is held for the next turn, not handed to the dying one."""
        async with self.gate:
            turn = self.turn
            if turn is None or turn.interrupting or self.backend is None:
                return
            turn.interrupting = True
            await self.backend.interrupt()

    async def close(self) -> None:
        """End the backend process, its running turn ``interrupted``; its session id stays, so the
        next input resumes it. What waited stays waiting for that input."""
        async with self.gate:
            await self._projects.cancel_requests(self)
            await self._drop(close=True)
            await self._end_turn(TurnEnd("interrupted"), start_next=False)

    async def reap(self, now: float) -> bool:
        """Close the process of a thread idle for ``REAP_AFTER`` with nothing of it running in the
        background (the backend says); never the top level's."""
        async with self.gate:
            if (self.top_level or self.backend is None or self.turn is not None or self._pending
                    or now - self.idle_since < REAP_AFTER or not self.backend.idle_ok()):
                return False
            await self._drop(close=True)
            return True

    async def _drop(self, *, close: bool) -> None:
        """Under the gate: forget the backend session (closing its process), keeping its id. Steers it
        never took go again with the next turn."""
        backend, self.backend, self._spec = self.backend, None, None
        self._generation += 1
        self._projects.bus.revoke(self._token)
        self._token = None
        if backend is None:
            return
        await self._remember(backend)
        lost = [one for one in self._unaccepted.values() if one.steer]
        self._unaccepted.clear()
        self._pending[:0] = lost
        if close:
            await _close(backend, self.thread)

    def dispose(self) -> None:
        """Stop handling events and taking input, for good (the project is gone, or the server stops)."""
        self._gone = True
        self._projects.bus.revoke(self._token)
        self._consumer.cancel()
        if self._flush is not None:
            self._flush.cancel()

    # ── what the backend reports ───────────────────────────────────────────────────────────────

    async def _consume(self) -> None:
        while True:
            generation, event = await self._events.get()
            async with self.gate:
                if generation != self._generation:
                    continue
                try:
                    await self._handle(event)
                except Exception:
                    logger.exception("Aleph: thread %s: handling %s failed", self.thread, type(event).__name__)

    async def _handle(self, event: Event) -> None:
        if isinstance(event, TurnEnd):
            await self._end_turn(event)
        elif isinstance(event, Exit):
            await self._drop(close=False)
            if self.turn is not None:
                await self._end_turn(TurnEnd("failed", event.error or PROCESS_EXITED))
            else:
                await self._submit()
        elif isinstance(event, Accepted):
            one = self._unaccepted.pop(event.key, None)
            if one is None:
                return
            await self._open_turn(one.cause, one.origin)
            if one.read_up_to is not None:
                await self.read(one.read_up_to, one.seq)
        elif isinstance(event, Ask):
            await self._projects.on_ask(self, event)
        else:
            turn = await self._open_turn("agent")
            if isinstance(event, Item):
                await self._item(turn, event)
            else:
                await self._delta(turn, event)

    async def _open_turn(self, cause: str, origin: Optional[str] = None) -> _Turn:
        """The open turn, or one the backend started by itself (it woke on its own, or ran a steer
        that missed its turn)."""
        if self.turn is None:
            self.turn = _Turn(_wire_id("t"), cause)
            started = self._scoped(turn=self.turn.id, cause=cause)
            if origin is not None:
                started["from"] = origin
            started["started_at"] = now_iso()
            await self._log("turn.started", started)
        return self.turn

    async def _item(self, turn: _Turn, event: Item) -> None:
        if event.key in turn.quiet:
            if event.status != "in_progress":
                turn.quiet.discard(event.key)
            return
        call = event.call
        if call is not None and call.server == tag.SERVER and event.key not in turn.items:
            if call.tool in tag.QUIET:
                if event.status == "in_progress":
                    turn.quiet.add(event.key)
                return
            if call.tool == tag.OPEN_THREAD and event.status == "in_progress":
                await self._park(turn, event.key, str(call.arguments.get("brief") or ""))
                return
        fields = dict(event.fields)
        if event.kind == "plan":
            await self._plan(turn, fields.get("steps"))
            return
        live = turn.items.get(event.key)
        if event.status == "in_progress":
            if live is not None:
                if live.kind != "thread":
                    live.fields.update({key: value for key, value in fields.items() if key != "text"})
                return
            live = _Live(_wire_id("i"), event.kind, fields, now_iso(), text=str(fields.pop("text", "") or ""))
            turn.items[event.key] = live
            if self.top_level and event.kind == "message" and reply_state(live.text, False) == "hold":
                live.held = True
                return
            if event.kind == "tool" and fields.get("tool") in _EDITS:
                files = await self._projects.on_edit(self, live.id, event)
                if files is not None:
                    live.fields["files"] = files
            await self._start_item(turn, live)
            return
        if live is None:  # instant: it completes as it starts
            live = _Live(_wire_id("i"), event.kind, {}, now_iso())
        else:
            del turn.items[event.key]
        if "text" in fields:
            live.text = str(fields.pop("text") or "")
        if live.kind != "thread":  # a thread item's one field is the server's own
            live.fields.update(fields)
        await self._complete_item(turn, live, "declined" if live.declined else event.status)

    async def _park(self, turn: _Turn, key: str, brief: str) -> None:
        """An ``open_thread`` call's item: its thread id is given now, and the item waits, parked by its
        brief, for the call to claim it (the call reaches the bus on its own way, before or after)."""
        thread = str(uuid.uuid4()).upper()
        live = _Live(_wire_id("i"), "thread", {"opened": thread}, now_iso())
        turn.items[key] = live
        await self._start_item(turn, live)
        async with self._parking:
            self._parked.setdefault(brief, []).append((live.id, thread))
            self._parking.notify_all()

    async def claim(self, brief: str) -> Optional[tuple[str, str]]:
        """``(item id, thread id)`` of the ``open_thread`` item seen with ``brief``, waiting a little for
        it; None when none shows."""
        async with self._parking:
            try:
                await asyncio.wait_for(self._parking.wait_for(lambda: brief in self._parked), CLAIM_SECONDS)
            except TimeoutError:
                return None
            parked = self._parked[brief]
            claimed = parked.pop(0)
            if not parked:
                del self._parked[brief]
            return claimed

    def opened(self, thread: str) -> None:
        """The thread of a parked ``open_thread`` item is open: the item completes ``completed``."""
        self._opened.add(thread)

    async def _delta(self, turn: _Turn, event: Delta) -> None:
        if event.steps is not None:
            await self._plan(turn, event.steps)
            return
        live = turn.items.get(event.key)
        if live is None:
            return
        if event.output is not None:
            live.fields["output_lines"] = await self._projects.on_output(self, live.id, event.output)
        if event.text is None:
            return
        live.text = event.text
        if live.held:
            if reply_state(live.text, False) == "hold":
                return
            await self._start_item(turn, live)
        else:
            self._grew(live)

    async def _plan(self, turn: _Turn, steps: Any) -> None:
        """One plan item per turn, whatever the backend sends: started by the first, grown by the rest,
        completed with the turn."""
        steps = list(steps or [])
        if turn.plan is None:
            turn.plan = _Live(_wire_id("i"), "plan", {}, now_iso(), steps=steps)
            await self._start_item(turn, turn.plan)
        else:
            turn.plan.steps = steps
            self._grew(turn.plan)

    async def _start_item(self, turn: _Turn, live: _Live) -> None:
        live.held = False
        if live.kind == "message":
            turn.spoke = True
        fields = {key: value for key, value in live.fields.items() if key not in _END_KEYS}
        await self._log("item.started", self._scoped(id=live.id, turn=turn.id, kind=live.kind, **fields,
                                                    status="in_progress", started_at=live.started_at))
        live.logged = True
        if live.text or live.steps:
            self._grew(live)

    async def _complete_item(self, turn: _Turn, live: _Live, status: str) -> None:
        if live.kind == "thread":
            # Whatever the backend says of the call: a thread item that opened none is failed (no row).
            status = "completed" if live.fields["opened"] in self._opened else "failed"
            self._opened.discard(live.fields["opened"])
        if self.top_level and live.kind == "message" and not live.logged:
            state = reply_state(live.text, True)
            if state == "silent":
                turn.silenced = True
            if state != "show":
                return
            await self._start_item(turn, live)
        if live.kind == "message":
            turn.spoke = True  # an instant one too: it completes as it starts
        self._dirty.pop(live, None)
        item = self._scoped(id=live.id, turn=turn.id, kind=live.kind, **live.fields, started_at=live.started_at)
        if live.kind in _TEXT_KINDS:
            item["text"] = live.text
        elif live.kind == "plan":
            item["steps"] = live.steps or []
        await self._log("item.completed", settle_item(item, status, now_iso()))
        self._projects.on_completed(self, item)

    async def _end_turn(self, end: TurnEnd, *, start_next: bool = True) -> None:
        """Under the gate: the one place a turn ends. Its open items settle, ``turn.completed`` is
        logged, and then whatever waited for it (steers held through an interrupt, every queued
        message) starts the next turn together."""
        turn = self.turn
        if turn is None:
            return
        # What the backend asked in this turn it can no longer take an answer to.
        await self._projects.cancel_requests(self, backend_only=True)
        status = "interrupted" if turn.interrupting and end.status == "completed" else end.status
        settled = "completed" if status == "completed" else "failed"
        for live in list(turn.items.values()):
            await self._complete_item(turn, live, "declined" if live.declined else settled)
        if turn.plan is not None:
            await self._complete_item(turn, turn.plan, settled)
        completed = self._scoped(turn=turn.id, status=status)
        if status == "failed":
            completed["error"] = end.error or "The turn failed."
        completed.update(silent=turn.silenced and not turn.spoke, usage=end.usage, finished_at=now_iso())
        await self._log("turn.completed", completed)
        self.turn = None
        self._projects.on_turn_end(self)
        self._parked.clear()  # its calls are over: what they never claimed opened nothing
        self.idle_since = time.monotonic()
        await self._remember(self.backend)
        if not start_next:
            return
        planned = self._spec is not None and self._spec.plan
        await self._submit()
        if self.turn is None:
            if planned and status == "completed" and turn.spoke:
                await self._projects.on_plan(self)
            self._projects.on_idle(self)

    # ── live growth ────────────────────────────────────────────────────────────────────────────

    def _grew(self, live: _Live) -> None:
        self._dirty[live] = None
        if self._flush is None:
            self._flush = asyncio.get_running_loop().call_later(DELTA_SECONDS, self._send_deltas)

    def _send_deltas(self) -> None:
        self._flush = None
        dirty, self._dirty = self._dirty, {}
        adapter = self._projects.adapter
        for live in dirty:
            if live.kind == "plan":
                adapter._live("item.delta", self._scoped(id=live.id, **{"from": 0}, steps=live.steps))
                continue
            start = _common_prefix(live.shown, live.text)
            if start == len(live.text) == len(live.shown):
                continue
            adapter._live("item.delta", self._scoped(id=live.id, **{"from": start}, text=live.text[start:]))
            live.shown = live.text

    def live_frames(self) -> list[str]:
        """Each running item with text or steps, whole, from 0: what a phone catching up shows."""
        turn = self.turn
        if turn is None:
            return []
        frames = []
        for live in [*turn.items.values(), *([turn.plan] if turn.plan else [])]:
            if not live.logged:
                continue
            if live.kind == "plan" and live.steps:
                frames.append(event_frame("item.delta", self._scoped(id=live.id, **{"from": 0}, steps=live.steps)))
            elif live.text:
                frames.append(event_frame("item.delta", self._scoped(id=live.id, **{"from": 0}, text=live.text)))
        return frames


async def _close(backend: BackendSession, thread: str) -> None:
    """End a backend session's process. One that fails to end cleanly is gone to the runtime all the
    same: what follows (its turn's end, the next process) must not wait on it."""
    try:
        await backend.close()
    except Exception as exc:
        logger.warning("Aleph: closing thread %s failed: %s", thread, exc)


class SessionHub:
    """The sessions of every project's threads, made on a thread's first input."""

    def __init__(self, projects: Projects) -> None:
        self.projects = projects
        self.sessions: dict[str, Session] = {}
        self._ended: set[str] = set()  # deleted projects: their threads take no input
        self._tasks: set[asyncio.Task] = set()
        self._reaper: Optional[asyncio.Task] = None

    def session(self, thread: ThreadRow) -> Session:
        if thread.project in self._ended:
            raise RpcError(INVALID_PARAMS, "no thread with that id")
        session = self.sessions.get(thread.id)
        if session is None:
            session = self.sessions[thread.id] = Session(self, thread)
        return session

    def start(self, waiting: list[ThreadRow], unbriefed: list[ThreadRow]) -> None:
        """At boot: start the threads whose queued messages waited through the restart, and those the
        agent opened whose brief never started their first turn (the others resume on their next
        input); and the reaper."""
        for coroutine in [*(self.session(thread).resume() for thread in waiting),
                          *(self.first_turn(thread) for thread in unbriefed)]:
            task = asyncio.create_task(coroutine)
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
        self._reaper = asyncio.create_task(self._reap_forever())

    async def first_turn(self, thread: ThreadRow) -> None:
        """An agent-opened thread's first turn: its brief, the agent's own."""
        try:
            await self.session(thread).submit(thread.payload["brief"], cause="agent")
        except Exception as exc:
            # Its turn failed in the log already, with the reason.
            logger.warning("Aleph: thread %s could not start: %s", thread.id, exc)

    async def stop(self) -> None:
        """The server stops: close every process without logging; the next start settles what was
        open and resumes the sessions."""
        for task in [*self._tasks, *([self._reaper] if self._reaper else [])]:
            task.cancel()
        sessions, self.sessions = list(self.sessions.values()), {}
        for session in sessions:
            session.dispose()
            backend, session.backend = session.backend, None
            if backend is not None:
                await _close(backend, session.thread)

    async def end(self, project_id: str, thread_id: Optional[str] = None) -> None:
        """Close the sessions of a project (it is being deleted: they go for good) or of one thread."""
        if thread_id is None:
            self._ended.add(project_id)
        for session in [session for session in self.sessions.values() if session.project == project_id
                        and thread_id in (None, session.thread)]:
            await session.close()
            if thread_id is None:
                session.dispose()
                del self.sessions[session.thread]

    async def reap(self, now: Optional[float] = None) -> list[str]:
        """Close every idle thread process ``Session.reap`` allows; the threads closed."""
        now = time.monotonic() if now is None else now
        return [session.thread for session in list(self.sessions.values()) if await session.reap(now)]

    async def _reap_forever(self) -> None:
        while True:
            await asyncio.sleep(REAP_EVERY)
            try:
                for thread in await self.reap():
                    logger.info("Aleph: closed the idle process of thread %s", thread)
            except Exception:
                logger.exception("Aleph: reaping idle threads failed")

    def live_frames(self) -> list[str]:
        return [frame for session in self.sessions.values() for frame in session.live_frames()]
