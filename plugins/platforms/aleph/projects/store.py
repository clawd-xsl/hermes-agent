"""The projects' tables in Aleph's store: projects, threads, turns, items and queued messages, and the
coordination journal (deliveries between a project's sessions, and the threads the agent opened).

They live in the chats' SQLite file, on its connection and behind its lock (``AlephStore`` composes
``ProjectTables``), and are folds of the log: every row changes in the transaction that appends the
event changing it, so a snapshot or a history page never disagrees with the events around it.

A project event is an ordinary ``events`` row. Its ``chat`` column holds the thread it belongs to
(the top level's thread id is the project's own), so cards, receipts and messages are found by
thread exactly as by chat; its ``project`` column lets the project's events go in one delete.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    payload TEXT NOT NULL,
    created_seq INTEGER NOT NULL,
    deleted_seq INTEGER
);
CREATE TABLE IF NOT EXISTS threads (
    id TEXT PRIMARY KEY,
    project TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_seq INTEGER NOT NULL,
    session_id TEXT,
    last_active_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS threads_project ON threads (project, created_seq);
CREATE TABLE IF NOT EXISTS turns (
    id TEXT PRIMARY KEY,
    thread TEXT NOT NULL,
    project TEXT NOT NULL,
    seq INTEGER NOT NULL,
    payload TEXT NOT NULL,
    open INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS turns_thread ON turns (thread, seq);
CREATE TABLE IF NOT EXISTS items (
    id TEXT PRIMARY KEY,
    thread TEXT NOT NULL,
    project TEXT NOT NULL,
    turn TEXT NOT NULL,
    seq INTEGER NOT NULL,
    payload TEXT NOT NULL,
    open INTEGER NOT NULL DEFAULT 1,
    out_start INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS items_thread ON items (thread, seq);
CREATE TABLE IF NOT EXISTS queued (
    id TEXT PRIMARY KEY,
    thread TEXT NOT NULL,
    project TEXT NOT NULL,
    seq INTEGER NOT NULL,
    payload TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'waiting'
);
CREATE INDEX IF NOT EXISTS queued_thread ON queued (thread, seq);
CREATE TABLE IF NOT EXISTS deliveries (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    project TEXT NOT NULL,
    source TEXT NOT NULL,
    target TEXT NOT NULL,
    request_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    text TEXT NOT NULL,
    status TEXT NOT NULL,
    detail TEXT NOT NULL DEFAULT '',
    created REAL NOT NULL,
    UNIQUE (source, request_id)
);
CREATE INDEX IF NOT EXISTS deliveries_inbox ON deliveries (target, status, seq);
CREATE TABLE IF NOT EXISTS opened (
    project TEXT NOT NULL,
    source TEXT NOT NULL,
    request_id TEXT NOT NULL,
    arguments TEXT NOT NULL,
    result TEXT NOT NULL,
    UNIQUE (source, request_id)
);
"""
TABLES = ("projects", "threads", "turns", "items", "queued", "deliveries", "opened")
# A delivery not settled yet; ``delivering`` is one being handed over right now.
PENDING = ("queued", "held", "delivering")
RATE_HOUR, RATE_MINUTE = 60, 10  # deliveries one session may send per hour, per minute
INBOX = 50  # deliveries pending for one target

# Events whose payload is the entity itself, named by its ``id``.
_PROJECT_ENTITY = frozenset({"project.new", "project.update"})
_THREAD_ENTITY = frozenset({"thread.new", "thread.update"})
# What counts as a thread being active, for the threads list's order.
_ACTIVITY = frozenset({"message.new", "message.queued", "turn.started", "turn.completed"})


class JournalError(ValueError):
    """A delivery refused, its message one the agent reads."""


@dataclass(frozen=True)
class Delivery:
    seq: int
    project: str
    source: str
    target: str
    request_id: str
    kind: str  # report or message
    text: str
    status: str  # queued, held, delivering, delivered, failed or uncertain
    detail: str
    created: float


_DELIVERY_COLUMNS = "seq, project, source, target, request_id, kind, text, status, detail, created"


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def _dump(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def top_level(project: dict[str, Any]) -> dict[str, Any]:
    """The top level of a project as a thread: it has no ``thread.new`` of its own, and its name and
    settings are the project's, so its own are always null."""
    return {"id": project["id"], "project": project["id"], "name": None, "created_at": project["created_at"],
            "opened_by": "user", "item": None, "brief": None, "model": None, "effort": None, "approval": None,
            "plan": False, "closed": False}


@dataclass(frozen=True)
class ProjectRow:
    id: str
    payload: dict[str, Any]  # as its latest project.new / project.update has it
    created_seq: int
    deleted_seq: Optional[int]


@dataclass(frozen=True)
class ThreadRow:
    id: str
    project: str
    payload: dict[str, Any]  # as its latest thread.new / thread.update has it
    created_seq: int
    session_id: Optional[str]
    last_active_at: str

    @property
    def top_level(self) -> bool:
        return self.id == self.project


@dataclass(frozen=True)
class ItemRow:
    id: str
    project: str
    payload: dict[str, Any]  # as its latest item.started / item.completed has it
    open: bool
    out_start: int  # a command's first output line still kept


class ProjectTables:
    """The projects' rows on ``AlephStore``'s connection. Methods ending in ``_locked`` run inside the
    store's lock (and, when they write, its append transaction); the others take the lock."""

    def __init__(self, db: sqlite3.Connection, lock: threading.Lock) -> None:
        self._db = db
        self._lock = lock

    @staticmethod
    def scope(event_type: str, payload: dict[str, Any]) -> tuple[Any, Optional[str]]:
        """``(chat, project)`` columns for an event: a project event is filed under its thread."""
        project = payload.get("project")
        if event_type in _PROJECT_ENTITY:
            return payload.get("id"), payload.get("id")
        if project is None:
            return payload.get("chat"), None
        if event_type in _THREAD_ENTITY:
            return payload.get("id"), project
        return payload.get("thread") or project, project

    # ── fold ───────────────────────────────────────────────────────────────────────────────────

    def fold_locked(self, seq: int, event_type: str, thread: str, payload: dict[str, Any]) -> None:
        """Bring the rows in step with one project event, in its append transaction."""
        db = self._db
        if event_type == "project.new":
            db.execute("INSERT INTO projects (id, payload, created_seq) VALUES (?, ?, ?)",
                       (payload["id"], _dump(payload), seq))
            db.execute("INSERT INTO threads (id, project, payload, created_seq, last_active_at) VALUES (?, ?, ?, ?, ?)",
                       (payload["id"], payload["id"], _dump(top_level(payload)), seq, payload["created_at"]))
        elif event_type == "project.update":
            db.execute("UPDATE projects SET payload = ? WHERE id = ?", (_dump(payload), payload["id"]))
        elif event_type == "project.delete":
            db.execute("UPDATE projects SET deleted_seq = ? WHERE id = ?", (seq, payload["project"]))
        elif event_type == "thread.new":
            db.execute("INSERT INTO threads (id, project, payload, created_seq, last_active_at) VALUES (?, ?, ?, ?, ?)",
                       (payload["id"], payload["project"], _dump(payload), seq, payload["created_at"]))
        elif event_type == "thread.update":
            db.execute("UPDATE threads SET payload = ? WHERE id = ?", (_dump(payload), payload["id"]))
        elif event_type == "thread.close":
            row = db.execute("SELECT payload FROM threads WHERE id = ?", (thread,)).fetchone()
            if row is not None:
                db.execute("UPDATE threads SET payload = ? WHERE id = ?",
                           (_dump({**json.loads(row[0]), "closed": True}), thread))
        elif event_type == "turn.started":
            db.execute("INSERT INTO turns (id, thread, project, seq, payload) VALUES (?, ?, ?, ?, ?)",
                       (payload["turn"], thread, payload["project"], seq, _dump(payload)))
        elif event_type == "turn.completed":
            row = db.execute("SELECT payload FROM turns WHERE id = ?", (payload["turn"],)).fetchone()
            if row is not None:
                db.execute("UPDATE turns SET payload = ?, open = 0 WHERE id = ?",
                           (_dump({**json.loads(row[0]), **payload}), payload["turn"]))
        elif event_type == "item.started":
            db.execute("INSERT INTO items (id, thread, project, turn, seq, payload) VALUES (?, ?, ?, ?, ?, ?)",
                       (payload["id"], thread, payload["project"], payload["turn"], seq, _dump(payload)))
        elif event_type == "item.completed":
            # An instant item (a model fallback) has no item.started: its completion is its place.
            db.execute("INSERT INTO items (id, thread, project, turn, seq, payload, open) VALUES (?, ?, ?, ?, ?, ?, 0) "
                       "ON CONFLICT(id) DO UPDATE SET payload = excluded.payload, open = 0",
                       (payload["id"], thread, payload["project"], payload["turn"], seq, _dump(payload)))
        elif event_type == "message.queued":
            db.execute("INSERT INTO queued (id, thread, project, seq, payload) VALUES (?, ?, ?, ?, ?)",
                       (payload["id"], thread, payload["project"], seq, _dump(payload)))
        elif event_type == "message.cancel":
            db.execute("UPDATE queued SET state = 'cancelled' WHERE id = ? AND state = 'waiting'", (payload["id"],))
        elif event_type == "message.new":
            # A queued message is logged again as its message.new when its turn takes it.
            db.execute("UPDATE queued SET state = 'sent' WHERE id = ? AND state = 'waiting'", (payload["id"],))
        if event_type in _ACTIVITY:
            db.execute("UPDATE threads SET last_active_at = ? WHERE id = ?", (_now_iso(), thread))

    # ── reads ──────────────────────────────────────────────────────────────────────────────────

    def project(self, project_id: str) -> Optional[ProjectRow]:
        with self._lock:
            row = self._db.execute("SELECT id, payload, created_seq, deleted_seq FROM projects WHERE id = ?",
                                   (project_id,)).fetchone()
        if row is None:
            return None
        return ProjectRow(row[0], json.loads(row[1]), int(row[2]), int(row[3]) if row[3] is not None else None)

    def live_projects(self) -> list[str]:
        """The projects that exist, oldest first."""
        with self._lock:
            rows = self._db.execute("SELECT id FROM projects WHERE deleted_seq IS NULL ORDER BY created_seq").fetchall()
        return [row[0] for row in rows]

    def thread(self, thread_id: str) -> Optional[ThreadRow]:
        with self._lock:
            row = self._db.execute("SELECT id, project, payload, created_seq, session_id, last_active_at "
                                   "FROM threads WHERE id = ?", (thread_id,)).fetchone()
        return _thread(row) if row is not None else None

    def queued(self, message_id: str) -> Optional[tuple[str, int]]:
        """``(thread, seq)`` of a message the person queued, whatever became of it: its id is taken."""
        with self._lock:
            row = self._db.execute("SELECT thread, seq FROM queued WHERE id = ?", (message_id,)).fetchone()
        return (row[0], int(row[1])) if row is not None else None

    def queued_state(self, message_id: str) -> Optional[str]:
        """``waiting``, ``sent`` or ``cancelled``, for a message the person queued."""
        with self._lock:
            row = self._db.execute("SELECT state FROM queued WHERE id = ?", (message_id,)).fetchone()
        return row[0] if row is not None else None

    def cancel_seq(self, message_id: str) -> Optional[int]:
        """The seq of the ``message.cancel`` that took a queued message back."""
        with self._lock:
            row = self._db.execute("SELECT e.seq FROM events e JOIN queued q ON q.thread = e.chat WHERE q.id = ? "
                                   "AND e.type = 'message.cancel' AND json_extract(e.payload, '$.id') = ?",
                                   (message_id, message_id)).fetchone()
        return int(row[0]) if row is not None else None

    def waiting(self, thread: str) -> list[tuple[int, dict[str, Any]]]:
        """A thread's queued messages still waiting, in order: ``(seq of message.queued, its payload)``."""
        with self._lock:
            rows = self._db.execute("SELECT seq, payload FROM queued WHERE thread = ? AND state = 'waiting' "
                                    "ORDER BY seq", (thread,)).fetchall()
        return [(int(seq), json.loads(body)) for seq, body in rows]

    def waiting_threads(self) -> list[ThreadRow]:
        """The open threads of live projects with queued messages waiting."""
        with self._lock:
            rows = self._db.execute(
                "SELECT t.id, t.project, t.payload, t.created_seq, t.session_id, t.last_active_at FROM threads t "
                "JOIN projects p ON p.id = t.project WHERE p.deleted_seq IS NULL AND EXISTS (SELECT 1 FROM queued q "
                "WHERE q.thread = t.id AND q.state = 'waiting') ORDER BY t.created_seq").fetchall()
        return [thread for thread in map(_thread, rows) if not thread.payload.get("closed")]

    def snapshot_locked(self, waiting: set[str]) -> list[dict[str, Any]]:
        """``hello``'s ``projects`` on a reset: every project that exists, oldest first, with all its
        threads but the top level (its ``project.new`` stands for it), closed ones included (their rows
        stay in the main transcript). ``waiting`` names the threads with a request open."""
        db = self._db
        running = {row[0] for row in db.execute("SELECT DISTINCT thread FROM turns WHERE open = 1")}
        queued: dict[str, list[dict[str, Any]]] = {}
        for thread, body in db.execute("SELECT thread, payload FROM queued WHERE state = 'waiting' ORDER BY seq"):
            queued.setdefault(thread, []).append(json.loads(body))
        projects = []
        for project_id, body in db.execute("SELECT id, payload FROM projects WHERE deleted_seq IS NULL "
                                           "ORDER BY created_seq").fetchall():
            threads = []
            for thread_id, thread_body, last_active_at in db.execute(
                    "SELECT id, payload, last_active_at FROM threads WHERE project = ? AND id != project "
                    "ORDER BY created_seq", (project_id,)).fetchall():
                state = "waiting" if thread_id in waiting else "running" if thread_id in running else "idle"
                threads.append({**json.loads(thread_body), "state": state, "last_active_at": last_active_at,
                                "queued": queued.get(thread_id, [])})
            projects.append({"project": json.loads(body), "threads": threads})
        return projects

    def entries_locked(self, thread: str, before: int, limit: int) -> list[dict[str, Any]]:
        """A thread's newest ``limit`` turns and items logged before ``before``, newest first, each
        ``{seq, turn}`` (its ``turn.started`` with its ``turn.completed`` fields once it ended) or
        ``{seq, item}`` (as it stands now, at its ``item.started`` seq)."""
        rows = self._db.execute(
            "SELECT seq, 'turn', payload FROM turns WHERE thread = ? AND seq < ? "
            "UNION ALL SELECT seq, 'item', payload FROM items WHERE thread = ? AND seq < ? "
            "ORDER BY seq DESC LIMIT ?", (thread, int(before), thread, int(before), int(limit))).fetchall()
        return [{"seq": int(seq), kind: json.loads(body)} for seq, kind, body in rows]

    def threads(self, project_id: str) -> list[ThreadRow]:
        """A project's threads but the top level, oldest first."""
        with self._lock:
            rows = self._db.execute("SELECT id, project, payload, created_seq, session_id, last_active_at FROM threads "
                                    "WHERE project = ? AND id != project ORDER BY created_seq", (project_id,)).fetchall()
        return [_thread(row) for row in rows]

    def thread_ids_locked(self, project_id: str) -> list[str]:
        return [row[0] for row in self._db.execute("SELECT id FROM threads WHERE project = ?", (project_id,))]

    # ── writes outside the fold ────────────────────────────────────────────────────────────────

    def set_session_id(self, thread: str, session_id: str) -> None:
        """The backend's own id for a thread's session, to resume it (no event: the phone never sees it)."""
        with self._lock:
            self._db.execute("UPDATE threads SET session_id = ? WHERE id = ?", (session_id, thread))

    def delete_project_locked(self, project_id: str) -> None:
        """Drop a project's turns, items and queued messages. Its project and thread rows stay, marked
        by ``project.delete``: their ids are never used again."""
        for table in ("turns", "items", "queued", "deliveries", "opened"):
            self._db.execute(f"DELETE FROM {table} WHERE project = ?", (project_id,))

    def settle_open(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """``(items, turns)`` still open, as they stand, oldest first: what a server that restarted
        settles as failed before anyone connects."""
        with self._lock:
            items = self._db.execute("SELECT payload FROM items WHERE open = 1 ORDER BY seq").fetchall()
            turns = self._db.execute("SELECT payload FROM turns WHERE open = 1 ORDER BY seq").fetchall()
        return [json.loads(row[0]) for row in items], [json.loads(row[0]) for row in turns]

    def unbriefed(self) -> list[ThreadRow]:
        """Open threads of live projects the agent opened whose brief never started a turn: the
        restart came between their ``thread.new`` and their first turn."""
        with self._lock:
            rows = self._db.execute(
                "SELECT t.id, t.project, t.payload, t.created_seq, t.session_id, t.last_active_at FROM threads t "
                "JOIN projects p ON p.id = t.project WHERE p.deleted_seq IS NULL "
                "AND json_extract(t.payload, '$.opened_by') = 'agent' AND json_extract(t.payload, '$.brief') IS NOT NULL "
                "AND NOT EXISTS (SELECT 1 FROM turns u WHERE u.thread = t.id) ORDER BY t.created_seq").fetchall()
        return [thread for thread in map(_thread, rows) if not thread.payload.get("closed")]

    def item(self, item_id: str) -> Optional[ItemRow]:
        with self._lock:
            row = self._db.execute("SELECT id, project, payload, open, out_start FROM items WHERE id = ?",
                                   (item_id,)).fetchone()
        return ItemRow(row[0], row[1], json.loads(row[2]), bool(row[3]), int(row[4])) if row is not None else None

    def set_out_start(self, item_id: str, out_start: int) -> None:
        """The index of a command's first output line still kept (no event: ``output.get`` says it)."""
        with self._lock:
            self._db.execute("UPDATE items SET out_start = ? WHERE id = ?", (out_start, item_id))

    # ── the coordination journal ───────────────────────────────────────────────────────────────

    def journal(self, project: str, source: str, target: str, request_id: str, kind: str, text: str,
                now: float) -> Delivery:
        """A delivery from session ``source`` to ``target``, ``queued``; the one already journaled
        under ``(source, request_id)`` when it is the same. ``JournalError`` for another delivery under
        that id, a source over its rate, a repeat of what it sent a minute ago, or a full inbox."""
        with self._lock:
            row = self._db.execute(f"SELECT {_DELIVERY_COLUMNS} FROM deliveries WHERE source = ? AND request_id = ?",
                                   (source, request_id)).fetchone()
            if row is not None:
                known = Delivery(*row)
                if (known.target, known.kind, known.text) != (target, kind, text):
                    raise JournalError("request_id already belongs to a different message")
                return known
            recent = self._db.execute("SELECT target, text, created FROM deliveries WHERE source = ? AND created > ?",
                                      (source, now - 3600)).fetchall()
            if len(recent) >= RATE_HOUR or sum(created > now - 60 for _, _, created in recent) >= RATE_MINUTE:
                raise JournalError("Message rate limit reached; do not retry immediately")
            if any(to == target and body == text and created > now - 60 for to, body, created in recent):
                raise JournalError("Duplicate message; reuse the original request_id to check its status")
            pending = self._db.execute(
                f"SELECT COUNT(*) FROM deliveries WHERE target = ? AND status IN ({','.join('?' * len(PENDING))})",
                (target, *PENDING)).fetchone()[0]
            if pending >= INBOX:
                raise JournalError("The target's inbox is full; do not retry immediately")
            cursor = self._db.execute(
                "INSERT INTO deliveries (project, source, target, request_id, kind, text, status, created) "
                "VALUES (?, ?, ?, ?, ?, ?, 'queued', ?)", (project, source, target, request_id, kind, text, now))
            return Delivery(int(cursor.lastrowid), project, source, target, request_id, kind, text, "queued", "", now)

    def delivery(self, source: str, request_id: str) -> Optional[Delivery]:
        with self._lock:
            row = self._db.execute(f"SELECT {_DELIVERY_COLUMNS} FROM deliveries WHERE source = ? AND request_id = ?",
                                   (source, request_id)).fetchone()
        return Delivery(*row) if row is not None else None

    def next_delivery(self, target: str) -> Optional[Delivery]:
        """The oldest delivery waiting for ``target`` (``queued`` or ``held``)."""
        with self._lock:
            row = self._db.execute(f"SELECT {_DELIVERY_COLUMNS} FROM deliveries WHERE target = ? "
                                   "AND status IN ('queued', 'held') ORDER BY seq LIMIT 1", (target,)).fetchone()
        return Delivery(*row) if row is not None else None

    def set_delivery(self, seq: int, status: str, detail: str = "") -> None:
        with self._lock:
            self._db.execute("UPDATE deliveries SET status = ?, detail = ? WHERE seq = ?", (status, detail, seq))

    def recover_deliveries(self, detail: str) -> int:
        """At boot: deliveries the last run was handing over may or may not have reached their target.
        They become ``uncertain``, never replayed. How many there were."""
        with self._lock:
            return self._db.execute("UPDATE deliveries SET status = 'uncertain', detail = ? WHERE status = 'delivering'",
                                    (detail,)).rowcount

    def delivery_targets(self) -> list[str]:
        """The sessions with deliveries waiting."""
        with self._lock:
            rows = self._db.execute("SELECT DISTINCT target FROM deliveries WHERE status IN ('queued', 'held')").fetchall()
        return [row[0] for row in rows]

    def opened(self, source: str, request_id: str) -> Optional[tuple[str, dict[str, Any]]]:
        """``(arguments, result)`` of the ``open_thread`` ``source`` made under ``request_id``."""
        with self._lock:
            row = self._db.execute("SELECT arguments, result FROM opened WHERE source = ? AND request_id = ?",
                                   (source, request_id)).fetchone()
        return (row[0], json.loads(row[1])) if row is not None else None

    def record_opened(self, project: str, source: str, request_id: str, arguments: str, result: dict[str, Any]) -> None:
        with self._lock:
            self._db.execute("INSERT INTO opened (project, source, request_id, arguments, result) VALUES (?, ?, ?, ?, ?)",
                             (project, source, request_id, arguments, _dump(result)))


def _thread(row) -> ThreadRow:
    return ThreadRow(row[0], row[1], json.loads(row[2]), int(row[3]), row[4], row[5])
