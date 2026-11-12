"""Aleph's durable state: the append-only event log, message index, chats, file registry, paired
devices (by their static X25519 public key), the pending pairing secret, and the server's static key.

One SQLite file under ``$HERMES_HOME/aleph/``. The log is what a phone replays after ``hello``, one log
for all chats; ``seq`` is assigned here and increases by exactly one per logged event. Deleting a side
chat takes its events out of the log, so a replay may skip the seqs it had. Methods are synchronous and thread-safe (one connection behind a lock); the adapter
calls them through ``asyncio.to_thread`` under its own append lock, which also fixes the order in
which events are broadcast.

``hermes aleph pair|devices|revoke`` open the same file from another process: pairing secrets and
revocations reach the gateway through it (the gateway polls for revoked devices).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# The conversation's layout. A store written under another one is not converted: its conversation
# (events, messages, chats, steps and their files) is dropped, keeping pairings and the agent's profile.
# Until Aleph's first release the number stays 1 (the current layout is always v1), and a deploy that
# changes the layout drops or converts the store by hand. From the first release on, every layout
# change bumps it.
SCHEMA_VERSION = "1"
_CONVERSATION_TABLES = ("events", "messages", "chats", "activity")
_META = "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);"
_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    seq INTEGER PRIMARY KEY,
    type TEXT NOT NULL,
    chat TEXT,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS events_chat ON events (chat);
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    seq INTEGER NOT NULL,
    chat TEXT NOT NULL,
    sender TEXT NOT NULL,
    kind TEXT NOT NULL,
    text TEXT NOT NULL,
    attachments TEXT NOT NULL,
    dispatched INTEGER NOT NULL DEFAULT 0,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL,
    reactions TEXT NOT NULL DEFAULT '{}',
    read INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS messages_chat ON messages (chat, seq);
CREATE TABLE IF NOT EXISTS chats (
    id TEXT PRIMARY KEY,
    title TEXT,
    created_at TEXT NOT NULL,
    created_seq INTEGER NOT NULL,
    last_active_at REAL NOT NULL,
    deleted_seq INTEGER
);
CREATE TABLE IF NOT EXISTS files (
    id TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    path TEXT NOT NULL,
    agent_path TEXT
);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS devices (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    model TEXT NOT NULL,
    public_key BLOB NOT NULL UNIQUE,
    created_at REAL NOT NULL,
    last_seen_at REAL,
    push_token TEXT,
    push_environment TEXT
);
CREATE TABLE IF NOT EXISTS activity (
    rowid INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL UNIQUE,
    chat TEXT NOT NULL,
    step TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS pairing (
    secret BLOB NOT NULL,
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL
);
"""

# The pairing secret: 32 random bytes in the QR code, the PSK of the pairing handshake.
SECRET_BYTES = 32
PAIRING_TTL_SECONDS = 10 * 60
SERVER_KEY_FILE = "server.key"


def fingerprint(public_key: bytes) -> str:
    """A short, readable fingerprint of a device key: ``3f9a 1c2b 77d0 e4a1``."""
    digest = hashlib.sha256(public_key).hexdigest()[:16]
    return " ".join(digest[i:i + 4] for i in range(0, 16, 4))


def load_or_create_server_key(root: Path) -> bytes:
    """The server's static X25519 private key (32 raw bytes in ``server.key``, mode 600), created the
    first time it is needed. The gateway and the CLI may race to create it: the loser reads it."""
    from plugins.platforms.aleph.noise import KeyPair
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / SERVER_KEY_FILE
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        private = path.read_bytes()
        if len(private) != 32:
            raise ValueError(f"{path} is not a 32-byte X25519 private key")
        return private
    private = KeyPair.generate().private
    with os.fdopen(fd, "wb") as handle:
        handle.write(private)
    return private


@dataclass(frozen=True)
class LoggedEvent:
    seq: int
    type: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class StoredMessage:
    id: str
    seq: int
    chat: str
    sender: str
    kind: str
    text: str
    attachments: list[dict[str, Any]]
    dispatched: bool


@dataclass(frozen=True)
class Chat:
    """A side chat (``main`` is not stored: it always exists)."""
    id: str
    title: Optional[str]
    created_at: str
    created_seq: int
    last_active_at: float
    deleted_seq: Optional[int]


_CHAT_COLUMNS = "id, title, created_at, created_seq, last_active_at, deleted_seq"


def _chat(row) -> Chat:
    return Chat(row[0], row[1], row[2], int(row[3]), float(row[4]), int(row[5]) if row[5] is not None else None)


@dataclass(frozen=True)
class StoredFile:
    id: str
    description: dict[str, Any]
    path: str  # Aleph's own copy (what GET serves)
    agent_path: Optional[str]  # the gateway media-cache copy handed to the agent, for uploads


@dataclass(frozen=True)
class Device:
    id: str
    name: str
    model: str
    public_key: bytes
    created_at: float
    last_seen_at: Optional[float]
    push_token: Optional[str]
    push_environment: Optional[str]


_DEVICE_COLUMNS = "id, name, model, public_key, created_at, last_seen_at, push_token, push_environment"


def _device(row) -> Device:
    return Device(row[0], row[1], row[2], bytes(row[3]), float(row[4]),
                  float(row[5]) if row[5] is not None else None, row[6], row[7])


class AlephStore:
    """The chats' log. ``history_limit`` bounds how many events are kept; older ones are pruned and
    a ``hello`` asking for them gets ``reset: true``."""

    def __init__(self, root: Path, *, history_limit: int = 20000) -> None:
        self.root = Path(root)
        self.files_dir = self.root / "files"
        self.files_dir.mkdir(parents=True, exist_ok=True)
        self.history_limit = max(100, int(history_limit))
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(self.root / "aleph.db"), check_same_thread=False, isolation_level=None)
        # The CLI writes to the same file while the gateway runs: wait for the other side's lock.
        self._db.execute("PRAGMA busy_timeout=5000")
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        with self._lock:
            self._db.executescript(_META)
            if self._meta_locked("schema") != SCHEMA_VERSION:
                self._drop_conversation_locked()
            self._db.executescript(_SCHEMA)
            self._set_meta_locked("schema", SCHEMA_VERSION)

    def _drop_conversation_locked(self) -> None:
        """Start the conversation over (a store from another layout): phones that knew it get
        ``reset: true``. Pairings, the server key and the agent's profile stay."""
        for table in _CONVERSATION_TABLES:
            self._db.execute(f"DROP TABLE IF EXISTS {table}")
        avatar = self._meta_locked("profile_avatar") or ""
        exists = self._db.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'files'").fetchone()
        if exists is not None:
            for file_id, path in self._db.execute("SELECT id, path FROM files WHERE id != ?", (avatar,)).fetchall():
                Path(path).unlink(missing_ok=True)
            self._db.execute("DELETE FROM files WHERE id != ?", (avatar,))
        self._db.execute("DELETE FROM meta WHERE key NOT LIKE 'profile_%'")

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # ── log ────────────────────────────────────────────────────────────────────────────────────

    def latest(self) -> int:
        with self._lock:
            return self._latest_locked()

    def _latest_locked(self) -> int:
        # A log that lost its newest events (pruned, or a deleted chat's) keeps counting from where
        # it was, so seq never repeats.
        row = self._db.execute("SELECT MAX(seq) FROM events").fetchone()
        return max(int(row[0] or 0), int(self._meta_locked("seq_floor") or 0))

    def oldest(self) -> Optional[int]:
        with self._lock:
            row = self._db.execute("SELECT MIN(seq) FROM events").fetchone()
            return int(row[0]) if row[0] is not None else None

    def pruned_through(self) -> int:
        """The newest seq pruned for age: a phone that stored less has missed what it cannot get
        back. (A deleted chat's events are gone too, but deliberately: that is no reset.)"""
        with self._lock:
            return int(self._meta_locked("pruned_through") or 0)

    def append(self, event_type: str, payload: dict[str, Any], *,
               message: Optional[dict[str, Any]] = None) -> LoggedEvent:
        """Log one event (and index the message it records) atomically; returns it with its seq. The
        event's chat is its payload's ``chat``."""
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                event = self._append_locked(event_type, payload, message)
                self._db.execute("COMMIT")
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
        return event

    def _append_locked(self, event_type: str, payload: dict[str, Any],
                       message: Optional[dict[str, Any]] = None) -> LoggedEvent:
        chat = payload.get("chat")
        if message is not None:
            self._keep_time_order_locked(chat, payload)
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        seq = self._latest_locked() + 1
        self._db.execute("INSERT INTO events (seq, type, chat, payload) VALUES (?, ?, ?, ?)",
                         (seq, event_type, chat, body))
        if message is not None:
            current = {key: value for key, value in payload.items() if key != "draft_id"}
            self._db.execute(
                "INSERT INTO messages (id, seq, chat, sender, kind, text, attachments, dispatched, payload, "
                "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?)",
                (message["id"], seq, chat, message["sender"], message.get("kind") or "text",
                 message.get("text") or "", json.dumps(message.get("attachments") or []),
                 json.dumps(current, ensure_ascii=False, separators=(",", ":")), payload.get("created_at") or ""))
            self._db.execute("UPDATE chats SET last_active_at = ? WHERE id = ?", (time.time(), chat))
        else:
            self._fold_locked(event_type, chat, payload)
        self._prune_locked(seq)
        return LoggedEvent(seq, event_type, payload)

    def _keep_time_order_locked(self, chat: Any, payload: dict[str, Any]) -> None:
        """A message is never older than the one logged before it in its chat (the clock stepped
        back, or two sends raced to the log): it takes that one's time."""
        stamp = payload.get("created_at")
        row = self._db.execute("SELECT created_at FROM messages WHERE chat = ? ORDER BY seq DESC LIMIT 1",
                               (chat,)).fetchone()
        if not isinstance(stamp, str) or row is None or not row[0]:
            return
        try:
            if datetime.fromisoformat(stamp) < datetime.fromisoformat(row[0]):
                payload["created_at"] = row[0]
        except ValueError:
            pass

    def _fold_locked(self, event_type: str, chat: Any, payload: dict[str, Any]) -> None:
        """Keep each message's current state in step with the event that changes it, in the same
        transaction, so a history page never disagrees with the events around it."""
        if event_type == "message.edit":
            row = self._db.execute("SELECT payload FROM messages WHERE id = ?", (payload.get("id"),)).fetchone()
            if row is not None:
                current = json.loads(row[0])
                current["text"] = payload.get("text") or ""
                self._db.execute("UPDATE messages SET text = ?, payload = ? WHERE id = ?",
                                 (current["text"], json.dumps(current, ensure_ascii=False, separators=(",", ":")),
                                  payload.get("id")))
        elif event_type == "message.delete":
            self._db.execute("DELETE FROM messages WHERE id = ?", (payload.get("id"),))
        elif event_type == "reaction":
            row = self._db.execute("SELECT reactions FROM messages WHERE id = ?",
                                   (payload.get("message_id"),)).fetchone()
            if row is not None:
                reactions = json.loads(row[0])
                reactions[str(payload.get("by"))] = payload.get("emoji")
                self._db.execute("UPDATE messages SET reactions = ? WHERE id = ?",
                                 (json.dumps(reactions, ensure_ascii=False), payload.get("message_id")))
        elif event_type == "receipt":
            row = self._db.execute("SELECT seq FROM messages WHERE id = ? AND chat = ?",
                                   (payload.get("read_up_to"), chat)).fetchone()
            if row is not None:
                self._db.execute("UPDATE messages SET read = 1 WHERE chat = ? AND sender = 'user' AND seq <= ? "
                                 "AND read = 0", (chat, int(row[0])))

    def _prune_locked(self, latest: int) -> None:
        # Prune in batches (10% slack) so a full log does not delete one row per append.
        cutoff = latest - self.history_limit
        oldest = self._db.execute("SELECT MIN(seq) FROM events").fetchone()[0]
        if oldest is not None and cutoff - int(oldest) >= self.history_limit // 10:
            self._db.execute("DELETE FROM events WHERE seq <= ?", (cutoff,))
            self._set_meta_locked("seq_floor", str(latest))
            self._set_meta_locked("pruned_through", str(cutoff))

    def events_after(self, after: int, *, until: Optional[int] = None) -> list[LoggedEvent]:
        query, args = "SELECT seq, type, payload FROM events WHERE seq > ?", [int(after)]
        if until is not None:
            query += " AND seq <= ?"
            args.append(int(until))
        with self._lock:
            rows = self._db.execute(query + " ORDER BY seq", args).fetchall()
        return [LoggedEvent(int(seq), kind, json.loads(payload)) for seq, kind, payload in rows]

    # ── messages ───────────────────────────────────────────────────────────────────────────────

    def message(self, message_id: str) -> Optional[StoredMessage]:
        with self._lock:
            row = self._db.execute(
                "SELECT id, seq, chat, sender, kind, text, attachments, dispatched FROM messages WHERE id = ?",
                (message_id,)).fetchone()
        if row is None:
            return None
        return StoredMessage(row[0], int(row[1]), row[2], row[3], row[4], row[5], json.loads(row[6]), bool(row[7]))

    def mark_dispatched(self, message_id: str) -> None:
        with self._lock:
            self._db.execute("UPDATE messages SET dispatched = 1 WHERE id = ?", (message_id,))

    def history(self, chat: str, before: int, limit: int) -> tuple[list[dict[str, Any]], Optional[int]]:
        """A history page: the newest ``limit`` messages of ``chat`` logged before ``before`` (a seq),
        oldest first, each as its ``message.new`` stands now (edits applied, deleted ones gone) with its
        ``seq``, ``reactions`` and, for the person's, ``read``; and the ``before`` of the next page, or
        None when nothing older exists."""
        with self._lock:
            rows = self._db.execute(
                "SELECT seq, sender, payload, reactions, read FROM messages WHERE chat = ? AND seq < ? "
                "ORDER BY seq DESC LIMIT ?", (chat, int(before), int(limit) + 1)).fetchall()
        more = len(rows) > limit
        page = []
        for seq, sender, body, reactions, read in reversed(rows[:limit]):
            message = json.loads(body)
            marks = json.loads(reactions)
            message.update(seq=int(seq), reactions={"user": marks.get("user"), "agent": marks.get("agent")})
            if sender == "user":
                message["read"] = bool(read)
            page.append(message)
        return page, (page[0]["seq"] if more and page else None)

    def user_message_ids(self, chat: str, limit: int) -> list[str]:
        """The person's most recent message ids in a chat, newest first."""
        with self._lock:
            rows = self._db.execute("SELECT id FROM messages WHERE chat = ? AND sender = 'user' "
                                    "ORDER BY seq DESC LIMIT ?", (chat, limit)).fetchall()
        return [row[0] for row in rows]

    def user_message_seq(self, message_id: str) -> Optional[int]:
        message = self.message(message_id)
        return message.seq if message is not None and message.sender == "user" else None

    # ── chats ──────────────────────────────────────────────────────────────────────────────────

    def chat(self, chat_id: str) -> Optional[Chat]:
        with self._lock:
            row = self._db.execute(f"SELECT {_CHAT_COLUMNS} FROM chats WHERE id = ?", (chat_id,)).fetchone()
        return _chat(row) if row is not None else None

    def chats(self) -> list[Chat]:
        """The side chats that exist (not deleted), oldest first."""
        with self._lock:
            rows = self._db.execute(f"SELECT {_CHAT_COLUMNS} FROM chats WHERE deleted_seq IS NULL "
                                    "ORDER BY created_seq").fetchall()
        return [_chat(row) for row in rows]

    def create_chat(self, chat_id: str, title: Optional[str], created_at: str) -> tuple[Chat, Optional[LoggedEvent]]:
        """Open a side chat and log its ``chat.new``, atomically; one that exists (or existed) is
        returned as it is, with no event."""
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                row = self._db.execute(f"SELECT {_CHAT_COLUMNS} FROM chats WHERE id = ?", (chat_id,)).fetchone()
                event = None
                if row is None:
                    event = self._append_locked("chat.new", {"chat": chat_id, "title": title, "created_at": created_at})
                    self._db.execute("INSERT INTO chats (id, title, created_at, created_seq, last_active_at) "
                                     "VALUES (?, ?, ?, ?, ?)", (chat_id, title, created_at, event.seq, time.time()))
                    row = self._db.execute(f"SELECT {_CHAT_COLUMNS} FROM chats WHERE id = ?", (chat_id,)).fetchone()
                self._db.execute("COMMIT")
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
        return _chat(row), event

    def set_chat_title(self, chat_id: str, title: str) -> None:
        with self._lock:
            self._db.execute("UPDATE chats SET title = ? WHERE id = ?", (title, chat_id))

    def idle_chats(self, before: float) -> list[str]:
        """Side chats nobody has written in since ``before`` (a time)."""
        with self._lock:
            rows = self._db.execute("SELECT id FROM chats WHERE deleted_seq IS NULL AND last_active_at < ?",
                                    (before,)).fetchall()
        return [row[0] for row in rows]

    def delete_chat(self, chat_id: str) -> LoggedEvent:
        """Drop a side chat's events, messages, steps and files, and log ``chat.delete``, atomically.
        The chat is remembered as deleted (its id is not reused)."""
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                file_ids = []
                for (attachments,) in self._db.execute("SELECT attachments FROM messages WHERE chat = ?", (chat_id,)):
                    file_ids += [str(item.get("id")) for item in json.loads(attachments) if isinstance(item, dict)]
                paths = []
                for file_id in file_ids:
                    row = self._db.execute("SELECT path FROM files WHERE id = ?", (file_id,)).fetchone()
                    if row is not None:
                        paths.append(row[0])
                        self._db.execute("DELETE FROM files WHERE id = ?", (file_id,))
                self._set_meta_locked("seq_floor", str(self._latest_locked()))
                for table in ("events", "messages", "activity"):
                    self._db.execute(f"DELETE FROM {table} WHERE chat = ?", (chat_id,))
                for key in (f"last_user_message:{chat_id}", f"last_receipt:{chat_id}"):
                    self._db.execute("DELETE FROM meta WHERE key = ?", (key,))
                event = self._append_locked("chat.delete", {"chat": chat_id})
                self._db.execute("UPDATE chats SET deleted_seq = ? WHERE id = ?", (event.seq, chat_id))
                self._db.execute("COMMIT")
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
        for path in paths:
            Path(path).unlink(missing_ok=True)
        return event

    # ── files ──────────────────────────────────────────────────────────────────────────────────

    def add_file(self, description: dict[str, Any], path: str, agent_path: Optional[str] = None) -> None:
        with self._lock:
            self._db.execute("INSERT INTO files (id, description, path, agent_path) VALUES (?, ?, ?, ?)",
                             (description["id"], json.dumps(description, ensure_ascii=False), path, agent_path))

    def delete_file(self, file_id: str) -> None:
        """Forget a stored file and remove its ciphertext (a replaced avatar)."""
        with self._lock:
            row = self._db.execute("SELECT path FROM files WHERE id = ?", (file_id,)).fetchone()
            self._db.execute("DELETE FROM files WHERE id = ?", (file_id,))
        if row is not None:
            Path(row[0]).unlink(missing_ok=True)

    def file(self, file_id: str) -> Optional[StoredFile]:
        with self._lock:
            row = self._db.execute("SELECT id, description, path, agent_path FROM files WHERE id = ?",
                                   (file_id,)).fetchone()
        return StoredFile(row[0], json.loads(row[1]), row[2], row[3]) if row is not None else None

    # ── devices ──────────────────────────────────────────────────────────────────────────────

    def devices(self) -> list[Device]:
        with self._lock:
            rows = self._db.execute(f"SELECT {_DEVICE_COLUMNS} FROM devices ORDER BY created_at").fetchall()
        return [_device(row) for row in rows]

    def device(self, device_id: str) -> Optional[Device]:
        with self._lock:
            row = self._db.execute(f"SELECT {_DEVICE_COLUMNS} FROM devices WHERE id = ?", (device_id,)).fetchone()
        return _device(row) if row is not None else None

    def device_for_key(self, public_key: bytes) -> Optional[Device]:
        """The device a static key belongs to; ``None`` for an unknown or revoked key."""
        with self._lock:
            row = self._db.execute(f"SELECT {_DEVICE_COLUMNS} FROM devices WHERE public_key = ?",
                                   (bytes(public_key),)).fetchone()
        return _device(row) if row is not None else None

    def device_ids(self) -> set[str]:
        with self._lock:
            return {row[0] for row in self._db.execute("SELECT id FROM devices")}

    def touch_device(self, device_id: str, at: Optional[float] = None) -> None:
        with self._lock:
            self._db.execute("UPDATE devices SET last_seen_at = ? WHERE id = ?", (at or time.time(), device_id))

    def revoke_device(self, device_id: str) -> bool:
        """Forget a device: its key no longer completes a handshake, and its push token goes with it."""
        with self._lock:
            return self._db.execute("DELETE FROM devices WHERE id = ?", (device_id,)).rowcount > 0

    def set_push_token(self, device_id: str, token: Optional[str], environment: Optional[str]) -> None:
        """Register (or with ``None`` drop) a device's APNs token. A token moves: it is cleared from
        any other device that had it (the same phone paired again), so it is never pushed twice."""
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                if token is not None:
                    self._db.execute("UPDATE devices SET push_token = NULL, push_environment = NULL "
                                     "WHERE push_token = ? AND id != ?", (token, device_id))
                self._db.execute("UPDATE devices SET push_token = ?, push_environment = ? WHERE id = ?",
                                 (token, environment if token is not None else None, device_id))
                self._db.execute("COMMIT")
            except BaseException:
                self._db.execute("ROLLBACK")
                raise

    def drop_push_token(self, token: str) -> None:
        """APNs said the token is dead (410 / BadDeviceToken)."""
        with self._lock:
            self._db.execute("UPDATE devices SET push_token = NULL, push_environment = NULL WHERE push_token = ?",
                             (token,))

    # ── pairing ──────────────────────────────────────────────────────────────────────────────

    def new_pairing(self, *, ttl: float = PAIRING_TTL_SECONDS) -> tuple[bytes, float]:
        """A fresh single-use pairing secret and when it expires. It replaces any pending one."""
        secret = secrets.token_bytes(SECRET_BYTES)
        now = time.time()
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                self._db.execute("DELETE FROM pairing")
                self._db.execute("INSERT INTO pairing (secret, created_at, expires_at) VALUES (?, ?, ?)",
                                 (secret, now, now + ttl))
                self._db.execute("COMMIT")
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
        return secret, now + ttl

    def pending_pairing(self) -> Optional[tuple[bytes, float]]:
        """(secret, expires_at) of the pending pairing, if one is still valid."""
        with self._lock:
            row = self._db.execute("SELECT secret, expires_at FROM pairing WHERE expires_at > ?",
                                   (time.time(),)).fetchone()
        return (bytes(row[0]), float(row[1])) if row is not None else None

    def complete_pairing(self, secret: bytes, public_key: bytes, *, name: str, model: str) -> Optional[Device]:
        """Spend ``secret`` (it must still be the pending one) and record the device it paired. A key
        that is already paired keeps its device (and id), under the new name."""
        now = time.time()
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                row = self._db.execute("SELECT secret FROM pairing WHERE expires_at > ?", (now,)).fetchone()
                if row is None or not hmac.compare_digest(bytes(row[0]), bytes(secret)):
                    self._db.execute("COMMIT")
                    return None
                self._db.execute("DELETE FROM pairing")
                existing = self._db.execute("SELECT id FROM devices WHERE public_key = ?",
                                            (bytes(public_key),)).fetchone()
                if existing is not None:
                    device_id = existing[0]
                    self._db.execute("UPDATE devices SET name = ?, model = ? WHERE id = ?", (name, model, device_id))
                else:
                    device_id = f"d_{secrets.token_hex(8)}"
                    self._db.execute("INSERT INTO devices (id, name, model, public_key, created_at) "
                                     "VALUES (?, ?, ?, ?, ?)", (device_id, name, model, bytes(public_key), now))
                row = self._db.execute(f"SELECT {_DEVICE_COLUMNS} FROM devices WHERE id = ?", (device_id,)).fetchone()
                self._db.execute("COMMIT")
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
        return _device(row)

    # ── activity ───────────────────────────────────────────────────────────────────────────────

    def add_step(self, step: dict[str, Any], *, keep: int = 500) -> None:
        """Record an activity step; only the most recent ``keep`` are kept (not in the event log)."""
        with self._lock:
            self._db.execute("INSERT INTO activity (id, chat, step) VALUES (?, ?, ?)",
                             (step["id"], step["chat"], json.dumps(step, ensure_ascii=False)))
            self._db.execute("DELETE FROM activity WHERE rowid <= (SELECT MAX(rowid) FROM activity) - ?", (keep,))

    def update_step(self, step: dict[str, Any]) -> None:
        with self._lock:
            self._db.execute("UPDATE activity SET step = ? WHERE id = ?",
                             (json.dumps(step, ensure_ascii=False), step["id"]))

    def steps(self, *, chat: Optional[str] = None, before: Optional[str] = None,
              limit: int = 50) -> tuple[list[dict[str, Any]], bool]:
        """The most recent steps (of one chat, or all), newest first, older than ``before`` (a step
        id) when given; and whether there are more."""
        where, args = [], []
        with self._lock:
            if before:
                row = self._db.execute("SELECT rowid FROM activity WHERE id = ?", (before,)).fetchone()
                if row is None:
                    return [], False
                where.append("rowid < ?")
                args.append(row[0])
            if chat is not None:
                where.append("chat = ?")
                args.append(chat)
            clause = f"WHERE {' AND '.join(where)} " if where else ""
            rows = self._db.execute(f"SELECT step FROM activity {clause}ORDER BY rowid DESC LIMIT ?",
                                    (*args, limit + 1)).fetchall()
        return [json.loads(r[0]) for r in rows[:limit]], len(rows) > limit

    # ── meta ───────────────────────────────────────────────────────────────────────────────────

    def meta(self, key: str) -> Optional[str]:
        with self._lock:
            return self._meta_locked(key)

    def set_meta(self, key: str, value: str) -> None:
        with self._lock:
            self._set_meta_locked(key, value)

    def _meta_locked(self, key: str) -> Optional[str]:
        row = self._db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row is not None else None

    def _set_meta_locked(self, key: str, value: str) -> None:
        self._db.execute("INSERT INTO meta (key, value) VALUES (?, ?) "
                         "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, value))
