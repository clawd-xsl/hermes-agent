"""Aleph's durable state: the append-only event log, message index, file registry, paired devices
(by their static X25519 public key), the pending pairing secret, and the server's static key.

One SQLite file under ``$HERMES_HOME/aleph/``. The log is what a phone replays after ``hello``;
``seq`` is assigned here and increases by exactly one per logged event, so a gap on the phone means a
missed event. Methods are synchronous and thread-safe (one connection behind a lock); the adapter
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
from pathlib import Path
from typing import Any, Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    seq INTEGER PRIMARY KEY,
    type TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    seq INTEGER NOT NULL,
    sender TEXT NOT NULL,
    kind TEXT NOT NULL,
    text TEXT NOT NULL,
    attachments TEXT NOT NULL,
    dispatched INTEGER NOT NULL DEFAULT 0
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
    sender: str
    kind: str
    text: str
    attachments: list[dict[str, Any]]
    dispatched: bool


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
    """The chat's log. ``history_limit`` bounds how many events are kept; older ones are pruned and
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
        self._db.executescript(_SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # ── log ────────────────────────────────────────────────────────────────────────────────────

    def latest(self) -> int:
        with self._lock:
            return self._latest_locked()

    def _latest_locked(self) -> int:
        row = self._db.execute("SELECT MAX(seq) FROM events").fetchone()
        if row[0] is not None:
            return int(row[0])
        # An emptied log keeps counting from where it was, so seq never repeats.
        return int(self._meta_locked("seq_floor") or 0)

    def oldest(self) -> Optional[int]:
        with self._lock:
            row = self._db.execute("SELECT MIN(seq) FROM events").fetchone()
            return int(row[0]) if row[0] is not None else None

    def append(self, event_type: str, payload: dict[str, Any], *,
               message: Optional[dict[str, Any]] = None) -> LoggedEvent:
        """Log one event (and index the message it records) atomically; returns it with its seq."""
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                seq = self._latest_locked() + 1
                self._db.execute("INSERT INTO events (seq, type, payload) VALUES (?, ?, ?)", (seq, event_type, body))
                if message is not None:
                    self._db.execute(
                        "INSERT INTO messages (id, seq, sender, kind, text, attachments, dispatched) "
                        "VALUES (?, ?, ?, ?, ?, ?, 0)",
                        (message["id"], seq, message["sender"], message.get("kind") or "text",
                         message.get("text") or "", json.dumps(message.get("attachments") or [])))
                self._prune_locked(seq)
                self._db.execute("COMMIT")
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
        return LoggedEvent(seq, event_type, payload)

    def _prune_locked(self, latest: int) -> None:
        # Prune in batches (10% slack) so a full log does not delete one row per append.
        cutoff = latest - self.history_limit
        oldest = self._db.execute("SELECT MIN(seq) FROM events").fetchone()[0]
        if oldest is not None and cutoff - int(oldest) >= self.history_limit // 10:
            self._db.execute("DELETE FROM events WHERE seq <= ?", (cutoff,))
            self._set_meta_locked("seq_floor", str(latest))

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
                "SELECT id, seq, sender, kind, text, attachments, dispatched FROM messages WHERE id = ?",
                (message_id,)).fetchone()
        if row is None:
            return None
        return StoredMessage(row[0], int(row[1]), row[2], row[3], row[4], json.loads(row[5]), bool(row[6]))

    def mark_dispatched(self, message_id: str) -> None:
        with self._lock:
            self._db.execute("UPDATE messages SET dispatched = 1 WHERE id = ?", (message_id,))

    def update_message_text(self, message_id: str, text: str) -> None:
        with self._lock:
            self._db.execute("UPDATE messages SET text = ? WHERE id = ?", (text, message_id))

    def forget_message(self, message_id: str) -> None:
        with self._lock:
            self._db.execute("DELETE FROM messages WHERE id = ?", (message_id,))

    def user_message_seq(self, message_id: str) -> Optional[int]:
        message = self.message(message_id)
        return message.seq if message is not None and message.sender == "user" else None

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
            self._db.execute("INSERT INTO activity (id, step) VALUES (?, ?)",
                             (step["id"], json.dumps(step, ensure_ascii=False)))
            self._db.execute("DELETE FROM activity WHERE rowid <= (SELECT MAX(rowid) FROM activity) - ?", (keep,))

    def update_step(self, step: dict[str, Any]) -> None:
        with self._lock:
            self._db.execute("UPDATE activity SET step = ? WHERE id = ?",
                             (json.dumps(step, ensure_ascii=False), step["id"]))

    def steps(self, *, before: Optional[str] = None, limit: int = 50) -> tuple[list[dict[str, Any]], bool]:
        """The most recent steps, newest first, older than ``before`` (a step id) when given; and
        whether there are more."""
        with self._lock:
            if before:
                row = self._db.execute("SELECT rowid FROM activity WHERE id = ?", (before,)).fetchone()
                if row is None:
                    return [], False
                rows = self._db.execute("SELECT step FROM activity WHERE rowid < ? ORDER BY rowid DESC LIMIT ?",
                                        (row[0], limit + 1)).fetchall()
            else:
                rows = self._db.execute("SELECT step FROM activity ORDER BY rowid DESC LIMIT ?",
                                        (limit + 1,)).fetchall()
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
