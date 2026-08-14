"""Crash-durable effect journal for the persistent Claude CLI runtime.

Claude Code's stream-json output and MCP control channel are independent. A
tool request can therefore arrive before the complete assistant record that
contains it. Conversation persistence cannot also be the admission record for
a side effect: waiting for that record creates a protocol cycle when Claude is
itself waiting for the MCP response.

This module is the deliberately small write-ahead boundary between those two
concerns. The CLI loopback records the exact tool intent before dispatch and
advances it through these states::

    prepared -> running -> completed -> reconciled

``running`` is intentionally not recoverable as ``prepared``. Hermes can die
after an external effect happened but before its result was recorded, so an
automatic retry could perform the effect twice. ``completed`` is safe to
replay because it contains the already-produced canonical Hermes tool row.

The journal is private runtime state, not conversation history. It lives next
to the Claude CLI session binding and is written mode 0600 using an atomic
replace followed by a directory fsync. The complete Claude assistant record
remains authoritative for the transcript and reconciles journal entries only
after the assistant row and cached tool results are durable there.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Optional


JournalState = Literal["prepared", "running", "completed", "reconciled"]
ClaimDisposition = Literal["execute", "replay", "unknown"]

_PROCESS_LOCKS_GUARD = threading.Lock()
_PROCESS_LOCKS: dict[str, threading.RLock] = {}


class ClaudeCliToolJournalError(RuntimeError):
    """The effect journal could not prove a safe execution transition."""


@dataclass(frozen=True)
class ToolIntent:
    tool_use_id: str
    name: str
    arguments: dict[str, Any]
    batch_id: str
    ordinal: int


@dataclass(frozen=True)
class ToolClaim:
    disposition: ClaimDisposition
    tool_use_id: str
    result_row: Optional[dict[str, Any]] = None


def _journal_root() -> Path:
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "runtime" / "claude-cli" / "effect-journal"


def _owner_digest(owner_key: str) -> str:
    return hashlib.sha256(owner_key.encode("utf-8")).hexdigest()


def _intent_signature(name: str, arguments: dict[str, Any]) -> str:
    payload = json.dumps(
        arguments,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(f"{name}\0{payload}".encode("utf-8")).hexdigest()


def _safe_copy(value: Any) -> Any:
    """Return a JSON-safe copy without mutating a live tool result."""
    cloned = copy.deepcopy(value)
    try:
        from agent.message_sanitization import _sanitize_structure_surrogates

        _sanitize_structure_surrogates(cloned)
    except Exception:
        pass
    return cloned


def _fsync_directory(path: Path) -> None:
    """Make a newly replaced journal name durable on POSIX filesystems."""
    if os.name == "nt" or not hasattr(os, "O_DIRECTORY"):
        return
    try:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _process_lock(path: Path) -> threading.RLock:
    key = str(path.resolve())
    with _PROCESS_LOCKS_GUARD:
        return _PROCESS_LOCKS.setdefault(key, threading.RLock())


@contextlib.contextmanager
def _exclusive_file_lock(path: Path):
    """Serialize journal read-modify-write across Hermes processes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        handle = path.open("a+b")
        os.chmod(path, 0o600)
    except OSError as exc:
        raise ClaudeCliToolJournalError(
            f"Claude CLI effect journal lock is unavailable: {path}: {exc}"
        ) from exc
    try:
        if os.name == "nt":  # pragma: no cover - exercised on Windows CI
            import msvcrt

            handle.seek(0)
            if handle.read(1) == b"":
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    except OSError as exc:
        handle.close()
        raise ClaudeCliToolJournalError(
            f"Claude CLI effect journal lock failed: {path}: {exc}"
        ) from exc
    try:
        yield
    finally:
        try:
            if os.name == "nt":  # pragma: no cover - exercised on Windows CI
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        handle.close()


class ClaudeCliToolJournal:
    """One serialized, crash-durable tool-effect journal per Hermes owner."""

    _SCHEMA_VERSION = 1
    _MAX_RECONCILED_RECORDS = 256

    def __init__(self, owner_key: str, *, root: Optional[Path] = None) -> None:
        self.owner_key = owner_key
        self.root = Path(root) if root is not None else _journal_root()
        self.path = self.root / f"{_owner_digest(owner_key)}.json"
        self.lock_path = self.path.with_suffix(".lock")
        self._lock = threading.RLock()
        self._records: dict[str, dict[str, Any]] = {}
        with (
            self._lock,
            _process_lock(self.lock_path),
            _exclusive_file_lock(self.lock_path),
        ):
            self._load()

    def _load(self) -> None:
        self._records = {}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except (OSError, ValueError, TypeError) as exc:
            raise ClaudeCliToolJournalError(
                f"Claude CLI effect journal is unreadable: {self.path}: {exc}"
            ) from exc
        if not isinstance(raw, dict) or raw.get("version") != self._SCHEMA_VERSION:
            raise ClaudeCliToolJournalError(
                f"Claude CLI effect journal has an unsupported format: {self.path}"
            )
        records = raw.get("records")
        if not isinstance(records, dict):
            raise ClaudeCliToolJournalError(
                f"Claude CLI effect journal has invalid records: {self.path}"
            )
        for tool_use_id, record in records.items():
            if not isinstance(tool_use_id, str) or not isinstance(record, dict):
                raise ClaudeCliToolJournalError(
                    f"Claude CLI effect journal has an invalid entry: {self.path}"
                )
            if record.get("state") not in {
                "prepared",
                "running",
                "completed",
                "reconciled",
            }:
                raise ClaudeCliToolJournalError(
                    f"Claude CLI effect journal has an invalid state: {self.path}"
                )
        self._records = records

    def _persist(self) -> None:
        from utils import atomic_json_write

        self.root.mkdir(parents=True, exist_ok=True)
        atomic_json_write(
            self.path,
            {
                "version": self._SCHEMA_VERSION,
                "owner_key_hash": _owner_digest(self.owner_key),
                "records": self._records,
            },
            indent=2,
            mode=0o600,
        )
        _fsync_directory(self.root)

    def _prune_reconciled(self) -> bool:
        """Bound audit history without ever discarding unresolved effects."""
        reconciled = sorted(
            (
                (tool_use_id, record)
                for tool_use_id, record in self._records.items()
                if record.get("state") == "reconciled"
            ),
            key=lambda item: float(item[1].get("updated_at") or 0.0),
            reverse=True,
        )
        stale = reconciled[self._MAX_RECONCILED_RECORDS :]
        for tool_use_id, _record in stale:
            self._records.pop(tool_use_id, None)
        return bool(stale)

    @staticmethod
    def _record_for(intent: ToolIntent) -> dict[str, Any]:
        now = time.time()
        return {
            "tool_use_id": intent.tool_use_id,
            "name": intent.name,
            "arguments": _safe_copy(intent.arguments),
            "signature": _intent_signature(intent.name, intent.arguments),
            "batch_id": intent.batch_id,
            "ordinal": int(intent.ordinal),
            "state": "prepared",
            "prepared_at": now,
            "updated_at": now,
        }

    @staticmethod
    def _assert_same_intent(record: dict[str, Any], intent: ToolIntent) -> None:
        expected = _intent_signature(intent.name, intent.arguments)
        if (
            str(record.get("name") or "") != intent.name
            or str(record.get("signature") or "") != expected
        ):
            raise ClaudeCliToolJournalError(
                "Claude CLI reused tool_use_id "
                f"{intent.tool_use_id!r} for a different tool intent"
            )

    def prepare_batch(self, intents: Iterable[ToolIntent]) -> None:
        """Durably record every currently known member of a model tool batch."""
        materialized = list(intents)
        if not materialized:
            return
        if any(not intent.tool_use_id or not intent.name for intent in materialized):
            raise ClaudeCliToolJournalError(
                "Claude CLI tool intents require non-empty ids and names"
            )
        with (
            self._lock,
            _process_lock(self.lock_path),
            _exclusive_file_lock(self.lock_path),
        ):
            self._load()
            changed = self._prune_reconciled()
            for intent in materialized:
                record = self._records.get(intent.tool_use_id)
                if record is None:
                    self._records[intent.tool_use_id] = self._record_for(intent)
                    changed = True
                else:
                    self._assert_same_intent(record, intent)
            if changed:
                self._persist()

    def claim_batch(self, intents: Iterable[ToolIntent]) -> list[ToolClaim]:
        """Atomically claim new intents and identify safe replay/uncertainty.

        Every ``prepared`` member advances to ``running`` in one durable write
        before the caller starts any external effect. A previously ``running``
        member is ``unknown`` and must never be invoked automatically again.
        """
        materialized = list(intents)
        self.prepare_batch(materialized)
        claims: list[ToolClaim] = []
        with (
            self._lock,
            _process_lock(self.lock_path),
            _exclusive_file_lock(self.lock_path),
        ):
            self._load()
            changed = False
            for intent in materialized:
                record = self._records[intent.tool_use_id]
                self._assert_same_intent(record, intent)
                state = record["state"]
                if state == "prepared":
                    record["state"] = "running"
                    record["running_at"] = time.time()
                    record["updated_at"] = record["running_at"]
                    changed = True
                    claims.append(ToolClaim("execute", intent.tool_use_id))
                elif state == "running":
                    claims.append(ToolClaim("unknown", intent.tool_use_id))
                elif state in {"completed", "reconciled"}:
                    result_row = record.get("result_row")
                    if not isinstance(result_row, dict):
                        raise ClaudeCliToolJournalError(
                            "Claude CLI effect journal completed without a result "
                            f"for {intent.tool_use_id}"
                        )
                    claims.append(
                        ToolClaim(
                            "replay",
                            intent.tool_use_id,
                            _safe_copy(result_row),
                        )
                    )
                else:  # guarded during load; defensive against memory corruption
                    raise ClaudeCliToolJournalError(
                        f"Unknown Claude CLI effect journal state: {state!r}"
                    )
            if changed:
                self._persist()
        return claims

    def complete(self, result_rows: Iterable[dict[str, Any]]) -> None:
        """Durably attach canonical Hermes tool rows after execution."""
        rows = [dict(row) for row in result_rows]
        if not rows:
            return
        with (
            self._lock,
            _process_lock(self.lock_path),
            _exclusive_file_lock(self.lock_path),
        ):
            self._load()
            changed = False
            for row in rows:
                tool_use_id = str(row.get("tool_call_id") or "")
                record = self._records.get(tool_use_id)
                if record is None:
                    raise ClaudeCliToolJournalError(
                        f"Tool result {tool_use_id!r} has no prepared CLI effect intent"
                    )
                state = record.get("state")
                if state == "running":
                    record["state"] = "completed"
                    record["result_row"] = _safe_copy(row)
                    record["completed_at"] = time.time()
                    record["updated_at"] = record["completed_at"]
                    changed = True
                elif state in {"completed", "reconciled"}:
                    if record.get("result_row") != row:
                        raise ClaudeCliToolJournalError(
                            "Claude CLI effect result changed for completed tool "
                            f"{tool_use_id}"
                        )
                else:
                    raise ClaudeCliToolJournalError(
                        "Cannot complete a Claude CLI tool effect before it is "
                        f"running: {tool_use_id}"
                    )
            if changed:
                self._persist()

    def complete_without_effect(self, result_rows: Iterable[dict[str, Any]]) -> None:
        """Complete host-skipped calls without marking an effect running."""
        rows = [dict(row) for row in result_rows]
        if not rows:
            return
        with (
            self._lock,
            _process_lock(self.lock_path),
            _exclusive_file_lock(self.lock_path),
        ):
            self._load()
            changed = False
            for row in rows:
                tool_use_id = str(row.get("tool_call_id") or "")
                record = self._records.get(tool_use_id)
                if record is None:
                    raise ClaudeCliToolJournalError(
                        f"Skipped result {tool_use_id!r} has no prepared CLI intent"
                    )
                state = record.get("state")
                if state == "prepared":
                    record["state"] = "completed"
                    record["result_row"] = _safe_copy(row)
                    record["completed_at"] = time.time()
                    record["updated_at"] = record["completed_at"]
                    changed = True
                elif state in {"completed", "reconciled"}:
                    if record.get("result_row") != row:
                        raise ClaudeCliToolJournalError(
                            f"Skipped CLI tool result changed for {tool_use_id}"
                        )
                else:
                    raise ClaudeCliToolJournalError(
                        f"Cannot host-skip {tool_use_id}; effect state is {state}"
                    )
            if changed:
                self._persist()

    def mark_reconciled(self, tool_use_ids: Iterable[str]) -> None:
        """Mark completed effects represented by durable transcript rows."""
        ids = {str(value) for value in tool_use_ids if str(value)}
        if not ids:
            return
        with (
            self._lock,
            _process_lock(self.lock_path),
            _exclusive_file_lock(self.lock_path),
        ):
            self._load()
            changed = False
            for tool_use_id in ids:
                record = self._records.get(tool_use_id)
                if record is None:
                    continue
                if record.get("state") == "completed":
                    record["state"] = "reconciled"
                    record["reconciled_at"] = time.time()
                    record["updated_at"] = record["reconciled_at"]
                    changed = True
            if changed:
                self._persist()

    def snapshot(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return _safe_copy(self._records)


__all__ = [
    "ClaudeCliToolJournal",
    "ClaudeCliToolJournalError",
    "ToolClaim",
    "ToolIntent",
]
