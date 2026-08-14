"""Durable inbox/outbox for reviewed ``session: main`` webhooks.

The generic delivery ledger records an *outbound final response* after a model
turn has completed.  Reviewed-main webhooks need a different durability
boundary: Hermes returns HTTP 202 before the reviewer runs, so the accepted
review input and any admitted handoff must survive a gateway restart.

This store keeps that small state machine in the existing profile ``state.db``:

    pending_review -> reviewing -> filtered
                                -> pending_main -> main_enqueuing -> completed

Failures return the row to the pending state with bounded backoff.  A new
adapter instance may reclaim an in-flight row owned by an older instance;
same-instance claims are protected by a lease so a lost completion callback
eventually self-heals without immediately duplicating a live model turn.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

PENDING_REVIEW = "pending_review"
REVIEWING = "reviewing"
PENDING_MAIN = "pending_main"
MAIN_ENQUEUING = "main_enqueuing"
FILTERED = "filtered"
COMPLETED = "completed"
ABANDONED = "abandoned"

TERMINAL_STATES = frozenset({FILTERED, COMPLETED, ABANDONED})
_PENDING_STATES = frozenset({PENDING_REVIEW, PENDING_MAIN})
_IN_FLIGHT_STATES = frozenset({REVIEWING, MAIN_ENQUEUING})

MAX_ATTEMPTS_PER_STAGE = 5
STALE_AFTER_SECONDS = 24 * 60 * 60
RETENTION_SECONDS = 7 * 24 * 60 * 60
LEASE_SECONDS = 15 * 60
MAX_TERMINAL_ROWS = 5_000
_RETRY_DELAYS_SECONDS = (2.0, 10.0, 30.0, 120.0, 300.0)

_DB_LOCK = threading.Lock()


@dataclass(frozen=True)
class AcceptResult:
    """Result of durably accepting a provider delivery."""

    created: bool
    receipt: Dict[str, Any]


def compute_receipt_id(profile: str, route: str, delivery_id: str) -> str:
    """Stable, profile-scoped identity for one external provider delivery."""
    material = f"{profile}\0{route}\0{delivery_id}"
    return hashlib.sha256(material.encode("utf-8", "replace")).hexdigest()[:32]


class WebhookReviewStore:
    """SQLite-backed reviewed-main receipt state machine."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        # Capture the active/default profile path when the shared webhook
        # adapter is created. Reviewer turns may later enter a named profile
        # runtime scope; their durable ingress ownership must stay in the one
        # listener's database rather than jump between profile homes.
        self.db_path = (
            Path(db_path) if db_path is not None else get_hermes_home() / "state.db"
        )
        self._schema_initialized = False

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            # Public operations serialize through _DB_LOCK, so this flag is
            # safe without a second lock. Journal-mode negotiation and DDL are
            # paid once per adapter lifecycle rather than on every webhook
            # state transition—the hot path remains one small transaction.
            if not self._schema_initialized:
                self._initialize_schema(conn)
                conn.commit()
                self._schema_initialized = True
        except Exception:
            conn.close()
            raise
        return conn

    @staticmethod
    def _initialize_schema(conn: sqlite3.Connection) -> None:
        from hermes_state import apply_wal_with_fallback

        apply_wal_with_fallback(conn, db_label="state.db (webhook_review_store)")
        conn.execute(
            """CREATE TABLE IF NOT EXISTS webhook_review_receipts (
                receipt_id TEXT PRIMARY KEY,
                profile TEXT NOT NULL,
                route TEXT NOT NULL,
                delivery_id TEXT NOT NULL,
                event_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                review_prompt TEXT NOT NULL,
                skills_json TEXT NOT NULL,
                main_source_json TEXT NOT NULL,
                handoff TEXT,
                state TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                next_attempt_at REAL NOT NULL,
                lease_until REAL,
                owner_token TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                last_error TEXT,
                UNIQUE(profile, route, delivery_id)
            )"""
        )
        conn.execute(
            """CREATE INDEX IF NOT EXISTS idx_webhook_review_due
               ON webhook_review_receipts(state, next_attempt_at)"""
        )

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    @staticmethod
    def _decode_row(row: sqlite3.Row | None) -> Optional[Dict[str, Any]]:
        if row is None:
            return None
        result = dict(row)
        try:
            result["skills"] = json.loads(result.pop("skills_json"))
        except Exception:
            result["skills"] = []
            result.pop("skills_json", None)
        try:
            result["main_source"] = json.loads(result.pop("main_source_json"))
        except Exception:
            result["main_source"] = {}
            result.pop("main_source_json", None)
        return result

    def accept(
        self,
        *,
        profile: str,
        route: str,
        delivery_id: str,
        event_id: str,
        event_type: str,
        review_prompt: str,
        skills: List[str],
        main_source: Dict[str, Any],
        now: Optional[float] = None,
    ) -> AcceptResult:
        """Persist a receipt before the HTTP handler returns 202.

        Repeating the same ``(profile, route, delivery_id)`` is idempotent and
        returns the existing state without replacing its progress.
        """
        accepted_at = time.time() if now is None else float(now)
        normalized_profile = str(profile or "default")
        receipt_id = compute_receipt_id(normalized_profile, route, delivery_id)
        skills_json = json.dumps(list(skills or []), separators=(",", ":"))
        source_json = json.dumps(main_source, separators=(",", ":"), sort_keys=True)
        with _DB_LOCK, self._transaction() as conn:
            cursor = conn.execute(
                """INSERT OR IGNORE INTO webhook_review_receipts
                   (receipt_id, profile, route, delivery_id, event_id,
                    event_type, review_prompt, skills_json, main_source_json,
                    state, attempts, next_attempt_at, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?)""",
                (
                    receipt_id,
                    normalized_profile,
                    route,
                    delivery_id,
                    event_id,
                    event_type,
                    review_prompt,
                    skills_json,
                    source_json,
                    PENDING_REVIEW,
                    accepted_at,
                    accepted_at,
                    accepted_at,
                ),
            )
            row = conn.execute(
                "SELECT * FROM webhook_review_receipts WHERE receipt_id=?",
                (receipt_id,),
            ).fetchone()
            created = cursor.rowcount == 1
        if created:
            self.prune(now=accepted_at)
        decoded = self._decode_row(row)
        if decoded is None:  # Defensive: INSERT/SELECT are one transaction.
            raise RuntimeError("durable webhook receipt disappeared after insert")
        return AcceptResult(created=created, receipt=decoded)

    def get(self, receipt_id: str) -> Optional[Dict[str, Any]]:
        with _DB_LOCK, self._transaction() as conn:
            row = conn.execute(
                "SELECT * FROM webhook_review_receipts WHERE receipt_id=?",
                (receipt_id,),
            ).fetchone()
        return self._decode_row(row)

    def claim_due(
        self,
        *,
        owner_token: str,
        limit: int = 4,
        now: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Claim due work for one adapter instance.

        In-flight work from another adapter lifecycle is immediately
        recoverable. Work owned by this instance is reclaimed only after its
        lease expires, protecting a live reviewer while self-healing a lost
        completion callback.
        """
        claimed_at = time.time() if now is None else float(now)
        stale_before = claimed_at - STALE_AFTER_SECONDS
        claimed: List[Dict[str, Any]] = []
        limit = max(1, min(int(limit), 100))
        with _DB_LOCK, self._transaction() as conn:
            candidates = conn.execute(
                """SELECT * FROM webhook_review_receipts
                   WHERE (
                       (state IN (?, ?) AND next_attempt_at <= ?)
                       OR
                       (state IN (?, ?) AND (
                           owner_token IS NULL OR owner_token != ?
                           OR lease_until IS NULL OR lease_until <= ?
                       ))
                   )
                   ORDER BY created_at ASC
                   LIMIT ?""",
                (
                    PENDING_REVIEW,
                    PENDING_MAIN,
                    claimed_at,
                    REVIEWING,
                    MAIN_ENQUEUING,
                    owner_token,
                    claimed_at,
                    limit,
                ),
            ).fetchall()
            for row in candidates:
                current = dict(row)
                receipt_id = str(current["receipt_id"])
                state = str(current["state"])
                current_owner = current.get("owner_token")
                attempts = int(current.get("attempts") or 0)
                if float(current.get("created_at") or 0) < stale_before:
                    cursor = conn.execute(
                        """UPDATE webhook_review_receipts
                           SET state=?, updated_at=?, owner_token=NULL,
                               lease_until=NULL, last_error=?
                           WHERE receipt_id=? AND state=? AND owner_token IS ?""",
                        (
                            ABANDONED,
                            claimed_at,
                            "receipt exceeded recovery window",
                            receipt_id,
                            state,
                            current_owner,
                        ),
                    )
                    if cursor.rowcount != 1:
                        continue
                    continue
                if attempts >= MAX_ATTEMPTS_PER_STAGE:
                    cursor = conn.execute(
                        """UPDATE webhook_review_receipts
                           SET state=?, updated_at=?, owner_token=NULL,
                               lease_until=NULL, last_error=?
                           WHERE receipt_id=? AND state=? AND owner_token IS ?""",
                        (
                            ABANDONED,
                            claimed_at,
                            "receipt exhausted retry budget",
                            receipt_id,
                            state,
                            current_owner,
                        ),
                    )
                    if cursor.rowcount != 1:
                        continue
                    continue

                target_state = (
                    REVIEWING
                    if state in {PENDING_REVIEW, REVIEWING}
                    else MAIN_ENQUEUING
                )
                next_attempt = attempts + 1
                cursor = conn.execute(
                    """UPDATE webhook_review_receipts
                       SET state=?, attempts=?, owner_token=?, lease_until=?,
                           updated_at=?, last_error=NULL
                       WHERE receipt_id=? AND state=? AND owner_token IS ?""",
                    (
                        target_state,
                        next_attempt,
                        owner_token,
                        claimed_at + LEASE_SECONDS,
                        claimed_at,
                        receipt_id,
                        state,
                        current_owner,
                    ),
                )
                # SQLite's process-local Python lock cannot coordinate two
                # gateway processes during a replace/restart overlap. This CAS
                # makes the claim itself authoritative across processes.
                if cursor.rowcount != 1:
                    continue
                updated = conn.execute(
                    "SELECT * FROM webhook_review_receipts WHERE receipt_id=?",
                    (receipt_id,),
                ).fetchone()
                decoded = self._decode_row(updated)
                if decoded is not None:
                    claimed.append(decoded)
        return claimed

    def next_wake_delay(
        self, *, owner_token: str, now: Optional[float] = None
    ) -> Optional[float]:
        """Seconds until any non-terminal receipt can next be claimed."""
        current_time = time.time() if now is None else float(now)
        with _DB_LOCK, self._transaction() as conn:
            rows = conn.execute(
                """SELECT state, next_attempt_at, lease_until, owner_token
                   FROM webhook_review_receipts
                   WHERE state IN (?, ?, ?, ?)""",
                (PENDING_REVIEW, PENDING_MAIN, REVIEWING, MAIN_ENQUEUING),
            ).fetchall()
        if not rows:
            return None
        due_times: List[float] = []
        for row in rows:
            state = str(row["state"])
            if state in _PENDING_STATES:
                due_times.append(float(row["next_attempt_at"] or current_time))
            elif row["owner_token"] != owner_token:
                due_times.append(current_time)
            else:
                due_times.append(float(row["lease_until"] or current_time))
        return max(0.0, min(due_times) - current_time)

    def mark_reviewed(self, receipt_id: str, *, owner_token: str, handoff: str) -> bool:
        """Persist the reviewer handoff before attempting main enqueue."""
        return self._finish_stage(
            receipt_id,
            owner_token=owner_token,
            expected_state=REVIEWING,
            state=PENDING_MAIN,
            handoff=str(handoff),
            reset_attempts=True,
        )

    def mark_filtered(self, receipt_id: str, *, owner_token: str) -> bool:
        return self._finish_stage(
            receipt_id,
            owner_token=owner_token,
            expected_state=REVIEWING,
            state=FILTERED,
        )

    def mark_completed(self, receipt_id: str, *, owner_token: str) -> bool:
        return self._finish_stage(
            receipt_id,
            owner_token=owner_token,
            expected_state=MAIN_ENQUEUING,
            state=COMPLETED,
        )

    def _finish_stage(
        self,
        receipt_id: str,
        *,
        owner_token: str,
        expected_state: str,
        state: str,
        handoff: Optional[str] = None,
        reset_attempts: bool = False,
    ) -> bool:
        now = time.time()
        assignments = [
            "state=?",
            "updated_at=?",
            "owner_token=NULL",
            "lease_until=NULL",
            "next_attempt_at=?",
            "last_error=NULL",
        ]
        values: List[Any] = [state, now, now]
        if handoff is not None:
            assignments.append("handoff=?")
            values.append(handoff)
        if reset_attempts:
            assignments.append("attempts=0")
        values.extend([receipt_id, expected_state, owner_token])
        with _DB_LOCK, self._transaction() as conn:
            cursor = conn.execute(
                f"""UPDATE webhook_review_receipts
                    SET {", ".join(assignments)}
                    WHERE receipt_id=? AND state=? AND owner_token=?""",
                values,
            )
        return cursor.rowcount == 1

    def retry(
        self,
        receipt_id: str,
        *,
        owner_token: str,
        error: str,
        now: Optional[float] = None,
    ) -> Optional[float]:
        """Return in-flight work to its pending state with bounded backoff.

        Returns the delay in seconds, or ``None`` when the retry budget has
        been exhausted and the receipt was abandoned.
        """
        failed_at = time.time() if now is None else float(now)
        with _DB_LOCK, self._transaction() as conn:
            row = conn.execute(
                """SELECT state, attempts FROM webhook_review_receipts
                   WHERE receipt_id=? AND owner_token=?""",
                (receipt_id, owner_token),
            ).fetchone()
            if row is None or str(row["state"]) not in _IN_FLIGHT_STATES:
                return None
            attempts = int(row["attempts"] or 0)
            if attempts >= MAX_ATTEMPTS_PER_STAGE:
                conn.execute(
                    """UPDATE webhook_review_receipts
                       SET state=?, updated_at=?, owner_token=NULL,
                           lease_until=NULL, last_error=?
                       WHERE receipt_id=? AND owner_token=?""",
                    (
                        ABANDONED,
                        failed_at,
                        str(error or "retry budget exhausted")[:500],
                        receipt_id,
                        owner_token,
                    ),
                )
                return None
            delay = _RETRY_DELAYS_SECONDS[
                min(max(attempts - 1, 0), len(_RETRY_DELAYS_SECONDS) - 1)
            ]
            pending_state = (
                PENDING_REVIEW if str(row["state"]) == REVIEWING else PENDING_MAIN
            )
            conn.execute(
                """UPDATE webhook_review_receipts
                   SET state=?, next_attempt_at=?, updated_at=?,
                       owner_token=NULL, lease_until=NULL, last_error=?
                   WHERE receipt_id=? AND owner_token=?""",
                (
                    pending_state,
                    failed_at + delay,
                    failed_at,
                    str(error or "reviewed-main processing failed")[:500],
                    receipt_id,
                    owner_token,
                ),
            )
        return delay

    def prune(self, *, now: Optional[float] = None) -> None:
        """Bound completed receipt retention without deleting owed work."""
        current_time = time.time() if now is None else float(now)
        cutoff = current_time - RETENTION_SECONDS
        try:
            with _DB_LOCK, self._transaction() as conn:
                conn.execute(
                    """DELETE FROM webhook_review_receipts
                       WHERE state IN (?, ?, ?) AND updated_at < ?""",
                    (FILTERED, COMPLETED, ABANDONED, cutoff),
                )
                terminal_count = conn.execute(
                    """SELECT COUNT(*) FROM webhook_review_receipts
                       WHERE state IN (?, ?, ?)""",
                    (FILTERED, COMPLETED, ABANDONED),
                ).fetchone()[0]
                excess = max(0, int(terminal_count) - MAX_TERMINAL_ROWS)
                if excess:
                    conn.execute(
                        """DELETE FROM webhook_review_receipts
                           WHERE receipt_id IN (
                               SELECT receipt_id FROM webhook_review_receipts
                               WHERE state IN (?, ?, ?)
                               ORDER BY updated_at ASC LIMIT ?
                           )""",
                        (FILTERED, COMPLETED, ABANDONED, excess),
                    )
        except Exception:
            logger.debug("webhook review receipt prune failed", exc_info=True)
