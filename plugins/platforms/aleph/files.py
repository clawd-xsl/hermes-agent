"""Aleph's file encryption and one-time file tickets (protocol "Files").

Each file is encrypted whole with ChaCha20-Poly1305 under its own random 32-byte key and a zero nonce
(every key encrypts exactly one file). The key travels only inside the encrypted channel, in the
file description; the HTTP side moves ciphertext against single-use tickets and needs no other
credential.

- Upload: ``file.ticket {size}`` -> ticket (5 minutes to ``PUT`` exactly ``size`` bytes of
  ciphertext). The uploaded ciphertext then waits for the ``message.send`` that names the ticket.
- Download: a ticket per link (10 minutes, one ``GET``), minted for each logged attachment and on
  ``file.ticket {id}``.

Tickets live in memory: after a gateway restart a phone asks ``file.ticket {id}`` again.
"""

from __future__ import annotations

import os
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

KEY_BYTES = 32
TAG_BYTES = 16
_ZERO_NONCE = b"\x00" * 12

UPLOAD_TICKET_SECONDS = 5 * 60
DOWNLOAD_TICKET_SECONDS = 10 * 60
# How long an uploaded file waits for the message that sends it.
UPLOADED_HOLD_SECONDS = 60 * 60


def encrypt_file(plaintext: bytes) -> tuple[bytes, bytes]:
    """(key, ciphertext) for one file."""
    key = secrets.token_bytes(KEY_BYTES)
    return key, ChaCha20Poly1305(key).encrypt(_ZERO_NONCE, plaintext, b"")


def decrypt_file(key: bytes, ciphertext: bytes) -> bytes:
    """Raises ``ValueError`` for a wrong key or damaged ciphertext."""
    if len(key) != KEY_BYTES:
        raise ValueError("a file key is 32 bytes")
    try:
        return ChaCha20Poly1305(key).decrypt(_ZERO_NONCE, ciphertext, b"")
    except InvalidTag as exc:
        raise ValueError("the file does not decrypt with its key") from exc


def file_url(ticket: str) -> str:
    """Where a ticket is used, relative to the base URL the phone already knows (public or local)."""
    return f"/aleph/v1/files/{ticket}"


@dataclass
class _Ticket:
    kind: str  # "upload" or "download"
    expires_at: float
    size: int = 0  # upload: the exact ciphertext size
    file_id: str = ""  # download
    uploaded: Optional[Path] = None  # upload: the ciphertext, once PUT


class TicketBook:
    def __init__(self, uploads_dir: Path, *, clock: Callable[[], float] = time.time) -> None:
        self.uploads_dir = Path(uploads_dir)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self._clock = clock
        self._tickets: dict[str, _Ticket] = {}

    def _prune(self) -> None:
        now = self._clock()
        for ticket, entry in list(self._tickets.items()):
            if entry.expires_at <= now:
                del self._tickets[ticket]
                if entry.uploaded is not None:
                    entry.uploaded.unlink(missing_ok=True)

    def issue_upload(self, size: int) -> str:
        self._prune()
        ticket = f"t_{secrets.token_urlsafe(24)}"
        self._tickets[ticket] = _Ticket("upload", self._clock() + UPLOAD_TICKET_SECONDS, size=size)
        return ticket

    def issue_download(self, file_id: str) -> str:
        self._prune()
        ticket = f"t_{secrets.token_urlsafe(24)}"
        self._tickets[ticket] = _Ticket("download", self._clock() + DOWNLOAD_TICKET_SECONDS, file_id=file_id)
        return ticket

    def upload_size(self, ticket: str) -> Optional[int]:
        """The size a ``PUT`` must have, while the ticket still takes one."""
        self._prune()
        entry = self._tickets.get(ticket)
        if entry is None or entry.kind != "upload" or entry.uploaded is not None:
            return None
        return entry.size

    def accept_upload(self, ticket: str, data: bytes) -> bool:
        entry = self._tickets.get(ticket)
        if entry is None or entry.kind != "upload" or entry.uploaded is not None or len(data) != entry.size:
            return False
        path = self.uploads_dir / f"{ticket}.bin"
        tmp = path.with_suffix(".part")
        tmp.write_bytes(data)
        os.replace(tmp, path)
        entry.uploaded = path
        entry.expires_at = self._clock() + UPLOADED_HOLD_SECONDS
        return True

    def uploaded(self, ticket: str) -> Optional[Path]:
        """The ciphertext a ``message.send`` attachment names (not yet spent)."""
        self._prune()
        entry = self._tickets.get(ticket)
        if entry is None or entry.kind != "upload" or entry.uploaded is None:
            return None
        return entry.uploaded

    def spend_upload(self, ticket: str) -> None:
        self._tickets.pop(ticket, None)

    def redeem_download(self, ticket: str) -> Optional[str]:
        """The file id a download ticket names; the ticket is spent."""
        self._prune()
        entry = self._tickets.get(ticket)
        if entry is None or entry.kind != "download":
            return None
        del self._tickets[ticket]
        return entry.file_id
