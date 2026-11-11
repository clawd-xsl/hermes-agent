"""Replies on Aleph (protocol "Replies"): a message can quote an earlier one in its chat. The server
puts a snapshot of the quoted message on the ``message.new`` it logs, so every device, and one that
catches up later, draws the same quote even after the original is gone."""

from __future__ import annotations

import unicodedata
from typing import Any, Optional

EXCERPT_CHARS = 200
_REGIONAL = range(0x1F1E6, 0x1F200)


def _extends(char: str) -> bool:
    """A code point that belongs to the grapheme before it."""
    point = ord(char)
    return (unicodedata.category(char) in ("Mn", "Me", "Mc") or point == 0x200D
            or 0xFE00 <= point <= 0xFE0F or 0x1F3FB <= point <= 0x1F3FF or 0xE0020 <= point <= 0xE007F)


def excerpt(text: str, limit: int = EXCERPT_CHARS) -> str:
    """The text on one line, at most ``limit`` characters, never cut inside a grapheme (an accent, a
    ZWJ emoji, a skin tone or a flag stays whole)."""
    flat = " ".join((text or "").split())
    if len(flat) <= limit:
        return flat
    cut = limit
    while cut > 0 and (_extends(flat[cut]) or flat[cut - 1] == "‍"):
        cut -= 1
    if cut > 0 and ord(flat[cut]) in _REGIONAL:
        run = 0
        while cut - run > 0 and ord(flat[cut - run - 1]) in _REGIONAL:
            run += 1
        if run % 2:  # the cut falls between a flag's two halves
            cut -= 1
    return flat[:cut].rstrip()


def snapshot(message_id: str, sender: str, text: str, attachments: list[dict[str, Any]],
             location: Optional[dict[str, Any]]) -> dict[str, Any]:
    """``reply_to`` for a ``message.new``: who wrote the quoted message, its start, and what it carried
    first (the kind of its first attachment, or ``location``), so a media-only message can be named."""
    attachment = None
    if attachments:
        attachment = str(attachments[0].get("kind") or "file")
    elif location is not None:
        attachment = "location"
    return {"id": message_id, "sender": sender, "text": excerpt(text), "attachment": attachment}
