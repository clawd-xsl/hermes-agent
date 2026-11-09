"""Emoji reactions on Aleph (protocol "Reactions"): one emoji per person per message, a tapback.

Either side reacts; the server logs each as a ``reaction`` event. The person's reactions reach the
agent with the next message they send, as a short note in front of it; a reaction alone never starts
a turn.
"""

from __future__ import annotations

import re
from typing import Any, Optional

QUOTE_CHARS = 60

# Code points that can be (the base of) an emoji. Hermes has no emoji data and the ``regex`` module
# (``\\X``) is not a dependency, so this follows UTS #51's sequence grammar over these ranges.
_PICTO = (
    "©®‼⁉™ℹ↔-↙↩↪⌚⌛⌨⏏⏩-⏳"
    "⏸-⏺Ⓜ▪▫▶◀◻-◾☀-➿⤴⤵⬅-⬇"
    "⬛⬜⭐⭕〰〽㊗㊙"
    "\U0001f000-\U0001f0ff\U0001f10d-\U0001f1ad\U0001f201-\U0001f2ff\U0001f300-\U0001f3fa"
    "\U0001f400-\U0001f64f\U0001f680-\U0001f6ff\U0001f7e0-\U0001f7f0\U0001f90c-\U0001f9ff"
    "\U0001fa70-\U0001faff"
)
_ELEMENT = f"[{_PICTO}](?:️|︎)?[\U0001f3fb-\U0001f3ff]?️?"
_EMOJI = re.compile(
    "(?:"
    "[\U0001f1e6-\U0001f1ff]{2}"                                  # a flag: two regional indicators
    "|[0-9#*]️?⃣"                                       # a keycap
    "|\U0001f3f4[\U000e0020-\U000e007e]+\U000e007f"               # a tag flag (🏴󠁧󠁢󠁳󠁣󠁴󠁿)
    f"|{_ELEMENT}(?:‍{_ELEMENT})*"                           # one emoji, or a ZWJ sequence
    ")"
)


def is_single_emoji(text: Any) -> bool:
    """One emoji grapheme: a flag, a skin tone or a ZWJ family each counts as one."""
    return isinstance(text, str) and 0 < len(text) <= 40 and _EMOJI.fullmatch(text) is not None


def quote(text: str, *, has_attachments: bool = False) -> str:
    """The start of a message, as the note quotes it."""
    flat = " ".join((text or "").split())
    if not flat:
        return "with an attachment" if has_attachments else "(empty)"
    return f'"{flat}"' if len(flat) <= QUOTE_CHARS else f'"{flat[:QUOTE_CHARS].rstrip()}…"'


def note(reacted: list[tuple[str, str, bool]]) -> str:
    """``[The person reacted ❤️ to your message "…"]`` lines for ``(emoji, text, has_attachments)``."""
    return "\n".join(f"[The person reacted {emoji} to your message {quote(text, has_attachments=has_attachments)}]"
                     for emoji, text, has_attachments in reacted)


def clean_emoji(raw: Any) -> Optional[str]:
    """A reaction's emoji, or None to take it back; ``ValueError`` for anything else."""
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise ValueError("emoji must be a string")
    emoji = raw.strip()
    if not emoji:
        return None
    if not is_single_emoji(emoji):
        raise ValueError(f"{emoji!r} is not a single emoji")
    return emoji
