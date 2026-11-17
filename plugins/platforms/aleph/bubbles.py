"""One reply as a few bubbles (protocol "Replies", "Bubbles"): the agent writes ``<message_break/>``
between the parts, and each part becomes a message of its own. Inside a fenced code block or an
inline code span the marker is text."""

from __future__ import annotations

import re

MARKER = "<message_break/>"
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")


def _marker_offsets(text: str) -> list[int]:
    """Where the markers that break the text start: not inside a fence or a code span."""
    offsets, fence, start = [], None, 0
    for line in text.split("\n"):
        opened = _FENCE.match(line)
        if fence is not None:
            if opened and opened.group(1)[0] == fence[0] and len(opened.group(1)) >= len(fence) \
                    and not line[opened.end():].strip():
                fence = None
        elif opened:
            fence = opened.group(1)
        else:
            offsets += [start + i for i in _line_markers(line)]
        start += len(line) + 1
    return offsets


def _line_markers(line: str) -> list[int]:
    found, i, span = [], 0, 0  # span: the backtick run that opened the code span we are in
    while i < len(line):
        if line[i] == "`":
            run = len(line[i:]) - len(line[i:].lstrip("`"))
            span = 0 if span == run else (run if span == 0 else span)
            i += run
        elif span == 0 and line.startswith(MARKER, i):
            found.append(i)
            i += len(MARKER)
        else:
            i += 1
    return found


def _trim(part: str) -> str:
    """A part without the blank lines around it (indentation inside is kept)."""
    lines = part.split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines).rstrip()


def _segments(text: str) -> list[str]:
    pieces, last = [], 0
    for offset in _marker_offsets(text):
        pieces.append(text[last:offset])
        last = offset + len(MARKER)
    pieces.append(text[last:])
    return pieces


def split(text: str) -> list[str]:
    """The parts of a finished reply, empty ones dropped."""
    return [part for part in map(_trim, _segments(text)) if part]


def split_streaming(text: str) -> tuple[list[str], str]:
    """A reply being written: the parts a marker has already closed (empty ones dropped), and the
    part still being written, without the start of a marker it may end in."""
    *closed, tail = _segments(text)
    for cut in range(min(len(MARKER) - 1, len(tail)), 0, -1):
        if MARKER.startswith(tail[-cut:]):
            tail = tail[:-cut]
            break
    return [part for part in map(_trim, closed) if part], _trim(tail)
