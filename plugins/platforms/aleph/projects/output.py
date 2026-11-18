"""Command output kept beside the log (``output.get``), and the connections following it live.

Each ``execute`` item's output is appended to ``<state>/projects/<project>/<item>.out`` as the command
writes it. Past ``REWRITE_BYTES`` the file is rewritten to its last ``KEEP_BYTES`` from a line's start,
and the item row's ``out_start``, the index of the first line still kept, moves on by the lines
dropped. Line indexes are the command's own, from its first line, whatever was dropped.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from plugins.platforms.aleph.server import INVALID_PARAMS, RpcError, event_frame

if TYPE_CHECKING:
    from plugins.platforms.aleph.projects.service import Projects
    from plugins.platforms.aleph.server import Connection

KEEP_BYTES = 8 * 1024 * 1024
REWRITE_BYTES = 16 * 1024 * 1024
MAX_LIMIT = 2000


@dataclass(eq=False)
class _Running:
    """A running command's output as kept: where its file is, and its lines so far."""
    path: Path
    start: int  # out_start
    size: int = 0  # bytes in the file
    lines: int = 0  # line feeds in the file
    tail: str = ""  # the partial last line
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)  # appending, against reading a page and following

    @property
    def total(self) -> int:
        return self.start + self.lines + (1 if self.tail else 0)


def _append(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "ab") as file:
        file.write(data)


def _trim(path: Path) -> tuple[int, int]:
    """Keep the file's last ``KEEP_BYTES`` from a line's start (or mid-line, for a line longer than
    that); ``(lines dropped, bytes kept)``."""
    data = path.read_bytes()
    cut = len(data) - KEEP_BYTES
    feed = data.find(b"\n", cut - 1)  # cut may fall right at a line's start
    cut = feed + 1 if feed != -1 else cut
    temp = path.with_suffix(".out.tmp")
    temp.write_bytes(data[cut:])
    os.replace(temp, path)
    return data.count(b"\n", 0, cut), len(data) - cut


def _read(path: Path) -> list[str]:
    """The kept lines, the last one partial while the command runs."""
    try:
        text = path.read_bytes().decode("utf-8", "replace")
    except FileNotFoundError:
        return []
    lines = text.split("\n")
    return lines[:-1] if lines[-1] == "" else lines


class Outputs:
    """Every command's kept output, and which item each connection follows (one at most)."""

    def __init__(self, projects: Projects) -> None:
        self.projects = projects
        self._running: dict[str, _Running] = {}  # by item, from its first output to its end
        self._following: dict[Connection, str] = {}

    def _path(self, project: str, item: str) -> Path:
        return self.projects.state_dir / project / f"{item}.out"

    async def append(self, project: str, item: str, chunk: str) -> int:
        """A running command's new output: kept, and sent to its followers; its ``output_lines`` now."""
        body = self._running.get(item)
        if body is None:
            body = self._running[item] = _Running(self._path(project, item), 0)
        if not chunk:
            return body.total
        async with body.lock:
            data = chunk.encode("utf-8", "surrogateescape")
            await asyncio.to_thread(_append, body.path, data)
            first = body.start + body.lines  # the line the chunk goes on with
            lines = (body.tail + chunk).split("\n")
            body.lines += len(lines) - 1
            body.tail = lines[-1]
            body.size += len(data)
            if body.size > REWRITE_BYTES:
                dropped, body.size = await asyncio.to_thread(_trim, body.path)
                body.start += dropped
                body.lines -= dropped
                tables = self.projects.tables
                async with self.projects.adapter.append_lock:
                    await asyncio.to_thread(tables.set_out_start, item, body.start)
            frame = event_frame("output.delta", {"item": item, "start": first,
                                                 "lines": lines if lines[-1] else lines[:-1]})
            for conn, followed in self._following.items():
                if followed == item:
                    conn.enqueue(frame)
            return body.total

    def finish(self, item: str) -> None:
        """The command ended: its followers stop following."""
        self._running.pop(item, None)
        for conn in [conn for conn, followed in self._following.items() if followed == item]:
            del self._following[conn]

    def forget(self, project: str) -> None:
        """The project is deleted: its commands' output goes with its folder of bodies."""
        root = self.projects.state_dir / project
        for item in [item for item, body in self._running.items() if body.path.parent == root]:
            self.finish(item)

    def drop(self, conn: Connection) -> None:
        """``output.stop``, or the connection closed."""
        self._following.pop(conn, None)

    async def get(self, params: dict[str, Any], conn: Optional[Connection]) -> dict[str, Any]:
        """``output.get``: a page of lines ending before ``before`` (null: the end), and with
        ``follow`` the item's new output to ``conn`` while it runs, in place of what it followed. A page
        without ``follow`` leaves the follow as it is."""
        item_id, before, limit = params.get("item"), params.get("before"), params.get("limit", 500)
        if before is not None and (isinstance(before, bool) or not isinstance(before, int) or before < 0):
            raise RpcError(INVALID_PARAMS, "before must be a line index or null")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_LIMIT:
            raise RpcError(INVALID_PARAMS, f"limit must be 1 to {MAX_LIMIT}")
        row = await asyncio.to_thread(self.projects.tables.item, item_id) if isinstance(item_id, str) else None
        if row is None or row.payload.get("tool") != "execute":
            raise RpcError(INVALID_PARAMS, "no command with that item id")
        body = self._running.get(row.id)
        async with body.lock if body is not None else contextlib.nullcontext():
            start = body.start if body is not None else row.out_start
            lines = await asyncio.to_thread(_read, self._path(row.project, row.id))
            total = start + len(lines)
            end = total if before is None else max(start, min(before, total))
            first = max(start, end - limit)
            running = row.open
            if params.get("follow") and conn is not None:
                # A follow asked for ends the one before, even when this command has already ended.
                if running:
                    self._following[conn] = row.id
                else:
                    self._following.pop(conn, None)
        return {"start": first, "lines": lines[first - start:end - start], "total": total, "running": running}

    async def lines(self, project: str, item: str, start: int) -> int:
        """How many lines a command that is no longer running wrote."""
        return start + len(await asyncio.to_thread(_read, self._path(project, item)))
