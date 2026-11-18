"""What a project session's agent process runs of Aleph's, by this file's absolute path.

The agent runs in the project's folder, where the plugin is not importable, so this is stdlib only.

- ``pipe <socket>``: the agent's stdio MCP server for the coordination tools. It connects to the bus
  (``bus.py``), sends the session's token (from ``TOKEN_ENV``, never the command line) as the first
  line, then copies bytes both ways: the bus itself speaks newline-delimited JSON-RPC MCP, as Claude
  Code and Codex do on stdio.
- ``stop``: Claude Code's Stop hook. The CLI's Stop payload says what the session still runs in the
  background (``background_tasks``: shells, subagents, monitors; ``session_crons``: non-durable
  CronCreate, ScheduleWakeup, /loop), which its stream-json ``result`` does not. The hook writes them
  to the file ``STOP_ENV`` names; ``idle`` reads it back for the reaper. A hook must never block or
  alter the CLI's turn: it never prints and always exits 0.
"""

from __future__ import annotations

import json
import os
import select
import socket
import sys
import time
from typing import Any, Optional

TOKEN_ENV = "ALEPH_BUS_TOKEN"
STOP_ENV = "ALEPH_STOP_REPORT"
CHUNK = 65536


def pipe(path: str) -> int:
    token = os.environ.get(TOKEN_ENV, "")
    if not token:
        print(f"{TOKEN_ENV} is not set", file=sys.stderr)
        return 2
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        connection.connect(path)
    except OSError as exc:
        print(f"cannot reach the Aleph bus: {exc}", file=sys.stderr)
        return 1
    stdin, stdout = sys.stdin.fileno(), sys.stdout.fileno()
    reading = [stdin, connection]
    try:
        connection.sendall(token.encode() + b"\n")
        return _copy(connection, stdin, stdout, reading)
    except OSError:
        return 0  # either side went away: the session is ending


def _copy(connection: socket.socket, stdin: int, stdout: int, reading: list) -> int:
    while True:
        ready, _, _ = select.select(reading, [], [])
        if stdin in ready:
            data = os.read(stdin, CHUNK)
            if data:
                connection.sendall(data)
            else:
                # The agent closed its end: the bus finishes what it was answering, then hangs up.
                connection.shutdown(socket.SHUT_WR)
                reading.remove(stdin)
        if connection in ready:
            data = connection.recv(CHUNK)
            if not data:
                return 0
            view = memoryview(data)
            while view:
                view = view[os.write(stdout, view):]


def stop_report(payload: Any, now: Optional[float] = None) -> dict[str, Any]:
    data = payload if isinstance(payload, dict) else {}

    def listed(value: Any) -> Optional[list]:
        # None, not []: the CLI did not say, so the session is not known to be idle.
        return value if isinstance(value, list) else None
    return {"session_id": str(data.get("session_id") or ""), "background_tasks": listed(data.get("background_tasks")),
            "session_crons": listed(data.get("session_crons")), "at": time.time() if now is None else now}


def stop() -> int:
    try:
        path = os.environ.get(STOP_ENV, "")
        if path:
            try:
                payload = json.loads(sys.stdin.read() or "{}")
            except ValueError:
                payload = {}
            partial = f"{path}.{os.getpid()}.tmp"
            with open(partial, "w", encoding="utf-8") as handle:
                json.dump(stop_report(payload), handle)
            os.replace(partial, path)
    except Exception:
        pass
    return 0


def idle(path: str, since: float) -> bool:
    """Whether the Stop hook wrote, at or after ``since`` (the turn's end), that nothing of the
    session runs in the background."""
    try:
        with open(path, encoding="utf-8") as handle:
            report = json.load(handle)
        at = float(report.get("at") or 0)
    except (OSError, ValueError, TypeError, AttributeError):
        return False
    tasks, crons = report.get("background_tasks"), report.get("session_crons")
    return at >= since and isinstance(tasks, list) and isinstance(crons, list) and not tasks and not crons


def main(argv: list[str]) -> int:
    if argv[:1] == ["pipe"] and len(argv) == 2:
        return pipe(argv[1])
    if argv == ["stop"]:
        return stop()
    print("usage: child.py pipe <socket> | stop", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
