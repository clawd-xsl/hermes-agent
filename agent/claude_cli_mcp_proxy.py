"""MCP stdio proxy for the in-process Claude tool loopback.

Claude Code launches this module from its private MCP configuration.  The
proxy contains no Hermes tool authority of its own; it forwards authenticated
requests to the parent process that owns the live AIAgent.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import sys
import threading
import uuid
from typing import Any


class LoopbackClient:
    def __init__(self) -> None:
        self._host = os.environ.get("HERMES_CLAUDE_LOOPBACK_HOST", "127.0.0.1")
        self._port = int(os.environ.get("HERMES_CLAUDE_LOOPBACK_PORT", "0"))
        self._token = os.environ.get("HERMES_CLAUDE_LOOPBACK_TOKEN", "")
        if not self._port or not self._token:
            raise RuntimeError("Claude tool loopback configuration is missing")
        self._lock = threading.Lock()

    def request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        request_id = uuid.uuid4().hex
        payload = json.dumps(
            {
                "id": request_id,
                "token": self._token,
                "method": method,
                "params": params or {},
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8") + b"\n"
        with self._lock, socket.create_connection((self._host, self._port), timeout=30) as sock:
            sock.settimeout(600)
            sock.sendall(payload)
            reader = sock.makefile("rb")
            raw = reader.readline(8 * 1024 * 1024 + 1)
        if not raw or len(raw) > 8 * 1024 * 1024:
            raise RuntimeError("Claude tool loopback returned an invalid response")
        response = json.loads(raw.decode("utf-8"))
        if response.get("id") != request_id:
            raise RuntimeError("Claude tool loopback response id mismatch")
        if response.get("error"):
            error = response["error"]
            raise RuntimeError(str(error.get("message") or error))
        return response.get("result")


async def run() -> None:
    try:
        import mcp.server.stdio
        import mcp.types as types
        from mcp.server.lowlevel import NotificationOptions, Server
        from mcp.server.models import InitializationOptions
    except ImportError as exc:
        raise RuntimeError(
            "Claude CLI runtime requires the Hermes MCP extra: pip install 'hermes-agent[mcp]'"
        ) from exc

    client = LoopbackClient()
    server = Server("hermes-agent-loopback")

    @server.list_tools()
    async def list_tools() -> list[Any]:
        rows = await asyncio.to_thread(client.request, "list_tools")
        return [
            types.Tool(
                name=row["name"],
                description=row.get("description") or "",
                inputSchema=row.get("inputSchema")
                or {"type": "object", "properties": {}},
            )
            for row in rows
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[Any]:
        result = await asyncio.to_thread(
            client.request,
            "call_tool",
            {"name": name, "arguments": arguments or {}},
        )
        return [types.TextContent(type="text", text=str(result))]

    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="hermes-agent-loopback",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


def main() -> None:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        return
    except Exception as exc:
        print(f"Hermes Claude MCP proxy failed: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
