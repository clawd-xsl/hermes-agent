"""MCP stdio proxy for the in-process Claude tool loopback.

Claude Code launches this module from its private MCP configuration.  The
proxy contains no Hermes tool authority of its own; it forwards authenticated
requests to the parent process that owns the live AIAgent.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import socket
import sys
import uuid
from typing import Any


def _normalized_mcp_content(result: Any) -> list[dict[str, str]]:
    """Translate Hermes tool content into MCP text/image content specs."""
    if not isinstance(result, list):
        if isinstance(result, str):
            text = result
        else:
            text = json.dumps(result, ensure_ascii=False, default=str)
        return [{"type": "text", "text": text}]

    normalized: list[dict[str, str]] = []
    for part in result:
        if not isinstance(part, dict):
            normalized.append({"type": "text", "text": str(part)})
            continue
        part_type = str(part.get("type") or "")
        if part_type in {"text", "input_text"}:
            normalized.append(
                {
                    "type": "text",
                    "text": str(part.get("text") or part.get("content") or ""),
                }
            )
            continue

        image_source = part.get("source")
        if part_type == "image" and isinstance(image_source, dict):
            if image_source.get("type") == "base64":
                data = str(image_source.get("data") or "")
                mime_type = str(
                    image_source.get("media_type") or "image/png"
                )
                try:
                    base64.b64decode(data, validate=True)
                except (ValueError, TypeError):
                    data = ""
                if data:
                    normalized.append(
                        {
                            "type": "image",
                            "data": data,
                            "mimeType": mime_type,
                        }
                    )
                    continue

        image_ref = part.get("image_url")
        if isinstance(image_ref, dict):
            image_ref = image_ref.get("url")
        if not image_ref:
            image_ref = part.get("url")
        if part_type in {"image_url", "input_image", "image"} and isinstance(
            image_ref, str
        ):
            if image_ref.startswith("data:"):
                header, separator, data = image_ref.partition(",")
                if separator and ";base64" in header.lower():
                    mime_type = header[5:].split(";", 1)[0] or "image/png"
                    try:
                        base64.b64decode(data, validate=True)
                    except (ValueError, TypeError):
                        data = ""
                    if data:
                        normalized.append(
                            {
                                "type": "image",
                                "data": data,
                                "mimeType": mime_type,
                            }
                        )
                        continue
            normalized.append(
                {
                    "type": "text",
                    "text": f"[Image available at URL: {image_ref}]",
                }
            )
            continue

        normalized.append(
            {
                "type": "text",
                "text": json.dumps(part, ensure_ascii=False, default=str),
            }
        )
    return normalized or [{"type": "text", "text": "(no output)"}]


class LoopbackClient:
    def __init__(self) -> None:
        self._host = os.environ.get("HERMES_CLAUDE_LOOPBACK_HOST", "127.0.0.1")
        self._port = int(os.environ.get("HERMES_CLAUDE_LOOPBACK_PORT", "0"))
        self._token = os.environ.get("HERMES_CLAUDE_LOOPBACK_TOKEN", "")
        try:
            self._request_timeout = max(
                1.0,
                float(
                    os.environ.get(
                        "HERMES_CLAUDE_LOOPBACK_TIMEOUT_SECONDS",
                        "600",
                    )
                ),
            )
        except (TypeError, ValueError):
            self._request_timeout = 600.0
        if not self._port or not self._token:
            raise RuntimeError("Claude tool loopback configuration is missing")

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
        # Each request owns an independent socket, so there is no shared
        # mutable transport state to protect. Keeping a client-wide lock here
        # silently serialized Claude's parallel MCP calls and defeated
        # Hermes' safe tool-batch planner in the parent process.
        with socket.create_connection((self._host, self._port), timeout=30) as sock:
            sock.settimeout(self._request_timeout)
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
        content: list[Any] = []
        for part in _normalized_mcp_content(result):
            if part["type"] == "image":
                content.append(
                    types.ImageContent(
                        type="image",
                        data=part["data"],
                        mimeType=part["mimeType"],
                    )
                )
            else:
                content.append(
                    types.TextContent(type="text", text=part["text"])
                )
        return content

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
