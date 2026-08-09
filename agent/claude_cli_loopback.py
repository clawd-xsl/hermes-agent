"""Authenticated in-process tool loopback for the persistent Claude CLI runtime.

Claude Code owns the model/tool loop when ``api_mode=claude_cli``.  It still
needs to execute the *same* tools a normal :class:`AIAgent` turn would execute,
including agent-owned tools such as ``memory`` and ``delegate_task``.  A
standalone MCP server cannot do that because those handlers need the live
agent instance.

This module keeps the authority boundary inside Hermes:

* a loopback-only TCP server lives in the Hermes process;
* every request carries an unguessable bearer token;
* the MCP stdio proxy started by Claude only forwards list/call requests;
* calls are dispatched through ``AIAgent._invoke_tool`` so middleware,
  approvals, hooks, memory, delegation, and session scoping remain intact.

The private JSONL protocol is intentionally tiny and is not a public API.
"""

from __future__ import annotations

import contextvars
import hashlib
import hmac
import json
import logging
import secrets
import socketserver
import threading
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Optional

logger = logging.getLogger(__name__)

_MAX_REQUEST_BYTES = 8 * 1024 * 1024


@dataclass
class _TurnBinding:
    task_id: str
    user_task: str
    messages: list[dict[str, Any]]
    context: contextvars.Context


def _schema_fingerprint(tools: list[dict[str, Any]]) -> str:
    payload = json.dumps(tools, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class _LoopbackTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = False
    daemon_threads = True


class ClaudeToolLoopback:
    """Expose one live AIAgent's exact tool surface to a Claude MCP proxy."""

    def __init__(self, agent: Any, *, serve: bool = True) -> None:
        self._agent = agent
        self._token = secrets.token_urlsafe(32)
        self._lock = threading.RLock()
        self._call_lock = threading.Lock()
        self._turn: Optional[_TurnBinding] = None
        owner = self

        class Handler(socketserver.StreamRequestHandler):
            def handle(self) -> None:
                while True:
                    raw = self.rfile.readline(_MAX_REQUEST_BYTES + 1)
                    if not raw:
                        return
                    if len(raw) > _MAX_REQUEST_BYTES or not raw.endswith(b"\n"):
                        return
                    response = owner._handle_request(raw)
                    self.wfile.write(
                        json.dumps(response, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                        + b"\n"
                    )
                    self.wfile.flush()

        self._server: Optional[_LoopbackTCPServer] = None
        self._thread: Optional[threading.Thread] = None
        if not serve:
            return
        self._server = _LoopbackTCPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="hermes-claude-tool-loopback",
            daemon=True,
        )
        self._thread.start()

    @property
    def address(self) -> tuple[str, int]:
        if self._server is None:
            raise RuntimeError("Claude tool loopback server is not running")
        host, port = self._server.server_address
        return str(host), int(port)

    @property
    def token(self) -> str:
        return self._token

    def bind_agent(self, agent: Any) -> None:
        """Rebind a pooled Claude process to the current AIAgent instance."""
        with self._lock:
            self._agent = agent

    def begin_turn(
        self,
        *,
        task_id: str,
        user_task: str,
        messages: list[dict[str, Any]],
    ) -> None:
        with self._lock:
            self._turn = _TurnBinding(
                task_id=task_id,
                user_task=user_task,
                messages=messages,
                context=contextvars.copy_context(),
            )

    def end_turn(self) -> None:
        with self._lock:
            self._turn = None

    def tool_definitions(self) -> list[dict[str, Any]]:
        with self._lock:
            tools = getattr(self._agent, "tools", None) or []
            return [dict(tool) for tool in tools if isinstance(tool, dict)]

    def fingerprint(self) -> str:
        return _schema_fingerprint(self.tool_definitions())

    def proxy_env(self) -> dict[str, str]:
        host, port = self.address
        return {
            "HERMES_CLAUDE_LOOPBACK_HOST": host,
            "HERMES_CLAUDE_LOOPBACK_PORT": str(port),
            "HERMES_CLAUDE_LOOPBACK_TOKEN": self._token,
        }

    def close(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._server = None
        self._thread = None

    def _handle_request(self, raw: bytes) -> dict[str, Any]:
        request_id: Any = None
        try:
            request = json.loads(raw.decode("utf-8"))
            request_id = request.get("id")
            supplied = str(request.get("token") or "")
            if not hmac.compare_digest(supplied, self._token):
                raise PermissionError("invalid loopback token")
            method = request.get("method")
            params = request.get("params") or {}
            if method == "list_tools":
                return {"id": request_id, "result": self._list_tools()}
            if method == "call_tool":
                return {"id": request_id, "result": self._call_tool(params)}
            raise ValueError(f"unknown loopback method: {method}")
        except Exception as exc:
            logger.debug("Claude tool loopback request failed", exc_info=True)
            return {
                "id": request_id,
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }

    def _list_tools(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for entry in self.tool_definitions():
            function = entry.get("function") or {}
            name = str(function.get("name") or "").strip()
            if not name:
                continue
            result.append(
                {
                    "name": name,
                    "description": str(function.get("description") or ""),
                    "inputSchema": function.get("parameters")
                    or {"type": "object", "properties": {}},
                }
            )
        return result

    def _call_tool(self, params: dict[str, Any]) -> str:
        name = str(params.get("name") or "").strip()
        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            raise TypeError("tool arguments must be an object")

        definitions = {tool["name"] for tool in self._list_tools()}
        if name not in definitions:
            raise PermissionError(f"tool is not enabled in this session: {name}")

        with self._lock:
            agent = self._agent
            turn = self._turn
        if turn is None:
            raise RuntimeError("no active Claude turn is bound to the loopback")

        tool_call_id = str(params.get("tool_call_id") or "") or secrets.token_hex(12)
        # A contextvars.Context cannot be entered by two threads concurrently.
        # Claude may issue parallel MCP calls, while AIAgent's stateful tools
        # are intentionally serialized, so one lock protects both invariants.
        #
        # Use Hermes' standard sequential executor instead of calling
        # ``_invoke_tool`` directly.  The executor is the contract boundary
        # for guardrails, approvals, checkpoints, middleware, result budgets,
        # progress callbacks, mutation verification, and agent-owned tools.
        # A scratch message list prevents the executor from projecting the
        # tool result into Hermes history before Claude emits its authoritative
        # assistant/tool protocol records.
        with self._call_lock:
            call_context = turn.context.copy()
            scratch_messages = list(turn.messages)
            tool_call = SimpleNamespace(
                id=tool_call_id,
                function=SimpleNamespace(
                    name=name,
                    arguments=json.dumps(arguments, ensure_ascii=False),
                ),
            )
            assistant = SimpleNamespace(tool_calls=[tool_call])

            def _execute() -> None:
                from agent.tool_executor import execute_tool_calls_sequential

                execute_tool_calls_sequential(
                    agent,
                    assistant,
                    scratch_messages,
                    turn.task_id,
                    persist_progress=False,
                )

            started = time.monotonic()
            call_context.run(_execute)
            logger.debug(
                "Claude MCP tool completed (name=%s duration_ms=%d)",
                name,
                int((time.monotonic() - started) * 1000),
            )

        for message in reversed(scratch_messages):
            if message.get("role") != "tool" or message.get("tool_call_id") != tool_call_id:
                continue
            content = message.get("content")
            if isinstance(content, str):
                return content
            return json.dumps(content, ensure_ascii=False)
        raise RuntimeError(f"Hermes tool executor produced no result for {name}")


__all__ = ["ClaudeToolLoopback"]
