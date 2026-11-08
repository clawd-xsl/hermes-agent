"""End-to-end MCP stdio handshake: the proxy subprocess Claude Code launches, against a live loopback.

Unit tests of the proxy helpers cannot catch an MCP SDK API change (mcp 2.x dropped the lowlevel
decorator registration, which made every Claude turn fail with "Hermes MCP loopback initialized with
status failed"). This drives the real proxy script over stdio with raw JSON-RPC, the way Claude does.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from plugins.claude_cli.loopback import ClaudeToolLoopback  # noqa: E402

_PROXY = Path(__file__).resolve().parents[3] / "plugins" / "claude_cli" / "mcp_proxy.py"


class _Agent:
    tools = [{
        "type": "function",
        "function": {
            "name": "echo",
            "description": "Echo a value",
            "parameters": {"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]},
        },
    }]

    def _execute_tool_calls(self, assistant, messages, task_id, _api_call_count=0):
        for call in assistant.tool_calls:
            value = json.loads(call.function.arguments)["value"]
            messages.append({"role": "tool", "tool_call_id": call.id, "content": f"echoed:{value}"})


class _Proxy:
    def __init__(self, env: dict[str, str]):
        self.proc = subprocess.Popen(
            [sys.executable, str(_PROXY)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, env={**os.environ, **env})
        self._next_id = 0

    def request(self, method: str, params: dict | None = None) -> dict:
        self._next_id += 1
        self._send({"jsonrpc": "2.0", "id": self._next_id, "method": method, "params": params or {}})
        while True:
            line = self.proc.stdout.readline()
            assert line, f"proxy closed stdout; stderr: {self.proc.stderr.read()}"
            message = json.loads(line)
            if message.get("id") == self._next_id:
                return message

    def notify(self, method: str) -> None:
        self._send({"jsonrpc": "2.0", "method": method})

    def _send(self, payload: dict) -> None:
        self.proc.stdin.write(json.dumps(payload) + "\n")
        self.proc.stdin.flush()

    def close(self) -> None:
        self.proc.stdin.close()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def test_proxy_serves_exact_schemas_and_routes_calls_through_the_loopback():
    agent = _Agent()
    loopback = ClaudeToolLoopback(agent, serve=True)
    history = [{"role": "user", "content": "go"}]
    loopback.begin_turn(task_id="t1", user_task="go", messages=history, projection_callback=history.extend)
    proxy = _Proxy(loopback.proxy_env())
    try:
        init = proxy.request("initialize", {
            "protocolVersion": "2025-06-18", "capabilities": {},
            "clientInfo": {"name": "claude-code", "version": "test"}})
        assert "result" in init, init
        assert init["result"]["capabilities"].get("tools") is not None
        proxy.notify("notifications/initialized")

        listed = proxy.request("tools/list")
        (tool,) = listed["result"]["tools"]
        assert tool["name"] == "echo"
        assert tool["inputSchema"] == _Agent.tools[0]["function"]["parameters"]

        # Claude emits the authoritative assistant record on stdout around the same time; here the call
        # arrives first, so the bridge correlates it by the MCP request id.
        result_box: dict = {}
        caller = threading.Thread(target=lambda: result_box.update(
            proxy.request("tools/call", {"name": "echo", "arguments": {"value": "hi"}})))
        caller.start()
        caller.join(timeout=30)
        assert not caller.is_alive()
        result = result_box["result"]
        assert result.get("isError") in (None, False)
        assert result["content"][0]["type"] == "text"
        assert "echoed:hi" in result["content"][0]["text"]
    finally:
        proxy.close()
        loopback.end_turn()
        loopback.close()
