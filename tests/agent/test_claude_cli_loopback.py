from __future__ import annotations

import json
import sys
import types

from agent.claude_cli_loopback import ClaudeToolLoopback


class _Agent:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "echo",
                "description": "Echo a value",
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                },
            },
        }
    ]


def _request(loopback: ClaudeToolLoopback, payload: dict) -> dict:
    return loopback._handle_request(json.dumps(payload).encode())


def test_loopback_requires_token_and_preserves_exact_schema():
    loopback = ClaudeToolLoopback(_Agent(), serve=False)
    try:
        denied = _request(
            loopback,
            {"id": "bad", "token": "wrong", "method": "list_tools"},
        )
        assert denied["error"]["type"] == "PermissionError"

        listed = _request(
            loopback,
            {"id": "ok", "token": loopback.token, "method": "list_tools"},
        )
        assert listed["result"] == [
            {
                "name": "echo",
                "description": "Echo a value",
                "inputSchema": _Agent.tools[0]["function"]["parameters"],
            }
        ]
    finally:
        loopback.close()


def test_loopback_uses_standard_executor_without_early_db_projection(monkeypatch):
    calls = []

    def fake_execute(agent, assistant, messages, task_id, *, persist_progress=True, **_kwargs):
        call = assistant.tool_calls[0]
        calls.append(
            {
                "agent": agent,
                "name": call.function.name,
                "args": json.loads(call.function.arguments),
                "task_id": task_id,
                "persist_progress": persist_progress,
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call.id,
                "content": "standard-executor-result",
            }
        )

    fake_module = types.ModuleType("agent.tool_executor")
    fake_module.execute_tool_calls_sequential = fake_execute
    monkeypatch.setitem(sys.modules, "agent.tool_executor", fake_module)
    agent = _Agent()
    loopback = ClaudeToolLoopback(agent, serve=False)
    try:
        history = [{"role": "user", "content": "run it"}]
        loopback.begin_turn(task_id="task-1", user_task="run it", messages=history)
        response = _request(
            loopback,
            {
                "id": "call",
                "token": loopback.token,
                "method": "call_tool",
                "params": {"name": "echo", "arguments": {"value": "hi"}},
            },
        )
        assert response["result"] == "standard-executor-result"
        assert calls == [
            {
                "agent": agent,
                "name": "echo",
                "args": {"value": "hi"},
                "task_id": "task-1",
                "persist_progress": False,
            }
        ]
        assert history == [{"role": "user", "content": "run it"}]
    finally:
        loopback.end_turn()
        loopback.close()
