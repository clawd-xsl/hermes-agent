from __future__ import annotations

import json
import sys
import threading
import time
import types
from types import SimpleNamespace

from agent.claude_agent_sdk_bridge import ClaudeAgentSdkToolBridge
from agent.claude_agent_sdk_journal import ClaudeToolJournal


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

    def _execute_tool_calls(
        self,
        assistant,
        messages,
        task_id,
        _api_call_count=0,
        *,
        persist_progress=True,
    ):
        from agent.tool_executor import execute_tool_calls_sequential

        return execute_tool_calls_sequential(
            self,
            assistant,
            messages,
            task_id,
            persist_progress=persist_progress,
        )


def _request(bridge: ClaudeAgentSdkToolBridge, payload: dict) -> dict:
    request_id = payload.get("id")
    try:
        method = payload.get("method")
        if method == "list_tools":
            result = bridge._list_tools()
        elif method == "call_tool":
            result = bridge._call_tool(payload.get("params") or {})
        else:
            raise ValueError(f"unknown bridge method: {method}")
        return {"id": request_id, "result": result}
    except Exception as exc:
        return {
            "id": request_id,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }


def test_in_process_bridge_preserves_exact_schema():
    bridge = ClaudeAgentSdkToolBridge(_Agent())
    try:
        listed = _request(
            bridge,
            {"id": "ok", "method": "list_tools"},
        )
        assert listed["result"] == [
            {
                "name": "echo",
                "description": "Echo a value",
                "inputSchema": _Agent.tools[0]["function"]["parameters"],
            }
        ]
    finally:
        bridge.close()


def test_tool_proposal_mode_never_invokes_agent_executor():
    class ProposalAgent(_Agent):
        def _execute_tool_calls(self, *args, **kwargs):
            raise AssertionError("proposal mode must not execute tools")

    bridge = ClaudeAgentSdkToolBridge(ProposalAgent())
    try:
        bridge.begin_turn(
            task_id="proposal",
            user_task="choose a tool",
            messages=[{"role": "user", "content": "choose a tool"}],
            execute_tools=False,
        )
        response = _request(
            bridge,
            {
                "id": "call",
                "method": "call_tool",
                "params": {"name": "echo", "arguments": {"value": "hi"}},
            },
        )

        assert "captured by the host" in response["result"]
    finally:
        bridge.end_turn()
        bridge.close()


def test_background_review_tool_policy_crosses_sdk_worker_thread(
    monkeypatch,
):
    """The turn's logical policy must survive the SDK worker-thread hop."""
    from hermes_cli.plugins import (
        clear_thread_tool_whitelist,
        get_pre_tool_call_block_message,
        set_thread_tool_whitelist,
    )
    from tools.terminal_tool import (
        _get_approval_callback,
        set_approval_callback,
    )
    from tools.feishu_doc_tool import (
        get_client as get_feishu_doc_client,
        set_client as set_feishu_doc_client,
    )
    from tools.feishu_drive_tool import (
        get_client as get_feishu_drive_client,
        set_client as set_feishu_drive_client,
    )

    monkeypatch.setattr(
        "hermes_cli.plugins.invoke_hook", lambda _hook_name, **_kwargs: []
    )
    agent = _Agent()
    policy_results = []
    approval_callbacks = []
    approval_callback = lambda *_args, **_kwargs: "deny"
    feishu_client = object()
    feishu_clients = []

    def execute_with_policy(
        assistant,
        messages,
        _task_id,
        _api_call_count=0,
        *,
        persist_progress=True,
    ):
        call = assistant.tool_calls[0]
        blocked = get_pre_tool_call_block_message(
            call.function.name,
            json.loads(call.function.arguments),
        )
        policy_results.append(blocked)
        approval_callbacks.append(_get_approval_callback())
        feishu_clients.append(
            (get_feishu_doc_client(), get_feishu_drive_client())
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call.id,
                "content": blocked or "executed",
            }
        )

    agent._execute_tool_calls = execute_with_policy
    bridge = ClaudeAgentSdkToolBridge(agent)
    history = [{"role": "user", "content": "run it"}]
    try:
        set_thread_tool_whitelist(
            {"memory"},
            "Background review denied non-whitelisted tool: {tool_name}",
        )
        set_approval_callback(approval_callback)
        set_feishu_doc_client(feishu_client)
        set_feishu_drive_client(feishu_client)
        bridge.begin_turn(
            task_id="background-review",
            user_task="review",
            messages=history,
            projection_callback=history.extend,
        )
        # The originating review thread clears its policy after the turn. The
        # binding must retain the captured context for the server-side call.
        clear_thread_tool_whitelist()
        set_approval_callback(None)
        set_feishu_doc_client(None)
        set_feishu_drive_client(None)
        authoritative = bridge.reconcile_authoritative_projection(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "native-echo",
                            "type": "function",
                            "function": {
                                "name": "echo",
                                "arguments": '{"value":"unsafe"}',
                            },
                        }
                    ],
                }
            ]
        )
        history.extend(authoritative)
        bridge.mark_authoritative_projection_persisted(
            authoritative, succeeded=True
        )

        response_holder = []
        request_thread = threading.Thread(
            target=lambda: response_holder.append(
                _request(
                    bridge,
                    {
                        "id": "call",
                        "method": "call_tool",
                        "params": {
                            "name": "echo",
                            "arguments": {"value": "unsafe"},
                        },
                    },
                )
            )
        )
        request_thread.start()
        request_thread.join(timeout=2.0)
        assert request_thread.is_alive() is False
        response = response_holder[0]

        assert response["result"] == (
            "Background review denied non-whitelisted tool: echo"
        )
        assert policy_results == [response["result"]]
        assert approval_callbacks == [approval_callback]
        assert feishu_clients == [(feishu_client, feishu_client)]
    finally:
        clear_thread_tool_whitelist()
        set_approval_callback(None)
        set_feishu_doc_client(None)
        set_feishu_drive_client(None)
        bridge.end_turn()
        bridge.close()


def test_bridge_persists_intent_before_standard_executor_and_result_after(
    monkeypatch,
):
    calls = []
    projection_order = []
    next_model_boundaries = []

    def fake_execute(
        agent,
        assistant,
        messages,
        task_id,
        api_call_count=0,
        *,
        persist_progress=True,
        **_kwargs,
    ):
        call = assistant.tool_calls[0]
        calls.append(
            {
                "agent": agent,
                "name": call.function.name,
                "args": json.loads(call.function.arguments),
                "task_id": task_id,
                "api_call_count": api_call_count,
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
    agent._api_call_count = 7

    def execute_with_count(
        assistant,
        messages,
        task_id,
        api_call_count=0,
        *,
        persist_progress=True,
    ):
        return fake_execute(
            agent,
            assistant,
            messages,
            task_id,
            api_call_count,
            persist_progress=persist_progress,
        )

    agent._execute_tool_calls = execute_with_count
    bridge = ClaudeAgentSdkToolBridge(agent)
    try:
        history = [{"role": "user", "content": "run it"}]

        def project(rows):
            projection_order.append((rows[0]["role"], len(calls)))
            history.extend(rows)

        bridge.begin_turn(
            task_id="task-1",
            user_task="run it",
            messages=history,
            projection_callback=project,
            before_next_model_callback=lambda: next_model_boundaries.append(
                [row["role"] for row in history]
            ),
        )
        bridge.register_tool_request(
            name="echo",
            arguments={"value": "hi"},
            claude_id="native-control-id",
        )
        authoritative = bridge.reconcile_authoritative_projection(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "native-control-id",
                            "type": "function",
                            "function": {
                                "name": "echo",
                                "arguments": '{"value":"hi"}',
                            },
                        }
                    ],
                }
            ]
        )
        project(authoritative)
        bridge.mark_authoritative_projection_persisted(
            authoritative, succeeded=True
        )
        response = _request(
            bridge,
            {
                "id": "call",
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
                "api_call_count": 7,
                "persist_progress": False,
            }
        ]
        assert [row["role"] for row in history] == ["user", "assistant", "tool"]
        tool_call_id = history[1]["tool_calls"][0]["id"]
        assert tool_call_id == "native-control-id"
        assert history[2]["tool_call_id"] == tool_call_id
        assert projection_order == [("assistant", 0), ("tool", 1)]
        assert next_model_boundaries == [["user", "assistant", "tool"]]
        assert bridge.reconcile_authoritative_projection(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "native-control-id",
                            "type": "function",
                            "function": {
                                "name": "echo",
                                "arguments": '{"value":"hi"}',
                            },
                        }
                    ],
                }
            ]
        ) == []
        assert bridge.reconcile_authoritative_projection(
            [
                {
                    "role": "tool",
                    "tool_call_id": "native-control-id",
                    "content": "standard-executor-result",
                }
            ]
        ) == []
        assert agent._stream_needs_break is True
    finally:
        bridge.end_turn()
        bridge.close()


def test_before_next_model_can_add_wire_only_tool_context():
    agent = _Agent()
    agent._execute_tool_calls = lambda assistant, messages, *_args, **_kwargs: (
        messages.append(
            {
                "role": "tool",
                "tool_call_id": assistant.tool_calls[0].id,
                "content": "standard-executor-result",
            }
        )
    )
    history = [{"role": "user", "content": "run it"}]
    bridge = ClaudeAgentSdkToolBridge(agent)
    try:
        bridge.begin_turn(
            task_id="task-wire-context",
            user_task="run it",
            messages=history,
            projection_callback=history.extend,
            before_next_model_callback=lambda: {
                "native-id": "standard-executor-result\n\nPRIVATE GUIDANCE"
            },
        )
        authoritative = bridge.reconcile_authoritative_projection(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "native-id",
                            "type": "function",
                            "function": {
                                "name": "echo",
                                "arguments": '{"value":"hi"}',
                            },
                        }
                    ],
                }
            ]
        )
        history.extend(authoritative)
        bridge.mark_authoritative_projection_persisted(
            authoritative,
            succeeded=True,
        )

        response = _request(
            bridge,
            {
                "id": "call",
                "method": "call_tool",
                "params": {"name": "echo", "arguments": {"value": "hi"}},
            },
        )

        assert response["result"].endswith("PRIVATE GUIDANCE")
        assert history[-1]["content"] == "standard-executor-result"
        assert "PRIVATE GUIDANCE" not in str(history)
    finally:
        bridge.end_turn()
        bridge.close()


def test_native_projection_repairs_duplicate_tool_ids_without_losing_calls():
    agent = _Agent()
    bridge = ClaudeAgentSdkToolBridge(agent)
    try:
        history = [{"role": "user", "content": "run both"}]
        bridge.begin_turn(
            task_id="duplicate-ids",
            user_task="run both",
            messages=history,
        )
        # Claude permission frames can expose the same native id for distinct
        # calls. Register both arrival orders before the assistant batch.
        bridge.register_tool_request(
            name="echo",
            arguments={"value": "first"},
            claude_id="reused-id",
        )
        bridge.register_tool_request(
            name="echo",
            arguments={"value": "second"},
            claude_id="reused-id",
        )

        projected = bridge.reconcile_authoritative_projection(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "reused-id",
                            "type": "function",
                            "function": {
                                "name": "echo",
                                "arguments": '{"value":"first"}',
                            },
                        },
                        {
                            "id": "reused-id",
                            "type": "function",
                            "function": {
                                "name": "echo",
                                "arguments": '{"value":"second"}',
                            },
                        },
                    ],
                }
            ]
        )

        calls = projected[0]["tool_calls"]
        assert [call["id"] for call in calls] == ["reused-id", "reused-id_d2"]
        assert [
            json.loads(call["function"]["arguments"])["value"]
            for call in calls
        ] == ["first", "second"]
    finally:
        bridge.close()


def test_authoritative_tool_intent_wins_race_without_duplicate_projection(
    monkeypatch,
):
    def fake_execute(_agent, assistant, messages, _task_id, **_kwargs):
        messages.append(
            {
                "role": "tool",
                "tool_call_id": assistant.tool_calls[0].id,
                "content": "done",
            }
        )

    fake_module = types.ModuleType("agent.tool_executor")
    fake_module.execute_tool_calls_sequential = fake_execute
    monkeypatch.setitem(sys.modules, "agent.tool_executor", fake_module)
    agent = _Agent()
    bridge = ClaudeAgentSdkToolBridge(agent)
    history = [{"role": "user", "content": "run it"}]
    projected = []

    def project(rows):
        projected.extend(rows)
        history.extend(rows)

    try:
        bridge.begin_turn(
            task_id="task-1",
            user_task="run it",
            messages=history,
            projection_callback=project,
        )
        authoritative = bridge.reconcile_authoritative_projection(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "claude-first",
                            "type": "function",
                            "function": {
                                "name": "echo",
                                "arguments": '{"value":"hi"}',
                            },
                        }
                    ],
                }
            ]
        )
        project(authoritative)
        bridge.mark_authoritative_projection_persisted(
            authoritative, succeeded=True
        )
        response = _request(
            bridge,
            {
                "id": "call",
                "method": "call_tool",
                "params": {"name": "echo", "arguments": {"value": "hi"}},
            },
        )

        assert response["result"] == "done"
        assert [row["role"] for row in projected] == ["assistant", "tool"]
        assert projected[1]["tool_call_id"] == "claude-first"
        assert bridge.reconcile_authoritative_projection(
            [
                {
                    "role": "tool",
                    "tool_call_id": "claude-first",
                    "content": "done",
                }
            ]
        ) == []
    finally:
        bridge.end_turn()
        bridge.close()


def test_authoritative_batch_executes_once_through_standard_batch_planner():
    agent = _Agent()
    executions = []

    def execute_batch(
        assistant,
        messages,
        task_id,
        api_call_count=0,
        *,
        persist_progress=True,
    ):
        executions.append(
            {
                "ids": [call.id for call in assistant.tool_calls],
                "task_id": task_id,
                "api_call_count": api_call_count,
                "persist_progress": persist_progress,
            }
        )
        for call in assistant.tool_calls:
            value = json.loads(call.function.arguments)["value"]
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": f"done:{value}",
                }
            )

    agent._execute_tool_calls = execute_batch
    bridge = ClaudeAgentSdkToolBridge(agent)
    history = [{"role": "user", "content": "run both"}]
    projection_batches = []

    def project(rows):
        projection_batches.append([dict(row) for row in rows])
        history.extend(rows)

    try:
        bridge.begin_turn(
            task_id="task-batch",
            user_task="run both",
            messages=history,
            projection_callback=project,
        )
        authoritative = bridge.reconcile_authoritative_projection(
            [
                {
                    "role": "assistant",
                    "content": "Checking both values.",
                    "tool_calls": [
                        {
                            "id": "native-a",
                            "type": "function",
                            "function": {
                                "name": "echo",
                                "arguments": '{"value":"a"}',
                            },
                        },
                        {
                            "id": "native-b",
                            "type": "function",
                            "function": {
                                "name": "echo",
                                "arguments": '{"value":"b"}',
                            },
                        },
                    ],
                }
            ]
        )
        project(authoritative)
        bridge.mark_authoritative_projection_persisted(
            authoritative, succeeded=True
        )

        responses = {}

        def request_value(value):
            responses[value] = _request(
                bridge,
                {
                    "id": f"call-{value}",
                    "method": "call_tool",
                    "params": {
                        "name": "echo",
                        "arguments": {"value": value},
                    },
                },
            )

        threads = [
            threading.Thread(target=request_value, args=(value,))
            for value in ("a", "b")
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=2.0)

        assert all(not thread.is_alive() for thread in threads)
        assert responses == {
            "a": {"id": "call-a", "result": "done:a"},
            "b": {"id": "call-b", "result": "done:b"},
        }
        assert executions == [
            {
                "ids": ["native-a", "native-b"],
                "task_id": "task-batch",
                "api_call_count": 0,
                "persist_progress": False,
            }
        ]
        assert [row["role"] for row in history] == [
            "user",
            "assistant",
            "tool",
            "tool",
        ]
        assert [row["tool_call_id"] for row in projection_batches[1]] == [
            "native-a",
            "native-b",
        ]
        assert bridge.reconcile_authoritative_projection(
            [
                {"role": "tool", "tool_call_id": "native-a", "content": "done:a"},
                {"role": "tool", "tool_call_id": "native-b", "content": "done:b"},
            ]
        ) == []
    finally:
        bridge.end_turn()
        bridge.close()


def test_authoritative_batch_does_not_execute_duplicate_calls_twice():
    agent = _Agent()
    executed_ids = []

    def execute_batch(
        assistant,
        messages,
        _task_id,
        _api_call_count=0,
        *,
        persist_progress=True,
    ):
        assert persist_progress is False
        for call in assistant.tool_calls:
            executed_ids.append(call.id)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": "executed",
                }
            )

    def deduplicate(calls):
        return calls[:1]

    agent._execute_tool_calls = execute_batch
    agent._deduplicate_tool_calls = deduplicate
    bridge = ClaudeAgentSdkToolBridge(agent)
    history = [{"role": "user", "content": "run once"}]

    try:
        bridge.begin_turn(
            task_id="task-dedupe",
            user_task="run once",
            messages=history,
            projection_callback=history.extend,
        )
        authoritative = bridge.reconcile_authoritative_projection(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": native_id,
                            "type": "function",
                            "function": {
                                "name": "echo",
                                "arguments": '{"value":"same"}',
                            },
                        }
                        for native_id in ("native-first", "native-duplicate")
                    ],
                }
            ]
        )
        history.extend(authoritative)
        bridge.mark_authoritative_projection_persisted(
            authoritative, succeeded=True
        )

        first = bridge._call_tool(
            {"name": "echo", "arguments": {"value": "same"}}
        )
        second = bridge._call_tool(
            {"name": "echo", "arguments": {"value": "same"}}
        )

        assert first == "executed"
        assert "duplicate call" in second
        assert executed_ids == ["native-first"]
        assert [row["tool_call_id"] for row in history[-2:]] == [
            "native-first",
            "native-duplicate",
        ]
    finally:
        bridge.end_turn()
        bridge.close()


def test_mcp_first_race_uses_effect_wal_without_waiting_for_authoritative_event(
    monkeypatch,
):
    executed = threading.Event()

    def fake_execute(_agent, assistant, messages, _task_id, **_kwargs):
        executed.set()
        messages.append(
            {
                "role": "tool",
                "tool_call_id": assistant.tool_calls[0].id,
                "content": "done",
            }
        )

    fake_module = types.ModuleType("agent.tool_executor")
    fake_module.execute_tool_calls_sequential = fake_execute
    monkeypatch.setitem(sys.modules, "agent.tool_executor", fake_module)
    agent = _Agent()
    bridge = ClaudeAgentSdkToolBridge(agent)
    history = [{"role": "user", "content": "run it"}]
    response_holder = []

    def project(rows):
        history.extend(rows)

    try:
        bridge.begin_turn(
            task_id="task-1",
            user_task="run it",
            messages=history,
            projection_callback=project,
        )
        bridge.register_tool_request(
            name="echo",
            arguments={"value": "hi"},
            claude_id="native-control-id",
        )
        request_thread = threading.Thread(
            target=lambda: response_holder.append(
                _request(
                    bridge,
                    {
                        "id": "call",
                        "method": "call_tool",
                        "params": {
                            "name": "echo",
                            "arguments": {"value": "hi"},
                        },
                    },
                )
            )
        )
        request_thread.start()
        deadline = time.monotonic() + 2.0
        with bridge._projection_condition:
            while (
                not any(entry.claimed for entry in bridge._tool_projections)
                and time.monotonic() < deadline
            ):
                bridge._projection_condition.wait(timeout=0.05)
        assert any(entry.claimed for entry in bridge._tool_projections)
        request_thread.join(timeout=2.0)
        assert request_thread.is_alive() is False
        assert executed.is_set() is True
        assert response_holder[0]["result"] == "done"
        # The tool result is held outside the transcript until the complete
        # assistant record arrives, preserving strict role alternation.
        assert history == [{"role": "user", "content": "run it"}]

        reconciled = bridge.reconcile_authoritative_projection(
            [
                {
                    "role": "assistant",
                    "content": "I will inspect this now.",
                    "reasoning": "The echo tool is the correct next step.",
                    "finish_reason": "tool_calls",
                    "tool_calls": [
                        {
                            "id": "native-control-id",
                            "type": "function",
                            "function": {
                                "name": "echo",
                                "arguments": '{"value":"hi"}',
                            },
                        }
                    ],
                }
            ]
        )
        project(reconciled)
        bridge.mark_authoritative_projection_persisted(
            reconciled, succeeded=True
        )
        assert history[1]["content"] == "I will inspect this now."
        assert history[1]["reasoning"] == (
            "The echo tool is the correct next step."
        )
        assert history[1]["finish_reason"] == "tool_calls"
        assert history[1]["tool_calls"][0]["id"] == "native-control-id"
        assert history[2]["tool_call_id"] == "native-control-id"
        assert len(history) == 3
    finally:
        bridge.end_turn()
        bridge.close()


def test_mcp_call_without_native_tool_id_fails_closed():
    agent = _Agent()
    bridge = ClaudeAgentSdkToolBridge(agent)
    try:
        bridge.begin_turn(
            task_id="missing-id",
            user_task="run it",
            messages=[{"role": "user", "content": "run it"}],
        )

        response = _request(
            bridge,
            {
                "id": "call",
                "method": "call_tool",
                "params": {"name": "echo", "arguments": {"value": "hi"}},
            },
        )

        assert response["error"]["type"] == "RuntimeError"
        assert "without a durable native tool_use_id" in response["error"]["message"]
    finally:
        bridge.end_turn()
        bridge.close()


def test_partial_stream_batch_is_wal_durable_and_keeps_standard_batch_planner(
    tmp_path,
):
    class BatchAgent(_Agent):
        def __init__(self):
            self.batch_sizes = []

        def _execute_tool_calls(
            self,
            assistant,
            messages,
            _task_id,
            _api_call_count=0,
            *,
            persist_progress=True,
        ):
            assert persist_progress is False
            self.batch_sizes.append(len(assistant.tool_calls))
            for call in assistant.tool_calls:
                value = json.loads(call.function.arguments)["value"]
                messages.append(
                    {
                        "role": "tool",
                        "name": call.function.name,
                        "tool_call_id": call.id,
                        "content": f"done:{value}",
                        "effect_disposition": "performed",
                    }
                )

    agent = BatchAgent()
    journal = ClaudeToolJournal("batch-owner", root=tmp_path)
    bridge = ClaudeAgentSdkToolBridge(
        agent,
        journal=journal,
    )
    history = [{"role": "user", "content": "run both"}]
    try:
        bridge.begin_turn(
            task_id="task-1",
            user_task="run both",
            messages=history,
            projection_callback=history.extend,
        )
        bridge.observe_stream_event(
            {"type": "message_start", "message": {"id": "assistant-1"}}
        )
        for index, tool_id, value in ((0, "tool-a", "a"), (1, "tool-b", "b")):
            bridge.observe_stream_event(
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {
                        "type": "tool_use",
                        "id": tool_id,
                        "name": "mcp__hermes__echo",
                        "input": {},
                    },
                }
            )
            bridge.observe_stream_event(
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": json.dumps({"value": value}),
                    },
                }
            )
            bridge.observe_stream_event(
                {"type": "content_block_stop", "index": index}
            )
        bridge.observe_stream_event({"type": "message_stop"})
        bridge.register_tool_request(
            name="echo", arguments={"value": "a"}, claude_id="tool-a"
        )
        bridge.register_tool_request(
            name="echo", arguments={"value": "b"}, claude_id="tool-b"
        )

        assert bridge._call_tool(
            {"name": "echo", "arguments": {"value": "a"}}
        ) == "done:a"
        assert bridge._call_tool(
            {"name": "echo", "arguments": {"value": "b"}}
        ) == "done:b"
        assert agent.batch_sizes == [2]
        assert history == [{"role": "user", "content": "run both"}]
        assert {
            row["state"] for row in journal.snapshot().values()
        } == {"completed"}

        authoritative = bridge.reconcile_authoritative_projection(
            [
                {
                    "role": "assistant",
                    "content": "Running both.",
                    "tool_calls": [
                        {
                            "id": "tool-a",
                            "type": "function",
                            "function": {
                                "name": "echo",
                                "arguments": '{"value":"a"}',
                            },
                        },
                        {
                            "id": "tool-b",
                            "type": "function",
                            "function": {
                                "name": "echo",
                                "arguments": '{"value":"b"}',
                            },
                        },
                    ],
                }
            ]
        )
        history.extend(authoritative)
        bridge.mark_authoritative_projection_persisted(
            authoritative, succeeded=True
        )

        assert [row["role"] for row in history] == [
            "user",
            "assistant",
            "tool",
            "tool",
        ]
        assert [row["tool_call_id"] for row in history[2:]] == [
            "tool-a",
            "tool-b",
        ]
        assert {
            row["state"] for row in journal.snapshot().values()
        } == {"reconciled"}
    finally:
        bridge.end_turn()
        bridge.close()


def test_partial_stream_repairs_identical_duplicate_ids_before_execution(tmp_path):
    agent = _Agent()
    journal = ClaudeToolJournal("duplicate-owner", root=tmp_path)
    bridge = ClaudeAgentSdkToolBridge(agent, journal=journal)
    try:
        bridge.begin_turn(
            task_id="duplicate-partial",
            user_task="run both",
            messages=[{"role": "user", "content": "run both"}],
        )
        bridge.register_tool_request(
            name="echo", arguments={"value": "same"}, claude_id="same-id"
        )
        bridge.observe_stream_event(
            {"type": "message_start", "message": {"id": "assistant-1"}}
        )
        for index in range(2):
            bridge.observe_stream_event(
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {
                        "type": "tool_use",
                        "id": "same-id",
                        "name": "mcp__hermes__echo",
                        "input": {"value": "same"},
                    },
                }
            )
        bridge.observe_stream_event({"type": "message_stop"})

        batch = bridge._tool_batches["assistant-1"]
        assert [entry.local_id for entry in batch.projections] == [
            "same-id",
            "same-id_d2",
        ]
        assert set(journal.snapshot()) == {"same-id", "same-id_d2"}
    finally:
        bridge.end_turn()
        bridge.close()


def test_tool_side_effect_is_refused_when_intent_persistence_fails(monkeypatch):
    executed = []
    stop_boundaries = []

    def fake_execute(*_args, **_kwargs):
        executed.append(True)

    fake_module = types.ModuleType("agent.tool_executor")
    fake_module.execute_tool_calls_sequential = fake_execute
    monkeypatch.setitem(sys.modules, "agent.tool_executor", fake_module)
    agent = _Agent()
    bridge = ClaudeAgentSdkToolBridge(agent)

    def fail_projection(_rows):
        agent._incremental_persistence_failed = True

    try:
        bridge.begin_turn(
            task_id="task-1",
            user_task="run it",
            messages=[{"role": "user", "content": "run it"}],
            projection_callback=fail_projection,
            before_next_model_callback=lambda: stop_boundaries.append(True),
        )
        authoritative = bridge.reconcile_authoritative_projection(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "native-id",
                            "type": "function",
                            "function": {
                                "name": "echo",
                                "arguments": '{"value":"hi"}',
                            },
                        }
                    ],
                }
            ]
        )
        fail_projection(authoritative)
        bridge.mark_authoritative_projection_persisted(
            authoritative, succeeded=False
        )
        response = _request(
            bridge,
            {
                "id": "call",
                "method": "call_tool",
                "params": {"name": "echo", "arguments": {"value": "hi"}},
            },
        )
        assert response["error"]["type"] == "RuntimeError"
        assert "refusing to admit" in response["error"]["message"]
        assert executed == []
        assert stop_boundaries == [True]
    finally:
        bridge.end_turn()
        bridge.close()


def test_bridge_refuses_followup_side_effect_after_guardrail_halt():
    class _Decision:
        pass

    agent = _Agent()
    decision = _Decision()
    agent._tool_guardrail_halt_decision = decision
    agent._guardrail_block_result = lambda seen: (
        "halted" if seen is decision else "wrong decision"
    )
    bridge = ClaudeAgentSdkToolBridge(agent)
    try:
        bridge.begin_turn(
            task_id="task-1",
            user_task="run it",
            messages=[{"role": "user", "content": "run it"}],
        )
        response = _request(
            bridge,
            {
                "id": "blocked-call",
                "method": "call_tool",
                "params": {"name": "echo", "arguments": {"value": "must not run"}},
            },
        )
        assert response["result"] == "halted"
    finally:
        bridge.end_turn()
        bridge.close()


def test_authoritative_housekeeping_batch_mutes_post_answer_tool_noise():
    agent = _Agent()
    agent._has_content_after_think_block = lambda content: bool(content)
    agent._has_stream_consumers = lambda: True
    agent._mute_post_response = False
    bridge = ClaudeAgentSdkToolBridge(agent)
    try:
        projected = bridge.reconcile_authoritative_projection(
            [
                {
                    "role": "assistant",
                    "content": "Done — I will remember that.",
                    "tool_calls": [
                        {
                            "id": "memory-1",
                            "type": "function",
                            "function": {
                                "name": "memory",
                                "arguments": '{"action":"add"}',
                            },
                        }
                    ],
                }
            ]
        )

        assert projected[0]["content"] == "Done — I will remember that."
        assert agent._last_content_with_tools == "Done — I will remember that."
        assert agent._last_content_tools_all_housekeeping is True
        assert agent._mute_post_response is True

        bridge.reconcile_authoritative_projection(
            [{"role": "assistant", "content": "Final follow-up."}]
        )
        assert agent._mute_post_response is False
    finally:
        bridge.close()


def test_authoritative_substantive_batch_never_inherits_housekeeping_mute():
    agent = _Agent()
    agent._has_content_after_think_block = lambda content: bool(content)
    agent._has_stream_consumers = lambda: True
    agent._last_content_with_tools = "stale answer"
    agent._last_content_tools_all_housekeeping = True
    agent._mute_post_response = True
    bridge = ClaudeAgentSdkToolBridge(agent)
    try:
        bridge.reconcile_authoritative_projection(
            [
                {
                    "role": "assistant",
                    "content": "I will inspect the repository now.",
                    "tool_calls": [
                        {
                            "id": "terminal-1",
                            "type": "function",
                            "function": {
                                "name": "terminal",
                                "arguments": '{"command":"pwd"}',
                            },
                        }
                    ],
                }
            ]
        )

        assert agent._last_content_with_tools is None
        assert agent._last_content_tools_all_housekeeping is False
        assert agent._mute_post_response is False
    finally:
        bridge.close()
def test_tool_batch_wait_inherits_native_turn_timeout():
    agent = _Agent()
    agent._claude_agent_sdk_session = SimpleNamespace(turn_timeout=1_800.0)
    bridge = ClaudeAgentSdkToolBridge(agent)
    try:
        assert bridge._tool_batch_timeout() == 1_800.0
    finally:
        bridge.close()
