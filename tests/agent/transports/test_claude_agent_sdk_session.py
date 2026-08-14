from __future__ import annotations

import asyncio
import json
import sys
import time

import pytest

pytest.importorskip("claude_agent_sdk")

from claude_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    ServerToolUseBlock,
    StreamEvent,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from agent.iteration_budget import IterationBudget
from agent.transports.claude_agent_sdk_session import (
    ClaudeAgentSdkSession,
    _AsyncRuntimeThread,
    _sdk_message_event,
)
from agent.transports.claude_agent_sdk_common import (
    ClaudeAgentSdkTurnResult,
    _bootstrap_image_blocks,
    _compose_user_content,
    _hermes_history_signature,
    serialize_history_for_bootstrap,
)


class _Agent:
    tools = []
    prefill_messages = []
    _incremental_persistence_failed = False
    _tool_guardrail_halt_decision = None
    _interrupt_requested = False
    _api_call_count = 1

    def __init__(self):
        self.iteration_budget = IterationBudget(20)

    def _execute_tool_calls(self, *_args, **_kwargs):
        raise AssertionError("toolless test must not execute a tool")


class _FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []
        self.interrupts = 0
        self.disconnected = False

    async def query(self, prompt, session_id="default"):
        self.prompts.append((prompt, session_id))

    async def receive_response(self):
        for message in self.responses.pop(0):
            await asyncio.sleep(0)
            yield message

    async def interrupt(self):
        self.interrupts += 1

    async def disconnect(self):
        self.disconnected = True


class _ToolAgent(_Agent):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "echo",
                "description": "Echo text",
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                },
            },
        }
    ]

    def __init__(self):
        super().__init__()
        self.batch_sizes = []
        self.executed = []

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
            arguments = json.loads(call.function.arguments)
            self.executed.append((call.id, call.function.name, arguments))
            messages.append({
                "role": "tool",
                "name": call.function.name,
                "tool_call_id": call.id,
                "content": f"echo:{arguments['value']}",
                "effect_disposition": "performed",
            })


class _DelayedAuthoritativeToolClient(_FakeClient):
    def __init__(self, sdk_tool):
        super().__init__([])
        self.sdk_tool = sdk_tool
        self.tool_response = None

    async def receive_response(self):
        partials = [
            {
                "type": "message_start",
                "message": {"id": "assistant-tool"},
            },
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "tool-1",
                    "name": "mcp__hermes__echo",
                    "input": {},
                },
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": '{"value":"hello"}',
                },
            },
            {"type": "content_block_stop", "index": 0},
            {"type": "message_stop"},
        ]
        for index, event in enumerate(partials):
            yield StreamEvent(
                uuid=f"partial-{index}", session_id="native-1", event=event
            )

        # This deliberately precedes the complete AssistantMessage. It is the
        # production ordering that used to deadlock: Claude waits here for MCP
        # while Hermes used to wait for the authoritative event below.
        self.tool_response = await self.sdk_tool.handler({"value": "hello"})
        yield AssistantMessage(
            content=[
                TextBlock(text="Checking."),
                ToolUseBlock(
                    id="tool-1",
                    name="mcp__hermes__echo",
                    input={"value": "hello"},
                ),
            ],
            model="claude-sonnet-4-6",
            message_id="assistant-tool",
            stop_reason="tool_use",
            session_id="native-1",
        )
        yield UserMessage(
            content=[ToolResultBlock(tool_use_id="tool-1", content="echo:hello")]
        )
        yield StreamEvent(
            uuid="final-start",
            session_id="native-1",
            event={"type": "message_start", "message": {"id": "assistant-final"}},
        )
        yield AssistantMessage(
            content=[TextBlock(text="Done.")],
            model="claude-sonnet-4-6",
            message_id="assistant-final",
            stop_reason="end_turn",
            session_id="native-1",
        )
        yield ResultMessage(
            subtype="success",
            duration_ms=20,
            duration_api_ms=10,
            is_error=False,
            num_turns=2,
            session_id="native-1",
            result="Done.",
            terminal_reason="completed",
        )


def _response(session_id: str, text: str):
    return [
        StreamEvent(
            uuid="stream-start",
            session_id=session_id,
            event={
                "type": "message_start",
                "message": {"id": f"message-{text}"},
            },
        ),
        StreamEvent(
            uuid="stream-text",
            session_id=session_id,
            event={
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": text},
            },
        ),
        AssistantMessage(
            content=[TextBlock(text=text)],
            model="claude-sonnet-4-6",
            usage={"input_tokens": 12, "output_tokens": 3},
            message_id=f"message-{text}",
            stop_reason="end_turn",
            session_id=session_id,
        ),
        ResultMessage(
            subtype="success",
            duration_ms=10,
            duration_api_ms=8,
            is_error=False,
            num_turns=1,
            session_id=session_id,
            result=text,
            usage={"input_tokens": 12, "output_tokens": 3},
            terminal_reason="completed",
        ),
    ]


def _session(tmp_path, monkeypatch, *, tools=None):
    monkeypatch.setattr(
        "agent.claude_agent_sdk_journal._journal_root",
        lambda: tmp_path / "effect-journal",
    )
    agent = _Agent()
    if tools is not None:
        agent.tools = tools
    session = ClaudeAgentSdkSession(
        owner_key="sdk-test-owner",
        agent=agent,
        cwd=str(tmp_path),
        model="anthropic/claude-sonnet-4-6",
        system_prompt="stable system prompt",
        persistent_binding=False,
    )
    return agent, session


def test_typed_sdk_assistant_message_projects_complete_native_fields():
    message = AssistantMessage(
        content=[TextBlock(text="hello")],
        model="claude-sonnet-4-6",
        usage={"input_tokens": 7},
        message_id="msg-1",
        stop_reason="end_turn",
        session_id="session-1",
    )

    event = _sdk_message_event(message)

    assert event["type"] == "assistant"
    assert event["session_id"] == "session-1"
    assert event["message"] == {
        "id": "msg-1",
        "model": "claude-sonnet-4-6",
        "content": [{"type": "text", "text": "hello"}],
        "usage": {"input_tokens": 7},
        "stop_reason": "end_turn",
    }


def test_bootstrap_preserves_prior_api_content_images_and_excludes_current_user():
    current = [
        {"type": "text", "text": "compare these"},
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,Q1VSUkVOVA=="},
        },
    ]
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "clean old", "api_content": "recalled old"},
        {
            "role": "user",
            "api_content": [
                {"type": "text", "text": "old image"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,T0xE"},
                },
            ],
        },
        {"role": "assistant", "content": "seen"},
        {"role": "user", "content": current},
    ]

    bootstrap = serialize_history_for_bootstrap(
        messages,
        prefill_messages=[{"role": "user", "content": "stable prefill"}],
    )
    content = _compose_user_content(
        current,
        bootstrap=bootstrap,
        bootstrap_attachments=_bootstrap_image_blocks(messages),
    )

    assert "stable prefill" in bootstrap
    assert "recalled old" in bootstrap
    assert "clean old" not in bootstrap
    assert "compare these" not in bootstrap
    assert '"role":"system"' not in bootstrap
    images = [block for block in content if block.get("type") == "image"]
    assert images == [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": "T0xE",
            },
        },
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": "Q1VSUkVOVA==",
            },
        },
    ]


def test_sdk_session_streams_and_projects_on_originating_turn_thread(
    tmp_path, monkeypatch
):
    agent, session = _session(tmp_path, monkeypatch)
    client = _FakeClient([_response("native-1", "hello")])
    runtime = _AsyncRuntimeThread("sdk-test")
    session._runtime = runtime
    session._client = client
    session._process_started_at = time.monotonic()
    monkeypatch.setattr(session, "ensure_started", lambda: None)
    messages = [{"role": "user", "content": "hi"}]
    deltas = []
    callback_threads = []

    def project(rows):
        callback_threads.append(__import__("threading").get_ident())
        messages.extend(rows)

    origin_thread = __import__("threading").get_ident()
    try:
        result = session.run_turn(
            agent=agent,
            user_input="hi",
            messages=messages,
            task_id="task",
            stream_callback=deltas.append,
            projection_callback=project,
        )
    finally:
        session.close()

    assert result.error is None
    assert result.final_text == "hello"
    assert result.model_iterations == 1
    assert result.native_session_id == "native-1"
    assert result.token_usage == {"input_tokens": 12, "output_tokens": 3}
    assert deltas == ["hello"]
    assert messages[-1] == {
        "role": "assistant",
        "content": "hello",
        "finish_reason": "end_turn",
    }
    assert callback_threads == [origin_thread]


def test_sdk_interrupt_arriving_during_startup_prevents_query(tmp_path, monkeypatch):
    agent, session = _session(tmp_path, monkeypatch)
    agent.iteration_budget = IterationBudget(1)

    def start_then_interrupt():
        session._interrupt_requested.set()

    monkeypatch.setattr(session, "ensure_started", start_then_interrupt)
    try:
        result = session.run_turn(
            agent=agent,
            user_input="stop while starting",
            messages=[{"role": "user", "content": "stop while starting"}],
            task_id="startup-interrupt",
        )
    finally:
        session.close()

    assert result.interrupted is True
    assert result.should_retire is True
    assert result.error is None
    assert result.budget_iterations == 0
    assert agent.iteration_budget.used == 0


def test_sdk_client_is_reused_for_multiple_turns(tmp_path, monkeypatch):
    agent, session = _session(tmp_path, monkeypatch)
    client = _FakeClient([
        _response("native-1", "first"),
        _response("native-1", "second"),
    ])
    runtime = _AsyncRuntimeThread("sdk-reuse")
    session._runtime = runtime
    session._client = client
    session._process_started_at = time.monotonic()
    starts = []
    monkeypatch.setattr(session, "ensure_started", lambda: starts.append(True))
    messages = []
    try:
        for prompt in ("one", "two"):
            messages.append({"role": "user", "content": prompt})
            result = session.run_turn(
                agent=agent,
                user_input=prompt,
                messages=messages,
                task_id="task",
                projection_callback=messages.extend,
            )
            assert result.error is None
    finally:
        session.close()

    assert len(client.prompts) == 2
    assert client.prompts[0][1] == client.prompts[1][1] == "default"
    assert starts == [True, True]
    assert session._turns_completed == 2


def test_sdk_resume_is_invalidated_when_hermes_history_diverges(tmp_path, monkeypatch):
    agent, session = _session(tmp_path, monkeypatch)
    session.persistent_binding = True
    session.native_session_id = "native-old"
    session._resume = True
    session._history_signature = _hermes_history_signature([
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer that was undone"},
    ])
    reset = []
    seen_resume = []

    def reset_fresh():
        reset.append(True)
        session.native_session_id = "native-new"
        session._resume = False
        session._history_signature = None

    def run_once(**_kwargs):
        seen_resume.append(session._resume)
        return ClaudeAgentSdkTurnResult(
            final_text="replacement answer",
            projected_messages=[{"role": "assistant", "content": "replacement answer"}],
            native_session_id=session.native_session_id,
        )

    monkeypatch.setattr(session, "_reset_fresh_binding", reset_fresh)
    monkeypatch.setattr(session, "_run_turn_once", run_once)
    monkeypatch.setattr(session, "sync_history_signature", lambda _messages: None)
    try:
        result = session.run_turn(
            agent=agent,
            user_input="replacement",
            messages=[{"role": "user", "content": "replacement"}],
            task_id="rewrite",
        )
    finally:
        session.close()

    assert result.final_text == "replacement answer"
    assert reset == [True]
    assert seen_resume == [False]


def test_sdk_missing_resume_retries_same_turn_with_fresh_binding(tmp_path, monkeypatch):
    agent, session = _session(tmp_path, monkeypatch)
    session.persistent_binding = True
    session.native_session_id = "native-missing"
    session._resume = True
    session._history_signature = _hermes_history_signature([])
    results = iter([
        ClaudeAgentSdkTurnResult(error="Session not found", should_retire=True),
        ClaudeAgentSdkTurnResult(final_text="recovered", native_session_id="native-new"),
    ])
    reset = []

    def reset_fresh():
        reset.append(True)
        session._resume = False

    monkeypatch.setattr(session, "_reset_fresh_binding", reset_fresh)
    monkeypatch.setattr(session, "_run_turn_once", lambda **_kwargs: next(results))
    monkeypatch.setattr(session, "sync_history_signature", lambda _messages: None)
    try:
        result = session.run_turn(
            agent=agent,
            user_input="hello",
            messages=[{"role": "user", "content": "hello"}],
            task_id="resume-recovery",
        )
    finally:
        session.close()

    assert result.final_text == "recovered"
    assert result.session_reuse == "resume_recovery"
    assert reset == [True]


def test_sdk_options_keep_hermes_as_only_agent_and_preserve_raw_cli_args(
    tmp_path, monkeypatch
):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "echo",
                "description": "Echo text",
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                },
            },
        }
    ]
    agent, session = _session(tmp_path, monkeypatch, tools=tools)
    session.command = sys.executable
    session.extra_args = ["--operator-custom-flag"]
    captured = {}

    async def fake_connect(client):
        captured["client"] = client

    async def fake_mcp_status(_client):
        return {"mcpServers": [{"name": "hermes", "status": "connected"}]}

    monkeypatch.setattr("claude_agent_sdk.ClaudeSDKClient.connect", fake_connect)
    monkeypatch.setattr(
        "claude_agent_sdk.ClaudeSDKClient.get_mcp_status", fake_mcp_status
    )
    asyncio.run(session._sdk_connect())
    try:
        client = captured["client"]
        options = client.options
        assert options.tools == ["ToolSearch"]
        assert options.allowed_tools == ["mcp__hermes__*"]
        assert options.setting_sources == []
        assert options.skills == []
        assert options.strict_mcp_config is True
        assert options.mcp_servers["hermes"]["type"] == "sdk"
        assert options.mcp_servers["hermes"]["alwaysLoad"] is True
        assert list(options.hooks) == ["PreToolUse"]
        command = client._custom_transport._build_command()
        assert command[0] == sys.executable
        assert command[1] == "--operator-custom-flag"
        assert "--disable-slash-commands" in command
        assert "--no-session-persistence" in command
    finally:
        asyncio.run(session._sdk_disconnect())
        session.close()


def test_sdk_bundled_cli_skips_only_redundant_cold_version_probe(tmp_path, monkeypatch):
    _agent, session = _session(tmp_path, monkeypatch)
    captured = {}
    probed = []

    async def fake_connect(client):
        captured["client"] = client

    async def base_version_probe(_transport):
        probed.append(True)

    monkeypatch.setattr("claude_agent_sdk.ClaudeSDKClient.connect", fake_connect)
    monkeypatch.setattr(
        "claude_agent_sdk._internal.transport.subprocess_cli."
        "SubprocessCLITransport._check_claude_version",
        base_version_probe,
    )
    asyncio.run(session._sdk_connect())
    try:
        client = captured["client"]
        assert client.options.cli_path is None
        asyncio.run(client._custom_transport._check_claude_version())
        assert probed == []
    finally:
        asyncio.run(session._sdk_disconnect())
        session.close()


def test_sdk_options_preserve_native_limits_and_json_settings(tmp_path, monkeypatch):
    agent, session = _session(tmp_path, monkeypatch)
    session.fast_mode = True
    session.reasoning_effort = "high"
    session.thinking_mode = "disabled"
    session.max_output_tokens = 4096
    session.api_retry_count = 2
    session.provider_request_timeout = 45.5
    session.auto_compaction_enabled = False

    env = session._build_env()

    assert env["ANTHROPIC_API_KEY"] == ""
    assert env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "4096"
    assert env["CLAUDE_CODE_MAX_RETRIES"] == "2"
    assert env["API_TIMEOUT_MS"] == "45500"
    assert env["DISABLE_COMPACT"] == "1"
    assert env["CLAUDE_CODE_DISABLE_FAST_MODE"] == ""

    session.command = sys.executable
    captured = {}

    async def fake_connect(client):
        captured["client"] = client

    monkeypatch.setattr("claude_agent_sdk.ClaudeSDKClient.connect", fake_connect)
    asyncio.run(session._sdk_connect())
    try:
        client = captured["client"]
        assert client.options.settings == '{"fastMode":true}'
        assert client.options.effort == "high"
        assert client.options.thinking == {"type": "disabled"}
        command = client._custom_transport._build_command()
        settings_index = command.index("--settings")
        assert json.loads(command[settings_index + 1]) == {"fastMode": True}
    finally:
        asyncio.run(session._sdk_disconnect())
        session.close()


def test_sdk_transport_error_includes_bounded_stderr_diagnostics(tmp_path, monkeypatch):
    agent, session = _session(tmp_path, monkeypatch)
    client = _FakeClient([[RuntimeError("wire closed")]])
    runtime = _AsyncRuntimeThread("sdk-stderr")
    session._runtime = runtime
    session._client = client
    monkeypatch.setattr(session, "ensure_started", lambda: None)
    session._stderr_line("oauth credential rejected")
    try:
        result = session.run_turn(
            agent=agent,
            user_input="hello",
            messages=[{"role": "user", "content": "hello"}],
            task_id="stderr",
        )
    finally:
        session.close()

    assert result.error == "wire closed\noauth credential rejected"
    assert result.should_retire is True


def test_explicit_relative_cli_override_resolves_from_session_cwd(
    tmp_path, monkeypatch
):
    _agent, session = _session(tmp_path, monkeypatch)
    executable = tmp_path / "bin" / "claude-custom"
    executable.parent.mkdir()
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    session.command = "bin/claude-custom"
    try:
        assert session._resolved_cli_path() == str(executable.resolve())
    finally:
        session.close()


def test_sdk_connect_fails_closed_when_hermes_mcp_is_not_connected(
    tmp_path, monkeypatch
):
    agent = _ToolAgent()
    monkeypatch.setattr(
        "agent.claude_agent_sdk_journal._journal_root",
        lambda: tmp_path / "effect-journal",
    )
    session = ClaudeAgentSdkSession(
        owner_key="mcp-failure",
        agent=agent,
        cwd=str(tmp_path),
        model="anthropic/claude-sonnet-4-6",
        system_prompt="stable system prompt",
        command=sys.executable,
        persistent_binding=False,
    )
    disconnected = []

    async def fake_connect(_client):
        return None

    async def fake_status(_client):
        return {
            "mcpServers": [
                {"name": "hermes", "status": "failed", "error": "bad bridge"}
            ]
        }

    async def fake_disconnect(_client):
        disconnected.append(True)

    monkeypatch.setattr("claude_agent_sdk.ClaudeSDKClient.connect", fake_connect)
    monkeypatch.setattr("claude_agent_sdk.ClaudeSDKClient.get_mcp_status", fake_status)
    monkeypatch.setattr("claude_agent_sdk.ClaudeSDKClient.disconnect", fake_disconnect)

    with pytest.raises(RuntimeError, match="failed to initialize: bad bridge"):
        asyncio.run(session._sdk_connect())

    assert disconnected == [True]
    assert session.is_alive is False
    session.close()


def test_sdk_connect_accepts_hermes_mcp_deferred_until_first_prompt(
    tmp_path, monkeypatch
):
    agent = _ToolAgent()
    monkeypatch.setattr(
        "agent.claude_agent_sdk_journal._journal_root",
        lambda: tmp_path / "effect-journal",
    )
    session = ClaudeAgentSdkSession(
        owner_key="mcp-deferred",
        agent=agent,
        cwd=str(tmp_path),
        model="anthropic/claude-sonnet-4-6",
        system_prompt="stable system prompt",
        command=sys.executable,
        persistent_binding=False,
    )

    async def fake_connect(_client):
        return None

    async def fake_status(_client):
        return {"mcpServers": []}

    monkeypatch.setattr("claude_agent_sdk.ClaudeSDKClient.connect", fake_connect)
    monkeypatch.setattr("claude_agent_sdk.ClaudeSDKClient.get_mcp_status", fake_status)

    asyncio.run(session._sdk_connect())
    try:
        assert session.is_alive is True
    finally:
        asyncio.run(session._sdk_disconnect())
        session.close()


def test_sdk_partial_batch_executes_before_delayed_authoritative_message(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "agent.claude_agent_sdk_journal._journal_root",
        lambda: tmp_path / "effect-journal",
    )
    agent = _ToolAgent()
    session = ClaudeAgentSdkSession(
        owner_key="deadlock-regression",
        agent=agent,
        cwd=str(tmp_path),
        model="anthropic/claude-sonnet-4-6",
        system_prompt="stable system prompt",
        persistent_binding=False,
    )
    client = _DelayedAuthoritativeToolClient(session._sdk_tools()[0])
    runtime = _AsyncRuntimeThread("sdk-deadlock")
    session._runtime = runtime
    session._client = client
    session._process_started_at = time.monotonic()
    session.turn_timeout = 5.0
    monkeypatch.setattr(session, "ensure_started", lambda: None)
    messages = [{"role": "user", "content": "use echo"}]
    try:
        result = session.run_turn(
            agent=agent,
            user_input="use echo",
            messages=messages,
            task_id="deadlock-regression",
            projection_callback=messages.extend,
        )
    finally:
        session.close()

    assert result.error is None
    assert result.final_text == "Done."
    assert agent.batch_sizes == [1]
    assert agent.executed == [("tool-1", "echo", {"value": "hello"})]
    assert client.tool_response == {"content": [{"type": "text", "text": "echo:hello"}]}
    assert [row["role"] for row in messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert messages[1]["tool_calls"][0]["id"] == "tool-1"
    assert messages[2]["tool_call_id"] == "tool-1"
    journal_rows = session.bridge._journal.snapshot()
    assert journal_rows["tool-1"]["state"] == "reconciled"


def test_sdk_retry_error_preserves_provider_category_and_status(tmp_path, monkeypatch):
    agent, session = _session(tmp_path, monkeypatch)
    client = _FakeClient([
        [
            SystemMessage(
                subtype="api_retry",
                data={
                    "attempt": 2,
                    "max_retries": 4,
                    "retry_delay_ms": 1500,
                    "error_status": 429,
                    "error": "rate_limit",
                },
            ),
            ResultMessage(
                subtype="error_during_execution",
                duration_ms=10,
                duration_api_ms=8,
                is_error=True,
                num_turns=1,
                session_id="native-1",
                errors=["Usage limit reached"],
            ),
        ]
    ])
    runtime = _AsyncRuntimeThread("sdk-retry")
    session._runtime = runtime
    session._client = client
    monkeypatch.setattr(session, "ensure_started", lambda: None)
    retries = []
    try:
        result = session.run_turn(
            agent=agent,
            user_input="retry",
            messages=[{"role": "user", "content": "retry"}],
            task_id="retry",
            api_retry_callback=retries.append,
        )
    finally:
        session.close()

    assert retries == [
        {
            "attempt": 2,
            "max_retries": 4,
            "retry_delay_ms": 1500,
            "error_status": 429,
            "error": "rate_limit",
        }
    ]
    assert result.error == "Usage limit reached"
    assert result.error_category == "rate_limit"
    assert result.error_status == 429
    assert result.should_retire is True


def test_sdk_empty_refusal_and_thinking_limit_keep_host_semantics(
    tmp_path, monkeypatch
):
    agent, session = _session(tmp_path, monkeypatch)
    client = _FakeClient([
        [
            AssistantMessage(
                content=[],
                model="claude-sonnet-4-6",
                message_id="refusal",
                stop_reason="refusal",
                session_id="native-1",
                usage={"input_tokens": 9, "output_tokens": 0},
            ),
            ResultMessage(
                subtype="success",
                duration_ms=10,
                duration_api_ms=8,
                is_error=False,
                num_turns=1,
                session_id="native-1",
                result="",
                stop_reason="refusal",
            ),
        ],
        [
            AssistantMessage(
                content=[
                    ThinkingBlock(thinking="still reasoning", signature="signature")
                ],
                model="claude-sonnet-4-6",
                message_id="thinking-limit",
                stop_reason="max_tokens",
                session_id="native-1",
            ),
            ResultMessage(
                subtype="success",
                duration_ms=10,
                duration_api_ms=8,
                is_error=False,
                num_turns=1,
                session_id="native-1",
                result="",
                stop_reason="max_tokens",
            ),
        ],
    ])
    runtime = _AsyncRuntimeThread("sdk-stop-semantics")
    session._runtime = runtime
    session._client = client
    monkeypatch.setattr(session, "ensure_started", lambda: None)
    observed = []
    try:
        refusal = session.run_turn(
            agent=agent,
            user_input="refuse",
            messages=[{"role": "user", "content": "refuse"}],
            task_id="refusal",
            iteration_post_callback=lambda *args: observed.append(args),
        )
        messages = [
            {"role": "user", "content": "refuse"},
            {"role": "user", "content": "hard"},
        ]
        # Keep this unit focused on terminal mapping rather than native-history
        # invalidation; pooled history behavior is covered by runtime tests.
        session._history_signature = None
        session._resume = False
        thinking = session.run_turn(
            agent=agent,
            user_input="hard",
            messages=messages,
            task_id="thinking-limit",
        )
    finally:
        session.close()

    assert refusal.last_stop_reason == "refusal"
    assert refusal.model_iterations == 1
    assert observed[0][1] == {
        "role": "assistant",
        "content": None,
        "finish_reason": "refusal",
    }
    assert thinking.last_stop_reason == "max_tokens"
    assert thinking.thinking_budget_exhausted is True
    assert thinking.model_iterations == 1


def test_sdk_structured_summary_is_toolless_and_does_not_consume_turn_budget(
    tmp_path, monkeypatch
):
    agent, session = _session(tmp_path, monkeypatch)
    agent.iteration_budget = IterationBudget(0)
    client = _FakeClient([
        [
            StreamEvent(
                uuid="summary-start",
                session_id="native-summary",
                event={"type": "message_start", "message": {"id": "summary"}},
            ),
            AssistantMessage(
                content=[TextBlock(text="display text")],
                model="claude-sonnet-4-6",
                message_id="summary",
                session_id="native-summary",
            ),
            ResultMessage(
                subtype="success",
                duration_ms=10,
                duration_api_ms=8,
                is_error=False,
                num_turns=1,
                session_id="native-summary",
                result="display text",
                structured_output={"answer": "yes", "count": 2},
            ),
        ]
    ])
    runtime = _AsyncRuntimeThread("sdk-summary")
    session._runtime = runtime
    session._client = client
    monkeypatch.setattr(session, "ensure_started", lambda: None)
    try:
        result = session.summarize(
            agent=agent,
            messages=[
                {"role": "user", "content": "do work"},
                {"role": "assistant", "content": "working"},
            ],
            prompt="summarize without tools",
        )
    finally:
        session.close()

    assert result.structured_output == {"answer": "yes", "count": 2}
    assert result.final_text == '{"answer":"yes","count":2}'
    assert result.budget_exhausted is False
    assert agent.iteration_budget.used == 0
    prompt, _session_id = client.prompts[0]
    assert "do work" in prompt
    assert prompt.endswith("summarize without tools")


def test_sdk_tool_proposal_captures_call_without_executing(tmp_path, monkeypatch):
    agent = _ToolAgent()
    monkeypatch.setattr(
        "agent.claude_agent_sdk_journal._journal_root",
        lambda: tmp_path / "effect-journal",
    )
    session = ClaudeAgentSdkSession(
        owner_key="tool-proposal",
        agent=agent,
        cwd=str(tmp_path),
        model="anthropic/claude-sonnet-4-6",
        system_prompt="stable system prompt",
        persistent_binding=False,
    )
    client = _FakeClient([
        [
            AssistantMessage(
                content=[
                    ToolUseBlock(
                        id="proposal-1",
                        name="mcp__hermes__echo",
                        input={"value": "candidate"},
                    )
                ],
                model="claude-sonnet-4-6",
                message_id="proposal",
                session_id="native-proposal",
            )
        ]
    ])
    runtime = _AsyncRuntimeThread("sdk-proposal")
    session._runtime = runtime
    session._client = client
    monkeypatch.setattr(session, "ensure_started", lambda: None)
    try:
        result = session.propose_tools(
            agent=agent,
            messages=[{"role": "assistant", "content": "prior context"}],
            prompt="choose the next tool",
        )
    finally:
        session.close()

    assert result.captured_tool_calls is True
    assert result.should_retire is True
    assert result.projected_messages[0]["tool_calls"][0]["function"] == {
        "name": "echo",
        "arguments": '{"value": "candidate"}',
    }
    assert agent.executed == []


def test_sdk_iteration_budget_stops_unreserved_native_step(tmp_path, monkeypatch):
    agent, session = _session(tmp_path, monkeypatch)
    agent.iteration_budget = IterationBudget(1)
    client = _FakeClient([
        [
            StreamEvent(
                uuid="first-start",
                session_id="native-1",
                event={"type": "message_start", "message": {"id": "first"}},
            ),
            AssistantMessage(
                content=[TextBlock(text="working")],
                model="claude-sonnet-4-6",
                message_id="first",
                session_id="native-1",
            ),
            StreamEvent(
                uuid="second-start",
                session_id="native-1",
                event={"type": "message_start", "message": {"id": "second"}},
            ),
        ]
    ])
    runtime = _AsyncRuntimeThread("sdk-budget")
    session._runtime = runtime
    session._client = client
    monkeypatch.setattr(session, "ensure_started", lambda: None)
    try:
        result = session.run_turn(
            agent=agent,
            user_input="loop",
            messages=[{"role": "user", "content": "loop"}],
            task_id="budget",
        )
    finally:
        session.close()

    assert result.budget_iterations == 1
    assert result.budget_exhausted is True
    assert result.should_retire is True
    assert agent.iteration_budget.used == 1
    assert client.interrupts == 1


def test_sdk_internal_tool_step_still_consumes_native_iteration_budget(
    tmp_path, monkeypatch
):
    agent, session = _session(tmp_path, monkeypatch)
    agent.iteration_budget = IterationBudget(1)
    client = _FakeClient([
        [
            StreamEvent(
                uuid="internal-start",
                session_id="native-1",
                event={"type": "message_start", "message": {"id": "internal"}},
            ),
            AssistantMessage(
                content=[
                    ServerToolUseBlock(
                        id="search-1",
                        name="tool_search_tool_regex",
                        input={"query": "echo"},
                    )
                ],
                model="claude-sonnet-4-6",
                message_id="internal",
                session_id="native-1",
            ),
            StreamEvent(
                uuid="next-start",
                session_id="native-1",
                event={"type": "message_start", "message": {"id": "next"}},
            ),
        ]
    ])
    runtime = _AsyncRuntimeThread("sdk-internal-budget")
    session._runtime = runtime
    session._client = client
    monkeypatch.setattr(session, "ensure_started", lambda: None)
    try:
        result = session.run_turn(
            agent=agent,
            user_input="find a tool",
            messages=[{"role": "user", "content": "find a tool"}],
            task_id="internal-budget",
        )
    finally:
        session.close()

    assert result.budget_iterations == 1
    assert result.budget_exhausted is True
    assert result.should_retire is True
    assert agent.iteration_budget.used == 1
    assert client.interrupts == 1


def test_sdk_manual_compaction_waits_for_native_boundary_without_turn_budget(
    tmp_path, monkeypatch
):
    agent, session = _session(tmp_path, monkeypatch)
    agent.iteration_budget = IterationBudget(0)
    session._turns_completed = 1
    client = _FakeClient([
        [
            StreamEvent(
                uuid="compact-start",
                session_id="native-1",
                event={"type": "message_start", "message": {"id": "compact"}},
            ),
            SystemMessage(
                subtype="compact_boundary",
                data={
                    "compact_metadata": {
                        "trigger": "manual",
                        "pre_tokens": 1234,
                    }
                },
            ),
            ResultMessage(
                subtype="success",
                duration_ms=10,
                duration_api_ms=8,
                is_error=False,
                num_turns=1,
                session_id="native-1",
                result="",
                terminal_reason="completed",
            ),
        ]
    ])
    runtime = _AsyncRuntimeThread("sdk-compact")
    session._runtime = runtime
    session._client = client
    monkeypatch.setattr(session, "ensure_started", lambda: None)
    try:
        result = session.compact(agent=agent, focus_topic="keep decisions")
    finally:
        session.close()

    assert result.error is None
    assert result.compacted is True
    assert result.compaction_count == 1
    assert result.compaction_metadata == {
        "trigger": "manual",
        "pre_tokens": 1234,
    }
    assert client.prompts == [("/compact keep decisions", "default")]
    assert agent.iteration_budget.used == 0
