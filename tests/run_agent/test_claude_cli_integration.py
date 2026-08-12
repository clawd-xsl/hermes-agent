from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import run_agent
from agent.transports.claude_cli_session import ClaudeCliTurnResult


class _FakeSession:
    def __init__(self):
        self.calls = []
        self.closed = False

    def run_turn(self, **kwargs):
        self.calls.append(kwargs)
        return ClaudeCliTurnResult(
            final_text="native answer",
            projected_messages=[{"role": "assistant", "content": "native answer"}],
            native_session_id="native-session",
            token_usage={"input_tokens": 8, "output_tokens": 2},
            last_call_usage={"input_tokens": 8, "output_tokens": 2},
            latency_ms={"first_text": 12, "total": 30},
        )

    def close(self):
        self.closed = True

    def interrupt(self):
        self.interrupted = True


class _SequenceSession(_FakeSession):
    def __init__(self, responses):
        super().__init__()
        self.responses = list(responses)

    def run_turn(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def _make_agent():
    return run_agent.AIAgent(
        model="claude-opus-4-6",
        provider="anthropic",
        api_mode="claude_cli",
        api_key="must-not-be-used",
        base_url="https://must-not-be-used.invalid",
        session_id="hermes-session",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )


def test_claude_cli_agent_does_not_construct_http_inference_client():
    agent = _make_agent()
    try:
        assert agent.api_mode == "claude_cli"
        assert agent.provider == "anthropic"
        assert agent.client is None
        assert agent.api_key == ""
        assert agent.base_url == ""
    finally:
        agent.close()


def test_main_native_structured_output_survives_turn_result(monkeypatch):
    agent = _make_agent()
    agent.request_overrides = {
        "response_format": {
            "type": "json_object",
        }
    }
    native = _SequenceSession(
        [
            ClaudeCliTurnResult(
                final_text='{"answer":"yes"}',
                structured_output={"answer": "yes"},
                projected_messages=[
                    {"role": "assistant", "content": '{"answer":"yes"}'}
                ],
                native_session_id="native-session",
                token_usage={"input_tokens": 8, "output_tokens": 2},
                last_call_usage={"input_tokens": 8, "output_tokens": 2},
            )
        ]
    )
    monkeypatch.setattr(
        "agent.claude_cli_runtime._get_session",
        lambda bound_agent, *, system_prompt: native,
    )
    with patch.object(agent, "_sync_external_memory_for_turn", return_value=None):
        result = agent.run_conversation("answer as JSON")
    try:
        assert result["final_response"] == '{"answer":"yes"}'
        assert result["structured_output"] == {"answer": "yes"}
        assert "ignored_request_controls" not in result
    finally:
        agent.close()


def test_native_session_uses_claude_specific_command_and_args(monkeypatch):
    from agent.claude_cli_runtime import _new_transient_session

    class _Loopback:
        def __init__(self, *_args, **_kwargs):
            pass

        def fingerprint(self):
            return "tools"

        def close(self):
            pass

    monkeypatch.setattr(
        "agent.transports.claude_cli_session.ClaudeToolLoopback",
        _Loopback,
    )
    monkeypatch.setattr("agent.claude_cli_runtime._runtime_config", lambda: {})
    agent = run_agent.AIAgent(
        model="claude-opus-4-6",
        provider="anthropic",
        api_mode="claude_cli",
        acp_command="/opt/claude",
        acp_args=["--debug-to-stderr"],
        service_tier="priority",
        session_id="custom-claude-command",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    session = _new_transient_session(
        agent,
        system_prompt="stable",
        turn_id="command-test",
    )
    try:
        assert session.command == "/opt/claude"
        assert session.extra_args == ["--debug-to-stderr"]
        assert session.fast_mode is True
    finally:
        session.close()
        agent.close()


def test_moa_turn_retires_private_guidance_at_turn_boundary(monkeypatch):
    agent = _make_agent()
    native = _FakeSession()

    def fake_get_session(bound_agent, *, system_prompt):
        bound_agent._claude_cli_session = native
        return native

    monkeypatch.setattr("agent.claude_cli_runtime._get_session", fake_get_session)
    monkeypatch.setattr(
        "agent.claude_cli_runtime.forget_claude_cli_binding",
        lambda _owner: None,
    )
    monkeypatch.setattr(
        "agent.moa_loop.aggregate_moa_context",
        lambda **_kwargs: "PRIVATE ADVISOR GUIDANCE",
    )

    with patch.object(agent, "_sync_external_memory_for_turn", return_value=None):
        result = agent.run_conversation(
            "use advisors",
            moa_config={
                "reference_models": [
                    {"provider": "openai", "model": "advisor"}
                ],
                "aggregator": {"provider": "anthropic", "model": "claude"},
            },
        )

    try:
        assert result["completed"] is True
        assert "PRIVATE ADVISOR GUIDANCE" in native.calls[0]["user_input"]
        assert native.closed is True
        assert agent._claude_cli_session is None
        assert "PRIVATE ADVISOR GUIDANCE" not in str(result["messages"])
    finally:
        agent.close()


def test_live_model_switch_can_enter_subscription_runtime_without_http_auth():
    agent = run_agent.AIAgent(
        model="old-model",
        provider="openrouter",
        api_mode="chat_completions",
        api_key="old-http-key",
        base_url="https://openrouter.ai/api/v1",
        session_id="switch-into-claude",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    try:
        with (
            patch("agent.credential_pool.load_pool", return_value=None),
            patch("agent.model_metadata.get_model_context_length", return_value=200_000),
            patch("hermes_cli.config.load_config", return_value={}),
        ):
            agent.switch_model(
                new_model="claude-opus-4-6",
                new_provider="anthropic",
                api_key="",
                base_url="",
                api_mode="claude_cli",
            )

        assert agent.model == "claude-opus-4-6"
        assert agent.provider == "anthropic"
        assert agent.api_mode == "claude_cli"
        assert agent.api_key == ""
        assert agent.base_url == ""
        assert agent.client is None
        assert agent._anthropic_client is None
        assert agent._credential_pool is None
        assert agent._primary_runtime["api_mode"] == "claude_cli"
        assert agent._primary_runtime["client_kwargs"] == {}
        assert agent._claude_cli_command is None
        assert agent._claude_cli_args == []
    finally:
        agent.close()


def test_live_model_switch_away_from_subscription_retires_native_session():
    agent = _make_agent()
    native = _FakeSession()
    agent._claude_cli_session = native
    try:
        with (
            patch("agent.credential_pool.load_pool", return_value=None),
            patch("agent.model_metadata.get_model_context_length", return_value=128_000),
            patch("hermes_cli.config.load_config", return_value={}),
        ):
            agent.switch_model(
                new_model="openai/gpt-5.4",
                new_provider="openrouter",
                api_key="new-http-key",
                base_url="https://openrouter.ai/api/v1",
                api_mode="chat_completions",
            )

        assert native.closed is True
        assert agent._claude_cli_session is None
        assert agent.provider == "openrouter"
        assert agent.api_mode == "chat_completions"
        assert agent.api_key == "new-http-key"
        assert agent.client is not None
        assert agent._primary_runtime["api_mode"] == "chat_completions"
    finally:
        agent.close()


def test_subscription_primary_restores_after_http_fallback_without_openai_client():
    agent = _make_agent()
    try:
        agent._fallback_activated = True
        agent.model = "fallback-model"
        agent.provider = "openrouter"
        agent.requested_provider = "openrouter"
        agent.base_url = "https://openrouter.ai/api/v1"
        agent.api_mode = "chat_completions"
        agent.api_key = "fallback-key"
        agent.client = MagicMock()
        agent._client_kwargs = {
            "api_key": "fallback-key",
            "base_url": "https://openrouter.ai/api/v1",
        }

        with (
            patch.object(
                agent,
                "_create_openai_client",
                side_effect=AssertionError(
                    "restoring claude_cli must not construct OpenAI"
                ),
            ),
            patch("agent.credential_pool.load_pool", return_value=None),
        ):
            restored = agent._restore_primary_runtime()

        assert restored is True
        assert agent.api_mode == "claude_cli"
        assert agent.provider == "anthropic"
        assert agent.model == "claude-opus-4-6"
        assert agent.api_key == ""
        assert agent.base_url == ""
        assert agent.client is None
        assert agent._fallback_activated is False
    finally:
        agent.close()


def test_conversation_loop_uses_native_runtime_and_persists_projection(monkeypatch):
    agent = _make_agent()
    native = _FakeSession()
    thinking = []
    agent.thinking_callback = thinking.append

    def fake_get_session(bound_agent, *, system_prompt):
        assert bound_agent is agent
        assert isinstance(system_prompt, str) and system_prompt
        bound_agent._claude_cli_session = native
        return native

    monkeypatch.setattr("agent.claude_cli_runtime._get_session", fake_get_session)
    with (
        patch.object(agent, "_sync_external_memory_for_turn", return_value=None),
        patch.object(agent, "_claim_stream_writer", return_value=1) as claim_writer,
    ):
        result = agent.run_conversation("hello")

    try:
        assert result["final_response"] == "native answer"
        assert result["completed"] is True
        assert result["agent_persisted"] is True
        assert result["claude_session_id"] == "native-session"
        assert result["claude_latency_ms"] == {"first_text": 12, "total": 30}
        assert native.calls[0]["agent"] is agent
        assert native.calls[0]["user_input"] == "hello"
        assert result["messages"][-1] == {
            "role": "assistant",
            "content": "native answer",
        }
        assert thinking[0]
        assert thinking[-1] == ""
        claim_writer.assert_called_once_with()
    finally:
        agent.close()


def test_native_turn_honors_interrupt_that_precedes_child_dispatch(monkeypatch):
    agent = _make_agent()
    agent.interrupt("stop before provider", hard_cancel=True)
    get_session = MagicMock(
        side_effect=AssertionError("interrupted turn must not start Claude")
    )
    monkeypatch.setattr("agent.claude_cli_runtime._get_session", get_session)

    with patch.object(agent, "_sync_external_memory_for_turn", return_value=None):
        result = agent.run_conversation("must not reach provider")

    try:
        assert result["interrupted"] is True
        assert result["turn_exit_reason"] == "interrupted_by_user"
        assert result["api_calls"] == 0
        assert result["completed"] is False
        get_session.assert_not_called()
    finally:
        agent.close()


def test_native_skill_review_cadence_preserves_mid_loop_skill_reset(monkeypatch):
    """A skill_manage call resets earlier private-loop iterations exactly once."""
    agent = _make_agent()
    agent._skill_nudge_interval = 2
    agent._iters_since_skill = 0
    agent.valid_tool_names = set(agent.valid_tool_names) | {"skill_manage"}

    class _SkillUsingSession(_FakeSession):
        def run_turn(self, **kwargs):
            self.calls.append(kwargs)
            observe = kwargs["iteration_post_callback"]
            observe(
                1,
                {"role": "assistant", "content": "", "tool_calls": []},
                {},
            )
            # This is the timing of the shared Hermes tool executor: the first
            # model step has completed, then skill_manage succeeds and resets
            # the cadence before Claude's follow-up model response.
            kwargs["agent"]._iters_since_skill = 0
            observe(
                2,
                {"role": "assistant", "content": "done"},
                {},
            )
            return ClaudeCliTurnResult(
                final_text="done",
                projected_messages=[
                    {"role": "assistant", "content": "done"}
                ],
                native_session_id="native-skill-reset",
                model_iterations=2,
            )

    native = _SkillUsingSession()
    monkeypatch.setattr(
        "agent.claude_cli_runtime._get_session",
        lambda bound_agent, **_kwargs: native,
    )
    with (
        patch.object(agent, "_sync_external_memory_for_turn", return_value=None),
        patch.object(agent, "_spawn_background_review") as review,
    ):
        result = agent.run_conversation("update the relevant skill")

    try:
        assert result["final_response"] == "done"
        assert agent._iters_since_skill == 1
        review.assert_not_called()
    finally:
        agent.close()


def test_native_budget_exhaustion_gets_isolated_toolless_summary(monkeypatch):
    agent = _make_agent()
    agent.max_iterations = 1
    native = _SequenceSession(
        [
            ClaudeCliTurnResult(
                final_text="working",
                projected_messages=[
                    {"role": "assistant", "content": "working"}
                ],
                model_iterations=1,
                budget_iterations=1,
                budget_exhausted=True,
                error="iteration budget exhausted",
                should_retire=True,
                native_session_id="exhausted-native",
                token_usage={"input_tokens": 8, "output_tokens": 2},
                last_call_usage={"input_tokens": 8, "output_tokens": 2},
            )
        ]
    )

    class SummarySession:
        def __init__(self):
            self.calls = []
            self.closed = False

        def summarize(self, **kwargs):
            self.calls.append(kwargs)
            return ClaudeCliTurnResult(
                final_text="Summary of completed work and remaining steps.",
                model_iterations=1,
                token_usage={"input_tokens": 12, "output_tokens": 5},
                last_call_usage={"input_tokens": 12, "output_tokens": 5},
            )

        def close(self):
            self.closed = True

        def interrupt(self):
            return None

    summary = SummarySession()

    def get_session(bound_agent, *, system_prompt):
        bound_agent._claude_cli_session = native
        return native

    def new_transient(bound_agent, **kwargs):
        assert bound_agent is agent
        assert kwargs["tool_definitions"] == []
        return summary

    monkeypatch.setattr("agent.claude_cli_runtime._get_session", get_session)
    monkeypatch.setattr(
        "agent.claude_cli_runtime._new_transient_session",
        new_transient,
    )
    monkeypatch.setattr(
        "agent.claude_cli_runtime._invalidate_persistent_native_history",
        lambda bound_agent, _session: setattr(
            bound_agent, "_claude_cli_session", None
        ),
    )
    with patch.object(agent, "_sync_external_memory_for_turn", return_value=None):
        result = agent.run_conversation("complete a long task")

    try:
        assert result["final_response"] == (
            "Summary of completed work and remaining steps."
        )
        assert result["messages"][-1] == {
            "role": "assistant",
            "content": "Summary of completed work and remaining steps.",
        }
        assert summary.calls
        assert any(
            row.get("content") == "working"
            for row in summary.calls[0]["messages"]
        )
        assert summary.closed is True
    finally:
        agent.close()


def test_native_runtime_receives_effective_user_and_system_context(monkeypatch):
    agent = _make_agent()
    native = _FakeSession()
    agent.ephemeral_system_prompt = "EPHEMERAL-SYSTEM-CONTEXT"

    def fake_get_session(bound_agent, *, system_prompt):
        assert bound_agent is agent
        assert "EPHEMERAL-SYSTEM-CONTEXT" in system_prompt
        bound_agent._claude_cli_session = native
        return native

    monkeypatch.setattr("agent.claude_cli_runtime._get_session", fake_get_session)
    monkeypatch.setattr(
        "agent.turn_context.compose_user_api_content",
        lambda content, *_args, **_kwargs: content + "\n\nRECALLED-AND-PLUGIN-CONTEXT",
    )
    with patch.object(agent, "_sync_external_memory_for_turn", return_value=None):
        result = agent.run_conversation("clean user text")

    try:
        assert result["completed"] is True
        assert native.calls[0]["user_input"] == (
            "clean user text\n\nRECALLED-AND-PLUGIN-CONTEXT"
        )
        # The transcript stays clean while the sidecar retains exact wire text.
        assert result["messages"][0]["content"] == "clean user text"
    finally:
        agent.close()


def test_native_runtime_uses_shared_finalizer_and_reenables_background_review(
    monkeypatch,
):
    agent = _make_agent()
    native = _FakeSession()
    agent._memory_nudge_interval = 1
    agent._skill_nudge_interval = 1
    agent.valid_tool_names.update({"memory", "skill_manage"})
    agent._memory_store = object()
    spawned = []
    lifecycle_calls = []

    def fake_get_session(bound_agent, *, system_prompt):
        bound_agent._claude_cli_session = native
        return native

    def fake_hook(name, **kwargs):
        lifecycle_calls.append((name, kwargs))
        if name == "transform_llm_output":
            return ["transformed native answer"]
        return []

    monkeypatch.setattr("agent.claude_cli_runtime._get_session", fake_get_session)
    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", fake_hook)
    monkeypatch.setattr(
        agent,
        "_spawn_background_review",
        lambda **kwargs: spawned.append(kwargs),
    )
    with patch.object(agent, "_sync_external_memory_for_turn", return_value=None):
        result = agent.run_conversation("remember this")

    try:
        assert result["final_response"] == "transformed native answer"
        assert result["response_transformed"] is True
        assert [
            name
            for name, _ in lifecycle_calls
            if name in {"transform_llm_output", "post_llm_call", "on_session_end"}
        ] == [
            "transform_llm_output",
            "post_llm_call",
            "on_session_end",
        ]
        assert len(spawned) == 1
        assert spawned[0]["review_memory"] is True
        assert spawned[0]["review_skills"] is True
        assert agent._iters_since_skill == 0
    finally:
        agent.close()


def test_native_runtime_fires_request_scoped_observer_hooks(monkeypatch):
    agent = _make_agent()
    native = _FakeSession()
    calls = []

    def fake_get_session(bound_agent, *, system_prompt):
        bound_agent._claude_cli_session = native
        return native

    monkeypatch.setattr("agent.claude_cli_runtime._get_session", fake_get_session)
    monkeypatch.setattr(
        "hermes_cli.lifecycle.has_hook",
        lambda name: name in {"pre_api_request", "post_api_request"},
    )
    monkeypatch.setattr(
        "hermes_cli.lifecycle.invoke_hook",
        lambda name, **kwargs: calls.append((name, kwargs)) or [],
    )
    with patch.object(agent, "_sync_external_memory_for_turn", return_value=None):
        result = agent.run_conversation("observe this")

    try:
        pre = [kwargs for name, kwargs in calls if name == "pre_api_request"]
        post = [kwargs for name, kwargs in calls if name == "post_api_request"]
        assert result["completed"] is True
        assert len(pre) == len(post) == 1
        assert pre[0]["api_request_id"] == post[0]["api_request_id"]
        assert pre[0]["api_mode"] == post[0]["api_mode"] == "claude_cli"
        assert pre[0]["request"]["body"]["transport"] == "claude_cli"
        assert post[0]["response"]["assistant_message"]["content"] == "native answer"
    finally:
        agent.close()


def test_native_runtime_fires_observer_hooks_for_each_inner_model_iteration(
    monkeypatch,
):
    agent = _make_agent()
    calls = []
    progress = []
    context_usage_updates = []
    agent._delegate_depth = 1
    agent.tool_progress_callback = lambda *args: progress.append(args)
    agent.context_compressor.update_from_response = (
        lambda usage: context_usage_updates.append(dict(usage))
    )

    class IteratingSession(_FakeSession):
        def run_turn(self, **kwargs):
            self.calls.append(kwargs)
            post = kwargs["iteration_post_callback"]
            before_next = kwargs["before_next_model_callback"]
            post(
                1,
                {
                    "role": "assistant",
                    "content": "checking",
                    "tool_calls": [
                        {
                            "id": "tool-1",
                            "type": "function",
                            "function": {"name": "echo", "arguments": "{}"},
                        }
                    ],
                },
                {"input_tokens": 5, "output_tokens": 2},
            )
            before_next()
            post(
                2,
                {"role": "assistant", "content": "native answer"},
                {
                    "input_tokens": 3,
                    "cache_read_input_tokens": 4,
                    "output_tokens": 1,
                },
            )
            return ClaudeCliTurnResult(
                final_text="native answer",
                projected_messages=[
                    {"role": "assistant", "content": "native answer"}
                ],
                model_iterations=2,
                native_session_id="native-session",
                token_usage={"input_tokens": 8, "output_tokens": 3},
                last_call_usage={
                    "input_tokens": 3,
                    "cache_read_input_tokens": 4,
                    "output_tokens": 1,
                },
            )

    native = IteratingSession()

    def fake_get_session(bound_agent, *, system_prompt):
        bound_agent._claude_cli_session = native
        return native

    monkeypatch.setattr("agent.claude_cli_runtime._get_session", fake_get_session)
    monkeypatch.setattr(
        "hermes_cli.lifecycle.has_hook",
        lambda name: name in {"pre_api_request", "post_api_request"},
    )
    monkeypatch.setattr(
        "hermes_cli.lifecycle.invoke_hook",
        lambda name, **kwargs: calls.append((name, kwargs)) or [],
    )
    with patch.object(agent, "_sync_external_memory_for_turn", return_value=None):
        result = agent.run_conversation("observe inner calls")

    try:
        pre = [kwargs for name, kwargs in calls if name == "pre_api_request"]
        post = [kwargs for name, kwargs in calls if name == "post_api_request"]
        assert result["completed"] is True
        assert len(pre) == len(post) == 2
        assert [item["api_call_count"] for item in pre] == [1, 2]
        assert [item["api_call_count"] for item in post] == [1, 2]
        assert [item["api_request_id"] for item in pre] == [
            item["api_request_id"] for item in post
        ]
        assert post[0]["finish_reason"] == "tool_calls"
        assert post[1]["finish_reason"] == "stop"
        assert post[1]["usage"]["prompt_tokens"] == 7
        assert progress == [
            ("_thinking", "checking"),
            ("_thinking", "native answer"),
        ]
        assert agent._last_turn_usage == {
            "prompt_tokens": 7,
            "completion_tokens": 1,
            "total_tokens": 8,
            "input_tokens": 3,
            "output_tokens": 1,
            "cache_read_tokens": 4,
            "cache_write_tokens": 0,
            "reasoning_tokens": 0,
        }
        assert context_usage_updates == [
            {
                "prompt_tokens": 5,
                "completion_tokens": 2,
                "total_tokens": 7,
                "input_tokens": 5,
                "output_tokens": 2,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
                "reasoning_tokens": 0,
            },
            {
                "prompt_tokens": 7,
                "completion_tokens": 1,
                "total_tokens": 8,
                "input_tokens": 3,
                "output_tokens": 1,
                "cache_read_tokens": 4,
                "cache_write_tokens": 0,
                "reasoning_tokens": 0,
            },
        ]
    finally:
        agent.close()


def test_native_context_engine_recovers_final_usage_from_result_frame(monkeypatch):
    agent = _make_agent()
    context_usage_updates = []
    agent.context_compressor.update_from_response = (
        lambda usage: context_usage_updates.append(dict(usage))
    )

    class ResultUsageSession(_FakeSession):
        def run_turn(self, **kwargs):
            self.calls.append(kwargs)
            post = kwargs["iteration_post_callback"]
            before_next = kwargs["before_next_model_callback"]
            # Some Claude stream-json versions omit usage from assistant
            # events and expose it only in the terminal result frame.
            post(
                1,
                {
                    "role": "assistant",
                    "content": "checking",
                    "tool_calls": [
                        {
                            "id": "tool-1",
                            "type": "function",
                            "function": {"name": "echo", "arguments": "{}"},
                        }
                    ],
                },
                {},
            )
            before_next()
            post(
                2,
                {"role": "assistant", "content": "native answer"},
                {},
            )
            return ClaudeCliTurnResult(
                final_text="native answer",
                projected_messages=[
                    {"role": "assistant", "content": "native answer"}
                ],
                model_iterations=2,
                native_session_id="native-session",
                token_usage={"input_tokens": 14, "output_tokens": 4},
                last_call_usage={
                    "input_tokens": 3,
                    "cache_read_input_tokens": 7,
                    "output_tokens": 1,
                },
            )

    native = ResultUsageSession()
    monkeypatch.setattr(
        "agent.claude_cli_runtime._get_session",
        lambda bound_agent, **_kwargs: native,
    )
    with patch.object(agent, "_sync_external_memory_for_turn", return_value=None):
        result = agent.run_conversation("recover result usage")

    try:
        assert result["completed"] is True
        assert context_usage_updates == [
            {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
                "reasoning_tokens": 0,
            },
            {
                "prompt_tokens": 10,
                "completion_tokens": 1,
                "total_tokens": 11,
                "input_tokens": 3,
                "output_tokens": 1,
                "cache_read_tokens": 7,
                "cache_write_tokens": 0,
                "reasoning_tokens": 0,
            },
        ]
    finally:
        agent.close()


def test_native_internal_continuation_pre_hook_sees_provisional_text(
    monkeypatch,
):
    agent = _make_agent()
    calls = []
    provisional = {"role": "assistant", "content": "partial answer"}

    class InternallyContinuingSession(_FakeSession):
        def run_turn(self, **kwargs):
            self.calls.append(kwargs)
            project = kwargs["projection_callback"]
            post = kwargs["iteration_post_callback"]
            before_next = kwargs["before_next_model_callback"]
            project([provisional])
            post(1, provisional, {"input_tokens": 5, "output_tokens": 2})
            before_next()
            final = {"role": "assistant", "content": "complete answer"}
            project([final])
            post(2, final, {"input_tokens": 7, "output_tokens": 3})
            return ClaudeCliTurnResult(
                final_text="complete answer",
                projected_messages=[provisional, final],
                model_iterations=2,
                native_session_id="native-session",
                token_usage={"input_tokens": 12, "output_tokens": 5},
                last_call_usage={"input_tokens": 7, "output_tokens": 3},
            )

    native = InternallyContinuingSession()

    def fake_get_session(bound_agent, *, system_prompt):
        bound_agent._claude_cli_session = native
        return native

    monkeypatch.setattr("agent.claude_cli_runtime._get_session", fake_get_session)
    monkeypatch.setattr(
        "hermes_cli.lifecycle.has_hook",
        lambda name: name in {"pre_api_request", "post_api_request"},
    )
    monkeypatch.setattr(
        "hermes_cli.lifecycle.invoke_hook",
        lambda name, **kwargs: calls.append((name, kwargs)) or [],
    )
    with patch.object(agent, "_sync_external_memory_for_turn", return_value=None):
        result = agent.run_conversation("continue without a tool")

    try:
        pre = [kwargs for name, kwargs in calls if name == "pre_api_request"]
        assert result["final_response"] == "complete answer"
        assert len(pre) == 2
        assert pre[1]["request_messages"][-1] == {
            "role": "assistant",
            "content": "partial answer",
        }
        assert [row["role"] for row in result["messages"]] == [
            "user",
            "assistant",
        ]
        assert result["messages"][-1]["content"] == (
            "partial answer\ncomplete answer"
        )
    finally:
        agent.close()


def test_native_pre_model_boundary_delivers_late_steer_in_tool_result(
    monkeypatch,
):
    agent = _make_agent()

    class LateSteerSession(_FakeSession):
        def run_turn(self, **kwargs):
            self.calls.append(kwargs)
            rows = [
                {
                    "role": "assistant",
                    "content": "checking",
                    "tool_calls": [
                        {
                            "id": "tool-1",
                            "type": "function",
                            "function": {"name": "echo", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "tool-1",
                    "content": "original tool result",
                },
            ]
            kwargs["projection_callback"](rows)
            assert kwargs["agent"].steer("use the corrected target") is True
            kwargs["before_next_model_callback"]()
            return ClaudeCliTurnResult(
                final_text="corrected answer",
                projected_messages=[
                    *rows,
                    {"role": "assistant", "content": "corrected answer"},
                ],
                model_iterations=2,
                native_session_id="native-session",
            )

    native = LateSteerSession()
    monkeypatch.setattr(
        "agent.claude_cli_runtime._get_session",
        lambda bound_agent, **_kwargs: native,
    )
    with patch.object(agent, "_sync_external_memory_for_turn", return_value=None):
        result = agent.run_conversation("inspect first")

    try:
        tool_row = next(
            row for row in result["messages"] if row.get("role") == "tool"
        )
        assert "original tool result" in tool_row["content"]
        assert "use the corrected target" in tool_row["content"]
        assert agent._pending_steer is None
    finally:
        agent.close()


def test_changed_context_selection_uses_isolated_one_turn_native_child(monkeypatch):
    agent = _make_agent()
    transient = _FakeSession()
    invalidated = []
    selected = [
        {"role": "system", "content": "selected system"},
        {"role": "user", "content": "retrieved branch context"},
        {"role": "assistant", "content": "branch answer"},
        {"role": "user", "content": "selected current input"},
    ]

    monkeypatch.setattr(
        "agent.claude_cli_runtime._select_native_request_context",
        lambda *_args, **_kwargs: (selected, True),
    )
    monkeypatch.setattr(
        "agent.claude_cli_runtime._new_transient_session",
        lambda bound_agent, **_kwargs: transient,
    )
    monkeypatch.setattr(
        "agent.claude_cli_runtime._get_session",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("persistent native history must not bypass selected context")
        ),
    )
    monkeypatch.setattr(
        "agent.claude_cli_runtime._invalidate_persistent_native_history",
        lambda bound_agent, previous: invalidated.append((bound_agent, previous)),
    )
    with patch.object(agent, "_sync_external_memory_for_turn", return_value=None):
        result = agent.run_conversation("clean current input")

    try:
        assert result["completed"] is True
        assert transient.closed is True
        assert transient.calls[0]["user_input"] == "selected current input"
        assert transient.calls[0]["bootstrap_messages"] == selected[1:]
        assert invalidated == [(agent, None)]
    finally:
        agent.close()


def test_shared_finalizer_does_not_micro_compact_native_history(monkeypatch):
    agent = _make_agent()
    native = _FakeSession()
    micro_calls = []
    agent.context_compressor._micro_compact_enabled = True
    agent.context_compressor._micro_compact = (
        lambda messages: micro_calls.append(list(messages)) or messages
    )

    def fake_get_session(bound_agent, *, system_prompt):
        bound_agent._claude_cli_session = native
        return native

    monkeypatch.setattr("agent.claude_cli_runtime._get_session", fake_get_session)
    with patch.object(agent, "_sync_external_memory_for_turn", return_value=None):
        result = agent.run_conversation("keep native history authoritative")

    try:
        assert result["completed"] is True
        assert micro_calls == []
    finally:
        agent.close()


def test_automatic_native_compaction_runs_hermes_boundary_and_carries_memory(
    monkeypatch,
):
    agent = _make_agent()

    class CompactingSession(_FakeSession):
        def run_turn(self, **kwargs):
            self.calls.append(kwargs)
            return ClaudeCliTurnResult(
                final_text="native answer",
                projected_messages=[
                    {"role": "assistant", "content": "native answer"}
                ],
                native_session_id="native-session",
                token_usage={"input_tokens": 8, "output_tokens": 2},
                last_call_usage={"input_tokens": 8, "output_tokens": 2},
                compacted=len(self.calls) == 1,
                compaction_count=1 if len(self.calls) == 1 else 0,
                compaction_metadata={"preTokens": 120_000},
            )

    native = CompactingSession()
    pre_calls = []
    switch_calls = []
    context_boundaries = []
    events = []
    statuses = []
    agent._memory_manager = SimpleNamespace(
        on_pre_compress=lambda messages: pre_calls.append(list(messages))
        or "provider checkpoint alpha",
        on_session_switch=lambda session_id, **kwargs: switch_calls.append(
            (session_id, kwargs)
        ),
    )
    agent.context_compressor.on_session_start = (
        lambda session_id, **kwargs: context_boundaries.append((session_id, kwargs))
    )
    agent.event_callback = lambda name, payload: events.append((name, payload))
    agent._emit_status = statuses.append

    def fake_get_session(bound_agent, *, system_prompt):
        bound_agent._claude_cli_session = native
        return native

    monkeypatch.setattr("agent.claude_cli_runtime._get_session", fake_get_session)
    with patch.object(agent, "_sync_external_memory_for_turn", return_value=None):
        first = agent.run_conversation("first message")
        second = agent.run_conversation("second message")

    try:
        assert first["completed"] is True
        assert second["completed"] is True
        assert len(pre_calls) == 1
        assert switch_calls == [
            (
                "hermes-session",
                {
                    "parent_session_id": "hermes-session",
                    "reset": False,
                    "reason": "compression",
                },
            )
        ]
        assert context_boundaries[0][1]["boundary_reason"] == "compression"
        assert agent.context_compressor.compression_count == 1
        assert events[0][0] == "session:compress"
        assert events[0][1]["automatic"] is True
        assert events[0][1]["native"] is True
        assert "provider checkpoint alpha" in native.calls[1]["user_input"]
        assert native.calls[1]["user_input"].endswith("second message")
        assert agent._claude_cli_pending_compaction_context == ""
        # Provider continuity is wire-only; the visible transcript stays clean.
        assert all(
            "provider checkpoint alpha" not in str(message.get("content", ""))
            for message in second["messages"]
        )
        second_user = next(
            message
            for message in reversed(second["messages"])
            if message.get("role") == "user"
        )
        assert "provider checkpoint alpha" in str(second_user["api_content"])
        assert statuses
    finally:
        agent.close()


def test_native_request_and_execution_middleware_are_not_bypassed(monkeypatch):
    agent = _make_agent()
    transient = _FakeSession()
    execution_calls = []

    def apply_request(request, **_context):
        changed = dict(request)
        changed["messages"] = [dict(row) for row in request["messages"]]
        changed["messages"][-1]["content"] = "middleware-selected input"
        changed["temperature"] = 0.2
        changed["transport"] = "openai"
        return SimpleNamespace(
            payload=changed,
            original_payload=request,
            trace=[{"source": "test-middleware"}],
        )

    def run_execution(request, next_call, **context):
        execution_calls.append((request, context))
        return next_call(request)

    monkeypatch.setattr(
        "hermes_cli.middleware.apply_llm_request_middleware",
        apply_request,
    )
    monkeypatch.setattr(
        "hermes_cli.middleware.run_llm_execution_middleware",
        run_execution,
    )
    monkeypatch.setattr(
        "agent.claude_cli_runtime._new_transient_session",
        lambda bound_agent, **_kwargs: transient,
    )
    with patch.object(agent, "_sync_external_memory_for_turn", return_value=None):
        result = agent.run_conversation("clean input")

    try:
        assert result["completed"] is True
        assert transient.calls[0]["user_input"] == "middleware-selected input"
        assert execution_calls[0][1]["middleware_trace"] == [
            {"source": "test-middleware"}
        ]
        assert result["ignored_request_controls"] == [
            "request_middleware.temperature",
            "request_middleware.transport",
        ]
    finally:
        agent.close()


def test_native_execution_middleware_cannot_silently_replace_transport(monkeypatch):
    agent = _make_agent()
    persistent = _FakeSession()

    def run_execution(request, next_call, **_context):
        changed = dict(request)
        changed["transport"] = "openai"
        return next_call(changed)

    monkeypatch.setattr(
        "hermes_cli.middleware.run_llm_execution_middleware",
        run_execution,
    )
    monkeypatch.setattr(
        "agent.claude_cli_runtime._get_session",
        lambda bound_agent, **_kwargs: persistent,
    )
    with patch.object(agent, "_sync_external_memory_for_turn", return_value=None):
        result = agent.run_conversation("clean input")

    try:
        assert result["completed"] is False
        assert "cannot replace the selected native transport" in result["error"]
    finally:
        agent.close()


def test_native_execution_middleware_mutation_uses_isolated_child(monkeypatch):
    agent = _make_agent()
    persistent = _FakeSession()
    transient = _FakeSession()
    invalidated = []

    def get_session(bound_agent, **_kwargs):
        bound_agent._claude_cli_session = persistent
        return persistent

    def run_execution(request, next_call, **_context):
        changed = dict(request)
        changed["messages"] = [dict(row) for row in request["messages"]]
        changed["messages"][-1]["content"] = "execution-selected input"
        return next_call(changed)

    monkeypatch.setattr("agent.claude_cli_runtime._get_session", get_session)
    monkeypatch.setattr(
        "agent.claude_cli_runtime._new_transient_session",
        lambda *_args, **_kwargs: transient,
    )
    monkeypatch.setattr(
        "hermes_cli.middleware.run_llm_execution_middleware",
        run_execution,
    )
    monkeypatch.setattr(
        "agent.claude_cli_runtime._invalidate_persistent_native_history",
        lambda bound_agent, previous: invalidated.append((bound_agent, previous)),
    )

    with patch.object(agent, "_sync_external_memory_for_turn", return_value=None):
        result = agent.run_conversation("original input")

    try:
        assert result["completed"] is True
        assert persistent.calls == []
        assert transient.calls[0]["user_input"] == "execution-selected input"
        assert invalidated == [(agent, persistent)]
    finally:
        agent.close()


def test_native_physical_attempt_runs_through_relay(monkeypatch):
    agent = _make_agent()
    native = _FakeSession()
    relay_calls = []
    completions = []

    def execute(request, callback, **context):
        relay_calls.append((request, context))
        return callback(request)

    monkeypatch.setattr(
        "agent.claude_cli_runtime._get_session",
        lambda bound_agent, **_kwargs: native,
    )
    monkeypatch.setattr("agent.relay_llm.execute", execute)
    monkeypatch.setattr(
        "agent.relay_llm.complete_logical_call",
        lambda request_id, **kwargs: completions.append((request_id, kwargs)),
    )

    with patch.object(agent, "_sync_external_memory_for_turn", return_value=None):
        result = agent.run_conversation("relay this")

    try:
        assert result["completed"] is True
        assert len(relay_calls) == 1
        assert relay_calls[0][1]["metadata"]["api_mode"] == "claude_cli"
        assert relay_calls[0][1]["metadata"]["call_role"] == "primary"
        assert relay_calls[0][1]["defer_logical_completion"] is True
        assert completions[0][1]["outcome"] == "success"
    finally:
        agent.close()


def test_native_relay_response_mutation_is_applied_and_retires_history(
    monkeypatch,
):
    agent = _make_agent()

    class _ProjectingSession(_FakeSession):
        def run_turn(self, **kwargs):
            self.calls.append(kwargs)
            original = {"role": "assistant", "content": "native answer"}
            # Real stream-json projects the assistant record before the Relay
            # / execution-middleware stack gets the completed turn result.
            kwargs["projection_callback"]([original])
            return ClaudeCliTurnResult(
                final_text="native answer",
                projected_messages=[original],
                native_session_id="native-session",
                model_iterations=1,
            )

    native = _ProjectingSession()
    invalidated = []

    def execute(request, callback, **_context):
        raw = callback(request)
        changed = dict(vars(raw))
        changed["final_text"] = "relay-filtered answer"
        changed["projected_messages"] = [
            {"role": "assistant", "content": "relay-filtered answer"}
        ]
        return SimpleNamespace(**changed)

    monkeypatch.setattr(
        "agent.claude_cli_runtime._get_session",
        lambda bound_agent, **_kwargs: native,
    )
    monkeypatch.setattr("agent.relay_llm.execute", execute)
    monkeypatch.setattr(
        "agent.claude_cli_runtime._invalidate_persistent_native_history",
        lambda bound_agent, previous: invalidated.append((bound_agent, previous)),
    )

    with patch.object(agent, "_sync_external_memory_for_turn", return_value=None):
        result = agent.run_conversation("filter this")

    try:
        assert result["final_response"] == "relay-filtered answer"
        assert [
            row.get("content")
            for row in result["messages"]
            if row.get("role") == "assistant"
        ] == ["relay-filtered answer"]
        assert invalidated == [(agent, native)]
    finally:
        agent.close()


def test_execution_middleware_short_circuit_invalidates_missed_native_history(
    monkeypatch,
):
    agent = _make_agent()
    native = _FakeSession()
    invalidated = []

    def get_session(bound_agent, **_kwargs):
        bound_agent._claude_cli_session = native
        return native

    def short_circuit(_request, _next_call, **_context):
        return ClaudeCliTurnResult(
            final_text="middleware answer",
            projected_messages=[
                {"role": "assistant", "content": "middleware answer"}
            ],
        )

    monkeypatch.setattr("agent.claude_cli_runtime._get_session", get_session)
    monkeypatch.setattr(
        "hermes_cli.middleware.run_llm_execution_middleware",
        short_circuit,
    )
    monkeypatch.setattr(
        "agent.claude_cli_runtime._invalidate_persistent_native_history",
        lambda bound_agent, previous: invalidated.append((bound_agent, previous)),
    )
    with patch.object(agent, "_sync_external_memory_for_turn", return_value=None):
        result = agent.run_conversation("cached request")

    try:
        assert result["final_response"] == "middleware answer"
        assert native.calls == []
        assert invalidated == [(agent, native)]
    finally:
        agent.close()


def test_native_startup_failure_counts_the_attempt_and_reports_it_to_hooks(
    monkeypatch,
):
    agent = _make_agent()
    agent._memory_nudge_interval = 1
    agent._skill_nudge_interval = 1
    agent._turns_since_memory = 0
    agent.valid_tool_names = set(agent.valid_tool_names) | {
        "memory",
        "skill_manage",
    }
    agent._memory_store = MagicMock()
    monkeypatch.setattr(
        "agent.claude_cli_runtime._get_session",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("native process unavailable")
        ),
    )

    with (
        patch.object(agent, "_sync_external_memory_for_turn", return_value=None),
        patch.object(agent, "_invoke_api_request_error_hook") as error_hook,
        patch.object(agent, "_spawn_background_review") as background_review,
    ):
        result = agent.run_conversation("try native")

    try:
        assert result["completed"] is False
        assert result["api_calls"] == 2
        assert agent._api_call_count == 2
        assert error_hook.call_count == 2
        assert error_hook.call_args.kwargs["api_call_count"] == 2
        assert error_hook.call_args.kwargs["retry_count"] == 1
        assert error_hook.call_args.kwargs["max_retries"] == 1
        assert error_hook.call_args.kwargs["reason"] == "unknown"
        assert error_hook.call_args.kwargs["retryable"] is True
        background_review.assert_called_once()
        assert background_review.call_args.kwargs["review_memory"] is True
        assert background_review.call_args.kwargs["review_skills"] is True
        assert agent._iters_since_skill == 0
    finally:
        agent.close()


def test_native_final_answer_gates_continue_on_same_warm_session(monkeypatch):
    agent = _make_agent()
    native = _SequenceSession(
        [
            ClaudeCliTurnResult(
                final_text="draft answer",
                projected_messages=[{"role": "assistant", "content": "draft answer"}],
                native_session_id="native-session",
            ),
            ClaudeCliTurnResult(
                final_text="verified answer",
                projected_messages=[{"role": "assistant", "content": "verified answer"}],
                native_session_id="native-session",
                session_reuse="warm_hit",
            ),
        ]
    )
    # The production session factory publishes the live child on the agent;
    # this test replaces that factory, so mirror the ownership side effect.
    agent._claude_cli_session = native
    checks = []
    request_counts = []
    execution_counts = []

    def continuation(*_args, **_kwargs):
        checks.append(True)
        if len(checks) == 1:
            return "run verification now", "_verification_stop_synthetic", "verification_required", 0
        return None, None, "", 0

    def apply_request(request, **context):
        request_counts.append(context["api_call_count"])
        return SimpleNamespace(
            payload=request,
            original_payload=request,
            trace=[],
        )

    def run_execution(request, next_call, **context):
        execution_counts.append(context["api_call_count"])
        return next_call(request)

    monkeypatch.setattr(
        "agent.claude_cli_runtime._get_session",
        lambda bound_agent, **_kwargs: native,
    )
    monkeypatch.setattr(
        "agent.claude_cli_runtime._native_continuation_nudge",
        continuation,
    )
    monkeypatch.setattr(
        "hermes_cli.middleware.apply_llm_request_middleware",
        apply_request,
    )
    monkeypatch.setattr(
        "hermes_cli.middleware.run_llm_execution_middleware",
        run_execution,
    )
    with patch.object(agent, "_sync_external_memory_for_turn", return_value=None):
        result = agent.run_conversation("make the change")

    try:
        assert result["final_response"] == "verified answer"
        assert result["api_calls"] == 2
        assert agent._api_call_count == 2
        assert len(native.calls) == 2
        assert request_counts == [1, 2]
        assert execution_counts == [1, 2]
        assert native.calls[1]["user_input"] == "run verification now"
        assert all(
            not row.get("_verification_stop_synthetic")
            for row in result["messages"]
        )
        # The native thread still contains the private verification nudge that
        # the shared finalizer removed from Hermes history.  It must not be
        # warm-resumed under a signature for the cleaned transcript.
        assert native.closed is True
        assert agent._claude_cli_session is None
    finally:
        agent.close()


def test_native_empty_response_retries_then_returns_visible_answer(monkeypatch):
    agent = _make_agent()
    agent._retry_status_buffer = [("status", "transient native retry")]
    native = _SequenceSession(
        [
            ClaudeCliTurnResult(
                final_text="",
                native_session_id="native-session",
                model_iterations=1,
            ),
            ClaudeCliTurnResult(
                final_text="answer after empty retry",
                projected_messages=[
                    {"role": "assistant", "content": "answer after empty retry"}
                ],
                native_session_id="native-session",
                model_iterations=1,
                session_reuse="warm_hit",
            ),
        ]
    )

    monkeypatch.setattr(
        "agent.claude_cli_runtime._get_session",
        lambda bound_agent, **_kwargs: native,
    )
    monkeypatch.setattr(
        "agent.claude_cli_runtime._native_empty_retry_wait",
        lambda *_args, **_kwargs: True,
    )
    with patch.object(agent, "_sync_external_memory_for_turn", return_value=None):
        result = agent.run_conversation("answer me")

    try:
        assert result["final_response"] == "answer after empty retry"
        assert result["api_calls"] == 2
        assert native.calls[1]["user_input"].startswith(
            "[System: Your previous response was empty."
        )
        assert all(
            not row.get("_empty_recovery_synthetic")
            for row in result["messages"]
        )
        assert agent._retry_status_buffer == []
    finally:
        agent.close()


def test_native_max_tokens_stop_continues_and_assembles_complete_answer(
    monkeypatch,
):
    agent = _make_agent()
    native = _SequenceSession(
        [
            ClaudeCliTurnResult(
                final_text="first half ",
                projected_messages=[
                    {
                        "role": "assistant",
                        "content": "first half ",
                        "finish_reason": "length",
                    }
                ],
                native_session_id="native-session",
                model_iterations=1,
                last_stop_reason="max_tokens",
            ),
            ClaudeCliTurnResult(
                final_text="second half",
                projected_messages=[
                    {"role": "assistant", "content": "second half"}
                ],
                native_session_id="native-session",
                model_iterations=1,
                session_reuse="warm_hit",
                last_stop_reason="end_turn",
            ),
        ]
    )

    monkeypatch.setattr(
        "agent.claude_cli_runtime._get_session",
        lambda bound_agent, **_kwargs: native,
    )
    with patch.object(agent, "_sync_external_memory_for_turn", return_value=None):
        result = agent.run_conversation("write the long answer")

    try:
        assert result["final_response"] == "first half second half"
        assert result["api_calls"] == 2
        assert result["partial"] is False
        assert native.calls[1]["user_input"].startswith(
            "[System: Your previous response was truncated by the output "
        )
        assert [row["role"] for row in result["messages"]] == [
            "user",
            "assistant",
            "user",
            "assistant",
        ]
    finally:
        agent.close()


def test_native_max_tokens_continuation_is_bounded_and_returns_partial(
    monkeypatch,
):
    agent = _make_agent()
    native = _SequenceSession(
        [
            ClaudeCliTurnResult(
                final_text=f"part-{attempt} ",
                projected_messages=[
                    {
                        "role": "assistant",
                        "content": f"part-{attempt} ",
                        "finish_reason": "length",
                    }
                ],
                native_session_id="native-session",
                model_iterations=1,
                session_reuse="warm_hit" if attempt else "cold_miss",
                last_stop_reason="max_tokens",
            )
            for attempt in range(4)
        ]
    )

    monkeypatch.setattr(
        "agent.claude_cli_runtime._get_session",
        lambda bound_agent, **_kwargs: native,
    )
    with patch.object(agent, "_sync_external_memory_for_turn", return_value=None):
        result = agent.run_conversation("write a very long answer")

    try:
        assert result["final_response"] == "part-0 part-1 part-2 part-3"
        assert result["api_calls"] == 4
        assert result["partial"] is True
        assert result["turn_exit_reason"] == "claude_cli_output_truncated"
        assert result["error"] == (
            "Response remained truncated after 4 continuation attempts"
        )
        assert len(native.calls) == 4
    finally:
        agent.close()


def test_native_thinking_budget_exhaustion_does_not_retry_continuation(
    monkeypatch,
):
    agent = _make_agent()
    native = _SequenceSession(
        [
            ClaudeCliTurnResult(
                final_text="",
                model_iterations=1,
                budget_iterations=1,
                last_stop_reason="max_tokens",
                thinking_budget_exhausted=True,
                native_session_id="native-session",
            )
        ]
    )
    agent._claude_cli_session = native
    monkeypatch.setattr(
        "agent.claude_cli_runtime._get_session",
        lambda _agent, **_kwargs: native,
    )

    with patch.object(agent, "_sync_external_memory_for_turn", return_value=None):
        result = agent.run_conversation("solve it")

    try:
        assert len(native.calls) == 1
        assert result["completed"] is False
        assert result["partial"] is True
        assert result["turn_exit_reason"] == (
            "claude_cli_thinking_budget_exhausted"
        )
        assert "Thinking Budget Exhausted" in result["final_response"]
        assert "reasoning" in result["error"].lower()
        assert native.closed is True
    finally:
        agent.close()


def test_native_refusal_is_terminal_policy_result_not_empty_retry(monkeypatch):
    agent = _make_agent()
    native = _SequenceSession(
        [
            ClaudeCliTurnResult(
                final_text="",
                native_session_id="native-session",
                model_iterations=1,
                last_stop_reason="refusal",
                token_usage={"input_tokens": 4, "output_tokens": 0},
            )
        ]
    )
    monkeypatch.setattr(
        "agent.claude_cli_runtime._get_session",
        lambda bound_agent, **_kwargs: native,
    )
    monkeypatch.setattr(agent, "_try_activate_fallback", lambda reason=None: False)
    with (
        patch.object(agent, "_sync_external_memory_for_turn", return_value=None),
        patch.object(agent, "_invoke_api_request_error_hook") as error_hook,
    ):
        result = agent.run_conversation("request that Claude refuses")

    try:
        assert len(native.calls) == 1
        assert result["api_calls"] == 1
        assert result["failed"] is True
        assert result["completed"] is False
        assert result["partial"] is False
        assert result["turn_exit_reason"] == "content_policy_blocked"
        assert result["error"] == (
            "content_policy_blocked: model declined (content_filter)"
        )
        assert "safety refusal" in result["final_response"]
        assert "fallback provider" in result["final_response"]
        error_hook.assert_called_once()
        assert error_hook.call_args.kwargs["reason"] == "content_policy_blocked"
        assert error_hook.call_args.kwargs["retryable"] is False
    finally:
        agent.close()


def test_native_refusal_activates_standard_fallback_same_turn(monkeypatch):
    agent = _make_agent()
    native = _SequenceSession(
        [
            ClaudeCliTurnResult(
                final_text="I cannot help with that.",
                projected_messages=[
                    {"role": "assistant", "content": "I cannot help with that."}
                ],
                native_session_id="native-session",
                model_iterations=1,
                last_stop_reason="refusal",
            )
        ]
    )
    fallback_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="fallback answered safely",
                    tool_calls=None,
                    reasoning=None,
                    reasoning_content=None,
                    reasoning_details=None,
                ),
                finish_reason="stop",
            )
        ],
        model="fallback/model",
        usage=None,
    )
    agent._fallback_chain = [{"provider": "openai", "model": "fallback/model"}]
    agent._fallback_index = 0
    activated_reasons = []

    def activate_fallback(reason=None):
        activated_reasons.append(getattr(reason, "value", reason))
        agent._fallback_index = 1
        agent._fallback_activated = True
        agent.api_mode = "chat_completions"
        agent.provider = "openai"
        agent.requested_provider = "openai"
        agent.model = "fallback/model"
        agent.base_url = "https://fallback.invalid/v1"
        agent.client = MagicMock()
        agent._transport_cache.clear()
        return True

    monkeypatch.setattr(
        "agent.claude_cli_runtime._get_session",
        lambda bound_agent, **_kwargs: native,
    )
    monkeypatch.setattr(agent, "_try_activate_fallback", activate_fallback)
    monkeypatch.setattr(
        agent,
        "_interruptible_api_call",
        lambda _kwargs: fallback_response,
    )
    with (
        patch.object(agent, "_sync_external_memory_for_turn", return_value=None),
        patch.object(agent, "_invoke_api_request_error_hook") as error_hook,
    ):
        result = agent.run_conversation("answer via whichever provider can")

    try:
        assert result["final_response"] == "fallback answered safely"
        assert result["api_calls"] == 2
        assert activated_reasons == ["content_policy_blocked"]
        assert error_hook.call_args.kwargs["reason"] == "content_policy_blocked"
        assert native.closed is True
        assert [row["content"] for row in result["messages"]] == [
            "answer via whichever provider can",
            "fallback answered safely",
        ]
    finally:
        agent.close()


def test_native_empty_after_housekeeping_uses_already_delivered_answer(
    monkeypatch,
):
    agent = _make_agent()
    native = _SequenceSession(
        [
            ClaudeCliTurnResult(
                final_text="",
                projected_messages=[
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
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "memory-1",
                        "content": "saved",
                    },
                ],
                native_session_id="native-session",
                model_iterations=1,
            )
        ]
    )

    monkeypatch.setattr(
        "agent.claude_cli_runtime._get_session",
        lambda bound_agent, **_kwargs: native,
    )
    with patch.object(agent, "_sync_external_memory_for_turn", return_value=None):
        result = agent.run_conversation("remember this")

    try:
        assert result["final_response"] == "Done — I will remember that."
        assert result["response_previewed"] is True
        assert len(native.calls) == 1
    finally:
        agent.close()


def test_native_empty_response_exhaustion_is_visible_and_drops_scaffolding(
    monkeypatch,
):
    agent = _make_agent()
    native = _SequenceSession(
        [
            ClaudeCliTurnResult(
                final_text="",
                native_session_id="native-session",
                model_iterations=1,
                session_reuse="warm_hit" if attempt else "cold_miss",
            )
            for attempt in range(4)
        ]
    )

    monkeypatch.setattr(
        "agent.claude_cli_runtime._get_session",
        lambda bound_agent, **_kwargs: native,
    )
    monkeypatch.setattr(
        "agent.claude_cli_runtime._native_empty_retry_wait",
        lambda *_args, **_kwargs: True,
    )
    with patch.object(agent, "_sync_external_memory_for_turn", return_value=None):
        result = agent.run_conversation("answer me")

    try:
        assert result["turn_exit_reason"] == "empty_response_exhausted"
        assert result["api_calls"] == 4
        assert "No reply:" in result["final_response"]
        assert len(native.calls) == 4
        assert all(
            not row.get("_empty_recovery_synthetic")
            for row in result["messages"]
        )
        assert all(
            not row.get("_empty_terminal_sentinel")
            for row in result["messages"]
        )
    finally:
        agent.close()


def test_native_tool_protocol_persistence_failure_is_terminal_and_visible(
    monkeypatch,
):
    agent = _make_agent()

    class _PersistenceFailingSession(_FakeSession):
        def run_turn(self, **kwargs):
            self.calls.append(kwargs)
            kwargs["agent"]._incremental_persistence_failed = True
            return ClaudeCliTurnResult(
                final_text="must not be reported as success",
                projected_messages=[
                    {
                        "role": "assistant",
                        "content": "must not be reported as success",
                    }
                ],
                native_session_id="native-session",
                model_iterations=1,
            )

    native = _PersistenceFailingSession()
    monkeypatch.setattr(
        "agent.claude_cli_runtime._get_session",
        lambda bound_agent, **_kwargs: native,
    )
    with patch.object(agent, "_sync_external_memory_for_turn", return_value=None):
        result = agent.run_conversation("change persistent state")

    try:
        assert result["turn_exit_reason"] == "session_persistence_failed"
        assert result["failed"] is True
        assert result["completed"] is False
        assert "session storage" in result["final_response"]
        assert result["error"] == result["final_response"]
        assert native.closed is True
    finally:
        agent.close()


def test_native_terminal_error_falls_through_to_standard_provider_same_turn(
    monkeypatch,
):
    agent = _make_agent()
    native = _SequenceSession(
        [
            ClaudeCliTurnResult(
                error="upstream unavailable",
                should_retire=True,
                native_session_id="native-session",
                model_iterations=1,
            )
        ]
    )
    fallback_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="fallback answer",
                    tool_calls=None,
                    reasoning=None,
                    reasoning_content=None,
                    reasoning_details=None,
                ),
                finish_reason="stop",
            )
        ],
        model="fallback/model",
        usage=None,
    )
    agent._fallback_chain = [{"provider": "openai", "model": "fallback/model"}]
    agent._fallback_index = 0

    def activate_fallback(reason=None):
        agent._fallback_index = 1
        agent._fallback_activated = True
        agent.api_mode = "chat_completions"
        agent.provider = "openai"
        agent.requested_provider = "openai"
        agent.model = "fallback/model"
        agent.base_url = "https://fallback.invalid/v1"
        agent.client = MagicMock()
        agent._transport_cache.clear()
        return True

    monkeypatch.setattr(
        "agent.claude_cli_runtime._get_session",
        lambda bound_agent, **_kwargs: native,
    )
    monkeypatch.setattr(agent, "_try_activate_fallback", activate_fallback)
    monkeypatch.setattr(
        agent,
        "_interruptible_api_call",
        lambda _kwargs: fallback_response,
    )
    with patch.object(agent, "_sync_external_memory_for_turn", return_value=None):
        result = agent.run_conversation("finish this")

    try:
        assert result["final_response"] == "fallback answer"
        assert result["api_calls"] == 2
        assert agent.provider == "openai"
        assert native.closed is True
        assert [row["content"] for row in result["messages"]] == [
            "finish this",
            "fallback answer",
        ]
    finally:
        agent.close()


def test_native_failure_before_provider_response_does_not_fabricate_usage(
    monkeypatch,
):
    agent = _make_agent()
    native = _SequenceSession(
        [
            ClaudeCliTurnResult(
                error="process exited before assistant response",
                should_retire=True,
                native_session_id="native-session",
            ),
            ClaudeCliTurnResult(
                error="process exited before assistant response",
                should_retire=True,
                native_session_id="native-session-2",
            ),
        ]
    )
    completed_usage = []
    session_api_calls_before = agent.session_api_calls
    agent._fallback_chain = []

    monkeypatch.setattr(
        "agent.claude_cli_runtime._get_session",
        lambda bound_agent, **_kwargs: native,
    )
    monkeypatch.setattr(agent, "_try_activate_fallback", lambda reason=None: False)
    monkeypatch.setattr(
        "agent.conversation_loop._notify_context_engine_turn_complete",
        lambda _agent, _messages, *, usage=None, **_kwargs: completed_usage.append(
            usage
        ),
    )
    with (
        patch.object(agent, "_sync_external_memory_for_turn", return_value=None),
        patch.object(agent, "_invoke_api_request_error_hook") as error_hook,
    ):
        result = agent.run_conversation("fail before response")

    try:
        assert result["failed"] is True
        assert result["api_calls"] == 2
        assert agent.session_api_calls == session_api_calls_before
        assert completed_usage == [None]
        assert error_hook.call_args.kwargs["retryable"] is True
        assert error_hook.call_args.kwargs["reason"] == "unknown"
        assert error_hook.call_count == 2
        assert error_hook.call_args.kwargs["retry_count"] == 1
        assert error_hook.call_args.kwargs["max_retries"] == 1
        assert native.closed is True
    finally:
        agent.close()


def test_native_transport_failure_restarts_once_before_fallback(monkeypatch):
    agent = _make_agent()
    first = _SequenceSession(
        [
            ClaudeCliTurnResult(
                error="local subprocess disconnected",
                should_retire=True,
                native_session_id="native-failed",
            )
        ]
    )
    second = _SequenceSession(
        [
            ClaudeCliTurnResult(
                final_text="answer after restart",
                projected_messages=[
                    {"role": "assistant", "content": "answer after restart"}
                ],
                native_session_id="native-recovered",
                model_iterations=1,
                token_usage={"input_tokens": 4, "output_tokens": 2},
                last_call_usage={"input_tokens": 4, "output_tokens": 2},
            )
        ]
    )
    sessions = iter([first, second])

    def get_session(bound_agent, **_kwargs):
        session = next(sessions)
        bound_agent._claude_cli_session = session
        return session

    monkeypatch.setattr("agent.claude_cli_runtime._get_session", get_session)
    with patch.object(agent, "_sync_external_memory_for_turn", return_value=None):
        result = agent.run_conversation("recover locally")

    try:
        assert result["final_response"] == "answer after restart"
        assert result["api_calls"] == 2
        assert first.closed is True
        assert len(first.calls) == 1
        assert len(second.calls) == 1
        assert [row["content"] for row in result["messages"]] == [
            "recover locally",
            "answer after restart",
        ]
    finally:
        agent.close()


def test_native_terminal_error_is_redacted_before_main_history(monkeypatch):
    agent = _make_agent()
    raw_error = (
        "failed at https://alice:super-secret@example.com/private"
    )
    native = _SequenceSession(
        [
            ClaudeCliTurnResult(
                error=raw_error,
                final_text=raw_error,
                projected_messages=[
                    {"role": "assistant", "content": raw_error}
                ],
                should_retire=True,
                native_session_id="native-failed",
                model_iterations=1,
            )
        ]
    )
    agent._fallback_chain = []

    monkeypatch.setattr(
        "agent.claude_cli_runtime._get_session",
        lambda bound_agent, **_kwargs: native,
    )
    monkeypatch.setattr(
        agent,
        "_try_activate_fallback",
        lambda reason=None: False,
    )
    with patch.object(agent, "_sync_external_memory_for_turn", return_value=None):
        result = agent.run_conversation("do not leak credentials")

    try:
        assert result["failed"] is True
        assert result["final_response"].startswith("Claude CLI turn failed:")
        assert "super-secret" not in result["final_response"]
        assert "super-secret" not in str(result["error"])
        assert all(
            "super-secret" not in str(row.get("content") or "")
            for row in result["messages"]
        )
        assert [row["role"] for row in result["messages"]] == [
            "user",
            "assistant",
        ]
    finally:
        agent.close()


def test_native_structured_billing_error_keeps_standard_recovery_metadata(
    monkeypatch,
):
    agent = _make_agent()
    native = _SequenceSession(
        [
            ClaudeCliTurnResult(
                error="Your account has no remaining usage",
                error_category="billing_error",
                error_status=402,
                should_retire=True,
                native_session_id="native-billing",
            )
        ]
    )
    agent._fallback_chain = []
    monkeypatch.setattr(
        "agent.claude_cli_runtime._get_session",
        lambda bound_agent, **_kwargs: native,
    )
    monkeypatch.setattr(agent, "_try_activate_fallback", lambda reason=None: False)

    with patch.object(agent, "_sync_external_memory_for_turn", return_value=None):
        result = agent.run_conversation("use subscription")

    try:
        assert result["failed"] is True
        assert result["failure_reason"] == "billing"
        assert result["billing_block"]["provider"] == "anthropic"
    finally:
        agent.close()


def test_native_terminal_context_error_reports_disabled_compaction(monkeypatch):
    agent = _make_agent()
    agent.compression_enabled = False
    native = _SequenceSession(
        [
            ClaudeCliTurnResult(
                error="prompt is too long for the model context window",
                terminal_result_received=True,
                should_retire=True,
                native_session_id="native-overflow",
            )
        ]
    )
    agent._fallback_chain = []
    monkeypatch.setattr(
        "agent.claude_cli_runtime._get_session",
        lambda bound_agent, **_kwargs: native,
    )
    monkeypatch.setattr(agent, "_try_activate_fallback", lambda reason=None: False)

    with patch.object(agent, "_sync_external_memory_for_turn", return_value=None):
        result = agent.run_conversation("too much history")

    try:
        assert result["failed"] is True
        assert result["failure_reason"] == "context_overflow"
        assert result["compaction_disabled"] is True
        assert not result.get("compression_exhausted")
        assert "auto-compaction is disabled" in result["final_response"]
        assert result["error"] == result["final_response"]
        assert result["messages"][-1]["content"] == result["final_response"]
    finally:
        agent.close()


def test_native_image_rejection_retries_on_clean_session_with_api_sidecar(
    monkeypatch,
):
    agent = _make_agent()
    rejected = _SequenceSession(
        [
            ClaudeCliTurnResult(
                error=(
                    "image dimensions exceed max allowed size: 8000 pixels"
                ),
                terminal_result_received=True,
                should_retire=True,
                native_session_id="native-image-rejected",
            )
        ]
    )
    recovered = _SequenceSession(
        [
            ClaudeCliTurnResult(
                final_text="image understood",
                projected_messages=[
                    {"role": "assistant", "content": "image understood"}
                ],
                model_iterations=1,
                native_session_id="native-image-recovered",
            )
        ]
    )
    sessions = iter([rejected, recovered])

    def get_session(bound_agent, **_kwargs):
        session = next(sessions)
        bound_agent._claude_cli_session = session
        return session

    def shrink(_agent, messages, **_kwargs):
        user_row = next(row for row in messages if row.get("role") == "user")
        user_row["api_content"] = [
            {"type": "text", "text": "inspect"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64,SMALL"},
            },
        ]
        return True

    monkeypatch.setattr("agent.claude_cli_runtime._get_session", get_session)
    monkeypatch.setattr(
        "agent.claude_cli_runtime._shrink_native_history_images", shrink
    )
    user_content = [
        {"type": "text", "text": "inspect"},
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,ORIGINAL"},
        },
    ]

    with patch.object(agent, "_sync_external_memory_for_turn", return_value=None):
        result = agent.run_conversation(user_content)

    try:
        assert result["final_response"] == "image understood"
        assert result["api_calls"] == 2
        assert rejected.closed is True
        assert recovered.calls[0]["user_input"][1]["image_url"]["url"].endswith(
            "SMALL"
        )
        user_row = next(
            row for row in result["messages"] if row.get("role") == "user"
        )
        assert user_row["content"][1]["image_url"]["url"].endswith("ORIGINAL")
        assert user_row["api_content"][1]["image_url"]["url"].endswith("SMALL")
    finally:
        agent.close()


def test_native_private_api_retry_surfaces_progress_and_error_observer(
    monkeypatch,
):
    agent = _make_agent()
    statuses = []

    class _RetryingSession(_FakeSession):
        def run_turn(self, **kwargs):
            self.calls.append(kwargs)
            kwargs["api_retry_callback"](
                {
                    "attempt": 1,
                    "max_retries": 2,
                    "retry_delay_ms": 2500,
                    "error_status": 429,
                    "error": "rate_limit",
                }
            )
            return ClaudeCliTurnResult(
                final_text="recovered",
                projected_messages=[
                    {"role": "assistant", "content": "recovered"}
                ],
                model_iterations=1,
                native_session_id="native-retried",
            )

    native = _RetryingSession()
    monkeypatch.setattr(
        "agent.claude_cli_runtime._get_session",
        lambda bound_agent, **_kwargs: native,
    )
    monkeypatch.setattr(agent, "_buffer_status", statuses.append)

    with (
        patch.object(agent, "_sync_external_memory_for_turn", return_value=None),
        patch.object(agent, "_invoke_api_request_error_hook") as error_hook,
    ):
        result = agent.run_conversation("recover from throttle")

    try:
        assert result["final_response"] == "recovered"
        assert statuses == [
            "⚠️ Claude API rate limit — retry 1/2 in 2.5s"
        ]
        assert error_hook.call_count == 1
        assert error_hook.call_args.kwargs["reason"] == "rate_limit"
        assert error_hook.call_args.kwargs["retry_count"] == 1
        assert error_hook.call_args.kwargs["max_retries"] == 2
    finally:
        agent.close()


def test_native_active_redirect_restarts_same_logical_turn_with_correction(monkeypatch):
    agent = _make_agent()

    class _RedirectingSession(_FakeSession):
        def run_turn(self, **kwargs):
            self.calls.append(kwargs)
            assert kwargs["agent"].redirect("use the corrected requirement") is True
            return ClaudeCliTurnResult(
                interrupted=True,
                should_retire=True,
                native_session_id="native-interrupted",
            )

    first = _RedirectingSession()
    second = _SequenceSession(
        [
            ClaudeCliTurnResult(
                final_text="corrected answer",
                projected_messages=[
                    {"role": "assistant", "content": "corrected answer"}
                ],
                native_session_id="native-corrected",
            )
        ]
    )
    sessions = iter([first, second])

    def get_session(bound_agent, **_kwargs):
        session = next(sessions)
        bound_agent._claude_cli_session = session
        return session

    monkeypatch.setattr("agent.claude_cli_runtime._get_session", get_session)
    with patch.object(agent, "_sync_external_memory_for_turn", return_value=None):
        result = agent.run_conversation("original requirement")

    try:
        assert result["final_response"] == "corrected answer"
        assert first.closed is True
        assert getattr(first, "interrupted", False) is True
        redirected_input = second.calls[0]["user_input"]
        assert "[Context from the interrupted assistant response]" in redirected_input
        assert redirected_input.endswith("use the corrected requirement")
        assert result["interrupted"] is False
        assert any(
            row.get("role") == "user"
            and row.get("content") == "use the corrected requirement"
            for row in result["messages"]
        )
    finally:
        agent.close()
