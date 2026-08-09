from __future__ import annotations

from unittest.mock import patch

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


def test_conversation_loop_uses_native_runtime_and_persists_projection(monkeypatch):
    agent = _make_agent()
    native = _FakeSession()

    def fake_get_session(bound_agent, *, system_prompt):
        assert bound_agent is agent
        assert isinstance(system_prompt, str) and system_prompt
        bound_agent._claude_cli_session = native
        return native

    monkeypatch.setattr("agent.claude_cli_runtime._get_session", fake_get_session)
    with patch.object(agent, "_sync_external_memory_for_turn", return_value=None):
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
    finally:
        agent.close()

