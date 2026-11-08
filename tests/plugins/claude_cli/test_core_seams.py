"""The claude_cli runtime through core's turn runtime seams (resolution, construction, auxiliary
routing, fallback binding, keyless gates, background-review replay, binding persistence)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from plugins.claude_cli.provider import ClaudeCliTurnRuntime, wants_persistent_binding

_MODEL_CFG = {"provider": "anthropic", "default": "claude-sonnet-4-6", "anthropic_runtime": "claude_cli",
              "claude_cli": {"command": "/opt/claude", "args": ["--verbose"]}}


@pytest.fixture
def claude_config(monkeypatch):
    from hermes_cli import runtime_provider as rp
    monkeypatch.setattr(rp, "_get_model_config", lambda: dict(_MODEL_CFG))
    return rp


def test_runtime_resolution_skips_the_credential_ladder(claude_config, monkeypatch):
    monkeypatch.setattr(claude_config, "_ladder_rungs",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("ladder must not run")))
    runtime = claude_config.resolve_runtime_provider(requested="anthropic")
    assert runtime["api_mode"] == "claude_cli"
    assert runtime["provider"] == "anthropic"
    assert runtime["api_key"] == "" and runtime["base_url"] == ""
    assert runtime["command"] == "/opt/claude" and runtime["args"] == ["--verbose"]


def test_unconfigured_command_defaults_to_the_claude_executable():
    overlay = ClaudeCliTurnRuntime().resolve_runtime(
        provider="anthropic", api_mode="", model_cfg={"anthropic_runtime": "claude_cli"})
    assert overlay["command"] == "claude"


def test_other_providers_and_runtimes_are_not_claimed():
    runtime = ClaudeCliTurnRuntime()
    assert runtime.resolve_runtime(provider="openai", api_mode="", model_cfg=_MODEL_CFG) is None
    assert runtime.resolve_runtime(provider="anthropic", api_mode="", model_cfg={"anthropic_runtime": "auto"}) is None


def _claude_agent(**kwargs):
    from run_agent import AIAgent
    return AIAgent(model="claude-sonnet-4-6", provider="anthropic", api_mode="claude_cli", api_key="",
                   base_url="", acp_command="/opt/claude", acp_args=["--verbose"], quiet_mode=True,
                   skip_context_files=True, skip_memory=True, **kwargs)


def test_agent_construction_builds_no_http_client():
    agent = _claude_agent()
    try:
        assert agent.api_mode == "claude_cli"
        assert agent.client is None and agent.api_key == "" and agent._client_kwargs == {}
        assert agent._claude_cli_command == "/opt/claude"
        assert agent._claude_cli_args == ["--verbose"]
    finally:
        agent.close()


def test_auxiliary_calls_on_the_main_provider_use_isolated_claude_children():
    from agent.auxiliary_client import resolve_provider_client
    from plugins.claude_cli.auxiliary import AsyncClaudeCliAuxiliaryClient, ClaudeCliAuxiliaryClient
    main = {"provider": "anthropic", "api_mode": "claude_cli", "model": "claude-sonnet-4-6"}
    client, model = resolve_provider_client("anthropic", "claude-haiku-4-5", main_runtime=main)
    assert isinstance(client, ClaudeCliAuxiliaryClient) and model == "claude-haiku-4-5"
    client, _ = resolve_provider_client("anthropic", "claude-haiku-4-5", main_runtime=main, async_mode=True)
    assert isinstance(client, AsyncClaudeCliAuxiliaryClient)
    # An explicit HTTP endpoint/key for the auxiliary slot stays authoritative.
    client, _ = resolve_provider_client("anthropic", "claude-haiku-4-5", main_runtime=main,
                                        explicit_api_key="sk-ant-test", explicit_base_url="https://api.anthropic.com")
    assert not isinstance(client, ClaudeCliAuxiliaryClient)


def test_auxiliary_adapter_maps_native_result(monkeypatch):
    from plugins.claude_cli.auxiliary import ClaudeCliAuxiliaryClient
    seen = {}

    def fake(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(final_text="title", projected_messages=[], last_stop_reason="end_turn",
                               token_usage={"input_tokens": 3, "cache_read_input_tokens": 4, "output_tokens": 2})
    monkeypatch.setattr("plugins.claude_cli.runtime.run_claude_cli_auxiliary_completion", fake)
    response = ClaudeCliAuxiliaryClient("claude-haiku-4-5").chat.completions.create(
        messages=[{"role": "user", "content": "hi"}], max_tokens=20,
        response_format={"type": "json_schema", "json_schema": {"schema": {"type": "object"}}})
    assert response.choices[0].message.content == "title"
    assert response.choices[0].finish_reason == "stop"
    assert response.usage.prompt_tokens == 7 and response.usage.completion_tokens == 2
    assert seen["json_schema"] == {"type": "object"} and seen["max_tokens"] == 20


def test_keyless_runtime_routes_pass_the_api_key_gates():
    from agent.turn_runtime_registry import route_authenticates_itself
    assert route_authenticates_itself({"api_mode": "claude_cli", "api_key": ""})
    assert not route_authenticates_itself({"api_mode": "anthropic_messages", "api_key": ""})
    assert not route_authenticates_itself(None)


def test_fallback_entry_with_explicit_claude_api_mode_binds_the_runtime():
    from agent.route_binding import bind_route_entry
    from run_agent import AIAgent
    agent = AIAgent(model="gpt-5.4", provider="openrouter", api_key="sk-or-test",
                    base_url="https://openrouter.ai/api/v1", quiet_mode=True, skip_context_files=True,
                    skip_memory=True)
    try:
        bound = bind_route_entry(
            agent, {"provider": "anthropic", "model": "claude-sonnet-4-6", "api_mode": "claude_cli"},
            "anthropic", "claude-sonnet-4-6")
        assert bound is not None
        assert agent.api_mode == "claude_cli" and agent.provider == "anthropic"
        assert agent.client is None and agent.api_key == "" and agent._credential_pool is None
        assert agent._fallback_activated is True
    finally:
        agent.close()


def test_fallback_identity_keeps_http_and_runtime_routes_distinct():
    from agent.turn_runtime_binding import runtime_identity_base_url
    assert runtime_identity_base_url("claude_cli", "") == "claude_cli://local"
    assert runtime_identity_base_url("anthropic_messages", "https://api.anthropic.com") == "https://api.anthropic.com"


def test_background_review_replay_is_bounded_for_runtime_forks():
    from agent.background_review import _review_replay_history
    fork = SimpleNamespace(api_mode="claude_cli")
    history = []
    for i in range(40):
        history.append({"role": "user", "content": f"question {i}"})
        history.append({"role": "assistant", "content": "x" * 20_000})
    replay = _review_replay_history(fork, history, routed=False)
    assert replay[0]["content"].startswith("[Earlier conversation digest")
    assert len(replay) < len(history)
    assert all(len(str(row.get("content") or "")) <= 24_000 for row in replay)
    assert all(len(str(row.get("content") or "")) <= 4_000 for row in replay[1:])
    http_fork = SimpleNamespace(api_mode="anthropic_messages")
    assert _review_replay_history(http_fork, history, routed=False) is history


@pytest.mark.parametrize("attrs,expected", [
    ({}, True),
    ({"_persist_disabled": True}, False),
    ({"_delegate_depth": 1}, False),
    ({"platform": "cron"}, False),
    ({"platform": "subagent"}, False),
    ({"platform": "cron", "_claude_cli_persistent_binding": True}, True),
    ({"_claude_cli_persistent_binding": False}, False),
])
def test_persistent_binding_policy(attrs, expected, monkeypatch):
    monkeypatch.setattr("agent.oneshot_footprint.agent_is_oneshot", lambda _agent: False)
    assert wants_persistent_binding(SimpleNamespace(**attrs)) is expected


def test_oneshot_sessions_do_not_persist_bindings(monkeypatch):
    monkeypatch.setattr("agent.oneshot_footprint.agent_is_oneshot", lambda _agent: True)
    assert wants_persistent_binding(SimpleNamespace()) is False


def test_release_detaches_borrowed_sessions_and_closes_owned_ones(monkeypatch):
    calls = []
    monkeypatch.setattr("plugins.claude_cli.runtime.release_claude_cli_session", lambda a: calls.append(("release", a)))
    monkeypatch.setattr("plugins.claude_cli.runtime.detach_claude_cli_session", lambda a: calls.append(("detach", a)))
    runtime = ClaudeCliTurnRuntime()
    owner, helper = SimpleNamespace(), SimpleNamespace(_borrows_session_runtime=True)
    runtime.release(owner, hard=False)
    assert calls == []  # cache eviction keeps the child pooled
    runtime.release(owner, hard=True)
    runtime.release(helper, hard=True)
    assert calls == [("release", owner), ("detach", helper)]
