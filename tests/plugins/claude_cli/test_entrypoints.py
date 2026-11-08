"""Runtime propagation guards for non-primary AIAgent construction sites."""

from __future__ import annotations

from types import SimpleNamespace


_CLAUDE_RUNTIME = {
    "provider": "anthropic",
    "requested_provider": "anthropic",
    "api_mode": "claude_cli",
    "api_key": "",
    "base_url": "",
    "command": "/opt/claude",
    "args": ["--debug-to-stderr"],
    "credential_pool": None,
}


class _FakeAgent:
    def __init__(self, captured: dict, **kwargs):
        object.__setattr__(self, "_captured", captured)
        captured.update(kwargs)
        self._session_messages = []
        self.context_compressor = SimpleNamespace(last_prompt_tokens=0)

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)
        if name == "_claude_cli_persistent_binding":
            self._captured[name] = value

    def run_conversation(self, *_args, **_kwargs):
        return {
            "final_response": "ok",
            "messages": [{"role": "assistant", "content": "ok"}],
            "completed": True,
            "api_calls": 1,
        }

    def _convert_to_trajectory_format(self, *_args, **_kwargs):
        return [{"from": "gpt", "value": "ok"}]

    def shutdown_memory_provider(self, *_args, **_kwargs):
        return None

    def close(self):
        return None


def test_oneshot_propagates_persistent_claude_command(monkeypatch):
    from hermes_cli import oneshot

    captured = {}
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {
            "model": {
                "default": "claude-opus-4-6",
                "provider": "anthropic",
            }
        },
    )
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **_kwargs: dict(_CLAUDE_RUNTIME),
    )
    monkeypatch.setattr(
        "hermes_cli.tools_config._get_platform_tools",
        lambda *_args, **_kwargs: set(),
    )
    monkeypatch.setattr(
        "hermes_cli.mcp_startup.ensure_mcp_discovery_before_agent_build",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(oneshot, "_create_session_db_for_oneshot", lambda: None)
    monkeypatch.setattr(
        "run_agent.AIAgent",
        lambda **kwargs: _FakeAgent(captured, **kwargs),
    )

    text, _result = oneshot._run_agent("hello")

    assert text == "ok"
    assert captured["api_mode"] == "claude_cli"
    assert captured["acp_command"] == "/opt/claude"
    assert captured["acp_args"] == ["--debug-to-stderr"]
