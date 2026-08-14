"""Runtime propagation guards for non-primary AIAgent construction sites."""

from __future__ import annotations

from types import SimpleNamespace


_CLAUDE_RUNTIME = {
    "provider": "anthropic",
    "requested_provider": "anthropic",
    "api_mode": "claude_agent_sdk",
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
        if name == "_claude_agent_sdk_persistent_binding":
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


def test_direct_runner_resolves_configured_persistent_claude_runtime(monkeypatch):
    import run_agent

    captured = {}
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **_kwargs: {**_CLAUDE_RUNTIME, "model": "claude-opus-4-6"},
    )
    monkeypatch.setattr(
        run_agent,
        "AIAgent",
        lambda **kwargs: _FakeAgent(captured, **kwargs),
    )

    run_agent.main(query="hello")

    assert captured["provider"] == "anthropic"
    assert captured["api_mode"] == "claude_agent_sdk"
    assert captured["api_key"] == ""
    assert captured["acp_command"] == "/opt/claude"
    assert captured["acp_args"] == ["--debug-to-stderr"]
    assert captured["_claude_agent_sdk_persistent_binding"] is False


def test_batch_worker_propagates_persistent_claude_runtime(monkeypatch):
    import batch_runner

    captured = {}
    monkeypatch.setattr(
        batch_runner,
        "AIAgent",
        lambda **kwargs: _FakeAgent(captured, **kwargs),
    )
    monkeypatch.setattr(
        batch_runner,
        "sample_toolsets_from_distribution",
        lambda _distribution: [],
    )
    config = {
        "distribution": "default",
        "model": "claude-opus-4-6",
        "max_iterations": 10,
        **_CLAUDE_RUNTIME,
    }

    result = batch_runner._process_single_prompt(
        0,
        {"prompt": "hello"},
        0,
        config,
    )

    assert result["success"] is True
    assert captured["provider"] == "anthropic"
    assert captured["api_mode"] == "claude_agent_sdk"
    assert captured["acp_command"] == "/opt/claude"
    assert captured["acp_args"] == ["--debug-to-stderr"]
    assert captured["_claude_agent_sdk_persistent_binding"] is False


def test_batch_worker_resolves_keychain_runtime_in_worker(monkeypatch, tmp_path):
    import batch_runner

    captured = {}
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **_kwargs: dict(_CLAUDE_RUNTIME),
    )
    monkeypatch.setattr(
        batch_runner,
        "_process_single_prompt",
        lambda _idx, _data, _batch, config: captured.update(config)
        or {
            "success": False,
            "trajectory": None,
            "tool_stats": {},
            "reasoning_stats": {},
        },
    )

    batch_runner._process_batch_worker(
        (
            0,
            [(0, {"prompt": "hello"})],
            tmp_path,
            set(),
            {
                "model": "claude-opus-4-6",
                "provider": None,
                "requested_provider": None,
                "api_mode": None,
                "api_key": None,
                "base_url": "https://openrouter.ai/api/v1",
                "resolve_runtime_in_worker": True,
                "verbose": False,
            },
        )
    )

    assert captured["provider"] == "anthropic"
    assert captured["api_mode"] == "claude_agent_sdk"
    assert captured["api_key"] == ""
    assert captured["base_url"] == ""


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
    assert captured["api_mode"] == "claude_agent_sdk"
    assert captured["acp_command"] == "/opt/claude"
    assert captured["acp_args"] == ["--debug-to-stderr"]
    assert captured["_claude_agent_sdk_persistent_binding"] is False


def test_feishu_comment_agent_propagates_persistent_claude_command(monkeypatch):
    from plugins.platforms.feishu import feishu_comment

    captured = {}
    monkeypatch.setattr(
        feishu_comment,
        "_resolve_model_and_runtime",
        lambda: ("claude-opus-4-6", dict(_CLAUDE_RUNTIME)),
    )
    monkeypatch.setattr(
        "run_agent.AIAgent",
        lambda **kwargs: _FakeAgent(captured, **kwargs),
    )

    assert (
        feishu_comment._run_comment_agent(
            "hello",
            object(),
            session_key="comment-doc:docx:file-token",
        )
        == "ok"
    )
    assert captured["requested_provider"] == "anthropic"
    assert captured["api_mode"] == "claude_agent_sdk"
    assert captured["acp_command"] == "/opt/claude"
    assert captured["acp_args"] == ["--debug-to-stderr"]
    assert captured["session_id"].startswith("feishu-comment-")
    assert "file-token" not in captured["session_id"]
