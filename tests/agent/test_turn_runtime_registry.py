"""Tests for the turn runtime seam (agent/turn_runtime_provider.py, agent/turn_runtime_registry.py).

A turn runtime owns a whole turn's model/tool loop; core call sites (turn dispatch, interrupt, steer,
compaction, teardown, runtime resolution) ask the registry instead of comparing ``api_mode`` strings.
"""
from __future__ import annotations

from typing import Any, Optional

import pytest

from agent import turn_runtime_registry as registry
from agent.turn_runtime_provider import TurnRuntimeProvider


class FakeRuntime(TurnRuntimeProvider):
    def __init__(self, name: str = "fake_runtime") -> None:
        self._name = name
        self.calls: list[tuple[str, Any]] = []
        self.turn_result: Optional[dict[str, Any]] = None
        self.steer_answer: Optional[bool] = None
        self.compact_answer: Any = None
        self.owns = True

    @property
    def name(self) -> str:
        return self._name

    def resolve_runtime(self, *, provider, api_mode, model_cfg):
        if provider == "anthropic" and (model_cfg or {}).get("anthropic_runtime") == self._name:
            return {"command": "fake-cli", "args": ["--x"], "api_mode": "ignored"}
        return None

    def init_client(self, agent):
        self.calls.append(("init_client", agent))
        return True

    def run_turn(self, agent, turn):
        self.calls.append(("run_turn", turn))
        if self.turn_result is not None:
            return self.turn_result
        return {"final_response": "from runtime", "messages": turn.messages, "api_calls": 1,
                "completed": True, "partial": False, "interrupted": False}

    def request_interrupt(self, agent):
        self.calls.append(("interrupt", agent))

    def request_steer(self, agent, text):
        self.calls.append(("steer", text))
        return self.steer_answer

    def release(self, agent, *, hard):
        self.calls.append(("release", hard))

    def owns_compaction(self, agent):
        return self.owns

    def compact(self, agent, messages, system_message, **kwargs):
        self.calls.append(("compact", kwargs))
        return self.compact_answer

    def names(self) -> list[str]:
        return [name for name, _ in self.calls]


@pytest.fixture
def fake_runtime(monkeypatch):
    registry._reset_for_tests()
    runtime = FakeRuntime()
    registry.register_provider(runtime)
    # A registered runtime must never need a discovery sweep.
    monkeypatch.setattr(registry, "_discover_once", lambda: None)
    yield runtime
    registry._reset_for_tests()


class TestLookup:
    def test_wire_modes_have_no_runtime_and_never_discover(self, monkeypatch):
        def boom():
            raise AssertionError("discovery must not run for a wire mode")
        monkeypatch.setattr(registry, "_discover_once", boom)
        for mode in ("chat_completions", "codex_responses", "anthropic_messages", "bedrock_converse", "", None):
            assert registry.turn_runtime_for(mode) is None

    def test_codex_app_server_is_builtin(self):
        from agent.codex_runtime import CodexAppServerTurnRuntime
        runtime = registry.turn_runtime_for("codex_app_server")
        assert isinstance(runtime, CodexAppServerTurnRuntime)
        assert registry.turn_runtime_for("CODEX_APP_SERVER") is runtime

    def test_plugin_cannot_replace_builtin(self, fake_runtime):
        registry.register_provider(FakeRuntime("codex_app_server"))
        assert registry.get_provider("codex_app_server") is None
        assert type(registry.turn_runtime_for("codex_app_server")).__name__ == "CodexAppServerTurnRuntime"

    def test_registered_runtime_found(self, fake_runtime):
        assert registry.turn_runtime_for("fake_runtime") is fake_runtime
        assert registry.is_turn_runtime_mode("fake_runtime")
        assert "fake_runtime" in registry.turn_runtime_names()
        assert fake_runtime in registry.all_turn_runtimes()

    def test_unknown_mode_runs_one_discovery_sweep(self, monkeypatch):
        registry._reset_for_tests()
        sweeps = []

        def discover():
            sweeps.append(1)
            registry.register_provider(FakeRuntime("late_runtime"))
        monkeypatch.setattr(registry, "_discover_once", discover)
        try:
            assert registry.turn_runtime_for("late_runtime").name == "late_runtime"
            assert sweeps == [1]
            assert registry.turn_runtime_for("late_runtime") is not None
            assert sweeps == [1]  # found without another sweep
        finally:
            registry._reset_for_tests()

    def test_overlay_forces_the_runtime_name(self, fake_runtime):
        overlay = registry.resolve_runtime_overlay(
            provider="anthropic", api_mode="anthropic_messages", model_cfg={"anthropic_runtime": "fake_runtime"})
        assert overlay == {"command": "fake-cli", "args": ["--x"], "api_mode": "fake_runtime"}
        assert registry.resolve_runtime_overlay(provider="openai", api_mode="", model_cfg={"x": 1}) is None
        assert registry.resolve_runtime_overlay(provider="anthropic", api_mode="", model_cfg=None) is None

    def test_call_runtime_hook_swallows_failures(self, fake_runtime):
        class Agent:
            api_mode = "fake_runtime"

        def broken(agent):
            raise RuntimeError("boom")
        fake_runtime.owns_compaction = broken
        assert registry.call_runtime_hook(Agent(), "owns_compaction", default="d") == "d"
        assert registry.call_runtime_hook(object(), "owns_compaction", default="none") == "none"


class TestRuntimeResolution:
    def test_plugin_runtime_resolved_before_credential_ladder(self, fake_runtime, monkeypatch):
        from hermes_cli import runtime_provider as rp

        def no_ladder(*a, **k):
            raise AssertionError("credential ladder must not run for a runtime-owned provider")
        monkeypatch.setattr(rp, "_ladder_rungs", no_ladder)
        monkeypatch.setattr(rp, "_get_model_config", lambda: {"provider": "anthropic",
                                                               "anthropic_runtime": "fake_runtime"})
        runtime = rp.resolve_runtime_provider(requested="anthropic")
        assert runtime["api_mode"] == "fake_runtime"
        assert runtime["provider"] == "anthropic"
        assert runtime["api_key"] == "" and runtime["base_url"] == ""
        assert runtime["command"] == "fake-cli" and runtime["args"] == ["--x"]

    def test_explicit_credentials_keep_the_http_route(self, fake_runtime, monkeypatch):
        from hermes_cli import runtime_provider as rp
        monkeypatch.setattr(rp, "_get_model_config", lambda: {"provider": "anthropic",
                                                               "anthropic_runtime": "fake_runtime"})
        sentinel = {"provider": "anthropic", "api_mode": "anthropic_messages", "api_key": "k", "base_url": "u"}
        monkeypatch.setattr(rp, "_ladder_rungs", lambda *a, **k: iter([dict(sentinel)]))
        monkeypatch.setattr("hermes_cli.runtime_provider_missing_key.raise_if_declared_key_missing",
                            lambda *a, **k: None)
        runtime = rp.resolve_runtime_provider(requested="anthropic", explicit_api_key="k")
        assert runtime["api_mode"] == "anthropic_messages"


def _make_agent(**kwargs):
    import run_agent
    return run_agent.AIAgent(
        api_key="stub", base_url="https://stub.invalid", provider="anthropic", api_mode="fake_runtime",
        quiet_mode=True, skip_context_files=True, skip_memory=True, **kwargs,
    )


class TestAgentIntegration:
    def test_runtime_owns_client_and_turn(self, fake_runtime):
        agent = _make_agent()
        assert agent.api_mode == "fake_runtime"
        assert agent.client is None
        assert "init_client" in fake_runtime.names()
        result = agent.run_conversation("hello")
        assert result["final_response"] == "from runtime"
        (turn,) = [arg for name, arg in fake_runtime.calls if name == "run_turn"]
        assert turn.continuation is False
        assert turn.messages[turn.current_turn_user_idx]["role"] == "user"

    def test_interrupt_reaches_runtime(self, fake_runtime):
        agent = _make_agent()
        agent.interrupt("stop")
        assert "interrupt" in fake_runtime.names()

    def test_redirect_falls_back_to_hermes_buffer_without_native_steer(self, fake_runtime):
        agent = _make_agent()
        fake_runtime.steer_answer = None
        agent.redirect("go left")
        assert ("steer", "go left") in fake_runtime.calls

    def test_native_steer_answer_is_final(self, fake_runtime):
        agent = _make_agent()
        fake_runtime.steer_answer = True
        assert agent.redirect("go right") is True
        fake_runtime.steer_answer = False
        assert agent.redirect("go right") is False

    def test_close_and_release_clients_release_the_runtime(self, fake_runtime):
        agent = _make_agent()
        agent.release_clients()
        agent.close()
        releases = [arg for name, arg in fake_runtime.calls if name == "release"]
        assert releases == [False, True]

    def test_runtime_returning_none_without_fallback_is_an_error(self, fake_runtime):
        from agent.conversation_loop import _turn_runtime_turn

        class Turn:
            api_call_count = 0
            continuation = False
            active_system_prompt = ""
        agent = _make_agent()
        fake_runtime.run_turn = lambda agent, turn: None
        with pytest.raises(RuntimeError, match="without activating a fallback"):
            _turn_runtime_turn(agent, Turn())


class TestCompactionRouting:
    def test_compress_context_routes_to_runtime(self, fake_runtime):
        agent = _make_agent()
        history = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
        fake_runtime.compact_answer = (history, "native prompt")
        out = agent._compress_context(history, "sys", force=True, focus_topic="keep x", trigger=None)
        assert out == (history, "native prompt")
        (kwargs,) = [arg for name, arg in fake_runtime.calls if name == "compact"]
        assert kwargs["force"] is True and kwargs["focus_topic"] == "keep x"

    def test_skipped_runtime_compaction_leaves_history(self, fake_runtime):
        agent = _make_agent()
        history = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
        fake_runtime.compact_answer = None
        messages, _prompt = agent._compress_context(history, "sys", force=False)
        assert messages is history

    def test_runtime_not_owning_compaction_uses_hermes_compressor(self, fake_runtime, monkeypatch):
        agent = _make_agent()
        fake_runtime.owns = False
        history = [{"role": "user", "content": "a"}]
        agent._compress_context(history, "sys", force=False)
        assert "compact" not in fake_runtime.names()
