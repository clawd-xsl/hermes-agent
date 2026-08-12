from __future__ import annotations

from types import SimpleNamespace

from agent.conversation_compression import (
    claude_cli_owns_context_compaction,
    compress_context,
)
from agent.context_engine import ContextEngine


class _Compressor:
    def __init__(self) -> None:
        self.compression_count = 0
        self.last_compression_rough_tokens = 0
        self.last_prompt_tokens = 100
        self.last_completion_tokens = 10
        self.awaiting_real_usage_after_compression = False
        self.boundaries = []

    def on_session_start(self, session_id, **kwargs):
        self.boundaries.append((session_id, kwargs))


class _MemoryManager:
    def __init__(self) -> None:
        self.pre = []
        self.switches = []

    def on_pre_compress(self, messages):
        self.pre.append(messages)
        return "remember checkpoint alpha"

    def on_session_switch(self, session_id, **kwargs):
        self.switches.append((session_id, kwargs))


class _ExternalEngine(ContextEngine):
    """Configured plugin engine that must retain compaction ownership."""

    protect_first_n = 1
    protect_last_n = 1
    threshold_tokens = 1
    context_length = 200_000
    last_prompt_tokens = 0
    compression_count = 0

    def __init__(self) -> None:
        self.calls = []

    @property
    def name(self):
        return "external-test-engine"

    def update_from_response(self, usage):
        return None

    def should_compress(self, prompt_tokens=None):
        return True

    def compress(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return [{"role": "user", "content": "external engine summary"}]


class _Agent:
    def __init__(self) -> None:
        self.api_mode = "claude_cli"
        self.session_id = "hermes-session"
        self.platform = "cli"
        self._cached_system_prompt = "stable prompt"
        self.context_compressor = _Compressor()
        self._memory_manager = _MemoryManager()
        self._session_db = None
        self.statuses = []
        self.status_events = []
        self.warnings = []
        self.events = []

    def _emit_status(self, text):
        self.statuses.append(text)

    def _emit_warning(self, text):
        self.warnings.append(text)

    def status_callback(self, kind, text):
        self.status_events.append((kind, text))

    def event_callback(self, name, payload):
        self.events.append((name, payload))

    def _build_system_prompt(self, _system_message):
        return "rebuilt prompt"


def test_manual_compress_delegates_to_native_boundary_and_runs_hermes_lifecycle(
    monkeypatch,
):
    agent = _Agent()
    messages = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
    ]
    calls = []

    def compact(bound_agent, *, focus_topic=None):
        calls.append((bound_agent, focus_topic))
        return SimpleNamespace(
            compacted=True,
            interrupted=False,
            error=None,
            native_session_id="native-session",
            compaction_metadata={"trigger": "manual"},
            last_call_usage={
                "input_tokens": 11,
                "cache_read_input_tokens": 22,
                "cache_creation_input_tokens": 3,
            },
        )

    monkeypatch.setattr(
        "agent.claude_cli_runtime.compact_claude_cli_history", compact
    )
    monkeypatch.setattr(
        "tools.file_tools.reset_file_dedup", lambda _task_id: None
    )
    monkeypatch.setattr(
        "tools.skills_tool.reset_skill_view_dedup", lambda _task_id: None
    )

    returned, prompt = compress_context(
        agent,
        messages,
        "ignored",
        approx_tokens=120_000,
        task_id="manual",
        focus_topic="keep architecture decisions",
        force=True,
    )

    assert returned is messages
    assert prompt == "stable prompt"
    assert calls[0][0] is agent
    assert "keep architecture decisions" in calls[0][1]
    assert "remember checkpoint alpha" in calls[0][1]
    assert agent._memory_manager.pre == [messages]
    assert agent._memory_manager.switches == [
        (
            "hermes-session",
            {
                "parent_session_id": "hermes-session",
                "reset": False,
                "reason": "compression",
            },
        )
    ]
    assert agent.context_compressor.boundaries[0][0] == "hermes-session"
    assert agent.context_compressor.boundaries[0][1]["boundary_reason"] == "compression"
    assert agent.context_compressor.compression_count == 1
    assert agent.context_compressor.last_prompt_tokens == -1
    assert agent.context_compressor.awaiting_real_usage_after_compression is True
    assert agent.context_compressor._last_native_compaction is True
    assert agent._last_native_compaction is True
    assert agent._last_native_compaction_prompt_tokens == 36
    assert agent.events[0][0] == "session:compress"
    assert agent.events[0][1]["native"] is True
    assert agent.events[0][1]["native_session_id"] == "native-session"
    assert agent.status_events[-1][0] == "compacted"


def test_failed_native_compaction_is_an_explicit_abort(monkeypatch):
    agent = _Agent()
    messages = [{"role": "user", "content": "question"}]
    monkeypatch.setattr(
        "agent.claude_cli_runtime.compact_claude_cli_history",
        lambda *_args, **_kwargs: SimpleNamespace(
            compacted=False,
            interrupted=False,
            error="native compact failed",
        ),
    )

    returned, _ = compress_context(agent, messages, "", force=True)

    assert returned is messages
    assert agent.context_compressor._last_compress_aborted is True
    assert agent.context_compressor._last_summary_error == "native compact failed"
    assert agent.context_compressor._last_native_compaction is False
    assert agent.events == []
    assert "native compact failed" in agent.warnings[0]


def test_automatic_compression_remains_owned_by_claude(monkeypatch):
    agent = _Agent()
    messages = [{"role": "user", "content": "question"}]
    monkeypatch.setattr(
        "agent.claude_cli_runtime.compact_claude_cli_history",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("automatic path must not issue /compact")
        ),
    )

    returned, prompt = compress_context(agent, messages, "", force=False)

    assert returned is messages
    assert prompt == "stable prompt"
    assert agent.context_compressor.compression_count == 0


def test_idle_auto_compression_requests_native_boundary(monkeypatch):
    agent = _Agent()
    messages = [{"role": "user", "content": "question"}]
    calls = []
    monkeypatch.setattr(
        "agent.claude_cli_runtime.compact_claude_cli_history",
        lambda bound_agent, *, focus_topic=None: calls.append(
            (bound_agent, focus_topic)
        ) or SimpleNamespace(
            compacted=True,
            interrupted=False,
            error=None,
            native_session_id="native-idle",
            compaction_metadata={"trigger": "idle"},
        ),
    )
    monkeypatch.setattr(
        "tools.file_tools.reset_file_dedup", lambda _task_id: None
    )
    monkeypatch.setattr(
        "tools.skills_tool.reset_skill_view_dedup", lambda _task_id: None
    )

    returned, prompt = compress_context(
        agent,
        messages,
        "",
        force=False,
        native_trigger_source="idle",
    )

    assert returned is messages
    assert prompt == "stable prompt"
    assert calls and calls[0][0] is agent
    assert agent._last_native_compaction is True
    assert agent.context_compressor._compression_telemetry_seed[
        "trigger_source"
    ] == "idle"


def test_configured_external_engine_retains_compaction_ownership(monkeypatch):
    from run_agent import AIAgent

    agent = AIAgent(
        model="claude-opus-4-6",
        provider="anthropic",
        api_mode="claude_cli",
        session_id="external-engine-owner",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    engine = _ExternalEngine()
    agent.context_compressor = engine
    agent._context_engine_is_external = True
    agent._compression_feasibility_checked = True
    monkeypatch.setattr(
        "agent.claude_cli_runtime.compact_claude_cli_history",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("external engine must not be replaced by native /compact")
        ),
    )
    monkeypatch.setattr(
        "tools.file_tools.reset_file_dedup", lambda _task_id: None
    )
    monkeypatch.setattr(
        "tools.skills_tool.reset_skill_view_dedup", lambda _task_id: None
    )
    messages = [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"},
    ]

    try:
        returned, _prompt = compress_context(
            agent,
            messages,
            "system",
            approx_tokens=100_000,
            force=True,
        )
    finally:
        agent.close()

    assert engine.calls and engine.calls[0][0] == messages
    assert engine.calls[0][1]["force"] is True
    assert returned is not messages
    assert returned[0]["content"] == "external engine summary"


def test_explicit_external_engine_selection_wins_over_native_default():
    agent = SimpleNamespace(
        api_mode="claude_cli",
        context_compressor=_Compressor(),
        _context_engine_is_external=True,
    )

    assert claude_cli_owns_context_compaction(agent) is False
