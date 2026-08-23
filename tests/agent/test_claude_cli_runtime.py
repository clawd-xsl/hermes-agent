from __future__ import annotations

from types import SimpleNamespace

from agent.claude_cli_runtime import (
    _attach_native_moa_context,
    _dedupe_final_projection,
    _native_housekeeping_fallback,
    _native_fast_mode,
    _native_display_error,
    _native_turn_signature,
    _prepend_native_compaction_context,
    _native_request_messages,
    _native_system_prompt_with_prefill,
    _native_moa_guidance,
    _native_json_schema,
    _native_timeouts,
    _native_unsupported_request_controls,
    _get_session,
    _owner_key,
    _reasoning_effort,
    _record_native_auto_compaction,
    _record_usage,
    _shrink_native_history_images,
    _thinking_mode,
    compact_claude_cli_history,
    release_claude_cli_session,
    run_claude_cli_auxiliary_completion,
    stamp_native_compaction_context,
)
from agent.transports.claude_cli_session import ClaudeCliTurnResult


class _Compressor:
    def __init__(self):
        self.usage = None

    def update_from_response(self, usage):
        self.usage = usage


class _SessionDB:
    def __init__(self):
        self.calls = []

    def update_token_counts(self, *args, **kwargs):
        self.calls.append((args, kwargs))


def test_usage_accounts_aggregate_turn_but_compressor_tracks_last_native_call():
    db = _SessionDB()
    agent = SimpleNamespace(
        session_api_calls=0,
        session_prompt_tokens=0,
        session_completion_tokens=0,
        session_total_tokens=0,
        session_input_tokens=0,
        session_output_tokens=0,
        session_cache_read_tokens=0,
        session_cache_write_tokens=0,
        session_estimated_cost_usd=0.0,
        context_compressor=_Compressor(),
        _session_db=db,
        _session_db_created=True,
        session_id="session-1",
        model="claude-opus-4-6",
    )

    result = _record_usage(
        agent,
        {
            "input_tokens": 100,
            "cache_read_input_tokens": 200,
            "cache_creation_input_tokens": 10,
            "output_tokens": 30,
        },
        {
            "input_tokens": 5,
            "cache_read_input_tokens": 95,
            "output_tokens": 12,
        },
    )

    assert result["prompt_tokens"] == 310
    assert result["total_tokens"] == 340
    assert result["last_prompt_tokens"] == 100
    assert result["last_usage"] == {
        "prompt_tokens": 100,
        "completion_tokens": 12,
        "total_tokens": 112,
        "input_tokens": 5,
        "output_tokens": 12,
        "cache_read_tokens": 95,
        "cache_write_tokens": 0,
        "reasoning_tokens": 0,
    }
    assert agent.session_total_tokens == 340
    assert agent.context_compressor.usage == {
        "prompt_tokens": 100,
        "completion_tokens": 12,
        "total_tokens": 112,
        "input_tokens": 5,
        "output_tokens": 12,
        "cache_read_tokens": 95,
        "cache_write_tokens": 0,
        "reasoning_tokens": 0,
    }
    assert db.calls[0][1]["billing_mode"] == "subscription_included"
    assert db.calls[0][1]["api_call_count"] == 1


def test_final_projection_is_not_duplicated():
    projected = [{"role": "assistant", "content": "done"}]
    assert _dedupe_final_projection(projected, "done") is projected
    assert _dedupe_final_projection([], "done") == [
        {"role": "assistant", "content": "done"}
    ]


def test_native_turn_signature_tracks_host_control_and_history_state():
    base = ClaudeCliTurnResult(
        final_text="done",
        projected_messages=[{"role": "assistant", "content": "done"}],
        native_session_id="native-1",
    )
    for field, value in (
        ("host_stop_reason", "guardrail_halt"),
        ("should_retire", True),
        ("budget_exhausted", True),
        ("native_session_id", "native-2"),
        ("compacted", True),
        ("compaction_count", 1),
        ("error_category", "rate_limit"),
        ("error_status", 429),
        ("terminal_result_received", True),
    ):
        changed = ClaudeCliTurnResult(**vars(base))
        setattr(changed, field, value)
        assert _native_turn_signature(changed) != _native_turn_signature(base)


def test_native_auto_compaction_closes_status_lifecycle_and_records_progress(
    monkeypatch,
):
    class _NativeCompressor:
        compression_count = 0
        _last_compression_made_progress = False

        def record_completed_compaction(self, **_kwargs):
            self._verify_compaction_cleared_threshold = True

    statuses = []
    status_events = []
    compressor = _NativeCompressor()
    agent = SimpleNamespace(
        session_id="hermes-session",
        platform="cli",
        context_compressor=compressor,
        _memory_manager=None,
        event_callback=None,
        status_callback=lambda kind, text: status_events.append((kind, text)),
        _emit_status=lambda text: statuses.append(text),
    )
    monkeypatch.setattr(
        "agent.conversation_compression._notify_context_engine_compression_complete",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr("tools.file_tools.reset_file_dedup", lambda _task_id: None)
    monkeypatch.setattr(
        "tools.skills_tool.reset_skill_view_dedup", lambda _task_id: None
    )

    recorded = _record_native_auto_compaction(
        agent,
        SimpleNamespace(
            compacted=True,
            compaction_count=1,
            compaction_metadata={"preTokens": 120_000},
            native_session_id="native-session",
        ),
        messages=[{"role": "user", "content": "hello"}],
        task_id="default",
    )

    assert recorded is True
    assert statuses and "Compacting context" in statuses[0]
    assert status_events[-1][0] == "compacted"
    assert compressor.compression_count == 1
    assert compressor._last_compression_made_progress is True
    assert compressor.last_compression_rough_tokens == 120_000


def test_native_display_error_is_bounded_and_force_redacted():
    agent = SimpleNamespace(
        _summarize_api_error=lambda error: str(error),
    )
    rendered = _native_display_error(
        agent,
        "failed at https://alice:super-secret@example.com/private",
    )

    assert rendered.startswith("Claude CLI turn failed:")
    assert "super-secret" not in rendered
    assert "example.com" in rendered


def test_native_housekeeping_fallback_only_reuses_completed_answer_text():
    agent = SimpleNamespace(_strip_think_blocks=lambda text: text)
    messages = [
        {"role": "user", "content": "remember this"},
        {
            "role": "assistant",
            "content": "Done — I will remember it.",
            "tool_calls": [
                {
                    "id": "memory-1",
                    "function": {"name": "memory", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "memory-1", "content": "saved"},
    ]
    assert _native_housekeeping_fallback(
        agent, messages, current_turn_user_idx=0
    ) == "Done — I will remember it."

    messages[1]["tool_calls"][0]["function"]["name"] = "terminal"
    assert (
        _native_housekeeping_fallback(
            agent, messages, current_turn_user_idx=0
        )
        is None
    )


def test_persistence_disabled_fork_cannot_share_foreground_native_history():
    foreground = SimpleNamespace(session_id="main", _persist_disabled=False)
    review = SimpleNamespace(session_id="main", _persist_disabled=True)
    assert _owner_key(foreground) == "main"
    assert _owner_key(review).startswith("ephemeral:")
    assert _owner_key(review) != _owner_key(foreground)


def test_one_shot_agent_uses_nonpersistent_native_history(monkeypatch):
    captured = {}
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
    }

    class Session:
        is_busy = False
        last_used_at = 0.0

        def __init__(self, **kwargs):
            captured.update(kwargs)

        def close(self):
            captured["closed"] = True

    monkeypatch.setattr(
        "agent.claude_cli_runtime.ClaudeCliSession",
        Session,
    )
    monkeypatch.setattr("agent.claude_cli_runtime._runtime_config", lambda: {})
    agent = SimpleNamespace(
        session_id="one-shot-native-binding-test",
        _gateway_session_key="",
        tools=[],
        session_cwd="/tmp",
        model="claude-opus-4-6",
        reasoning_config=None,
        _persist_disabled=False,
        _claude_cli_persistent_binding=False,
        _claude_cli_command="claude",
        _claude_cli_args=[],
        service_tier=None,
        request_overrides={
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "answer", "schema": schema},
            }
        },
        compression_enabled=False,
    )

    session = _get_session(agent, system_prompt="stable")
    release_claude_cli_session(agent)

    assert captured["persistent_binding"] is False
    assert captured["auto_compaction_enabled"] is False
    assert captured["json_schema"] == schema
    assert session is not None
    assert captured["closed"] is True


def test_native_provider_timeout_reaches_child_and_expands_implicit_turn_cap(
    monkeypatch,
):
    monkeypatch.setattr(
        "hermes_cli.timeouts.get_provider_request_timeout",
        lambda provider, model: 900.0
        if (provider, model) == ("anthropic", "claude-opus-4-6")
        else None,
    )
    agent = SimpleNamespace(provider="anthropic", model="claude-opus-4-6")

    turn_timeout, request_timeout = _native_timeouts(agent, {})

    assert request_timeout == 900.0
    assert turn_timeout == 905.0


def test_explicit_native_turn_cap_remains_independent_of_provider_timeout(
    monkeypatch,
):
    monkeypatch.setattr(
        "hermes_cli.timeouts.get_provider_request_timeout",
        lambda _provider, _model: 900.0,
    )
    agent = SimpleNamespace(provider="anthropic", model="claude-opus-4-6")

    turn_timeout, request_timeout = _native_timeouts(
        agent,
        {"turn_timeout_seconds": 120.0},
    )

    assert request_timeout == 900.0
    assert turn_timeout == 120.0


def test_native_prefill_examples_live_in_stable_system_prefix_not_request_rows():
    agent = SimpleNamespace(
        prefill_messages=[
            {"role": "user", "content": "answer in the demonstrated style"},
            {"role": "assistant", "content": "demonstrated answer"},
        ]
    )

    system_prompt = _native_system_prompt_with_prefill(agent, "stable identity")
    request = _native_request_messages(
        agent,
        [{"role": "user", "content": "live question"}],
        system_prompt=system_prompt,
    )

    assert system_prompt.startswith("stable identity\n\n<hermes_prefill_examples>")
    assert "demonstrated answer" in system_prompt
    assert [row["role"] for row in request] == ["system", "user"]
    assert request[-1]["content"] == "live question"
    assert _native_system_prompt_with_prefill(agent, system_prompt) == system_prompt


def test_reasoning_effort_maps_hermes_edges_to_claude_cli_levels():
    assert _reasoning_effort(SimpleNamespace(reasoning_config={"effort": "minimal"})) == "low"
    assert _reasoning_effort(SimpleNamespace(reasoning_config={"effort": "ultra"})) == "max"
    assert _reasoning_effort(SimpleNamespace(reasoning_config={"effort": "xhigh"})) == "xhigh"
    assert _reasoning_effort(SimpleNamespace(reasoning_config={"effort": "none"})) is None


def test_disabled_reasoning_is_forwarded_as_native_thinking_override():
    assert _thinking_mode(SimpleNamespace(reasoning_config={"enabled": False})) == "disabled"
    assert _thinking_mode(SimpleNamespace(reasoning_config={"effort": "none"})) == "disabled"
    assert _thinking_mode(SimpleNamespace(reasoning_config={"effort": "low"})) is None
    assert _thinking_mode(SimpleNamespace(reasoning_config=None)) is None


def test_native_request_view_substitutes_multimodal_api_content_sidecar():
    api_content = [
        {"type": "text", "text": "wire context"},
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,AA=="},
        },
    ]
    rows = _native_request_messages(
        SimpleNamespace(prefill_messages=[]),
        [
            {
                "role": "user",
                "content": "clean transcript",
                "api_content": api_content,
            }
        ],
        system_prompt="system",
    )
    assert rows[-1]["content"] == api_content
    assert "api_content" not in rows[-1]


def test_native_compaction_continuity_preserves_multimodal_user_input():
    image = {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,AA=="},
    }
    content = _prepend_native_compaction_context([image], "remember checkpoint")
    assert content[0]["type"] == "text"
    assert "remember checkpoint" in content[0]["text"]
    assert content[1] is image


def test_native_compaction_continuity_stamps_wire_sidecar_once():
    message = {
        "role": "user",
        "content": "clean question",
        "api_content": "question with recalled memory",
    }

    first = stamp_native_compaction_context(message, "remember checkpoint")
    second = stamp_native_compaction_context(message, "remember checkpoint")

    assert first == second == message["api_content"]
    assert first.count("remember checkpoint") == 1
    assert first.endswith("question with recalled memory")
    assert message["content"] == "clean question"


def test_native_moa_guidance_is_attached_to_current_multimodal_user(monkeypatch):
    seen = {}

    def aggregate(**kwargs):
        seen.update(kwargs)
        return "PRIVATE ADVISOR GUIDANCE"

    monkeypatch.setattr("agent.moa_loop.aggregate_moa_context", aggregate)
    rows = _attach_native_moa_context(
        SimpleNamespace(),
        [
            {"role": "system", "content": "system"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "question"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AA=="},
                    },
                ],
            },
        ],
        original_user_message=[{"type": "text", "text": "question"}],
        moa_config={
            "reference_models": [{"provider": "openai", "model": "advisor"}],
            "aggregator": {"provider": "anthropic", "model": "claude"},
        },
    )

    assert seen["user_prompt"] == "question"
    assert rows[-1]["content"][-1] == {
        "type": "text",
        "text": "\n\nPRIVATE ADVISOR GUIDANCE",
    }


def test_native_moa_guidance_helper_recomputes_each_call(monkeypatch):
    calls = []

    def aggregate(**kwargs):
        calls.append(kwargs["api_messages"])
        return f"guidance-{len(calls)}"

    monkeypatch.setattr("agent.moa_loop.aggregate_moa_context", aggregate)
    config = {
        "reference_models": [{"provider": "openai", "model": "advisor"}],
        "aggregator": {"provider": "anthropic", "model": "claude"},
    }

    first = _native_moa_guidance(
        SimpleNamespace(),
        [{"role": "user", "content": "step one"}],
        original_user_message="task",
        moa_config=config,
    )
    second = _native_moa_guidance(
        SimpleNamespace(),
        [{"role": "tool", "content": "step two"}],
        original_user_message="task",
        moa_config=config,
    )

    assert (first, second) == ("guidance-1", "guidance-2")
    assert calls[1][-1]["content"] == "step two"


def test_native_request_controls_are_never_silently_claimed_as_applied():
    statuses = []
    agent = SimpleNamespace(
        max_tokens=4096,
        service_tier="priority",
        request_overrides={
            "speed": "fast",
            "temperature": 0.2,
            "extra_body": {"foo": "bar"},
        },
        _emit_status=statuses.append,
    )
    controls = _native_unsupported_request_controls(agent)
    assert controls == [
        "request_overrides.extra_body.foo",
        "request_overrides.temperature",
    ]
    assert "cannot forward" in statuses[0]


def test_native_structured_output_is_supported_not_reported_as_ignored():
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }
    agent = SimpleNamespace(
        max_tokens=4096,
        service_tier=None,
        request_overrides={
            "extra_body": {
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "answer", "schema": schema},
                }
            }
        },
        _emit_status=lambda _message: None,
    )

    assert _native_json_schema(agent) == schema
    assert _native_unsupported_request_controls(agent) == []


def test_native_json_object_maps_to_generic_object_schema():
    agent = SimpleNamespace(
        request_overrides={"response_format": {"type": "json_object"}}
    )
    assert _native_json_schema(agent) == {"type": "object"}


def test_native_image_shrink_uses_api_content_without_rewriting_visible_image(
    monkeypatch,
):
    original = [
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,ORIGINAL"},
        }
    ]
    messages = [{"role": "user", "content": original}]

    def shrink(rows, **_kwargs):
        rows[0]["content"] = [
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64,SMALL"},
            }
        ]
        return True

    monkeypatch.setattr(
        "agent.conversation_compression.try_shrink_image_parts_in_messages",
        shrink,
    )
    monkeypatch.setattr(
        "agent.conversation_loop._image_error_max_dimension",
        lambda _error: 2000,
    )

    assert _shrink_native_history_images(
        SimpleNamespace(),
        messages,
        error="image dimensions exceed max allowed size: 2000 pixels",
    )
    assert messages[0]["content"] is original
    assert messages[0]["content"][0]["image_url"]["url"].endswith("ORIGINAL")
    assert messages[0]["api_content"][0]["image_url"]["url"].endswith("SMALL")


def test_hermes_fast_state_maps_to_native_claude_fast_mode():
    assert _native_fast_mode(
        SimpleNamespace(service_tier="priority", request_overrides={}),
        {},
    )
    assert _native_fast_mode(
        SimpleNamespace(
            service_tier=None,
            request_overrides={"speed": "fast"},
        ),
        {},
    )
    assert _native_fast_mode(
        SimpleNamespace(service_tier=None, request_overrides={}),
        {"fast_mode": True},
    )
    assert not _native_fast_mode(
        SimpleNamespace(service_tier=None, request_overrides={}),
        {},
    )


def test_auxiliary_completion_is_isolated_toolless_and_closes(monkeypatch):
    created = []

    class Session:
        turn_timeout = 600.0

        def summarize(self, *, agent, messages, prompt):
            assert agent.tools == []
            assert agent.iteration_budget is None
            assert messages == [{"role": "assistant", "content": "prior"}]
            assert prompt == [
                {"type": "text", "text": "inspect"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AA=="},
                },
            ]
            return SimpleNamespace(
                final_text="result",
                error=None,
                interrupted=False,
            )

        def close(self):
            created.append("closed")

    session = Session()

    def _new(agent, **kwargs):
        created.append((agent, kwargs))
        return session

    monkeypatch.setattr("agent.claude_cli_runtime._new_transient_session", _new)

    result = run_claude_cli_auxiliary_completion(
        messages=[
            {"role": "system", "content": "system one"},
            {"role": "system", "content": "system two"},
            {"role": "assistant", "content": "prior"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "inspect"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AA=="},
                    },
                ],
            },
        ],
        model="anthropic/claude-opus-4-6",
        timeout=12.5,
        max_tokens=2048,
        reasoning_config={"effort": "high"},
    )

    assert result.final_text == "result"
    assert session.turn_timeout == 12.5
    assert created[0][1]["system_prompt"] == "system one\n\nsystem two"
    assert created[0][1]["model"] == "anthropic/claude-opus-4-6"
    assert created[0][1]["tool_definitions"] == []
    assert created[0][0].max_tokens == 2048
    assert created[-1] == "closed"


def test_auxiliary_completion_rejects_non_user_terminal_message():
    try:
        run_claude_cli_auxiliary_completion(
            messages=[{"role": "assistant", "content": "orphan"}],
            model="claude-opus-4-6",
        )
    except ValueError as exc:
        assert "final user message" in str(exc)
    else:  # pragma: no cover - makes the contract failure explicit
        raise AssertionError("expected ValueError")


def test_auxiliary_completion_captures_tool_proposal_without_executing(
    monkeypatch,
):
    created = []
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Look something up",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    class Session:
        turn_timeout = 600.0

        def propose_tools(self, *, agent, messages, prompt):
            assert agent.tools == tools
            assert messages == []
            assert prompt == "find it"
            return SimpleNamespace(
                final_text="",
                error=None,
                interrupted=False,
                projected_messages=[
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "tool_1",
                                "type": "function",
                                "function": {
                                    "name": "lookup",
                                    "arguments": "{}",
                                },
                            }
                        ],
                    }
                ],
            )

        def close(self):
            created.append("closed")

    def _new(agent, **kwargs):
        created.append((agent, kwargs))
        return Session()

    monkeypatch.setattr("agent.claude_cli_runtime._new_transient_session", _new)

    result = run_claude_cli_auxiliary_completion(
        messages=[{"role": "user", "content": "find it"}],
        model="claude-opus-4-6",
        tools=tools,
        tool_choice="required",
    )

    assert result.projected_messages[0]["tool_calls"][0]["id"] == "tool_1"
    assert created[0][1]["tool_definitions"] == tools
    assert "must respond by proposing at least one" in created[0][1][
        "system_prompt"
    ]
    assert created[-1] == "closed"


def test_manual_compaction_reuses_effective_persistent_session_prompt(monkeypatch):
    calls = []

    class Session:
        def compact(self, *, agent, focus_topic=None):
            calls.append((agent, focus_topic))
            return SimpleNamespace(
                compacted=True,
                should_retire=False,
                token_usage={},
            )

    session = Session()
    prompts = []
    monkeypatch.setattr(
        "agent.claude_cli_runtime._get_session",
        lambda agent, *, system_prompt: prompts.append(system_prompt) or session,
    )
    agent = SimpleNamespace(
        _cached_system_prompt="stable",
        ephemeral_system_prompt="turn-scoped",
    )

    result = compact_claude_cli_history(agent, focus_topic="keep decisions")

    assert result.compacted is True
    assert prompts == ["stable\n\nturn-scoped"]
    assert calls == [(agent, "keep decisions")]


class TestReleaseSessionsByOwner:
    """One-shot runs must hand their pooled child back when they finish.

    Webhook deliveries and cron fires end in the gateway/scheduler, which hold
    a session id but no AIAgent. Before this path existed their child stayed
    pooled until LRU pressure evicted it — hours of ~265MB apiece, and a
    delivery burst could push a live chat session out of the pool.
    """

    class _FakeSession:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    def _pool(self, monkeypatch, **entries):
        from agent import claude_cli_runtime as rt

        monkeypatch.setattr(rt, "_SESSIONS", dict(entries), raising=False)
        return rt

    def test_closes_and_unpools_the_named_owner(self, monkeypatch):
        victim, keep = self._FakeSession(), self._FakeSession()
        rt = self._pool(monkeypatch, webhook_1=victim, signal_main=keep)

        assert rt.release_claude_cli_sessions_by_owner("webhook_1") == 1
        assert victim.closed is True
        assert keep.closed is False
        assert "webhook_1" not in rt._SESSIONS
        assert "signal_main" in rt._SESSIONS

    def test_releases_both_sides_of_a_compression_rotation(self, monkeypatch):
        before, after = self._FakeSession(), self._FakeSession()
        rt = self._pool(monkeypatch, cron_orig=before, cron_tip=after)

        assert rt.release_claude_cli_sessions_by_owner("cron_tip", "cron_orig") == 2
        assert before.closed is True
        assert after.closed is True
        assert rt._SESSIONS == {}

    def test_unknown_blank_and_duplicate_keys_are_harmless(self, monkeypatch):
        only = self._FakeSession()
        rt = self._pool(monkeypatch, real=only)

        assert rt.release_claude_cli_sessions_by_owner("missing", "", None) == 0
        assert only.closed is False
        # A rotation that never happened passes the same id twice.
        assert rt.release_claude_cli_sessions_by_owner("real", "real") == 1
        assert only.closed is True

    def test_a_failing_close_does_not_strand_the_other_victim(self, monkeypatch):
        class _Angry(self._FakeSession):
            def close(self):
                raise RuntimeError("child already gone")

        angry, ok = _Angry(), self._FakeSession()
        rt = self._pool(monkeypatch, a=angry, b=ok)

        # Both leave the pool even though one child refuses to die.
        assert rt.release_claude_cli_sessions_by_owner("a", "b") == 2
        assert ok.closed is True
        assert rt._SESSIONS == {}
