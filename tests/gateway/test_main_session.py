from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import HomeChannel, Platform, PlatformConfig
from gateway.main_session import (
    MainSessionUnavailable,
    enqueue_main_session_turn,
    normalize_session_mode,
    resolve_main_session_source,
)
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key


def _runner_with_home(source: SessionSource | None = None):
    adapter = MagicMock()
    adapter.is_connected = True
    adapter._active_sessions = {}
    adapter._pending_messages = {}
    adapter.handle_message = AsyncMock()
    adapter.build_source.side_effect = lambda **kwargs: SessionSource(
        platform=Platform.SIGNAL, **kwargs
    )

    home = HomeChannel(
        platform=Platform.SIGNAL,
        chat_id="+15551234567",
        name="Note to Self",
    )
    adapter.config = PlatformConfig(enabled=True, home_channel=home)
    entry = SimpleNamespace(origin=source) if source is not None else None
    runner = SimpleNamespace(
        _running=True,
        _draining=False,
        _external_drain_active=False,
        _running_agents={},
        _sessions={},
        _session_sources={},
        adapters={Platform.SIGNAL: adapter},
        config=SimpleNamespace(
            platforms={
                Platform.SIGNAL: PlatformConfig(
                    enabled=True,
                    home_channel=home,
                )
            }
        ),
        session_store=SimpleNamespace(
            list_sessions=lambda: [entry] if entry is not None else []
        ),
    )
    runner._adapter_for_source = lambda resolved: runner.adapters.get(resolved.platform)
    runner._session_key_for_source = lambda resolved: build_session_key(resolved)
    runner._cache_session_source = lambda key, resolved: runner._session_sources.update(
        {key: resolved}
    )
    runner._sessions_map = lambda: GatewayRunner._sessions_map(runner)
    runner._session_state = lambda key: GatewayRunner._session_state(runner, key)
    runner._peek_session_state = lambda key: GatewayRunner._peek_session_state(runner, key)
    runner._enqueue_fifo = lambda key, event, selected: GatewayRunner._enqueue_fifo(
        runner, key, event, selected
    )
    return runner, adapter


def test_session_mode_is_only_main_or_isolated():
    assert normalize_session_mode(None) == "isolated"
    assert normalize_session_mode(" MAIN ") == "main"
    with pytest.raises(ValueError, match="main"):
        normalize_session_mode("fork")


def test_resolver_reuses_durable_exact_home_source():
    persisted = SessionSource(
        platform=Platform.SIGNAL,
        chat_id="+15551234567",
        chat_name="Note to Self",
        chat_type="dm",
        user_id="+15551234567",
        user_id_alt="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
    )
    runner, _ = _runner_with_home(persisted)

    resolved = resolve_main_session_source(runner)

    assert resolved == persisted
    assert build_session_key(resolved) == "agent:main:signal:dm:+15551234567"


def test_resolver_uses_captured_session_key_with_profile_identity():
    default_source = SessionSource(
        platform=Platform.SIGNAL,
        chat_id="+15551234567",
        chat_type="dm",
    )
    work_source = SessionSource(
        platform=Platform.SIGNAL,
        chat_id="+15551234567",
        chat_type="dm",
        user_id="+15551234567",
        scope_id="workspace-1",
        profile="work",
    )
    runner, _ = _runner_with_home()
    runner._profile_adapters = {"work": runner.adapters}
    runner._session_sources = {
        "agent:work:signal:dm:+15551234567": work_source,
        "agent:main:signal:dm:+15551234567": default_source,
    }

    resolved = resolve_main_session_source(
        runner,
        origin={
            "platform": "signal",
            "chat_id": "+15551234567",
            "session_key": "agent:work:signal:dm:+15551234567",
            "profile": "work",
        },
        profile="work",
    )

    assert resolved == work_source
    assert resolved.scope_id == "workspace-1"
    assert resolved.profile == "work"


def test_resolver_stamps_profile_on_new_dm_home():
    runner, adapter = _runner_with_home()
    runner._profile_adapters = {"work": runner.adapters}

    resolved = resolve_main_session_source(runner, profile="work")

    assert resolved.profile == "work"
    adapter.build_source.assert_called_once()


def test_resolver_rejects_without_live_home():
    runner, adapter = _runner_with_home()
    runner.config.platforms = {}
    adapter.config = PlatformConfig(enabled=True)

    with pytest.raises(MainSessionUnavailable, match="home_channel"):
        resolve_main_session_source(runner)


@pytest.mark.asyncio
async def test_idle_main_turn_enters_normal_adapter_pipeline():
    source = SessionSource(
        platform=Platform.SIGNAL,
        chat_id="+15551234567",
        chat_type="dm",
        user_id="+15551234567",
    )
    runner, adapter = _runner_with_home(source)

    result = await enqueue_main_session_turn(
        runner,
        source=source,
        text="scheduled prompt",
        event_id="cron:j1:1",
        metadata={"trigger": "cron"},
    )

    assert result.queued is False
    event = adapter.handle_message.await_args.args[0]
    assert event.internal is True
    assert event.text == "scheduled prompt"
    assert event.metadata == {"main_session": True, "trigger": "cron"}


@pytest.mark.asyncio
async def test_busy_main_turns_stay_separate_fifo_entries():
    source = SessionSource(
        platform=Platform.SIGNAL,
        chat_id="+15551234567",
        chat_type="dm",
        user_id="+15551234567",
    )
    runner, adapter = _runner_with_home(source)
    session_key = build_session_key(source)
    adapter._active_sessions[session_key] = object()

    first = await enqueue_main_session_turn(
        runner, source=source, text="cron one", event_id="cron:1"
    )
    second = await enqueue_main_session_turn(
        runner, source=source, text="hook two", event_id="hook:2"
    )

    assert first.queued is True
    assert second.queued is True
    assert adapter._pending_messages[session_key].text == "cron one"
    overflow = runner._session_state(session_key).conversation.queued_events
    assert [event.text for event in overflow] == ["hook two"]
    adapter.handle_message.assert_not_awaited()
