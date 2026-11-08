"""``session: main``: automation turns enter the real home conversation's FIFO.

Resolution is unit-tested against a stub runner; admission runs a real GatewayRunner,
DiscordAdapter, SessionAuthority and drain over SQLite (the turn itself is a stand-in), so a
cron/webhook turn is proven to queue behind a running human turn as its own turn, in order.
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import HomeChannel, Platform, PlatformConfig
from gateway.main_session import (
    CRON_PRODUCER, WEBHOOK_PRODUCER, MainSessionUnavailable, enqueue_main_session_turn,
    normalize_session_mode, resolve_main_session_source)
from gateway.session import SessionSource, build_session_key
from tests.waiting import async_wait_until

_HOME_ID = "+15551234567"


def _runner_with_home(*persisted: SessionSource, profiles=("default",)):
    adapter = MagicMock()
    adapter.is_connected = True
    adapter._active_sessions = {}
    adapter.build_source.side_effect = lambda **kwargs: SessionSource(platform=Platform.SIGNAL, **kwargs)
    home = HomeChannel(platform=Platform.SIGNAL, chat_id=_HOME_ID, name="Note to Self")
    adapter.config = PlatformConfig(enabled=True, home_channel=home)
    adapters = {Platform.SIGNAL: adapter}
    entries = [SimpleNamespace(origin=src, session_key=build_session_key(src)) for src in persisted]
    runner = SimpleNamespace(
        _running=True, _draining=False, _external_drain_active=False, _running_agents={}, _session_sources={},
        adapters=adapters, config=SimpleNamespace(platforms={}),
        session_store=SimpleNamespace(list_sessions=lambda: list(entries)))
    runner._adapters_for_profile = lambda profile: adapters if (profile or "default") in profiles else {}
    return runner, adapter


def test_session_mode_is_only_main_or_isolated():
    assert normalize_session_mode(None) == "isolated"
    assert normalize_session_mode("") == "isolated"
    assert normalize_session_mode(" MAIN ") == "main"
    assert normalize_session_mode(None, default="main") == "main"
    with pytest.raises(ValueError, match="main"):
        normalize_session_mode("fork")


def test_resolver_reuses_the_persisted_exact_home_source():
    persisted = SessionSource(platform=Platform.SIGNAL, chat_id=_HOME_ID, chat_name="Note to Self",
                              chat_type="dm", user_id=_HOME_ID,
                              user_id_alt="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
    runner, adapter = _runner_with_home(persisted)
    resolved = resolve_main_session_source(runner)
    assert resolved == persisted
    assert build_session_key(resolved) == f"agent:main:signal:dm:{_HOME_ID}"
    adapter.build_source.assert_not_called()


def test_resolver_prefers_the_jobs_captured_session_key_and_profile():
    default_source = SessionSource(platform=Platform.SIGNAL, chat_id=_HOME_ID, chat_type="dm")
    work_source = SessionSource(platform=Platform.SIGNAL, chat_id=_HOME_ID, chat_type="dm", user_id=_HOME_ID,
                                scope_id="workspace-1", profile="work")
    runner, _ = _runner_with_home(profiles=("default", "work"))
    runner._session_sources = {"agent:work:signal:dm:x": work_source, "agent:main:signal:dm:x": default_source}
    resolved = resolve_main_session_source(
        runner, origin={"platform": "signal", "chat_id": _HOME_ID, "session_key": "agent:work:signal:dm:x",
                        "profile": "work"})
    assert resolved == work_source
    assert resolved.scope_id == "workspace-1"


def test_resolver_rebuilds_a_never_seen_dm_home_stamped_with_the_profile():
    runner, adapter = _runner_with_home(profiles=("work",))
    resolved = resolve_main_session_source(runner, profile="work")
    assert resolved.profile == "work"
    assert resolved.chat_id == _HOME_ID and resolved.chat_type == "dm"
    adapter.build_source.assert_called_once()


def test_resolver_refuses_a_never_seen_thread_home():
    runner, adapter = _runner_with_home()
    adapter.config.home_channel = HomeChannel(platform=Platform.SIGNAL, chat_id=_HOME_ID, name="t", thread_id="9")
    with pytest.raises(MainSessionUnavailable, match="group/thread"):
        resolve_main_session_source(runner)


def test_resolver_needs_a_live_home():
    runner, adapter = _runner_with_home()
    adapter.config = PlatformConfig(enabled=True)
    with pytest.raises(MainSessionUnavailable, match="home_channel"):
        resolve_main_session_source(runner)
    runner._running = False
    with pytest.raises(MainSessionUnavailable, match="running"):
        resolve_main_session_source(runner)


def test_resolver_refuses_ambiguous_homes_unless_the_origin_names_one():
    runner, signal = _runner_with_home()
    telegram = MagicMock()
    telegram.config = PlatformConfig(enabled=True, home_channel=HomeChannel(
        platform=Platform.TELEGRAM, chat_id="777", name="tg"))
    runner.adapters[Platform.TELEGRAM] = telegram
    with pytest.raises(MainSessionUnavailable, match="ambiguous"):
        resolve_main_session_source(runner)
    resolved = resolve_main_session_source(runner, origin={"platform": "signal", "chat_id": _HOME_ID})
    assert resolved.platform == Platform.SIGNAL


@pytest.mark.asyncio
async def test_draining_gateway_refuses_the_turn():
    source = SessionSource(platform=Platform.SIGNAL, chat_id=_HOME_ID, chat_type="dm", user_id=_HOME_ID)
    runner, _ = _runner_with_home(source)
    runner._draining = True
    with pytest.raises(MainSessionUnavailable, match="draining"):
        await enqueue_main_session_turn(runner, source=source, text="x", producer=CRON_PRODUCER)


@pytest.mark.asyncio
async def test_without_an_authority_the_turn_is_admitted_through_the_adapter(monkeypatch):
    source = SessionSource(platform=Platform.SIGNAL, chat_id=_HOME_ID, chat_type="dm", user_id=_HOME_ID)
    runner, adapter = _runner_with_home(source)
    runner._adapter_for_source = lambda resolved: adapter
    runner._session_key_for_source = build_session_key
    runner._cache_session_source = lambda key, resolved: runner._session_sources.update({key: resolved})
    monkeypatch.setattr("gateway.session_automation.admit_route_wake", AsyncMock(return_value=False))

    async def accept(event):
        event._gateway_accepted = True
    adapter.handle_message = AsyncMock(side_effect=accept)
    adapter.gateway_runner = runner
    result = await enqueue_main_session_turn(runner, source=source, text="scheduled prompt",
                                             producer=CRON_PRODUCER, metadata={"trigger": "cron"})
    assert (result.queued, result.durable) == (False, False)
    event = adapter.handle_message.await_args.args[0]
    assert event.internal is True and event.text == "scheduled prompt"
    assert event.metadata == {"main_session": True, "trigger": "cron"}
    assert event.message_id.startswith("cron-main:")


@pytest.mark.asyncio
async def test_slash_text_is_never_a_main_turn():
    source = SessionSource(platform=Platform.SIGNAL, chat_id=_HOME_ID, chat_type="dm", user_id=_HOME_ID)
    runner, _ = _runner_with_home(source)
    with pytest.raises(ValueError, match="slash"):
        await enqueue_main_session_turn(runner, source=source, text="/reset", producer=CRON_PRODUCER)


@pytest.mark.asyncio
async def test_main_turns_queue_behind_a_running_human_turn_as_separate_turns(monkeypatch):
    from gateway.config import GatewayConfig
    from gateway.platforms.base import SendResult
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority
    from gateway.session_results import execution_result
    from plugins.platforms.discord.adapter import DiscordAdapter

    monkeypatch.setenv("DISCORD_ALLOWED_USERS", "42")
    runner = GatewayRunner(GatewayConfig())
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="fixture-token", typing_indicator=False,
                                            home_channel=HomeChannel(platform=Platform.DISCORD, chat_id="42",
                                                                     name="Home")))

    async def send(chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="sent")
    monkeypatch.setattr(adapter, "send", send)
    adapter.gateway_runner = runner
    adapter._running = True
    runner.adapters = {Platform.DISCORD: adapter}
    runner._running = True
    await initialize_session_authority(runner, profile_id="default", instance_id="owner")
    runner._wire_adapter_handlers(adapter)
    human = adapter.build_source(chat_id="42", chat_type="dm", user_id="42")
    running, finish = asyncio.Event(), asyncio.Event()
    executed = []

    class Agent:
        def interrupt(self, *args, **kwargs):
            pass

    async def turn(event, source, quick_key, run_generation):
        executed.append(event.text)
        if event.text == "A":
            runner._session_state(quick_key).turn.agent = Agent()
            running.set()
            await finish.wait()
        captured = execution_result.get()
        if captured is not None:
            captured["result"] = {"final_response": "reply", "messages": []}
        return "reply"
    monkeypatch.setattr(runner, "_handle_message_with_agent", turn)
    await adapter.handle_message(MessageEvent(text="A", source=human, message_id="A"))
    await asyncio.wait_for(running.wait(), 10)

    source = resolve_main_session_source(runner)
    assert build_session_key(source) == runner.session_store._generate_session_key(human)
    first = await enqueue_main_session_turn(runner, source=source, text="cron one", producer=CRON_PRODUCER)
    second = await enqueue_main_session_turn(runner, source=source, text="hook two", producer=WEBHOOK_PRODUCER)
    assert first.durable and second.durable
    assert first.queued and second.queued
    assert executed == ["A"]  # never interrupts the running turn
    finish.set()
    await async_wait_until(lambda: len(executed) == 3, timeout=15, what="both automation turns to run")
    assert executed == ["A", "cron one", "hook two"]
