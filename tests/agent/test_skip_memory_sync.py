"""``skip_memory_sync``: a read-only memory turn keeps recall but never writes back.

Used by one-shot classifier runs over untrusted external events (webhook main-session review):
their exchange must not be fed to external memory providers or the context engine.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from run_agent import AIAgent
from tests.agent.test_skip_background_review import _run_finalize, _stub_agent_for_finalize


def _make_agent(**kwargs) -> AIAgent:
    return AIAgent(
        model="openai/gpt-4o-mini", provider="openrouter", api_key="sk-dummy",
        base_url="https://openrouter.ai/api/v1", quiet_mode=True, skip_context_files=True,
        skip_memory=True, platform="cli", **kwargs,
    )


def test_flag_defaults_off_and_persists():
    assert _make_agent().skip_memory_sync is False
    assert _make_agent(skip_memory_sync=True).skip_memory_sync is True


def test_turn_is_not_synced_to_external_memory():
    agent = _make_agent(skip_memory_sync=True)
    manager = MagicMock()
    agent._memory_manager = manager
    agent._sync_external_memory_for_turn(original_user_message="external webhook payload",
                                         final_response="concise handoff", interrupted=False, messages=[])
    manager.sync_all.assert_not_called()
    manager.queue_prefetch_all.assert_not_called()


def test_ordinary_turn_still_syncs():
    agent = _make_agent()
    manager = MagicMock()
    agent._memory_manager = manager
    agent._sync_external_memory_for_turn(original_user_message="hello", final_response="hi",
                                         interrupted=False, messages=[])
    manager.sync_all.assert_called_once()


def test_context_engine_does_not_observe_the_turn(monkeypatch):
    import agent.conversation_loop as loop
    calls = []
    monkeypatch.setattr(loop, "_notify_context_engine_turn_complete", lambda *a, **k: calls.append(a))
    for flag, expected in ((True, 0), (False, 1)):
        calls.clear()
        agent = _make_agent(skip_memory_sync=flag)
        _stub_agent_for_finalize(agent)
        _run_finalize(agent)
        assert len(calls) == expected
