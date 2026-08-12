"""Regression tests for background review agent cleanup."""

from __future__ import annotations

import run_agent as run_agent_module
from run_agent import AIAgent


def _bare_agent() -> AIAgent:
    agent = object.__new__(AIAgent)
    agent.model = "fake-model"
    agent.platform = "telegram"
    agent.provider = "openai"
    agent.base_url = ""
    agent.api_key = ""
    agent.api_mode = ""
    agent.session_id = "test-session"
    agent._parent_session_id = ""
    agent._credential_pool = None
    agent._memory_store = object()
    agent._memory_enabled = True
    agent._user_profile_enabled = False
    agent._cached_system_prompt = "test-cached-system-prompt"
    import datetime as _dt
    agent.session_start = _dt.datetime(2026, 1, 1, 12, 0, 0)
    agent._MEMORY_REVIEW_PROMPT = "review memory"
    agent._SKILL_REVIEW_PROMPT = "review skills"
    agent._COMBINED_REVIEW_PROMPT = "review both"
    agent.background_review_callback = None
    agent.status_callback = None
    agent._safe_print = lambda *_args, **_kwargs: None
    return agent


class ImmediateThread:
    def __init__(self, *, target, daemon=None, name=None):
        self._target = target

    def start(self):
        self._target()


def test_background_review_shuts_down_memory_provider_before_close(monkeypatch):
    events = []

    class FakeReviewAgent:
        def __init__(self, **kwargs):
            events.append(("init", kwargs))
            self._session_messages = []

        def run_conversation(self, **kwargs):
            events.append(("run_conversation", kwargs))

        def shutdown_memory_provider(self):
            events.append(("shutdown_memory_provider", None))

        def close(self):
            events.append(("close", None))

    monkeypatch.setattr(run_agent_module, "AIAgent", FakeReviewAgent)
    monkeypatch.setattr(run_agent_module.threading, "Thread", ImmediateThread)

    agent = _bare_agent()

    AIAgent._spawn_background_review(
        agent,
        messages_snapshot=[{"role": "user", "content": "hello"}],
        review_memory=True,
    )

    assert [name for name, _payload in events] == [
        "init",
        "run_conversation",
        "shutdown_memory_provider",
        "close",
    ]


def test_background_review_fork_opts_out_of_session_finalization(monkeypatch):
    """The review fork shares the parent's live session_id, so it must set
    ``_end_session_on_close = False``. Otherwise close() (now finalizing owned
    session rows) would end the still-active parent session mid-conversation
    every time the review fires (~every 10 turns). Regression for #12029.
    """
    seen = {}

    class FakeReviewAgent:
        def __init__(self, **kwargs):
            self._session_messages = []
            # Default matches AIAgent.__init__ (agent_init.py): owns its row.
            self._end_session_on_close = True

        def __setattr__(self, name, value):
            object.__setattr__(self, name, value)
            if name == "_end_session_on_close":
                seen["end_session_on_close"] = value

        def run_conversation(self, **kwargs):
            # By the time the fork runs, the opt-out must already be applied.
            seen["at_run_time"] = self._end_session_on_close

        def shutdown_memory_provider(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(run_agent_module, "AIAgent", FakeReviewAgent)
    monkeypatch.setattr(run_agent_module.threading, "Thread", ImmediateThread)

    agent = _bare_agent()

    AIAgent._spawn_background_review(
        agent,
        messages_snapshot=[{"role": "user", "content": "hello"}],
        review_memory=True,
    )

    assert seen.get("end_session_on_close") is False
    assert seen.get("at_run_time") is False


def test_background_review_inherits_keychain_claude_runtime_and_isolates_history(
    monkeypatch,
):
    """The concrete review fork must remain usable without an HTTP API key.

    This covers the user-reported loophole end to end at the constructor
    boundary: the native runtime, command/args, and reasoning settings reach
    the child, while its harness turn is transient and cannot mutate or
    compact the parent's Hermes/native session history.
    """
    captured = {}

    class FakeReviewAgent:
        def __init__(self, **kwargs):
            captured["kwargs"] = dict(kwargs)
            self._session_messages = []

        def run_conversation(self, **kwargs):
            captured["run"] = dict(kwargs)
            captured["state"] = {
                "persist_disabled": self._persist_disabled,
                "session_db": self._session_db,
                "session_json_enabled": self._session_json_enabled,
                "session_id": self.session_id,
                "compression_enabled": self.compression_enabled,
                "end_session_on_close": self._end_session_on_close,
            }
            return {"final_response": "", "messages": []}

        def shutdown_memory_provider(self):
            pass

        def close(self):
            pass

    agent = _bare_agent()
    agent.model = "claude-sonnet-4-6"
    agent.provider = "anthropic"
    agent.api_mode = "claude_cli"
    agent.api_key = ""
    agent.base_url = ""
    agent.acp_command = "/opt/claude"
    agent.acp_args = ["--setting-sources", ""]
    agent.reasoning_config = {"enabled": True, "effort": "high"}
    agent.ephemeral_system_prompt = "stable gateway context"
    agent.prefill_messages = []
    agent.enabled_toolsets = ["memory", "skills", "terminal"]
    agent.disabled_toolsets = []
    agent.request_overrides = {}
    agent.max_tokens = None
    agent._credential_pool = None

    monkeypatch.setattr(run_agent_module, "AIAgent", FakeReviewAgent)
    monkeypatch.setattr(run_agent_module.threading, "Thread", ImmediateThread)
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {})

    snapshot = [
        {
            "role": "user" if i % 2 == 0 else "assistant",
            "content": f"historical row {i}",
        }
        for i in range(40)
    ]
    AIAgent._spawn_background_review(
        agent,
        messages_snapshot=snapshot,
        review_memory=True,
        review_skills=True,
    )

    kwargs = captured["kwargs"]
    assert kwargs["provider"] == "anthropic"
    assert kwargs["model"] == "claude-sonnet-4-6"
    assert kwargs["api_mode"] == "claude_cli"
    assert kwargs["api_key"] is None
    assert kwargs["acp_command"] == "/opt/claude"
    assert kwargs["acp_args"] == ["--setting-sources", ""]
    assert kwargs["reasoning_config"] == agent.reasoning_config
    assert kwargs["ephemeral_system_prompt"] == agent.ephemeral_system_prompt
    review_history = captured["run"]["conversation_history"]
    assert review_history[0]["content"].startswith(
        "[Earlier conversation digest"
    )
    assert review_history[-1] == snapshot[-1]
    assert len(review_history) < len(snapshot)
    assert captured["state"] == {
        "persist_disabled": True,
        "session_db": None,
        "session_json_enabled": False,
        "session_id": agent.session_id,
        "compression_enabled": False,
        "end_session_on_close": False,
    }










# ---------------------------------------------------------------------------
# memory_notifications mode: off | on | verbose
# ---------------------------------------------------------------------------

import json as _json

from agent.background_review import summarize_background_review_actions


def _memory_add_review():
    """A minimal review transcript: one memory add (assistant call + tool result)."""
    return [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_mem1",
                    "function": {
                        "name": "memory",
                        "arguments": _json.dumps(
                            {
                                "action": "add",
                                "target": "memory",
                                "content": "User prefers terse replies",
                            }
                        ),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_mem1",
            "content": _json.dumps(
                {"success": True, "message": "Entry added.", "target": "memory"}
            ),
        },
    ]


def _skill_patch_review():
    return [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_skill1",
                    "function": {
                        "name": "skill_manage",
                        "arguments": _json.dumps(
                            {"action": "patch", "name": "demo", "old_string": "a", "new_string": "b"}
                        ),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_skill1",
            "content": _json.dumps(
                {
                    "success": True,
                    "message": "Patched SKILL.md in skill 'demo' (1 replacement).",
                    "_change": {"old": "a", "new": "b"},
                }
            ),
        },
    ]


def test_memory_notifications_off_returns_nothing():
    actions = summarize_background_review_actions(
        _memory_add_review(), [], notification_mode="off"
    )
    assert actions == []








def test_skill_patch_off_silent_verbose_shows_diff():
    assert (
        summarize_background_review_actions(
            _skill_patch_review(), [], notification_mode="off"
        )
        == []
    )
    verbose = summarize_background_review_actions(
        _skill_patch_review(), [], notification_mode="verbose"
    )
    assert len(verbose) == 1
    assert "demo" in verbose[0] and "→" in verbose[0]
