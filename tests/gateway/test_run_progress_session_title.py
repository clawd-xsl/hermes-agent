"""The session-title callback reaches an adapter that shows titles itself (``on_session_title``), with
every title the titler produces; adapters without the method keep Hermes's own rename lanes."""

from __future__ import annotations

from types import SimpleNamespace

SOURCE = SimpleNamespace(chat_id="9B2E", platform="aleph", thread_id=None)


def _mixin(adapter):
    from gateway.run_turn_progress import GatewayTurnProgressMixin
    mixin = GatewayTurnProgressMixin.__new__(GatewayTurnProgressMixin)
    mixin._ctx = SimpleNamespace(source=SOURCE, session_key="k")
    mixin._runner = SimpleNamespace(
        _delivery_adapter_for=lambda source: adapter,
        _recover_discord_auto_thread_source=lambda source, key: source,
        _is_telegram_topic_lane=lambda source: False,
        _is_discord_auto_thread_lane=lambda source: False,
        _is_relay_discord_channel_lane=lambda source: False)
    return mixin


def test_an_adapter_that_shows_titles_hears_each_one():
    seen = []

    class Wants:
        def on_session_title(self, source, title, title_source):
            seen.append((source, title, title_source))
    agent = SimpleNamespace(session_id="s1")
    mixin = _mixin(Wants())
    mixin._attach_session_title_callback(agent, mixin._ctx)
    agent._on_session_title("Weather in Paris", "derived")
    agent._on_session_title("Paris weather this weekend", "llm")
    assert seen == [(SOURCE, "Weather in Paris", "derived"), (SOURCE, "Paris weather this weekend", "llm")]


def test_other_adapters_are_unchanged():
    agent = SimpleNamespace(session_id="s1")
    mixin = _mixin(SimpleNamespace())
    mixin._attach_session_title_callback(agent, mixin._ctx)
    assert not hasattr(agent, "_on_session_title")  # no rename lane for this source
