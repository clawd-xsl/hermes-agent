"""The progress callback reports tool starts and finishes to an adapter that keeps its own record of
the agent's steps (``on_tool_activity``), and to no other."""

from __future__ import annotations

from types import SimpleNamespace

SOURCE = SimpleNamespace(chat_id="main", platform="aleph")


def test_the_progress_callback_reports_tools_to_an_adapter_that_wants_them():
    from gateway.run_turn_progress import GatewayTurnProgressMixin
    seen = []

    class Wants:
        def on_tool_activity(self, source, event):
            seen.append((source, event))

    class Plain:
        pass

    def runner_with(adapter):
        mixin = GatewayTurnProgressMixin.__new__(GatewayTurnProgressMixin)
        mixin._ctx = SimpleNamespace(source=SOURCE, _run_still_current=lambda: True, _live_status_mode="verb")
        mixin._runner = SimpleNamespace(_delivery_adapter_for=lambda source: adapter)
        return mixin
    runner = runner_with(Wants())
    runner._progress_adapter_activity("tool.started", "web_search", {"query": "x"}, {})
    runner._progress_adapter_activity("tool.completed", "web_search", None, {"is_error": True, "duration": 1.5})
    runner._progress_adapter_activity("_thinking", "_thinking", None, {})
    runner._progress_adapter_activity("tool.started", "_thinking", None, {})
    assert [e["type"] for _, e in seen] == ["tool.started", "tool.completed"]
    assert seen[0][1] == {"type": "tool.started", "tool": "web_search", "args": {"query": "x"}, "is_error": None,
                          "duration": None, "live_status": "verb"}
    assert seen[1][1]["is_error"] is True and seen[1][1]["duration"] == 1.5
    runner_with(Plain())._progress_adapter_activity("tool.started", "web_search", {}, {})  # nothing to call
