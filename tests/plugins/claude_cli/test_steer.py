"""Mid-turn /steer rides on Claude's MCP tool result (plugins/claude_cli/steer.py)."""
from __future__ import annotations

from agent.prompt_builder import STEER_MARKER_OPEN, steer_user_row
from plugins.claude_cli.steer import deliver_pending_steer_in_tool_results, fold_steer_rows_into_tool_results


class _Agent:
    def __init__(self, steer=None):
        self._steer = steer
        self.requeued = []

    def _drain_pending_steer(self):
        steer, self._steer = self._steer, None
        return steer


def test_standard_executor_steer_row_folds_into_last_tool_result():
    rows = [
        {"role": "assistant", "tool_calls": [{"id": "a"}, {"id": "b"}]},
        {"role": "tool", "tool_call_id": "a", "content": "first"},
        {"role": "tool", "tool_call_id": "b", "content": "second"},
        steer_user_row("go left"),
    ]
    assert fold_steer_rows_into_tool_results(rows, 1) == 1
    assert [row["role"] for row in rows] == ["assistant", "tool", "tool"]
    assert rows[1]["content"] == "first"
    assert rows[2]["content"].startswith("second\n\n" + STEER_MARKER_OPEN)
    assert "go left" in rows[2]["content"]


def test_steer_without_a_preceding_tool_result_stays():
    rows = [{"role": "assistant", "content": "x"}, steer_user_row("later")]
    assert fold_steer_rows_into_tool_results(rows, 0) == 0
    assert len(rows) == 2


def test_multimodal_tool_result_gets_a_text_part():
    rows = [{"role": "tool", "content": [{"type": "image_url", "image_url": {"url": "data:"}}]},
            steer_user_row("zoom in")]
    fold_steer_rows_into_tool_results(rows, 0)
    assert rows[0]["content"][-1]["type"] == "text"
    assert "zoom in" in rows[0]["content"][-1]["text"]


def test_pending_steer_is_drained_onto_the_trailing_tool_result():
    agent = _Agent("use the corrected target")
    messages = [{"role": "user", "content": "go"}, {"role": "tool", "content": "original tool result"}]
    assert deliver_pending_steer_in_tool_results(agent, messages, 1) is True
    assert "use the corrected target" in messages[-1]["content"]
    assert len(messages) == 2  # no standalone user row Claude would never see mid-loop
    assert deliver_pending_steer_in_tool_results(agent, messages, 1) is False


def test_no_trailing_tool_rows_leaves_the_steer_pending():
    agent = _Agent("keep me")
    assert deliver_pending_steer_in_tool_results(agent, [{"role": "user", "content": "x"}], 0) is False
    assert agent._steer == "keep me"
