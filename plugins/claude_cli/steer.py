"""Mid-turn ``/steer`` delivery inside Claude's native tool loop.

Hermes delivers a steer as a standalone user row after the newest tool result
(``agent_runtime_helpers.apply_pending_steer_to_tool_results``). Claude Code's inner loop is private:
between two of its model requests the only host-controlled bytes are the MCP tool results, so a
separate user row would never reach Claude while it is still mid-turn. The steer therefore rides on
the last tool result of the batch, wrapped in the same self-describing out-of-band marker. The
Hermes mirror keeps that exact content: it is what Claude saw, and the native thread (not the mirror)
is what later turns continue from.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _append_marker(content: Any, marker: str) -> Any:
    if isinstance(content, list):
        return [*content, {"type": "text", "text": marker.lstrip()}]
    return f"{content if content is not None else ''}{marker}"


def _last_tool_row(rows: list, start: int = 0) -> Optional[dict]:
    for row in reversed(rows[start:]):
        if isinstance(row, dict) and row.get("role") == "tool":
            return row
    return None


def fold_steer_rows_into_tool_results(rows: list, start: int = 0) -> int:
    """Move standalone steer rows the standard executor appended at ``rows[start:]`` onto the tool
    result before each; returns how many were folded. A steer with no tool result before it stays."""
    from agent.prompt_builder import STEER_DISPLAY_KIND
    folded = 0
    index = max(0, start)
    while index < len(rows):
        row = rows[index]
        if isinstance(row, dict) and row.get("role") == "user" and row.get("display_kind") == STEER_DISPLAY_KIND:
            target = _last_tool_row(rows[:index], start)
            if target is not None:
                target["content"] = _append_marker(target.get("content"), "\n\n" + str(row.get("content") or ""))
                del rows[index]
                folded += 1
                continue
        index += 1
    return folded


def deliver_pending_steer_in_tool_results(agent: Any, messages: list, trailing_tool_rows: int) -> bool:
    """Drain a pending steer onto the last of the ``trailing_tool_rows`` tool results; True when one was
    delivered. With no tool result to carry it, the steer is requeued for the next user turn."""
    if trailing_tool_rows <= 0 or not messages:
        return False
    drain = getattr(agent, "_drain_pending_steer", None)
    steer_text = drain() if callable(drain) else None
    if not steer_text:
        return False
    target = _last_tool_row(messages, max(0, len(messages) - trailing_tool_rows))
    if target is None:
        from agent.agent_runtime_helpers import _requeue_pending_steer
        _requeue_pending_steer(agent, steer_text)
        return False
    from agent.prompt_builder import format_steer_marker
    target["content"] = _append_marker(target.get("content"), format_steer_marker(steer_text))
    logger.info("Delivered /steer inside Claude's native tool result (%d chars)", len(steer_text))
    return True
