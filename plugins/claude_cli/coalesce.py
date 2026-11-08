"""Reassemble Claude Code's per-block assistant records into complete messages.

Claude Code 2.1 emits one ``assistant`` stream-json record per content block (``thinking``, then
``tool_use``, then ``text``), all sharing the message id and with no ``stop_reason``. The turn reader
was written for one complete record per message: it de-duplicates on ``(type, message id)`` and
projects, counts and reports each record as a whole model step, so everything after the first block
(the tool calls, the text) was dropped from Hermes' transcript.

:class:`AssistantRecordCoalescer` sits between the stdout queue and the reader. Records of one message
are merged (a block repeated verbatim, as a replay emits it, is kept once) and released as one
complete record once known complete: its ``message_stop`` (or the next message's ``message_start``),
the next non-stream event (the tool result, ``result``), or a quiet poll while it holds tool calls.
Partial ``stream_event`` deltas pass through immediately, so streaming is unaffected;
an older Claude Code that already emits complete records passes through unchanged.
"""

from __future__ import annotations

import collections
import copy
import json
import queue
from typing import Any, Callable, Optional


def _block_key(block: Any) -> str:
    try:
        return json.dumps(block, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return repr(block)


def _assistant_id(event: Any) -> Optional[str]:
    if not isinstance(event, dict) or event.get("type") != "assistant":
        return None
    message = event.get("message")
    message_id = message.get("id") if isinstance(message, dict) else None
    return str(message_id) if message_id else None


class AssistantRecordCoalescer:
    """Pull events from a ``queue.Queue`` with per-message assistant records merged."""

    def __init__(self, source: "queue.Queue[Any] | Callable[[], queue.Queue[Any]]"):
        # A callable is re-read on every pull, so a child restarted mid-read is followed.
        self._source = source if callable(source) else (lambda: source)
        self._ready: collections.deque[Any] = collections.deque()
        self._pending: Optional[dict[str, Any]] = None
        self._pending_id: Optional[str] = None
        self._pending_keys: set[str] = set()

    def get(self, timeout: float) -> Any:
        """Next event, or raise ``queue.Empty`` after ``timeout`` with nothing ready.

        A held message is released when it is known complete (``message_stop``, the next message's
        ``message_start``, any non-stream event) or, holding tool calls, when the stream goes quiet
        for a whole poll: Claude streams deltas while it still generates, and a tool proposal or a
        loopback batch may be waiting on exactly this record."""
        while not self._ready:
            try:
                event = self._source().get(timeout=timeout)
            except queue.Empty:
                if self._pending is not None and self._holds_tool_use():
                    self._ready.append(self.flush())
                    continue
                raise
            self._accept(event)
        return self._ready.popleft()

    def flush(self) -> Optional[dict[str, Any]]:
        """Release the held message (end of stream)."""
        held = self._pending
        self._pending, self._pending_id, self._pending_keys = None, None, set()
        return held

    def _accept(self, event: Any) -> None:
        message_id = _assistant_id(event)
        if message_id is not None and message_id == self._pending_id:
            self._merge(event)
            return
        if isinstance(event, dict) and event.get("type") == "stream_event":
            inner = event.get("event")
            inner_type = inner.get("type") if isinstance(inner, dict) else None
            if inner_type in {"message_stop", "message_start"} and self._pending is not None:
                # The held message is complete (its stop, or the next request has begun).
                self._ready.append(self.flush())
            self._ready.append(event)
            return
        held = self.flush()
        if held is not None:
            self._ready.append(held)
        if message_id is not None:
            self._hold(event, message_id)
        else:
            self._ready.append(event)

    def _holds_tool_use(self) -> bool:
        content = (self._pending or {}).get("message", {}).get("content") or []
        return any(isinstance(block, dict) and block.get("type") == "tool_use" for block in content)

    def _hold(self, event: dict[str, Any], message_id: str) -> None:
        held = copy.copy(event)
        message = dict(event.get("message") or {})
        blocks = list(message.get("content") or []) if isinstance(message.get("content"), list) else []
        message["content"] = []
        held["message"] = message
        self._pending, self._pending_id, self._pending_keys = held, message_id, set()
        self._append_blocks(blocks)

    def _merge(self, event: dict[str, Any]) -> None:
        assert self._pending is not None
        message = event.get("message") or {}
        blocks = message.get("content") if isinstance(message.get("content"), list) else []
        self._append_blocks(blocks)
        held_message = self._pending["message"]
        # Later records carry the freshest metadata (usage, a stop_reason once Claude sends one).
        for key, value in message.items():
            if key != "content" and value is not None:
                held_message[key] = value
        for key, value in event.items():
            if key not in {"message", "type"} and value is not None:
                self._pending[key] = value

    def _append_blocks(self, blocks: list) -> None:
        assert self._pending is not None
        content = self._pending["message"]["content"]
        for block in blocks:
            key = _block_key(block)
            if key in self._pending_keys:
                continue
            self._pending_keys.add(key)
            content.append(block)
