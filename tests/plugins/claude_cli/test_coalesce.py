"""Claude Code 2.1 per-block assistant records are reassembled into complete messages."""
from __future__ import annotations

import queue

import pytest

from plugins.claude_cli.coalesce import AssistantRecordCoalescer


def _assistant(message_id, *blocks, **extra):
    return {"type": "assistant", "message": {"id": message_id, "role": "assistant", "content": list(blocks), **extra}}


def _drain(events):
    source = queue.Queue()
    for event in events:
        source.put(event)
    coalescer = AssistantRecordCoalescer(source)
    out = []
    while True:
        try:
            out.append(coalescer.get(timeout=0.01))
        except queue.Empty:
            break
    held = coalescer.flush()
    return out, held


def test_per_block_records_merge_into_one_message_released_by_the_tool_result():
    thinking = {"type": "thinking", "thinking": "plan"}
    use = {"type": "tool_use", "id": "toolu_1", "name": "mcp__hermes__terminal", "input": {"command": "ls"}}
    delta = {"type": "stream_event", "event": {"type": "content_block_delta"}}
    result = {"type": "user", "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1"}]}}
    out, held = _drain([_assistant("m1", thinking), delta, _assistant("m1", use, usage={"output_tokens": 5}), result])
    assert [e["type"] for e in out] == ["stream_event", "assistant", "user"]
    merged = out[1]["message"]
    assert merged["content"] == [thinking, use]
    assert merged["usage"] == {"output_tokens": 5}
    assert held is None


def test_text_before_tool_use_stays_in_the_same_row():
    text = {"type": "text", "text": "Let me check."}
    use = {"type": "tool_use", "id": "t", "name": "mcp__hermes__x", "input": {}}
    out, _ = _drain([_assistant("m1", text), _assistant("m1", use), {"type": "result"}])
    assert [b["type"] for b in out[0]["message"]["content"]] == ["text", "tool_use"]
    assert out[1]["type"] == "result"


def test_a_new_message_id_releases_the_previous_message():
    out, held = _drain([_assistant("m1", {"type": "text", "text": "a"}), _assistant("m2", {"type": "text", "text": "b"})])
    assert [e["message"]["id"] for e in out] == ["m1"]
    assert held["message"]["id"] == "m2"


def test_replayed_identical_blocks_are_kept_once_and_complete_records_pass_through():
    block = {"type": "text", "text": "same"}
    out, _ = _drain([_assistant("m1", block), _assistant("m1", block), {"type": "result"}])
    assert out[0]["message"]["content"] == [block]
    complete = _assistant("m9", {"type": "thinking", "thinking": "x"}, {"type": "text", "text": "y"}, stop_reason="end_turn")
    out, _ = _drain([complete, {"type": "result"}])
    assert out[0] == complete


def test_message_stop_releases_before_the_stop_event():
    use = {"type": "tool_use", "id": "t", "name": "mcp__hermes__x", "input": {}}
    stop = {"type": "stream_event", "event": {"type": "message_stop"}}
    out, held = _drain([_assistant("m1", use), stop])
    assert [e["type"] for e in out] == ["assistant", "stream_event"]
    assert held is None


def test_parallel_tool_uses_streamed_back_to_back_stay_one_message():
    a = {"type": "tool_use", "id": "a", "name": "mcp__hermes__x", "input": {}}
    b = {"type": "tool_use", "id": "b", "name": "mcp__hermes__x", "input": {}}
    delta = {"type": "stream_event", "event": {"type": "content_block_delta"}}
    out, _ = _drain([_assistant("m1", a), delta, _assistant("m1", b), {"type": "stream_event", "event": {"type": "message_stop"}}])
    assistant = [e for e in out if e["type"] == "assistant"]
    assert len(assistant) == 1 and [blk["id"] for blk in assistant[0]["message"]["content"]] == ["a", "b"]


def test_quiet_stream_releases_held_tool_calls():
    source = queue.Queue()
    source.put(_assistant("m1", {"type": "tool_use", "id": "t", "name": "mcp__hermes__x", "input": {}}))
    coalescer = AssistantRecordCoalescer(source)
    released = coalescer.get(timeout=0.01)
    assert released["message"]["content"][0]["type"] == "tool_use"


def test_timeout_never_releases_a_held_text_message():
    source = queue.Queue()
    source.put(_assistant("m1", {"type": "thinking", "thinking": "x"}))
    coalescer = AssistantRecordCoalescer(source)
    with pytest.raises(queue.Empty):
        coalescer.get(timeout=0.01)
    source.put(_assistant("m1", {"type": "text", "text": "done"}))
    source.put({"type": "result"})
    assert [b["type"] for b in coalescer.get(timeout=0.01)["message"]["content"]] == ["thinking", "text"]
