"""propagate_context_to_thread carries a restricted turn's tool whitelist into the worker.

Background review and side-question forks restrict tool dispatch with a thread-local whitelist. A worker
that dispatches tools for that turn (a parallel tool batch, a turn runtime's tool loopback) must not run
them unrestricted.
"""
from __future__ import annotations

import threading

from hermes_cli.plugins import (
    clear_thread_tool_whitelist,
    get_pre_tool_call_block_message,
    get_thread_tool_whitelist_state,
    set_thread_tool_whitelist,
)
from tools.thread_context import propagate_context_to_thread


def _run_in_thread(target):
    box = {}
    worker = threading.Thread(target=lambda: box.setdefault("value", target()))
    worker.start()
    worker.join(timeout=5)
    return box.get("value")


def test_worker_inherits_and_then_clears_the_whitelist(monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_k: [])
    set_thread_tool_whitelist({"read_file"}, deny_msg_fmt="denied: {tool_name}")
    try:
        wrapped = propagate_context_to_thread(
            lambda: (get_pre_tool_call_block_message("terminal", {}),
                     get_pre_tool_call_block_message("read_file", {})))
    finally:
        clear_thread_tool_whitelist()

    blocked, allowed = _run_in_thread(wrapped)
    assert blocked == "denied: terminal"
    assert allowed is None


def test_worker_thread_is_left_unrestricted_after_the_call():
    set_thread_tool_whitelist({"read_file"})
    try:
        wrapped = propagate_context_to_thread(lambda: get_thread_tool_whitelist_state())
    finally:
        clear_thread_tool_whitelist()
    results = []

    def worker():
        results.append(wrapped())
        results.append(get_thread_tool_whitelist_state())

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(timeout=5)
    assert results[0][0] == {"read_file"}
    assert results[1] is None


def test_unrestricted_parent_installs_nothing():
    clear_thread_tool_whitelist()
    wrapped = propagate_context_to_thread(lambda: get_thread_tool_whitelist_state())
    assert _run_in_thread(wrapped) is None
