"""Compaction for the ``claude_cli`` runtime: Claude's native ``/compact`` on the authoritative thread.

``compress_context`` routes here (``ClaudeCliTurnRuntime.compact``) while the runtime owns the
context. Claude compacts automatically on context pressure, so automatic Hermes triggers are skipped;
an explicit ``/compress [focus]`` (``force``) and an opt-in idle compaction (``trigger="idle"``) cross
to the native ``/compact``. Hermes' own transcript is deliberately kept whole for search and crash
recovery: message counts do not shrink even though Claude's live context does.
"""

from __future__ import annotations

import contextlib
import logging
import time
from typing import Any, Optional

logger = logging.getLogger("agent.conversation_compression")

#: Automatic triggers that intentionally request a native boundary (others are pressure-driven,
#: which Claude handles itself).
_NATIVE_AUTOMATIC_TRIGGERS = frozenset({"idle"})


def claude_cli_owns_context_compaction(agent: Any) -> bool:
    """Whether Claude's native history owns this agent's compaction policy.

    The built-in ``ContextCompressor`` must not rewrite Hermes' mirror of the live thread. A configured
    context-engine plugin is different: it is the user-selected policy, keeps ownership, and the
    persistent native binding is rebuilt from its committed transcript at the resulting boundary.
    Failure to classify never lets the host rewrite a live native thread behind Claude's back."""
    if getattr(agent, "api_mode", None) != "claude_cli":
        return False
    selected_external = getattr(agent, "_context_engine_is_external", None)
    if isinstance(selected_external, bool):
        return not selected_external
    engine = getattr(agent, "context_compressor", None)
    try:
        from agent.context_compressor import ContextCompressor
        from agent.context_engine import ContextEngine
        return not (isinstance(engine, ContextEngine) and not isinstance(engine, ContextCompressor))
    except Exception:
        return True


def _keep_last_exchanges(verbatim_tail: Optional[list]) -> Optional[int]:
    """``/compress here N`` hands core the kept tail; Claude needs it as an instruction instead."""
    if not verbatim_tail:
        return None
    count = sum(1 for row in verbatim_tail if isinstance(row, dict) and row.get("role") == "user")
    return count or None


def _native_focus(focus_topic: Optional[str], keep_last: Optional[int], memory_context: str) -> Optional[str]:
    parts: list[str] = []
    if isinstance(focus_topic, str) and focus_topic.strip():
        parts.append(focus_topic.strip())
    if isinstance(keep_last, int) and keep_last > 0:
        parts.append(
            f"Keep the most recent {keep_last} user exchange(s), including their tool calls and results, "
            "in full detail; compact only older context.")
    if memory_context.strip():
        parts.append(f"Preserve these memory-provider insights in the compacted context:\n{memory_context.strip()}")
    return "\n\n".join(parts) or None


def _clear_attempt_state(agent: Any, compressor: Any) -> None:
    agent._last_native_compaction = False
    # Gateway hygiene re-baselines its pressure meter on this transport-reported value: clear it per
    # attempt so a failed /compact can never reuse a stale boundary from an earlier turn.
    agent._last_native_compaction_prompt_tokens = None
    agent._last_compaction_in_place = False
    agent._last_compression_attempt_in_place = None
    for name, value in (
        ("_last_native_compaction", False), ("_last_summary_dropped_count", 0),
        ("_last_summary_fallback_used", False), ("_last_summary_error", None),
        ("_last_compress_aborted", False), ("_last_compression_made_progress", False),
    ):
        with contextlib.suppress(Exception):
            setattr(compressor, name, value)


def _native_prompt_tokens(result: Any) -> int:
    """Claude's post-boundary context size: newer releases report the last model call separately,
    older ones aggregate turn usage. Either beats estimating Hermes' unpruned mirror."""
    usage = getattr(result, "last_call_usage", None)
    if not isinstance(usage, dict) or not usage:
        usage = getattr(result, "token_usage", None)
    if not isinstance(usage, dict) or not usage:
        return 0
    try:
        return sum(int(usage.get(key) or 0) for key in (
            "input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"))
    except (TypeError, ValueError):
        return 0


def _record_boundary(agent: Any, compressor: Any, approx_tokens: Optional[int]) -> None:
    compressor.compression_count = int(getattr(compressor, "compression_count", 0) or 0) + 1
    compressor._last_compression_made_progress = True
    compressor.last_compression_rough_tokens = int(approx_tokens or 0)
    compressor.last_prompt_tokens = -1
    compressor.last_completion_tokens = 0
    compressor.awaiting_real_usage_after_compression = True
    record = getattr(type(compressor), "record_completed_compaction", None)
    if callable(record):
        record(compressor, used_fallback=False)
    else:
        compressor._verify_compaction_cleared_threshold = True


def _notify_boundary(agent: Any, result: Any, *, defer_notification: bool) -> None:
    from agent.conversation_compression import (
        _notify_context_engine_compression_complete, _queue_context_engine_compression_notification,
    )
    session_id = getattr(agent, "session_id", None) or ""
    if defer_notification:
        _queue_context_engine_compression_notification(agent, new_session_id=session_id, old_session_id=session_id)
    else:
        _notify_context_engine_compression_complete(agent, new_session_id=session_id, old_session_id=session_id)
    memory_manager = getattr(agent, "_memory_manager", None)
    if memory_manager is not None:
        try:
            memory_manager.on_session_switch(session_id, parent_session_id=session_id, reset=False, reason="compression")
        except Exception:
            logger.debug("memory manager on_session_switch (Claude native) failed", exc_info=True)
    callback = getattr(agent, "event_callback", None)
    if callback:
        try:
            callback("session:compress", {
                "platform": getattr(agent, "platform", None) or "", "session_id": session_id,
                "old_session_id": "", "in_place": True, "native": True, "runtime": "claude_cli",
                "native_session_id": getattr(result, "native_session_id", None),
                "compression_count": getattr(agent.context_compressor, "compression_count", 0),
            })
        except Exception:
            logger.debug("event_callback error on Claude native session:compress", exc_info=True)


def compact_via_claude_cli(
    agent: Any, messages: list, system_message: Optional[str], *, force: bool, focus_topic: Optional[str] = None,
    verbatim_tail: Optional[list] = None, approx_tokens: Optional[int] = None, task_id: str = "default",
    trigger: Optional[str] = None, commit_fence: Any = None, attempt: Any = None,
    defer_context_engine_notification: bool = False,
) -> Optional[tuple[list, str]]:
    """Native ``/compact`` for an explicit or idle request; None (skip) for pressure-driven triggers,
    which Claude's own automatic compaction already handles."""
    if not force and str(trigger or "").strip().lower() not in _NATIVE_AUTOMATIC_TRIGGERS:
        logger.info("Claude CLI owns automatic native context compaction; Hermes automatic compression skipped "
                    "(trigger=%s)", trigger or "auto")
        return None
    from agent.conversation_compression import (
        COMPACTION_STATUS, _CompressionActivityHeartbeat, _emit_compaction_done, _existing_system_prompt,
        _pre_compress_memory_context, _reset_read_dedup_caches,
    )
    from agent.conversation_compression_telemetry import _emit_bypassed_attempt_telemetry
    from plugins.claude_cli.runtime import compact_claude_cli_history

    started_at = getattr(attempt, "started_at", None) or time.monotonic()
    attempt_seed = getattr(attempt, "seed", None)

    def _record(commit_status: str, failure_class: Optional[str], method: str = "none") -> None:
        _emit_bypassed_attempt_telemetry(
            agent, started_at, commit_status=commit_status, failure_class=failure_class,
            approx_tokens=approx_tokens, route="claude_cli", method=method, attempt_seed=attempt_seed)

    compressor = agent.context_compressor
    _clear_attempt_state(agent, compressor)
    existing_prompt = _existing_system_prompt(agent, system_message)
    session_id = getattr(agent, "session_id", None) or "none"
    logger.info("Claude native compaction started: session=%s messages=%d tokens=~%s focus=%r trigger=%s",
                session_id, len(messages), f"{approx_tokens:,}" if approx_tokens else "unknown", focus_topic,
                trigger or ("manual" if force else "auto"))
    with contextlib.suppress(Exception):
        agent._emit_status(COMPACTION_STATUS)
    memory_context = _pre_compress_memory_context(agent, messages, False)
    native_focus = _native_focus(focus_topic, _keep_last_exchanges(verbatim_tail), memory_context)

    heartbeat = _CompressionActivityHeartbeat(agent, emit_client_status=True, commit_fence=commit_fence).start()
    commit_entered = False
    try:
        if commit_fence is not None:
            commit_entered = commit_fence.begin_commit(getattr(agent, "_hard_interrupt_requested", None))
            if not commit_entered:
                heartbeat.stop("context compression cancelled")
                _record("aborted", "commit_fence_cancelled")
                return messages, existing_prompt
        result = compact_claude_cli_history(agent, focus_topic=native_focus)
    except BaseException as exc:
        heartbeat.stop("context compression failed")
        with contextlib.suppress(Exception):
            compressor._last_compress_aborted = True
            compressor._last_summary_error = str(exc)
        _record("aborted", f"exception:{type(exc).__name__}")
        raise
    finally:
        if commit_entered and commit_fence is not None:
            commit_fence.finish_commit()

    error = getattr(result, "error", None)
    interrupted = bool(getattr(result, "interrupted", False))
    if error or interrupted or not getattr(result, "compacted", False):
        heartbeat.stop("context compression failed")
        failure = str(error or ("Claude native compaction was interrupted" if interrupted
                                else "Claude did not confirm a compaction boundary"))
        with contextlib.suppress(Exception):
            compressor._last_compress_aborted = True
            compressor._last_summary_error = failure
        with contextlib.suppress(Exception):
            agent._emit_warning(f"⚠ Claude native compaction failed: {failure}")
        _record("failed", "claude_native_compaction_failed")
        return messages, existing_prompt

    prompt_tokens = _native_prompt_tokens(result)
    if prompt_tokens > 0:
        agent._last_native_compaction_prompt_tokens = prompt_tokens
    heartbeat.stop("context compression completed")
    try:
        _record_boundary(agent, compressor, approx_tokens)
    except Exception:
        logger.debug("Claude native compaction bookkeeping failed", exc_info=True)
    _notify_boundary(agent, result, defer_notification=defer_context_engine_notification)
    _reset_read_dedup_caches(task_id, session_id=getattr(agent, "session_id", None) or "")
    agent._last_native_compaction = True
    with contextlib.suppress(Exception):
        compressor._last_native_compaction = True
    _record("committed", None, method="provider")
    _emit_compaction_done(agent)
    logger.info("Claude native compaction done: session=%s native_session=%s metadata=%s", session_id,
                getattr(result, "native_session_id", None) or "", getattr(result, "compaction_metadata", None) or {})
    return messages, existing_prompt
