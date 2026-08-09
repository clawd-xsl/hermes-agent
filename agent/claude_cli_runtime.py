"""AIAgent integration for the persistent Claude CLI runtime."""

from __future__ import annotations

import atexit
import hashlib
import json
import logging
import threading
from typing import Any, Dict, List, Optional

from agent.transports.claude_cli_session import ClaudeCliSession

logger = logging.getLogger(__name__)

_POOL_LOCK = threading.RLock()
_SESSIONS: dict[str, ClaudeCliSession] = {}
_MAX_LIVE_SESSIONS = 8


def _owner_key(agent: Any) -> str:
    session_id = str(getattr(agent, "session_id", "") or "").strip()
    if session_id:
        return session_id
    stable = str(getattr(agent, "_gateway_session_key", "") or "").strip()
    return stable or f"ephemeral:{id(agent)}"


def _tool_fingerprint(agent: Any) -> str:
    payload = json.dumps(
        getattr(agent, "tools", None) or [],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _runtime_config() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        cfg = load_config() or {}
        model_cfg = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
        raw = model_cfg.get("claude_cli") if isinstance(model_cfg, dict) else {}
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def _get_session(agent: Any, *, system_prompt: str) -> ClaudeCliSession:
    from agent.runtime_cwd import resolve_agent_cwd

    owner_key = _owner_key(agent)
    cwd = getattr(agent, "session_cwd", None) or str(resolve_agent_cwd())
    model = str(getattr(agent, "model", "") or "")
    tools_hash = _tool_fingerprint(agent)
    cfg = _runtime_config()
    command = str(cfg.get("command") or "claude")
    try:
        turn_timeout = float(cfg.get("turn_timeout_seconds") or 600)
    except (TypeError, ValueError):
        turn_timeout = 600.0
    turn_timeout = max(30.0, min(turn_timeout, 3600.0))

    with _POOL_LOCK:
        session = _SESSIONS.get(owner_key)
        if session is not None and not session.compatible(
            cwd=cwd,
            model=model,
            tool_fingerprint=tools_hash,
        ):
            _SESSIONS.pop(owner_key, None)
            session.close()
            session = None
        if session is None:
            if len(_SESSIONS) >= _MAX_LIVE_SESSIONS:
                idle = [
                    (key, candidate)
                    for key, candidate in _SESSIONS.items()
                    if not candidate.is_busy
                ]
                if idle:
                    victim_key, victim = min(idle, key=lambda row: row[1].last_used_at)
                    _SESSIONS.pop(victim_key, None)
                    victim.close()
            session = ClaudeCliSession(
                owner_key=owner_key,
                agent=agent,
                cwd=cwd,
                model=model,
                system_prompt=system_prompt,
                command=command,
                turn_timeout=turn_timeout,
            )
            _SESSIONS[owner_key] = session
        agent._claude_cli_session = session
        return session


def _retire_session(agent: Any, session: ClaudeCliSession) -> None:
    with _POOL_LOCK:
        owner_key = _owner_key(agent)
        if _SESSIONS.get(owner_key) is session:
            _SESSIONS.pop(owner_key, None)
    try:
        session.close()
    finally:
        if getattr(agent, "_claude_cli_session", None) is session:
            agent._claude_cli_session = None


def release_claude_cli_session(agent: Any) -> None:
    """Hard-close the native session owned by an AIAgent session boundary."""
    session = getattr(agent, "_claude_cli_session", None)
    if session is None:
        return
    _retire_session(agent, session)


def close_all_claude_cli_sessions() -> None:
    with _POOL_LOCK:
        sessions = list(_SESSIONS.values())
        _SESSIONS.clear()
    for session in sessions:
        try:
            session.close()
        except Exception:
            logger.debug("Claude CLI session cleanup failed", exc_info=True)


atexit.register(close_all_claude_cli_sessions)


def _record_usage(
    agent: Any,
    usage: dict[str, int],
    last_call_usage: Optional[dict[str, int]] = None,
) -> dict[str, Any]:
    uncached = int(usage.get("input_tokens") or 0)
    cache_read = int(usage.get("cache_read_input_tokens") or 0)
    cache_write = int(usage.get("cache_creation_input_tokens") or 0)
    output = int(usage.get("output_tokens") or 0)
    prompt = uncached + cache_read + cache_write
    total = prompt + output
    fields = {
        "session_api_calls": 1,
        "session_prompt_tokens": prompt,
        "session_completion_tokens": output,
        "session_total_tokens": total,
        "session_input_tokens": uncached,
        "session_output_tokens": output,
        "session_cache_read_tokens": cache_read,
        "session_cache_write_tokens": cache_write,
    }
    for name, increment in fields.items():
        setattr(agent, name, int(getattr(agent, name, 0) or 0) + increment)
    compressor_usage = last_call_usage or usage
    last_uncached = int(compressor_usage.get("input_tokens") or 0)
    last_cache_read = int(compressor_usage.get("cache_read_input_tokens") or 0)
    last_cache_write = int(compressor_usage.get("cache_creation_input_tokens") or 0)
    last_output = int(compressor_usage.get("output_tokens") or 0)
    last_prompt = last_uncached + last_cache_read + last_cache_write
    compressor = getattr(agent, "context_compressor", None)
    if compressor is not None:
        compressor.update_from_response(
            {
                "prompt_tokens": last_prompt,
                "completion_tokens": last_output,
                "total_tokens": last_prompt + last_output,
            }
        )

    agent.session_cost_status = "included"
    agent.session_cost_source = "none"
    session_db = getattr(agent, "_session_db", None)
    session_id = getattr(agent, "session_id", None)
    if session_db is not None and session_id:
        try:
            if not getattr(agent, "_session_db_created", False):
                agent._ensure_db_session()
            session_db.update_token_counts(
                session_id,
                input_tokens=uncached,
                output_tokens=output,
                cache_read_tokens=cache_read,
                cache_write_tokens=cache_write,
                model=agent.model,
                cost_status="included",
                cost_source="none",
                billing_provider="anthropic",
                billing_base_url="claude-cli://local",
                billing_mode="subscription_included",
                api_call_count=1,
            )
        except Exception:
            logger.debug("Claude CLI token persistence failed", exc_info=True)

    return {
        "prompt_tokens": prompt,
        "completion_tokens": output,
        "total_tokens": total,
        "input_tokens": uncached,
        "output_tokens": output,
        "cache_read_tokens": cache_read,
        "cache_write_tokens": cache_write,
        "last_prompt_tokens": last_prompt,
        "estimated_cost_usd": getattr(agent, "session_estimated_cost_usd", 0.0),
        "cost_status": "included",
        "cost_source": "none",
    }


def _dedupe_final_projection(
    projected: list[dict[str, Any]], final_text: str
) -> list[dict[str, Any]]:
    if not final_text:
        return projected
    for message in reversed(projected):
        if message.get("role") == "assistant":
            if message.get("content") == final_text and not message.get("tool_calls"):
                return projected
            break
    return [*projected, {"role": "assistant", "content": final_text}]


def run_claude_cli_turn(
    agent: Any,
    *,
    user_message: str,
    original_user_message: Any,
    messages: List[Dict[str, Any]],
    effective_task_id: str,
    active_system_prompt: str,
    should_review_memory: bool = False,
) -> Dict[str, Any]:
    """Run one Hermes turn through the session's persistent Claude child."""
    try:
        agent._reset_stream_delivery_tracking()
        session = _get_session(agent, system_prompt=active_system_prompt)
        turn = session.run_turn(
            agent=agent,
            user_input=user_message,
            messages=messages,
            task_id=effective_task_id,
            stream_callback=getattr(agent, "_fire_stream_delta", None),
        )
    except Exception as exc:
        logger.exception("Claude CLI turn failed")
        session = getattr(agent, "_claude_cli_session", None)
        if session is not None:
            _retire_session(agent, session)
        agent._stream_callback = None
        agent.clear_interrupt()
        error_text = f"Claude CLI turn failed: {exc}"
        messages.append({"role": "assistant", "content": error_text})
        if getattr(agent, "_session_db", None) is not None:
            try:
                agent._flush_messages_to_session_db(messages)
            except Exception:
                logger.debug("Claude CLI startup-error flush failed", exc_info=True)
        return {
            "final_response": error_text,
            "messages": messages,
            "api_calls": 0,
            "completed": False,
            "partial": True,
            "error": str(exc),
            "agent_persisted": True,
        }

    if turn.should_retire:
        _retire_session(agent, session)

    if turn.error and not turn.final_text:
        turn.final_text = f"Claude CLI turn failed: {turn.error}"

    projected = _dedupe_final_projection(turn.projected_messages, turn.final_text)
    if projected:
        messages.extend(projected)
        if getattr(agent, "_session_db", None) is not None:
            try:
                agent._flush_messages_to_session_db(messages)
            except Exception:
                logger.debug("Claude CLI projected-message flush failed", exc_info=True)

    agent._iters_since_skill = (
        getattr(agent, "_iters_since_skill", 0) + turn.tool_iterations
    )

    if not turn.interrupted and turn.error is None:
        try:
            agent._sync_external_memory_for_turn(
                original_user_message=original_user_message,
                final_response=turn.final_text,
                interrupted=False,
                messages=messages,
            )
        except Exception:
            logger.debug("Claude CLI external-memory sync failed", exc_info=True)

    # Do not fork the main Claude subscription session for Hermes' legacy
    # background memory/skill review. Claude's native session and the memory /
    # skill tools already cover this path; a second CLI history would add
    # latency and a second, divergent context owner.

    usage_result = _record_usage(agent, turn.token_usage, turn.last_call_usage)
    try:
        from hermes_cli.plugins import invoke_hook as _invoke_hook

        _invoke_hook(
            "on_session_end",
            session_id=agent.session_id,
            task_id=effective_task_id,
            turn_id=getattr(agent, "_current_turn_id", "") or "",
            completed=not turn.interrupted and turn.error is None,
            interrupted=turn.interrupted,
            model=agent.model,
            platform=getattr(agent, "platform", None) or "",
        )
    except Exception:
        logger.debug("Claude CLI on_session_end hook failed", exc_info=True)
    agent._stream_callback = None
    agent.clear_interrupt()
    return {
        "final_response": turn.final_text,
        "messages": messages,
        "api_calls": 1,
        "completed": not turn.interrupted and turn.error is None,
        "partial": turn.interrupted or turn.error is not None,
        "error": turn.error,
        "agent_persisted": True,
        "claude_session_id": turn.native_session_id,
        "claude_session_reuse": turn.session_reuse,
        "claude_latency_ms": turn.latency_ms,
        **usage_result,
    }


__all__ = [
    "run_claude_cli_turn",
    "release_claude_cli_session",
    "close_all_claude_cli_sessions",
]
