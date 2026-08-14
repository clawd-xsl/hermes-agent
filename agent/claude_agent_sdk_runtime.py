"""AIAgent integration for the persistent Claude Agent SDK runtime."""

from __future__ import annotations

import atexit
import hashlib
import json
import logging
import random
import re
import threading
import time
import uuid
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from agent.transports.claude_agent_sdk_session import (
    ClaudeAgentSdkSession,
    ClaudeAgentSdkTurnResult,
    forget_claude_agent_sdk_binding,
)

logger = logging.getLogger(__name__)

_POOL_LOCK = threading.RLock()
_SESSIONS: dict[str, ClaudeAgentSdkSession] = {}
_MAX_LIVE_SESSIONS = 8
_MAX_NATIVE_TRANSPORT_RETRIES = 1


def _owner_key(agent: Any) -> str:
    # Background review forks deliberately share the foreground Hermes
    # session_id for prompt parity, but must never share its native Claude
    # history. Their harness prompt would otherwise become a durable main-
    # session user turn even though Hermes DB persistence is disabled.
    if getattr(agent, "_persist_disabled", False):
        return f"ephemeral:{id(agent)}"
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
        raw = model_cfg.get("claude_agent_sdk") if isinstance(model_cfg, dict) else {}
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def _native_json_schema(agent: Any) -> Optional[dict[str, Any]]:
    """Translate Hermes/OpenAI structured-output config to Claude's schema.

    Main-agent ``request_overrides`` uses the same OpenAI-shaped payload as
    plugin/auxiliary calls.  Claude Code exposes an equivalent stable
    ``--json-schema`` process option, so this is a supported control rather
    than an API-wire exception.
    """
    overrides = getattr(agent, "request_overrides", None)
    if not isinstance(overrides, dict):
        return None
    response_format = overrides.get("response_format")
    extra_body = overrides.get("extra_body")
    if response_format is None and isinstance(extra_body, dict):
        response_format = extra_body.get("response_format")
    if not isinstance(response_format, dict):
        return None
    format_type = str(response_format.get("type") or "").strip().lower()
    if format_type == "json_object":
        return {"type": "object"}
    if format_type != "json_schema":
        return None
    wrapper = response_format.get("json_schema")
    if not isinstance(wrapper, dict):
        return None
    schema = wrapper.get("schema")
    return dict(schema) if isinstance(schema, dict) else None


def _native_unsupported_request_controls(agent: Any) -> list[str]:
    """Surface API-wire controls the persistent Claude process cannot accept."""
    overrides = getattr(agent, "request_overrides", None)
    control_signature = (
        getattr(agent, "max_tokens", None),
        str(getattr(agent, "service_tier", None) or ""),
        json.dumps(overrides, sort_keys=True, default=str)
        if isinstance(overrides, dict)
        else "",
    )
    if (
        getattr(agent, "_claude_agent_sdk_unsupported_controls_signature", None)
        == control_signature
    ):
        cached = getattr(agent, "_claude_agent_sdk_unsupported_controls", None)
        if isinstance(cached, list):
            return list(cached)
    controls: list[str] = []
    service_tier = str(getattr(agent, "service_tier", None) or "").strip().lower()
    if service_tier and service_tier not in {"fast", "priority"}:
        controls.append("service_tier")
    if isinstance(overrides, dict):
        native_json_schema = _native_json_schema(agent)
        for key in sorted(overrides):
            value = overrides.get(key)
            if key in {"model", "messages", "tools"}:
                continue
            if key == "speed" and str(value or "").strip().lower() == "fast":
                continue
            if (
                key == "service_tier"
                and str(value or "").strip().lower() in {"fast", "priority"}
            ):
                continue
            if key == "response_format" and native_json_schema is not None:
                continue
            if key == "extra_body" and isinstance(value, dict):
                unsupported_extra = [
                    nested_key
                    for nested_key in sorted(value)
                    if not (
                        nested_key == "response_format"
                        and native_json_schema is not None
                    )
                ]
                controls.extend(
                    f"request_overrides.extra_body.{nested_key}"
                    for nested_key in unsupported_extra
                )
                continue
            controls.append(f"request_overrides.{key}")
    controls = list(dict.fromkeys(controls))
    agent._claude_agent_sdk_unsupported_controls_signature = control_signature
    agent._claude_agent_sdk_unsupported_controls = controls
    if controls:
        message = (
            "Persistent Claude Agent SDK cannot forward API-wire controls: "
            + ", ".join(controls)
            + ". Claude low-latency mode is supported through Hermes /fast "
            "or model.claude_agent_sdk.fast_mode."
        )
        logger.warning(message)
        try:
            agent._emit_status("⚠️ " + message)
        except Exception:
            pass
    return controls


def _reasoning_effort(agent: Any) -> Optional[str]:
    """Map Hermes' effort vocabulary onto Claude Code's CLI levels."""
    raw = getattr(agent, "reasoning_config", None)
    if not isinstance(raw, dict) or raw.get("enabled") is False:
        return None
    effort = str(raw.get("effort") or "").strip().lower()
    if not effort or effort == "none":
        return None
    return {
        "minimal": "low",
        "ultra": "max",
    }.get(effort, effort if effort in {"low", "medium", "high", "xhigh", "max"} else None)


def _thinking_mode(agent: Any) -> Optional[str]:
    """Return an explicit Claude Code thinking override when Hermes disables it.

    Omitting ``--effort`` is not enough: Claude Code then uses its own default
    (currently adaptive thinking).  Hermes' ``reasoning_effort: none`` contract
    therefore needs the native ``--thinking disabled`` switch.
    """
    raw = getattr(agent, "reasoning_config", None)
    if not isinstance(raw, dict):
        return None
    if raw.get("enabled") is False:
        return "disabled"
    if str(raw.get("effort") or "").strip().lower() == "none":
        return "disabled"
    return None


def _native_fast_mode(agent: Any, cfg: dict[str, Any]) -> bool:
    """Map Hermes' existing fast-mode state onto Claude Code's native mode."""
    if cfg.get("fast_mode") is True:
        return True
    service_tier = str(getattr(agent, "service_tier", None) or "").strip().lower()
    if service_tier in {"fast", "priority"}:
        return True
    overrides = getattr(agent, "request_overrides", None)
    if not isinstance(overrides, dict):
        return False
    return (
        str(overrides.get("speed") or "").strip().lower() == "fast"
        or str(overrides.get("service_tier") or "").strip().lower()
        in {"fast", "priority"}
    )


def _native_max_output_tokens(agent: Any) -> Optional[int]:
    """Return Hermes' validated response cap for Claude Code."""
    value = getattr(agent, "max_tokens", None)
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _native_api_retry_count(agent: Any) -> Optional[int]:
    """Translate Hermes' total-attempt ceiling to Claude's retry count."""
    value = getattr(agent, "_api_max_retries", None)
    if value is None:
        return None
    try:
        total_attempts = max(1, int(value))
    except (TypeError, ValueError):
        return None
    return total_attempts - 1


def _native_timeouts(
    agent: Any,
    cfg: dict[str, Any],
) -> tuple[float, Optional[float]]:
    """Resolve the whole-turn and inner provider-request deadlines.

    ``model.claude_agent_sdk.turn_timeout_seconds`` bounds Claude Code's complete
    private tool loop.  The provider timeout is the existing cross-transport
    config contract and therefore has to reach Claude Code's ``API_TIMEOUT_MS``
    as well.  When the Claude-specific whole-turn cap is left implicit, grow
    it to leave the configured provider request enough room to finish; an
    explicit whole-turn cap remains authoritative and may deliberately stop a
    multi-step native turn sooner.
    """
    raw_turn_timeout = cfg.get("turn_timeout_seconds")
    explicit_turn_timeout = raw_turn_timeout not in (None, "")
    try:
        turn_timeout = float(raw_turn_timeout or 600)
    except (TypeError, ValueError):
        turn_timeout = 600.0
        explicit_turn_timeout = False
    turn_timeout = max(30.0, min(turn_timeout, 3600.0))

    try:
        from hermes_cli.timeouts import get_provider_request_timeout

        provider_timeout = get_provider_request_timeout(
            str(getattr(agent, "provider", "") or ""),
            str(getattr(agent, "model", "") or ""),
        )
    except Exception:
        provider_timeout = None
    if provider_timeout is not None:
        provider_timeout = float(provider_timeout)
        if not explicit_turn_timeout:
            turn_timeout = max(turn_timeout, provider_timeout + 5.0)
    return turn_timeout, provider_timeout


def _shrink_native_history_images(
    agent: Any,
    messages: list[dict[str, Any]],
    *,
    error: str,
) -> bool:
    """Create API-only shrunken image sidecars for one native retry.

    A rejected image remains in Claude's own history and poisons every later
    request. The standard HTTP loop can mutate a request-local copy; the
    persistent runtime must additionally cold-bootstrap a clean native thread.
    Preserve the original visible attachment in ``content`` and store only the
    provider-safe replacement in Hermes' existing ``api_content`` sidecar.
    """
    try:
        from agent.conversation_compression import (
            try_shrink_image_parts_in_messages,
        )
        from agent.conversation_loop import _image_error_max_dimension

        max_dimension = _image_error_max_dimension(RuntimeError(error)) or 8000
    except Exception:
        logger.debug("Claude image-shrink setup failed", exc_info=True)
        return False

    changed = False
    for message in messages:
        if not isinstance(message, dict):
            continue
        effective = message.get("api_content", message.get("content"))
        if not isinstance(effective, list):
            continue
        wrapper = {"role": message.get("role"), "content": effective}
        if try_shrink_image_parts_in_messages(
            [wrapper],
            max_dimension=max_dimension,
        ):
            message["api_content"] = wrapper["content"]
            changed = True
    return changed


_NATIVE_PREFILL_START = "<hermes_prefill_examples>"


def _native_system_prompt_with_prefill(agent: Any, system_prompt: str) -> str:
    """Keep Hermes' per-call few-shot prefill effective in a native thread.

    Claude's bidirectional stream-json protocol accepts new user inputs but
    cannot insert historical assistant/user rows immediately after the system
    prompt on every request.  A persistent process would otherwise see the
    configured prefill only as quoted bootstrap data on its first turn.  Fold
    the stable examples into the native system prefix once instead: they stay
    at the same cached position for every model iteration without entering the
    visible Hermes transcript.
    """
    base = str(system_prompt or "")
    if _NATIVE_PREFILL_START in base:
        return base
    configured = getattr(agent, "prefill_messages", None) or []
    if not isinstance(configured, list):
        return base

    from agent.message_content import flatten_message_text

    examples: list[dict[str, Any]] = []
    for message in configured:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").strip()
        if role not in {"user", "assistant", "tool"}:
            continue
        content = message.get("content")
        visible = flatten_message_text(content)
        if isinstance(content, list) and any(
            isinstance(part, dict)
            and str(part.get("type") or "").lower()
            in {"image", "image_url", "input_image", "document"}
            for part in content
        ):
            visible = (visible + "\n[non-text example attachment]").strip()
        row: dict[str, Any] = {"role": role, "content": visible}
        if message.get("tool_calls"):
            row["tool_calls"] = message["tool_calls"]
        if message.get("tool_call_id"):
            row["tool_call_id"] = message["tool_call_id"]
        examples.append(row)
    if not examples:
        return base

    block = (
        f"{_NATIVE_PREFILL_START}\n"
        "The following role-tagged messages are stable few-shot examples "
        "configured by Hermes. Apply their behavior to every turn; they are "
        "examples, not new conversation messages.\n"
        + json.dumps(examples, ensure_ascii=False, separators=(",", ":"))
        + "\n</hermes_prefill_examples>"
    )
    return "\n\n".join(part for part in (base.strip(), block) if part)


def _get_session(agent: Any, *, system_prompt: str) -> ClaudeAgentSdkSession:
    from agent.runtime_cwd import resolve_agent_cwd

    system_prompt = _native_system_prompt_with_prefill(agent, system_prompt)
    owner_key = _owner_key(agent)
    cwd = getattr(agent, "session_cwd", None) or str(resolve_agent_cwd())
    model = str(getattr(agent, "model", "") or "")
    tools_hash = _tool_fingerprint(agent)
    reasoning_effort = _reasoning_effort(agent)
    thinking_mode = _thinking_mode(agent)
    persistent_binding = (
        bool(getattr(agent, "_claude_agent_sdk_persistent_binding", True))
        and not getattr(agent, "_persist_disabled", False)
    )
    cfg = _runtime_config()
    command = str(
        getattr(agent, "_claude_agent_sdk_command", None)
        or cfg.get("command")
        or ""
    )
    configured_args = (
        getattr(agent, "_claude_agent_sdk_args", None) or cfg.get("args") or []
    )
    extra_args = [str(arg) for arg in configured_args]
    fast_mode = _native_fast_mode(agent, cfg)
    max_output_tokens = _native_max_output_tokens(agent)
    api_retry_count = _native_api_retry_count(agent)
    auto_compaction_enabled = bool(
        getattr(agent, "compression_enabled", True)
    )
    json_schema = _native_json_schema(agent)
    turn_timeout, provider_request_timeout = _native_timeouts(agent, cfg)

    with _POOL_LOCK:
        session = _SESSIONS.get(owner_key)
        if session is not None and not session.compatible(
            cwd=cwd,
            model=model,
            command=command,
            extra_args=extra_args,
            tool_fingerprint=tools_hash,
            system_prompt=system_prompt,
            reasoning_effort=reasoning_effort,
            thinking_mode=thinking_mode,
            fast_mode=fast_mode,
            max_output_tokens=max_output_tokens,
            api_retry_count=api_retry_count,
            turn_timeout=turn_timeout,
            provider_request_timeout=provider_request_timeout,
            persistent_binding=persistent_binding,
            auto_compaction_enabled=auto_compaction_enabled,
            json_schema=json_schema,
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
            session = ClaudeAgentSdkSession(
                owner_key=owner_key,
                agent=agent,
                cwd=cwd,
                model=model,
                system_prompt=system_prompt,
                command=command,
                extra_args=extra_args,
                turn_timeout=turn_timeout,
                reasoning_effort=reasoning_effort,
                thinking_mode=thinking_mode,
                fast_mode=fast_mode,
                max_output_tokens=max_output_tokens,
                api_retry_count=api_retry_count,
                provider_request_timeout=provider_request_timeout,
                persistent_binding=persistent_binding,
                auto_compaction_enabled=auto_compaction_enabled,
                json_schema=json_schema,
            )
            _SESSIONS[owner_key] = session
        agent._claude_agent_sdk_session = session
        return session


def _new_transient_session(
    agent: Any,
    *,
    system_prompt: str,
    turn_id: str,
    model: Optional[str] = None,
    tool_definitions: Optional[list[dict[str, Any]]] = None,
    json_schema: Optional[dict[str, Any]] = None,
) -> ClaudeAgentSdkSession:
    """Create an unbound one-turn child for request-only context selection."""
    from agent.runtime_cwd import resolve_agent_cwd

    system_prompt = _native_system_prompt_with_prefill(agent, system_prompt)
    cfg = _runtime_config()
    command = str(
        getattr(agent, "_claude_agent_sdk_command", None)
        or cfg.get("command")
        or ""
    )
    configured_args = (
        getattr(agent, "_claude_agent_sdk_args", None) or cfg.get("args") or []
    )
    extra_args = [str(arg) for arg in configured_args]
    fast_mode = _native_fast_mode(agent, cfg)
    turn_timeout, provider_request_timeout = _native_timeouts(agent, cfg)
    return ClaudeAgentSdkSession(
        owner_key=f"transient:{id(agent)}:{turn_id}",
        agent=agent,
        cwd=getattr(agent, "session_cwd", None) or str(resolve_agent_cwd()),
        model=str(model if model is not None else getattr(agent, "model", "") or ""),
        system_prompt=system_prompt,
        command=command,
        extra_args=extra_args,
        turn_timeout=turn_timeout,
        reasoning_effort=_reasoning_effort(agent),
        thinking_mode=_thinking_mode(agent),
        fast_mode=fast_mode,
        max_output_tokens=_native_max_output_tokens(agent),
        api_retry_count=_native_api_retry_count(agent),
        provider_request_timeout=provider_request_timeout,
        persistent_binding=False,
        tool_definitions=tool_definitions,
        auto_compaction_enabled=bool(
            getattr(agent, "compression_enabled", True)
        ),
        json_schema=json_schema,
    )


def run_claude_agent_sdk_auxiliary_completion(
    *,
    messages: list[dict[str, Any]],
    model: str,
    timeout: Optional[float] = None,
    max_tokens: Optional[int] = None,
    reasoning_config: Optional[dict[str, Any]] = None,
    tools: Optional[list[dict[str, Any]]] = None,
    tool_choice: Any = None,
    json_schema: Optional[dict[str, Any]] = None,
) -> Any:
    """Run one isolated auxiliary completion via Claude Code auth.

    The persistent main Claude process cannot be reused here: auxiliary calls
    routinely have different system prompts and selected histories, and
    injecting them into the main native thread would corrupt both its context
    and prompt-cache prefix.  A transient child preserves keychain/subscription
    auth while keeping the authoritative main session untouched.
    """
    if not isinstance(messages, list) or not messages:
        raise ValueError("Claude Agent SDK auxiliary completion requires messages")

    system_parts: list[str] = []
    conversation: list[dict[str, Any]] = []
    for row in messages:
        if not isinstance(row, dict):
            continue
        if row.get("role") == "system":
            content = row.get("content")
            if isinstance(content, str) and content.strip():
                system_parts.append(content.strip())
            continue
        conversation.append(dict(row))

    if not conversation or conversation[-1].get("role") != "user":
        raise ValueError(
            "Claude Agent SDK auxiliary completion requires a final user message"
        )

    tool_definitions = [
        dict(tool) for tool in (tools or []) if isinstance(tool, dict)
    ]
    if tool_choice == "none":
        tool_definitions = []

    system_prompt = "\n\n".join(system_parts).strip() or (
        "You are a helpful AI assistant. Follow the user's instructions "
        "precisely and return only the requested result."
    )
    if tool_definitions:
        required_name = ""
        if isinstance(tool_choice, dict):
            function = tool_choice.get("function")
            if isinstance(function, dict):
                required_name = str(function.get("name") or "").strip()
        if required_name:
            system_prompt += (
                "\n\nYou must respond by proposing a call to the tool named "
                f"{required_name}. Do not replace it with prose."
            )
        elif tool_choice == "required":
            system_prompt += (
                "\n\nYou must respond by proposing at least one of the "
                "available tools. Do not replace the tool call with prose."
            )
    current_user = conversation[-1].get(
        "api_content", conversation[-1].get("content", "")
    )
    turn_id = f"aux:{uuid.uuid4().hex}"
    agent = SimpleNamespace(
        model=str(model or ""),
        max_tokens=max_tokens,
        tools=tool_definitions,
        reasoning_config=(
            dict(reasoning_config) if isinstance(reasoning_config, dict) else None
        ),
        session_cwd=None,
        prefill_messages=[],
        iteration_budget=None,
        _api_max_retries=None,
        step_callback=None,
        _checkpoint_mgr=None,
        compression_enabled=True,
    )
    session = _new_transient_session(
        agent,
        system_prompt=system_prompt,
        turn_id=turn_id,
        model=str(model or ""),
        tool_definitions=tool_definitions,
        json_schema=json_schema,
    )
    if timeout is not None:
        try:
            session.turn_timeout = max(1.0, float(timeout))
            # The auxiliary OpenAI-shaped client exposes one timeout knob.
            # Preserve its historical contract for both the host wait and the
            # child request rather than allowing provider config to override a
            # more specific call-site deadline.
            session.provider_request_timeout = max(
                1.0,
                session.turn_timeout - 5.0,
            )
        except (TypeError, ValueError):
            pass
    try:
        if tool_definitions:
            result = session.propose_tools(
                agent=agent,
                messages=conversation[:-1],
                prompt=current_user,
            )
        else:
            result = session.summarize(
                agent=agent,
                messages=conversation[:-1],
                prompt=current_user,
            )
    finally:
        session.close()

    if getattr(result, "interrupted", False):
        raise RuntimeError("Claude Agent SDK auxiliary completion was interrupted")
    error = str(getattr(result, "error", "") or "").strip()
    if error:
        raise RuntimeError(error)
    has_tool_calls = any(
        isinstance(row, dict)
        and row.get("role") == "assistant"
        and bool(row.get("tool_calls"))
        for row in (getattr(result, "projected_messages", None) or [])
    )
    if (
        not has_tool_calls
        and not str(getattr(result, "final_text", "") or "").strip()
    ):
        raise RuntimeError("Claude Agent SDK auxiliary completion returned empty output")
    return result


def _retire_session(agent: Any, session: ClaudeAgentSdkSession) -> None:
    with _POOL_LOCK:
        owner_key = _owner_key(agent)
        if _SESSIONS.get(owner_key) is session:
            _SESSIONS.pop(owner_key, None)
    try:
        session.close()
    finally:
        if getattr(agent, "_claude_agent_sdk_session", None) is session:
            agent._claude_agent_sdk_session = None


def _invalidate_persistent_native_history(
    agent: Any, previous_session: Optional[ClaudeAgentSdkSession]
) -> None:
    """Drop a warm thread that missed an isolated selected-context turn."""
    owner_key = _owner_key(agent)
    if previous_session is not None:
        _retire_session(agent, previous_session)
    forget_claude_agent_sdk_binding(owner_key)
    if getattr(agent, "_claude_agent_sdk_session", None) is previous_session:
        agent._claude_agent_sdk_session = None


def release_claude_agent_sdk_session(agent: Any) -> None:
    """Hard-close the native session owned by an AIAgent session boundary."""
    session = getattr(agent, "_claude_agent_sdk_session", None)
    if session is None:
        return
    _retire_session(agent, session)


def detach_claude_agent_sdk_session(agent: Any) -> None:
    """Leave a pooled native thread alive while cleaning up a helper agent.

    Gateway compression helpers temporarily bind the real conversation's
    Claude thread so they can issue native ``/compact``.  Their AIAgent object
    is disposable, but the just-compacted process is not: the main agent built
    immediately afterwards should rebind and reuse it.  Compatibility is still
    rechecked by :func:`_get_session`; this only prevents helper ``close()``
    from treating the shared native thread as its own hard boundary.
    """
    if getattr(agent, "_claude_agent_sdk_session", None) is not None:
        agent._claude_agent_sdk_session = None


def close_all_claude_agent_sdk_sessions() -> None:
    with _POOL_LOCK:
        sessions = list(_SESSIONS.values())
        _SESSIONS.clear()
    for session in sessions:
        try:
            session.close()
        except Exception:
            logger.debug("Claude Agent SDK session cleanup failed", exc_info=True)


def compact_claude_agent_sdk_history(
    agent: Any,
    *,
    focus_topic: Optional[str] = None,
) -> Any:
    """Run Claude Code's native ``/compact`` on this Hermes session.

    The native thread, rather than Hermes' OpenAI-style mirror, is the context
    actually sent to Claude.  Manual compression therefore has to cross this
    transport boundary; rewriting the mirror would report success without
    reducing the live context.
    """
    system_prompt = getattr(agent, "_cached_system_prompt", None)
    if not system_prompt:
        system_prompt = agent._build_system_prompt(None)
    ephemeral = str(getattr(agent, "ephemeral_system_prompt", "") or "").strip()
    if ephemeral:
        system_prompt = (str(system_prompt or "") + "\n\n" + ephemeral).strip()

    session = _get_session(agent, system_prompt=str(system_prompt or ""))
    try:
        result = session.compact(agent=agent, focus_topic=focus_topic)
    except Exception:
        _invalidate_persistent_native_history(agent, session)
        raise

    if getattr(result, "should_retire", False):
        _invalidate_persistent_native_history(agent, session)
    usage = getattr(result, "token_usage", None)
    if isinstance(usage, dict) and usage:
        _record_usage(
            agent,
            usage,
            getattr(result, "last_call_usage", None),
            api_call_count=max(1, int(getattr(result, "model_iterations", 0) or 0)),
        )
    return result


atexit.register(close_all_claude_agent_sdk_sessions)


def _normalize_native_usage(usage: Optional[dict[str, int]]) -> dict[str, int]:
    """Normalize one Claude usage record to the ContextEngine contract."""
    raw = usage if isinstance(usage, dict) else {}
    uncached = int(raw.get("input_tokens") or 0)
    cache_read = int(raw.get("cache_read_input_tokens") or 0)
    cache_write = int(raw.get("cache_creation_input_tokens") or 0)
    output = int(raw.get("output_tokens") or 0)
    prompt = uncached + cache_read + cache_write
    return {
        "prompt_tokens": prompt,
        "completion_tokens": output,
        "total_tokens": prompt + output,
        "input_tokens": uncached,
        "output_tokens": output,
        "cache_read_tokens": cache_read,
        "cache_write_tokens": cache_write,
        "reasoning_tokens": 0,
    }


def _record_usage(
    agent: Any,
    usage: dict[str, int],
    last_call_usage: Optional[dict[str, int]] = None,
    *,
    api_call_count: int = 1,
    update_context_engine: bool = True,
) -> dict[str, Any]:
    uncached = int(usage.get("input_tokens") or 0)
    cache_read = int(usage.get("cache_read_input_tokens") or 0)
    cache_write = int(usage.get("cache_creation_input_tokens") or 0)
    output = int(usage.get("output_tokens") or 0)
    prompt = uncached + cache_read + cache_write
    total = prompt + output
    fields = {
        "session_api_calls": api_call_count,
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
    last_usage = _normalize_native_usage(last_call_usage or usage)
    last_prompt = last_usage["prompt_tokens"]
    compressor = getattr(agent, "context_compressor", None)
    if update_context_engine and compressor is not None:
        compressor.update_from_response(last_usage)

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
                billing_base_url="claude-agent-sdk://local",
                billing_mode="subscription_included",
                api_call_count=api_call_count,
            )
        except Exception:
            logger.debug("Claude Agent SDK token persistence failed", exc_info=True)

    return {
        "prompt_tokens": prompt,
        "completion_tokens": output,
        "total_tokens": total,
        "input_tokens": uncached,
        "output_tokens": output,
        "cache_read_tokens": cache_read,
        "cache_write_tokens": cache_write,
        "last_prompt_tokens": last_prompt,
        "last_usage": last_usage,
        "estimated_cost_usd": getattr(agent, "session_estimated_cost_usd", 0.0),
        "cost_status": "included",
        "cost_source": "none",
    }


def _advance_skill_review_cadence(agent: Any, iterations: int) -> None:
    """Match the standard loop's conditional per-provider-step counter."""
    if (
        iterations > 0
        and getattr(agent, "_skill_nudge_interval", 0) > 0
        and "skill_manage" in getattr(agent, "valid_tool_names", set())
    ):
        agent._iters_since_skill = (
            getattr(agent, "_iters_since_skill", 0) + iterations
        )


_PENDING_NATIVE_COMPACTION_CONTEXT = "_claude_agent_sdk_pending_compaction_context"
_STAMPED_NATIVE_COMPACTION_CONTEXT = "_claude_agent_sdk_compaction_context"


def _prepend_native_compaction_context(content: Any, context: str) -> Any:
    """Attach one-shot provider continuity without dirtying visible history."""
    context = str(context or "").strip()
    if not context:
        return content
    prefix = (
        "[Hermes continuity checkpoint recovered at Claude's last automatic "
        "compaction boundary. Preserve and use it as prior context:]\n"
        f"{context}\n\n"
    )
    if isinstance(content, list):
        return [{"type": "text", "text": prefix}, *content]
    if content is None:
        return prefix.rstrip()
    return prefix + str(content)


def stamp_native_compaction_context(
    message: Dict[str, Any],
    context: str,
) -> Any:
    """Stamp post-compaction continuity onto the durable wire sidecar.

    Turn setup calls this before its crash-resilience persist, while recursive
    native continuations call it from ``run_claude_agent_sdk_turn``.  The private
    marker prevents the latter from prepending the same checkpoint twice.
    Visible ``content`` remains clean; ``api_content`` records exactly what the
    native thread received so a cold bootstrap can reproduce the prefix.
    """
    normalized = str(context or "").strip()
    effective = message.get("api_content", message.get("content"))
    if not normalized:
        return effective
    if message.get(_STAMPED_NATIVE_COMPACTION_CONTEXT) == normalized:
        return effective
    stamped = _prepend_native_compaction_context(effective, normalized)
    message["api_content"] = stamped
    message[_STAMPED_NATIVE_COMPACTION_CONTEXT] = normalized
    return stamped


def _record_native_auto_compaction(
    agent: Any,
    turn: Any,
    *,
    messages: list[dict[str, Any]],
    task_id: str,
) -> bool:
    """Bridge a Claude-owned automatic compaction into Hermes lifecycle.

    Claude emits the boundary only after it has completed its native summary,
    so a memory provider's ``on_pre_compress`` return value cannot be inserted
    retroactively.  We still run the hook against Hermes' lossless transcript
    (preserving extraction side effects) and carry any returned continuity text
    into the next native user input exactly once.
    """
    if not bool(getattr(turn, "compacted", False)):
        return False

    boundary_count = max(1, int(getattr(turn, "compaction_count", 0) or 0))
    session_id = str(getattr(agent, "session_id", "") or "")
    memory_manager = getattr(agent, "_memory_manager", None)
    if memory_manager is not None:
        try:
            from agent.conversation_compression import sanitize_memory_context

            continuity = memory_manager.on_pre_compress(messages)
            if isinstance(continuity, str):
                continuity = sanitize_memory_context(continuity).strip()
                if continuity:
                    existing = str(
                        getattr(agent, _PENDING_NATIVE_COMPACTION_CONTEXT, "") or ""
                    ).strip()
                    setattr(
                        agent,
                        _PENDING_NATIVE_COMPACTION_CONTEXT,
                        "\n\n".join(part for part in (existing, continuity) if part),
                    )
        except Exception:
            logger.debug(
                "memory manager on_pre_compress (Claude automatic) failed",
                exc_info=True,
            )

    compressor = getattr(agent, "context_compressor", None)
    if compressor is not None:
        try:
            compressor.compression_count = int(
                getattr(compressor, "compression_count", 0) or 0
            ) + boundary_count
            compressor._last_compression_made_progress = True
            metadata = getattr(turn, "compaction_metadata", None) or {}
            rough_tokens = (
                metadata.get("preTokens")
                or metadata.get("pre_tokens")
                or metadata.get("input_tokens")
                or 0
            )
            compressor.last_compression_rough_tokens = int(rough_tokens or 0)
            compressor.last_prompt_tokens = -1
            compressor.last_completion_tokens = 0
            compressor.awaiting_real_usage_after_compression = True
            record_boundary = getattr(
                type(compressor), "record_completed_compaction", None
            )
            if callable(record_boundary):
                for _ in range(boundary_count):
                    record_boundary(compressor, used_fallback=False)
            else:
                compressor._verify_compaction_cleared_threshold = True
            compressor._last_native_compaction = True
        except Exception:
            logger.debug(
                "Claude automatic compaction bookkeeping failed", exc_info=True
            )

    try:
        from agent.conversation_compression import (
            COMPACTION_STATUS,
            _emit_compaction_done,
            _notify_context_engine_compression_complete,
        )

        agent._emit_status(COMPACTION_STATUS)
        # The boundary is reported only after Claude has already completed its
        # private summary.  Emit both edges back-to-back so gateway/TUI state
        # cannot remain stuck in ``compacting`` until some unrelated status.
        _emit_compaction_done(agent)
        _notify_context_engine_compression_complete(
            agent,
            new_session_id=session_id,
            old_session_id=session_id,
        )
    except Exception:
        logger.debug(
            "Claude automatic context-engine compaction notification failed",
            exc_info=True,
        )

    if memory_manager is not None:
        try:
            memory_manager.on_session_switch(
                session_id,
                parent_session_id=session_id,
                reset=False,
                reason="compression",
            )
        except Exception:
            logger.debug(
                "memory manager on_session_switch (Claude automatic) failed",
                exc_info=True,
            )

    if getattr(agent, "event_callback", None):
        try:
            agent.event_callback(
                "session:compress",
                {
                    "platform": getattr(agent, "platform", None) or "",
                    "session_id": session_id,
                    "old_session_id": "",
                    "in_place": True,
                    "native": True,
                    "automatic": True,
                    "runtime": "claude_agent_sdk",
                    "native_session_id": getattr(turn, "native_session_id", None),
                    "compression_count": getattr(compressor, "compression_count", 0),
                    "boundary_count": boundary_count,
                },
            )
        except Exception:
            logger.debug(
                "event_callback error on Claude automatic session:compress",
                exc_info=True,
            )

    for module_name, reset_name in (
        ("tools.file_tools", "reset_file_dedup"),
        ("tools.skills_tool", "reset_skill_view_dedup"),
    ):
        try:
            module = __import__(module_name, fromlist=[reset_name])
            getattr(module, reset_name)(task_id)
        except Exception:
            pass

    agent._last_native_compaction = True
    agent._last_compaction_in_place = False
    logger.info(
        "Claude automatic compaction observed: session=%s native_session=%s "
        "boundaries=%d metadata=%s",
        session_id or "none",
        getattr(turn, "native_session_id", None) or "",
        boundary_count,
        getattr(turn, "compaction_metadata", None) or {},
    )
    return True


def _dedupe_final_projection(
    projected: list[dict[str, Any]],
    final_text: str,
    *,
    replace_terminal_assistant: bool = False,
) -> list[dict[str, Any]]:
    if not final_text:
        return projected
    if replace_terminal_assistant:
        # A controlled host-side finalizer (currently the iteration-limit
        # summary) supersedes Claude's unfinished terminal prose.  Keep the
        # crash-durable tool protocol, but do not merge a provisional
        # ``working`` answer into the final summary merely to repair adjacent
        # assistant roles.
        projected = list(projected)
        while projected:
            message = projected[-1]
            if (
                not isinstance(message, dict)
                or message.get("role") != "assistant"
                or message.get("tool_calls")
            ):
                break
            projected.pop()
    for message in reversed(projected):
        if message.get("role") == "assistant":
            if message.get("content") == final_text and not message.get("tool_calls"):
                return projected
            break
    return [*projected, {"role": "assistant", "content": final_text}]


def _native_turn_signature(turn: Any) -> str:
    """Stable subset used to detect execution-middleware history divergence."""
    payload = {
        "final_text": getattr(turn, "final_text", None),
        "projected_messages": getattr(turn, "projected_messages", None),
        "error": getattr(turn, "error", None),
        "error_category": getattr(turn, "error_category", None),
        "error_status": getattr(turn, "error_status", None),
        "terminal_result_received": getattr(
            turn, "terminal_result_received", None
        ),
        "interrupted": getattr(turn, "interrupted", None),
        # These fields decide whether the native thread may be resumed and
        # how Hermes finalizes the turn.  Execution middleware that drops or
        # rewrites a host stop must invalidate the physical history just like
        # a response-text rewrite; otherwise a killed/partial Claude thread
        # can be treated as a faithful warm prefix on the next user turn.
        "host_stop_reason": getattr(turn, "host_stop_reason", None),
        "should_retire": getattr(turn, "should_retire", None),
        "budget_exhausted": getattr(turn, "budget_exhausted", None),
        "last_stop_reason": getattr(turn, "last_stop_reason", None),
        "native_session_id": getattr(turn, "native_session_id", None),
        "compacted": getattr(turn, "compacted", None),
        "compaction_count": getattr(turn, "compaction_count", None),
    }
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _coerce_native_turn_result(value: Any) -> ClaudeAgentSdkTurnResult:
    """Normalize Relay/middleware JSON projections back to the native result."""
    if isinstance(value, ClaudeAgentSdkTurnResult):
        return value
    try:
        from agent.relay_llm import _jsonable

        payload = _jsonable(value)
    except Exception:
        payload = None
    if not isinstance(payload, dict) or "projected_messages" not in payload:
        raise TypeError(
            "Claude LLM execution middleware must return ClaudeAgentSdkTurnResult"
        )
    allowed = ClaudeAgentSdkTurnResult.__dataclass_fields__
    return ClaudeAgentSdkTurnResult(
        **{key: item for key, item in payload.items() if key in allowed}
    )


def _native_request_messages(
    agent: Any,
    messages: list[dict[str, Any]],
    *,
    system_prompt: str,
) -> list[dict[str, Any]]:
    """Build the JSON-shaped request view used by native observer hooks."""
    request_messages: list[dict[str, Any]] = []
    if system_prompt:
        request_messages.append({"role": "system", "content": system_prompt})
    for message in messages:
        if not isinstance(message, dict):
            continue
        row = dict(message)
        sidecar = row.pop("api_content", None)
        if sidecar is not None and row.get("role") in {"user", "assistant"}:
            row["content"] = sidecar
        for key in tuple(row):
            if key.startswith("_") or key in {"display_kind", "display_metadata"}:
                row.pop(key, None)
        request_messages.append(row)
    return request_messages


def _native_moa_guidance(
    agent: Any,
    request_messages: list[dict[str, Any]],
    *,
    original_user_message: Any,
    moa_config: Optional[dict[str, Any]],
) -> str:
    """Run the configured one-shot advisor fan-out for one model step."""
    if not moa_config:
        return ""
    try:
        from agent.message_content import flatten_message_text
        from agent.moa_loop import _preset_temperature, aggregate_moa_context

        return aggregate_moa_context(
            user_prompt=(
                original_user_message
                if isinstance(original_user_message, str)
                else flatten_message_text(original_user_message)
            ),
            api_messages=request_messages,
            reference_models=moa_config.get("reference_models") or [],
            aggregator=moa_config.get("aggregator") or {},
            temperature=_preset_temperature(moa_config, "reference_temperature"),
            aggregator_temperature=_preset_temperature(
                moa_config, "aggregator_temperature"
            ),
            reference_max_tokens=moa_config.get("reference_max_tokens"),
            reference_timeout=(
                float(moa_config["reference_timeout"])
                if moa_config.get("reference_timeout")
                else None
            ),
            degraded_reference_policy=str(
                moa_config.get("degraded_reference_policy") or "loud"
            ),
            agent=agent,
        )
    except Exception:
        logger.warning("Claude native MoA context aggregation failed", exc_info=True)
        return ""


def _append_private_context(content: Any, guidance: str) -> Any:
    """Append request-only guidance without mutating the source content."""
    if isinstance(content, str):
        return content + "\n\n" + guidance
    if isinstance(content, list):
        return [*content, {"type": "text", "text": "\n\n" + guidance}]
    return (str(content or "") + "\n\n" + guidance).strip()


def _attach_native_moa_context(
    agent: Any,
    request_messages: list[dict[str, Any]],
    *,
    original_user_message: Any,
    moa_config: Optional[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Run Hermes' one-shot MoA fan-out and attach its private guidance."""
    guidance = _native_moa_guidance(
        agent,
        request_messages,
        original_user_message=original_user_message,
        moa_config=moa_config,
    )
    if not guidance:
        return request_messages

    rows = [dict(row) for row in request_messages]
    for row in reversed(rows):
        if row.get("role") != "user":
            continue
        row["content"] = _append_private_context(row.get("content", ""), guidance)
        break
    return rows


def _select_native_request_context(
    agent: Any,
    request_messages: list[dict[str, Any]],
    conversation_messages: list[dict[str, Any]],
    *,
    current_turn_user_idx: int,
) -> tuple[list[dict[str, Any]], bool]:
    """Apply ContextEngine selection and report whether bytes changed.

    A changed request cannot be sent through an already-warm native history:
    Claude would still retain every message the engine removed. The caller
    therefore uses an isolated one-turn child only for changed selections.
    """
    incoming = (
        conversation_messages[current_turn_user_idx]
        if 0 <= current_turn_user_idx < len(conversation_messages)
        else None
    )
    try:
        from agent.conversation_loop import _apply_context_engine_selection

        selected = _apply_context_engine_selection(
            agent,
            request_messages,
            conversation_messages,
            incoming,
            logger=logger,
        )
    except Exception:
        logger.warning("Claude native context selection failed open", exc_info=True)
        return request_messages, False
    return selected, selected != request_messages


def _apply_native_request_middleware(
    agent: Any,
    *,
    request_messages: list[dict[str, Any]],
    effective_task_id: str,
    turn_id: str,
    api_request_id: str,
    api_call_count: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Apply the standard LLM request middleware to the native request view."""
    request = {
        "model": agent.model,
        "messages": request_messages,
        "tools": agent.tools or [],
        "transport": "claude_agent_sdk",
    }
    try:
        from hermes_cli.middleware import apply_llm_request_middleware

        result = apply_llm_request_middleware(
            request,
            task_id=effective_task_id,
            turn_id=turn_id,
            api_request_id=api_request_id,
            session_id=agent.session_id or "",
            platform=agent.platform or "",
            model=agent.model,
            provider=agent.provider,
            base_url="claude-agent-sdk://local",
            api_mode=agent.api_mode,
            api_call_count=api_call_count,
        )
        return result.payload, list(result.trace)
    except Exception:
        logger.warning("Claude native request middleware failed open", exc_info=True)
        return request, []


def _invoke_native_pre_request_hook(
    agent: Any,
    *,
    request_messages: list[dict[str, Any]],
    original_user_message: Any,
    messages: list[dict[str, Any]],
    effective_task_id: str,
    turn_id: str,
    api_request_id: str,
    started_at: float,
    middleware_trace: list[dict[str, Any]],
    request_model: Optional[str] = None,
    request_tools: Optional[list[dict[str, Any]]] = None,
    api_call_count: int = 1,
) -> None:
    try:
        from hermes_cli.lifecycle import has_hook, invoke_hook

        if not has_hook("pre_api_request"):
            return
        hook_model = request_model or agent.model
        hook_tools = request_tools if request_tools is not None else (agent.tools or [])
        request_kwargs = {
            "model": hook_model,
            "messages": request_messages,
            "tools": hook_tools,
            "transport": "claude_agent_sdk",
        }
        request_chars = sum(
            len(str(message.get("content") or "")) for message in request_messages
        )
        invoke_hook(
            "pre_api_request",
            task_id=effective_task_id,
            turn_id=turn_id,
            api_request_id=api_request_id,
            session_id=agent.session_id or "",
            user_message=original_user_message,
            conversation_history=list(messages),
            platform=agent.platform or "",
            model=hook_model,
            provider=agent.provider,
            base_url="claude-agent-sdk://local",
            api_mode=agent.api_mode,
            api_call_count=api_call_count,
            retry_count=0,
            request_messages=list(request_messages),
            message_count=len(request_messages),
            tool_count=len(hook_tools),
            approx_input_tokens=max(0, request_chars // 4),
            request_char_count=request_chars,
            max_tokens=agent.max_tokens,
            started_at=started_at,
            middleware_trace=list(middleware_trace),
            request=agent._api_request_payload_for_hook(request_kwargs),
        )
    except Exception:
        logger.debug("Claude Agent SDK pre_api_request hook failed", exc_info=True)


def _invoke_native_post_request_hook(
    agent: Any,
    *,
    turn: Any,
    request_messages: list[dict[str, Any]],
    effective_task_id: str,
    turn_id: str,
    api_request_id: str,
    started_at: float,
    ended_at: float,
    request_model: Optional[str] = None,
    api_call_count: int = 1,
) -> None:
    try:
        from hermes_cli.lifecycle import has_hook, invoke_hook

        if not has_hook("post_api_request"):
            return
        hook_model = request_model or agent.model
        usage = {
            "prompt_tokens": int(turn.token_usage.get("input_tokens") or 0)
            + int(turn.token_usage.get("cache_read_input_tokens") or 0)
            + int(turn.token_usage.get("cache_creation_input_tokens") or 0),
            "completion_tokens": int(turn.token_usage.get("output_tokens") or 0),
        }
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
        assistant_message = {
            "role": "assistant",
            "content": turn.final_text,
            "tool_calls": [
                call
                for message in turn.projected_messages
                for call in (message.get("tool_calls") or [])
            ],
        }
        finish_reason = _native_finish_reason(
            getattr(turn, "last_stop_reason", None),
            has_tool_calls=bool(assistant_message["tool_calls"]),
            error=bool(turn.error),
        )
        response = agent._sanitize_hook_payload(
            {
                "model": hook_model,
                "finish_reason": finish_reason,
                "assistant_message": assistant_message,
                "usage": usage,
                "native_session_id": turn.native_session_id,
            }
        )
        invoke_hook(
            "post_api_request",
            task_id=effective_task_id,
            turn_id=turn_id,
            api_request_id=api_request_id,
            session_id=agent.session_id or "",
            platform=agent.platform or "",
            model=hook_model,
            provider=agent.provider,
            base_url="claude-agent-sdk://local",
            api_mode=agent.api_mode,
            api_call_count=api_call_count,
            api_duration=max(0.0, ended_at - started_at),
            started_at=started_at,
            ended_at=ended_at,
            finish_reason=finish_reason,
            message_count=len(request_messages),
            response_model=hook_model,
            response=response,
            usage=usage,
            assistant_message=assistant_message,
            assistant_content_chars=len(turn.final_text or ""),
            assistant_tool_call_count=len(assistant_message["tool_calls"]),
        )
    except Exception:
        logger.debug("Claude Agent SDK post_api_request hook failed", exc_info=True)


def _native_finish_reason(
    stop_reason: Any,
    *,
    has_tool_calls: bool = False,
    error: bool = False,
) -> str:
    """Map Claude wire stop reasons onto Hermes/OpenAI observer semantics."""
    if error:
        return "error"
    if has_tool_calls:
        return "tool_calls"
    normalized = str(stop_reason or "").strip().lower()
    return {
        "": "stop",
        "end_turn": "stop",
        "stop_sequence": "stop",
        "tool_use": "tool_calls",
        "max_tokens": "length",
        "model_context_window_exceeded": "length",
        "refusal": "content_filter",
    }.get(normalized, normalized)


def _invoke_native_post_iteration_hook(
    agent: Any,
    *,
    assistant_message: dict[str, Any],
    usage: dict[str, int],
    effective_task_id: str,
    turn_id: str,
    api_request_id: str,
    api_call_count: int,
    started_at: float,
    ended_at: float,
    request_model: Optional[str] = None,
) -> None:
    """Emit standard progress + observer events for one Claude response."""
    assistant_text = str(assistant_message.get("content") or "").strip()
    progress_callback = getattr(agent, "tool_progress_callback", None)
    if assistant_text and callable(progress_callback):
        progress_text = re.sub(
            r"</?(?:REASONING_SCRATCHPAD|think|reasoning)>",
            "",
            assistant_text,
            flags=re.IGNORECASE,
        ).strip()
        first_line = progress_text.split("\n", 1)[0][:80] if progress_text else ""
        try:
            if first_line and getattr(agent, "_delegate_depth", 0) > 0:
                progress_callback("_thinking", first_line)
            elif progress_text:
                progress_callback(
                    "reasoning.available",
                    "_thinking",
                    progress_text[:500],
                    None,
                )
        except Exception:
            pass

    try:
        from hermes_cli.lifecycle import has_hook, invoke_hook

        if not has_hook("post_api_request"):
            return
        hook_model = request_model or agent.model
        prompt_tokens = sum(
            int(usage.get(key) or 0)
            for key in (
                "input_tokens",
                "cache_read_input_tokens",
                "cache_creation_input_tokens",
            )
        )
        completion_tokens = int(usage.get("output_tokens") or 0)
        normalized_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        finish_reason = _native_finish_reason(
            assistant_message.get("finish_reason"),
            has_tool_calls=bool(assistant_message.get("tool_calls")),
        )
        response = agent._sanitize_hook_payload(
            {
                "model": hook_model,
                "finish_reason": finish_reason,
                "assistant_message": assistant_message,
                "usage": normalized_usage,
                "native_session_id": getattr(
                    getattr(agent, "_claude_agent_sdk_session", None),
                    "native_session_id",
                    None,
                ),
            }
        )
        invoke_hook(
            "post_api_request",
            task_id=effective_task_id,
            turn_id=turn_id,
            api_request_id=api_request_id,
            session_id=agent.session_id or "",
            platform=agent.platform or "",
            model=hook_model,
            provider=agent.provider,
            base_url="claude-agent-sdk://local",
            api_mode=agent.api_mode,
            api_call_count=api_call_count,
            api_duration=max(0.0, ended_at - started_at),
            started_at=started_at,
            ended_at=ended_at,
            finish_reason=finish_reason,
            response_model=hook_model,
            response=response,
            usage=normalized_usage,
            assistant_message=assistant_message,
            assistant_content_chars=len(
                str(assistant_message.get("content") or "")
            ),
            assistant_tool_call_count=len(
                assistant_message.get("tool_calls") or []
            ),
        )
    except Exception:
        logger.debug(
            "Claude Agent SDK per-iteration post_api_request hook failed",
            exc_info=True,
        )


def _last_native_assistant(messages: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "assistant":
            return message
    return None


def _native_empty_retry_wait(agent: Any, attempt: int) -> bool:
    """Wait between empty native replies while remaining interruptible."""
    from agent.retry_utils import jittered_backoff

    wait_time = jittered_backoff(attempt, base_delay=5.0, max_delay=60.0)
    agent._buffer_status(
        f"⚠️ Empty response from Claude Agent SDK — retrying "
        f"({attempt}/3) in {wait_time:.0f}s"
    )
    deadline = time.time() + wait_time
    while time.time() < deadline:
        if getattr(agent, "_interrupt_requested", False):
            return False
        time.sleep(min(0.2, max(0.0, deadline - time.time())))
    return True


def _native_housekeeping_fallback(
    agent: Any,
    messages: list[dict[str, Any]],
    *,
    current_turn_user_idx: int,
) -> Optional[str]:
    """Recover visible text delivered alongside housekeeping tool calls."""
    housekeeping = {"memory", "todo", "skill_manage", "session_search"}
    start = max(-1, int(current_turn_user_idx)) + 1
    for row in reversed(messages[start:]):
        if not isinstance(row, dict) or row.get("role") != "assistant":
            continue
        tool_calls = row.get("tool_calls") or []
        if not tool_calls:
            continue
        names = {
            str((call.get("function") or {}).get("name") or "")
            for call in tool_calls
            if isinstance(call, dict)
        }
        if not names or not names.issubset(housekeeping):
            return None
        content = row.get("content")
        if not isinstance(content, str) or not content.strip():
            return None
        strip_think = getattr(agent, "_strip_think_blocks", None)
        if callable(strip_think):
            content = strip_think(content)
        return content.strip() or None
    return None


def _native_iteration_limit_summary(
    agent: Any,
    *,
    messages: list[dict[str, Any]],
    system_prompt: str,
    turn_id: str,
) -> Optional[str]:
    """Match Hermes' terminal toolless summary without mutating native history."""
    summary_prompt = (
        "You've reached the maximum number of tool-calling iterations allowed. "
        "Please provide a final response summarizing what you've found and "
        "accomplished so far, without calling any more tools."
    )
    previous_session = getattr(agent, "_claude_agent_sdk_session", None)
    summary_session = _new_transient_session(
        agent,
        system_prompt=system_prompt,
        turn_id=f"{turn_id}:iteration-summary",
        tool_definitions=[],
        json_schema=_native_json_schema(agent),
    )
    agent._claude_agent_sdk_session = summary_session
    try:
        agent._emit_status("⚠️ Iteration budget exhausted — asking Claude to summarise")
        result = summary_session.summarize(
            agent=agent,
            messages=messages,
            prompt=summary_prompt,
        )
        usage = getattr(result, "token_usage", None)
        if isinstance(usage, dict) and usage:
            # Account the extra subscription call, then let the main turn's
            # usage update below restore compressor pressure to the last
            # authoritative main-thread iteration.
            _record_usage(
                agent,
                usage,
                getattr(result, "last_call_usage", None),
                api_call_count=max(
                    1,
                    int(getattr(result, "model_iterations", 0) or 0),
                ),
            )
        if getattr(result, "error", None) or getattr(result, "interrupted", False):
            logger.warning(
                "Claude iteration-limit summary failed: %s",
                getattr(result, "error", None) or "interrupted",
            )
            return None
        text = str(getattr(result, "final_text", "") or "").strip()
        return text or None
    except Exception:
        logger.warning("Claude iteration-limit summary failed", exc_info=True)
        return None
    finally:
        summary_session.close()
        agent._claude_agent_sdk_session = previous_session


def _native_error_classification(
    agent: Any,
    error: str,
    *,
    category: Optional[str] = None,
    status_code: Optional[int] = None,
) -> Any:
    """Classify a terminal Claude subprocess error once for hooks/failover."""
    try:
        from agent.error_classifier import (
            ClassifiedError,
            FailoverReason,
            classify_api_error,
        )

        normalized_category = str(category or "").strip().lower()
        category_map = {
            "authentication_failed": (
                FailoverReason.auth,
                False,
                False,
                True,
                True,
            ),
            "oauth_org_not_allowed": (
                FailoverReason.auth_permanent,
                False,
                False,
                False,
                True,
            ),
            "billing_error": (
                FailoverReason.billing,
                False,
                False,
                True,
                True,
            ),
            "rate_limit": (
                FailoverReason.rate_limit,
                True,
                False,
                True,
                True,
            ),
            "overloaded": (
                FailoverReason.overloaded,
                True,
                False,
                False,
                True,
            ),
            "server_error": (
                FailoverReason.server_error,
                True,
                False,
                False,
                True,
            ),
            "model_not_found": (
                FailoverReason.model_not_found,
                False,
                False,
                False,
                True,
            ),
            "invalid_request": (
                FailoverReason.format_error,
                False,
                False,
                False,
                True,
            ),
            # Claude's category means the configured response cap itself was
            # rejected/exhausted, not that input history should be compressed.
            "max_output_tokens": (
                FailoverReason.format_error,
                False,
                False,
                False,
                False,
            ),
        }
        mapped = category_map.get(normalized_category)
        if mapped is not None:
            reason, retryable, should_compress, rotate, fallback = mapped
            return ClassifiedError(
                reason=reason,
                status_code=status_code,
                provider=str(getattr(agent, "provider", "") or ""),
                model=str(getattr(agent, "model", "") or ""),
                message=str(error),
                retryable=retryable,
                should_compress=should_compress,
                should_rotate_credential=rotate,
                should_fallback=fallback,
            )

        return classify_api_error(
            RuntimeError(error),
            provider=str(getattr(agent, "provider", "") or ""),
            model=str(getattr(agent, "model", "") or ""),
        )
    except Exception:
        logger.debug("Claude terminal error classification failed", exc_info=True)
        return None


def _native_error_failover_reason(agent: Any, error: str) -> Any:
    """Return the shared classifier reason used by Hermes' fallback chain."""
    classified = _native_error_classification(agent, error)
    return getattr(classified, "reason", None)


def _native_display_error(agent: Any, error: Any) -> str:
    """Return the standard bounded, force-redacted native error surface."""
    raw = str(error or "Unknown Claude Agent SDK error")
    try:
        summary = agent._summarize_api_error(RuntimeError(raw))
    except Exception:
        summary = raw[:500]
    try:
        from agent.redact import redact_sensitive_text

        summary = redact_sensitive_text(
            str(summary),
            force=True,
            redact_url_credentials=True,
        )
    except Exception:
        summary = str(summary)
    return "Claude Agent SDK turn failed: " + summary.strip()


def _sanitize_native_terminal_error_surface(agent: Any, turn: Any) -> None:
    """Redact a terminal error before its assistant projection is committed.

    Some Claude Agent SDK versions repeat an error result as both ``final_text`` and
    a text-only assistant event.  Sanitizing only the result envelope is too
    late: the projection has already become durable main-session history by
    then.  Preserve a genuine partial answer, but replace the CLI's duplicated
    raw error wherever it is acting as the answer surface.
    """
    raw_error = str(getattr(turn, "error", "") or "").strip()
    if not raw_error:
        return
    raw_final = str(getattr(turn, "final_text", "") or "").strip()
    if raw_final and raw_final != raw_error:
        return

    display_error = _native_display_error(agent, raw_error)
    turn.final_text = display_error
    for row in getattr(turn, "projected_messages", None) or []:
        if not isinstance(row, dict):
            continue
        if row.get("role") != "assistant" or row.get("tool_calls"):
            continue
        content = str(row.get("content") or "").strip()
        if content in {raw_error, raw_final}:
            row["content"] = display_error


def _native_fallback_handoff(
    agent: Any,
    *,
    messages: list[dict[str, Any]],
    api_call_count: int,
    reason: Any,
    pending_verification_response: Optional[str],
    pending_verification_response_previewed: bool,
) -> Optional[dict[str, Any]]:
    """Activate Hermes' configured fallback and hand the prepared turn back.

    ``conversation_loop`` recognizes the private marker and falls through to
    its normal provider loop with the existing turn locals.  That preserves
    the single prologue/user-row invariant while restoring the complete
    retry/fallback machinery for a terminal native failure.
    """
    if not getattr(agent, "_has_pending_fallback", lambda: False)():
        return None
    try:
        activated = agent._try_activate_fallback(reason=reason)
    except Exception:
        logger.warning("Claude Agent SDK fallback activation failed", exc_info=True)
        return None
    if not activated:
        return None

    # A partial native assistant message is useful context for the fallback,
    # but it is not a completed transcript turn. Pair it with a private user
    # continuation so strict providers see valid role alternation; both rows
    # are removed from durable/returned history by the shared finalizer.
    tail = messages[-1] if messages else None
    if (
        isinstance(tail, dict)
        and tail.get("role") == "assistant"
        and not tail.get("tool_calls")
    ):
        tail["_provider_fallback_synthetic"] = True
        tail.pop("_db_persisted", None)
        messages.append(
            {
                "role": "user",
                "content": (
                    "[System: The previous inference runtime stopped before "
                    "finishing. Continue the same task from the transcript "
                    "and produce the complete answer.]"
                ),
                "_provider_fallback_synthetic": True,
            }
        )

    agent._session_messages = messages
    return {
        "_claude_agent_sdk_continue_standard_loop": True,
        "api_calls": api_call_count,
        "pending_verification_response": pending_verification_response,
        "pending_verification_response_previewed": (
            pending_verification_response_previewed
        ),
    }


def _native_continuation_nudge(
    agent: Any,
    *,
    user_message: Any,
    final_response: str,
    messages: list[dict[str, Any]],
    ack_continuations: int,
) -> tuple[Optional[str], Optional[str], str, int]:
    """Apply the standard final-answer continuation gates to a native turn.

    Returns ``(nudge, synthetic_flag, reason, ack_continuations)``.  Claude
    owns the inner model/tool loop, so these gates run between native user
    inputs instead of inside Hermes' chat-completions loop.
    """
    try:
        from agent.agent_runtime_helpers import intent_ack_continuation_mode

        ack_mode = intent_ack_continuation_mode(agent)
        if (
            ack_mode != "off"
            and agent.valid_tool_names
            and ack_continuations < 2
            and agent._looks_like_codex_intermediate_ack(
                user_message=user_message,
                assistant_content=final_response,
                messages=messages,
                require_workspace=(ack_mode == "codex_only"),
            )
        ):
            return (
                "[System: Continue now. Execute the required tool calls and only "
                "send your final answer after completing the task.]",
                None,
                "intent_ack_continuation",
                ack_continuations + 1,
            )
    except Exception:
        logger.debug("Claude intent-ack continuation check failed", exc_info=True)

    try:
        from agent.verification_stop import (
            build_verify_on_stop_nudge,
            verify_on_stop_enabled,
        )

        verify_nudge = (
            build_verify_on_stop_nudge(
                session_id=getattr(agent, "session_id", None),
                changed_paths=getattr(agent, "_turn_file_mutation_paths", set()),
                attempts=getattr(agent, "_verification_stop_nudges", 0),
            )
            if verify_on_stop_enabled()
            else None
        )
    except Exception:
        logger.debug("Claude verification stop-loop check failed", exc_info=True)
        verify_nudge = None
    if verify_nudge:
        agent._verification_stop_nudges = (
            getattr(agent, "_verification_stop_nudges", 0) + 1
        )
        return (
            verify_nudge,
            "_verification_stop_synthetic",
            "verification_required",
            ack_continuations,
        )

    edited = sorted(getattr(agent, "_turn_file_mutation_paths", set()) or [])
    attempt = getattr(agent, "_pre_verify_nudges", 0)
    try:
        from agent.verify_hooks import max_verify_nudges
        from hermes_cli.lifecycle import has_hook
        from hermes_cli.plugins import get_pre_verify_continue_message

        pre_verify_nudge = None
        if edited and has_hook("pre_verify") and attempt < max_verify_nudges():
            coding = getattr(agent, "_resolved_is_coding", None)
            if coding is None:
                from agent.coding_context import is_coding_context

                coding = bool(
                    is_coding_context(platform=getattr(agent, "platform", "") or "")
                )
                agent._resolved_is_coding = coding
            pre_verify_nudge = get_pre_verify_continue_message(
                session_id=getattr(agent, "session_id", None) or "",
                platform=getattr(agent, "platform", "") or "",
                model=getattr(agent, "model", "") or "",
                coding=coding,
                attempt=attempt,
                final_response=final_response,
                changed_paths=edited,
            )
    except Exception:
        logger.debug("Claude pre_verify hook check failed", exc_info=True)
        pre_verify_nudge = None
    if pre_verify_nudge:
        agent._pre_verify_nudges = attempt + 1
        return (
            pre_verify_nudge,
            "_pre_verify_synthetic",
            "verify_hook_continue",
            ack_continuations,
        )

    try:
        from agent.kanban_stop import build_kanban_stop_nudge

        kanban_nudge = build_kanban_stop_nudge(
            messages=messages,
            attempts=getattr(agent, "_kanban_stop_nudges", 0),
        )
    except Exception:
        logger.debug("Claude kanban stop-loop check failed", exc_info=True)
        kanban_nudge = None
    if kanban_nudge:
        agent._kanban_stop_nudges = getattr(agent, "_kanban_stop_nudges", 0) + 1
        return (
            kanban_nudge,
            "_kanban_stop_synthetic",
            "kanban_terminal_required",
            ack_continuations,
        )

    return None, None, "", ack_continuations


def run_claude_agent_sdk_turn(
    agent: Any,
    *,
    user_message: Any,
    original_user_message: Any,
    messages: List[Dict[str, Any]],
    conversation_history: List[Dict[str, Any]],
    effective_task_id: str,
    turn_id: str,
    current_turn_user_idx: int,
    active_system_prompt: str,
    should_review_memory: bool = False,
    moa_config: Optional[dict[str, Any]] = None,
    _moa_applied: bool = False,
    _api_calls_before: int = 0,
    _ack_continuations: int = 0,
    _empty_retries: int = 0,
    _post_tool_empty_retried: bool = False,
    _pending_verification_response: Optional[str] = None,
    _pending_verification_response_previewed: bool = False,
    _native_transport_retries: int = 0,
    _length_continue_retries: int = 0,
    _truncated_response_parts: Optional[list[str]] = None,
    _truncated_parts_previewed: bool = True,
    _image_shrink_retries: int = 0,
) -> Dict[str, Any]:
    """Run one Hermes turn through the session's persistent Claude child."""
    truncated_response_parts = list(_truncated_response_parts or [])
    # The standard loop checks this before its first provider request. Native
    # dispatch must do the same: a stop can land after shared turn setup but
    # before the Claude session installs its active-request abort callback.
    # Starting the child anyway would burn a request the user already stopped.
    if (
        bool(getattr(agent, "_interrupt_requested", False))
        and not bool(getattr(agent, "_has_pending_redirect", lambda: False)())
    ):
        from agent.turn_finalizer import finalize_turn

        result = finalize_turn(
            agent,
            final_response=None,
            api_call_count=_api_calls_before,
            interrupted=True,
            failed=False,
            messages=messages,
            conversation_history=conversation_history,
            effective_task_id=effective_task_id,
            turn_id=turn_id,
            user_message=user_message,
            original_user_message=original_user_message,
            _should_review_memory=should_review_memory,
            _turn_exit_reason="interrupted_by_user",
            _pending_verification_response=_pending_verification_response,
            _pending_verification_response_previewed=(
                _pending_verification_response_previewed
            ),
        )
        result.update(
            {
                "partial": True,
                "error": None,
                "agent_persisted": True,
                "claude_session_id": getattr(
                    getattr(agent, "_claude_agent_sdk_session", None),
                    "native_session_id",
                    None,
                ),
            }
        )
        return result
    unsupported_request_controls = _native_unsupported_request_controls(agent)
    effective_user_input = user_message
    if 0 <= current_turn_user_idx < len(messages):
        current_user = messages[current_turn_user_idx]
        if isinstance(current_user, dict) and current_user.get("role") == "user":
            effective_user_input = current_user.get(
                "api_content", current_user.get("content", user_message)
            )
    pending_compaction_context = str(
        getattr(agent, _PENDING_NATIVE_COMPACTION_CONTEXT, "") or ""
    ).strip()
    if pending_compaction_context:
        if 0 <= current_turn_user_idx < len(messages):
            current_user = messages[current_turn_user_idx]
            if isinstance(current_user, dict) and current_user.get("role") == "user":
                effective_user_input = stamp_native_compaction_context(
                    current_user,
                    pending_compaction_context,
                )
            else:
                effective_user_input = _prepend_native_compaction_context(
                    effective_user_input,
                    pending_compaction_context,
                )
        else:
            effective_user_input = _prepend_native_compaction_context(
                effective_user_input,
                pending_compaction_context,
            )
    # Tool intent/results must enter the Hermes transcript immediately around
    # execution, but a text-only terminal assistant row is still subject to
    # Relay / LLM execution middleware response transforms. Keep an exact list
    # of the authoritative rows already committed in memory so the transformed
    # result can be reconciled without retaining the pre-middleware answer.
    incrementally_projected_rows: list[dict[str, Any]] = []
    api_request_id = f"{turn_id}:claude-agent-sdk:{uuid.uuid4().hex[:12]}"
    agent._current_api_request_id = api_request_id
    native_effective_system_prompt = _native_system_prompt_with_prefill(
        agent, active_system_prompt
    )
    baseline_request_messages = _native_request_messages(
        agent,
        messages,
        system_prompt=native_effective_system_prompt,
    )
    if pending_compaction_context:
        # Keep observer hooks/context engines faithful to the bytes sent while
        # retaining a clean user-visible transcript and stable system prompt.
        for row in reversed(baseline_request_messages):
            if row.get("role") == "user":
                row["content"] = effective_user_input
                break
    if moa_config and not _moa_applied:
        baseline_request_messages = _attach_native_moa_context(
            agent,
            baseline_request_messages,
            original_user_message=original_user_message,
            moa_config=moa_config,
        )
        for row in reversed(baseline_request_messages):
            if row.get("role") == "user":
                effective_user_input = row.get("content", effective_user_input)
                break
    request_messages, context_selection_changed = _select_native_request_context(
        agent,
        baseline_request_messages,
        messages,
        current_turn_user_idx=current_turn_user_idx,
    )
    middleware_request, middleware_trace = _apply_native_request_middleware(
        agent,
        request_messages=request_messages,
        effective_task_id=effective_task_id,
        turn_id=turn_id,
        api_request_id=api_request_id,
        api_call_count=_api_calls_before + 1,
    )
    middleware_messages = middleware_request.get("messages")
    if not (
        isinstance(middleware_messages, list)
        and middleware_messages
        and all(isinstance(row, dict) for row in middleware_messages)
    ):
        logger.warning(
            "Claude request middleware returned invalid messages; using the "
            "pre-middleware request"
        )
        middleware_messages = request_messages
    request_messages = middleware_messages
    native_model = str(middleware_request.get("model") or agent.model)
    native_tools = middleware_request.get("tools")
    if not isinstance(native_tools, list):
        logger.warning(
            "Claude request middleware returned invalid tools; using the "
            "agent's stable tool schema"
        )
        native_tools = agent.tools or []
    middleware_changed = (
        request_messages != baseline_request_messages
        or native_model != agent.model
        or native_tools != (agent.tools or [])
    )
    unsupported_keys = set(middleware_request) - {
        "model",
        "messages",
        "tools",
        "transport",
    }
    middleware_controls = [
        f"request_middleware.{key}" for key in sorted(unsupported_keys)
    ]
    if middleware_request.get("transport", "claude_agent_sdk") != "claude_agent_sdk":
        # The middleware request is a logical view over an already-selected
        # native transport.  Pretending a transport rewrite took effect would
        # be worse than rejecting it: the persistent Claude process would
        # still handle the request.  Restore the truthful value and surface
        # the unsupported control in the turn result.
        middleware_controls.append("request_middleware.transport")
        middleware_request["transport"] = "claude_agent_sdk"
    if middleware_controls:
        unsupported_request_controls = list(
            dict.fromkeys([*unsupported_request_controls, *middleware_controls])
        )
        middleware_warning = (
            "Claude request middleware produced controls that cannot be "
            "forwarded to the CLI: " + ", ".join(middleware_controls)
        )
        logger.warning(middleware_warning)
        try:
            agent._emit_status("⚠️ " + middleware_warning)
        except Exception:
            pass
    request_context_changed = context_selection_changed or middleware_changed
    native_system_prompt = native_effective_system_prompt
    native_user_input = effective_user_input
    native_bootstrap_messages: Optional[list[dict[str, Any]]] = None
    if request_context_changed:
        selected_system = [
            row.get("content")
            for row in request_messages
            if row.get("role") == "system" and isinstance(row.get("content"), str)
        ]
        selected_conversation = [
            row for row in request_messages if row.get("role") != "system"
        ]
        if selected_conversation and selected_conversation[-1].get("role") == "user":
            native_system_prompt = "\n\n".join(selected_system).strip()
            native_bootstrap_messages = selected_conversation
            native_user_input = selected_conversation[-1].get("content", effective_user_input)
        else:
            # A provider request must end in the active user turn. An engine
            # returning another shape is already outside its documented
            # contract; fail open instead of silently dropping the real input.
            logger.warning(
                "Claude context selection did not end in a user message; "
                "using the persistent unmodified native context"
            )
            request_messages = baseline_request_messages
            native_model = agent.model
            native_tools = agent.tools or []
            request_context_changed = False
    request_started_at = time.time()
    observer_request_messages = [dict(row) for row in request_messages]
    native_iteration_pre_count = 1
    native_iteration_post_count = 0
    native_context_usage_updates = 0
    native_skill_cadence_updates = 0
    # Claude's stream-json assistant event does not always carry usage.  The
    # terminal result frame can be the first place the final iteration's usage
    # appears (via ``usage.iterations``).  Do not eagerly turn a missing event
    # payload into a zero-valued ContextEngine update: doing so would make the
    # final accounting path believe the response had already been observed and
    # silently discard the authoritative result-frame usage.
    native_pending_context_usage_updates = 0
    native_iteration_started_at: dict[int, float] = {1: request_started_at}

    def _native_iteration_request_id(index: int) -> str:
        return api_request_id if index == 1 else f"{api_request_id}:{index}"

    agent._api_call_count = _api_calls_before + 1
    _invoke_native_pre_request_hook(
        agent,
        request_messages=request_messages,
        original_user_message=original_user_message,
        messages=messages,
        effective_task_id=effective_task_id,
        turn_id=turn_id,
        api_request_id=api_request_id,
        started_at=request_started_at,
        middleware_trace=middleware_trace,
        request_model=native_model,
        request_tools=native_tools,
        api_call_count=_api_calls_before + 1,
    )
    try:
        from utils import env_var_enabled

        if env_var_enabled("HERMES_DUMP_REQUESTS"):
            agent._dump_api_request_debug(
                middleware_request,
                reason="preflight:claude_agent_sdk",
            )
    except Exception:
        logger.debug("Claude native request debug dump failed", exc_info=True)

    def _before_next_native_model() -> Optional[dict[str, Any]]:
        nonlocal native_iteration_pre_count
        # Close the narrow race after the standard Hermes tool executor's
        # own steer drain but before the MCP result is returned to Claude.  A
        # steer arriving in that window must still ride on the final tool
        # result, matching conversation_loop's pre-next-request drain.
        trailing_tool_rows = 0
        for row in reversed(messages):
            if not isinstance(row, dict) or row.get("role") != "tool":
                break
            trailing_tool_rows += 1
        if trailing_tool_rows:
            try:
                agent._apply_pending_steer_to_tool_results(
                    messages,
                    trailing_tool_rows,
                )
                # projection_callback copied observer rows before this last
                # boundary. Keep the observer payload faithful to the exact
                # tool result Claude is about to receive.
                for source, observed in zip(
                    messages[-trailing_tool_rows:],
                    observer_request_messages[-trailing_tool_rows:],
                ):
                    if (
                        isinstance(source, dict)
                        and isinstance(observed, dict)
                        and source.get("role") == observed.get("role") == "tool"
                    ):
                        observed["content"] = source.get("content")
            except Exception:
                logger.debug("Claude pre-model steer drain failed", exc_info=True)
        wire_result_overrides: dict[str, Any] = {}
        if moa_config and trailing_tool_rows:
            # Standard Hermes recomputes one-shot MoA guidance before every
            # acting-model iteration. Claude's inner loop is private, but the
            # final MCP result boundary is still host-controlled. Attach the
            # fresh guidance only to the wire copy of the last tool result so
            # it reaches the next native request without entering Hermes'
            # visible/durable transcript.
            moa_request = _native_request_messages(
                agent,
                messages,
                system_prompt=native_effective_system_prompt,
            )
            guidance = _native_moa_guidance(
                agent,
                moa_request,
                original_user_message=original_user_message,
                moa_config=moa_config,
            )
            if guidance:
                source = messages[-1]
                tool_call_id = str(source.get("tool_call_id") or "")
                augmented = _append_private_context(
                    source.get("content", ""),
                    guidance,
                )
                if tool_call_id:
                    wire_result_overrides[tool_call_id] = augmented
                if observer_request_messages:
                    for observed in reversed(observer_request_messages):
                        if (
                            observed.get("role") == "tool"
                            and str(observed.get("tool_call_id") or "")
                            == tool_call_id
                        ):
                            observed["content"] = augmented
                            break
        native_iteration_pre_count += 1
        index = native_iteration_pre_count
        # Keep agent-level introspection/tool middleware aligned with the
        # standard loop, which publishes the current provider-call ordinal
        # before every model step.
        agent._api_call_count = _api_calls_before + index
        started_at = time.time()
        native_iteration_started_at[index] = started_at
        _invoke_native_pre_request_hook(
            agent,
            request_messages=list(observer_request_messages),
            original_user_message=original_user_message,
            messages=messages,
            effective_task_id=effective_task_id,
            turn_id=turn_id,
            api_request_id=_native_iteration_request_id(index),
            started_at=started_at,
            # Request middleware owns the outer native turn. It cannot mutate
            # Claude Code's private follow-up request, so only the first
            # observer row carries its trace.
            middleware_trace=[],
            request_model=native_model,
            request_tools=native_tools,
            api_call_count=_api_calls_before + index,
        )
        return wire_result_overrides or None

    def _after_native_model(
        index: int,
        assistant_message: dict[str, Any],
        iteration_usage: dict[str, int],
    ) -> None:
        nonlocal native_iteration_post_count
        nonlocal native_context_usage_updates
        nonlocal native_pending_context_usage_updates
        nonlocal native_skill_cadence_updates
        native_iteration_post_count += 1
        # Match the standard loop's ordering relative to skill_manage: count
        # each completed provider step here, before Claude can execute the
        # tool calls proposed by that response.  A successful skill_manage
        # then resets the counter, and only later model steps count again.
        # Adding the whole private-loop total after run_turn() would resurrect
        # the pre-reset iterations and spuriously trigger background review.
        _advance_skill_review_cadence(agent, 1)
        native_skill_cadence_updates += 1
        ended_at = time.time()
        started_at = native_iteration_started_at.get(index, request_started_at)
        _invoke_native_post_iteration_hook(
            agent,
            assistant_message=assistant_message,
            usage=iteration_usage,
            effective_task_id=effective_task_id,
            turn_id=turn_id,
            api_request_id=_native_iteration_request_id(index),
            api_call_count=_api_calls_before + index,
            started_at=started_at,
            ended_at=ended_at,
            request_model=native_model,
        )
        # ContextEngine.update_from_response is an every-provider-response
        # contract. Claude's private tool loop can make several paid model
        # calls inside one outer Hermes turn; updating only once with the final
        # call silently disabled adaptive context policy for earlier steps.
        compressor = getattr(agent, "context_compressor", None)
        if compressor is not None and iteration_usage:
            try:
                # Any earlier response still lacking usage can no longer be
                # matched with the terminal frame (which describes the latest
                # iteration). Preserve the one-update-per-response contract for
                # those gaps before recording this response's real usage.
                for _ in range(native_pending_context_usage_updates):
                    compressor.update_from_response(_normalize_native_usage({}))
                    native_context_usage_updates += 1
                native_pending_context_usage_updates = 0
                compressor.update_from_response(
                    _normalize_native_usage(iteration_usage)
                )
                native_context_usage_updates += 1
            except Exception:
                logger.warning(
                    "Claude per-iteration ContextEngine usage update failed",
                    exc_info=True,
                )
        elif compressor is not None:
            native_pending_context_usage_updates += 1

    def _on_native_api_retry(event: dict[str, Any]) -> None:
        """Surface Claude's private provider retry without rebuilding it."""
        attempt = max(1, int(event.get("attempt") or 1))
        max_retries = max(attempt, int(event.get("max_retries") or attempt))
        delay_ms = max(0, int(event.get("retry_delay_ms") or 0))
        category = str(event.get("error") or "unknown")
        status_code = event.get("error_status")
        delay = f" in {delay_ms / 1000:g}s" if delay_ms else ""
        agent._buffer_status(
            "⚠️ Claude API "
            + category.replace("_", " ")
            + f" — retry {attempt}/{max_retries}{delay}"
        )
        touch_activity = getattr(agent, "_touch_activity", None)
        if callable(touch_activity):
            touch_activity(
                f"Claude API retry {attempt}/{max_retries}: {category}"
            )
        classified = _native_error_classification(
            agent,
            category.replace("_", " "),
            category=category,
            status_code=(
                int(status_code) if isinstance(status_code, (int, float)) else None
            ),
        )
        try:
            index = max(1, native_iteration_pre_count)
            agent._invoke_api_request_error_hook(
                task_id=effective_task_id,
                turn_id=turn_id,
                api_request_id=_native_iteration_request_id(index),
                api_call_count=_api_calls_before + index,
                api_start_time=native_iteration_started_at.get(
                    index, request_started_at
                ),
                api_kwargs={
                    "model": native_model,
                    "messages": list(observer_request_messages),
                    "tools": native_tools,
                    "transport": "claude_agent_sdk",
                },
                error_type="ClaudeAgentSdkApiRetry",
                error_message=category,
                status_code=getattr(classified, "status_code", status_code),
                retry_count=attempt,
                max_retries=max_retries,
                retryable=True,
                reason=(
                    getattr(getattr(classified, "reason", None), "value", None)
                    or category
                ),
            )
        except Exception:
            logger.debug("Claude API retry observer failed", exc_info=True)

    def _persist_native_projection(rows: list[dict[str, Any]]) -> None:
        # A terminal text-only assistant record arrives immediately before the
        # result frame. Defer the row itself (not merely its DB write) those few
        # milliseconds so Relay / execution middleware can transform it before
        # it becomes conversation history, and so an active-turn redirect can
        # discard it just like the standard HTTP loop does. Tool calls/results
        # remain crash-durable immediately.
        has_tool_protocol_rows = any(
            row.get("role") == "tool" or row.get("tool_calls")
            for row in rows
            if isinstance(row, dict)
        )
        if not has_tool_protocol_rows:
            # Do not commit a provisional text-only assistant row yet, but do
            # retain it in the observer view. Claude may start an internal
            # continuation/retry before the terminal result frame; the next
            # pre_api_request hook must then describe the history that native
            # Claude actually saw rather than the previous outer request.
            observer_request_messages.extend(
                dict(row) for row in rows if isinstance(row, dict)
            )
            return
        messages.extend(rows)
        observer_request_messages.extend(dict(row) for row in rows)
        incrementally_projected_rows.extend(rows)
        if getattr(agent, "_session_db", None) is None:
            return
        try:
            flushed = agent._flush_messages_to_session_db(messages, conversation_history)
        except Exception:
            flushed = False
            logger.warning(
                "Claude incremental projected-message flush failed",
                exc_info=True,
            )
        if flushed is False:
            agent._incremental_persistence_failed = True
            logger.warning(
                "Claude native tool/progress records could not be persisted "
                "incrementally (session=%s)",
                getattr(agent, "session_id", None),
            )
    session: Optional[ClaudeAgentSdkSession] = None
    previous_session = getattr(agent, "_claude_agent_sdk_session", None)
    transient_session = False
    redirect_crossed_response = False
    native_request_performed = False
    native_result_signature: Optional[str] = None
    native_request_used_pending_context = False
    thinking_cleared = False
    persistent_history_invalidated = False

    def _clear_native_thinking() -> None:
        nonlocal thinking_cleared
        if thinking_cleared:
            return
        thinking_cleared = True
        callback = getattr(agent, "thinking_callback", None)
        if callable(callback):
            try:
                callback("")
            except Exception:
                pass

    def _native_stream_delta(delta: str) -> None:
        if delta:
            _clear_native_thinking()
        callback = getattr(agent, "_fire_stream_delta", None)
        if callable(callback):
            callback(delta)

    try:
        agent._reset_stream_delivery_tracking()
        thinking_callback = getattr(agent, "thinking_callback", None)
        if getattr(agent, "quiet_mode", False) and callable(thinking_callback):
            try:
                from agent.display import KawaiiSpinner

                face = random.choice(KawaiiSpinner.get_thinking_faces())
                verb = random.choice(KawaiiSpinner.get_thinking_verbs())
                thinking_callback(f"{face} {verb}...")
            except Exception:
                logger.debug("Claude native thinking indicator failed", exc_info=True)
        if request_context_changed:
            session = _new_transient_session(
                agent,
                system_prompt=native_system_prompt,
                turn_id=turn_id,
                model=native_model,
                tool_definitions=native_tools,
                json_schema=_native_json_schema(agent),
            )
            transient_session = True
            agent._claude_agent_sdk_session = session
        else:
            session = _get_session(agent, system_prompt=native_system_prompt)
        def _perform_native_physical_request(payload: dict[str, Any]) -> Any:
            nonlocal session
            nonlocal transient_session
            nonlocal previous_session
            nonlocal persistent_history_invalidated
            nonlocal native_request_performed
            nonlocal native_result_signature
            nonlocal native_request_used_pending_context
            request_user_input = native_user_input
            request_bootstrap_messages = native_bootstrap_messages
            if payload != middleware_request:
                unsupported_execution_keys = set(payload) - {
                    "model",
                    "messages",
                    "tools",
                    "transport",
                }
                if unsupported_execution_keys:
                    raise RuntimeError(
                        "Claude LLM execution middleware produced unsupported "
                        "native request keys: "
                        + ", ".join(sorted(unsupported_execution_keys))
                    )
                if payload.get("transport") != "claude_agent_sdk":
                    raise RuntimeError(
                        "Claude LLM execution middleware cannot replace the "
                        "selected native transport"
                    )
                execution_messages = payload.get("messages")
                if not (
                    isinstance(execution_messages, list)
                    and execution_messages
                    and all(isinstance(row, dict) for row in execution_messages)
                    and execution_messages[-1].get("role") == "user"
                ):
                    raise RuntimeError(
                        "Claude LLM execution middleware must return messages "
                        "ending in the active user turn"
                    )
                execution_model = str(payload.get("model") or native_model)
                execution_tools = payload.get("tools")
                if not isinstance(execution_tools, list):
                    raise RuntimeError(
                        "Claude LLM execution middleware must return tools as a list"
                    )
                execution_system = "\n\n".join(
                    str(row.get("content") or "")
                    for row in execution_messages
                    if row.get("role") == "system"
                    and isinstance(row.get("content"), str)
                ).strip()
                execution_conversation = [
                    dict(row)
                    for row in execution_messages
                    if row.get("role") != "system"
                ]

                # The already-prepared persistent process cannot accept a
                # different model/toolset/history without corrupting its
                # cached prefix. Honor the execution middleware in a one-turn
                # child, then invalidate the main native binding because it
                # necessarily missed this logical turn.
                if session is not None:
                    if transient_session:
                        session.close()
                        agent._claude_agent_sdk_session = previous_session
                    else:
                        _invalidate_persistent_native_history(agent, session)
                        persistent_history_invalidated = True
                        if previous_session is session:
                            previous_session = None
                    session = None
                session = _new_transient_session(
                    agent,
                    system_prompt=execution_system,
                    turn_id=f"{turn_id}:execution-middleware",
                    model=execution_model,
                    tool_definitions=execution_tools,
                    json_schema=_native_json_schema(agent),
                )
                transient_session = True
                agent._claude_agent_sdk_session = session
                request_user_input = execution_conversation[-1].get(
                    "content", native_user_input
                )
                request_bootstrap_messages = execution_conversation
            touch_activity = getattr(agent, "_touch_activity", None)
            if callable(touch_activity):
                touch_activity(
                    f"starting Claude Agent SDK request #{_api_calls_before + 1}"
                )
            agent._api_call_count = _api_calls_before + 1
            # The standard streaming transport claims the agent's
            # single-writer token before consuming any deltas.  Native Claude
            # also streams into the same callbacks, so it must participate in
            # that fence; otherwise a superseded redirect/fallback stream can
            # keep writing after a newer request has taken ownership.
            claim_stream_writer = getattr(agent, "_claim_stream_writer", None)
            if callable(claim_stream_writer):
                claim_stream_writer()
            native_result = session.run_turn(
                agent=agent,
                user_input=request_user_input,
                messages=messages,
                task_id=effective_task_id,
                stream_callback=_native_stream_delta,
                projection_callback=_persist_native_projection,
                bootstrap_messages=request_bootstrap_messages,
                before_next_model_callback=_before_next_native_model,
                iteration_post_callback=_after_native_model,
                api_retry_callback=_on_native_api_retry,
            )
            native_request_performed = True
            native_result_signature = _native_turn_signature(native_result)
            native_request_used_pending_context = bool(
                pending_compaction_context and not transient_session
            )
            return native_result

        def _perform_native_request(payload: dict[str, Any]) -> Any:
            from agent import relay_llm

            return relay_llm.execute(
                payload,
                _perform_native_physical_request,
                session_id=str(agent.session_id or ""),
                name=str(agent.provider or "anthropic"),
                model_name=str(native_model or agent.model or ""),
                metadata={
                    "api_mode": "claude_agent_sdk",
                    "api_request_id": api_request_id,
                    "call_role": (
                        "delegated"
                        if getattr(agent, "is_subagent", False)
                        else "fallback"
                        if int(getattr(agent, "_fallback_index", 0) or 0) > 0
                        else "primary"
                    ),
                    "retry_count": 0,
                    "transport": "claude_agent_sdk",
                },
                defer_logical_completion=True,
            )

        from hermes_cli.middleware import run_llm_execution_middleware

        model_request_active = getattr(agent, "_model_request_active", None)
        redirect_lock = getattr(agent, "_pending_redirect_lock", None)
        previous_abort = getattr(agent, "_active_request_abort", None)
        agent._active_request_abort = lambda _reason="": session.interrupt()
        if redirect_lock is not None:
            with redirect_lock:
                if model_request_active is not None:
                    model_request_active.set()
        elif model_request_active is not None:
            model_request_active.set()
        try:
            turn = run_llm_execution_middleware(
                middleware_request,
                _perform_native_request,
                original_request=middleware_request,
                task_id=effective_task_id,
                turn_id=turn_id,
                api_request_id=api_request_id,
                session_id=agent.session_id or "",
                platform=agent.platform or "",
                model=agent.model,
                provider=agent.provider,
                base_url="claude-agent-sdk://local",
                api_mode=agent.api_mode,
                api_call_count=_api_calls_before + 1,
                middleware_trace=list(middleware_trace),
            )
        finally:
            _clear_native_thinking()
            agent._active_request_abort = previous_abort
            if redirect_lock is not None:
                with redirect_lock:
                    if model_request_active is not None:
                        model_request_active.clear()
                    redirect_crossed_response = bool(agent._pending_redirect)
            else:
                if model_request_active is not None:
                    model_request_active.clear()
                redirect_crossed_response = agent._has_pending_redirect()
        turn = _coerce_native_turn_result(turn)
        try:
            from agent import relay_llm

            relay_llm.complete_logical_call(
                api_request_id,
                outcome=(
                    "cancelled"
                    if getattr(turn, "interrupted", False)
                    else "failed"
                    if getattr(turn, "error", None)
                    else "success"
                ),
            )
        except Exception:
            logger.debug(
                "Claude Agent SDK Relay logical-call completion failed",
                exc_info=True,
            )
        middleware_diverged_from_native = (
            not native_request_performed
            or native_result_signature != _native_turn_signature(turn)
        )
        if middleware_diverged_from_native and not transient_session:
            _invalidate_persistent_native_history(agent, session)
            session = None
    except Exception as exc:
        _clear_native_thinking()
        logger.exception("Claude Agent SDK turn failed")
        if session is not None:
            if transient_session:
                session.close()
                agent._claude_agent_sdk_session = previous_session
                if not persistent_history_invalidated:
                    _invalidate_persistent_native_history(agent, previous_session)
            else:
                _invalidate_persistent_native_history(agent, session)
        error_text = _native_display_error(agent, exc)
        classified_error = _native_error_classification(agent, str(exc))
        classified_reason = getattr(classified_error, "reason", None)
        failed_api_call_count = _api_calls_before + 1
        agent._api_call_count = failed_api_call_count
        _advance_skill_review_cadence(agent, 1)
        try:
            agent._invoke_api_request_error_hook(
                task_id=effective_task_id,
                turn_id=turn_id,
                api_request_id=api_request_id,
                api_call_count=failed_api_call_count,
                api_start_time=request_started_at,
                api_kwargs={
                    "model": agent.model,
                    "messages": request_messages,
                    "tools": agent.tools or [],
                    "transport": "claude_agent_sdk",
                },
                error_type=type(exc).__name__,
                error_message=str(exc),
                status_code=getattr(classified_error, "status_code", None),
                retry_count=_native_transport_retries,
                max_retries=_MAX_NATIVE_TRANSPORT_RETRIES,
                retryable=getattr(classified_error, "retryable", None),
                reason=(
                    getattr(classified_reason, "value", None)
                    or "claude_agent_sdk_startup_error"
                ),
            )
        except Exception:
            logger.debug("Claude Agent SDK api_request_error hook failed", exc_info=True)

        if (
            bool(getattr(classified_error, "retryable", False))
            and not bool(getattr(classified_error, "should_fallback", False))
            and _native_transport_retries < _MAX_NATIVE_TRANSPORT_RETRIES
            and not bool(getattr(agent, "_interrupt_requested", False))
        ):
            agent._buffer_status(
                "⚠️ Claude Agent SDK transport stopped before a response — "
                "restarting it once"
            )
            agent._turn_received_provider_response = False
            return run_claude_agent_sdk_turn(
                agent,
                user_message=user_message,
                original_user_message=original_user_message,
                messages=messages,
                conversation_history=conversation_history,
                effective_task_id=effective_task_id,
                turn_id=turn_id,
                current_turn_user_idx=current_turn_user_idx,
                active_system_prompt=active_system_prompt,
                should_review_memory=should_review_memory,
                moa_config=moa_config,
                _moa_applied=_moa_applied,
                _api_calls_before=failed_api_call_count,
                _ack_continuations=_ack_continuations,
                _empty_retries=_empty_retries,
                _post_tool_empty_retried=_post_tool_empty_retried,
                _pending_verification_response=_pending_verification_response,
                _pending_verification_response_previewed=(
                    _pending_verification_response_previewed
                ),
                _native_transport_retries=_native_transport_retries + 1,
                _length_continue_retries=_length_continue_retries,
                _truncated_response_parts=truncated_response_parts,
                _truncated_parts_previewed=_truncated_parts_previewed,
                _image_shrink_retries=_image_shrink_retries,
            )

        messages.append({"role": "assistant", "content": error_text})

        handoff = _native_fallback_handoff(
            agent,
            messages=messages,
            api_call_count=failed_api_call_count,
            reason=classified_reason,
            pending_verification_response=_pending_verification_response,
            pending_verification_response_previewed=(
                _pending_verification_response_previewed
            ),
        )
        if handoff is not None:
            agent._turn_received_provider_response = False
            return handoff

        from agent.turn_finalizer import finalize_turn

        result = finalize_turn(
            agent,
            final_response=error_text,
            api_call_count=failed_api_call_count,
            interrupted=bool(getattr(agent, "_interrupt_requested", False)),
            failed=True,
            messages=messages,
            conversation_history=conversation_history,
            effective_task_id=effective_task_id,
            turn_id=turn_id,
            user_message=user_message,
            original_user_message=original_user_message,
            # Match the standard loop's terminal-error finalization.  The
            # trigger was computed before native dispatch and must not be
            # silently disabled merely because the Claude child failed to
            # start; the background reviewer is persistence-isolated and can
            # still consolidate the completed logical turn/error context.
            _should_review_memory=should_review_memory,
            _turn_exit_reason="claude_agent_sdk_startup_error",
        )
        result.update({"partial": True, "error": error_text, "agent_persisted": True})
        return result

    if transient_session:
        session.close()
        agent._claude_agent_sdk_session = previous_session
        if not persistent_history_invalidated:
            _invalidate_persistent_native_history(agent, previous_session)

    if native_request_used_pending_context:
        # Compare before clearing: an observer or nested boundary may have
        # appended newer continuity while this request was running.
        current_pending = str(
            getattr(agent, _PENDING_NATIVE_COMPACTION_CONTEXT, "") or ""
        ).strip()
        if current_pending == pending_compaction_context:
            setattr(agent, _PENDING_NATIVE_COMPACTION_CONTEXT, "")

    persistence_failed = bool(
        getattr(agent, "_incremental_persistence_failed", False)
    )
    if persistence_failed:
        turn.error = (
            "Claude Agent SDK turn stopped because session storage could not persist "
            "the tool protocol. Free disk space and retry."
        )
        turn.final_text = turn.error
        turn.should_retire = True

    # Third-party/custom session adapters may implement the older run_turn
    # contract and not invoke per-iteration callbacks. Preserve the prior
    # one-event observer behavior as a compatibility fallback; the built-in
    # transport emits one post event for every provider response above.
    if native_iteration_post_count == 0:
        _invoke_native_post_request_hook(
            agent,
            turn=turn,
            request_messages=request_messages,
            effective_task_id=effective_task_id,
            turn_id=turn_id,
            api_request_id=api_request_id,
            started_at=request_started_at,
            ended_at=time.time(),
            request_model=native_model,
            api_call_count=(
                _api_calls_before
                + max(1, int(getattr(turn, "model_iterations", 0) or 0))
            ),
        )
    touch_activity = getattr(agent, "_touch_activity", None)
    if callable(touch_activity):
        touch_activity(
            "Claude Agent SDK request completed"
            if not turn.error
            else "Claude Agent SDK request ended with an error"
        )

    terminal_classified_error = None
    if (
        turn.error
        and not persistence_failed
        and not turn.interrupted
        and not getattr(turn, "budget_exhausted", False)
    ):
        try:
            classified_error = _native_error_classification(
                agent,
                str(turn.error),
                category=getattr(turn, "error_category", None),
                status_code=getattr(turn, "error_status", None),
            )
            terminal_classified_error = classified_error
            classified_reason = getattr(classified_error, "reason", None)
            agent._invoke_api_request_error_hook(
                task_id=effective_task_id,
                turn_id=turn_id,
                api_request_id=api_request_id,
                api_call_count=(
                    _api_calls_before
                    + max(1, int(turn.model_iterations or 0))
                ),
                api_start_time=request_started_at,
                api_kwargs={
                    "model": native_model,
                    "messages": request_messages,
                    "tools": native_tools,
                    "transport": "claude_agent_sdk",
                },
                error_type="ClaudeAgentSdkRuntimeError",
                error_message=str(turn.error),
                status_code=getattr(classified_error, "status_code", None),
                retry_count=_native_transport_retries,
                max_retries=_MAX_NATIVE_TRANSPORT_RETRIES,
                retryable=getattr(classified_error, "retryable", None),
                reason=(
                    getattr(classified_reason, "value", None)
                    or "claude_agent_sdk_runtime_error"
                ),
            )
        except Exception:
            logger.debug("Claude Agent SDK api_request_error hook failed", exc_info=True)

    try:
        from agent.error_classifier import FailoverReason

        terminal_reason = getattr(terminal_classified_error, "reason", None)
    except Exception:
        FailoverReason = None  # type: ignore[assignment]
        terminal_reason = None
    if (
        FailoverReason is not None
        and terminal_reason == FailoverReason.image_too_large
        and _image_shrink_retries < 1
        and not turn.interrupted
        and _shrink_native_history_images(
            agent,
            messages,
            error=str(turn.error or ""),
        )
    ):
        agent._buffer_status(
            "📐 Image exceeded Claude's size limit — resized the API copy "
            "and retrying once"
        )
        if session is not None:
            if transient_session:
                session.close()
                agent._claude_agent_sdk_session = previous_session
            else:
                _invalidate_persistent_native_history(agent, session)
        return run_claude_agent_sdk_turn(
            agent,
            user_message=user_message,
            original_user_message=original_user_message,
            messages=messages,
            conversation_history=conversation_history,
            effective_task_id=effective_task_id,
            turn_id=turn_id,
            current_turn_user_idx=current_turn_user_idx,
            active_system_prompt=active_system_prompt,
            should_review_memory=should_review_memory,
            moa_config=moa_config,
            _moa_applied=_moa_applied,
            _api_calls_before=(
                _api_calls_before
                + max(1, int(getattr(turn, "model_iterations", 0) or 0))
            ),
            _ack_continuations=_ack_continuations,
            _empty_retries=_empty_retries,
            _post_tool_empty_retried=_post_tool_empty_retried,
            _pending_verification_response=_pending_verification_response,
            _pending_verification_response_previewed=(
                _pending_verification_response_previewed
            ),
            _native_transport_retries=_native_transport_retries,
            _length_continue_retries=_length_continue_retries,
            _truncated_response_parts=truncated_response_parts,
            _truncated_parts_previewed=_truncated_parts_previewed,
            _image_shrink_retries=_image_shrink_retries + 1,
        )

    if turn.should_retire and not transient_session:
        _invalidate_persistent_native_history(agent, session)
        session = None

    empty_response_exhausted = False
    if getattr(turn, "budget_exhausted", False):
        # Session.run_turn uses ``error`` internally to stop the native child,
        # but iteration exhaustion is a controlled Hermes exit, not a provider
        # failure. If a verification gate had deliberately withheld a valid
        # candidate, let the shared finalizer restore that exact answer.
        turn.error = None
        if _pending_verification_response:
            turn.final_text = None
        else:
            # Text-only native projections are deliberately deferred until
            # Relay / execution middleware has completed.  The iteration-limit
            # summarizer still needs that unfinished work as input even though
            # it must not become a second adjacent assistant row in the main
            # transcript.  Add only the not-yet-committed suffix to the
            # isolated summary view.
            summary_messages = list(messages)
            committed_count = len(incrementally_projected_rows)
            native_projection = list(turn.projected_messages or [])
            if (
                committed_count <= len(native_projection)
                and native_projection[:committed_count]
                == incrementally_projected_rows
            ):
                summary_messages.extend(native_projection[committed_count:])
            else:
                summary_messages.extend(
                    row
                    for row in native_projection
                    if isinstance(row, dict)
                    and row.get("role") == "assistant"
                    and not row.get("tool_calls")
                )
            turn.final_text = _native_iteration_limit_summary(
                agent,
                messages=summary_messages,
                system_prompt=native_system_prompt,
                turn_id=turn_id,
            ) or (
                "I reached Hermes' iteration limit before I could finish, and "
                "the final summary request did not complete."
            )

    api_call_count = max(
        1,
        int(getattr(turn, "model_iterations", 0) or 0),
        int(getattr(turn, "budget_iterations", 0) or 0),
    )
    # Older/custom Claude session adapters may not implement the per-iteration
    # callback. Preserve their prior cadence behavior, while the built-in
    # adapter has already counted each step at the correct tool-reset boundary.
    _advance_skill_review_cadence(
        agent,
        max(0, api_call_count - native_skill_cadence_updates),
    )
    # Decide whether the provider actually responded before synthesizing or
    # persisting any terminal-error projection.  A transport-only failure is
    # retried on a fresh child without polluting the main session first.
    provider_response_received = bool(
        int(getattr(turn, "model_iterations", 0) or 0) > 0
        or getattr(turn, "token_usage", None)
        or getattr(turn, "last_call_usage", None)
        or any(
            isinstance(row, dict) and row.get("role") == "assistant"
            for row in (getattr(turn, "projected_messages", None) or [])
        )
        or (
            not turn.error
            and not turn.interrupted
            and bool(str(getattr(turn, "final_text", "") or "").strip())
        )
    )
    total_api_call_count = _api_calls_before + api_call_count
    agent._turn_received_provider_response = provider_response_received
    agent._api_call_count = total_api_call_count

    if (
        turn.error
        and not provider_response_received
        and not bool(getattr(turn, "terminal_result_received", False))
        and not persistence_failed
        and not turn.interrupted
        and not getattr(turn, "budget_exhausted", False)
        and not incrementally_projected_rows
        and bool(getattr(terminal_classified_error, "retryable", False))
        and not bool(
            getattr(terminal_classified_error, "should_fallback", False)
        )
        and _native_transport_retries < _MAX_NATIVE_TRANSPORT_RETRIES
    ):
        agent._buffer_status(
            "⚠️ Claude Agent SDK transport stopped before a response — "
            "restarting it once"
        )
        return run_claude_agent_sdk_turn(
            agent,
            user_message=user_message,
            original_user_message=original_user_message,
            messages=messages,
            conversation_history=conversation_history,
            effective_task_id=effective_task_id,
            turn_id=turn_id,
            current_turn_user_idx=current_turn_user_idx,
            active_system_prompt=active_system_prompt,
            should_review_memory=should_review_memory,
            moa_config=moa_config,
            _moa_applied=_moa_applied,
            _api_calls_before=total_api_call_count,
            _ack_continuations=_ack_continuations,
            _empty_retries=_empty_retries,
            _post_tool_empty_retried=_post_tool_empty_retried,
            _pending_verification_response=_pending_verification_response,
            _pending_verification_response_previewed=(
                _pending_verification_response_previewed
            ),
            _native_transport_retries=_native_transport_retries + 1,
            _length_continue_retries=_length_continue_retries,
            _truncated_response_parts=truncated_response_parts,
            _truncated_parts_previewed=_truncated_parts_previewed,
            _image_shrink_retries=_image_shrink_retries,
        )

    if (
        turn.error
        and not persistence_failed
        and not turn.interrupted
        and not getattr(turn, "budget_exhausted", False)
    ):
        _sanitize_native_terminal_error_surface(agent, turn)
        try:
            from agent.error_classifier import FailoverReason

            if (
                getattr(terminal_classified_error, "reason", None)
                in {
                    FailoverReason.context_overflow,
                    FailoverReason.payload_too_large,
                    FailoverReason.long_context_tier,
                }
                and not bool(getattr(agent, "compression_enabled", True))
            ):
                # Match the standard loop's user-facing disabled-compaction
                # contract, not merely its metadata flag.  This must happen
                # before the terminal projection is appended so the delivered
                # answer and durable assistant row remain identical.
                actionable = (
                    "Context overflow and auto-compaction is disabled "
                    "(compression.enabled: false). Run /compress to compact "
                    "manually, /new to start fresh, or switch to a "
                    "larger-context model."
                )
                turn.error = actionable
                turn.final_text = actionable
                for row in getattr(turn, "projected_messages", None) or []:
                    if (
                        isinstance(row, dict)
                        and row.get("role") == "assistant"
                        and not row.get("tool_calls")
                    ):
                        row["content"] = actionable
        except Exception:
            logger.debug(
                "Claude disabled-compaction error enrichment failed",
                exc_info=True,
            )

    projected = _dedupe_final_projection(
        turn.projected_messages,
        turn.final_text,
        replace_terminal_assistant=bool(
            getattr(turn, "budget_exhausted", False)
            and not _pending_verification_response
        ),
    )
    # Claude's private loop can emit more than one completed assistant record
    # without a host-visible user/tool row between them (for example an
    # internal continuation). The native history may represent that protocol,
    # but Hermes' durable transcript must retain its strict role-alternation
    # invariant. Reuse the same lossless repair applied before standard HTTP
    # requests, before any deferred text projection is written to SessionDB.
    try:
        repairs = agent._repair_message_sequence(projected)
        if repairs:
            logger.info(
                "Repaired %d Claude native projection alternation violation(s)",
                repairs,
            )
    except Exception:
        logger.warning(
            "Claude native projection alternation repair failed",
            exc_info=True,
        )
    committed_count = len(incrementally_projected_rows)
    if (
        committed_count <= len(projected)
        and projected[:committed_count] == incrementally_projected_rows
    ):
        remaining_projection = projected[committed_count:]
    else:
        # Response middleware may replace the result projection wholesale.
        # Tool rows already executed through the authoritative Hermes pipeline
        # cannot be rewritten retroactively; retain those committed rows and
        # apply only the middleware's terminal text projection.
        remaining_projection = [
            row
            for row in projected
            if isinstance(row, dict)
            and row.get("role") == "assistant"
            and not row.get("tool_calls")
        ]
    if remaining_projection:
        messages.extend(remaining_projection)
        if getattr(agent, "_session_db", None) is not None:
            try:
                agent._flush_messages_to_session_db(messages)
            except Exception:
                logger.debug("Claude Agent SDK projected-message flush failed", exc_info=True)

    if not transient_session and not middleware_diverged_from_native:
        _record_native_auto_compaction(
            agent,
            turn,
            messages=messages,
            task_id=effective_task_id,
        )
    if provider_response_received:
        # Flush usage-less non-final iterations now. Keep the final pending
        # response for _record_usage(), which can recover its authoritative
        # usage from the terminal result frame's last_call_usage field.
        compressor = getattr(agent, "context_compressor", None)
        if compressor is not None and native_pending_context_usage_updates > 1:
            try:
                for _ in range(native_pending_context_usage_updates - 1):
                    compressor.update_from_response(_normalize_native_usage({}))
                    native_context_usage_updates += 1
                native_pending_context_usage_updates = 1
            except Exception:
                logger.warning(
                    "Claude pending ContextEngine usage update failed",
                    exc_info=True,
                )
        usage_result = _record_usage(
            agent,
            turn.token_usage,
            turn.last_call_usage,
            api_call_count=api_call_count,
            update_context_engine=(
                native_pending_context_usage_updates > 0
                or native_context_usage_updates == 0
            ),
        )
        # Context engines observe the latest assembled provider request, not
        # the cumulative usage for Claude's whole private tool loop. Session/DB
        # accounting above deliberately remains aggregate.
        agent._last_turn_usage = dict(usage_result["last_usage"])
    if redirect_crossed_response and agent.clear_interrupt(preserve_redirect=True):
        # The final text-only projection is deliberately not flushed by the
        # incremental callback, so a response that crossed the redirect can be
        # removed without leaving a stale assistant bubble in durable history.
        while messages:
            tail = messages[-1]
            if not (
                isinstance(tail, dict)
                and tail.get("role") == "assistant"
                and not tail.get("tool_calls")
                and not tail.get("_db_persisted")
            ):
                break
            messages.pop()
        correction = agent._drain_pending_redirect()
        if correction:
            from agent.conversation_loop import _apply_active_turn_redirect

            _apply_active_turn_redirect(agent, messages, correction)
            if isinstance(original_user_message, str):
                original_user_message = (
                    f"{original_user_message}\n\n"
                    f"User correction during the turn: {correction}"
                )
            try:
                agent._persist_session(messages, conversation_history)
            except Exception:
                logger.warning("Claude redirected turn checkpoint persist failed", exc_info=True)
            if not transient_session and session is not None:
                _invalidate_persistent_native_history(agent, session)
            return run_claude_agent_sdk_turn(
                agent,
                user_message=user_message,
                original_user_message=original_user_message,
                messages=messages,
                conversation_history=conversation_history,
                effective_task_id=effective_task_id,
                turn_id=turn_id,
                current_turn_user_idx=len(messages) - 1,
                active_system_prompt=active_system_prompt,
                should_review_memory=should_review_memory,
                moa_config=moa_config,
                _moa_applied=False,
                _api_calls_before=max(_api_calls_before, total_api_call_count - 1),
                _ack_continuations=_ack_continuations,
                _empty_retries=_empty_retries,
                _post_tool_empty_retried=_post_tool_empty_retried,
                _pending_verification_response=_pending_verification_response,
                _pending_verification_response_previewed=(
                    _pending_verification_response_previewed
                ),
                _length_continue_retries=_length_continue_retries,
                _truncated_response_parts=truncated_response_parts,
                _truncated_parts_previewed=_truncated_parts_previewed,
                _image_shrink_retries=_image_shrink_retries,
            )

    guardrail = getattr(agent, "_tool_guardrail_halt_decision", None)
    if guardrail is not None:
        turn.final_text = agent._toolguard_controlled_halt_response(guardrail)
        if not messages or messages[-1].get("content") != turn.final_text:
            messages.append({"role": "assistant", "content": turn.final_text})
    elif (
        getattr(turn, "budget_exhausted", False)
        and not _pending_verification_response
    ):
        turn.final_text = (
            turn.final_text.strip()
            if isinstance(turn.final_text, str) and turn.final_text.strip()
            else "I stopped because the native Claude turn reached Hermes' iteration limit."
        )

    if (
        guardrail is None
        and turn.error
        and not persistence_failed
        and not turn.interrupted
        and not getattr(turn, "budget_exhausted", False)
    ):
        handoff = _native_fallback_handoff(
            agent,
            messages=messages,
            api_call_count=total_api_call_count,
            reason=(
                getattr(terminal_classified_error, "reason", None)
                or _native_error_failover_reason(agent, str(turn.error))
            ),
            pending_verification_response=_pending_verification_response,
            pending_verification_response_previewed=(
                _pending_verification_response_previewed
            ),
        )
        if handoff is not None:
            if session is not None and not transient_session:
                _invalidate_persistent_native_history(agent, session)
            return handoff
        display_error = _native_display_error(agent, turn.error)
        if not turn.final_text or str(turn.final_text).strip() == str(turn.error).strip():
            turn.final_text = display_error
            messages.append({"role": "assistant", "content": turn.final_text})

    can_continue = not (
        guardrail is not None
        or getattr(turn, "budget_exhausted", False)
        or turn.interrupted
        or turn.error
    )
    length_continuation_exhausted = False
    content_policy_blocked = False
    native_stop_reason = str(
        getattr(turn, "last_stop_reason", "") or ""
    ).strip().lower()
    if can_continue and native_stop_reason == "refusal":
        # Claude's wire-level refusal is a successful provider response, not a
        # transport error and not an empty answer. Match the standard loop:
        # emit the policy error observer, try a different configured provider
        # once, and otherwise surface an actionable terminal refusal without
        # burning three deterministic empty-response retries.
        refusal_text = str(turn.final_text or "").strip()
        try:
            from agent.error_classifier import FailoverReason

            refusal_reason = FailoverReason.content_policy_blocked
            agent._invoke_api_request_error_hook(
                task_id=effective_task_id,
                turn_id=turn_id,
                api_request_id=api_request_id,
                api_call_count=total_api_call_count,
                api_start_time=request_started_at,
                api_kwargs={
                    "model": native_model,
                    "messages": request_messages,
                    "tools": native_tools,
                    "transport": "claude_agent_sdk",
                },
                error_type="ContentPolicyBlocked",
                error_message=(
                    refusal_text
                    or "model declined to respond (content_filter)"
                ),
                status_code=None,
                retry_count=0,
                max_retries=0,
                retryable=False,
                reason=refusal_reason.value,
            )
        except Exception:
            logger.debug(
                "Claude Agent SDK content-policy error hook failed",
                exc_info=True,
            )
            refusal_reason = "content_policy_blocked"

        if getattr(agent, "_has_pending_fallback", lambda: False)():
            agent._buffer_status(
                "⚠️ Model declined to respond (safety refusal) — "
                "trying fallback..."
            )
        handoff = _native_fallback_handoff(
            agent,
            messages=messages,
            api_call_count=total_api_call_count,
            reason=refusal_reason,
            pending_verification_response=_pending_verification_response,
            pending_verification_response_previewed=(
                _pending_verification_response_previewed
            ),
        )
        if handoff is not None:
            if session is not None and not transient_session:
                _invalidate_persistent_native_history(agent, session)
            return handoff

        from agent.conversation_loop import _CONTENT_POLICY_RECOVERY_HINT

        refusal_detail = (
            f"Model's explanation: {refusal_text}"
            if refusal_text
            else "The model returned no explanation."
        )
        turn.final_text = (
            "⚠️  The model declined to respond to this request "
            "(safety refusal — not a Hermes/gateway failure).\n\n"
            f"{refusal_detail}\n\n{_CONTENT_POLICY_RECOVERY_HINT}"
        )
        turn.error = (
            "content_policy_blocked: "
            + (refusal_text or "model declined (content_filter)")
        )
        content_policy_blocked = True
        can_continue = False
        agent._emit_status(
            "⚠️ The model declined to respond to this request "
            "(safety refusal)."
        )
    elif (
        can_continue
        and native_stop_reason
        in {"max_tokens", "model_context_window_exceeded"}
        and bool(getattr(turn, "thinking_budget_exhausted", False))
    ):
        # Match the standard loop's reasoning-only exhaustion branch. There is
        # no visible answer to continue. The configured native output cap is a
        # child-wide startup setting, so repeating the same continuation under
        # that unchanged cap is both expensive and ineffective.
        turn.final_text = (
            "⚠️ **Thinking Budget Exhausted**\n\n"
            "Claude used the available output budget on reasoning and had "
            "none left for the visible response.\n\n"
            "Lower the reasoning effort with `/thinkon low` or `/thinkoff`, "
            "then retry."
        )
        turn.error = (
            "Model used all output tokens on reasoning with none left for "
            "the response. Lower reasoning effort and retry."
        )
        can_continue = False
        agent._emit_status(
            "💭 Claude exhausted its output budget on reasoning; no visible "
            "response was produced."
        )
    elif (
        can_continue
        and native_stop_reason
        in {"max_tokens", "model_context_window_exceeded"}
    ):
        # Claude Code does not automatically finish an output-limited answer:
        # it ends the native turn and waits for another user input.  Preserve
        # the standard Hermes loop's bounded continuation contract instead of
        # surfacing a partial answer as a successful final response.
        partial_text = str(turn.final_text or "")
        if partial_text:
            truncated_response_parts.append(partial_text)
        next_length_retry = _length_continue_retries + 1
        if next_length_retry < 4:
            agent._buffer_status(
                "↻ Claude response reached its output limit — continuing "
                f"({next_length_retry}/4)"
            )
            from agent.conversation_loop import _get_continuation_prompt

            messages.append(
                {
                    "role": "user",
                    "content": _get_continuation_prompt(False),
                }
            )
            agent._session_messages = messages
            try:
                agent._flush_messages_to_session_db(
                    messages, conversation_history
                )
            except Exception:
                logger.debug(
                    "Claude output-length continuation flush failed",
                    exc_info=True,
                )
            return run_claude_agent_sdk_turn(
                agent,
                user_message=user_message,
                original_user_message=original_user_message,
                messages=messages,
                conversation_history=conversation_history,
                effective_task_id=effective_task_id,
                turn_id=turn_id,
                current_turn_user_idx=len(messages) - 1,
                active_system_prompt=active_system_prompt,
                should_review_memory=should_review_memory,
                moa_config=moa_config,
                _moa_applied=False,
                _api_calls_before=total_api_call_count,
                _ack_continuations=_ack_continuations,
                _empty_retries=_empty_retries,
                _post_tool_empty_retried=_post_tool_empty_retried,
                _pending_verification_response=(
                    _pending_verification_response
                ),
                _pending_verification_response_previewed=(
                    _pending_verification_response_previewed
                ),
                _length_continue_retries=next_length_retry,
                _truncated_response_parts=truncated_response_parts,
                _truncated_parts_previewed=(
                    _truncated_parts_previewed
                    and agent._interim_content_was_streamed(partial_text)
                ),
                _image_shrink_retries=_image_shrink_retries,
            )
        length_continuation_exhausted = True
        turn.final_text = "".join(truncated_response_parts).strip()
        turn.error = "Response remained truncated after 4 continuation attempts"
        can_continue = False
    elif can_continue and truncated_response_parts:
        # A later native turn completed normally.  Return the same assembled
        # answer the standard loop returns while keeping each real assistant
        # chunk in the durable role-alternating transcript.
        terminal_text = str(turn.final_text or "")
        turn.final_text = (
            "".join(truncated_response_parts) + terminal_text
        ).strip()
        if (
            _truncated_parts_previewed
            and agent._interim_content_was_streamed(terminal_text)
        ):
            agent._response_was_previewed = True

    if can_continue and not str(turn.final_text or "").strip():
        housekeeping_fallback = _native_housekeeping_fallback(
            agent,
            messages,
            current_turn_user_idx=current_turn_user_idx,
        )
        if housekeeping_fallback:
            logger.info(
                "Claude returned empty after housekeeping tools — using "
                "the already-delivered assistant content"
            )
            agent._emit_status(
                "↻ Empty response after housekeeping tools — using the "
                "earlier content as the final answer"
            )
            turn.final_text = housekeeping_fallback
            agent._response_was_previewed = True

    if can_continue and not str(turn.final_text or "").strip():
        recent_tool = any(
            isinstance(row, dict) and row.get("role") == "tool"
            for row in messages[-8:]
        )
        if recent_tool and not _post_tool_empty_retried:
            nudge = (
                "You just executed tool calls but returned an empty response. "
                "Please process the tool results above and continue with the task."
            )
            next_empty_retries = _empty_retries
            next_post_tool_retry = True
            agent._buffer_status(
                "⚠️ Claude Agent SDK returned empty after tool calls — "
                "nudging it to continue"
            )
            retry_ready = True
        elif _empty_retries < 3:
            next_empty_retries = _empty_retries + 1
            next_post_tool_retry = _post_tool_empty_retried
            nudge = (
                "[System: Your previous response was empty. Continue the "
                "same task now and produce a visible final answer.]"
            )
            retry_ready = _native_empty_retry_wait(agent, next_empty_retries)
        else:
            next_empty_retries = _empty_retries
            next_post_tool_retry = _post_tool_empty_retried
            nudge = ""
            retry_ready = False

        if nudge and retry_ready:
            messages.append(
                {
                    "role": "assistant",
                    "content": "(empty)",
                    "_empty_recovery_synthetic": True,
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": nudge,
                    "_empty_recovery_synthetic": True,
                }
            )
            agent._session_messages = messages
            return run_claude_agent_sdk_turn(
                agent,
                user_message=user_message,
                original_user_message=original_user_message,
                messages=messages,
                conversation_history=conversation_history,
                effective_task_id=effective_task_id,
                turn_id=turn_id,
                current_turn_user_idx=len(messages) - 1,
                active_system_prompt=active_system_prompt,
                should_review_memory=should_review_memory,
                moa_config=moa_config,
                _moa_applied=False,
                _api_calls_before=total_api_call_count,
                _ack_continuations=_ack_continuations,
                _empty_retries=next_empty_retries,
                _post_tool_empty_retried=next_post_tool_retry,
                _pending_verification_response=_pending_verification_response,
                _pending_verification_response_previewed=(
                    _pending_verification_response_previewed
                ),
                _length_continue_retries=_length_continue_retries,
                _truncated_response_parts=truncated_response_parts,
                _truncated_parts_previewed=_truncated_parts_previewed,
                _image_shrink_retries=_image_shrink_retries,
            )

        if nudge and not retry_ready and getattr(agent, "_interrupt_requested", False):
            turn.interrupted = True
            turn.final_text = (
                "Operation interrupted while retrying an empty response from "
                "Claude Agent SDK."
            )
        else:
            handoff = _native_fallback_handoff(
                agent,
                messages=messages,
                api_call_count=total_api_call_count,
                reason=None,
                pending_verification_response=_pending_verification_response,
                pending_verification_response_previewed=(
                    _pending_verification_response_previewed
                ),
            )
            if handoff is not None:
                if session is not None and not transient_session:
                    _invalidate_persistent_native_history(agent, session)
                return handoff
            empty_response_exhausted = True
            turn.final_text = "(empty)"
            messages.append(
                {
                    "role": "assistant",
                    "content": "(empty)",
                    "_empty_terminal_sentinel": True,
                }
            )

    can_continue = bool(
        can_continue and not turn.interrupted and not empty_response_exhausted
    )
    if can_continue and turn.final_text:
        nudge, synthetic_flag, continuation_reason, _ack_continuations = (
            _native_continuation_nudge(
                agent,
                user_message=user_message,
                final_response=turn.final_text,
                messages=messages,
                ack_continuations=_ack_continuations,
            )
        )
        if nudge:
            assistant_row = _last_native_assistant(messages)
            if assistant_row is not None:
                assistant_row["finish_reason"] = continuation_reason
                if continuation_reason == "kanban_terminal_required":
                    assistant_row["_kanban_stop_synthetic"] = True
                assistant_row.pop("_db_persisted", None)
                agent._db_flush_scan_prefix = None
                agent._emit_interim_assistant_message(assistant_row)
            try:
                agent._flush_messages_to_session_db(messages, conversation_history)
            except Exception:
                logger.debug("Claude continuation interim flush failed", exc_info=True)
            user_nudge: dict[str, Any] = {"role": "user", "content": nudge}
            if synthetic_flag:
                user_nudge[synthetic_flag] = True
            messages.append(user_nudge)
            agent._session_messages = messages
            if continuation_reason == "kanban_terminal_required":
                agent._emit_status(
                    "⚠️ Kanban worker tried to exit without "
                    "kanban_complete/kanban_block — nudging to finish"
                )
            pending_response = _pending_verification_response
            pending_previewed = _pending_verification_response_previewed
            if continuation_reason != "intent_ack_continuation":
                pending_response = turn.final_text
                pending_previewed = agent._interim_content_was_streamed(
                    turn.final_text or ""
                )
            return run_claude_agent_sdk_turn(
                agent,
                user_message=user_message,
                original_user_message=original_user_message,
                messages=messages,
                conversation_history=conversation_history,
                effective_task_id=effective_task_id,
                turn_id=turn_id,
                current_turn_user_idx=len(messages) - 1,
                active_system_prompt=active_system_prompt,
                should_review_memory=should_review_memory,
                moa_config=moa_config,
                _moa_applied=False,
                _api_calls_before=total_api_call_count,
                _ack_continuations=_ack_continuations,
                _empty_retries=_empty_retries,
                _post_tool_empty_retried=_post_tool_empty_retried,
                _pending_verification_response=pending_response,
                _pending_verification_response_previewed=pending_previewed,
                _length_continue_retries=_length_continue_retries,
                _truncated_response_parts=truncated_response_parts,
                _truncated_parts_previewed=_truncated_parts_previewed,
                _image_shrink_retries=_image_shrink_retries,
            )

    interrupted = bool(turn.interrupted or getattr(agent, "_interrupt_requested", False))
    failed = bool(turn.error) and not interrupted
    if guardrail is not None:
        exit_reason = "guardrail_halt"
    elif persistence_failed:
        exit_reason = "session_persistence_failed"
    elif getattr(turn, "budget_exhausted", False):
        exit_reason = "claude_agent_sdk_budget_exhausted"
    elif interrupted:
        exit_reason = "interrupted_by_user"
    elif empty_response_exhausted:
        exit_reason = "empty_response_exhausted"
    elif bool(getattr(turn, "thinking_budget_exhausted", False)):
        exit_reason = "claude_agent_sdk_thinking_budget_exhausted"
    elif length_continuation_exhausted:
        exit_reason = "claude_agent_sdk_output_truncated"
    elif content_policy_blocked:
        exit_reason = "content_policy_blocked"
    elif failed:
        exit_reason = "claude_agent_sdk_error"
    else:
        exit_reason = "text_response(claude_agent_sdk)"

    if interrupted:
        clear_status = getattr(agent, "_clear_status_buffer", None)
        if callable(clear_status):
            clear_status()
    elif failed or empty_response_exhausted:
        flush_status = getattr(agent, "_flush_status_buffer", None)
        if callable(flush_status):
            flush_status()
    else:
        emit_fallback_notice = getattr(
            agent, "_emit_pending_fallback_notice", None
        )
        if callable(emit_fallback_notice):
            emit_fallback_notice()
        clear_status = getattr(agent, "_clear_status_buffer", None)
        if callable(clear_status):
            clear_status()

    if moa_config and session is not None and not transient_session:
        # MoA guidance is deliberately request-only on every Hermes transport.
        # Claude's stateful child would otherwise retain the private advisor
        # blocks after this turn even though Hermes' durable/cold-bootstrap
        # transcript does not. Reuse the process for the complete MoA tool
        # loop, then retire it at the real turn boundary so the next ordinary
        # turn starts from an exact Hermes history instead of stale private
        # guidance.
        _invalidate_persistent_native_history(agent, session)
        session = None

    from agent.turn_finalizer import finalize_turn

    result = finalize_turn(
        agent,
        final_response=turn.final_text,
        api_call_count=total_api_call_count,
        interrupted=interrupted,
        failed=failed,
        messages=messages,
        conversation_history=conversation_history,
        effective_task_id=effective_task_id,
        turn_id=turn_id,
        user_message=user_message,
        original_user_message=original_user_message,
        _should_review_memory=should_review_memory,
        _turn_exit_reason=exit_reason,
        _pending_verification_response=_pending_verification_response,
        _pending_verification_response_previewed=(
            _pending_verification_response_previewed
        ),
    )
    result.update({
        "partial": interrupted or (failed and not content_policy_blocked),
        "error": (
            turn.final_text
            if persistence_failed
            else str(turn.error)
            if content_policy_blocked
            else str(turn.error)
            if length_continuation_exhausted
            else _native_display_error(agent, turn.error)
            if turn.error
            else None
        ),
        "agent_persisted": True,
        "claude_session_id": turn.native_session_id,
        "claude_agent_sdk_session_reuse": turn.session_reuse,
        "claude_agent_sdk_latency_ms": turn.latency_ms,
        "claude_fast_mode": _native_fast_mode(agent, _runtime_config()),
    })
    if turn.structured_output is not None:
        result["structured_output"] = turn.structured_output
    terminal_reason = getattr(terminal_classified_error, "reason", None)
    terminal_reason_value = getattr(terminal_reason, "value", None)
    if failed and terminal_reason_value:
        result["failure_reason"] = terminal_reason_value
        try:
            from agent.error_classifier import FailoverReason

            if terminal_reason == FailoverReason.billing:
                from agent.conversation_loop import _billing_block_dict

                result["billing_block"] = _billing_block_dict(
                    str(getattr(agent, "provider", "") or "anthropic"),
                    "claude-agent-sdk://local",
                    str(getattr(agent, "model", "") or ""),
                    str(result.get("final_response") or ""),
                )
            if terminal_reason in {
                FailoverReason.context_overflow,
                FailoverReason.payload_too_large,
                FailoverReason.long_context_tier,
            }:
                if not bool(getattr(agent, "compression_enabled", True)):
                    result["compaction_disabled"] = True
                else:
                    # A terminal overflow means Claude's enabled native
                    # autocompactor already failed to recover. Preserve the
                    # gateway's standard clean-session recovery signal.
                    result["compression_exhausted"] = True
        except Exception:
            logger.debug(
                "Claude terminal error metadata enrichment failed",
                exc_info=True,
            )
    if unsupported_request_controls:
        result["ignored_request_controls"] = unsupported_request_controls
        # The shared result schema reports HTTP service_tier from request
        # overrides. It was not applied to this native subscription request.
        result["service_tier"] = None
    return result


__all__ = [
    "run_claude_agent_sdk_turn",
    "compact_claude_agent_sdk_history",
    "release_claude_agent_sdk_session",
    "detach_claude_agent_sdk_session",
    "close_all_claude_agent_sdk_sessions",
]
