"""OpenAI-shaped auxiliary client backed by isolated Claude Code children.

Claude Code subscription/keychain auth has no exportable HTTP key, so auxiliary work on the main
provider (titles, goal judging, smart approvals, summaries, vision, plugin LLM calls) runs through a
transient native child that never touches the persistent main-session thread
(``runtime.run_claude_cli_auxiliary_completion``). Served by ``ClaudeCliTurnRuntime.auxiliary_client``
via ``agent/auxiliary_turn_runtime.py``.
"""

from __future__ import annotations

import asyncio
import contextvars
import threading
from types import SimpleNamespace
from typing import Any, Optional

_STOP_REASONS = {
    "end_turn": "stop", "stop_sequence": "stop", "tool_use": "tool_calls", "max_tokens": "length",
    "model_context_window_exceeded": "length", "refusal": "content_filter",
}


def _json_schema(kwargs: dict[str, Any]) -> Optional[dict[str, Any]]:
    extra_body = kwargs.get("extra_body")
    response_format = kwargs.get("response_format")
    if response_format is None and isinstance(extra_body, dict):
        response_format = extra_body.get("response_format")
    if not isinstance(response_format, dict):
        return None
    format_type = str(response_format.get("type") or "").strip()
    if format_type == "json_schema":
        wrapper = response_format.get("json_schema")
        schema = wrapper.get("schema") if isinstance(wrapper, dict) else None
        return schema if isinstance(schema, dict) else None
    if format_type == "json_object":
        return {"type": "object"}
    return None


def _reasoning_config(kwargs: dict[str, Any]) -> Any:
    config = kwargs.get("_reasoning_config")
    extra_body = kwargs.get("extra_body")
    if config is None and isinstance(extra_body, dict) and isinstance(extra_body.get("reasoning"), dict):
        config = extra_body["reasoning"]
    return config


def _tool_calls(rows: list) -> tuple[list, Any]:
    reasoning, calls = None, []
    for row in reversed(rows or []):
        if not (isinstance(row, dict) and row.get("role") == "assistant"):
            continue
        reasoning = reasoning or row.get("reasoning") or None
        if row.get("tool_calls") and not calls:
            for call in row.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                function = call.get("function") or {}
                calls.append(SimpleNamespace(
                    id=str(call.get("id") or ""), type="function",
                    function=SimpleNamespace(name=str(function.get("name") or ""),
                                             arguments=str(function.get("arguments") or "{}"))))
        if reasoning and calls:
            break
    return calls, reasoning


class _ClaudeCliCompletionsAdapter:
    """``chat.completions.create`` over one isolated native completion."""

    def __init__(self, model: str):
        self._model = model

    def create(self, **kwargs: Any) -> Any:
        from plugins.claude_cli.runtime import run_claude_cli_auxiliary_completion

        model = str(kwargs.get("model") or self._model or "")
        native_kwargs: dict[str, Any] = {
            "messages": kwargs.get("messages") or [], "model": model, "timeout": kwargs.get("timeout"),
            "reasoning_config": _reasoning_config(kwargs),
        }
        max_tokens = kwargs.get("max_tokens")
        if max_tokens is None:
            max_tokens = kwargs.get("max_completion_tokens")
        if max_tokens is not None:
            native_kwargs["max_tokens"] = max_tokens
        schema = _json_schema(kwargs)
        if schema is not None:
            native_kwargs["json_schema"] = schema
        if kwargs.get("tools"):
            native_kwargs["tools"] = kwargs["tools"]
            if kwargs.get("tool_choice") is not None:
                native_kwargs["tool_choice"] = kwargs["tool_choice"]
        result = run_claude_cli_auxiliary_completion(**native_kwargs)

        raw_usage = getattr(result, "token_usage", None) or {}
        prompt_tokens = sum(int(raw_usage.get(key, 0) or 0) for key in (
            "input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"))
        completion_tokens = int(raw_usage.get("output_tokens", 0) or 0)
        calls, reasoning = _tool_calls(getattr(result, "projected_messages", None) or [])
        message = SimpleNamespace(role="assistant", content=str(getattr(result, "final_text", "") or ""),
                                  tool_calls=calls or None, reasoning=reasoning)
        stop = str(getattr(result, "last_stop_reason", "") or "").strip().lower()
        finish_reason = "tool_calls" if calls else _STOP_REASONS.get(stop, stop or "stop")
        return SimpleNamespace(
            choices=[SimpleNamespace(index=0, message=message, finish_reason=finish_reason)], model=model,
            usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                                  total_tokens=prompt_tokens + completion_tokens))


class _ChatShim:
    def __init__(self, completions: Any):
        self.completions = completions


class ClaudeCliAuxiliaryClient:
    """Chat-completions facade backed by Claude Code's own login."""

    def __init__(self, model: str):
        self.chat = _ChatShim(_ClaudeCliCompletionsAdapter(model))
        self.api_key = ""
        self.base_url = "claude-cli://local"

    def close(self) -> None:
        return None


class _AsyncClaudeCliCompletionsAdapter:
    def __init__(self, sync_adapter: _ClaudeCliCompletionsAdapter):
        self._sync = sync_adapter

    async def create(self, **kwargs: Any) -> Any:
        # Not asyncio's process-global default executor: the Claude child is already a blocking subprocess
        # boundary, and hosts that create/tear down short-lived loops would wait on its shutdown. A one-shot
        # daemon thread keeps ContextVars (runtime/provider routing) without becoming loop-owned state.
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        context = contextvars.copy_context()

        def _deliver(value: Any = None, error: Optional[BaseException] = None) -> None:
            if future.done():
                return
            if error is not None:
                future.set_exception(error)
            else:
                future.set_result(value)

        def _run() -> None:
            try:
                value = context.run(self._sync.create, **kwargs)
            except BaseException as exc:
                loop.call_soon_threadsafe(_deliver, None, exc)
            else:
                loop.call_soon_threadsafe(_deliver, value, None)

        threading.Thread(target=_run, daemon=True, name="hermes-claude-cli-aux").start()
        return await future


class AsyncClaudeCliAuxiliaryClient:
    def __init__(self, sync_wrapper: ClaudeCliAuxiliaryClient):
        self.chat = _ChatShim(_AsyncClaudeCliCompletionsAdapter(sync_wrapper.chat.completions))
        self.api_key = sync_wrapper.api_key
        self.base_url = sync_wrapper.base_url

    async def close(self) -> None:
        return None
