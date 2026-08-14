"""Persistent Claude Agent SDK transport for Hermes.

One SDK client stays connected on a dedicated asyncio thread for the lifetime
of each pooled Hermes conversation. Hermes supplies the stable system prompt
and exact tool surface; Claude owns native history and model/tool iteration
scheduling.
"""

from __future__ import annotations

import asyncio
import base64
import collections
import concurrent.futures
import json
import logging
import os
import queue
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

from agent.claude_agent_sdk_bridge import ClaudeAgentSdkToolBridge
from agent.transports.claude_agent_sdk_common import (
    ClaudeAgentSdkTurnResult,
    _bootstrap_image_blocks,
    _coerce_text,
    _compose_user_content,
    _forget_binding,
    _hermes_history_signature,
    _load_binding,
    _load_binding_history_signature,
    _previous_tool_batch,
    _save_binding,
    forget_claude_agent_sdk_binding,
    serialize_history_for_bootstrap,
)


logger = logging.getLogger(__name__)

_STDERR_TAIL_LINES = 80
_SDK_TURN_END = object()
_PARTIAL_BATCH_ASSEMBLY_SECONDS = 1.0
_PROCESS_ENV_CLEAR = {
    "API_TIMEOUT_MS",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_API_KEY_OLD",
    "ANTHROPIC_API_TOKEN",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_CUSTOM_HEADERS",
    "ANTHROPIC_OAUTH_TOKEN",
    "ANTHROPIC_UNIX_SOCKET",
    "CLAUDE_CODE_API_KEY_FILE_DESCRIPTOR",
    "CLAUDE_CODE_DISABLE_FAST_MODE",
    "CLAUDE_CODE_MAX_OUTPUT_TOKENS",
    "CLAUDE_CODE_MAX_RETRIES",
    "CLAUDE_CODE_OAUTH_REFRESH_TOKEN",
    "CLAUDE_CODE_OAUTH_SCOPES",
    "CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_FOUNDRY",
    "CLAUDE_CODE_USE_VERTEX",
}


def _strip_mcp_prefix(name: str) -> str:
    prefix = "mcp__hermes__"
    return name[len(prefix) :] if name.startswith(prefix) else name


def _content_blocks(message: Any) -> list[dict[str, Any]]:
    if not isinstance(message, dict):
        return []
    content = message.get("content")
    if not isinstance(content, list):
        return []
    return [entry for entry in content if isinstance(entry, dict)]


def _normalized_mcp_content(result: Any) -> list[dict[str, str]]:
    """Translate Hermes text/multimodal tool content into MCP content."""
    if not isinstance(result, list):
        text = (
            result
            if isinstance(result, str)
            else json.dumps(result, ensure_ascii=False, default=str)
        )
        return [{"type": "text", "text": text}]
    normalized: list[dict[str, str]] = []
    for part in result:
        if not isinstance(part, dict):
            normalized.append({"type": "text", "text": str(part)})
            continue
        part_type = str(part.get("type") or "")
        if part_type in {"text", "input_text"}:
            normalized.append({
                "type": "text",
                "text": str(part.get("text") or part.get("content") or ""),
            })
            continue
        source = part.get("source")
        if part_type == "image" and isinstance(source, dict):
            if source.get("type") == "base64":
                data = str(source.get("data") or "")
                try:
                    base64.b64decode(data, validate=True)
                except (TypeError, ValueError):
                    data = ""
                if data:
                    normalized.append({
                        "type": "image",
                        "data": data,
                        "mimeType": str(source.get("media_type") or "image/png"),
                    })
                    continue
        image_ref = part.get("image_url")
        if isinstance(image_ref, dict):
            image_ref = image_ref.get("url")
        if not image_ref:
            image_ref = part.get("url")
        if part_type in {"image_url", "input_image", "image"} and isinstance(
            image_ref, str
        ):
            if image_ref.startswith("data:"):
                header, separator, data = image_ref.partition(",")
                if separator and ";base64" in header.lower():
                    try:
                        base64.b64decode(data, validate=True)
                    except (TypeError, ValueError):
                        data = ""
                    if data:
                        normalized.append({
                            "type": "image",
                            "data": data,
                            "mimeType": header[5:].split(";", 1)[0] or "image/png",
                        })
                        continue
            normalized.append({
                "type": "text",
                "text": f"[Image available at URL: {image_ref}]",
            })
            continue
        normalized.append({
            "type": "text",
            "text": json.dumps(part, ensure_ascii=False, default=str),
        })
    return normalized or [{"type": "text", "text": "(no output)"}]


class _AsyncRuntimeThread:
    """Own one asyncio context for an SDK client for its complete lifetime."""

    def __init__(self, owner_key: str) -> None:
        self.owner_key = owner_key
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._ready = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name=f"hermes-claude-sdk-{owner_key[:24]}",
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(timeout=10.0):
            raise RuntimeError("Claude Agent SDK async runtime did not start")

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        # Some hardened containers deny writes to asyncio's socketpair wakeup
        # fd (EPERM).  ``call_soon_threadsafe`` deliberately swallows that
        # OSError, leaving submitted work queued forever while select() has no
        # deadline.  Detect that exact platform limitation and install a tiny
        # timer tick only there; normal hosts keep the zero-wakeup idle loop.
        needs_poll_fallback = False
        wake_socket = getattr(loop, "_csock", None)
        if wake_socket is not None:
            try:
                wake_socket.send(b"\0")
            except OSError:
                needs_poll_fallback = True
        if needs_poll_fallback:

            def _poll_threadsafe_queue() -> None:
                if not loop.is_closed():
                    loop.call_later(0.01, _poll_threadsafe_queue)

            loop.call_soon(_poll_threadsafe_queue)
        self._ready.set()
        try:
            loop.run_forever()
        finally:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()

    @property
    def is_alive(self) -> bool:
        return self._thread.is_alive() and self._loop is not None

    def submit(self, coroutine: Any) -> concurrent.futures.Future[Any]:
        loop = self._loop
        if loop is None or not self._thread.is_alive():
            raise RuntimeError("Claude Agent SDK async runtime is unavailable")
        return asyncio.run_coroutine_threadsafe(coroutine, loop)

    def call(self, coroutine: Any, *, timeout: Optional[float] = None) -> Any:
        return self.submit(coroutine).result(timeout=timeout)

    def stop(self) -> None:
        loop = self._loop
        if loop is not None and self._thread.is_alive():
            loop.call_soon_threadsafe(loop.stop)
            self._thread.join(timeout=10.0)
        self._loop = None


def _sdk_message_event(message: Any) -> dict[str, Any]:
    """Convert typed Agent SDK messages to Claude's stable stream-json shape."""
    kind = type(message).__name__
    if kind == "StreamEvent":
        return {
            "type": "stream_event",
            "uuid": getattr(message, "uuid", ""),
            "session_id": getattr(message, "session_id", ""),
            "event": getattr(message, "event", {}) or {},
        }
    if kind == "SystemMessage":
        data = dict(getattr(message, "data", {}) or {})
        data.setdefault("type", "system")
        data.setdefault("subtype", getattr(message, "subtype", ""))
        return data
    if kind == "AssistantMessage":
        blocks = []
        for block in getattr(message, "content", []) or []:
            block_kind = type(block).__name__
            if block_kind == "TextBlock":
                blocks.append({"type": "text", "text": getattr(block, "text", "")})
            elif block_kind == "ThinkingBlock":
                blocks.append({
                    "type": "thinking",
                    "thinking": getattr(block, "thinking", ""),
                    "signature": getattr(block, "signature", ""),
                })
            elif block_kind == "ToolUseBlock":
                blocks.append({
                    "type": "tool_use",
                    "id": getattr(block, "id", ""),
                    "name": getattr(block, "name", ""),
                    "input": getattr(block, "input", {}) or {},
                })
            elif block_kind == "ServerToolUseBlock":
                blocks.append({
                    "type": "server_tool_use",
                    "id": getattr(block, "id", ""),
                    "name": getattr(block, "name", ""),
                    "input": getattr(block, "input", {}) or {},
                })
            elif block_kind == "ServerToolResultBlock":
                blocks.append({
                    "type": "server_tool_result",
                    "tool_use_id": getattr(block, "tool_use_id", ""),
                    "content": getattr(block, "content", None),
                })
        event = {
            "type": "assistant",
            "session_id": getattr(message, "session_id", ""),
            "message": {
                "id": getattr(message, "message_id", "") or "",
                "model": getattr(message, "model", "") or "",
                "content": blocks,
                "usage": getattr(message, "usage", None),
                "stop_reason": getattr(message, "stop_reason", None),
            },
        }
        assistant_error = getattr(message, "error", None)
        if assistant_error is not None:
            event["message"]["error"] = assistant_error
        return event
    if kind == "UserMessage":
        content = getattr(message, "content", "")
        if isinstance(content, list):
            blocks = []
            for block in content:
                block_kind = type(block).__name__
                if block_kind == "TextBlock":
                    blocks.append({"type": "text", "text": getattr(block, "text", "")})
                elif block_kind == "ToolResultBlock":
                    blocks.append({
                        "type": "tool_result",
                        "tool_use_id": getattr(block, "tool_use_id", ""),
                        "content": getattr(block, "content", None),
                        "is_error": getattr(block, "is_error", None),
                    })
            content = blocks
        return {"type": "user", "message": {"role": "user", "content": content}}
    if kind == "ResultMessage":
        return {
            "type": "result",
            "subtype": getattr(message, "subtype", ""),
            "duration_ms": getattr(message, "duration_ms", 0),
            "duration_api_ms": getattr(message, "duration_api_ms", 0),
            "is_error": bool(getattr(message, "is_error", False)),
            "num_turns": getattr(message, "num_turns", 0),
            "session_id": getattr(message, "session_id", ""),
            "stop_reason": getattr(message, "stop_reason", None),
            "result": getattr(message, "result", None),
            "structured_output": getattr(message, "structured_output", None),
            "usage": getattr(message, "usage", None),
            "errors": getattr(message, "errors", None),
            "error_status": getattr(message, "api_error_status", None),
            "terminal_reason": getattr(message, "terminal_reason", None),
        }
    return {"type": "_ignored", "sdk_type": kind}


class ClaudeAgentSdkSession:
    """One reusable Agent SDK client and native Claude history binding."""

    def __init__(
        self,
        *,
        owner_key: str,
        agent: Any,
        cwd: str,
        model: str,
        system_prompt: str,
        command: str = "",
        extra_args: Optional[list[str]] = None,
        turn_timeout: float = 600.0,
        reasoning_effort: Optional[str] = None,
        thinking_mode: Optional[str] = None,
        fast_mode: bool = False,
        max_output_tokens: Optional[int] = None,
        api_retry_count: Optional[int] = None,
        provider_request_timeout: Optional[float] = None,
        persistent_binding: bool = True,
        tool_definitions: Optional[list[dict[str, Any]]] = None,
        auto_compaction_enabled: bool = True,
        json_schema: Optional[dict[str, Any]] = None,
    ) -> None:
        self.owner_key = owner_key
        self.cwd = cwd
        self.model = model.split("/", 1)[-1]
        self.system_prompt = system_prompt
        self.command = command
        self.extra_args = list(extra_args or [])
        self.turn_timeout = max(1.0, float(turn_timeout))
        self.reasoning_effort = reasoning_effort
        self.thinking_mode = thinking_mode
        self.fast_mode = bool(fast_mode)
        self.max_output_tokens = (
            int(max_output_tokens)
            if max_output_tokens is not None and int(max_output_tokens) > 0
            else None
        )
        self.api_retry_count = (
            max(0, int(api_retry_count)) if api_retry_count is not None else None
        )
        self.provider_request_timeout = (
            float(provider_request_timeout)
            if provider_request_timeout is not None
            and float(provider_request_timeout) > 0
            else None
        )
        self.persistent_binding = bool(persistent_binding)
        self.auto_compaction_enabled = bool(auto_compaction_enabled)
        self.json_schema = dict(json_schema) if isinstance(json_schema, dict) else None
        self.bridge = ClaudeAgentSdkToolBridge(
            agent,
            tool_definitions=tool_definitions,
            owner_key=owner_key,
        )
        self.tool_fingerprint = self.bridge.fingerprint()
        self.native_session_id = (
            _load_binding(owner_key) if self.persistent_binding else None
        )
        self._resume = bool(self.native_session_id)
        self._history_signature = (
            _load_binding_history_signature(owner_key)
            if self.persistent_binding and self._resume
            else None
        )
        self._runtime: Optional[_AsyncRuntimeThread] = None
        self._client: Any = None
        self._transport: Any = None
        self._runtime_dir: Optional[Path] = None
        self._stderr: collections.deque[str] = collections.deque(
            maxlen=_STDERR_TAIL_LINES
        )
        self._turn_lock = threading.Lock()
        self._interrupt_requested = threading.Event()
        self._closed = False
        self._created_at = time.monotonic()
        self._last_used_at = self._created_at
        self._process_started_at: Optional[float] = None
        self._turns_completed = 0

    @property
    def is_alive(self) -> bool:
        if (
            self._closed
            or self._client is None
            or self._runtime is None
            or not self._runtime.is_alive
        ):
            return False
        process = getattr(self._transport, "_process", None)
        return process is None or getattr(process, "returncode", None) is None

    @property
    def is_busy(self) -> bool:
        return self._turn_lock.locked()

    @property
    def last_used_at(self) -> float:
        return self._last_used_at

    def bind_agent(self, agent: Any) -> None:
        self.bridge.bind_agent(agent)

    def sync_history_signature(self, messages: list[dict[str, Any]]) -> None:
        self._history_signature = _hermes_history_signature(messages)
        if self.persistent_binding and self.native_session_id:
            _save_binding(
                self.owner_key,
                self.native_session_id,
                history_signature=self._history_signature,
            )

    def compatible(
        self,
        *,
        cwd: str,
        model: str,
        command: str,
        extra_args: list[str],
        tool_fingerprint: str,
        system_prompt: str,
        reasoning_effort: Optional[str],
        thinking_mode: Optional[str],
        fast_mode: bool,
        max_output_tokens: Optional[int],
        api_retry_count: Optional[int],
        turn_timeout: float,
        provider_request_timeout: Optional[float],
        persistent_binding: bool,
        auto_compaction_enabled: bool = True,
        json_schema: Optional[dict[str, Any]] = None,
    ) -> bool:
        return (
            os.path.realpath(cwd) == os.path.realpath(self.cwd)
            and model.split("/", 1)[-1] == self.model
            and command == self.command
            and list(extra_args) == self.extra_args
            and tool_fingerprint == self.tool_fingerprint
            and system_prompt == self.system_prompt
            and reasoning_effort == self.reasoning_effort
            and thinking_mode == self.thinking_mode
            and bool(fast_mode) is self.fast_mode
            and max_output_tokens == self.max_output_tokens
            and api_retry_count == self.api_retry_count
            and float(turn_timeout) == self.turn_timeout
            and provider_request_timeout == self.provider_request_timeout
            and bool(persistent_binding) is self.persistent_binding
            and bool(auto_compaction_enabled) is self.auto_compaction_enabled
            and (dict(json_schema) if isinstance(json_schema, dict) else None)
            == self.json_schema
        )

    def _system_prompt_path(self) -> Path:
        if self._runtime_dir is None:
            self._runtime_dir = Path(tempfile.mkdtemp(prefix="hermes-claude-sdk-"))
            os.chmod(self._runtime_dir, 0o700)
        path = self._runtime_dir / "system-prompt.md"
        path.write_text(self.system_prompt, encoding="utf-8")
        os.chmod(path, 0o600)
        return path

    def _build_env(self) -> dict[str, str]:
        # Agent SDK merges this mapping over inherited environment variables;
        # empty values preserve the old runtime's credential-isolation rule.
        env = {key: "" for key in _PROCESS_ENV_CLEAR}
        env.update({
            "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
            "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1",
            "CLAUDE_CODE_DISABLE_BUNDLED_SKILLS": "1",
            "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",
            "CLAUDE_CODE_DISABLE_CRON": "1",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "CLAUDE_CODE_DISABLE_OFFICIAL_MARKETPLACE_AUTOINSTALL": "1",
            "CLAUDE_CODE_SKIP_PLUGIN_MCP_SERVERS": "1",
            "DISABLE_AUTOUPDATER": "1",
        })
        if not self.fast_mode:
            env["CLAUDE_CODE_DISABLE_FAST_MODE"] = "1"
        if self.max_output_tokens is not None:
            env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(self.max_output_tokens)
        if self.api_retry_count is not None:
            env["CLAUDE_CODE_MAX_RETRIES"] = str(self.api_retry_count)
        request_timeout = (
            self.provider_request_timeout
            if self.provider_request_timeout is not None
            else max(1.0, self.turn_timeout - 5.0)
        )
        env["API_TIMEOUT_MS"] = str(max(1_000, int(request_timeout * 1_000)))
        if not self.auto_compaction_enabled:
            env["DISABLE_COMPACT"] = "1"
        return env

    def _ensure_sdk_dependency(self) -> None:
        try:
            from tools.lazy_deps import ensure

            ensure("provider.claude_agent_sdk", prompt=False)
            import claude_agent_sdk  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                "Claude Agent SDK runtime requires its optional dependency. Install with "
                "`pip install 'hermes-agent[claude-agent-sdk]'`."
            ) from exc

    def _resolved_cli_path(self) -> Optional[str]:
        if not self.command:
            return None
        resolved = shutil.which(self.command)
        if resolved is None:
            candidate = Path(self.command).expanduser()
            if not candidate.is_absolute():
                candidate = Path(self.cwd) / candidate
            if candidate.is_file():
                resolved = str(candidate.resolve())
        if resolved is None:
            raise RuntimeError(
                f"Configured Claude Code executable not found: {self.command}"
            )
        return resolved

    def _stderr_line(self, line: str) -> None:
        self._stderr.append(str(line).rstrip())

    def stderr_tail(self) -> str:
        return "\n".join(self._stderr)

    def _error_with_stderr(self, detail: Any) -> str:
        """Include the child process stderr in transport failures."""
        message = str(detail or "Claude Agent SDK transport failed")
        stderr = self.stderr_tail().strip()
        return f"{message}\n{stderr}" if stderr else message

    def _sdk_tools(self) -> list[Any]:
        from claude_agent_sdk import tool

        sdk_tools = []
        for definition in self.bridge.tool_definitions():
            function = definition.get("function") or {}
            name = str(function.get("name") or "").strip()
            if not name:
                continue
            description = str(function.get("description") or "")
            schema = function.get("parameters") or {
                "type": "object",
                "properties": {},
            }

            async def _handler(
                arguments: dict[str, Any], _name: str = name
            ) -> dict[str, Any]:
                # Query routes stdout control requests concurrently. Partial
                # message events are consumed on this same SDK loop, so wait
                # only for the already-emitted message_stop to seal the full
                # parallel batch. Never wait for the later authoritative
                # AssistantMessage: Claude may withhold it until this MCP
                # result returns. The PreToolUse singleton remains a bounded
                # fallback for any future CLI that omits partial events.
                deadline = time.monotonic() + min(
                    _PARTIAL_BATCH_ASSEMBLY_SECONDS,
                    self.turn_timeout,
                )
                while (
                    not self.bridge.has_partial_batch(
                        name=_name, arguments=arguments or {}
                    )
                    and time.monotonic() < deadline
                ):
                    await asyncio.sleep(0.001)
                result = await asyncio.to_thread(
                    self.bridge._call_tool,
                    {"name": _name, "arguments": arguments or {}},
                )
                return {"content": _normalized_mcp_content(result)}

            sdk_tools.append(tool(name, description, schema)(_handler))
        return sdk_tools

    async def _sdk_connect(self) -> None:
        if self._client is not None:
            return
        from claude_agent_sdk import (
            ClaudeAgentOptions,
            ClaudeSDKClient,
            HookMatcher,
            create_sdk_mcp_server,
        )
        from claude_agent_sdk._internal.transport.subprocess_cli import (
            SubprocessCLITransport,
        )

        sdk_tools = self._sdk_tools()

        async def _pre_tool_use(
            hook_input: dict[str, Any],
            tool_use_id: Optional[str],
            _context: dict[str, Any],
        ) -> dict[str, Any]:
            raw_name = str(hook_input.get("tool_name") or "")
            arguments = hook_input.get("tool_input")
            arguments = dict(arguments) if isinstance(arguments, dict) else {}
            native_id = str(hook_input.get("tool_use_id") or tool_use_id or "")
            if not raw_name.startswith("mcp__hermes__") or not native_id:
                return {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": (
                            "Hermes permits only tools from its in-process SDK MCP server."
                        ),
                    }
                }
            self.bridge.register_tool_request(
                name=_strip_mcp_prefix(raw_name),
                arguments=arguments,
                claude_id=native_id,
            )
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "allow",
                    "updatedInput": arguments,
                }
            }

        has_tools = bool(sdk_tools)
        mcp_servers = (
            {
                "hermes": {
                    **create_sdk_mcp_server("hermes", tools=sdk_tools),
                    # Current Claude Code initializes SDK MCP servers lazily.
                    # Require the Hermes bridge on the first model turn so its
                    # tools cannot be omitted from the initial tool surface.
                    "alwaysLoad": True,
                }
            }
            if has_tools
            else {}
        )
        extra_args: dict[str, Optional[str]] = {
            "disable-slash-commands": None,
            "prompt-suggestions": "false",
            "no-chrome": None,
            "replay-user-messages": None,
        }
        if not self.persistent_binding:
            extra_args["no-session-persistence"] = None
        options = ClaudeAgentOptions(
            tools=["ToolSearch"] if has_tools else [],
            allowed_tools=["mcp__hermes__*"] if has_tools else [],
            system_prompt={"type": "file", "path": str(self._system_prompt_path())},
            mcp_servers=mcp_servers,
            strict_mcp_config=True,
            model=self.model or None,
            cwd=self.cwd,
            cli_path=self._resolved_cli_path(),
            settings=(
                json.dumps({"fastMode": True}, separators=(",", ":"))
                if self.fast_mode
                else None
            ),
            env=self._build_env(),
            extra_args=extra_args,
            stderr=self._stderr_line,
            include_partial_messages=True,
            max_buffer_size=8 * 1024 * 1024,
            hooks={
                "PreToolUse": [
                    HookMatcher(matcher="mcp__hermes__.*", hooks=[_pre_tool_use])
                ]
            }
            if has_tools
            else None,
            setting_sources=[],
            skills=[],
            resume=self.native_session_id if self._resume else None,
            session_id=(
                None if self._resume else self.native_session_id or str(uuid.uuid4())
            ),
            effort=self.reasoning_effort,
            thinking=(
                {"type": "disabled"} if self.thinking_mode == "disabled" else None
            ),
            output_format=(
                {"type": "json_schema", "schema": self.json_schema}
                if self.json_schema is not None
                else None
            ),
        )
        if not self.native_session_id and options.session_id:
            self.native_session_id = str(options.session_id)

        raw_extra_args = list(self.extra_args)
        using_version_matched_bundle = not bool(self.command)

        class _ConfiguredCliTransport(SubprocessCLITransport):
            def _build_command(transport_self) -> list[str]:
                command = super()._build_command()
                return [command[0], *raw_extra_args, *command[1:]]

            async def _check_claude_version(transport_self) -> None:
                # The pinned SDK wheel and its bundled CLI are released as one
                # versioned artifact. Spawning a second `claude -v` process on
                # every cold session can only re-confirm that invariant and is
                # measurable on first-token latency. Keep the SDK's check for
                # an operator-supplied external executable.
                if using_version_matched_bundle:
                    return
                await super()._check_claude_version()

        transport = _ConfiguredCliTransport(_empty_prompt_stream(), options)
        client = ClaudeSDKClient(options=options, transport=transport)
        try:
            await client.connect()
            if has_tools:
                status = await client.get_mcp_status()
                servers = status.get("mcpServers") if isinstance(status, dict) else None
                rows = servers if isinstance(servers, list) else []
                hermes = next(
                    (
                        row
                        for row in rows
                        if isinstance(row, dict)
                        and str(row.get("name") or "") == "hermes"
                    ),
                    None,
                )
                if hermes is None:
                    # Claude Code may not materialize a lazily registered SDK
                    # MCP server until the first prompt.  The bridge is still
                    # fail-safe here: only mcp__hermes__* is allowed, and an
                    # absent server exposes no effecting tools.  alwaysLoad
                    # above makes the server mandatory for that first turn.
                    logger.debug(
                        "Hermes SDK MCP status deferred until the first prompt"
                    )
                elif str(hermes.get("status") or "").lower() not in {
                    "connected",
                    "pending",
                }:
                    detail = str(
                        hermes.get("error") or hermes.get("status") or "unknown"
                    )
                    raise RuntimeError(
                        "Hermes SDK MCP server failed to initialize: " + detail[:300]
                    )
        except BaseException:
            await client.disconnect()
            raise
        self._transport = transport
        self._client = client
        self._process_started_at = time.monotonic()
        if self.persistent_binding and self.native_session_id:
            _save_binding(self.owner_key, self.native_session_id)
        logger.info(
            "Claude Agent SDK session connected (owner=%s resume=%s)",
            self.owner_key,
            self._resume,
        )

    def ensure_started(self) -> None:
        if self._closed:
            raise RuntimeError("Claude Agent SDK session is closed")
        if self.is_alive:
            return
        self._ensure_sdk_dependency()
        if self._runtime is None:
            self._runtime = _AsyncRuntimeThread(self.owner_key)
        self._runtime.call(self._sdk_connect(), timeout=min(90.0, self.turn_timeout))

    @staticmethod
    def _stream_delta(event: dict[str, Any]) -> str:
        if event.get("type") != "stream_event":
            return ""
        inner = event.get("event")
        if not isinstance(inner, dict) or inner.get("type") != "content_block_delta":
            return ""
        delta = inner.get("delta")
        if not isinstance(delta, dict) or delta.get("type") != "text_delta":
            return ""
        return str(delta.get("text") or "")

    @staticmethod
    def _reasoning_delta(event: dict[str, Any]) -> str:
        if event.get("type") != "stream_event":
            return ""
        inner = event.get("event")
        if not isinstance(inner, dict) or inner.get("type") != "content_block_delta":
            return ""
        delta = inner.get("delta")
        if not isinstance(delta, dict) or delta.get("type") != "thinking_delta":
            return ""
        return str(delta.get("thinking") or delta.get("text") or "")

    @staticmethod
    def _tool_gen_started(event: dict[str, Any]) -> Optional[str]:
        if event.get("type") != "stream_event":
            return None
        inner = event.get("event")
        if not isinstance(inner, dict) or inner.get("type") != "content_block_start":
            return None
        block = inner.get("content_block")
        if not isinstance(block, dict) or block.get("type") != "tool_use":
            return None
        name = str(block.get("name") or "")
        return _strip_mcp_prefix(name) if name.startswith("mcp__hermes__") else None

    @staticmethod
    def _project_record(
        event: dict[str, Any],
        allowed_tool_ids: Optional[set[str]] = None,
    ) -> tuple[list[dict[str, Any]], int]:
        event_type = event.get("type")
        message = event.get("message")
        blocks = _content_blocks(message)
        if event_type == "assistant":
            text = "".join(
                str(block.get("text") or "")
                for block in blocks
                if block.get("type") == "text"
            )
            reasoning = "".join(
                str(block.get("thinking") or block.get("text") or "")
                for block in blocks
                if block.get("type") == "thinking"
            )
            all_uses = [block for block in blocks if block.get("type") == "tool_use"]
            uses = [
                block
                for block in all_uses
                if str(block.get("name") or "").startswith("mcp__hermes__")
            ]
            if uses:
                tool_calls = []
                for block in uses:
                    tool_calls.append({
                        "id": str(block.get("id") or uuid.uuid4().hex),
                        "type": "function",
                        "function": {
                            "name": _strip_mcp_prefix(str(block.get("name") or "")),
                            "arguments": json.dumps(
                                block.get("input") or {}, ensure_ascii=False
                            ),
                        },
                    })
                row: dict[str, Any] = {
                    "role": "assistant",
                    "content": text or None,
                    "tool_calls": tool_calls,
                }
                stop_reason = str(message.get("stop_reason") or "").strip()
                if stop_reason:
                    row["finish_reason"] = (
                        "length"
                        if stop_reason
                        in {"max_tokens", "model_context_window_exceeded"}
                        else stop_reason
                    )
                if reasoning:
                    row["reasoning"] = reasoning
                return [row], len(uses)
            if text and not all_uses:
                row = {"role": "assistant", "content": text}
                stop_reason = str(message.get("stop_reason") or "").strip()
                if stop_reason:
                    row["finish_reason"] = (
                        "length"
                        if stop_reason
                        in {"max_tokens", "model_context_window_exceeded"}
                        else stop_reason
                    )
                if reasoning:
                    row["reasoning"] = reasoning
                return [row], 0
        if event_type == "user":
            results: list[dict[str, Any]] = []
            for block in blocks:
                if block.get("type") != "tool_result":
                    continue
                tool_use_id = str(block.get("tool_use_id") or "")
                if allowed_tool_ids is not None and tool_use_id not in allowed_tool_ids:
                    continue
                content = block.get("content")
                if isinstance(content, list):
                    content = "\n".join(
                        str(part.get("text") or "")
                        for part in content
                        if isinstance(part, dict) and part.get("type") == "text"
                    )
                results.append({
                    "role": "tool",
                    "tool_call_id": tool_use_id,
                    "content": "" if content is None else str(content),
                })
            return results, 0
        return [], 0

    @staticmethod
    def _usage(event: dict[str, Any]) -> dict[str, int]:
        raw = event.get("usage")
        if not isinstance(raw, dict):
            message = event.get("message")
            raw = message.get("usage") if isinstance(message, dict) else None
        if not isinstance(raw, dict):
            return {}
        keys = (
            "input_tokens",
            "output_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
        )
        return {
            key: int(raw[key]) for key in keys if isinstance(raw.get(key), (int, float))
        }

    @classmethod
    def _last_iteration_usage(cls, event: dict[str, Any]) -> dict[str, int]:
        raw = event.get("usage")
        if not isinstance(raw, dict) or not isinstance(raw.get("iterations"), list):
            return {}
        for iteration in reversed(raw["iterations"]):
            if not isinstance(iteration, dict):
                continue
            nested = iteration.get("usage")
            candidate = nested if isinstance(nested, dict) else iteration
            usage = cls._usage({"usage": candidate})
            if usage:
                return usage
        return {}

    @staticmethod
    def _resume_binding_missing(error: Optional[str]) -> bool:
        text = (error or "").lower()
        return bool(
            "session" in text
            and any(
                marker in text
                for marker in ("not found", "does not exist", "no conversation found")
            )
        )

    async def _sdk_stream_turn(
        self,
        prompt: Any,
        output: "queue.Queue[Any]",
    ) -> None:
        client = self._client
        if client is None:
            output.put(RuntimeError("Claude Agent SDK client is not connected"))
            return
        try:
            if isinstance(prompt, str):
                await client.query(prompt, session_id="default")
            else:

                async def _one_message():
                    yield {
                        "type": "user",
                        "message": {"role": "user", "content": prompt},
                        "parent_tool_use_id": None,
                    }

                await client.query(_one_message(), session_id="default")
            async for message in client.receive_response():
                if type(message).__name__ == "StreamEvent":
                    partial = getattr(message, "event", None)
                    if isinstance(partial, dict):
                        self.bridge.observe_stream_event(partial)
                output.put(message)
        except BaseException as exc:
            output.put(exc)
        finally:
            output.put(_SDK_TURN_END)

    async def _sdk_interrupt(self) -> None:
        if self._client is not None:
            await self._client.interrupt()

    async def _sdk_disconnect(self) -> None:
        client = self._client
        self._client = None
        self._transport = None
        self._process_started_at = None
        if client is not None:
            await client.disconnect()

    def _disconnect_client(self) -> None:
        runtime = self._runtime
        if runtime is None or not runtime.is_alive:
            self._client = None
            self._transport = None
            self._process_started_at = None
            return
        try:
            runtime.call(self._sdk_disconnect(), timeout=20.0)
        except Exception:
            logger.warning("Claude Agent SDK disconnect failed", exc_info=True)
            self._client = None
            self._transport = None
            self._process_started_at = None

    def _reset_fresh_binding(self) -> None:
        self._disconnect_client()
        if self.persistent_binding:
            _forget_binding(self.owner_key)
        self.native_session_id = str(uuid.uuid4())
        self._resume = False
        self._history_signature = None
        self._turns_completed = 0
        self._stderr.clear()

    def run_turn(
        self,
        *,
        agent: Any,
        user_input: Any,
        messages: list[dict[str, Any]],
        task_id: str,
        stream_callback: Optional[Callable[[str], None]] = None,
        projection_callback: Optional[Callable[[list[dict[str, Any]]], None]] = None,
        bootstrap_messages: Optional[list[dict[str, Any]]] = None,
        before_next_model_callback: Optional[
            Callable[[], Optional[dict[str, Any]]]
        ] = None,
        iteration_post_callback: Optional[
            Callable[[int, dict[str, Any], dict[str, int]], None]
        ] = None,
        api_retry_callback: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> ClaudeAgentSdkTurnResult:
        with self._turn_lock:
            self.bridge.bind_agent(agent)
            incoming_prefix_signature = _hermes_history_signature(messages[:-1])
            history_diverged = (
                self._history_signature is not None
                and incoming_prefix_signature != self._history_signature
            )
            unverifiable_resume = (
                self._resume and self._history_signature is None
            )
            if history_diverged or unverifiable_resume:
                logger.info(
                    "Claude Agent SDK history invalidated before turn: owner=%s reason=%s",
                    self.owner_key,
                    "Hermes transcript diverged"
                    if history_diverged
                    else "binding has no transcript signature",
                )
                self._reset_fresh_binding()
            resumed_at_start = self._resume
            result = self._run_turn_once(
                user_input=user_input,
                messages=messages,
                task_id=task_id,
                stream_callback=stream_callback,
                projection_callback=projection_callback,
                bootstrap_messages=bootstrap_messages,
                before_next_model_callback=before_next_model_callback,
                iteration_post_callback=iteration_post_callback,
                api_retry_callback=api_retry_callback,
                operation="turn",
            )
            if resumed_at_start and self._resume_binding_missing(result.error):
                logger.warning(
                    "Claude Agent SDK session %s is unavailable; retrying with a fresh binding",
                    self.native_session_id,
                )
                self._reset_fresh_binding()
                result = self._run_turn_once(
                    user_input=user_input,
                    messages=messages,
                    task_id=task_id,
                    stream_callback=stream_callback,
                    projection_callback=projection_callback,
                    bootstrap_messages=bootstrap_messages,
                    before_next_model_callback=before_next_model_callback,
                    iteration_post_callback=iteration_post_callback,
                    api_retry_callback=api_retry_callback,
                    operation="turn",
                )
                result.session_reuse = "resume_recovery"
            self._last_used_at = time.monotonic()
            if (
                result.error is None
                and not result.interrupted
                and not result.should_retire
            ):
                mirrored_messages = list(messages)
                projected = list(result.projected_messages or [])
                if projected and not (
                    len(mirrored_messages) >= len(projected)
                    and mirrored_messages[-len(projected) :] == projected
                ):
                    mirrored_messages.extend(projected)
                self.sync_history_signature(mirrored_messages)
            return result

    def compact(
        self,
        *,
        agent: Any,
        focus_topic: Optional[str] = None,
    ) -> ClaudeAgentSdkTurnResult:
        with self._turn_lock:
            self.bridge.bind_agent(agent)
            if not (self._resume or self._turns_completed):
                return ClaudeAgentSdkTurnResult(
                    native_session_id=self.native_session_id,
                    error=(
                        "Claude native history is not available yet; send one "
                        "message in this session before using /compress"
                    ),
                    session_reuse="cold_miss",
                )
            focus = str(focus_topic or "").strip()
            command = "/compact" + (f" {focus}" if focus else "")
            restore_auto_compaction_disabled = not self.auto_compaction_enabled
            if restore_auto_compaction_disabled:
                self._disconnect_client()
                self.auto_compaction_enabled = True
            try:
                result = self._run_turn_once(
                    user_input=command,
                    messages=[],
                    task_id="claude-agent-sdk-native-compaction",
                    stream_callback=None,
                    projection_callback=None,
                    bootstrap_messages=[],
                    before_next_model_callback=None,
                    iteration_post_callback=None,
                    api_retry_callback=None,
                    operation="compact",
                )
            finally:
                if restore_auto_compaction_disabled:
                    self._disconnect_client()
                    self.auto_compaction_enabled = False
            if result.error is None and not result.interrupted and not result.compacted:
                result.error = "Claude Agent SDK completed /compact without a compact_boundary event"
            self._last_used_at = time.monotonic()
            return result

    def summarize(
        self,
        *,
        agent: Any,
        messages: list[dict[str, Any]],
        prompt: str,
    ) -> ClaudeAgentSdkTurnResult:
        with self._turn_lock:
            self.bridge.bind_agent(agent)
            result = self._run_turn_once(
                user_input=prompt,
                messages=[*messages, {"role": "user", "content": prompt}],
                task_id="claude-agent-sdk-iteration-summary",
                stream_callback=None,
                projection_callback=None,
                bootstrap_messages=None,
                before_next_model_callback=None,
                iteration_post_callback=None,
                api_retry_callback=None,
                operation="summary",
            )
            self._last_used_at = time.monotonic()
            return result

    def propose_tools(
        self,
        *,
        agent: Any,
        messages: list[dict[str, Any]],
        prompt: Any,
    ) -> ClaudeAgentSdkTurnResult:
        with self._turn_lock:
            self.bridge.bind_agent(agent)
            result = self._run_turn_once(
                user_input=prompt,
                messages=[*messages, {"role": "user", "content": prompt}],
                task_id="claude-agent-sdk-auxiliary-tool-proposal",
                stream_callback=None,
                projection_callback=None,
                bootstrap_messages=None,
                before_next_model_callback=None,
                iteration_post_callback=None,
                api_retry_callback=None,
                operation="tool_proposal",
            )
            self._last_used_at = time.monotonic()
            return result

    def _run_turn_once(
        self,
        *,
        user_input: Any,
        messages: list[dict[str, Any]],
        task_id: str,
        stream_callback: Optional[Callable[[str], None]],
        projection_callback: Optional[Callable[[list[dict[str, Any]]], None]],
        bootstrap_messages: Optional[list[dict[str, Any]]],
        before_next_model_callback: Optional[Callable[[], Optional[dict[str, Any]]]],
        iteration_post_callback: Optional[
            Callable[[int, dict[str, Any], dict[str, int]], None]
        ],
        api_retry_callback: Optional[Callable[[dict[str, Any]], None]],
        operation: str = "turn",
    ) -> ClaudeAgentSdkTurnResult:
        is_compaction = operation == "compact"
        is_summary = operation == "summary"
        is_tool_proposal = operation == "tool_proposal"
        bound_agent = getattr(self.bridge, "_agent", None)
        iteration_budget = getattr(bound_agent, "iteration_budget", None)
        reuse = (
            "warm_hit"
            if self._turns_completed and self.is_alive
            else "native_resume"
            if self._resume
            else "cold_miss"
        )
        reserved_model_iterations = 0
        if operation == "turn" and iteration_budget is not None:
            if not iteration_budget.consume():
                return ClaudeAgentSdkTurnResult(
                    native_session_id=self.native_session_id,
                    session_reuse=reuse,
                    latency_ms={"process_start": 0, "total": 0},
                    budget_exhausted=True,
                    should_retire=True,
                    error=(
                        "Claude Agent SDK iteration budget exhausted before "
                        "the first native model step"
                    ),
                )
            reserved_model_iterations = 1

        self._interrupt_requested.clear()
        launch_started = time.monotonic()
        was_alive = self.is_alive
        try:
            self.ensure_started()
        except Exception as exc:
            if reserved_model_iterations and iteration_budget is not None:
                iteration_budget.refund()
            return ClaudeAgentSdkTurnResult(
                native_session_id=self.native_session_id,
                session_reuse=reuse,
                error=str(exc),
                should_retire=True,
                latency_ms={
                    "process_start": int((time.monotonic() - launch_started) * 1000),
                    "total": int((time.monotonic() - launch_started) * 1000),
                },
            )
        turn_started = time.monotonic()
        result = ClaudeAgentSdkTurnResult(
            native_session_id=self.native_session_id,
            session_reuse=reuse,
            latency_ms={
                "process_start": (
                    int((turn_started - launch_started) * 1000) if not was_alive else 0
                )
            },
            budget_iterations=reserved_model_iterations,
        )
        if self._interrupt_requested.is_set():
            if reserved_model_iterations and iteration_budget is not None:
                iteration_budget.refund()
                result.budget_iterations = 0
            result.interrupted = True
            result.should_retire = True
            result.latency_ms["total"] = int((time.monotonic() - turn_started) * 1000)
            return result

        user_text = _coerce_text(user_input)
        bootstrap = (
            ""
            if self._resume or self._turns_completed
            else serialize_history_for_bootstrap(
                bootstrap_messages if bootstrap_messages is not None else messages,
                prefill_messages=None,
            )
        )
        prefill_messages = (
            getattr(bound_agent, "prefill_messages", None)
            if bootstrap_messages is None
            else None
        )
        bootstrap_attachments = (
            []
            if not bootstrap
            else _bootstrap_image_blocks(
                bootstrap_messages if bootstrap_messages is not None else messages,
                prefill_messages=prefill_messages,
            )
        )
        prompt = _compose_user_content(
            user_input,
            bootstrap=bootstrap,
            bootstrap_attachments=bootstrap_attachments,
        )

        def _project_from_bridge(rows: list[dict[str, Any]]) -> None:
            result.projected_messages.extend(rows)
            if projection_callback is not None:
                projection_callback(rows)

        callback_lock = threading.RLock()

        def _before_next_model() -> Optional[dict[str, Any]]:
            nonlocal reserved_model_iterations
            with callback_lock:
                if getattr(bound_agent, "_incremental_persistence_failed", False):
                    result.host_stop_reason = "session_persistence_failed"
                    result.error = (
                        "Claude Agent SDK turn stopped because session storage "
                        "could not persist the tool protocol. Free disk space and retry."
                    )
                    result.should_retire = True
                    self.interrupt()
                    return None
                if (
                    getattr(bound_agent, "_tool_guardrail_halt_decision", None)
                    is not None
                ):
                    result.host_stop_reason = "guardrail_halt"
                    result.should_retire = True
                    self.interrupt()
                    return None
                if operation == "turn" and iteration_budget is not None:
                    if not iteration_budget.consume():
                        result.budget_exhausted = True
                        result.error = (
                            "Claude Agent SDK iteration budget exhausted before "
                            "the next native model step"
                        )
                        result.should_retire = True
                        self.interrupt()
                        return None
                    reserved_model_iterations += 1
                    result.budget_iterations += 1
                if callable(before_next_model_callback):
                    return before_next_model_callback()
                return None

        self.bridge.begin_turn(
            task_id=task_id,
            user_task=user_text,
            messages=messages,
            projection_callback=_project_from_bridge,
            execute_tools=not is_tool_proposal,
            before_next_model_callback=_before_next_model,
        )

        first_record_at: Optional[float] = None
        first_text_at: Optional[float] = None
        current_stream_text = ""
        final_candidate = ""
        seen_message_ids: set[tuple[str, str]] = set()
        projected_tool_ids: set[str] = set()
        events: "queue.Queue[Any]" = queue.Queue()
        runtime = self._runtime
        if runtime is None:
            self.bridge.end_turn()
            result.error = "Claude Agent SDK async runtime disappeared"
            result.should_retire = True
            return result
        turn_future = runtime.submit(self._sdk_stream_turn(prompt, events))
        deadline = time.monotonic() + self.turn_timeout
        last_activity_heartbeat = turn_started
        terminal_seen = False
        last_assistant_error: Optional[str] = None
        native_model_steps_started = 0
        try:
            while time.monotonic() < deadline:
                try:
                    incoming = events.get(timeout=0.25)
                except queue.Empty:
                    now = time.monotonic()
                    if now - last_activity_heartbeat >= 30.0:
                        touch_activity = getattr(bound_agent, "_touch_activity", None)
                        if callable(touch_activity):
                            touch_activity("waiting for Claude Agent SDK response")
                        last_activity_heartbeat = now
                    if turn_future.done():
                        try:
                            turn_future.result()
                        except Exception as exc:
                            result.error = self._error_with_stderr(exc)
                            result.should_retire = True
                        break
                    continue
                if incoming is _SDK_TURN_END:
                    break
                if isinstance(incoming, BaseException):
                    if (
                        self._interrupt_requested.is_set()
                        and not result.host_stop_reason
                    ):
                        result.interrupted = True
                    else:
                        result.error = self._error_with_stderr(incoming)
                    result.should_retire = True
                    break

                now = time.monotonic()
                last_activity_heartbeat = now
                if first_record_at is None:
                    first_record_at = now
                    result.latency_ms["first_record"] = int((now - turn_started) * 1000)
                event = _sdk_message_event(incoming)
                event_type = str(event.get("type") or "")
                event_session_id = event.get("session_id") or event.get("sessionId")
                if event_session_id:
                    self.native_session_id = str(event_session_id)
                    result.native_session_id = self.native_session_id
                    if self.persistent_binding:
                        _save_binding(self.owner_key, self.native_session_id)

                inner = event.get("event")
                if event_type == "stream_event" and isinstance(inner, dict):
                    if inner.get("type") == "message_start" and not is_compaction:
                        native_model_steps_started += 1
                        current_stream_text = ""
                        touch_activity = getattr(bound_agent, "_touch_activity", None)
                        if callable(touch_activity):
                            touch_activity(
                                "starting Claude native model iteration "
                                f"#{native_model_steps_started}"
                            )
                        reset_stream_tracking = getattr(
                            bound_agent, "_reset_stream_delivery_tracking", None
                        )
                        if callable(reset_stream_tracking):
                            reset_stream_tracking()
                        checkpoint_mgr = getattr(bound_agent, "_checkpoint_mgr", None)
                        if checkpoint_mgr is not None:
                            try:
                                checkpoint_mgr.new_turn()
                            except Exception:
                                logger.debug(
                                    "Claude SDK checkpoint iteration reset failed",
                                    exc_info=True,
                                )
                        if (
                            operation == "turn"
                            and iteration_budget is not None
                            and native_model_steps_started > reserved_model_iterations
                        ):
                            if iteration_budget.consume():
                                reserved_model_iterations += 1
                                result.budget_iterations += 1
                                if callable(before_next_model_callback):
                                    before_next_model_callback()
                            else:
                                result.budget_exhausted = True
                                result.error = (
                                    "Claude Agent SDK iteration budget exhausted "
                                    "before an internal native model step"
                                )
                                result.should_retire = True
                                self.interrupt()
                                break
                        step_callback = getattr(bound_agent, "step_callback", None)
                        if callable(step_callback) and not is_summary:
                            try:
                                step_callback(
                                    native_model_steps_started,
                                    _previous_tool_batch(result.projected_messages),
                                )
                            except Exception:
                                logger.debug(
                                    "Claude SDK step callback failed", exc_info=True
                                )

                delta = "" if is_compaction else self._stream_delta(event)
                if delta:
                    current_stream_text += delta
                    if first_text_at is None:
                        first_text_at = now
                        result.latency_ms["first_text"] = int(
                            (now - turn_started) * 1000
                        )
                    if stream_callback is not None:
                        try:
                            stream_callback(delta)
                        except Exception:
                            logger.debug(
                                "Claude SDK stream callback failed", exc_info=True
                            )

                reasoning_delta = (
                    "" if is_compaction or is_summary else self._reasoning_delta(event)
                )
                if reasoning_delta:
                    callback = getattr(bound_agent, "_fire_reasoning_delta", None)
                    if callable(callback):
                        callback(reasoning_delta)
                tool_gen_name = None if is_compaction else self._tool_gen_started(event)
                if tool_gen_name:
                    callback = getattr(bound_agent, "_fire_tool_gen_started", None)
                    if callable(callback):
                        callback(tool_gen_name)

                if event_type == "system" and event.get("subtype") == "api_retry":
                    retry_event = {
                        "attempt": int(event.get("attempt") or 0),
                        "max_retries": int(event.get("max_retries") or 0),
                        "retry_delay_ms": int(event.get("retry_delay_ms") or 0),
                        "error_status": event.get("error_status"),
                        "error": str(event.get("error") or "unknown"),
                    }
                    result.last_api_retry = retry_event
                    if callable(api_retry_callback):
                        try:
                            api_retry_callback(dict(retry_event))
                        except Exception:
                            logger.debug(
                                "Claude SDK API retry callback failed",
                                exc_info=True,
                            )
                    continue
                if (
                    event_type == "system"
                    and event.get("subtype") == "compact_boundary"
                ):
                    result.compacted = True
                    result.compaction_count += 1
                    metadata = event.get("compact_metadata")
                    if isinstance(metadata, dict):
                        result.compaction_metadata = dict(metadata)
                    continue

                message = event.get("message")
                message_id = (
                    str(message.get("id") or "") if isinstance(message, dict) else ""
                )
                if event_type == "assistant" and isinstance(message, dict):
                    assistant_error = str(message.get("error") or "").strip()
                    if assistant_error:
                        last_assistant_error = assistant_error
                    stop_reason = str(message.get("stop_reason") or "").strip()
                    if stop_reason:
                        result.last_stop_reason = stop_reason
                    if stop_reason in {"max_tokens", "model_context_window_exceeded"}:
                        blocks = _content_blocks(message)
                        has_reasoning = any(
                            block.get("type") == "thinking"
                            and bool(
                                str(
                                    block.get("thinking") or block.get("text") or ""
                                ).strip()
                            )
                            for block in blocks
                        )
                        has_visible_text = any(
                            block.get("type") == "text"
                            and bool(str(block.get("text") or "").strip())
                            for block in blocks
                        )
                        has_tool_use = any(
                            block.get("type") == "tool_use" for block in blocks
                        )
                        if has_reasoning and not has_visible_text and not has_tool_use:
                            result.thinking_budget_exhausted = True

                record_key = (event_type, message_id)
                if not is_compaction and (
                    not message_id or record_key not in seen_message_ids
                ):
                    raw_projected, iterations = self._project_record(
                        event, allowed_tool_ids=projected_tool_ids
                    )
                    if message_id:
                        seen_message_ids.add(record_key)
                    for raw_message in raw_projected:
                        for tool_call in raw_message.get("tool_calls") or []:
                            raw_id = str(tool_call.get("id") or "")
                            if raw_id:
                                projected_tool_ids.add(raw_id)
                    projected = self.bridge.reconcile_authoritative_projection(
                        raw_projected
                    )
                    if projected:
                        result.projected_messages.extend(projected)
                        projection_succeeded = True
                        if projection_callback is not None:
                            try:
                                projection_callback(projected)
                            except Exception:
                                projection_succeeded = False
                                logger.warning(
                                    "Claude SDK incremental projection callback failed",
                                    exc_info=True,
                                )
                        if getattr(
                            bound_agent, "_incremental_persistence_failed", False
                        ):
                            projection_succeeded = False
                        self.bridge.mark_authoritative_projection_persisted(
                            projected, succeeded=projection_succeeded
                        )
                    if (
                        event_type == "assistant"
                        and raw_projected
                        and not raw_projected[0].get("tool_calls")
                        and raw_projected[0].get("content")
                    ):
                        final_candidate = str(raw_projected[0]["content"])
                    has_internal_only_tool_use = bool(
                        event_type == "assistant"
                        and not raw_projected
                        and any(
                            block.get("type") in {"tool_use", "server_tool_use"}
                            for block in _content_blocks(message)
                        )
                    )
                    if event_type == "assistant" and not has_internal_only_tool_use:
                        result.model_iterations += 1
                        observer_assistant = (
                            dict(raw_projected[0])
                            if raw_projected
                            else {"role": "assistant", "content": None}
                        )
                        if not raw_projected:
                            observer_blocks = _content_blocks(message)
                            reasoning = "".join(
                                str(block.get("thinking") or block.get("text") or "")
                                for block in observer_blocks
                                if block.get("type") == "thinking"
                            )
                            if reasoning:
                                observer_assistant["reasoning"] = reasoning
                            observer_stop_reason = str(
                                message.get("stop_reason") or ""
                            ).strip()
                            if observer_stop_reason:
                                observer_assistant["finish_reason"] = (
                                    "length"
                                    if observer_stop_reason
                                    in {
                                        "max_tokens",
                                        "model_context_window_exceeded",
                                    }
                                    else observer_stop_reason
                                )
                        if observer_assistant.get("tool_calls"):
                            interim = getattr(
                                bound_agent, "_emit_interim_assistant_message", None
                            )
                            if callable(interim):
                                interim(observer_assistant)
                        if callable(iteration_post_callback):
                            try:
                                iteration_post_callback(
                                    result.model_iterations,
                                    observer_assistant,
                                    self._usage(event),
                                )
                            except Exception:
                                logger.debug(
                                    "Claude SDK iteration post callback failed",
                                    exc_info=True,
                                )
                        if is_tool_proposal and observer_assistant.get("tool_calls"):
                            result.captured_tool_calls = True
                            result.final_text = str(
                                observer_assistant.get("content") or ""
                            )
                            result.should_retire = True
                            self.interrupt()
                            break
                    result.tool_iterations += iterations

                usage = self._usage(event)
                if usage:
                    result.token_usage = usage
                last_call_usage = self._last_iteration_usage(event)
                if last_call_usage:
                    result.last_call_usage = last_call_usage

                if event_type == "result":
                    terminal_seen = True
                    result.terminal_result_received = True
                    terminal_stop_reason = str(
                        event.get("stop_reason") or event.get("stopReason") or ""
                    ).strip()
                    if terminal_stop_reason:
                        result.last_stop_reason = terminal_stop_reason
                    terminal_reason = str(event.get("terminal_reason") or "")
                    if (
                        terminal_reason.startswith("aborted")
                        and not result.host_stop_reason
                    ):
                        result.interrupted = True
                    if event.get("is_error"):
                        errors = event.get("errors")
                        detail = (
                            "; ".join(str(value) for value in errors)
                            if isinstance(errors, list) and errors
                            else str(
                                event.get("result") or "Claude Agent SDK turn failed"
                            )
                        )
                        result.error = detail
                        result.error_category = (
                            str(result.last_api_retry.get("error") or "").strip()
                            or last_assistant_error
                            or (
                                str(event.get("subtype") or "").strip()
                                if str(event.get("subtype") or "").strip() != "success"
                                else ""
                            )
                            or None
                        )
                        raw_status = event.get("error_status")
                        if raw_status is None:
                            raw_status = result.last_api_retry.get("error_status")
                        try:
                            result.error_status = (
                                int(raw_status) if raw_status is not None else None
                            )
                        except (TypeError, ValueError):
                            result.error_status = None
                        result.should_retire = True
                    structured_output = event.get("structured_output")
                    if structured_output is not None:
                        result.structured_output = structured_output
                        result.final_text = (
                            structured_output
                            if isinstance(structured_output, str)
                            else json.dumps(
                                structured_output,
                                ensure_ascii=False,
                                separators=(",", ":"),
                            )
                        )
                    else:
                        terminal = event.get("result")
                        result.final_text = (
                            str(terminal)
                            if isinstance(terminal, str) and terminal
                            else final_candidate or current_stream_text
                        )
                    break
            else:
                result.error = f"Claude Agent SDK exceeded the {self.turn_timeout:.0f}s turn timeout"
                result.should_retire = True
                self.interrupt()

            if not terminal_seen and result.error is None and not result.interrupted:
                if turn_future.done():
                    try:
                        turn_future.result()
                    except Exception as exc:
                        result.error = self._error_with_stderr(exc)
                if result.error is None and not is_tool_proposal:
                    result.error = "Claude Agent SDK ended without a terminal result"
                result.should_retire = True
            if not result.final_text:
                result.final_text = final_candidate or current_stream_text
            if not result.final_text:
                for projected in reversed(result.projected_messages):
                    if projected.get("role") == "assistant" and projected.get(
                        "content"
                    ):
                        result.final_text = str(projected["content"])
                        break
            if (
                result.error is None
                and not result.interrupted
                and not result.should_retire
                and self.native_session_id
            ):
                if self.persistent_binding:
                    _save_binding(self.owner_key, self.native_session_id)
                self._resume = True
                self._turns_completed += 1
            result.latency_ms["total"] = int((time.monotonic() - turn_started) * 1000)
            if self._process_started_at is not None:
                result.latency_ms["process_age"] = int(
                    (time.monotonic() - self._process_started_at) * 1000
                )
            logger.info(
                "Claude Agent SDK turn complete (owner=%s model=%s reuse=%s "
                "latency_ms=%s interrupted=%s error=%s)",
                self.owner_key,
                self.model,
                reuse,
                result.latency_ms,
                result.interrupted,
                bool(result.error),
            )
            return result
        finally:
            self.bridge.end_turn()

    def interrupt(self) -> None:
        self._interrupt_requested.set()
        runtime = self._runtime
        if runtime is not None and runtime.is_alive and self._client is not None:
            interrupt = self._sdk_interrupt()
            try:
                runtime.submit(interrupt)
            except Exception:
                interrupt.close()
                logger.debug(
                    "Claude Agent SDK interrupt scheduling failed", exc_info=True
                )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._disconnect_client()
        if self._runtime is not None:
            self._runtime.stop()
            self._runtime = None
        self.bridge.close()
        if self._runtime_dir is not None:
            try:
                import shutil as _shutil

                _shutil.rmtree(self._runtime_dir)
            except OSError:
                logger.debug(
                    "Claude SDK runtime directory cleanup failed", exc_info=True
                )
            self._runtime_dir = None


async def _empty_prompt_stream():
    if False:  # pragma: no cover - establishes the async-generator protocol
        yield {}


__all__ = [
    "ClaudeAgentSdkSession",
    "ClaudeAgentSdkTurnResult",
    "_bootstrap_image_blocks",
    "_compose_user_content",
    "_hermes_history_signature",
    "forget_claude_agent_sdk_binding",
    "serialize_history_for_bootstrap",
]
