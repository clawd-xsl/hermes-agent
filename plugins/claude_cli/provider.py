"""The ``claude_cli`` turn runtime: Hermes turns on one long-lived Claude Code process per session.

Registered through ``ctx.register_turn_runtime`` (``agent/turn_runtime_provider.py``). Selected by
``model.anthropic_runtime: claude_cli`` with ``model.provider: anthropic``, or by an explicit
``api_mode: claude_cli`` on a fallback entry. The runtime itself lives in ``runtime.py`` (turn
driver), ``session.py`` (the stream-json child), ``loopback.py`` / ``mcp_proxy.py`` (Hermes tools
over a private MCP server) and ``journal.py`` (tool-effect write-ahead log).
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Optional

from agent.turn_runtime_provider import TurnRuntimeProvider

logger = logging.getLogger(__name__)

RUNTIME_NAME = "claude_cli"
_PROVIDER = "anthropic"

_FALLBACK_CONTINUATION_PROMPT = (
    "[System: The previous inference runtime stopped before finishing. "
    "Continue the same task from the transcript and produce the complete answer.]"
)


def claude_cli_process_config(model_cfg: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    """``model.claude_cli.command`` / ``args`` as the runtime keys AIAgent construction reads."""
    raw = (model_cfg or {}).get("claude_cli")
    cfg = raw if isinstance(raw, Mapping) else {}
    # Default to the plain executable: core treats a route with a ``command`` as an external-process
    # route that needs no API key (``hermes doctor``, the TUI model gate, delegation).
    command = str(cfg.get("command") or "").strip() or "claude"
    raw_args = cfg.get("args")
    args = [str(arg) for arg in raw_args] if isinstance(raw_args, (list, tuple)) else []
    return {"command": command, "args": args}


def wants_persistent_binding(agent: Any) -> bool:
    """Whether this agent's native session should be resumable across processes.

    Long-lived conversations keep a durable binding; one-shot surfaces (cron fires, delegated
    children, background-review forks, side agents such as ``/background``, ``chat -q``) never reuse their generated session ids, so a
    binding would only fill the native-session pool and runtime directory. An explicit
    ``agent._claude_cli_persistent_binding`` bool always wins."""
    explicit = getattr(agent, "_claude_cli_persistent_binding", None)
    if isinstance(explicit, bool):
        return explicit
    if getattr(agent, "_persist_disabled", False) or getattr(agent, "side_agent", False) is True:
        return False
    if int(getattr(agent, "_delegate_depth", 0) or 0) > 0:
        return False
    if str(getattr(agent, "platform", "") or "").strip().lower() in {"cron", "subagent"}:
        return False
    try:
        from agent.oneshot_footprint import agent_is_oneshot
        if agent_is_oneshot(agent):
            return False
    except Exception:
        pass
    return True


def _effective_system_prompt(agent: Any, active_system_prompt: Any) -> str:
    prompt = str(active_system_prompt or "")
    ephemeral = getattr(agent, "ephemeral_system_prompt", None)
    if ephemeral:
        prompt = (prompt + "\n\n" + str(ephemeral)).strip()
    return prompt


def _ensure_continuation_row(messages: list) -> None:
    """A generic-loop turn handed to this runtime mid-way must end on a user row: pair a partial
    assistant answer with a private continuation (both dropped from durable history)."""
    tail = messages[-1] if messages else None
    if isinstance(tail, dict) and tail.get("role") == "user":
        return
    if isinstance(tail, dict) and tail.get("role") == "assistant" and not tail.get("tool_calls"):
        tail["_provider_fallback_synthetic"] = True
        tail.pop("_db_persisted", None)
    messages.append({"role": "user", "content": _FALLBACK_CONTINUATION_PROMPT, "_provider_fallback_synthetic": True})


def _sync_native_history_signature(agent: Any, messages: list) -> None:
    """Claude's native transcript is authoritative for continuation. A display-only interruption row
    or cleaned recovery scaffolding in Hermes' mirror is no reason to drop the binding and replay; the
    signature only describes the mirror for diagnostics."""
    session = getattr(agent, "_claude_cli_session", None)
    sync = getattr(session, "sync_history_signature", None)
    if callable(sync):
        try:
            sync(messages)
        except Exception:
            logger.debug("Claude native transcript signature sync failed", exc_info=True)


class ClaudeCliTurnRuntime(TurnRuntimeProvider):
    """Persistent Claude Code stream-json runtime with Hermes tools over an MCP loopback."""

    @property
    def name(self) -> str:
        return RUNTIME_NAME

    @property
    def display_name(self) -> str:
        return "Claude Code"

    # -- selection ---------------------------------------------------------------------------------

    def resolve_runtime(self, *, provider: str, api_mode: str, model_cfg: Optional[Mapping[str, Any]]):
        if (provider or "").strip().lower() != _PROVIDER or not model_cfg:
            return None
        if str(model_cfg.get("anthropic_runtime") or "").strip().lower() != RUNTIME_NAME:
            return None
        return {
            "provider": _PROVIDER, "api_mode": RUNTIME_NAME, "base_url": "", "api_key": "",
            "source": "claude-code", "credential_pool": None, **claude_cli_process_config(model_cfg),
        }

    # -- construction --------------------------------------------------------------------------------

    def init_client(self, agent: Any) -> bool:
        # Claude Code owns inference, authentication, native history and compaction: no Anthropic/OpenAI
        # HTTP client, and no subscription credential copied into Hermes.
        agent.client = None
        agent.api_key = ""
        agent.base_url = ""
        agent._client_kwargs = {}
        agent._anthropic_client = None
        agent._is_anthropic_oauth = False
        # ``acp_command`` is shared by several subprocess transports; keep a Claude-specific copy so a
        # later fallback cannot mistake a stale Copilot command for the Claude executable.
        agent._claude_cli_command = getattr(agent, "acp_command", None) or None
        agent._claude_cli_args = list(getattr(agent, "acp_args", None) or [])
        if not getattr(agent, "quiet_mode", True):
            print(f"🤖 AI Agent initialized with model: {agent.model} (persistent Claude Code)")
        return True

    # -- the turn --------------------------------------------------------------------------------------

    def run_turn(self, agent: Any, turn: Any) -> Optional[dict[str, Any]]:
        from plugins.claude_cli.runtime import run_claude_cli_turn

        kwargs: dict[str, Any] = dict(
            user_message=turn.user_message, original_user_message=turn.original_user_message,
            messages=turn.messages, conversation_history=turn.conversation_history,
            effective_task_id=turn.effective_task_id, turn_id=turn.turn_id,
            current_turn_user_idx=turn.current_turn_user_idx,
            active_system_prompt=_effective_system_prompt(agent, turn.active_system_prompt),
            should_review_memory=turn._should_review_memory, moa_config=turn.moa_config,
        )
        if getattr(turn, "continuation", False):
            # The generic loop prepared this turn and failed over to Claude: do not rerun turn setup or
            # MoA fan-out, and keep its API calls / held verification answer in the turn's accounting.
            _ensure_continuation_row(turn.messages)
            kwargs.update(
                current_turn_user_idx=len(turn.messages) - 1, _moa_applied=True,
                _api_calls_before=int(turn.api_call_count or 0),
                _pending_verification_response=turn._pending_verification_response,
                _pending_verification_response_previewed=bool(turn._pending_verification_response_previewed),
            )
        result = run_claude_cli_turn(agent, **kwargs)
        if not result.get("_claude_cli_continue_standard_loop"):
            _sync_native_history_signature(agent, turn.messages)
            return result
        # A terminal native failure activated the next configured provider: the generic loop continues
        # the already-prepared turn (no second user row, no repeated pre_llm_call / memory prefetch).
        turn.api_call_count = max(0, int(result.get("api_calls") or 0))
        turn._pending_verification_response = result.get("pending_verification_response")
        turn._pending_verification_response_previewed = bool(result.get("pending_verification_response_previewed"))
        if turn.messages and turn.messages[-1].get("_provider_fallback_synthetic"):
            turn.current_turn_user_idx = len(turn.messages) - 1
        return None

    # -- in-flight control ---------------------------------------------------------------------------

    def request_interrupt(self, agent: Any) -> None:
        # A persistent subprocess, not an interruptible HTTP request: stop the child's turn now; the
        # runtime retires it and resumes native history on the next turn.
        session = getattr(agent, "_claude_cli_session", None)
        if session is not None:
            session.interrupt()

    def release(self, agent: Any, *, hard: bool) -> None:
        from plugins.claude_cli.runtime import detach_claude_cli_session, release_claude_cli_session
        if not hard:
            # Cache eviction keeps the child pooled so the rebuilt agent avoids a cold start.
            return
        if getattr(agent, "_borrows_session_runtime", False):
            detach_claude_cli_session(agent)
            return
        release_claude_cli_session(agent)

    # -- compaction ------------------------------------------------------------------------------------

    def owns_compaction(self, agent: Any) -> bool:
        from plugins.claude_cli.compaction import claude_cli_owns_context_compaction
        return claude_cli_owns_context_compaction(agent)

    def native_auto_compaction(self, agent: Any) -> bool:
        return self.owns_compaction(agent)

    def compact(self, agent: Any, messages: list, system_message: Any, **kwargs: Any) -> Any:
        from plugins.claude_cli.compaction import compact_via_claude_cli
        return compact_via_claude_cli(agent, messages, system_message, **kwargs)

    # -- auxiliary calls --------------------------------------------------------------------------------

    def auxiliary_client(self, *, model: str, async_mode: bool, is_vision: bool = False, task: Any = None):
        from plugins.claude_cli.auxiliary import ClaudeCliAuxiliaryClient, AsyncClaudeCliAuxiliaryClient
        if not model:
            logger.warning("claude_cli auxiliary client requested without a model")
            return None
        client = ClaudeCliAuxiliaryClient(model)
        return (AsyncClaudeCliAuxiliaryClient(client), model) if async_mode else (client, model)
