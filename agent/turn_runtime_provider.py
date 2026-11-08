"""Turn Runtime Provider ABC.

A turn runtime is an agent runtime that owns a whole turn's model/tool loop in an external
process (a long-lived subprocess, an app server) instead of Hermes issuing one provider request
per model step. The codex app-server runtime (``agent/codex_runtime.py``) is the built-in one;
plugins add others via :meth:`PluginContext.register_turn_runtime`.

A runtime is keyed by the ``api_mode`` it owns: an agent whose ``api_mode`` names a registered
runtime hands every turn to :meth:`TurnRuntimeProvider.run_turn` instead of entering the generic
loop, and every place that would otherwise act on Hermes' local mirror of the conversation
(interrupt, steer, compaction, teardown) asks the runtime first. Core keeps everything around the
turn: turn context (system prompt, memory prefetch, plugin context), session persistence,
finalization hooks, delivery.

Runtime objects receive live ``AIAgent`` instances, so they always run in-process; they are not
supported under ``plugins.isolation: host``.
"""

from __future__ import annotations

import abc
from typing import Any, Iterable, Mapping, Optional

from agent.provider_base import ProviderBase


class TurnRuntimeProvider(ProviderBase):
    """Abstract base class for a runtime that owns a turn's model/tool loop.

    :attr:`name` is the ``api_mode`` this runtime owns (``codex_app_server``, ``claude_cli``).
    """

    # -- selection ---------------------------------------------------------------------------

    def resolve_runtime(
        self, *, provider: str, api_mode: str, model_cfg: Optional[Mapping[str, Any]],
    ) -> Optional[dict[str, Any]]:
        """Runtime overrides when config opts the resolved ``provider`` into this runtime, else None.

        Called by ``hermes_cli.runtime_provider`` after the provider's own route is known. The
        returned mapping is merged over the resolved runtime and must set ``api_mode`` to
        :attr:`name`; it may clear credentials the runtime does not use (``api_key``,
        ``base_url``) and add the keys AIAgent construction reads (``command``, ``args``)."""
        return None

    # -- agent construction ------------------------------------------------------------------

    def init_client(self, agent: Any) -> bool:
        """Set up ``agent`` for this runtime at construction; True when the runtime owns inference
        and Hermes must not build an HTTP client for the main model (``agent.client`` stays None).
        The default False keeps the normal client (the runtime may still use it for side calls)."""
        return False

    # -- the turn ------------------------------------------------------------------------------

    @abc.abstractmethod
    def run_turn(self, agent: Any, turn: Any) -> Optional[dict[str, Any]]:
        """Run one prepared user turn; return the turn result, or None to continue the same turn on
        the generic loop (a failure activated a fallback provider and rewrote ``agent.api_mode``).

        ``turn`` is the conversation loop's state (``agent.conversation_loop._LoopState``): the
        prepared ``messages`` (user row already appended at ``current_turn_user_idx``),
        ``active_system_prompt``, ``user_message`` / ``original_user_message``,
        ``conversation_history``, ``effective_task_id``, ``turn_id``, ``moa_config``,
        ``_should_review_memory`` and the running ``api_call_count``. A runtime returning None
        updates ``turn.api_call_count`` (and any projected rows in ``turn.messages``) first.
        ``turn.continuation`` is True when the generic loop activated this runtime mid-turn as a
        fallback: the user row is already in ``messages`` and the turn context already ran."""

    # -- in-flight control -------------------------------------------------------------------

    def request_interrupt(self, agent: Any) -> None:
        """Stop the in-flight turn as soon as possible (called from any thread)."""

    def request_steer(self, agent: Any, text: str) -> Optional[bool]:
        """Deliver mid-turn steer ``text`` natively: True accepted, False refused, None (the default)
        when the runtime has no native steer and Hermes should buffer it as usual (the runtime then
        drains ``agent``'s pending steer itself, e.g. into its next tool result)."""
        return None

    def release(self, agent: Any, *, hard: bool) -> None:
        """Release the runtime's resources for ``agent``. ``hard`` is ``AIAgent.close()`` (a real
        conversation boundary: /new, /reset, expiry); soft is cache eviction
        (``release_clients()``), after which a rebuilt agent may resume the same session. A helper
        agent that only borrowed the live session's runtime (``agent._borrows_session_runtime``,
        e.g. the gateway's manual-compression agent) must leave that runtime running on close."""

    def release_sessions(self, session_ids: Iterable[str]) -> int:
        """Release what the runtime holds for Hermes sessions that ended with no live ``AIAgent`` to
        :meth:`release` (a finished webhook delivery or cron fire: the gateway and the scheduler hold
        only session ids by then). Several ids arrive when a compression mid-run rotated the work onto
        a continuation. Returns how many were released; the default holds nothing per session."""
        return 0

    # -- compaction ----------------------------------------------------------------------------

    def owns_compaction(self, agent: Any) -> bool:
        """True while the runtime's native thread is the authoritative context, so Hermes must not
        rewrite its local mirror to make room (it would not shrink the context the model sees)."""
        return True

    def native_auto_compaction(self, agent: Any) -> bool:
        """True when the runtime compacts automatically on context pressure, so Hermes skips its own
        preflight/threshold compression (only explicit requests reach :meth:`compact`)."""
        return True

    def compact(
        self, agent: Any, messages: list, system_message: Optional[str], *, force: bool,
        focus_topic: Optional[str] = None, verbatim_tail: Optional[list] = None,
        approx_tokens: Optional[int] = None, task_id: str = "default", trigger: Optional[str] = None,
        commit_fence: Any = None, attempt: Any = None, defer_context_engine_notification: bool = False,
    ) -> Optional[tuple[list, str]]:
        """Handle a ``compress_context`` call while :meth:`owns_compaction` is True; return
        ``(messages, system_prompt)`` like ``compress_context``, or None to skip (Hermes then leaves
        history unchanged). ``force`` is a manual ``/compress``; ``trigger`` says why an automatic
        attempt runs (``"idle"``, ``"overflow"``, ...); ``verbatim_tail`` holds the exchanges
        ``/compress here N`` keeps verbatim after ``messages``. ``attempt`` is the compression
        attempt record core opened (telemetry, ``restore_compressor``); ``commit_fence`` the
        cooperative commit fence, if any; ``defer_context_engine_notification`` asks to queue the
        context-engine hook until a manual host commits its outer history transaction."""
        return None

    # -- auxiliary calls --------------------------------------------------------------------------

    def auxiliary_client(
        self, *, model: str, async_mode: bool, is_vision: bool = False, task: Optional[str] = None,
    ) -> Optional[tuple[Any, str]]:
        """``(client, model)`` for an auxiliary call (titles, summaries, vision, judges) routed to this
        runtime — the main provider's auxiliary route while the main agent runs on it, or an explicit
        ``api_mode`` naming it. The client exposes ``.chat.completions.create()`` (awaitable when
        ``async_mode``) and must never touch a live main-session thread. None keeps the normal
        provider resolution (e.g. the provider's HTTP credentials)."""
        return None

    # -- input shaping ---------------------------------------------------------------------------

    def image_input_mode(self, agent: Any) -> Optional[str]:
        """``"text"`` to force attached images to path references, ``"native"`` to allow content
        parts, None to keep the normal routing decision."""
        return None
