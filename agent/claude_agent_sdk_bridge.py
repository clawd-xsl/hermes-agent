"""In-process Hermes tool bridge for the persistent Claude Agent SDK client.

Agent SDK hosts the MCP server in-process and invokes this bridge directly.
Calls still run through the live :class:`AIAgent` executor so middleware,
approvals, hooks, memory, delegation, batching, guardrails, and session
scoping remain identical to the standard provider loop.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import logging
import secrets
import threading
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, Optional

from agent.claude_agent_sdk_journal import (
    ClaudeToolJournal,
    ToolIntent,
)

logger = logging.getLogger(__name__)

_HOUSEKEEPING_TOOLS = frozenset({
    "memory",
    "todo",
    "skill_manage",
    "session_search",
})

def _strip_mcp_prefix(name: str) -> str:
    prefix = "mcp__hermes__"
    return name[len(prefix) :] if name.startswith(prefix) else name


@dataclass
class _TurnBinding:
    task_id: str
    user_task: str
    messages: list[dict[str, Any]]
    context: contextvars.Context
    projection_callback: Optional[Callable[[list[dict[str, Any]]], None]] = None
    execute_tools: bool = True
    tool_executor: Optional[Callable[..., Any]] = None
    before_next_model_callback: Optional[
        Callable[[], Optional[dict[str, Any]]]
    ] = None


@dataclass
class _ToolProjection:
    signature: str
    local_id: str
    name: str = ""
    arguments: Optional[dict[str, Any]] = None
    claude_id: Optional[str] = None
    batch_id: Optional[str] = None
    authoritative_seen: bool = False
    claimed: bool = False
    intent_persisted: bool = False
    intent_persistence_failed: bool = False
    result_ready: bool = False
    result_projected: bool = False
    result_content: Any = None
    result_row: Optional[dict[str, Any]] = None


@dataclass
class _ToolBatch:
    batch_id: str
    projections: list[_ToolProjection]
    started: bool = False
    completed: bool = False
    sealed: bool = False
    authoritative_seen: bool = False
    authoritative_persisted: bool = False
    journal_prepared: bool = False
    execution_error: Optional[str] = None


def _tool_signature(name: str, arguments: Any) -> str:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (TypeError, ValueError):
            pass
    try:
        encoded = json.dumps(
            arguments,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError):
        encoded = str(arguments)
    return f"{name}\0{encoded}"


def _normalized_arguments(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return dict(arguments)
    if isinstance(arguments, str):
        try:
            decoded = json.loads(arguments)
        except (TypeError, ValueError):
            return {}
        if isinstance(decoded, dict):
            return decoded
    return {}


def _schema_fingerprint(tools: list[dict[str, Any]]) -> str:
    payload = json.dumps(tools, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _unique_projection_id(
    raw_id: str,
    projections: list[_ToolProjection],
) -> str:
    """Return the standard deterministic duplicate-id repair for one turn."""
    base = str(raw_id or "").strip() or secrets.token_hex(12)
    used = {entry.local_id for entry in projections}
    if base not in used:
        return base
    suffix = 2
    candidate = f"{base}_d{suffix}"
    while candidate in used:
        suffix += 1
        candidate = f"{base}_d{suffix}"
    logger.warning(
        "Claude reused tool call id %s within one native turn; renamed the "
        "duplicate to %s to keep Hermes call/result pairing lossless.",
        base,
        candidate,
    )
    return candidate


class ClaudeAgentSdkToolBridge:
    """Expose one live AIAgent's exact tool surface to Agent SDK MCP."""

    def __init__(
        self,
        agent: Any,
        *,
        tool_definitions: Optional[list[dict[str, Any]]] = None,
        owner_key: Optional[str] = None,
        journal: Optional[ClaudeToolJournal] = None,
    ) -> None:
        self._agent = agent
        self._tool_definitions_override = tool_definitions
        self._lock = threading.RLock()
        self._projection_condition = threading.Condition(self._lock)
        self._turn: Optional[_TurnBinding] = None
        self._tool_projections: list[_ToolProjection] = []
        self._tool_batches: dict[str, _ToolBatch] = {}
        # Production sessions always supply owner_key. The no-journal path is
        # useful for direct unit callers and genuinely toolless SDK sessions.
        # A session with any executable tool cannot be constructed without the
        # durable journal.
        effective_tools = (
            tool_definitions
            if tool_definitions is not None
            else getattr(agent, "tools", None) or []
        )
        self._journal = journal or (
            ClaudeToolJournal(owner_key) if owner_key and effective_tools else None
        )
        self._stream_batch_id: Optional[str] = None
        self._stream_tool_blocks: dict[int, dict[str, Any]] = {}

    def bind_agent(self, agent: Any) -> None:
        """Rebind a pooled Claude process to the current AIAgent instance."""
        with self._lock:
            self._agent = agent

    def _tool_batch_timeout(self) -> float:
        """Use the owning native turn deadline for MCP batch coordination."""
        session = getattr(self._agent, "_claude_agent_sdk_session", None)
        try:
            timeout = float(getattr(session, "turn_timeout", 600.0))
        except (TypeError, ValueError):
            timeout = 600.0
        return max(1.0, timeout)

    def begin_turn(
        self,
        *,
        task_id: str,
        user_task: str,
        messages: list[dict[str, Any]],
        projection_callback: Optional[
            Callable[[list[dict[str, Any]]], None]
        ] = None,
        execute_tools: bool = True,
        before_next_model_callback: Optional[
            Callable[[], Optional[dict[str, Any]]]
        ] = None,
    ) -> None:
        tool_executor = None
        if execute_tools:
            # Capture this on the originating agent-turn thread, before the
            # MCP server hop. In addition to ContextVars, Hermes' standard
            # wrapper carries the CLI/ACP approval and sudo callbacks that are
            # deliberately thread-local for cross-session isolation.
            from tools.thread_context import propagate_context_to_thread

            tool_executor = propagate_context_to_thread(
                self._agent._execute_tool_calls
            )
        with self._lock:
            self._tool_projections = []
            self._tool_batches = {}
            self._stream_batch_id = None
            self._stream_tool_blocks = {}
            self._turn = _TurnBinding(
                task_id=task_id,
                user_task=user_task,
                messages=messages,
                context=contextvars.copy_context(),
                projection_callback=projection_callback,
                execute_tools=execute_tools,
                tool_executor=tool_executor,
                before_next_model_callback=before_next_model_callback,
            )

    def end_turn(self) -> None:
        with self._lock:
            self._turn = None
            self._tool_projections = []
            self._tool_batches = {}
            self._stream_batch_id = None
            self._stream_tool_blocks = {}
            self._projection_condition.notify_all()

    @staticmethod
    def _intent(entry: _ToolProjection, *, ordinal: int) -> ToolIntent:
        return ToolIntent(
            tool_use_id=entry.local_id,
            name=entry.name,
            arguments=dict(entry.arguments or {}),
            batch_id=str(entry.batch_id or ""),
            ordinal=ordinal,
        )

    def _prepare_batch_journal(self, batch: _ToolBatch) -> None:
        if self._journal is None or batch.journal_prepared:
            return
        self._journal.prepare_batch(
            self._intent(entry, ordinal=index)
            for index, entry in enumerate(batch.projections)
        )
        batch.journal_prepared = True

    def observe_stream_event(self, event: dict[str, Any]) -> None:
        """Capture a complete provisional tool batch from partial API events.

        ``message_stop`` necessarily precedes tool execution at the provider
        protocol level, even when Claude delays the later, complete assistant
        record until after MCP returns.  Its tool ids/names/JSON inputs are
        sufficient for the effect WAL and for Hermes' normal batch planner;
        narration/reasoning still comes only from the authoritative record.
        """
        if not isinstance(event, dict):
            return
        event_type = str(event.get("type") or "")
        with self._projection_condition:
            if event_type == "message_start":
                message = event.get("message")
                raw_id = (
                    str(message.get("id") or "")
                    if isinstance(message, dict)
                    else ""
                )
                self._stream_batch_id = raw_id or secrets.token_hex(12)
                self._stream_tool_blocks = {}
                return
            if self._stream_batch_id is None:
                return
            if event_type == "content_block_start":
                index = int(event.get("index") or 0)
                block = event.get("content_block")
                if not isinstance(block, dict) or block.get("type") != "tool_use":
                    return
                raw_name = str(block.get("name") or "")
                if not raw_name.startswith("mcp__hermes__"):
                    return
                self._stream_tool_blocks[index] = {
                    "id": str(block.get("id") or ""),
                    "name": _strip_mcp_prefix(raw_name),
                    "input": (
                        dict(block.get("input") or {})
                        if isinstance(block.get("input"), dict)
                        else {}
                    ),
                    "input_json": "",
                }
                return
            if event_type == "content_block_delta":
                index = int(event.get("index") or 0)
                pending = self._stream_tool_blocks.get(index)
                delta = event.get("delta")
                if pending is None or not isinstance(delta, dict):
                    return
                if delta.get("type") == "input_json_delta":
                    pending["input_json"] += str(delta.get("partial_json") or "")
                return
            if event_type == "content_block_stop":
                index = int(event.get("index") or 0)
                pending = self._stream_tool_blocks.get(index)
                if pending is None or not pending.get("input_json"):
                    return
                try:
                    decoded = json.loads(pending["input_json"])
                except (TypeError, ValueError):
                    return
                if isinstance(decoded, dict):
                    pending["input"] = decoded
                return
            if event_type != "message_stop":
                return

            batch_id = self._stream_batch_id
            blocks = [self._stream_tool_blocks[key] for key in sorted(self._stream_tool_blocks)]
            self._stream_batch_id = None
            self._stream_tool_blocks = {}
            if not blocks:
                return
            projections: list[_ToolProjection] = []
            matched_projection_ids: set[int] = set()
            for block in blocks:
                claude_id = str(block.get("id") or "")
                name = str(block.get("name") or "")
                arguments = _normalized_arguments(block.get("input") or {})
                signature = _tool_signature(name, arguments)
                match = next(
                    (
                        entry
                        for entry in self._tool_projections
                        if entry.claude_id == claude_id
                        and entry.signature == signature
                        and id(entry) not in matched_projection_ids
                    ),
                    None,
                )
                if match is None:
                    match = _ToolProjection(
                        signature=signature,
                        local_id=_unique_projection_id(
                            claude_id, self._tool_projections
                        ),
                        name=name,
                        arguments=arguments,
                        claude_id=claude_id or None,
                    )
                    self._tool_projections.append(match)
                matched_projection_ids.add(id(match))
                match.batch_id = batch_id
                projections.append(match)
            batch = self._tool_batches.get(batch_id)
            if batch is None:
                batch = _ToolBatch(
                    batch_id=batch_id,
                    projections=projections,
                    sealed=True,
                )
                self._tool_batches[batch_id] = batch
            else:
                if batch.projections != projections:
                    batch.journal_prepared = False
                batch.projections = projections
                batch.sealed = True
            self._prepare_batch_journal(batch)
            self._projection_condition.notify_all()

    def reconcile_authoritative_projection(
        self, rows: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Deduplicate Claude events against pre-execution durable rows.

        Claude's stream and its MCP request travel on different threads. The
        assistant tool-use event can arrive either before or after the MCP
        call. This registry makes both orders converge on one Hermes tool pair.
        """
        reconciled: list[dict[str, Any]] = []
        with self._lock:
            for row in rows:
                if not isinstance(row, dict):
                    continue
                if row.get("role") == "assistant" and row.get("tool_calls"):
                    self._classify_assistant_tool_narration(row)
                    kept_calls: list[dict[str, Any]] = []
                    row_projections: list[_ToolProjection] = []
                    matched_projection_ids: set[int] = set()
                    for call in row.get("tool_calls") or []:
                        function = call.get("function") or {}
                        name = str(function.get("name") or "")
                        arguments = _normalized_arguments(
                            function.get("arguments") or "{}"
                        )
                        signature = _tool_signature(
                            name, arguments
                        )
                        claude_id = str(call.get("id") or "")
                        match = next(
                            (
                                entry
                                for entry in self._tool_projections
                                if entry.claude_id == claude_id
                                and entry.signature == signature
                                and id(entry) not in matched_projection_ids
                            ),
                            None,
                        )
                        if match is None:
                            match = next(
                                (
                                    entry
                                    for entry in self._tool_projections
                                    if entry.signature == signature
                                    and entry.claude_id is None
                                    and id(entry) not in matched_projection_ids
                                ),
                                None,
                            )
                        if match is not None:
                            matched_projection_ids.add(id(match))
                            match.claude_id = claude_id
                            match.name = name
                            match.arguments = arguments
                            match.authoritative_seen = True
                            if match.intent_persisted:
                                continue
                            kept_call = dict(call)
                            kept_call["id"] = match.local_id
                            kept_calls.append(kept_call)
                            row_projections.append(match)
                            continue
                        local_id = _unique_projection_id(
                            claude_id, self._tool_projections
                        )
                        match = _ToolProjection(
                            signature=signature,
                            local_id=local_id,
                            name=name,
                            arguments=arguments,
                            claude_id=claude_id or None,
                            authoritative_seen=True,
                        )
                        self._tool_projections.append(match)
                        kept_call = dict(call)
                        kept_call["id"] = local_id
                        kept_calls.append(kept_call)
                        row_projections.append(match)
                    if kept_calls:
                        existing_batch_id = next(
                            (
                                entry.batch_id
                                for entry in row_projections
                                if entry.batch_id is not None
                            ),
                            None,
                        )
                        batch_id = existing_batch_id or secrets.token_hex(12)
                        batch = self._tool_batches.get(batch_id)
                        if batch is None:
                            batch = _ToolBatch(
                                batch_id=batch_id,
                                projections=list(row_projections),
                            )
                            self._tool_batches[batch_id] = batch
                        else:
                            # Partial stream events can establish this batch
                            # before the complete assistant record.  Keep its
                            # original order, adding any forward-compatible
                            # call that appeared only in the authoritative
                            # record rather than replacing cached executions.
                            known = {id(entry) for entry in batch.projections}
                            added = [
                                entry
                                for entry in row_projections
                                if id(entry) not in known
                            ]
                            if added:
                                batch.projections.extend(added)
                                batch.journal_prepared = False
                        for entry in row_projections:
                            entry.batch_id = batch_id
                        batch.sealed = True
                        batch.authoritative_seen = True
                        self._prepare_batch_journal(batch)
                        kept = dict(row)
                        kept["tool_calls"] = kept_calls
                        reconciled.append(kept)
                        # When Claude withheld the complete assistant record
                        # until after MCP returned, the effect journal already
                        # contains the completed results.  Persist the
                        # authoritative assistant row and those cached results
                        # together now; the later native user/tool-result event
                        # is a duplicate projection and will be dropped below.
                        if batch.completed:
                            for entry in batch.projections:
                                if entry.result_ready and entry.result_row is not None:
                                    reconciled.append(dict(entry.result_row))
                    continue

                if row.get("role") == "assistant":
                    # Match the standard loop's final text branch. A previous
                    # answer+housekeeping batch may have muted post-response
                    # diagnostics; a new text-only response restores them.
                    try:
                        self._agent._mute_post_response = False
                    except Exception:
                        pass

                if row.get("role") == "tool":
                    claude_id = str(row.get("tool_call_id") or "")
                    match = next(
                        (
                            entry
                            for entry in self._tool_projections
                            if entry.claude_id == claude_id
                        ),
                        None,
                    )
                    if match is not None and match.result_projected:
                        continue
                    if match is not None and match.local_id != claude_id:
                        kept = dict(row)
                        kept["tool_call_id"] = match.local_id
                        reconciled.append(kept)
                    else:
                        reconciled.append(row)
                    continue

                reconciled.append(row)
            self._projection_condition.notify_all()
        return reconciled

    def _classify_assistant_tool_narration(self, row: dict[str, Any]) -> None:
        """Mirror standard-loop answer+housekeeping output semantics.

        Claude streams assistant text before its authoritative assistant event
        arrives. Once that event shows the complete tool batch, Hermes can
        distinguish a delivered answer followed only by housekeeping from
        narration before substantive work. The distinction controls both
        empty-follow-up recovery and IM/tool-progress noise.
        """
        agent = self._agent
        calls = row.get("tool_calls") or []
        names = {
            str((call.get("function") or {}).get("name") or "")
            for call in calls
            if isinstance(call, dict)
        }
        all_housekeeping = bool(names) and names.issubset(_HOUSEKEEPING_TOOLS)

        if not all_housekeeping:
            agent._last_content_with_tools = None
            agent._last_content_tools_all_housekeeping = False
            agent._mute_post_response = False
            return

        content = row.get("content")
        has_visible_content = bool(content)
        content_check = getattr(agent, "_has_content_after_think_block", None)
        if callable(content_check):
            try:
                has_visible_content = bool(content_check(content))
            except Exception:
                has_visible_content = bool(content)
        if not has_visible_content:
            return

        agent._last_content_with_tools = content
        agent._last_content_tools_all_housekeeping = True
        has_stream_consumers = getattr(agent, "_has_stream_consumers", None)
        if callable(has_stream_consumers):
            try:
                agent._mute_post_response = bool(has_stream_consumers())
            except Exception:
                pass

    def register_tool_request(
        self, *, name: str, arguments: dict[str, Any], claude_id: str
    ) -> None:
        """Capture the native tool id exposed by Claude's permission event."""
        if not claude_id:
            return
        signature = _tool_signature(name, arguments)
        with self._lock:
            existing = next(
                (
                    entry
                    for entry in self._tool_projections
                    if entry.claude_id == claude_id
                    and entry.signature == signature
                ),
                None,
            )
            if existing is not None:
                if self._journal is not None and existing.batch_id:
                    batch = self._tool_batches.get(existing.batch_id)
                    if batch is not None:
                        self._prepare_batch_journal(batch)
                return
            local_id = _unique_projection_id(
                claude_id, self._tool_projections
            )
            projection = _ToolProjection(
                signature=signature,
                local_id=local_id,
                name=name,
                arguments=dict(arguments),
                claude_id=claude_id,
            )
            self._tool_projections.append(projection)
            # Hooks can arrive even when partial streaming is unavailable.
            # A singleton provisional batch is sufficient to break the
            # protocol cycle; a later stream/authoritative record merges it
            # into the complete batch before execution whenever available.
            batch_id = f"hook:{claude_id}"
            projection.batch_id = batch_id
            batch = _ToolBatch(
                batch_id=batch_id,
                projections=[projection],
                sealed=True,
            )
            self._tool_batches[batch_id] = batch
            self._prepare_batch_journal(batch)
            self._projection_condition.notify_all()

    def has_partial_batch(self, *, name: str, arguments: dict[str, Any]) -> bool:
        """Whether partial message_stop sealed this call's complete batch."""
        signature = _tool_signature(name, arguments)
        with self._lock:
            return any(
                entry.signature == signature
                and entry.batch_id is not None
                and not entry.batch_id.startswith("hook:")
                and not entry.claimed
                for entry in self._tool_projections
            )

    def mark_authoritative_projection_persisted(
        self, rows: list[dict[str, Any]], *, succeeded: bool
    ) -> None:
        """Release any MCP call waiting for its streamed intent to persist."""
        with self._projection_condition:
            ids = {
                str(call.get("id") or "")
                for row in rows
                if isinstance(row, dict) and row.get("role") == "assistant"
                for call in (row.get("tool_calls") or [])
            }
            for entry in self._tool_projections:
                if entry.local_id in ids or entry.claude_id in ids:
                    if succeeded:
                        entry.intent_persisted = True
                    else:
                        entry.intent_persistence_failed = True
            for batch in self._tool_batches.values():
                if any(
                    entry.local_id in ids or entry.claude_id in ids
                    for entry in batch.projections
                ):
                    batch.authoritative_persisted = bool(succeeded)
            # A delayed authoritative projection can contain both the
            # assistant tool calls and cached results in one atomic flush.
            persisted_result_ids = {
                str(row.get("tool_call_id") or "")
                for row in rows
                if isinstance(row, dict) and row.get("role") == "tool"
            }
            for entry in self._tool_projections:
                if entry.local_id in persisted_result_ids:
                    entry.result_projected = bool(succeeded)
            if succeeded and self._journal is not None:
                reconciled = [
                    entry.local_id
                    for entry in self._tool_projections
                    if entry.intent_persisted and entry.result_projected
                ]
                self._journal.mark_reconciled(reconciled)
            self._projection_condition.notify_all()

    def tool_definitions(self) -> list[dict[str, Any]]:
        with self._lock:
            tools = (
                self._tool_definitions_override
                if self._tool_definitions_override is not None
                else getattr(self._agent, "tools", None) or []
            )
            return [dict(tool) for tool in tools if isinstance(tool, dict)]

    def fingerprint(self) -> str:
        return _schema_fingerprint(self.tool_definitions())

    def close(self) -> None:
        return None

    def _list_tools(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for entry in self.tool_definitions():
            function = entry.get("function") or {}
            name = str(function.get("name") or "").strip()
            if not name:
                continue
            result.append(
                {
                    "name": name,
                    "description": str(function.get("description") or ""),
                    "inputSchema": function.get("parameters")
                    or {"type": "object", "properties": {}},
                }
            )
        return result

    def _call_tool(self, params: dict[str, Any]) -> Any:
        name = str(params.get("name") or "").strip()
        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            raise TypeError("tool arguments must be an object")

        definitions = {tool["name"] for tool in self._list_tools()}
        if name not in definitions:
            raise PermissionError(f"tool is not enabled in this session: {name}")

        with self._lock:
            agent = self._agent
            turn = self._turn
        if turn is None:
            raise RuntimeError("no active Claude turn is bound to the SDK bridge")

        # Auxiliary callers sometimes ask the model to *propose* a tool call
        # (MCP sampling and the MoA acting aggregator) and must receive that
        # structured call back for their own host loop to execute.  The
        # transient Claude session is stopped as soon as its authoritative
        # assistant tool-use record arrives; this fail-closed response covers
        # the small stream/MCP race before that stop and guarantees the
        # proposal can never execute inside Hermes as a side effect.
        if not turn.execute_tools:
            return (
                "Tool proposal captured by the host. Do not continue or "
                "attempt to execute it inside this auxiliary session."
            )

        # A halt decision is turn-scoped and terminal. If Claude asks for a
        # call without first emitting a new authoritative tool batch, answer
        # immediately and never allow the follow-up side effect. Normal
        # authoritative batches run through Hermes' executor below, which
        # applies the same guardrail semantics as the standard loop.
        halt_decision = getattr(agent, "_tool_guardrail_halt_decision", None)
        with self._lock:
            has_pending_projection = any(
                entry.signature == _tool_signature(name, arguments)
                and not entry.claimed
                for entry in self._tool_projections
            )
        if halt_decision is not None and not has_pending_projection:
            return agent._guardrail_block_result(halt_decision)

        # Claude's MCP request is an execution-control event, not a transcript
        # event.  It may arrive before the complete assistant message and must
        # never wait for that message (Claude can itself be waiting for this
        # MCP result).  The partial-stream batch or PreToolUse registration is
        # written to the independent effect WAL first.  The complete event
        # remains authoritative for later transcript projection only.
        signature = _tool_signature(name, arguments)
        with self._projection_condition:
            projection = next(
                (
                    entry
                    for entry in self._tool_projections
                    if entry.signature == signature
                    and entry.claude_id is not None
                    and not entry.claimed
                ),
                None,
            )
            if projection is None:
                raw_local_id = str(params.get("tool_call_id") or "")
                if not raw_local_id:
                    raise RuntimeError(
                        "Claude MCP request arrived without a durable native "
                        "tool_use_id; refusing to execute the side effect"
                    )
                projection = _ToolProjection(
                    signature=signature,
                    local_id=_unique_projection_id(
                        raw_local_id, self._tool_projections
                    ),
                    name=name,
                    arguments=dict(arguments),
                    claude_id=raw_local_id or None,
                    claimed=True,
                )
                self._tool_projections.append(projection)
            else:
                projection.name = name
                projection.arguments = dict(arguments)
                projection.claimed = True
            batch = (
                self._tool_batches.get(projection.batch_id)
                if projection.batch_id is not None
                else None
            )
            if batch is None:
                batch_id = f"mcp:{projection.local_id}"
                projection.batch_id = batch_id
                batch = _ToolBatch(
                    batch_id=batch_id,
                    projections=[projection],
                    sealed=True,
                )
                self._tool_batches[batch_id] = batch
            self._prepare_batch_journal(batch)
            self._projection_condition.notify_all()

            if (
                projection.intent_persistence_failed
                or getattr(agent, "_incremental_persistence_failed", False)
            ):
                before_next_model = turn.before_next_model_callback
                if callable(before_next_model):
                    turn.context.copy().run(before_next_model)
                raise RuntimeError(
                    "Claude transcript persistence already failed; refusing "
                    f"to admit side-effecting tool {name}"
                )

            if not batch.started:
                batch.started = True
                is_batch_leader = True
            else:
                is_batch_leader = False

            if not is_batch_leader:
                completion_deadline = time.monotonic() + self._tool_batch_timeout()
                while not batch.completed and time.monotonic() < completion_deadline:
                    self._projection_condition.wait(timeout=0.1)
                if not batch.completed:
                    raise RuntimeError(
                        f"Hermes tool batch timed out while waiting for {name}"
                    )
                if batch.execution_error is not None:
                    raise RuntimeError(batch.execution_error)
                return projection.result_content

        projection_callback = turn.projection_callback
        call_context = turn.context.copy()
        scratch_messages = list(turn.messages)
        initial_message_count = len(scratch_messages)
        all_tool_calls = [
            SimpleNamespace(
                id=item.local_id,
                function=SimpleNamespace(
                    name=item.name,
                    arguments=json.dumps(
                        item.arguments or {}, ensure_ascii=False
                    ),
                ),
            )
            for item in batch.projections
        ]
        execution_calls: list[Any] = []
        skipped_results: dict[str, dict[str, Any]] = {}
        replayed_results: dict[str, dict[str, Any]] = {}

        if self._journal is not None:
            claims = self._journal.claim_batch(
                self._intent(entry, ordinal=index)
                for index, entry in enumerate(batch.projections)
            )
            claims_by_id = {claim.tool_use_id: claim for claim in claims}
            unknown_ids = {
                claim.tool_use_id
                for claim in claims
                if claim.disposition == "unknown"
            }
            for call in all_tool_calls:
                claim = claims_by_id[call.id]
                if claim.disposition == "replay":
                    assert claim.result_row is not None
                    replayed_results[call.id] = dict(claim.result_row)
                elif claim.disposition == "execute":
                    execution_calls.append(call)
            if unknown_ids:
                # An earlier process may have completed an external side
                # effect after the durable ``running`` transition.  Abort all
                # still-new calls in this batch rather than cross that
                # uncertainty boundary or risk a duplicate side effect.
                for call in execution_calls:
                    skipped_results[call.id] = {
                        "role": "tool",
                        "name": call.function.name,
                        "tool_call_id": call.id,
                        "content": (
                            "[Tool execution skipped — another tool in this "
                            "Claude batch has an unknown outcome after a prior "
                            "process interruption.]"
                        ),
                        "effect_disposition": "none",
                    }
                execution_calls = []
                for tool_use_id in unknown_ids:
                    entry = next(
                        item
                        for item in batch.projections
                        if item.local_id == tool_use_id
                    )
                    skipped_results[tool_use_id] = {
                        "role": "tool",
                        "name": entry.name,
                        "tool_call_id": tool_use_id,
                        "content": (
                            "[Tool outcome unknown — Hermes restarted after "
                            "admitting this side effect and will not execute it "
                            "again automatically. Verify the external state "
                            "before retrying.]"
                        ),
                        "effect_disposition": "unknown",
                    }
        else:
            execution_calls = list(all_tool_calls)

        cap_delegate_calls = getattr(agent, "_cap_delegate_task_calls", None)
        if callable(cap_delegate_calls):
            capped_calls = list(cap_delegate_calls(execution_calls))
            capped_identities = {id(call) for call in capped_calls}
            for call in execution_calls:
                if id(call) not in capped_identities:
                    skipped_results[call.id] = {
                        "role": "tool",
                        "name": call.function.name,
                        "tool_call_id": call.id,
                        "content": (
                            "[Tool execution skipped — the delegate_task batch "
                            "exceeded Hermes' max_concurrent_children limit.]"
                        ),
                        "effect_disposition": "none",
                    }
            execution_calls = capped_calls

        deduplicate_calls = getattr(agent, "_deduplicate_tool_calls", None)
        if callable(deduplicate_calls):
            unique_calls = list(deduplicate_calls(execution_calls))
            unique_identities = {id(call) for call in unique_calls}
            for call in execution_calls:
                if id(call) not in unique_identities:
                    skipped_results[call.id] = {
                        "role": "tool",
                        "name": call.function.name,
                        "tool_call_id": call.id,
                        "content": (
                            "[Tool execution skipped — duplicate call in the "
                            "same assistant batch.]"
                        ),
                        "effect_disposition": "none",
                    }
            execution_calls = unique_calls

        assistant = SimpleNamespace(tool_calls=execution_calls)

        display_callback = getattr(agent, "stream_delta_callback", None)
        if display_callback is not None:
            try:
                display_callback(None)
            except Exception:
                pass
        thinking_callback = getattr(agent, "thinking_callback", None)
        if callable(thinking_callback):
            try:
                thinking_callback("")
            except Exception:
                pass

        started = time.monotonic()
        try:
            if execution_calls:
                if turn.tool_executor is None:
                    raise RuntimeError(
                        "Claude turn has no captured Hermes tool executor"
                    )
                turn.tool_executor(
                    assistant,
                    scratch_messages,
                    turn.task_id,
                    int(getattr(agent, "_api_call_count", 0) or 0),
                    persist_progress=False,
                )
            tool_names = {call.function.name for call in execution_calls}
            if execution_calls and tool_names == {"execute_code"}:
                budget = getattr(agent, "iteration_budget", None)
                if budget is not None:
                    budget.refund()

            executed_result_rows = [
                dict(message)
                for message in scratch_messages[initial_message_count:]
                if isinstance(message, dict) and message.get("role") == "tool"
            ]
            results_by_id = {
                str(message.get("tool_call_id") or ""): message
                for message in executed_result_rows
            }
            results_by_id.update(replayed_results)
            results_by_id.update(skipped_results)
            missing = [
                item.name
                for item in batch.projections
                if item.local_id not in results_by_id
            ]
            if missing:
                raise RuntimeError(
                    "Hermes tool executor produced no result for: "
                    + ", ".join(missing)
                )

            # Claude's native protocol requires one result per emitted call,
            # in the original assistant order. Filtered duplicates/excess
            # delegations therefore get explicit non-effect results rather
            # than disappearing as they can on the host-owned HTTP wire.
            result_rows = [
                results_by_id[item.local_id]
                for item in batch.projections
            ]

            if self._journal is not None:
                durable_rows = [
                    row
                    for row in result_rows
                    if str(row.get("tool_call_id") or "")
                    not in replayed_results
                    and str(row.get("tool_call_id") or "")
                    not in {
                        claim.tool_use_id
                        for claim in claims
                        if claim.disposition == "unknown"
                    }
                ]
                self._journal.complete(durable_rows)

            # If the authoritative assistant row won the race, append its
            # results now exactly as the standard Hermes loop does.  If MCP
            # won, retain the completed rows in the effect WAL/in-memory
            # registry; reconcile_authoritative_projection() appends them
            # immediately after the delayed authoritative row arrives.
            projected_now = bool(batch.authoritative_persisted)
            if projected_now and projection_callback is not None:
                projection_callback(result_rows)
            elif projection_callback is None:
                turn.messages.extend(result_rows)
                projected_now = True
            if projected_now and getattr(
                agent, "_incremental_persistence_failed", False
            ):
                before_next_model = turn.before_next_model_callback
                if callable(before_next_model):
                    call_context.run(before_next_model)
                raise RuntimeError(
                    "Claude tool results could not be persisted after "
                    "executing the authoritative batch"
                )

            # Claude will issue its next provider request only after every MCP
            # result in this batch has returned.  This callback therefore sits
            # on the last host-controlled boundary where Hermes can emit the
            # documented per-provider ``pre_api_request`` observer event.
            before_next_model = turn.before_next_model_callback
            wire_result_overrides: dict[str, Any] = {}
            if callable(before_next_model):
                candidate = call_context.run(before_next_model)
                if isinstance(candidate, dict):
                    wire_result_overrides = candidate

            agent._stream_needs_break = True
            with self._projection_condition:
                for item in batch.projections:
                    item.result_ready = True
                    item.result_projected = projected_now
                    item.result_row = dict(results_by_id[item.local_id])
                    item.result_content = wire_result_overrides.get(
                        item.local_id,
                        results_by_id[item.local_id].get("content"),
                    )
                batch.completed = True
                if (
                    projected_now
                    and self._journal is not None
                    and batch.authoritative_persisted
                ):
                    self._journal.mark_reconciled(
                        item.local_id for item in batch.projections
                    )
                self._projection_condition.notify_all()
            logger.debug(
                "Claude MCP tool batch completed (tools=%s duration_ms=%d)",
                ",".join(item.name for item in batch.projections),
                int((time.monotonic() - started) * 1000),
            )
            return projection.result_content
        except BaseException as exc:
            with self._projection_condition:
                batch.execution_error = (
                    "Hermes authoritative tool batch failed: " + str(exc)
                )
                batch.completed = True
                self._projection_condition.notify_all()
            raise

__all__ = ["ClaudeAgentSdkToolBridge"]
