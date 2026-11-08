"""Bind a live agent onto a turn runtime's route (``agent.turn_runtime_registry``).

Every place that swaps an agent's route in place — fallback activation (``route_binding``), primary
restore (``agent_runtime_helpers._rebuild_primary_client``), ``/model`` (``switch_model``) — builds
an HTTP client for the destination. A turn runtime that owns inference and authenticates itself has
no client, endpoint or key, so those sites call :func:`bind_turn_runtime_client` instead.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from agent.turn_runtime_provider import TurnRuntimeProvider

logger = logging.getLogger(__name__)


def runtime_for_route(api_mode: Any) -> Optional[TurnRuntimeProvider]:
    from agent.turn_runtime_registry import turn_runtime_for
    return turn_runtime_for(api_mode)


def runtime_identity_base_url(api_mode: Any, base_url: Any) -> str:
    """The endpoint a backend identity compares: a turn runtime is its own deployment (``<mode>://local``)
    even when provider + model match an HTTP route, so HTTP -> runtime stays a viable fallback while the
    same runtime twice still de-duplicates."""
    runtime = runtime_for_route(api_mode)
    if runtime is not None:
        return f"{runtime.name}://local"
    return str(base_url or "")


def bind_turn_runtime_client(agent: Any, runtime: TurnRuntimeProvider) -> None:
    """Clear every HTTP client/credential carrier and let ``runtime`` own inference, so a stale key can
    neither be required nor forwarded to the runtime's own process."""
    agent.client = None
    agent._client_kwargs = {}
    agent.api_key = ""
    agent.base_url = ""
    agent._anthropic_client = None
    agent._anthropic_api_key = ""
    agent._anthropic_base_url = ""
    agent._is_anthropic_oauth = False
    agent._credential_pool = None
    agent._credential_pool_entry_id = None
    if hasattr(agent, "_transport_cache"):
        agent._transport_cache.clear()
    runtime.init_client(agent)


def release_left_runtime(agent: Any, old_api_mode: Any, old_model: Any) -> None:
    """After a committed route swap, let the runtime the agent left (or whose model changed) release the
    child it held for the old route; a failed swap rolls back first and never reaches this."""
    old_runtime = runtime_for_route(old_api_mode)
    if old_runtime is None:
        return
    if runtime_for_route(getattr(agent, "api_mode", None)) is old_runtime and getattr(agent, "model", None) == old_model:
        return
    try:
        old_runtime.release(agent, hard=True)
    except Exception:
        logger.debug("route swap: %s release failed", old_runtime.name, exc_info=True)
