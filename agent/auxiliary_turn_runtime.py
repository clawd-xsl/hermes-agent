"""Auxiliary routing onto a turn runtime (``agent.turn_runtime_registry``).

A main agent whose turns run on a turn runtime that authenticates itself (a CLI child logged in to
its own account) has no HTTP credential for ``resolve_provider_client`` to find, so auxiliary work
(titles, summaries, vision, judges, plugin LLM calls) on the main provider would silently fail. The
runtime may supply its own isolated auxiliary client; this module decides when it applies.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from agent.turn_runtime_provider import TurnRuntimeProvider

logger = logging.getLogger("agent.auxiliary_client")


def _main_api_mode(main_runtime: Optional[dict[str, Any]], main_provider: str) -> str:
    from agent.auxiliary_client import _runtime_main_value
    from agent.turn_runtime_registry import configured_runtime_mode
    mode = ""
    if isinstance(main_runtime, dict):
        mode = str(main_runtime.get("api_mode") or "")
    mode = mode or str(_runtime_main_value("api_mode") or "")
    return mode or configured_runtime_mode(main_provider) or ""


def runtime_for_auxiliary(
    provider: str, api_mode: Optional[str], explicit_base_url: Any, explicit_api_key: Any,
    main_runtime: Optional[dict[str, Any]],
) -> Optional[TurnRuntimeProvider]:
    """The turn runtime an auxiliary request for ``provider`` should use, or None.

    An explicit ``api_mode`` naming a runtime selects it. Otherwise a request with no explicit
    endpoint/key for the provider the main agent runs on inherits the main runtime: auxiliary slots
    persist only provider + model, never the main transport."""
    from agent.turn_runtime_registry import turn_runtime_for
    if api_mode:
        return turn_runtime_for(api_mode)
    if explicit_base_url or explicit_api_key or not provider or provider == "auto":
        return None
    from agent.auxiliary_client import _read_main_provider
    main_provider = ""
    if isinstance(main_runtime, dict):
        main_provider = str(main_runtime.get("provider") or "")
    main_provider = (main_provider or str(_read_main_provider() or "")).strip().lower()
    if main_provider != provider:
        return None
    return turn_runtime_for(_main_api_mode(main_runtime, main_provider))


def resolve_runtime_auxiliary_client(
    provider: str, model: Optional[str], *, async_mode: bool, is_vision: bool, task: Optional[str],
    api_mode: Optional[str], explicit_base_url: Any, explicit_api_key: Any, main_runtime: Optional[dict[str, Any]],
) -> Optional[tuple[Any, Optional[str]]]:
    """``(client, model)`` from the applicable runtime, or None to continue normal resolution."""
    runtime = runtime_for_auxiliary(provider, api_mode, explicit_base_url, explicit_api_key, main_runtime)
    if runtime is None:
        return None
    if not model:
        from agent.auxiliary_client import _read_main_model
        model = _read_main_model()
    try:
        return runtime.auxiliary_client(model=model or "", async_mode=async_mode, is_vision=is_vision, task=task)
    except Exception:
        logger.warning("turn runtime %s auxiliary client failed", runtime.name, exc_info=True)
        return None


def main_route_api_mode(runtime_api_mode: str, main_provider: str) -> str:
    """Step-1 auxiliary routing: a turn-external caller (no live runtime bound) still routes the main
    provider through the configured runtime."""
    if runtime_api_mode:
        return runtime_api_mode
    from agent.turn_runtime_registry import configured_runtime_mode
    return configured_runtime_mode(main_provider) or ""
