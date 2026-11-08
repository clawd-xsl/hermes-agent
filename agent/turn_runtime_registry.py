"""Turn runtime registry.

Maps an ``api_mode`` to the :class:`agent.turn_runtime_provider.TurnRuntimeProvider` that owns
turns in that mode. The codex app-server runtime is built in; plugins register others through
:meth:`PluginContext.register_turn_runtime`. Core call sites never compare ``api_mode`` against a
runtime's name: they ask :func:`turn_runtime_for` and call the runtime's hooks.

A lookup for an ``api_mode`` that is neither built in, registered, nor one of the HTTP wire modes
runs one plugin discovery sweep first, so a caller that never imported ``model_tools`` (runtime
resolution, a detached worker) still finds a plugin runtime selected in config.yaml.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional

from agent.provider_registry import ProviderRegistry, lower_key
from agent.turn_runtime_provider import TurnRuntimeProvider

logger = logging.getLogger(__name__)

#: Runtimes implemented in core. Plugins cannot register these names.
BUILTIN_RUNTIMES = frozenset({"codex_app_server"})

#: Per-request HTTP wire modes: never a turn runtime, so a lookup never triggers discovery.
_WIRE_API_MODES = frozenset({
    "", "chat_completions", "codex_responses", "anthropic_messages", "bedrock_converse",
})


def _builtin_collision(name: str) -> None:
    logger.warning("Turn runtime '%s' is built in; a plugin cannot replace it", name)


_registry: ProviderRegistry[TurnRuntimeProvider] = ProviderRegistry(
    label="Turn runtime", provider_cls=TurnRuntimeProvider, logger=logger, normalize=lower_key,
    builtin_names=BUILTIN_RUNTIMES, on_builtin_collision=_builtin_collision,
)
_registry.export(globals())

_builtin_lock = threading.Lock()
_builtins: dict[str, TurnRuntimeProvider] = {}


def _builtin(name: str) -> Optional[TurnRuntimeProvider]:
    if name not in BUILTIN_RUNTIMES:
        return None
    with _builtin_lock:
        runtime = _builtins.get(name)
        if runtime is None:
            from agent.codex_runtime import CodexAppServerTurnRuntime
            runtime = _builtins[name] = CodexAppServerTurnRuntime()
        return runtime


def _discover_once() -> None:
    try:
        from hermes_cli.plugins import discover_plugins
        discover_plugins()
    except Exception:
        logger.debug("turn runtime lookup: plugin discovery failed", exc_info=True)


def turn_runtime_for(api_mode: Any) -> Optional[TurnRuntimeProvider]:
    """The runtime owning turns in ``api_mode``, or None for a per-request wire mode."""
    if not isinstance(api_mode, str):
        return None
    name = lower_key(api_mode)
    if name in _WIRE_API_MODES:
        return None
    builtin = _builtin(name)
    if builtin is not None:
        return builtin
    runtime = _registry.get_provider(name)
    if runtime is None:
        _discover_once()
        runtime = _registry.get_provider(name)
    return runtime


def agent_turn_runtime(agent: Any) -> Optional[TurnRuntimeProvider]:
    """The runtime owning ``agent``'s turns (by its current ``api_mode``), or None."""
    return turn_runtime_for(getattr(agent, "api_mode", None))


def is_turn_runtime_mode(api_mode: Any) -> bool:
    """True when ``api_mode`` names a built-in or registered turn runtime."""
    return turn_runtime_for(api_mode) is not None


def all_turn_runtimes() -> list[TurnRuntimeProvider]:
    """Every built-in and currently registered runtime (no discovery sweep: teardown paths use this)."""
    builtins = [rt for rt in (_builtin(name) for name in sorted(BUILTIN_RUNTIMES)) if rt is not None]
    return builtins + list(_registry.list_providers())


def release_runtime_sessions(*session_ids: Any) -> int:
    """Ask every runtime to release what it holds for ``session_ids`` (one-shot runs that ended with no
    live agent); the total released. Blank and duplicate ids are dropped; a failing runtime is logged
    and skipped so the others still release."""
    ids = list(dict.fromkeys(str(sid).strip() for sid in session_ids if sid and str(sid).strip()))
    if not ids:
        return 0
    released = 0
    for runtime in all_turn_runtimes():
        try:
            released += int(runtime.release_sessions(list(ids)) or 0)
        except Exception:
            logger.debug("turn runtime %s.release_sessions() failed", runtime.name, exc_info=True)
    return released


def turn_runtime_names() -> frozenset[str]:
    """Every runtime name currently available (built in + registered)."""
    return BUILTIN_RUNTIMES | frozenset(p.name for p in _registry.list_providers())


def resolve_runtime_overlay(
    *, provider: str, api_mode: str, model_cfg: Optional[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    """The first registered plugin runtime opting ``provider`` in via ``model_cfg``, as overrides.

    Built-in runtimes keep their own selection in ``hermes_cli.runtime_provider``."""
    if not model_cfg:
        return None
    _discover_once()
    for runtime in _registry.list_providers():
        try:
            overlay = runtime.resolve_runtime(provider=provider, api_mode=api_mode, model_cfg=model_cfg)
        except Exception:
            logger.warning("turn runtime %s.resolve_runtime() raised", runtime.name, exc_info=True)
            continue
        if overlay:
            overlay = dict(overlay)
            overlay["api_mode"] = runtime.name
            return overlay
    return None


def configured_runtime_mode(provider: Any) -> Optional[str]:
    """The plugin runtime ``config.yaml``'s ``model`` section opts ``provider`` into, or None. For readers
    with no live runtime bound (turn-external auxiliary calls, status surfaces)."""
    if not isinstance(provider, str) or not provider.strip():
        return None
    try:
        from hermes_cli.config import load_config_readonly
        cfg = load_config_readonly() or {}
    except Exception:
        return None
    model_cfg = cfg.get("model") if isinstance(cfg, dict) else None
    if not isinstance(model_cfg, dict):
        return None
    overlay = resolve_runtime_overlay(provider=lower_key(provider), api_mode="", model_cfg=model_cfg)
    return overlay["api_mode"] if overlay else None


def route_authenticates_itself(runtime: Any) -> bool:
    """True for a resolved runtime dict routed to a turn runtime: it carries no API key by design, so
    "no api_key" gates (``/btw``, ``/background``, delegation) must not treat it as unconfigured."""
    return isinstance(runtime, dict) and is_turn_runtime_mode(runtime.get("api_mode"))


def call_runtime_hook(agent: Any, hook: str, *args: Any, default: Any = None, **kwargs: Any) -> Any:
    """Call ``hook`` on ``agent``'s runtime, logging (not raising) a failure; ``default`` when the
    agent has no runtime or the hook raised."""
    runtime = agent_turn_runtime(agent)
    if runtime is None:
        return default
    try:
        return getattr(runtime, hook)(agent, *args, **kwargs)
    except Exception:
        logger.debug("turn runtime %s.%s() failed", runtime.name, hook, exc_info=True)
        return default
