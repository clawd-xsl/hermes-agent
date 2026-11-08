"""Route automation (cron fires, webhook events) into the gateway's real home conversation.

``session: main`` is deliberately not a second agent runtime. It resolves the configured home
conversation and commits a producer wake for that exact route, the way Kanban push wakes and
``/loop`` ticks do (``gateway.session_automation.admit_route_wake``): under a session authority the
wake lands in the durable FIFO behind any turn already admitted there; otherwise it is admitted as
an internal ``MessageEvent`` through the adapter (``gateway.wake.admit_internal_event``), which
queues it as its own turn behind a busy session. Either way the turn reuses the live agent, the
full transcript, the provider session and prompt cache, and the normal delivery path.

The receipt means "committed to the home conversation's FIFO", never model execution or delivery.
"""

from __future__ import annotations

import dataclasses
import logging
import uuid
from dataclasses import dataclass
from typing import Any, Mapping, Optional

logger = logging.getLogger(__name__)

SESSION_ISOLATED = "isolated"
SESSION_MAIN = "main"
SESSION_MODES = frozenset({SESSION_ISOLATED, SESSION_MAIN})

# Producer names for ``admit_route_wake``; listed in ``session_automation._ROUTE_WAKE_PRODUCERS`` so a
# wake still queued when the home route rotates (/new, /reset) follows it to the fresh session.
CRON_PRODUCER = "cron-main"
WEBHOOK_PRODUCER = "webhook-main"


class MainSessionUnavailable(RuntimeError):
    """A real, live main conversation cannot be resolved or does not accept the turn right now."""


@dataclass(frozen=True)
class MainSessionEnqueueResult:
    """Identity of a turn the home conversation's FIFO accepted."""

    session_key: str
    platform: str
    chat_id: str
    queued: bool  # the home session was busy: the turn waits behind the running one
    durable: bool  # committed to the session authority's ledger (survives a restart)


def normalize_session_mode(value: Any, *, default: str = SESSION_ISOLATED) -> str:
    """Normalize a user-facing ``main|isolated`` value; blank means ``default``; anything else raises."""
    raw = default if value is None or str(value).strip() == "" else value
    mode = str(raw).strip().lower()
    if mode not in SESSION_MODES:
        raise ValueError(f"Invalid session mode {value!r}; expected one of: {', '.join(sorted(SESSION_MODES))}")
    return mode


def _normalized_profile(value: Any) -> str:
    raw = str(value or "").strip()
    return "default" if raw in {"", "default", "main"} else raw


def _origin_source(origin: Any, *, profile: Optional[str] = None):
    """The ``SessionSource`` a cron job's stored origin names, or None when it names no live platform."""
    from gateway.config import Platform
    from gateway.session import SessionSource
    if not isinstance(origin, Mapping):
        return None
    platform_name = str(origin.get("platform") or "").strip().lower()
    chat_id = str(origin.get("chat_id") or "").strip()
    if not platform_name or not chat_id:
        return None
    try:
        platform = Platform(platform_name)
    except ValueError:
        return None

    def _opt(key: str) -> Optional[str]:
        return str(origin.get(key) or "").strip() or None

    chat_type = (_opt("chat_type") or ("group" if chat_id.startswith("group:") else "dm")).lower()
    return SessionSource(
        platform=platform, chat_id=chat_id, chat_name=_opt("chat_name"), chat_type=chat_type,
        user_id=_opt("user_id") or chat_id, user_name=_opt("user_name"), thread_id=_opt("thread_id"),
        scope_id=_opt("scope_id") or _opt("guild_id"), profile=profile or _opt("profile"))


def _source_matches_home(source: Any, platform: Any, home: Any, *, profile: Optional[str]) -> bool:
    return bool(
        source is not None
        and getattr(source, "platform", None) == platform
        and str(getattr(source, "chat_id", "")) == str(home.chat_id)
        and str(getattr(source, "thread_id", "") or "") == str(home.thread_id or "")
        and _normalized_profile(getattr(source, "profile", None)) == _normalized_profile(profile))


def _home_candidates(runner: Any, profile: Optional[str]) -> list[tuple[Any, Any, Any]]:
    """``(platform, home_channel, adapter)`` for every connected adapter of *profile* with a home."""
    adapters = runner._adapters_for_profile(profile) if hasattr(runner, "_adapters_for_profile") else (
        getattr(runner, "adapters", {}) or {})
    primary_configs = getattr(getattr(runner, "config", None), "platforms", {}) or {}
    candidates = []
    for platform, adapter in adapters.items():
        # A secondary profile's adapter carries its own loaded config; runner.config is only the
        # launch profile's, so read home_channel off the adapter to stay inside the routed profile.
        platform_config = getattr(adapter, "config", None) or primary_configs.get(platform)
        home = getattr(platform_config, "home_channel", None)
        if (adapter is not None and getattr(platform_config, "enabled", False) and home is not None
                and str(getattr(home, "chat_id", "") or "").strip()):
            candidates.append((platform, home, adapter))
    return candidates


def _pick_home(candidates: list, fallback: Any, profile: Optional[str]) -> tuple[Any, Any, Any]:
    if len(candidates) > 1 and fallback is not None:
        matching = [c for c in candidates if _source_matches_home(fallback, c[0], c[1], profile=profile)]
        if len(matching) == 1:
            candidates = matching
    if not candidates:
        raise MainSessionUnavailable("session: main needs one connected platform with home_channel configured")
    if len(candidates) != 1:
        names = ", ".join(sorted(str(getattr(p, "value", p)) for p, _, _ in candidates))
        raise MainSessionUnavailable(
            "session: main is ambiguous because multiple connected home channels are configured "
            f"({names}); keep one home channel or create the job from the intended home conversation")
    return candidates[0]


def _known_home_source(runner: Any, platform: Any, home: Any, profile: Optional[str], origin_key: str):
    """The exact source the home conversation was last seen with: the job's captured session key,
    then the newest live source, then the durable routing index (survives a gateway restart)."""
    live_sources = getattr(runner, "_session_sources", {}) or {}
    ordered = ([live_sources.get(origin_key)] if origin_key else []) + list(reversed(list(live_sources.values())))
    for source in ordered:
        if _source_matches_home(source, platform, home, profile=profile):
            return source
    store = getattr(runner, "session_store", None)
    if store is None:
        return None
    entries = store.list_sessions()  # newest first, lock-held
    if origin_key:
        entries = sorted(entries, key=lambda entry: getattr(entry, "session_key", "") != origin_key)
    for entry in entries:
        source = getattr(entry, "origin", None)
        if _source_matches_home(source, platform, home, profile=profile):
            return source
    return None


def _require_live(runner: Any) -> None:
    if runner is None or not getattr(runner, "_running", False):
        raise MainSessionUnavailable("session: main requires a running messaging gateway")
    if getattr(runner, "_draining", False) or getattr(runner, "_external_drain_active", False):
        raise MainSessionUnavailable("session: main is temporarily unavailable while the gateway is draining")


def resolve_main_session_source(runner: Any, *, origin: Any = None, profile: Optional[str] = None):
    """Resolve the one home conversation that defines ``main`` for *profile*.

    A configured ``home_channel`` is authoritative; the live/persisted source of that conversation is
    preferred so group, thread and user identity match the real inbound session key byte for byte.
    ``origin`` (a cron job's creating chat) disambiguates several homes and is the fallback when no
    home is configured. A DM home never seen yet is rebuilt from its chat id; a group or thread home
    never seen is refused rather than silently keyed into a different session.
    """
    _require_live(runner)
    fallback = _origin_source(origin, profile=profile)
    profile = profile or (fallback.profile if fallback is not None else None)
    candidates = _home_candidates(runner, profile)
    if not candidates and fallback is not None:
        adapters = runner._adapters_for_profile(profile) if hasattr(runner, "_adapters_for_profile") else {}
        if adapters.get(fallback.platform) is not None:
            return fallback
    platform, home, adapter = _pick_home(candidates, fallback, profile)
    origin_key = str(origin.get("session_key") or "").strip() if isinstance(origin, Mapping) else ""
    known = _known_home_source(runner, platform, home, profile, origin_key)
    if known is not None:
        return dataclasses.replace(known, profile=profile or known.profile)
    chat_id = str(home.chat_id)
    if home.thread_id or chat_id.startswith("group:"):
        raise MainSessionUnavailable(
            "session: main could not recover the exact group/thread home session; "
            "send one message in that home conversation first")
    source = adapter.build_source(chat_id=chat_id, chat_name=getattr(home, "name", None) or "Home",
                                  chat_type="dm", user_id=getattr(home, "user_id", None) or chat_id,
                                  user_name=getattr(home, "name", None) or None,
                                  scope_id=getattr(home, "scope_id", None))
    return dataclasses.replace(source, profile=profile or source.profile)


async def enqueue_main_session_turn(runner: Any, *, source: Any, text: str, producer: str,
                                    metadata: Optional[dict[str, Any]] = None) -> MainSessionEnqueueResult:
    """Commit one automation turn to the exact home conversation's FIFO (see module docstring).

    Raises ``MainSessionUnavailable`` (or the store's ``RuntimeStoreError`` / ``WakeNotAccepted``)
    when the turn was not accepted, so a caller can report the fire failed or let a provider retry."""
    from gateway.platforms.event import MessageEvent, MessageType
    from gateway.session_automation import admit_route_wake
    from gateway.wake import admit_internal_event

    prompt = str(text or "").strip()
    if not prompt:
        raise ValueError("A main-session turn requires non-empty text")
    if prompt.startswith("/"):
        raise ValueError("A main-session turn is never a slash command")
    _require_live(runner)
    adapter = runner._adapter_for_source(source)
    if adapter is None or not getattr(adapter, "is_connected", True):
        raise MainSessionUnavailable(f"session: main home adapter {source.platform.value} is not connected")
    session_key = runner._session_key_for_source(source)
    runner._cache_session_source(session_key, source)
    queued = (session_key in (getattr(adapter, "_active_sessions", {}) or {})
              or session_key in (getattr(runner, "_running_agents", {}) or {}))
    durable = bool(await admit_route_wake(runner, adapter, source, prompt, producer))
    if not durable:
        # No authority serves the route, or it holds no committed human turn yet: the adapter path
        # queues an internal event as its own FIFO turn (never merged into a waiting human message).
        event = MessageEvent(text=prompt, message_type=MessageType.TEXT, source=source, internal=True,
                             message_id=f"{producer}:{uuid.uuid4().hex}",
                             metadata={"main_session": True, **(metadata or {})})
        await admit_internal_event(adapter, event)
    logger.info("main session: %s turn accepted for %s (queued=%s durable=%s)",
                producer, session_key, queued, durable)
    return MainSessionEnqueueResult(session_key=session_key, platform=source.platform.value,
                                    chat_id=str(source.chat_id), queued=queued, durable=durable)
