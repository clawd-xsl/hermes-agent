"""Route internal work into the gateway's real home conversation.

``session: main`` is deliberately not a second agent runtime.  It resolves the
configured home conversation, creates an internal ``MessageEvent`` for that
exact session key, and uses the gateway's existing per-session FIFO.  The turn
therefore reuses the live agent, full transcript, provider session, prompt
cache, and normal delivery path.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource


SESSION_ISOLATED = "isolated"
SESSION_MAIN = "main"
SESSION_MODES = frozenset({SESSION_ISOLATED, SESSION_MAIN})


class MainSessionUnavailable(RuntimeError):
    """Raised when a real, live main conversation cannot be resolved."""


@dataclass(frozen=True)
class MainSessionEnqueueResult:
    """Identity of a turn accepted by the main-session FIFO."""

    session_key: str
    platform: str
    chat_id: str
    queued: bool


def normalize_session_mode(value: Any, *, default: str = SESSION_ISOLATED) -> str:
    """Normalize a user-facing ``main|isolated`` value or raise clearly."""
    raw = default if value is None or str(value).strip() == "" else value
    mode = str(raw).strip().lower()
    if mode not in SESSION_MODES:
        raise ValueError(
            f"Invalid session mode {value!r}; expected one of: "
            f"{', '.join(sorted(SESSION_MODES))}"
        )
    return mode


def _origin_source(origin: Any, *, profile: Optional[str] = None) -> Optional[SessionSource]:
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
    thread_id = str(origin.get("thread_id") or "").strip() or None
    chat_type = str(origin.get("chat_type") or "").strip().lower()
    if not chat_type:
        chat_type = "group" if chat_id.startswith("group:") else "dm"
    origin_profile = str(origin.get("profile") or "").strip() or None
    return SessionSource(
        platform=platform,
        chat_id=chat_id,
        chat_name=str(origin.get("chat_name") or "").strip() or None,
        chat_type=chat_type,
        user_id=str(origin.get("user_id") or chat_id).strip() or None,
        user_name=str(origin.get("user_name") or "").strip() or None,
        thread_id=thread_id,
        scope_id=(
            str(origin.get("scope_id") or origin.get("guild_id") or "").strip()
            or None
        ),
        profile=profile or origin_profile,
    )


def _adapter_map_for_profile(runner: Any, profile: Optional[str]) -> Mapping[Any, Any]:
    if profile and profile not in {"default", "main"}:
        profile_adapters = getattr(runner, "_profile_adapters", {}) or {}
        adapters = profile_adapters.get(profile)
        if adapters:
            return adapters
    return getattr(runner, "adapters", {}) or {}


def _normalized_profile(value: Any) -> str:
    raw = str(value or "").strip()
    return "default" if raw in {"", "default", "main"} else raw


def _source_matches_home(
    source: Any,
    platform: Platform,
    home: Any,
    *,
    profile: Optional[str] = None,
) -> bool:
    return bool(
        source
        and getattr(source, "platform", None) == platform
        and str(getattr(source, "chat_id", "")) == str(home.chat_id)
        and (str(getattr(source, "thread_id", "") or "") == str(home.thread_id or ""))
        and _normalized_profile(getattr(source, "profile", None))
        == _normalized_profile(profile)
    )


def resolve_main_session_source(
    runner: Any,
    *,
    origin: Any = None,
    profile: Optional[str] = None,
) -> SessionSource:
    """Resolve the one live home conversation that defines ``main``.

    A configured ``home_channel`` is authoritative.  The stored gateway
    routing index is consulted first so group/thread/user identity matches the
    real inbound session key byte-for-byte, including after a gateway restart.
    ``origin`` is a compatibility fallback for jobs created in a live chat
    before a home channel was configured.
    """
    if runner is None or not getattr(runner, "_running", False):
        raise MainSessionUnavailable(
            "session: main requires a running messaging gateway"
        )

    adapters = _adapter_map_for_profile(runner, profile)
    candidates: list[tuple[Platform, Any, Any]] = []
    runner_config = getattr(runner, "config", None)
    primary_configs = getattr(runner_config, "platforms", {}) or {}
    for platform, adapter in adapters.items():
        # Secondary multiplexed profiles own separately-loaded adapter configs;
        # runner.config is only the active profile.  Reading home_channel from
        # the adapter keeps `session: main` inside the routed profile.
        platform_config = getattr(adapter, "config", None)
        if platform_config is None:
            platform_config = primary_configs.get(platform)
        home = getattr(platform_config, "home_channel", None)
        if (
            getattr(platform_config, "enabled", False)
            and home is not None
            and str(getattr(home, "chat_id", "") or "").strip()
            and adapter is not None
        ):
            candidates.append((platform, home, adapter))

    fallback = _origin_source(origin, profile=profile)
    resolved_profile = profile or (fallback.profile if fallback is not None else None)
    origin_session_key = (
        str(origin.get("session_key") or "").strip()
        if isinstance(origin, Mapping)
        else ""
    )
    if len(candidates) > 1 and fallback is not None:
        matching = [
            candidate
            for candidate in candidates
            if _source_matches_home(
                fallback,
                candidate[0],
                candidate[1],
                profile=resolved_profile,
            )
        ]
        if len(matching) == 1:
            candidates = matching

    if not candidates:
        if fallback is not None and adapters.get(fallback.platform) is not None:
            return fallback
        raise MainSessionUnavailable(
            "session: main needs one connected platform with home_channel configured"
        )
    if len(candidates) != 1:
        names = ", ".join(sorted(platform.value for platform, _, _ in candidates))
        raise MainSessionUnavailable(
            "session: main is ambiguous because multiple connected home channels "
            f"are configured ({names}); keep one home channel or create the job "
            "from the intended home conversation"
        )

    platform, home, adapter = candidates[0]

    # First prefer the exact source captured when the job was created.  The
    # durable session key disambiguates same-chat profile/scope variants that
    # cannot safely be reconstructed from platform + chat_id alone.
    live_sources = getattr(runner, "_session_sources", {}) or {}
    if origin_session_key:
        exact = live_sources.get(origin_session_key)
        if _source_matches_home(exact, platform, home, profile=resolved_profile):
            return dataclasses.replace(
                exact,
                profile=resolved_profile or exact.profile,
            )

    # Otherwise prefer the newest matching source observed by this live runner.
    for source in reversed(list(live_sources.values())):
        if _source_matches_home(source, platform, home, profile=resolved_profile):
            return dataclasses.replace(source, profile=resolved_profile or source.profile)

    # Then use the durable gateway routing index. list_sessions() is public,
    # lock-held, and newest-first, so it is safe across cron worker threads.
    store = getattr(runner, "session_store", None)
    if store is not None:
        try:
            entries = store.list_sessions()
            if origin_session_key:
                entries = sorted(
                    entries,
                    key=lambda entry: getattr(entry, "session_key", "")
                    != origin_session_key,
                )
            for entry in entries:
                source = getattr(entry, "origin", None)
                if _source_matches_home(
                    source,
                    platform,
                    home,
                    profile=resolved_profile,
                ):
                    return dataclasses.replace(
                        source,
                        profile=resolved_profile or source.profile,
                    )
        except Exception:
            pass

    # A brand-new gateway may have a home configured before that conversation
    # has ever received a turn.  DMs (the personal-assistant case) can be
    # reconstructed exactly from chat_id; group/thread homes should normally
    # have a persisted source and are rejected to avoid silently creating a
    # different session key.
    chat_id = str(home.chat_id)
    thread_id = str(home.thread_id) if home.thread_id else None
    if thread_id or chat_id.startswith("group:"):
        raise MainSessionUnavailable(
            "session: main could not recover the exact group/thread home session; "
            "send one message in that home conversation first"
        )
    source = adapter.build_source(
        chat_id=chat_id,
        chat_name=getattr(home, "name", None) or "Home",
        chat_type="dm",
        user_id=chat_id,
        user_name=getattr(home, "name", None) or None,
    )
    return dataclasses.replace(source, profile=resolved_profile or source.profile)


async def enqueue_main_session_turn(
    runner: Any,
    *,
    source: SessionSource,
    text: str,
    event_id: str,
    raw_message: Any = None,
    metadata: Optional[dict[str, Any]] = None,
) -> MainSessionEnqueueResult:
    """Accept one internal turn into the exact main-session FIFO."""
    prompt = str(text or "").strip()
    if not prompt:
        raise ValueError("A main-session turn requires non-empty text")
    if runner is None or not getattr(runner, "_running", False):
        raise MainSessionUnavailable(
            "session: main requires a running messaging gateway"
        )
    if getattr(runner, "_draining", False) or getattr(
        runner, "_external_drain_active", False
    ):
        raise MainSessionUnavailable(
            "session: main is temporarily unavailable while the gateway is draining"
        )

    adapter = runner._adapter_for_source(source)
    if adapter is None or not getattr(adapter, "is_connected", True):
        raise MainSessionUnavailable(
            f"session: main home adapter {source.platform.value} is not connected"
        )

    event = MessageEvent(
        text=prompt,
        message_type=MessageType.TEXT,
        source=source,
        raw_message=raw_message,
        message_id=str(event_id),
        internal=True,
        metadata={"main_session": True, **(metadata or {})},
    )
    session_key = runner._session_key_for_source(source)
    runner._cache_session_source(session_key, source)

    adapter_active = session_key in (getattr(adapter, "_active_sessions", {}) or {})
    runner_active = session_key in (getattr(runner, "_running_agents", {}) or {})
    queued = adapter_active or runner_active
    if queued:
        # The explicit FIFO keeps simultaneous cron/webhook events as separate
        # user turns.  BasePlatformAdapter's generic busy path intentionally
        # merges ordinary text bursts, which is wrong for scheduled events.
        runner._enqueue_fifo(session_key, event, adapter)
    else:
        await adapter.handle_message(event)

    return MainSessionEnqueueResult(
        session_key=session_key,
        platform=source.platform.value,
        chat_id=str(source.chat_id),
        queued=queued,
    )
