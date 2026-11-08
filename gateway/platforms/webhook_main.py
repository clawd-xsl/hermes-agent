"""Webhook route policy for ``session: main`` and per-route runtime knobs.

Route keys (each overridable per route, platform-wide default under ``platforms.webhook.extra``):

- ``session``: ``isolated`` (default; the delivery runs in its own one-shot session) or ``main``
  (the event enters the gateway home conversation, ``gateway/main_session.py``).
- ``filter_before_main`` (default true): a main route first runs a one-shot reviewer — an ordinary
  durable webhook delivery with zero tools and read-only memory (``chat_type="webhook_review"``)
  whose destination is the main conversation. A bare ``NO_REPLY`` filters the event; anything else
  is a concise handoff that wakes main. ``false`` (trusted routes only) commits the rendered event
  straight into main.
- ``model`` / ``provider``: the isolated run's (or reviewer's) model. Delivery chat ids are one-shot,
  so each delivery inherits a stable route-scoped parent channel whose synthesized
  ``ChannelOverride`` reuses the gateway's normal channel model/credential resolution.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

REVIEW_CHAT_TYPE = "webhook_review"
MAIN_DESTINATION = "main_session"
MODEL_OVERRIDE_CHANNEL_PREFIX = "__webhook_model__:"
MAX_MAIN_HANDOFF_CHARS = 2_000


def coerce_bool(value: Any, *, default: bool) -> bool:
    """YAML/JSON booleans without treating the string "false" as true."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "on", "1"}:
            return True
        if normalized in {"false", "no", "off", "0"}:
            return False
        return default
    return bool(value)


def _clean(value: Any) -> Optional[str]:
    return (str(value).strip() or None) if value is not None else None


def _route_or_default(adapter, route: dict, key: str, default: Any) -> Any:
    value = route.get(key)
    return adapter.config.extra.get(key, default) if value is None or value == "" else value


def route_session_mode(adapter, route: dict) -> str:
    """The route's ``session`` over the platform default; raises ValueError on an unknown mode."""
    from gateway.main_session import normalize_session_mode
    return normalize_session_mode(_route_or_default(adapter, route, "session", None))


def route_reviews_before_main(adapter, route: dict) -> bool:
    return coerce_bool(_route_or_default(adapter, route, "filter_before_main", True), default=True)


def route_model_override(adapter, route: dict) -> tuple[Optional[str], Optional[str]]:
    extra = adapter.config.extra
    return (_clean(route.get("model")) or _clean(extra.get("model")),
            _clean(route.get("provider")) or _clean(extra.get("provider")))


def model_parent_id(adapter, route_name: str, route: Optional[dict]) -> Optional[str]:
    """The route-scoped parent channel a delivery inherits its model from, or None."""
    if not isinstance(route, dict) or not any(route_model_override(adapter, route)):
        return None
    return f"{MODEL_OVERRIDE_CHANNEL_PREFIX}{route_name}"


def sync_model_channel_overrides(adapter) -> None:
    """Publish one synthesized ``ChannelOverride`` per route with a model/provider override (and drop
    the ones a removed route left). An explicit ``channel_overrides`` entry under the reserved id wins."""
    from gateway.config import ChannelOverride
    overrides = adapter.config.channel_overrides
    managed = adapter.__dict__.setdefault("_managed_model_channels", set())
    for channel_id in managed:
        overrides.pop(channel_id, None)
    managed.clear()
    for route_name, route in adapter._routes.items():
        channel_id = model_parent_id(adapter, str(route_name), route)
        if channel_id is None:
            continue
        if channel_id in overrides:
            logger.warning("[webhook] Reserved channel override %s already exists; keeping the explicit "
                           "channel_overrides entry", channel_id)
            continue
        model, provider = route_model_override(adapter, route)
        overrides[channel_id] = ChannelOverride(model=model, provider=provider)
        managed.add(channel_id)


def build_review_prompt(*, route_name: str, event_type: str, rendered_event: str) -> str:
    """The reviewer's turn (the system prompt is untouched). The output contract is restated after the
    event body, closest to generation: reviewers that justified a skip ("routine. NO_REPLY") read as
    handoffs and woke main for nothing."""
    return (
        "Review this authenticated external webhook event and decide whether the main "
        "personal-assistant conversation should be interrupted.\n\n"
        "If the event is routine, duplicate, irrelevant, or needs no user attention, reply with "
        "exactly NO_REPLY and nothing else.\n\n"
        "If it matters, reply only with a concise, self-contained handoff for the main assistant: "
        "identify the source, what changed, why it matters, and any likely next action. Do not "
        "address the user, send messages, or mutate external state in this review turn.\n\n"
        f"Route: {route_name}\nEvent type: {event_type}\n\n"
        f"<external_event>\n{rendered_event}\n</external_event>\n\n"
        "Output format (your reply is parsed by code, not read by a person):\n"
        "- Skip: the entire reply is the single token NO_REPLY. Do not explain or justify the "
        "decision; no words before or after it, not even on the same line.\n"
        "- Interrupt: the entire reply is the handoff, and it never contains NO_REPLY.\n"
        '- Correct skip reply: "NO_REPLY"\n'
        '- Wrong skip replies, which are treated as handoffs and wrongly interrupt the user: '
        '"This is routine, no action needed. NO_REPLY" and "Sent by the owner.\\n\\nNO_REPLY".'
    )


def main_event_text(route_name: str, event_type: str, body: str) -> str:
    return f"[External event: {route_name} / {event_type}]\n\n{body}"


async def enqueue_main(adapter, *, profile: Optional[str], text: str):
    """Resolve the routed profile's home conversation and commit *text* there; raises when main is
    unavailable (the caller answers 503 or fails the reviewer's handoff)."""
    import asyncio
    from gateway.main_session import WEBHOOK_PRODUCER, enqueue_main_session_turn, resolve_main_session_source
    runner = adapter.gateway_runner
    with adapter._profile_scope(profile):
        source = await asyncio.to_thread(resolve_main_session_source, runner,
                                         profile=profile if isinstance(profile, str) else None)
        return await enqueue_main_session_turn(runner, source=source, text=text, producer=WEBHOOK_PRODUCER,
                                               metadata={"trigger": "webhook"})


async def handoff_to_main(adapter, chat_id: str, content: str, delivery: dict, metadata: Optional[dict]):
    """``send()`` for a reviewer delivery: only its final (notify-marked) reply is a handoff; interim
    status sends are acknowledged and dropped. A ``NO_REPLY`` never gets here (``send`` filters
    silence first). A failed handoff fails the send."""
    from gateway.platforms.base import SendResult
    if not (metadata or {}).get("notify"):
        logger.debug("[webhook] Reviewer %s: non-final send not handed to main", chat_id)
        return SendResult(success=True)
    route = str(delivery.get("route") or "")
    event_type = str((delivery.get("deliver_extra") or {}).get("event_type") or "unknown")
    handoff = str(content or "").strip()
    if len(handoff) > MAX_MAIN_HANDOFF_CHARS:
        logger.warning("[webhook] Truncating oversized reviewer handoff route=%s chars=%d", route, len(handoff))
        handoff = handoff[: MAX_MAIN_HANDOFF_CHARS - 1].rstrip() + "…"
    text = main_event_text(route, event_type, handoff)
    skills = (adapter._routes.get(route) or {}).get("skills") or []
    if skills:
        text = adapter._apply_skills(text, skills)
    try:
        accepted = await enqueue_main(adapter, profile=delivery.get("profile"), text=text)
    except Exception as exc:
        logger.exception("[webhook] Reviewed main-session handoff failed route=%s chat=%s", route, chat_id)
        return SendResult(success=False, error=f"Main conversation unavailable: {exc}")
    logger.info("[webhook] Main-session review admitted route=%s session=%s queued=%s", route,
                accepted.session_key, accepted.queued)
    return SendResult(success=True)


def release_one_shot_session(authority, receipt) -> None:
    """After a delivery's admission settled and its session was ended: release what a turn runtime
    pooled for the one-shot session (both sides of a mid-run compression rotation)."""
    from agent.turn_runtime_registry import release_runtime_sessions
    sid = receipt.ref.session_id
    tip = None
    try:
        tip = authority.db.get_compression_tip(sid)
    except Exception:
        logger.debug("[webhook] Compression tip lookup failed for %s", sid, exc_info=True)
    release_runtime_sessions(tip, sid)
