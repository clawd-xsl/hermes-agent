"""Aleph plugin registration: env enablement, home channel and ``register``."""

from __future__ import annotations

import logging
from typing import Any, Optional

from gateway.config import PlatformConfig
from gateway.platforms._shared import get_scoped_secret

from plugins.platforms.aleph import cli as _cli
from plugins.platforms.aleph.adapter import (
    CHAT_ID, CHAT_NAME, MAX_MESSAGE_LENGTH, OWNER_ID, AlephAdapter, check_aleph_requirements)
from plugins.platforms.aleph.standalone import standalone_send
from plugins.platforms.aleph.tools import register_tools

logger = logging.getLogger(__name__)

_HOME_OFF = {"off", "none", "false", "0", "no"}
_TRUE = {"1", "true", "yes", "on"}


def _env(name: str) -> str:
    return str(get_scoped_secret(name, "") or "").strip()


def _env_enablement() -> Optional[dict[str, Any]]:
    """``env_enablement_fn``: seed ``extra`` (and the home channel) from env before the adapter is
    built. ``None`` unless ``ALEPH_ENABLED`` is true (devices pair; there is no shared secret)."""
    if not _enabled_by_env():
        return None
    seed: dict[str, Any] = {"allow_from": [OWNER_ID]}
    for env, key in (("ALEPH_HOST", "host"), ("ALEPH_PORT", "port"), ("ALEPH_PUBLIC_URL", "public_url"),
                     ("ALEPH_HISTORY_LIMIT", "history_limit"), ("ALEPH_STATE_DIR", "state_dir")):
        if value := _env(env):
            seed[key] = value
    home = _env("ALEPH_HOME_CHANNEL") or CHAT_ID
    if home.lower() not in _HOME_OFF:
        seed["home_channel"] = {"chat_id": home, "name": _env("ALEPH_HOME_CHANNEL_NAME") or CHAT_NAME}
    return seed


def _enabled_by_env() -> bool:
    return _env("ALEPH_ENABLED").lower() in _TRUE


def _is_connected(config: PlatformConfig) -> bool:
    """Setup / env-enablement probe: ``ALEPH_ENABLED`` (``platforms.aleph.enabled`` in config.yaml
    enables it without this probe). The probe hands in a transient ``enabled=True`` view, so the
    config's own flag says nothing here."""
    return _enabled_by_env()


def _config_connected(config: PlatformConfig) -> bool:
    """``get_connected_platforms``: an enabled Aleph needs nothing else (no shared secret)."""
    return bool(getattr(config, "enabled", False))


def _validate_config(config: PlatformConfig) -> bool:
    return True


# System-prompt platform hint (PlatformEntry.platform_hint); cached-prompt bytes, keep stable.
_PLATFORM_HINT = (
    "You are on Aleph, the person's own iOS app for talking to you. It renders full Markdown "
    "(headings, lists, tables, code blocks, links) and shows your reply as it is written. "
    "You can send files: write MEDIA:/absolute/path/to/file in your response; images, video, "
    "audio and documents arrive as attachments the person can open. "
    "Aleph's own tools are loaded on demand (call them through tool_call): aleph_set_profile changes "
    "your name and avatar there, and only you can change them, so when the person asks for a new name "
    "or picture use it, not Hermes's config (aleph_get_profile reads them); aleph_react puts an emoji "
    "on the person's message, as they would tap back."
)


def register(ctx) -> None:
    """Plugin entry point, called by the Hermes plugin system."""
    from gateway.platforms.send_targets import StandaloneSendPolicy
    ctx.register_platform(
        name="aleph", label="Aleph", adapter_factory=AlephAdapter,
        dashboard_description="The Aleph iOS app: end-to-end encrypted, paired by QR code, with push.",
        check_fn=check_aleph_requirements, validate_config=_validate_config, is_connected=_is_connected,
        config_connected_fn=_config_connected,
        required_env=["ALEPH_ENABLED"], install_hint="Aleph needs aiohttp (bundled with the gateway)",
        env_enablement_fn=_env_enablement, cron_deliver_env_var="ALEPH_HOME_CHANNEL",
        # Every send_message delivery goes through the live adapter (it owns the event log).
        standalone_send=StandaloneSendPolicy(label="Aleph", media=True, text_direct=True),
        standalone_sender_fn=standalone_send, max_message_length=MAX_MESSAGE_LENGTH,
        emoji="ℵ", pii_safe=False, allow_update_command=True, platform_hint=_PLATFORM_HINT,
        # Drafts stream whenever the global ``streaming.enabled`` switch is on. Tool activity is not
        # part of the conversation: no progress rows and no "still working" notices; the status line
        # and activity history say it instead.
        display_tier="high", display_overrides={"streaming": True, "tool_progress": "off", "live_status": "full",
                                                "long_running_notifications": False},
    )
    register_tools(ctx)
    ctx.register_cli_command(
        name="aleph", help="Pair and manage Aleph devices",
        description="Pair the Aleph iOS app with this agent, list paired devices, revoke one.",
        setup_fn=_cli.register_cli, handler_fn=_cli.dispatch)
