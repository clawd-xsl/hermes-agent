"""Signal plugin registration: configured/validate checks and ``register``."""

import logging

from gateway.config import PlatformConfig
from gateway.platforms.send_targets import NOT_EXPLICIT, StandaloneSendPolicy

# Absolute imports (see __init__.py): one canonical module per file.
from plugins.platforms.signal.adapter import (
    MAX_MESSAGE_LENGTH, SignalAdapter, check_signal_requirements, validate_signal_config)
from plugins.platforms.signal.setup import interactive_setup
from plugins.platforms.signal.setup import read_signal_yaml_config
from plugins.platforms.signal.standalone import standalone_send

logger = logging.getLogger(__name__)


def _is_connected(config: PlatformConfig) -> bool:
    """The given config, else the profile's config.yaml: setup/status callers pass an empty
    PlatformConfig, and the runtime's settings live only in YAML (no env credentials)."""
    if validate_signal_config(config):
        return True
    try:
        return validate_signal_config(PlatformConfig(enabled=True, extra=read_signal_yaml_config()))
    except Exception:
        return False


def _validate_config(config: PlatformConfig) -> bool:
    """``validate_config``: name the missing signal-ts settings when the adapter cannot start."""
    if validate_signal_config(config):
        return True
    logger.warning("Signal: configure an existing state_path and sdk_path (and node_command when not on PATH) "
                   "under platforms.signal.extra")
    return False


# System-prompt platform hint (PlatformEntry.platform_hint); cached-prompt bytes, keep stable.
_PLATFORM_HINT = (
    "You are on Signal. Standard markdown (**bold**, *italic*, ~~strike~~, # headers, `code`) auto-converts to "
    "Signal formatting; bullets render as \u2022. No tables \u2014 use bullets or labeled lines. "
    "You can send files natively: write MEDIA:/absolute/path/to/file in your response. "
    "Images (.png, .jpg, .webp) send as photos, other files as documents; ![alt](url) sends as photos."
)
def _parse_send_target(ref):
    # "group:<id>" is a native group target; an empty id is not explicit.
    stripped = ref.strip()
    if not stripped.startswith("group:"):
        return None
    group_id = stripped[len("group:"):].strip()
    return (f"group:{group_id}", None) if group_id else NOT_EXPLICIT


def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system."""
    from . import gateway_env  # env -> GatewayConfig bridge (gateway.config_env)
    ctx.register_platform(
        target_ref_syntax_fn=_parse_send_target,
        # Every send_message delivery goes through standalone_send, which hands it to the LIVE
        # adapter (one signal-ts client per linked device); attachments ride the final chunk.
        standalone_send=StandaloneSendPolicy(label="Signal", media=True, empty_list_between_chunks=True,
                                             text_direct=True),
        name="signal", label="Signal", adapter_factory=SignalAdapter,
        dashboard_description=("Runs a persistent signal-ts linked device inside the gateway (no signal-cli "
                               "daemon). Configure it with `hermes gateway setup`."),
        docs_url="https://hermes-agent.nousresearch.com/docs/user-guide/messaging/signal",
        dashboard_env_vars=(), dashboard_required_env=(),
        check_fn=check_signal_requirements, validate_config=_validate_config,
        is_connected=_is_connected, required_env=[],
        config_connected_fn=validate_signal_config,
        install_hint="Signal: signal-ts runtime not configured", setup_fn=interactive_setup,
        cron_deliver_env_var="SIGNAL_HOME_CHANNEL",
        # send_message splits at the adapter's limit through this field.
        standalone_sender_fn=standalone_send, max_message_length=MAX_MESSAGE_LENGTH,
        emoji="📡", pii_safe=True, allow_update_command=True,
        platform_hint=_PLATFORM_HINT,
        display_tier="low",
        env_overrides=gateway_env.ENV_OVERRIDES,
        env_enable_credentials=gateway_env.ENV_ENABLE_CREDENTIALS, **gateway_env.AUTHZ,
    )
