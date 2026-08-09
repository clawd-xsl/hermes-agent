"""Tests for the startup allowlist warning check in gateway/run.py."""

import os
from unittest.mock import patch

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.run import _config_has_user_allowlist


def _would_warn(config=None):
    """Replicate the startup allowlist warning logic. Returns True if warning fires."""
    _any_allowlist = any(
        os.getenv(v)
        for v in ("TELEGRAM_ALLOWED_USERS", "DISCORD_ALLOWED_USERS",
                   "WHATSAPP_ALLOWED_USERS", "SLACK_ALLOWED_USERS",
                   "EMAIL_ALLOWED_USERS",
                   "SMS_ALLOWED_USERS", "MATTERMOST_ALLOWED_USERS",
                   "MATRIX_ALLOWED_USERS", "DINGTALK_ALLOWED_USERS", "FEISHU_ALLOWED_USERS", "WECOM_ALLOWED_USERS",
                   "GATEWAY_ALLOWED_USERS")
    )
    _any_allowlist = _any_allowlist or _config_has_user_allowlist(config)
    _allow_all = os.getenv("GATEWAY_ALLOW_ALL_USERS", "").lower() in {"true", "1", "yes"} or any(
        os.getenv(v, "").lower() in {"true", "1", "yes"}
        for v in ("TELEGRAM_ALLOW_ALL_USERS", "DISCORD_ALLOW_ALL_USERS",
                   "WHATSAPP_ALLOW_ALL_USERS", "SLACK_ALLOW_ALL_USERS",
                   "EMAIL_ALLOW_ALL_USERS",
                   "SMS_ALLOW_ALL_USERS", "MATTERMOST_ALLOW_ALL_USERS",
                   "MATRIX_ALLOW_ALL_USERS", "DINGTALK_ALLOW_ALL_USERS", "FEISHU_ALLOW_ALL_USERS", "WECOM_ALLOW_ALL_USERS")
    )
    return not _any_allowlist and not _allow_all


class TestAllowlistStartupCheck:

    def test_no_config_emits_warning(self):
        with patch.dict(os.environ, {}, clear=True):
            assert _would_warn() is True

    def test_signal_config_allowlist_suppresses_warning(self):
        config = GatewayConfig(
            platforms={
                Platform.SIGNAL: PlatformConfig(
                    enabled=True,
                    extra={"allow_from": ["+15551234567"]},
                )
            }
        )
        assert _would_warn(config) is False
