"""Tests for Signal media delivery in send_message_tool.py."""

import asyncio
from unittest.mock import MagicMock, AsyncMock, patch

from gateway.config import Platform
from gateway.platforms.base import SendResult


def _live_signal_adapter():
    adapter = MagicMock()
    adapter.send = AsyncMock(return_value=SendResult(success=True))
    adapter.send_files = AsyncMock(return_value=SendResult(success=True))
    runner = MagicMock()
    runner.adapters = {Platform.SIGNAL: adapter}
    return runner, adapter


class TestSendSignalMediaFiles:
    """Test that _send_signal correctly handles media_files parameter."""

    def test_send_signal_basic_text_without_media(self):
        """Backward compatibility: text-only signal messages work."""
        from tools.send_message_tool import _send_signal

        runner, adapter = _live_signal_adapter()
        with patch("gateway.run._gateway_runner_ref", return_value=runner):
            result = asyncio.run(_send_signal({}, "+155****9999", "Hello world"))

        assert result["success"] is True
        assert result["platform"] == "signal"
        assert result["chat_id"] == "+155****9999"
        adapter.send.assert_awaited_once_with(
            chat_id="+155****9999", content="Hello world"
        )

    def test_send_signal_with_attachments(self, tmp_path):
        """Signal messages with media_files include attachments in JSON-RPC."""
        from tools.send_message_tool import _send_signal

        img_path = tmp_path / "test.png"
        img_path.write_bytes(b"\x89PNG")

        runner, adapter = _live_signal_adapter()
        with patch("gateway.run._gateway_runner_ref", return_value=runner):
            result = asyncio.run(
                _send_signal(
                    {},
                    "+155****9999",
                    "Check this out",
                    media_files=[(str(img_path), False)],
                )
            )

        assert result["success"] is True
        assert result["platform"] == "signal"
        adapter.send_files.assert_awaited_once_with(
            "+155****9999", [str(img_path)], caption="Check this out"
        )

    def test_send_signal_with_missing_media_file(self):
        """Missing media files should generate warnings but not fail."""
        from tools.send_message_tool import _send_signal

        runner, adapter = _live_signal_adapter()
        with patch("gateway.run._gateway_runner_ref", return_value=runner):
            result = asyncio.run(
                _send_signal(
                    {},
                    "+155****9999",
                    "File missing?",
                    media_files=[("/nonexistent.png", False)],
                )
            )

        assert result["success"] is True  # Should succeed despite missing file
        assert "warnings" in result
        assert "Some media files were skipped" in str(result["warnings"])
        adapter.send.assert_awaited_once()

    def test_send_signal_without_live_gateway_fails_cleanly(self):
        from tools.send_message_tool import _send_signal

        with patch("gateway.run._gateway_runner_ref", return_value=None):
            result = asyncio.run(_send_signal({}, "+155****9999", "Hello"))

        assert "error" in result
        assert "persistent signal-ts adapter" in result["error"]


class TestSendSignalMediaRestrictions:
    """Test that the restriction block handles Signal media correctly."""

    def test_signal_allows_text_only_media_via_send_to_platform(self):
        """Signal should accept text-only media files (no message) via _send_to_platform."""
        from tools.send_message_tool import _send_to_platform

        mock_result = {"success": True, "platform": "signal"}
        with patch("tools.send_message_tool._send_signal", new=AsyncMock(return_value=mock_result)):
            config = MagicMock()
            config.platforms = {Platform.SIGNAL: MagicMock(enabled=True)}
            config.get_home_channel.return_value = None

            result = asyncio.run(
                _send_to_platform(
                    Platform.SIGNAL,
                    config,
                    "+155****9999",
                    "",  # Empty message - media is the message
                    media_files=[("/tmp/test.png", False)]
                )
            )

            assert result["success"] is True

    def test_non_media_platforms_reject_text_only_media(self):
        """Mattermost should reject text-only media (no MESSAGE content)."""
        from tools.send_message_tool import _send_to_platform

        config = MagicMock()
        config.platforms = {Platform.MATTERMOST: MagicMock(enabled=True)}
        config.get_home_channel.return_value = None

        # Empty message with media_files should trigger restriction block
        result = asyncio.run(
            _send_to_platform(
                Platform.MATTERMOST,
                config,
                "channel-id",
                "",  # Empty message - media is the only content
                media_files=[("/tmp/test.png", False)]
            )
        )

        assert "error" in result
        assert "only supported for" in result["error"]


class TestSendSignalMediaWarningMessages:
    """Test warning messages are updated to include signal."""

    def test_warning_includes_signal_when_media_omitted(self):
        """Non-media platforms should show a warning mentioning signal in the supported list."""
        from tools.send_message_tool import _send_to_platform
        from hermes_cli.plugins import discover_plugins
        from gateway.platform_registry import platform_registry

        config = MagicMock()
        config.platforms = {Platform.MATTERMOST: MagicMock(enabled=True)}
        config.get_home_channel.return_value = None

        # Patch a currently non-media platform's standalone sender so delivery
        # succeeds and the media-omitted warning can be asserted independently.
        discover_plugins()
        mattermost_entry = platform_registry.get("mattermost")
        original_sender = mattermost_entry.standalone_sender_fn
        mattermost_entry.standalone_sender_fn = AsyncMock(return_value={"success": True})
        try:
            result = asyncio.run(
                _send_to_platform(
                    Platform.MATTERMOST,
                    config,
                    "channel-id",
                    "Test message with media",
                    media_files=[("/tmp/test.png", False)]
                )
            )
        finally:
            mattermost_entry.standalone_sender_fn = original_sender

        assert result.get("warnings") is not None
        # Check that the warning mentions signal as supported
        found = any("signal" in w.lower() for w in result["warnings"])
        assert found, f"Expected 'signal' in warnings but got: {result.get('warnings')}"


class TestSendSignalGroupChats:
    """Test that _send_signal handles group chats correctly."""

    def test_send_signal_group_with_attachments(self, tmp_path):
        """Group chat messages with attachments should use groupId parameter."""
        from tools.send_message_tool import _send_signal

        img_path = tmp_path / "test_attachment.pdf"
        img_path.write_bytes(b"%PDF-1.4")

        runner, adapter = _live_signal_adapter()
        with patch("gateway.run._gateway_runner_ref", return_value=runner):
            result = asyncio.run(
                _send_signal(
                    {},
                    "group:abc123==",
                    "Group file",
                    media_files=[(str(img_path), False)],
                )
            )

        assert result["success"] is True
        adapter.send_files.assert_awaited_once_with(
            "group:abc123==", [str(img_path)], caption="Group file"
        )


class TestSendSignalConfigLoading:
    """Verify Signal config loading works."""

    def test_signal_platform_exists(self):
        """Platform.SIGNAL should be a valid platform."""
        assert hasattr(Platform, "SIGNAL")
        assert Platform.SIGNAL.value == "signal"
