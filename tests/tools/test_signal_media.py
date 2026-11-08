"""Signal send_message delivery: every send goes through the live gateway adapter (one signal-ts
client per linked device), never a second out-of-process client."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from gateway.config import Platform
from gateway.platforms.base import SendResult


def _live_runner():
    adapter = MagicMock()
    adapter.send = AsyncMock(return_value=SendResult(success=True))
    adapter.send_files = AsyncMock(return_value=SendResult(success=True))
    runner = SimpleNamespace(adapters={Platform.SIGNAL: adapter}, _gateway_loop=None)
    return runner, adapter


def _send(*args, runner, **kwargs):
    from plugins.platforms.signal.standalone import _send_signal
    with patch("gateway.run._gateway_runner_ref", return_value=runner):
        return asyncio.run(_send_signal(*args, **kwargs))


class TestSendSignalViaLiveAdapter:
    def test_text_uses_adapter_send(self):
        runner, adapter = _live_runner()
        result = _send("+155****9999", "Hello world", runner=runner)
        assert result["success"] is True and result["platform"] == "signal"
        assert result["chat_id"] == "+155****9999"
        adapter.send.assert_awaited_once_with(chat_id="+155****9999", content="Hello world")
        adapter.send_files.assert_not_awaited()

    def test_media_uses_send_files_with_caption(self, tmp_path):
        img = tmp_path / "test.png"
        img.write_bytes(b"\x89PNG")
        runner, adapter = _live_runner()
        result = _send("+155****9999", "Check this out", media_files=[(str(img), False)], runner=runner)
        assert result["success"] is True
        adapter.send_files.assert_awaited_once_with("+155****9999", [str(img)], caption="Check this out")

    def test_missing_media_file_warns_but_sends_text(self):
        runner, adapter = _live_runner()
        result = _send("+155****9999", "File missing?", media_files=[("/nonexistent.png", False)], runner=runner)
        assert result["success"] is True
        assert "Some media files were skipped" in str(result["warnings"])
        adapter.send.assert_awaited_once()

    def test_only_missing_media_and_no_text_fails(self):
        runner, _ = _live_runner()
        result = _send("+155****9999", "", media_files=[("/nonexistent.png", False)], runner=runner)
        assert "no valid media" in result["error"]

    def test_adapter_failure_is_reported(self):
        runner, adapter = _live_runner()
        adapter.send = AsyncMock(return_value=SendResult(success=False, error="RPC send failed"))
        result = _send("group:abc", "hi", runner=runner)
        assert "RPC send failed" in result["error"]

    def test_group_chat_id_is_redacted_in_result(self):
        runner, _ = _live_runner()
        assert _send("group:abc", "hi", runner=runner)["chat_id"] == "group:***"

    def test_without_live_gateway_fails_cleanly(self):
        result = _send("+155****9999", "Hello", runner=None)
        assert "running gateway" in result["error"]


class TestSignalSendToPlatformRouting:
    def test_media_only_send_reaches_the_live_adapter(self, tmp_path):
        from tools.send_message_tool import _send_to_platform

        img = tmp_path / "test.png"
        img.write_bytes(b"\x89PNG")
        runner, adapter = _live_runner()
        with patch("gateway.run._gateway_runner_ref", return_value=runner):
            result = asyncio.run(_send_to_platform(
                Platform.SIGNAL, MagicMock(extra={}), "+155****9999", "", media_files=[(str(img), False)]))
        assert result["success"] is True
        adapter.send_files.assert_awaited_once()

    def test_text_send_reaches_the_live_adapter(self):
        from tools.send_message_tool import _send_to_platform

        runner, adapter = _live_runner()
        with patch("gateway.run._gateway_runner_ref", return_value=runner):
            result = asyncio.run(_send_to_platform(Platform.SIGNAL, MagicMock(extra={}), "+155****9999", "hello"))
        assert result["success"] is True
        adapter.send.assert_awaited_once_with(chat_id="+155****9999", content="hello")

    def test_cron_without_gateway_gets_a_clear_error(self):
        from tools.send_message_tool import _send_to_platform

        with patch("gateway.run._gateway_runner_ref", return_value=None):
            result = asyncio.run(_send_to_platform(Platform.SIGNAL, MagicMock(extra={}), "+155****9999", "hello"))
        assert "running gateway" in result["error"]
