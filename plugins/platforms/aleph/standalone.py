"""send_message / cron delivery for Aleph (the plugin's ``standalone_sender_fn``).

The event log and its ``seq`` belong to the running gateway's adapter, so an out-of-band send is
handed to that adapter on the gateway's loop; without a live gateway it fails with an explicit
error rather than writing a log another process owns."""

from __future__ import annotations

import logging
import os

from tools.send_message_senders import _error, _live_adapter, _success

logger = logging.getLogger(__name__)

NO_LIVE_GATEWAY_ERROR = ("Aleph delivery needs the running gateway: its adapter owns the chat's event log. "
                         "Start the gateway (or run the job inside it) and retry.")


async def standalone_send(pconfig, chat_id, message, *, thread_id=None, media_files=None, force_document=False):
    del pconfig, thread_id, force_document
    from gateway.config import Platform

    runner, adapter = _live_adapter(Platform("aleph"))
    if adapter is None:
        return {"error": NO_LIVE_GATEWAY_ERROR}
    paths, skipped = [], False
    for media_path, _is_voice in media_files or []:
        if os.path.exists(media_path):
            paths.append(media_path)
        else:
            skipped = True
            logger.warning("Aleph media file not found, skipping: %s", media_path)
    if media_files and not paths and not (message or "").strip():
        return _error("Aleph send failed: no valid media files")

    def make_coro():
        if paths:
            return adapter.send_files(chat_id, paths, caption=message or "")
        return adapter.send(chat_id=chat_id, content=message)

    from tools.send_message_tool import _dispatch_on_gateway_loop
    try:
        sent = await _dispatch_on_gateway_loop(runner, make_coro, "send_message: failed to schedule Aleph send")
    except Exception as exc:
        return _error(f"Aleph send failed: {exc}")
    if isinstance(sent, dict):
        return sent
    if not sent.success:
        return _error(f"Aleph send failed: {sent.error or 'unknown adapter error'}")
    warnings = ["Some media files were skipped (not found on disk)"] if skipped else []
    return _success("aleph", chat_id, warnings)
