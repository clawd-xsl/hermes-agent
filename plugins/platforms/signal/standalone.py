"""send_message / cron delivery for Signal (the plugin's ``standalone_sender_fn``).

signal-ts is a linked device whose protocol state belongs to exactly one live client: a second,
one-shot client would race the gateway's ratchets and Signal would disconnect one of them
(ConnectedElsewhere). So every out-of-band send is delivered through the running gateway's adapter
(its formatting, chunking, attachment batching and shared rate-limit scheduler), dispatched on the
gateway's event loop; without a live gateway the send fails with an explicit error instead of
opening a second connection."""

import logging
import os

from tools.send_message_senders import _error, _live_adapter, _success

logger = logging.getLogger(__name__)

NO_LIVE_GATEWAY_ERROR = (
    "Signal delivery needs the running gateway: its persistent signal-ts connection is the only "
    "Signal client for this linked device. Start the gateway (or run the job inside it) and retry."
)


async def _send_signal(chat_id, message, media_files=None):
    """Send text and/or files through the live gateway adapter."""
    from gateway.config import Platform

    runner, adapter = _live_adapter(Platform.SIGNAL)
    if adapter is None:
        return {"error": NO_LIVE_GATEWAY_ERROR}
    paths, skipped = [], False
    for media_path, _is_voice in media_files or []:
        if os.path.exists(media_path):
            paths.append(media_path)
        else:
            skipped = True
            logger.warning("Signal media file not found, skipping: %s", media_path)
    if media_files and not paths and not (message or "").strip():
        return _error("Signal send failed: no valid media files")

    def make_coro():
        if paths:
            return adapter.send_files(chat_id, paths, caption=message or "")
        return adapter.send(chat_id=chat_id, content=message)

    from tools.send_message_tool import _dispatch_on_gateway_loop
    try:
        sent = await _dispatch_on_gateway_loop(runner, make_coro, "send_message: failed to schedule Signal send")
    except Exception as e:
        return _error(f"Signal send failed: {e}")
    if isinstance(sent, dict):  # the gateway loop could not take the send
        return sent
    if not sent.success:
        return _error(f"Signal send failed: {sent.error or 'unknown adapter error'}")
    warnings = ["Some media files were skipped (not found on disk)"] if skipped else []
    # Result-safe chat identifier for tool transcripts/log consumers.
    return _success("signal", "group:***" if str(chat_id).startswith("group:") else chat_id, warnings)


async def standalone_send(pconfig, chat_id, message, *, thread_id=None, media_files=None, force_document=False):
    """``standalone_sender_fn``: thread_id and force_document have no Signal equivalent."""
    del pconfig, thread_id, force_document  # runtime paths belong to the connected adapter
    return await _send_signal(chat_id, message, media_files=media_files)
