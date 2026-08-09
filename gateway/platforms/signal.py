"""Signal messenger adapter backed by a persistent ``signal-ts`` process.

The Node sidecar embeds libsignal directly.  It keeps one authenticated socket
and one durable protocol-state owner alive for the entire gateway lifetime,
avoiding signal-cli's Java process and HTTP/SSE + JSON-RPC latency.
"""

import asyncio
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import unquote

from gateway.config import Platform, PlatformConfig, _signal_direct_runtime_configured
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    ProcessingOutcome,
    SendResult,
    cache_image_from_bytes,
    cache_audio_from_bytes,
    cache_document_from_bytes,
    cache_image_from_url,
)
from gateway.platforms.helpers import redact_phone
from gateway.platforms.media_cache import DEFAULT_EXT_TO_MIME, mime_for_ext
from tools.audio_container import CONTAINER_TO_EXT, sniff_container
from gateway.platforms.signal_format import markdown_to_signal
from gateway.platforms.signal_rate_limit import (
    SIGNAL_BATCH_PACING_NOTICE_THRESHOLD,
    SIGNAL_MAX_ATTACHMENTS_PER_MSG,
    SIGNAL_RATE_LIMIT_MAX_ATTEMPTS,
    SignalRateLimitError,
    _extract_retry_after_seconds,
    _format_wait,
    _is_signal_rate_limit_error,
    _signal_send_timeout,
    get_scheduler,
)
from gateway.platforms.signal_ts import (
    SignalTsCallError,
    SignalTsProcessError,
    SignalTsSidecar,
    expand_runtime_path,
    resolve_node_executable,
)
from hermes_cli.config import get_hermes_home
from utils import is_truthy_value

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SIGNAL_MAX_ATTACHMENT_SIZE = 100 * 1024 * 1024  # 100 MB
MAX_MESSAGE_LENGTH = 8000  # Signal message size limit
TYPING_INTERVAL = 8.0  # seconds between typing indicator refreshes
SIGNAL_TS_RETRY_DELAY_INITIAL = 1.0
SIGNAL_TS_RETRY_DELAY_MAX = 10.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_comma_list(value: str) -> List[str]:
    """Split a comma-separated string into a list, stripping whitespace."""
    return [v.strip() for v in value.split(",") if v.strip()]


def _guess_extension(data: bytes) -> str:
    """Guess file extension from magic bytes.

    Android Signal delivers voice notes as raw ADTS AAC frames, which share
    the ``0xFF 0xFx`` sync word with MPEG-1/2 Layer 3 (MP3). The byte-1
    layout disambiguates: ADTS packs ``ID layer protection_absent`` into
    bits 3-0, where ``ID`` is 0 for MPEG-2/4 AAC and ``layer`` is always
    0 for ADTS. A real MP3 frame has ``ID=1`` and ``layer`` in {1, 2, 3}.
    """
    if data[:4] == b"\x89PNG":
        return ".png"
    if data[:2] == b"\xff\xd8":
        return ".jpg"
    if data[:4] == b"GIF8":
        return ".gif"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    if data[:4] == b"%PDF":
        return ".pdf"
    # Audio/AV containers: delegate to the shared central sniffer
    # (tools/audio_container.py) — ONE module owns magic-byte container
    # detection. It handles the brand/form-type disambiguations this
    # function used to carry locally: RIFF/WAVE vs WEBP (WEBP is claimed
    # above, before delegation), ftyp audio brands ("M4A ", "M4B ") vs
    # video brands (isom/mp42/avc1/qt), and MP3 vs ADTS AAC sync words.
    container = sniff_container(data)
    if container is not None:
        return CONTAINER_TO_EXT[container]
    if data[:2] == b"PK":
        return ".zip"
    return ".bin"


def _is_image_ext(ext: str) -> bool:
    return ext.lower() in {".jpg", ".jpeg", ".png", ".gif", ".webp"}


def _is_audio_ext(ext: str) -> bool:
    return ext.lower() in {".mp3", ".wav", ".ogg", ".m4a", ".aac"}


# Historical Signal ext→mime table now lives in
# gateway.platforms.media_cache.DEFAULT_EXT_TO_MIME (byte-identical);
# kept as a module alias for backwards compatibility with any callers
# that referenced the private name.
_EXT_TO_MIME = DEFAULT_EXT_TO_MIME


def _ext_to_mime(ext: str) -> str:
    """Map file extension to MIME type."""
    # preserves historical signal mapping (shared table matches verbatim)
    return mime_for_ext(ext, fallback="application/octet-stream")


def _remux_aac_to_m4a(aac_data: bytes) -> Optional[Tuple[bytes, str]]:
    """Losslessly remux raw ADTS AAC bytes into an MP4 (.m4a) container.

    Used by the Signal attachment cache so Android voice notes land on disk
    in a container that every major STT API (Groq, OpenAI, xAI, Mistral
    Voxtral) will accept. ``ffmpeg -c:a copy`` is a single demux/remux —
    no re-encode, no quality loss, sub-100ms for typical voice-note sizes.

    Returns ``(m4a_bytes, ".m4a")`` on success, or ``None`` if ffmpeg is
    missing, input is invalid, or remux fails for any reason. Callers
    must treat ``None`` as "pass through unchanged" and not raise.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        # Common Homebrew/local prefixes on macOS dev hosts.
        for prefix in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"):
            if os.path.isfile(prefix) and os.access(prefix, os.X_OK):
                ffmpeg = prefix
                break
    if not ffmpeg:
        logger.debug("Signal: ffmpeg not found, skipping AAC→M4A remux")
        return None
    try:
        with tempfile.NamedTemporaryFile(suffix=".aac", delete=False) as src:
            src.write(aac_data)
            src_path = src.name
        dst_path = src_path[:-4] + ".m4a"
        try:
            proc = subprocess.run(
                [ffmpeg, "-y", "-loglevel", "error", "-i", src_path,
                 "-c:a", "copy", "-movflags", "+faststart", dst_path],
                capture_output=True, timeout=10,
            )
            if proc.returncode != 0:
                logger.warning(
                    "Signal: AAC→M4A remux failed (ffmpeg exit %d): %s",
                    proc.returncode, proc.stderr.decode("utf-8", "replace")[:300],
                )
                return None
            with open(dst_path, "rb") as f:
                return f.read(), ".m4a"
        finally:
            for p in (src_path, dst_path):
                try:
                    os.unlink(p)
                except OSError:
                    pass
    except subprocess.TimeoutExpired:
        logger.warning("Signal: AAC→M4A remux timed out (>10s)")
        return None
    except Exception:
        logger.exception("Signal: AAC→M4A remux error")
        return None


def _render_mentions(text: str, mentions: list) -> str:
    """Replace Signal mention placeholders (\\uFFFC) with readable @identifiers.

    Signal encodes @mentions as the Unicode object replacement character
    with out-of-band metadata containing the mentioned user's UUID/number.
    """
    if not mentions or "\uFFFC" not in text:
        return text
    # Sort mentions by start position (reverse) to replace from end to start
    # so indices don't shift as we replace
    sorted_mentions = sorted(mentions, key=lambda m: m.get("start", 0), reverse=True)
    for mention in sorted_mentions:
        start = mention.get("start", 0)
        length = mention.get("length", 1)
        # Use the mention's number or UUID as the replacement
        identifier = mention.get("number") or mention.get("uuid") or "user"
        replacement = f"@{identifier}"
        text = text[:start] + replacement + text[start + length:]
    return text


def _is_signal_service_id(value: str) -> bool:
    """Return True if *value* already looks like a Signal service identifier."""
    if not value:
        return False
    if value.startswith("PNI:") or value.startswith("u:"):
        return True
    try:
        uuid.UUID(value)
        return True
    except (ValueError, AttributeError, TypeError):
        return False


def _looks_like_e164_number(value: str) -> bool:
    """Return True for a plausible E.164 phone number."""
    if not value or not value.startswith("+"):
        return False
    digits = value[1:]
    return digits.isdigit() and 7 <= len(digits) <= 15


def check_signal_requirements(config: PlatformConfig) -> bool:
    """Return whether the direct Signal runtime has complete local config."""
    return _signal_direct_runtime_configured(config)


# ---------------------------------------------------------------------------
# Signal Adapter
# ---------------------------------------------------------------------------

class SignalAdapter(BasePlatformAdapter):
    """Signal messenger adapter using a direct, persistent signal-ts client."""

    platform = Platform.SIGNAL
    enforces_own_access_policy = True
    # Signal has no real edit API for already-sent messages. Mark it explicitly
    # so streaming suppresses the visible cursor instead of leaving a stale tofu
    # square behind in chat clients when edit attempts fail.
    SUPPORTS_MESSAGE_EDITING = False

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.SIGNAL)

        extra = config.extra or {}
        self.account = extra.get("account", "")
        self.state_path = expand_runtime_path(str(extra.get("state_path") or ""))
        self.sdk_path = expand_runtime_path(str(extra.get("sdk_path") or ""))
        self.node_command = str(extra.get("node_command") or "node")
        self.node_executable = resolve_node_executable(self.node_command)
        self.startup_timeout = float(extra.get("startup_timeout_seconds", 30.0))
        self.call_timeout = float(extra.get("call_timeout_seconds", 30.0))
        self._signal_cache_dir = get_hermes_home() / "cache" / "signal-ts"
        self.ignore_stories = is_truthy_value(extra.get("ignore_stories"), default=True)

        # Signal behavior is config.yaml-owned.  ``.env`` remains reserved for
        # secrets; this linked-device runtime has no user-entered secret value.
        group_allowed = extra.get("group_allow_from", "")
        if isinstance(group_allowed, list):
            self.group_allow_from = {str(value).strip() for value in group_allowed if str(value).strip()}
        else:
            self.group_allow_from = set(_parse_comma_list(str(group_allowed)))

        # Mention filter — only respond in groups when the bot account is @mentioned.
        self.require_mention = is_truthy_value(
            extra.get("require_mention"), default=False
        )

        # DM allowlist is enforced before dispatch.  Keeping it on the adapter
        # also lets reaction hooks skip unauthorized senders
        # (reactions fire before run.py's auth gate, so without this check
        # every inbound DM from any contact gets a 👀 reaction).
        # "*" means all users allowed (open mode); empty means no restriction
        # recorded at adapter level (run.py still enforces auth separately).
        dm_allowed = extra.get("allow_from", "")
        if isinstance(dm_allowed, list):
            self.dm_allow_from = {str(value).strip() for value in dm_allowed if str(value).strip()}
        else:
            self.dm_allow_from = set(_parse_comma_list(str(dm_allowed)))
        self._dm_policy = "allowlist" if self.dm_allow_from else "pairing"
        self._group_policy = "allowlist" if self.group_allow_from else "disabled"
        self.reactions_enabled = is_truthy_value(
            extra.get("reactions"), default=True
        )

        # Persistent signal-ts runtime and reconnect supervisor.
        self._sidecar: Optional[SignalTsSidecar] = None
        self._sidecar_task: Optional[asyncio.Task] = None
        self._typing_tasks: Dict[str, asyncio.Task] = {}
        # Per-chat typing-indicator backoff. When Signal reports
        # NETWORK_FAILURE (recipient offline / unroutable), base.py's
        # _keep_typing refresh loop would otherwise hammer sendTyping every
        # ~2s indefinitely, producing WARNING-level log spam and pointless
        # RPC traffic. We track consecutive failures per chat and skip the
        # sidecar calls during a cooldown window instead.
        self._typing_failures: Dict[str, int] = {}
        self._typing_skip_until: Dict[str, float] = {}
        self._running = False

        # Normalize account for self-message filtering
        self._account_normalized = self.account.strip()

        # Track recently sent message timestamps to prevent echo-back loops
        # in Note to Self / self-chat mode and linked-device group sync-sents.
        # OrderedDict[timestamp_ms -> insertion_monotonic_seconds] gives us
        # LRU eviction (popitem(last=False) drops oldest) plus a TTL so that
        # under chatty groups a still-pending echo cannot be evicted just
        # because >50 outbounds happened. With a 5-minute TTL the cap only
        # matters for runaway producers, not normal traffic bursts.
        self._recent_sent_timestamps: "OrderedDict[int, float]" = OrderedDict()
        self._max_recent_timestamps = 512
        self._recent_sent_ttl_seconds = 300.0
        # Keep a separate bounded cache of outbound Signal message timestamps.
        # Signal quote.id is the timestamp of the quoted message, so this lets
        # inbound replies identify that the user replied to a message sent by
        # this bot even after the self-sync echo was filtered above.
        # OrderedDict (not set) so the cap evicts the OLDEST timestamp in FIFO
        # order — a plain set.pop() removes an arbitrary element, which could
        # drop a still-recent timestamp and miss a genuine reply-to-own-message.
        self._sent_message_timestamps: "OrderedDict[str, None]" = OrderedDict()
        self._max_sent_message_timestamps = 500
        # Signal increasingly exposes ACI/PNI UUIDs as stable recipient IDs.
        # Keep a best-effort mapping so outbound sends can upgrade from a
        # phone number to the corresponding UUID when Signal resolves it.
        self._recipient_uuid_by_number: Dict[str, str] = {}
        self._recipient_number_by_uuid: Dict[str, str] = {}
        self._recipient_cache_lock = asyncio.Lock()

        logger.info("Signal adapter initialized: runtime=signal-ts account=%s groups=%s",
                     redact_phone(self.account),
                     "enabled" if self.group_allow_from else "disabled")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        """Start one direct Signal client and its reconnect supervisor."""
        if not self.node_executable or not self.state_path.is_file() or not self.sdk_path.exists():
            logger.error(
                "Signal: state_path, sdk_path, and an executable node_command are required"
            )
            return False

        # Acquire scoped lock to prevent duplicate Signal listeners for the same phone
        lock_acquired = False
        try:
            if not self._acquire_platform_lock('signal-phone', self.account, 'Signal account'):
                return False
            lock_acquired = True
        except Exception as e:
            logger.warning("Signal: Could not acquire phone lock (non-fatal): %s", e)

        try:
            sidecar = self._new_sidecar()
            await sidecar.start()
            self._sidecar = sidecar
            if not self.account and sidecar.account:
                self.account = sidecar.account
                self._account_normalized = self.account.strip()
            self._running = True
            self._sidecar_task = asyncio.create_task(self._supervise_sidecar(sidecar))
            logger.info(
                "Signal: signal-ts connected (account=%s aci=%s)",
                redact_phone(self.account),
                sidecar.aci or "unknown",
            )
            return True
        except Exception as exc:
            logger.error("Signal: signal-ts startup failed: %s", exc)
            if lock_acquired:
                self._release_platform_lock()
            return False

    async def disconnect(self) -> None:
        """Stop the persistent signal-ts runtime and clean up."""
        self._running = False

        sidecar = self._sidecar
        self._sidecar = None
        if sidecar:
            await sidecar.close()
        if self._sidecar_task:
            await asyncio.gather(self._sidecar_task, return_exceptions=True)
            self._sidecar_task = None

        # Cancel all typing tasks
        for task in self._typing_tasks.values():
            task.cancel()
        self._typing_tasks.clear()

        self._release_platform_lock()

        logger.info("Signal: disconnected")

    # ------------------------------------------------------------------
    # Persistent runtime supervision
    # ------------------------------------------------------------------

    def _new_sidecar(self) -> SignalTsSidecar:
        assert self.node_executable
        return SignalTsSidecar(
            node_executable=self.node_executable,
            sdk_path=self.sdk_path,
            state_path=self.state_path,
            cache_dir=self._signal_cache_dir,
            expected_account=self.account or None,
            on_envelope=self._handle_envelope,
            startup_timeout=self.startup_timeout,
            call_timeout=self.call_timeout,
        )

    async def _supervise_sidecar(self, sidecar: SignalTsSidecar) -> None:
        """Restart a lost Signal socket with bounded exponential backoff."""
        backoff = SIGNAL_TS_RETRY_DELAY_INITIAL
        current = sidecar
        while self._running:
            await current.wait_closed()
            if not self._running:
                return
            if self._sidecar is current:
                self._sidecar = None
            logger.warning("Signal: signal-ts connection lost; reconnecting in %.1fs", backoff)
            try:
                await asyncio.sleep(backoff)
                replacement = self._new_sidecar()
                await replacement.start()
                self._sidecar = replacement
                current = replacement
                backoff = SIGNAL_TS_RETRY_DELAY_INITIAL
                logger.info("Signal: signal-ts reconnected")
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.warning("Signal: signal-ts reconnect failed: %s", exc)
                backoff = min(backoff * 2, SIGNAL_TS_RETRY_DELAY_MAX)

    # ------------------------------------------------------------------
    # Message Handling
    # ------------------------------------------------------------------

    async def _handle_envelope(self, envelope: dict) -> None:
        """Process an incoming signal-ts envelope in the gateway contract."""
        # Unwrap nested envelope if present
        envelope_data = envelope.get("envelope", envelope)

        # Handle syncMessage: extract "Note to Self" messages (sent to own account)
        # while still filtering other sync events (read receipts, typing, etc.)
        is_note_to_self = False
        if "syncMessage" in envelope_data:
            sync_msg = envelope_data.get("syncMessage")
            if sync_msg and isinstance(sync_msg, dict):
                sent_msg = sync_msg.get("sentMessage")
                if sent_msg and isinstance(sent_msg, dict):
                    dest = sent_msg.get("destinationNumber") or sent_msg.get("destination")
                    sent_ts = sent_msg.get("timestamp")
                    sent_msg_group_info = sent_msg.get("groupInfo") or {}
                    sent_msg_group_id = sent_msg_group_info.get("groupId") if sent_msg_group_info else None
                    if dest == self._account_normalized or sent_msg_group_id:
                        # Check if this is an echo of our own outbound reply
                        if self._consume_sent_timestamp(sent_ts):
                            return
                        # Genuine user Note to Self — promote to dataMessage
                        is_note_to_self = True
                        envelope_data = {**envelope_data, "dataMessage": sent_msg}
            if not is_note_to_self:
                return

        # Extract sender info
        sender = (
            envelope_data.get("sourceNumber")
            or envelope_data.get("sourceUuid")
            or envelope_data.get("source")
        )
        sender_name = envelope_data.get("sourceName", "")
        sender_uuid = envelope_data.get("sourceUuid", "")
        self._remember_recipient_identifiers(sender, sender_uuid)

        if not sender:
            logger.debug("Signal: ignoring envelope with no sender")
            return

        # Config-owned intake policy.  Pairing mode deliberately forwards the
        # DM so GatewayAuthorizationMixin can issue/verify a pairing code; an
        # explicit allowlist is enforced before any reactions or agent work.
        if self._dm_policy == "allowlist":
            sender_ids = {str(sender)}
            if sender_uuid:
                sender_ids.add(str(sender_uuid))
            if is_note_to_self and self.account:
                sender_ids.add(self.account)
            if "*" not in self.dm_allow_from and not (sender_ids & self.dm_allow_from):
                logger.debug("Signal: ignoring sender outside configured allow_from")
                return

        # Self-message filtering — prevent reply loops (but allow Note to Self)
        if self._account_normalized and sender == self._account_normalized and not is_note_to_self:
            return

        # Filter stories
        if self.ignore_stories and envelope_data.get("storyMessage"):
            return

        # Get data message — also check editMessage (edited messages contain
        # their updated dataMessage inside editMessage.dataMessage)
        data_message = (
            envelope_data.get("dataMessage")
            or (envelope_data.get("editMessage") or {}).get("dataMessage")
        )
        if not data_message:
            return

        # Check for group message
        group_info = data_message.get("groupInfo")
        group_id = group_info.get("groupId") if group_info else None
        is_group = bool(group_id)

        # Group message filtering — config-owned ``group_allow_from``:
        # - Empty → groups disabled (default safe behavior)
        # - Group IDs → only those groups allowed
        # - "*" → all groups allowed
        if is_group:
            if not self.group_allow_from:
                logger.debug("Signal: ignoring group message (group_allow_from is empty)")
                return
            if "*" not in self.group_allow_from and group_id not in self.group_allow_from:
                logger.debug("Signal: group %s not in allowlist", group_id[:8] if group_id else "?")
                return

        # Build chat info
        chat_id = sender if not is_group else f"group:{group_id}"
        chat_type = "group" if is_group else "dm"

        # Extract text and render mentions
        text = data_message.get("message", "")
        mentions = data_message.get("mentions", [])
        if text and mentions:
            text = _render_mentions(text, mentions)

        # Mention filter: in groups, only process messages that @mention the bot account
        if is_group and self.require_mention:
            account_norm = self._account_normalized
            # Check rendered mention tags OR raw mention metadata
            mentioned_in_text = account_norm and (
                f"@{account_norm}" in (text or "")
            )
            mentioned_in_metadata = any(
                m.get("number") == account_norm or m.get("uuid") == account_norm
                for m in (data_message.get("mentions") or [])
            )
            if not mentioned_in_text and not mentioned_in_metadata:
                logger.debug(
                    "Signal: ignoring group message (require_mention=true, bot not mentioned)"
                )
                return

        # Strip the bot's own @mention from any group message so the agent
        # doesn't misinterpret "@+155****4567 say hello" as a directive to
        # contact that phone number. _render_mentions replaces the Signal
        # ￼ placeholder with @<number-or-uuid>, which looks like an
        # addressee to the LLM rather than a self-reference. Applies to every
        # group (not just require_mention groups) so the self-mention is
        # cleaned wherever it appears.
        if is_group and text:
            account_norm = self._account_normalized
            if account_norm:
                text = text.replace(f"@{account_norm}", "")
                # Also strip if the mention was rendered using the bot's UUID
                bot_uuid = self._recipient_uuid_by_number.get(account_norm)
                if bot_uuid:
                    text = text.replace(f"@{bot_uuid}", "")
                # Tidy the spacing the removed mention left behind: collapse the
                # double-space at a mid-sentence removal and trim the ends.
                # Only touches the doubled space the removal introduced, so
                # intentional newlines in a multi-line message are preserved.
                text = text.replace("  ", " ").strip()

        # Extract quote (reply-to) context from Signal dataMessage. Signal's
        # quote.id is the timestamp of the quoted message; quote.author points
        # at the quoted sender when available. Preserve both so the gateway can
        # tell the agent when the user replied to a specific assistant message.
        quote_data = data_message.get("quote") or {}
        reply_to_id = str(quote_data.get("id")) if quote_data.get("id") else None
        reply_to_text = quote_data.get("text")
        reply_to_author = self._extract_quote_author(quote_data)
        reply_to_author_name = quote_data.get("authorName") or quote_data.get("authorProfileName")
        reply_to_is_own = self._quote_references_own_message(reply_to_id, reply_to_author)

        # Process attachments
        attachments_data = data_message.get("attachments", [])
        media_urls = []
        media_types = []

        if attachments_data and not getattr(self, "ignore_attachments", False):
            for att in attachments_data:
                att_id = att.get("id")
                att_size = att.get("size", 0)
                if not att_id:
                    continue
                if att_size > SIGNAL_MAX_ATTACHMENT_SIZE:
                    logger.warning("Signal: attachment too large (%d bytes), skipping", att_size)
                    continue
                try:
                    cached_path, ext = await self._fetch_attachment(att_id)
                    if cached_path:
                        # Use contentType from Signal if available, else map from extension
                        content_type = att.get("contentType") or _ext_to_mime(ext)
                        media_urls.append(cached_path)
                        media_types.append(content_type)
                except Exception:
                    logger.exception("Signal: failed to fetch attachment %s", att_id)

        # Skip envelopes with no meaningful content (no text, no attachments).
        # Catches profile key updates, empty messages, and other metadata-only
        # envelopes that still carry a dataMessage wrapper but have nothing
        # worth processing. Profile-key and metadata updates must not trigger a
        # full agent turn.
        if (not text or not text.strip()) and not media_urls:
            logger.debug(
                "Signal: skipping contentless envelope from %s (%d attachments)",
                redact_phone(sender), len(media_urls) if media_urls else 0,
            )
            return

        # Build session source
        source = self.build_source(
            chat_id=chat_id,
            chat_name=group_info.get("groupName") if group_info else sender_name,
            chat_type=chat_type,
            user_id=sender,
            user_name=sender_name or sender,
            user_id_alt=sender_uuid if sender_uuid else None,
            chat_id_alt=group_id if is_group else None,
        )

        # Determine message type from media
        msg_type = MessageType.TEXT
        if media_types:
            if any(mt.startswith("audio/") for mt in media_types):
                msg_type = MessageType.VOICE
            elif any(mt.startswith("image/") for mt in media_types):
                msg_type = MessageType.PHOTO
            elif any(mt.startswith("video/") for mt in media_types):
                msg_type = MessageType.VIDEO
            else:
                # Catch-all: application/*, text/*, and unknown MIME types are
                # treated as documents so run.py's document-context injection
                # surfaces the cached file path to the agent (same pattern as
                # WhatsApp/Slack/BlueBubbles/Mattermost).
                msg_type = MessageType.DOCUMENT

        # Parse timestamp from envelope data (milliseconds since epoch)
        ts_ms = envelope_data.get("timestamp", 0)
        if ts_ms:
            try:
                timestamp = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
            except (ValueError, OSError):
                timestamp = datetime.now(tz=timezone.utc)
        else:
            timestamp = datetime.now(tz=timezone.utc)

        # Build and dispatch event.
        # Store raw envelope data in raw_message so on_processing_start/complete
        # can extract targetAuthor + targetTimestamp for sendReaction.
        event = MessageEvent(
            source=source,
            text=text or "",
            message_type=msg_type,
            media_urls=media_urls,
            media_types=media_types,
            timestamp=timestamp,
            raw_message={
                "sender": sender,
                "timestamp_ms": ts_ms,
                "quote": quote_data if quote_data else None,
            },
            reply_to_message_id=reply_to_id,
            reply_to_text=reply_to_text,
            reply_to_author_id=reply_to_author,
            reply_to_author_name=reply_to_author_name,
            reply_to_is_own_message=reply_to_is_own,
        )

        logger.debug("Signal: message from %s in %s: %s",
                      redact_phone(sender), chat_id[:20], (text or "")[:50])

        await self.handle_message(event)

    def _remember_recipient_identifiers(self, number: Optional[str], service_id: Optional[str]) -> None:
        """Cache any number↔UUID mapping observed from Signal envelopes."""
        if not number or not service_id or not _is_signal_service_id(service_id):
            return
        self._recipient_uuid_by_number[number] = service_id
        self._recipient_number_by_uuid[service_id] = number

    @staticmethod
    def _extract_quote_author(quote_data: Any) -> Optional[str]:
        """Return the best available Signal sender identifier from quote metadata."""
        if not isinstance(quote_data, dict):
            return None
        for key in (
            "author",
            "authorNumber",
            "authorUuid",
            "authorAci",
            "authorServiceId",
            "authorServiceIdString",
        ):
            value = quote_data.get(key)
            if value:
                return str(value)
        return None

    def _quote_references_own_message(
        self,
        reply_to_id: Optional[str],
        reply_to_author: Optional[str],
    ) -> bool:
        """True when a Signal quote points at this adapter's outbound message."""
        if reply_to_id and str(reply_to_id) in self._sent_message_timestamps:
            return True
        if not reply_to_author:
            return False
        author = str(reply_to_author).strip()
        if self._account_normalized and author == self._account_normalized:
            return True
        cached_uuid = self._recipient_uuid_by_number.get(self._account_normalized)
        if cached_uuid and author == cached_uuid:
            return True
        cached_number = self._recipient_number_by_uuid.get(author)
        return bool(cached_number and cached_number == self._account_normalized)

    def _remember_sent_message_timestamp(self, timestamp: Any) -> None:
        """Keep a bounded cache of outbound Signal timestamps for quote matching."""
        if timestamp is None:
            return
        key = str(timestamp)
        # Re-insert to mark most-recently-used so eviction drops genuinely old
        # timestamps, not a recently re-seen one.
        self._sent_message_timestamps.pop(key, None)
        self._sent_message_timestamps[key] = None
        # FIFO-evict the oldest entry once over the cap.
        while len(self._sent_message_timestamps) > self._max_sent_message_timestamps:
            self._sent_message_timestamps.popitem(last=False)

    def _extract_contact_uuid(self, contact: Any, phone_number: str) -> Optional[str]:
        """Best-effort extraction of a Signal service ID from listContacts output."""
        if not isinstance(contact, dict):
            return None

        number = contact.get("number")
        recipient = contact.get("recipient")
        service_id = contact.get("uuid") or contact.get("serviceId")
        if not service_id:
            profile = contact.get("profile")
            if isinstance(profile, dict):
                service_id = profile.get("serviceId") or profile.get("uuid")

        if service_id and _is_signal_service_id(service_id):
            matches_number = number == phone_number or recipient == phone_number
            if matches_number:
                return service_id
        return None

    async def _resolve_recipient(self, chat_id: str) -> str:
        """Return the preferred Signal recipient identifier for a direct chat."""
        if (
            not chat_id
            or chat_id.startswith("group:")
            or _is_signal_service_id(chat_id)
            or not _looks_like_e164_number(chat_id)
        ):
            return chat_id

        cached = self._recipient_uuid_by_number.get(chat_id)
        if cached:
            return cached

        async with self._recipient_cache_lock:
            cached = self._recipient_uuid_by_number.get(chat_id)
            if cached:
                return cached

            contacts = await self._rpc("listContacts", {
                "account": self.account,
                "allRecipients": True,
            })
            if isinstance(contacts, list):
                for contact in contacts:
                    number = contact.get("number") if isinstance(contact, dict) else None
                    service_id = self._extract_contact_uuid(contact, chat_id)
                    if number and service_id:
                        self._remember_recipient_identifiers(number, service_id)

            return self._recipient_uuid_by_number.get(chat_id, chat_id)

    # ------------------------------------------------------------------
    # Attachment Handling
    # ------------------------------------------------------------------

    async def _fetch_attachment(self, attachment_id: str) -> tuple:
        """Download an attachment through the live signal-ts client."""
        result = await self._rpc("getAttachment", {
            "account": self.account,
            "id": attachment_id,
        })

        if not result:
            return None, ""

        if not isinstance(result, dict) or not result.get("path"):
            logger.warning("Signal: signal-ts attachment response missing path")
            return None, ""
        staged_path = Path(str(result["path"])).resolve()
        cache_root = self._signal_cache_dir.resolve()
        if not staged_path.is_relative_to(cache_root):
            raise SignalTsProcessError("signal-ts returned an attachment outside its cache")
        try:
            raw_data = await asyncio.to_thread(staged_path.read_bytes)
        finally:
            try:
                staged_path.unlink()
            except OSError:
                pass
        ext = _guess_extension(raw_data)

        # Android Signal voice notes are raw ADTS AAC streams. Most STT
        # providers (Groq Whisper, OpenAI Whisper) reject raw ADTS — they
        # require AAC to be muxed into an MP4 container. Remux losslessly
        # with ``ffmpeg -c:a copy`` so the cached file is a normal .m4a.
        # No re-encode, sub-100ms on a Pi 5. Graceful no-op if ffmpeg is
        # absent: the raw ADTS file is cached as-is and STT may reject it
        # (there is no downstream sniff-and-remux fallback).
        if ext == ".aac":
            remuxed: Optional[Tuple[bytes, str]] = await asyncio.to_thread(_remux_aac_to_m4a, raw_data)
            if remuxed is not None:
                raw_data, ext = remuxed

        if _is_image_ext(ext):
            path = cache_image_from_bytes(raw_data, ext)
        elif _is_audio_ext(ext):
            path = cache_audio_from_bytes(raw_data, ext)
        else:
            path = cache_document_from_bytes(raw_data, ext)

        return path, ext

    # ------------------------------------------------------------------
    # Persistent signal-ts commands
    # ------------------------------------------------------------------

    async def _rpc(
        self,
        method: str,
        params: dict,
        rpc_id: str = None,
        *,
        log_failures: bool = True,
        raise_on_rate_limit: bool = False,
        timeout: float = 30.0,
    ) -> Any:
        """Invoke one operation on the live signal-ts process.

        When ``log_failures=False``, error and exception paths log at DEBUG
        instead of WARNING — used by the typing-indicator path to silence
        repeated NETWORK_FAILURE spam for unreachable recipients while
        still preserving visibility for the first occurrence and for
        unrelated RPCs.

        When ``raise_on_rate_limit=True``, a Signal ``[429]`` /
        ``RateLimitException`` response raises ``SignalRateLimitError``
        instead of being swallowed — lets callers (multi-attachment send)
        opt into backoff-retry without changing default behaviour.
        """
        sidecar = self._sidecar
        if not sidecar or not sidecar.running:
            logger.warning("Signal: signal-ts call attempted while disconnected")
            return None

        try:
            return await sidecar.call(method, params, timeout=timeout)

        except SignalRateLimitError:
            raise
        except SignalTsCallError as e:
            if raise_on_rate_limit and _is_signal_rate_limit_error(e.details or str(e)):
                raise SignalRateLimitError(
                    str(e), retry_after=_extract_retry_after_seconds(e.details or str(e))
                ) from e
            if log_failures:
                logger.warning("Signal signal-ts %s failed: %s", method, e)
            else:
                logger.debug("Signal signal-ts %s failed: %s", method, e)
            return None
        except SignalTsProcessError as e:
            if log_failures:
                logger.warning("Signal signal-ts process failure during %s: %s", method, e)
            else:
                logger.debug("Signal signal-ts process failure during %s: %s", method, e)
            return None

    # ------------------------------------------------------------------
    # Formatting — markdown → Signal body ranges
    # ------------------------------------------------------------------

    @staticmethod
    def _markdown_to_signal(text: str) -> tuple[str, list[str]]:
        """Backward-compatible wrapper around shared Signal formatting helper."""
        return markdown_to_signal(text)

    def format_message(self, content: str) -> str:
        """Strip markdown for plain-text fallback (used by base class).

        The actual rich formatting happens in send() via _markdown_to_signal().
        """
        # This is only called if someone uses the base-class send path.
        # Our send() override bypasses this entirely.
        return content

    def _validate_send_result(self, result: Any) -> tuple[bool, Optional[str]]:
        """Validate normalized Signal send response results.

        Returns (success, error_message).
        """
        if not result or not isinstance(result, dict):
            return True, None

        results = result.get("results")
        if isinstance(results, list):
            for r in results:
                if not isinstance(r, dict):
                    continue
                rtype = r.get("type")
                if rtype and rtype != "SUCCESS":
                    return False, str(rtype)
                if "success" in r and not r.get("success"):
                    fail = r.get("failure")
                    if fail:
                        return False, str(fail)
                    return False, "Recipient delivery failed"
        return True, None

    # ------------------------------------------------------------------
    # Sending
    # ------------------------------------------------------------------

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a text message with native Signal formatting."""
        await self._stop_typing_indicator(chat_id)

        plain_text, text_styles = self._markdown_to_signal(content)

        params: Dict[str, Any] = {
            "account": self.account,
            "message": plain_text,
        }

        if text_styles:
            if len(text_styles) == 1:
                params["textStyle"] = text_styles[0]
            else:
                params["textStyles"] = text_styles

        if chat_id.startswith("group:"):
            params["groupId"] = chat_id[6:]
        else:
            params["recipient"] = [await self._resolve_recipient(chat_id)]

        logger.info("[Signal] Sending response (%d chars) to %s", len(plain_text), chat_id)
        result = await self._rpc("send", params)

        if result is not None:
            success, err_msg = self._validate_send_result(result)
            if not success:
                return SendResult(success=False, error=err_msg, raw_response=result)
            self._track_sent_timestamp(result)
            # Signal has no editable message identifier. Returning None keeps the
            # stream consumer on the non-edit fallback path instead of pretending
            # future edits can remove an in-progress cursor from the chat thread.
            return SendResult(success=True, message_id=None)
        return SendResult(success=False, error="RPC send failed")

    def _track_sent_timestamp(self, rpc_result) -> None:
        """Record outbound message timestamp for echo-back filtering."""
        ts = rpc_result.get("timestamp") if isinstance(rpc_result, dict) else None
        if ts:
            self._remember_sent_message_timestamp(ts)
            now = time.monotonic()
            # Re-insert to mark as most-recently-used.
            self._recent_sent_timestamps.pop(ts, None)
            self._recent_sent_timestamps[ts] = now
            # Drop entries older than TTL first (cheap O(k) where k=expired).
            cutoff = now - self._recent_sent_ttl_seconds
            while self._recent_sent_timestamps:
                oldest_ts, oldest_at = next(iter(self._recent_sent_timestamps.items()))
                if oldest_at < cutoff:
                    self._recent_sent_timestamps.popitem(last=False)
                else:
                    break
            # Hard cap as a last-resort guard against runaway producers.
            while len(self._recent_sent_timestamps) > self._max_recent_timestamps:
                self._recent_sent_timestamps.popitem(last=False)

    def _consume_sent_timestamp(self, ts) -> bool:
        """Pop a timestamp if it matches one we sent. Returns True on echo."""
        if ts and ts in self._recent_sent_timestamps:
            self._recent_sent_timestamps.pop(ts, None)
            return True
        return False

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """Send a typing indicator.

        base.py's ``_keep_typing`` refresh loop calls this every ~2s while
        the agent is processing. If Signal returns NETWORK_FAILURE for
        this recipient (offline, unroutable, group membership lost, etc.)
        the unmitigated behaviour is: a WARNING log every 2 seconds for as
        long as the agent keeps running. Instead we:

        - silence the WARNING after the first consecutive failure (subsequent
          attempts log at DEBUG) so transport issues are still visible once
          but don't flood the log,
        - skip the RPC entirely during an exponential cooldown window once
          three consecutive failures have happened, so we stop hammering
          the sidecar with requests it can't deliver.

        A successful sendTyping clears the counters.
        """
        now = time.monotonic()
        skip_until = self._typing_skip_until.get(chat_id, 0.0)
        if now < skip_until:
            return

        params: Dict[str, Any] = {
            "account": self.account,
        }

        if chat_id.startswith("group:"):
            params["groupId"] = chat_id[6:]
        else:
            params["recipient"] = [await self._resolve_recipient(chat_id)]

        fails = self._typing_failures.get(chat_id, 0)
        result = await self._rpc(
            "sendTyping",
            params,
            rpc_id="typing",
            log_failures=(fails == 0),
        )

        if result is None:
            fails += 1
            self._typing_failures[chat_id] = fails
            # After 3 consecutive failures, back off exponentially (16s,
            # 32s, 60s cap) to stop spamming Signal for a recipient
            # that clearly isn't reachable right now.
            if fails >= 3:
                backoff = min(60.0, 16.0 * (2 ** (fails - 3)))
                self._typing_skip_until[chat_id] = now + backoff
        else:
            self._typing_failures.pop(chat_id, None)
            self._typing_skip_until.pop(chat_id, None)

    async def send_multiple_images(
        self,
        chat_id: str,
        images: List[Tuple[str, str]],
        metadata: Optional[Dict[str, Any]] = None,
        human_delay: float = 0.0,
    ) -> None:
        """Send a batch of images via chunked Signal RPC calls.

        Per-image alt texts are dropped — Signal's send RPC only carries
        one shared message body. Bad images (download failure, missing
        file, oversize) are skipped with a warning so one bad URL
        doesn't lose the rest of the batch. ``human_delay`` is ignored:
        the rate-limit scheduler handles inter-batch pacing.
        """
        if not images:
            return

        scheduler = get_scheduler()
        logger.info(
            "Signal send_multiple_images: received %d image(s) for %s — "
            "scheduler state: %s",
            len(images), chat_id[:30], scheduler.state(),
        )

        await self._stop_typing_indicator(chat_id)

        attachments: List[str] = []
        skipped_download = 0
        skipped_missing = 0
        skipped_oversize = 0
        for image_url, _alt_text in images:
            if image_url.startswith("file://"):
                file_path = unquote(image_url[7:])
            else:
                try:
                    file_path = await cache_image_from_url(image_url)
                except Exception as e:
                    logger.warning("Signal: failed to download image %s: %s", image_url, e)
                    skipped_download += 1
                    continue

            if not file_path or not Path(file_path).exists():
                logger.warning("Signal: image file not found for %s", image_url)
                skipped_missing += 1
                continue

            file_size = Path(file_path).stat().st_size
            if file_size > SIGNAL_MAX_ATTACHMENT_SIZE:
                logger.warning(
                    "Signal: image too large (%d bytes), skipping %s", file_size, image_url
                )
                skipped_oversize += 1
                continue

            attachments.append(file_path)

        if not attachments:
            logger.error(
                "Signal: no valid images in batch of %d "
                "(download=%d missing=%d oversize=%d)",
                len(images), skipped_download, skipped_missing, skipped_oversize,
            )
            return

        logger.info(
            "Signal send_multiple_images: %d/%d images valid, sending in chunks",
            len(attachments), len(images),
        )

        base_params: Dict[str, Any] = {
            "account": self.account,
            "message": "",
        }
        if chat_id.startswith("group:"):
            base_params["groupId"] = chat_id[6:]
        else:
            base_params["recipient"] = [await self._resolve_recipient(chat_id)]

        att_batches = [
            attachments[i:i + SIGNAL_MAX_ATTACHMENTS_PER_MSG]
            for i in range(0, len(attachments), SIGNAL_MAX_ATTACHMENTS_PER_MSG)
        ]

        for idx, att_batch in enumerate(att_batches):
            n = len(att_batch)
            estimated = scheduler.estimate_wait(n)
            logger.debug(
                "Signal batch %d/%d: %d attachments, estimated wait=%.1fs",
                idx + 1, len(att_batches), n, estimated,
            )
            if estimated >= SIGNAL_BATCH_PACING_NOTICE_THRESHOLD:
                await self._notify_batch_pacing(
                    chat_id, idx + 1, len(att_batches), estimated
                )

            params = dict(base_params, attachments=att_batch)
            send_timeout = _signal_send_timeout(n)

            for attempt in range(1, SIGNAL_RATE_LIMIT_MAX_ATTEMPTS + 1):
                await scheduler.acquire(n)
                try:
                    _rpc_t0 = time.monotonic()
                    result = await self._rpc(
                        "send", params, raise_on_rate_limit=True, timeout=send_timeout,
                    )
                    _rpc_duration = time.monotonic() - _rpc_t0
                    if result is not None:
                        success, err_msg = self._validate_send_result(result)
                        if success:
                            self._track_sent_timestamp(result)
                            await scheduler.report_rpc_duration(_rpc_duration, n)
                            logger.info(
                                "Signal batch %d/%d: %d attachments sent in %.1fs "
                                "(attempt %d/%d)",
                                idx + 1, len(att_batches), n, _rpc_duration,
                                attempt, SIGNAL_RATE_LIMIT_MAX_ATTEMPTS,
                            )
                        else:
                            logger.error(
                                "Signal: RPC send failed for batch %d/%d (%d attachments, "
                                "attempt %d/%d, rpc_duration=%.1fs): %s",
                                idx + 1, len(att_batches), n,
                                attempt, SIGNAL_RATE_LIMIT_MAX_ATTEMPTS,
                                _rpc_duration, err_msg,
                            )
                            # Retry transient (non-rate-limit) failures once
                            if attempt < SIGNAL_RATE_LIMIT_MAX_ATTEMPTS:
                                backoff = 2.0 ** attempt
                                logger.info(
                                    "Signal: retrying batch %d/%d after %.1fs backoff",
                                    idx + 1, len(att_batches), backoff,
                                )
                                await asyncio.sleep(backoff)
                                continue
                    else:
                        # Assume the server didn't accept the batch, don't deduce tokens
                        logger.error(
                            "Signal: RPC send failed for batch %d/%d (%d attachments, "
                            "attempt %d/%d, rpc_duration=%.1fs)",
                            idx + 1, len(att_batches), n,
                            attempt, SIGNAL_RATE_LIMIT_MAX_ATTEMPTS,
                            _rpc_duration,
                        )
                        # Retry transient (non-rate-limit) failures once
                        if attempt < SIGNAL_RATE_LIMIT_MAX_ATTEMPTS:
                            backoff = 2.0 ** attempt
                            logger.info(
                                "Signal: retrying batch %d/%d after %.1fs backoff",
                                idx + 1, len(att_batches), backoff,
                            )
                            await asyncio.sleep(backoff)
                            continue
                    break
                except SignalRateLimitError as e:
                    scheduler.feedback(e.retry_after, n)
                    if attempt >= SIGNAL_RATE_LIMIT_MAX_ATTEMPTS:
                        logger.error(
                            "Signal: rate-limit retries exhausted on batch %d/%d "
                            "(%d attachments lost, server retry_after=%s)",
                            idx + 1, len(att_batches), n,
                            f"{e.retry_after:.0f}s" if e.retry_after else "unknown",
                        )
                        break
                    logger.warning(
                        "Signal: rate-limited on batch %d/%d "
                        "(attempt %d/%d, server retry_after=%s); "
                        "scheduler will pace the retry",
                        idx + 1, len(att_batches),
                        attempt, SIGNAL_RATE_LIMIT_MAX_ATTEMPTS,
                        f"{e.retry_after:.0f}s" if e.retry_after else "unknown",
                    )

    async def _notify_batch_pacing(
        self,
        chat_id: str,
        next_batch_idx: int,
        total_batches: int,
        wait_s: float,
    ) -> None:
        """Inform the user when an inter-batch pacing wait crosses the
        notice threshold. Best-effort; logs and continues on failure."""
        try:
            await self.send(
                chat_id,
                f"(More images coming — pausing ~{_format_wait(wait_s)} "
                f"for Signal rate limit, batch {next_batch_idx}/{total_batches}.)",
            )
        except Exception as e:
            logger.warning("Signal: failed to send pacing notice: %s", e)

    async def send_files(
        self,
        chat_id: str,
        file_paths: List[str],
        caption: str = "",
    ) -> SendResult:
        """Send local files through the already-connected signal-ts session."""
        await self._stop_typing_indicator(chat_id)
        attachments: list[str] = []
        for raw_path in file_paths:
            path = Path(raw_path).expanduser()
            try:
                size = path.stat().st_size
            except OSError:
                logger.warning("Signal: attachment does not exist: %s", path)
                continue
            if size > SIGNAL_MAX_ATTACHMENT_SIZE:
                logger.warning("Signal: attachment too large (%d bytes): %s", size, path)
                continue
            attachments.append(str(path.resolve()))
        if not attachments:
            return SendResult(success=False, error="No valid Signal attachments")

        plain_text, text_styles = self._markdown_to_signal(caption)
        base_params: Dict[str, Any] = {"account": self.account}
        if chat_id.startswith("group:"):
            base_params["groupId"] = chat_id[6:]
        else:
            base_params["recipient"] = [await self._resolve_recipient(chat_id)]

        scheduler = get_scheduler()
        batches = [
            attachments[index:index + SIGNAL_MAX_ATTACHMENTS_PER_MSG]
            for index in range(0, len(attachments), SIGNAL_MAX_ATTACHMENTS_PER_MSG)
        ]
        for index, batch in enumerate(batches):
            count = len(batch)
            estimated = scheduler.estimate_wait(count)
            if estimated >= SIGNAL_BATCH_PACING_NOTICE_THRESHOLD:
                await self._notify_batch_pacing(
                    chat_id, index + 1, len(batches), estimated
                )
            params = {
                **base_params,
                "message": plain_text if index == 0 else "",
                "attachments": batch,
            }
            if index == 0 and plain_text and text_styles:
                params["textStyle" if len(text_styles) == 1 else "textStyles"] = (
                    text_styles[0] if len(text_styles) == 1 else text_styles
                )

            delivered = False
            for attempt in range(1, SIGNAL_RATE_LIMIT_MAX_ATTEMPTS + 1):
                await scheduler.acquire(count)
                started = time.monotonic()
                try:
                    result = await self._rpc(
                        "send",
                        params,
                        raise_on_rate_limit=True,
                        timeout=_signal_send_timeout(count),
                    )
                except SignalRateLimitError as exc:
                    scheduler.feedback(exc.retry_after, count)
                    if attempt >= SIGNAL_RATE_LIMIT_MAX_ATTEMPTS:
                        return SendResult(
                            success=False,
                            error=f"Signal rate limit exhausted on attachment batch {index + 1}",
                        )
                    continue
                if result is None:
                    if attempt >= SIGNAL_RATE_LIMIT_MAX_ATTEMPTS:
                        return SendResult(
                            success=False,
                            error=f"Signal attachment batch {index + 1} failed",
                        )
                    await asyncio.sleep(2.0 ** attempt)
                    continue
                success, error = self._validate_send_result(result)
                if not success:
                    return SendResult(success=False, error=error, raw_response=result)
                self._track_sent_timestamp(result)
                await scheduler.report_rpc_duration(
                    time.monotonic() - started, count
                )
                delivered = True
                break
            if not delivered:
                return SendResult(
                    success=False,
                    error=f"Signal attachment batch {index + 1} was not delivered",
                )

        return SendResult(success=True, message_id=None)

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        """Send an image. Supports http(s):// and file:// URLs."""
        await self._stop_typing_indicator(chat_id)

        # Resolve image to local path
        if image_url.startswith("file://"):
            file_path = unquote(image_url[7:])
        else:
            # Download remote image to cache
            try:
                file_path = await cache_image_from_url(image_url)
            except Exception as e:
                logger.warning("Signal: failed to download image: %s", e)
                return SendResult(success=False, error=str(e))

        if not file_path or not Path(file_path).exists():
            return SendResult(success=False, error="Image file not found")

        # Validate size
        file_size = Path(file_path).stat().st_size
        if file_size > SIGNAL_MAX_ATTACHMENT_SIZE:
            return SendResult(success=False, error=f"Image too large ({file_size} bytes)")

        params: Dict[str, Any] = {
            "account": self.account,
            "message": caption or "",
            "attachments": [file_path],
        }

        if chat_id.startswith("group:"):
            params["groupId"] = chat_id[6:]
        else:
            params["recipient"] = [await self._resolve_recipient(chat_id)]

        result = await self._rpc("send", params)
        if result is not None:
            success, err_msg = self._validate_send_result(result)
            if not success:
                return SendResult(success=False, error=err_msg, raw_response=result)
            self._track_sent_timestamp(result)
            return SendResult(success=True)
        return SendResult(success=False, error="RPC send with attachment failed")

    async def _send_attachment(
        self,
        chat_id: str,
        file_path: str,
        media_label: str,
        caption: Optional[str] = None,
    ) -> SendResult:
        """Send any file as a Signal attachment via RPC.

        Shared implementation for send_document, send_image_file, send_voice,
        and send_video — avoids duplicating the validation/routing/RPC logic.
        """
        await self._stop_typing_indicator(chat_id)

        try:
            file_size = Path(file_path).stat().st_size
        except FileNotFoundError:
            return SendResult(success=False, error=f"{media_label} file not found: {file_path}")

        if file_size > SIGNAL_MAX_ATTACHMENT_SIZE:
            return SendResult(success=False, error=f"{media_label} too large ({file_size} bytes)")

        params: Dict[str, Any] = {
            "account": self.account,
            "message": caption or "",
            "attachments": [file_path],
        }

        if chat_id.startswith("group:"):
            params["groupId"] = chat_id[6:]
        else:
            params["recipient"] = [await self._resolve_recipient(chat_id)]

        result = await self._rpc("send", params)
        if result is not None:
            success, err_msg = self._validate_send_result(result)
            if not success:
                return SendResult(success=False, error=err_msg, raw_response=result)
            self._track_sent_timestamp(result)
            return SendResult(success=True)
        return SendResult(success=False, error=f"RPC send {media_label.lower()} failed")

    async def send_document(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str] = None,
        filename: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        """Send a document/file attachment."""
        return await self._send_attachment(chat_id, file_path, "File", caption)

    async def send_image_file(
        self,
        chat_id: str,
        image_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        """Send a local image file as a native Signal attachment.

        Called by the gateway media delivery flow when MEDIA: tags containing
        image paths are extracted from agent responses.
        """
        return await self._send_attachment(chat_id, image_path, "Image", caption)

    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        """Send an audio file as a Signal attachment.

        Signal does not distinguish voice messages from file attachments at
        the API level, so this routes through the same RPC send path.
        """
        return await self._send_attachment(chat_id, audio_path, "Audio", caption)

    async def send_video(
        self,
        chat_id: str,
        video_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        """Send a video file as a Signal attachment."""
        return await self._send_attachment(chat_id, video_path, "Video", caption)

    # ------------------------------------------------------------------
    # Typing Indicators
    # ------------------------------------------------------------------

    async def _stop_typing_indicator(self, chat_id: str) -> None:
        """Stop a typing indicator loop for a chat."""
        task = self._typing_tasks.pop(chat_id, None)
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Send an explicit stop-typing RPC so the recipient's device drops the
        # indicator immediately instead of waiting for Signal's ~5s built-in
        # timeout.  Failures are best-effort — the backoff state must still be
        # cleared so the next agent turn starts clean.
        try:
            params: Dict[str, Any] = {"account": self.account}
            if chat_id.startswith("group:"):
                params["groupId"] = chat_id[6:]
            else:
                params["recipient"] = [await self._resolve_recipient(chat_id)]
            params["stop"] = True
            await self._rpc(
                "sendTyping",
                params,
                rpc_id="typing-stop",
                log_failures=False,
            )
        except Exception:
            # Best-effort: any RPC failure (or recipient-resolution failure)
            # must not prevent backoff cleanup.
            pass

        self._typing_failures.pop(chat_id, None)
        self._typing_skip_until.pop(chat_id, None)

    async def stop_typing(self, chat_id: str) -> None:
        """Public interface for stopping typing — called by base adapter's
        _keep_typing finally block to clean up platform-level typing tasks."""
        await self._stop_typing_indicator(chat_id)

    # ------------------------------------------------------------------
    # Reactions
    # ------------------------------------------------------------------

    async def send_reaction(
        self,
        chat_id: str,
        emoji: str,
        target_author: str,
        target_timestamp: int,
    ) -> bool:
        """Send a reaction emoji through the persistent Signal runtime.

        Args:
            chat_id: The chat (phone number or "group:<id>")
            emoji: Reaction emoji string (e.g. "👀", "✅")
            target_author: Phone number / UUID of the message author
            target_timestamp: Signal timestamp (ms) of the message to react to
        """
        params: Dict[str, Any] = {
            "account": self.account,
            "emoji": emoji,
            "targetAuthor": target_author,
            "targetTimestamp": target_timestamp,
        }

        if chat_id.startswith("group:"):
            params["groupId"] = chat_id[6:]
        else:
            params["recipient"] = [chat_id]

        result = await self._rpc("sendReaction", params)
        if result is not None:
            return True
        logger.debug("Signal: sendReaction failed (chat=%s, emoji=%s)", chat_id[:20], emoji)
        return False

    async def remove_reaction(
        self,
        chat_id: str,
        target_author: str,
        target_timestamp: int,
    ) -> bool:
        """Remove a reaction by sending an empty-string emoji."""
        params: Dict[str, Any] = {
            "account": self.account,
            # Signal's protocol requires the original emoji on a removal.
            # This adapter uses 👀 as its only transient lifecycle reaction.
            "emoji": "👀",
            "targetAuthor": target_author,
            "targetTimestamp": target_timestamp,
            "remove": True,
        }

        if chat_id.startswith("group:"):
            params["groupId"] = chat_id[6:]
        else:
            params["recipient"] = [chat_id]

        result = await self._rpc("sendReaction", params)
        return result is not None

    # ------------------------------------------------------------------
    # Processing Lifecycle Hooks (reactions as progress indicators)
    # ------------------------------------------------------------------

    def _extract_reaction_target(self, event: MessageEvent) -> Optional[tuple]:
        """Extract (target_author, target_timestamp) from a MessageEvent.

        Returns None if the event doesn't carry the raw Signal envelope data
        needed for sendReaction.
        """
        raw = event.raw_message
        if not isinstance(raw, dict):
            return None
        author = raw.get("sender")
        ts = raw.get("timestamp_ms")
        if not author or not ts:
            return None
        return (author, ts)

    def _reactions_enabled(self, event: "MessageEvent" = None) -> bool:
        """Check if message reactions are enabled for this event.

        Two gates:
        1. ``platforms.signal.extra.reactions`` in config.yaml.
        2. DM allowlist — if ``allow_from`` is set, only react to
           messages from senders in that list.  This prevents unauthorized
           contacts from seeing the 👀 reaction (which fires before run.py's
           auth gate and would otherwise reveal that a bot is listening).
        """
        if not self.reactions_enabled:
            return False
        if event is not None:
            sender = getattr(getattr(event, "source", None), "user_id", None)
            if sender and "*" not in self.dm_allow_from and sender not in self.dm_allow_from:
                return False
        return True

    async def on_processing_start(self, event: MessageEvent) -> None:
        """React with 👀 when processing begins."""
        if not self._reactions_enabled(event):
            return
        target = self._extract_reaction_target(event)
        if target:
            await self.send_reaction(event.source.chat_id, "👀", *target)

    async def on_processing_complete(self, event: MessageEvent, outcome: "ProcessingOutcome") -> None:
        """Swap the 👀 reaction for ✅ (success) or ❌ (failure).

        On CANCELLED we leave the 👀 in place — no terminal outcome means
        the reaction should keep reflecting "in progress" (matches Telegram).
        """
        if not self._reactions_enabled(event):
            return
        if outcome == ProcessingOutcome.CANCELLED:
            return
        target = self._extract_reaction_target(event)
        if not target:
            return
        chat_id = event.source.chat_id
        # Remove the in-progress reaction, then add the final one
        await self.remove_reaction(chat_id, *target)
        if outcome == ProcessingOutcome.SUCCESS:
            await self.send_reaction(chat_id, "✅", *target)
        elif outcome == ProcessingOutcome.FAILURE:
            await self.send_reaction(chat_id, "❌", *target)

    # ------------------------------------------------------------------
    # Chat Info
    # ------------------------------------------------------------------

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Get information about a chat/contact."""
        if chat_id.startswith("group:"):
            return {
                "name": chat_id,
                "type": "group",
                "chat_id": chat_id,
            }

        # Try to resolve contact name
        result = await self._rpc("getContact", {
            "account": self.account,
            "contactAddress": chat_id,
        })

        name = chat_id
        if result and isinstance(result, dict):
            name = result.get("name") or result.get("profileName") or chat_id

        return {
            "name": name,
            "type": "dm",
            "chat_id": chat_id,
        }
