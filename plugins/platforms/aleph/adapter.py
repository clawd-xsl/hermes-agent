"""Aleph platform adapter: the iOS app as a Hermes messaging channel.

A direct conversation between one person and the agent, spoken over Aleph's own WebSocket protocol
(Aleph repo, ``docs/protocol.md``): the main chat (``main``, the home channel) and side chats the person
opens beside it, each a DM with a ``chat_id`` of its own, so a Hermes session of its own. Everything
that changes a thread is logged, all chats in one log with one ``seq``, so a phone that was offline
catches up; drafts and typing are live only. Every event about a chat names it (``chat``).

Map onto the base adapter:

- inbound ``message.send`` -> ``handle_message`` (idempotent by the phone's message id), echoed as
  ``message.new``;
- ``send`` / ``edit_message`` / ``delete_message`` -> ``message.new`` / ``.edit`` / ``.delete``;
- ``send_draft`` -> live ``draft`` (whole text); the next text ``send`` carries its ``draft_id``;
- ``send_typing`` / ``stop_typing`` / status text -> live ``typing``;
- ``_send_exec_approval_prompt`` / ``send_clarify`` -> ``approval.request`` / ``clarify.request``;
- ``send_image_file`` / ``send_document`` / ``send_voice`` / ``send_video`` -> attachments;
- ``on_processing_start`` -> ``receipt`` (the turn that takes the person's messages has started);
- ``chat.create`` / ``chat.delete`` -> a side chat opened, or ended (``/new``) and dropped;
- ``on_session_title`` -> ``chat.update`` (a side chat's session titles).

Everything to and from the phone is end-to-end encrypted with Noise (``noise.py``): a device pairs
once by scanning a QR code (``hermes aleph pair``, then the ``/aleph/v1/pair`` handshake) and is known
by its static key from then on. Files are encrypted with keys of their own (``files.py``). Logged events
that notify are pushed through APNs, content-free, to every device with a push token that is not
active (protocol "Push"); pushing never holds up logging or delivery.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import math
import mimetypes
import os
import re
import secrets
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from gateway.config import Platform, PlatformConfig
from gateway.platforms._shared import extra_or_secret
from gateway.platforms.base import (
    BasePlatformAdapter, ExecApprovalPrompt, MessageEvent, MessageType, SendResult, cache_media_bytes_async)
from gateway.platforms.helpers import MessageDeduplicator

from plugins.platforms.aleph import noise
from plugins.platforms.aleph import activity, reactions, replies
from plugins.platforms.aleph.profile import DEFAULT_NAME, Avatar
from plugins.platforms.aleph.files import TicketBook, decrypt_file, encrypt_file, file_url
from plugins.platforms.aleph.noise import b64url, from_b64url
from plugins.platforms.aleph.push import (
    BODY_MESSAGE, ENVIRONMENTS, ApnsConfig, ApnsSender, build_payload, push_body_for)
from plugins.platforms.aleph.schedule import Schedule, ScheduleError
from plugins.platforms.aleph.server import (
    BUSY, INVALID_PARAMS, METHOD_NOT_FOUND, AlephServer, Connection, RpcError, event_frame)
from plugins.platforms.aleph.store import AlephStore, Chat, Device, LoggedEvent, load_or_create_server_key

logger = logging.getLogger(__name__)

PLATFORM_NAME = "aleph"
HISTORY_PAGE_DEFAULT = 200
HISTORY_PAGE_MAX = 500
# Pages being read at once across connections; a load past this is told to try again (4002).
MAX_HISTORY_LOADS = 4
SCHEDULE_CALLS = frozenset({"schedule.list", "schedule.pause", "schedule.resume", "schedule.run",
                            "schedule.delete"})
CHAT_ID = "main"  # the main chat: the home channel, always there
CHAT_NAME = "Aleph"
MAX_CHAT_ID = 128
MAX_CHAT_TITLE = 200
# A side chat nobody has written in for this long is deleted (``side_chat_idle_days``; 0 keeps them).
DEFAULT_SIDE_CHAT_IDLE_DAYS = 30
IDLE_SWEEP_SECONDS = 3600
# Every paired device is the owner's: every inbound message is from this one sender.
OWNER_ID = "owner"
# The agent's name until it picks its own with the ``aleph_set_profile`` tool (protocol "Agent profile").
AGENT_NAME = DEFAULT_NAME
OWNER_NAME = "Owner"
DEFAULT_PORT = 8650
DEFAULT_HOST = "127.0.0.1"
DEFAULT_HISTORY_LIMIT = 20000
# Generous: Aleph renders long Markdown itself, and drafts stream the whole reply anyway.
MAX_MESSAGE_LENGTH = 32000
MAX_TEXT_CHARS = 64000
MAX_ATTACHMENTS = 20
MAX_UPLOAD_BYTES = 200 * 1024 * 1024
MAX_PLACE_TEXT = 200
# A text file this small is put into the message itself, as other channels do; larger ones and
# binary files reach the agent as a path it reads.
MAX_INLINE_TEXT_BYTES = 100 * 1024
# What the phone may tell about an upload (layout before download), within sane bounds.
MAX_MEDIA_SIDE = 100_000
MAX_MEDIA_SECONDS = 7 * 24 * 3600
# How often the gateway reloads paired keys and the pending pairing written by the CLI (another
# process): a revoked device is cut off, and a new pairing becomes reachable, within this.
DEVICE_POLL_SECONDS = 1.0
LAST_SEEN_INTERVAL = 60.0
_PUSH_TOKEN = re.compile(r"^[0-9a-fA-F]{16,400}$")

# ExecApprovalPrompt styles -> protocol styles.
_APPROVAL_STYLES = {"primary": "primary", "danger": "destructive"}


def now_iso() -> str:
    """ISO 8601 with the local offset and milliseconds (protocol "Frames")."""
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def check_aleph_requirements() -> bool:
    try:
        import aiohttp
    except ImportError:
        return False
    return True


def setting(config_extra: Optional[dict], key: str, env: str, default: Any = "") -> str:
    return str(extra_or_secret(config_extra or {}, key, env, default) or "").strip()


def resolve_state_dir(extra: Optional[dict]) -> Path:
    state_dir = setting(extra, "state_dir", "ALEPH_STATE_DIR")
    if state_dir:
        return Path(state_dir).expanduser()
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "aleph"


def apns_config(extra: Optional[dict]) -> Optional[ApnsConfig]:
    return ApnsConfig.from_settings(
        key_file=setting(extra, "apns_key_file", "ALEPH_APNS_KEY_FILE"),
        key_id=setting(extra, "apns_key_id", "ALEPH_APNS_KEY_ID"),
        team_id=setting(extra, "apns_team_id", "ALEPH_APNS_TEAM_ID"),
        topic=setting(extra, "apns_topic", "ALEPH_APNS_TOPIC"))


def _file_kind(cached_kind: str, mime: str, name: str) -> str:
    """Protocol attachment kind: image, video, audio, voice or file."""
    if cached_kind == "image":
        return "image"
    if cached_kind == "video":
        return "video"
    if cached_kind == "audio":
        return "voice" if (mime in ("audio/ogg", "audio/opus") or "voice" in name.lower()) else "audio"
    return "file"


def upload_limit() -> int:
    """The largest file the phone may upload: 200 MB, or less when the gateway's own inbound media cap
    (``gateway.max_inbound_media_bytes``, 128 MB by default) is lower, since the file must fit it to
    reach the agent."""
    from gateway.platforms.base import get_inbound_media_max_bytes
    try:
        gateway_cap = int(get_inbound_media_max_bytes())
    except Exception:
        gateway_cap = 0
    return min(MAX_UPLOAD_BYTES, gateway_cap) if gateway_cap > 0 else MAX_UPLOAD_BYTES


def _place_text(value: Any, field: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise RpcError(INVALID_PARAMS, f"location.{field} must be a string")
    text = " ".join(value.split())  # one line
    if len(text) > MAX_PLACE_TEXT:
        raise RpcError(INVALID_PARAMS, f"location.{field} is longer than {MAX_PLACE_TEXT} characters")
    return text or None


def clean_location(raw: Any) -> Optional[dict[str, Any]]:
    """``message.send``'s ``location``: ``{latitude, longitude, name?, address?}``, validated."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise RpcError(INVALID_PARAMS, "location must be an object")
    point: dict[str, Any] = {}
    for field, bound in (("latitude", 90.0), ("longitude", 180.0)):
        value = raw.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) \
                or not -bound <= value <= bound:
            raise RpcError(INVALID_PARAMS, f"location.{field} must be a number from {-bound:g} to {bound:g}")
        point[field] = float(value)
    for field in ("name", "address"):
        text = _place_text(raw.get(field), field)
        if text:
            point[field] = text
    return point


def location_note(point: dict[str, Any], text: str = "") -> str:
    """The note a shared pin reaches the agent as, the way other channels word it, then the message's
    own text."""
    from urllib.parse import quote
    lat, lon = point["latitude"], point["longitude"]
    lines = ["[The user shared a location pin.]"]
    if point.get("name"):
        lines.append(f"Place: {point['name']}")
    if point.get("address"):
        lines.append(f"Address: {point['address']}")
    link = f"https://maps.apple.com/?ll={lat},{lon}" + (f"&q={quote(point['name'])}" if point.get("name") else "")
    lines += [f"latitude: {lat}", f"longitude: {lon}", f"Map: {link}"]
    note = "\n".join(lines)
    return f"{note}\n\n{text}" if text.strip() else note


def _upload_hints(upload: dict[str, Any]) -> dict[str, Any]:
    """``width``/``height``/``duration`` the phone gave for an upload, keeping only sane values."""
    hints: dict[str, Any] = {}
    for field in ("width", "height"):
        value = upload.get(field)
        if isinstance(value, int) and not isinstance(value, bool) and 0 < value <= MAX_MEDIA_SIDE:
            hints[field] = value
    duration = upload.get("duration")
    if isinstance(duration, (int, float)) and not isinstance(duration, bool) and math.isfinite(duration) \
            and 0 <= duration <= MAX_MEDIA_SECONDS:
        hints["duration"] = round(float(duration), 3)
    return hints


def _inline_text(path: str, name: str, mime: str, kind: str) -> Optional[str]:
    """A small text file's content, to put into the message; None for anything else."""
    from gateway.platforms.base import _TEXT_INJECT_EXTENSIONS
    if kind != "file":
        return None
    ext = os.path.splitext(name or path)[1].lower()
    if ext not in _TEXT_INJECT_EXTENSIONS and not mime.startswith("text/"):
        return None
    try:
        if os.path.getsize(path) > MAX_INLINE_TEXT_BYTES:
            return None
        with open(path, "rb") as handle:
            return handle.read().decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _media_message_type(kind: str) -> MessageType:
    return {"image": MessageType.PHOTO, "video": MessageType.VIDEO, "audio": MessageType.AUDIO,
            "voice": MessageType.VOICE}.get(kind, MessageType.DOCUMENT)


def _image_size(path: str) -> Optional[tuple[int, int]]:
    try:
        from PIL import Image
        with Image.open(path) as image:
            return int(image.width), int(image.height)
    except Exception:
        return None


@dataclass(frozen=True)
class ReplaySnapshot:
    latest: int
    reset: bool
    events: list[LoggedEvent]
    # On a reset: every side chat that exists, as its ``chat.new`` payload.
    chats: Optional[list[dict[str, Any]]] = None


class AlephAdapter(BasePlatformAdapter):
    """The Aleph channel; see the module docstring for the mapping."""

    supports_code_blocks = True
    supports_status_text = True
    # Asks the gateway to tag tool-progress sends (``metadata["_tool_progress"]``) so they are
    # logged as ``kind: tool`` rows instead of agent text.
    MARKS_TOOL_PROGRESS = True
    MAX_MESSAGE_LENGTH = MAX_MESSAGE_LENGTH

    def __init__(self, config: PlatformConfig) -> None:
        extra = config.extra if isinstance(config.extra, dict) else {}
        config.extra = extra
        # A paired device's token authenticates the one owner; the gateway's allowlist admits them.
        extra.setdefault("allow_from", [OWNER_ID])
        super().__init__(config, Platform(PLATFORM_NAME))
        self.chat_id = CHAT_ID
        self.host = str(extra_or_secret(extra, "host", "ALEPH_HOST", DEFAULT_HOST) or DEFAULT_HOST).strip()
        self.port = self._int_setting(extra, "port", "ALEPH_PORT", DEFAULT_PORT)
        self.history_limit = self._int_setting(extra, "history_limit", "ALEPH_HISTORY_LIMIT", DEFAULT_HISTORY_LIMIT)
        self.side_chat_idle_days = self._int_setting(extra, "side_chat_idle_days", "ALEPH_SIDE_CHAT_IDLE_DAYS",
                                                     DEFAULT_SIDE_CHAT_IDLE_DAYS)
        self.state_dir = resolve_state_dir(extra)
        self.apns_config = apns_config(extra)
        self._push: Optional[ApnsSender] = None
        self._push_tasks: set[asyncio.Task] = set()
        self._push_problems_reported: set[tuple[str, str]] = set()
        self._device_watch: Optional[asyncio.Task] = None
        self._idle_sweep: Optional[asyncio.Task] = None
        self._schedule: Optional[Schedule] = None
        self._history_loads = 0
        # The side chats that exist (``main`` always does), and those deleted while running: what the
        # gateway still sends them (the /new that ended the session, a late reply) goes nowhere.
        self._chats: dict[str, Chat] = {}
        self._deleted_chats: set[str] = set()
        self._chat_lock = asyncio.Lock()
        self._last_seen: dict[str, float] = {}
        self.static: Optional[noise.KeyPair] = None
        # What the handshakes may consult before anyone is authenticated: memory only.
        self._device_keys: dict[bytes, Device] = {}
        self._pending: Optional[tuple[bytes, float]] = None
        self._tickets: Optional[TicketBook] = None
        # The agent's profile (name and avatar), loaded at connect; only the aleph tools change it.
        self._profile_name = AGENT_NAME
        self._avatar: Optional[dict[str, Any]] = None  # the avatar's file description, without a url
        self._avatar_digest = ""
        self._avatar_updated_at = ""
        self._profile_lock = asyncio.Lock()
        # Activity (protocol "Activity"): the message each chat's turn answers, its open steps (in
        # start order), and the status line they give.
        self._turn_message: dict[str, Optional[str]] = {}
        self._open_steps: dict[str, list[tuple[str, dict[str, Any]]]] = {}  # chat -> [(raw tool, step)]
        self._activity_status: dict[str, Optional[str]] = {}
        # The person's reactions since their last message in each chat: chat -> {message id -> emoji},
        # oldest first. They reach the agent in front of the next message there (protocol "Reactions").
        self._pending_reactions: dict[str, dict[str, str]] = {}
        self._dedup = MessageDeduplicator(max_size=5000, ttl_seconds=3600)
        self._store: Optional[AlephStore] = None
        self._server: Optional[AlephServer] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self.append_lock = asyncio.Lock()
        self._intake_lock = asyncio.Lock()
        # Live state replayed to a phone that connects mid-turn.
        self._draft_wire_ids: dict[int, str] = {}  # stream consumer draft id -> wire draft id
        self._open_draft: dict[str, tuple[str, str]] = {}  # chat -> (wire draft id, latest text)
        self._typing: dict[str, Optional[str]] = {}  # chat -> status while the agent works
        self._draft_prefix = secrets.token_hex(3)
        # Cards awaiting an answer.
        self._approvals: dict[str, dict[str, Any]] = {}
        self._clarifies: dict[str, dict[str, Any]] = {}
        # Receipt watermarks, per chat: the newest of the person's messages handed to the gateway, and
        # the newest a receipt covered.
        self._last_user_message: dict[str, str] = {}
        self._last_receipt: dict[str, str] = {}

    @staticmethod
    def _int_setting(extra: dict, key: str, env: str, default: int) -> int:
        raw = extra_or_secret(extra, key, env, default)
        try:
            return int(raw)
        except (TypeError, ValueError):
            logger.warning("Aleph: %s=%r is not a number; using %d", env, raw, default)
            return default

    # ── lifecycle ──────────────────────────────────────────────────────────────────────────────

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        self._loop = asyncio.get_running_loop()
        try:
            self._store = await asyncio.to_thread(AlephStore, self.state_dir, history_limit=self.history_limit)
        except Exception as exc:
            logger.error("Aleph: cannot open the event log in %s: %s", self.state_dir, exc)
            self._set_fatal_error("storage", f"cannot open {self.state_dir}: {exc}", retryable=False)
            return False
        try:
            self.static = noise.KeyPair.from_private(
                await asyncio.to_thread(load_or_create_server_key, self.state_dir))
        except (OSError, ValueError) as exc:
            logger.error("Aleph: cannot read the server key in %s: %s", self.state_dir, exc)
            self._set_fatal_error("storage", f"cannot read the server key: {exc}", retryable=False)
            await asyncio.to_thread(self._store.close)
            self._store = None
            return False
        self._tickets = TicketBook(self._store.files_dir / "uploads")
        await self._reload_devices()
        self._chats = {chat.id: chat for chat in await asyncio.to_thread(self._store.chats)}
        self._last_user_message, self._last_receipt = {}, {}
        for chat in (CHAT_ID, *self._chats):
            for marks, key in ((self._last_user_message, "last_user_message"), (self._last_receipt, "last_receipt")):
                value = await asyncio.to_thread(self._store.meta, f"{key}:{chat}")
                if value:
                    marks[chat] = value
        await self._load_profile()
        try:
            pending = json.loads(await asyncio.to_thread(self._store.meta, "pending_reactions") or "{}")
            self._pending_reactions = pending if isinstance(pending, dict) else {}
        except ValueError:
            self._pending_reactions = {}
        self._server = AlephServer(self, host=self.host, port=self.port, static=self.static)
        try:
            await self._server.start()
        except OSError as exc:
            logger.error("Aleph: cannot listen on %s:%d: %s", self.host, self.port, exc)
            self._set_fatal_error("listen", f"cannot listen on {self.host}:{self.port}: {exc}", retryable=True)
            await asyncio.to_thread(self._store.close)
            self._store = self._server = None
            return False
        port = self._server.bound_port or self.port
        logger.info("Aleph: listening on ws://%s:%d/aleph/v1/ws", self.host, port)
        self._push = ApnsSender.create(self.apns_config)
        devices = await asyncio.to_thread(self._store.devices)
        logger.info("Aleph: %d paired device(s); push %s", len(devices),
                    f"on ({self.apns_config.topic})" if self._push else "off")
        self._device_watch = asyncio.create_task(self._watch_devices())
        self._schedule = Schedule(lambda: self._live("schedule.changed", {}))
        self._schedule.start()
        if self.side_chat_idle_days > 0:
            self._idle_sweep = asyncio.create_task(self._sweep_idle_chats())
        self._mark_connected(listener_base=f"http://{self.host}:{port}")
        return True

    async def disconnect(self) -> None:
        server, store, push, watch = self._server, self._store, self._push, self._device_watch
        sweep, self._idle_sweep = self._idle_sweep, None
        schedule, self._schedule = self._schedule, None
        if schedule is not None:
            await schedule.stop()
        self._server = self._store = self._push = self._device_watch = None
        for task in (watch, sweep):
            if task is not None:
                task.cancel()
        for task in list(self._push_tasks):
            task.cancel()
        if push is not None:
            await push.close()
        if server is not None:
            await server.stop()
        if store is not None:
            await asyncio.to_thread(store.close)
        self._mark_disconnected()

    @property
    def store(self) -> AlephStore:
        if self._store is None:
            raise RuntimeError("Aleph adapter is not connected")
        return self._store

    # ── fan-out ────────────────────────────────────────────────────────────────────────────────

    async def _log(self, event_type: str, payload: dict[str, Any], *,
                   message: Optional[dict[str, Any]] = None, push_body: Optional[str] = None) -> LoggedEvent:
        """Append one logged event and broadcast it, in seq order; push it if it notifies (or, with
        ``push_body``, regardless of its kind)."""
        async with self.append_lock:
            event = await asyncio.to_thread(self.store.append, event_type, payload, message=message)
            self._broadcast(event)
        self._schedule_push(event, push_body)
        return event

    def _broadcast(self, event: LoggedEvent) -> None:
        """Under ``append_lock``: a logged event to every connected phone."""
        if self._server is not None:
            self._server.broadcast(event_frame(event.type, event.payload, event.seq))

    # ── devices ────────────────────────────────────────────────────────────────────────────────

    def known_device(self, public_key: Optional[bytes]) -> Optional[Device]:
        """The paired device a handshake's static key belongs to, or ``None`` (unknown or revoked).
        From memory: called before the peer is authenticated."""
        return self._device_keys.get(bytes(public_key)) if public_key else None

    async def device_seen(self, device: Device) -> None:
        """After a handshake: record when the device was last seen (at most once a minute)."""
        now = time.time()
        if self._store is not None and now - self._last_seen.get(device.id, 0.0) >= LAST_SEEN_INTERVAL:
            self._last_seen[device.id] = now
            await asyncio.to_thread(self.store.touch_device, device.id, now)

    def pending_pairing(self) -> Optional[tuple[bytes, float]]:
        """(secret, expires_at) of the pending pairing, from memory."""
        pending = self._pending
        return pending if pending is not None and pending[1] > time.time() else None

    async def _reload_devices(self) -> None:
        store = self._store
        if store is None:
            return
        devices = await asyncio.to_thread(store.devices)
        self._device_keys = {device.public_key: device for device in devices}
        self._pending = await asyncio.to_thread(store.pending_pairing)

    async def complete_pairing(self, secret: bytes, public_key: bytes, *, name: str,
                               model: str) -> Optional[dict[str, Any]]:
        """The pairing handshake's first message decrypted with ``secret``: record the device, spend
        the secret, and tell every device in the thread. ``None`` if the secret was replaced meanwhile."""
        if self._store is None:
            return None
        device = await asyncio.to_thread(self.store.complete_pairing, secret, public_key, name=name, model=model)
        if device is None:
            return None
        self._pending = None
        self._device_keys[device.public_key] = device
        logger.info("Aleph: paired %s (%s, %s)", device.name, device.model or "unknown model", device.id)
        message_id = str(uuid.uuid4())
        text = f"New device paired: {device.name}"
        await self._log("message.new", {"chat": CHAT_ID, "id": message_id, "sender": "agent", "text": text,
                                        "attachments": [], "created_at": now_iso(), "kind": "system"},
                        message={"id": message_id, "sender": "agent", "kind": "system", "text": text,
                                 "attachments": []},
                        push_body=BODY_MESSAGE)
        return {"device_id": device.id, "agent": {"name": self._profile_name}}

    async def _watch_devices(self) -> None:
        """Pick up what the CLI (another process) changed: revoked devices lose their connections,
        and a new pairing becomes reachable."""
        while True:
            await asyncio.sleep(DEVICE_POLL_SECONDS)
            try:
                server = self._server
                if server is None or self._store is None:
                    return
                await self._reload_devices()
                if server.connections:
                    await server.close_devices({device.id for device in self._device_keys.values()})
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.debug("Aleph: device check failed", exc_info=True)

    async def _register_push(self, conn: Connection, params: dict[str, Any]) -> None:
        token, environment = params.get("token"), params.get("environment")
        if token is not None:
            if not isinstance(token, str) or not _PUSH_TOKEN.match(token):
                raise RpcError(INVALID_PARAMS, "token must be an APNs device token in hex, or null")
            if environment not in ENVIRONMENTS:
                raise RpcError(INVALID_PARAMS, "environment must be development or production")
            token = token.lower()
        await asyncio.to_thread(self.store.set_push_token, conn.device_id, token,
                                environment if token is not None else None)
        logger.info("Aleph: %s %s push (%s)", conn.device_id, "registered for" if token else "unregistered from",
                    environment if token else "none")

    def _report_push_config_problem(self, result) -> None:
        """A setup problem fails every push the same way: say so once (per topic and reason) at
        ERROR, with what to fix, and keep the rest quiet."""
        topic = self.apns_config.topic if self.apns_config else "?"
        key = (topic, result.reason)
        if key in self._push_problems_reported:
            logger.debug("Aleph: push failed again: %d %s", result.status, result.reason)
            return
        self._push_problems_reported.add(key)
        logger.error("Aleph: push is not working for %s: APNs %d %s: %s. Pushes keep failing the same "
                     "way until it is fixed (reported once).", topic, result.status, result.reason,
                     result.config_problem)

    # ── push ───────────────────────────────────────────────────────────────────────────────────

    def _schedule_push(self, event: LoggedEvent, body: Optional[str] = None) -> None:
        """Fire and forget: a slow or failing APNs never delays the log or the WebSocket."""
        if self._push is None:
            return
        body = body or push_body_for(event.type, event.payload)
        if body is None:
            return
        task = asyncio.create_task(self._push_event(event, body))
        self._push_tasks.add(task)
        task.add_done_callback(self._push_tasks.discard)

    async def _push_event(self, event: LoggedEvent, body: str) -> None:
        sender, store, server = self._push, self._store, self._server
        if sender is None or store is None:
            return
        try:
            devices = await asyncio.to_thread(store.devices)
            active = server.active_device_ids() if server is not None else set()
            targets = [d for d in devices if d.push_token and d.id not in active]
            if not targets:
                return
            payload = build_payload(body=body, seq=event.seq, chat=str(event.payload.get("chat") or CHAT_ID))
            results = await asyncio.gather(
                *(sender.send(d.push_token, d.push_environment or "production", payload) for d in targets),
                return_exceptions=True)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("Aleph: push for seq %d failed", event.seq, exc_info=True)
            return
        for device, result in zip(targets, results):
            if isinstance(result, BaseException):
                logger.warning("Aleph: push to %s failed: %s", device.id, result)
            elif result.token_is_dead:
                logger.info("Aleph: APNs rejected %s's push token (%d %s); dropping it",
                            device.id, result.status, result.reason)
                await asyncio.to_thread(store.drop_push_token, device.push_token)
            elif result.config_problem:
                self._report_push_config_problem(result)
            elif not result.ok:
                logger.warning("Aleph: push to %s failed: %d %s", device.id, result.status, result.reason)
            else:
                self._push_problems_reported.clear()  # fixed: a new breakage gets reported again
                logger.info("Aleph: pushed seq %d to %s (%s): APNs %d", event.seq, device.id,
                            device.push_environment or "production", result.status)

    def _live(self, event_type: str, payload: dict[str, Any]) -> None:
        """Broadcast a live-only event; safe from any thread (status text arrives off-loop)."""
        def emit() -> None:
            if self._server is not None:
                self._server.broadcast(event_frame(event_type, payload))
        self._on_loop(emit)

    def live_state_frames(self) -> list[str]:
        """What a phone connecting mid-turn needs to show the present, in every chat: typing, the open
        draft, and the steps still running."""
        frames = []
        for chat, status in self._typing.items():
            if self._exists(chat):
                frames.append(event_frame("typing", self._typing_payload(chat, True, status)))
        for chat, (wire_id, text) in self._open_draft.items():
            if self._exists(chat):
                frames.append(event_frame("draft", {"chat": chat, "draft_id": wire_id, "text": text}))
        for chat, open_steps in self._open_steps.items():
            if self._exists(chat):
                frames += [event_frame("activity", dict(step)) for _raw, step in open_steps]
        return frames

    async def replay_snapshot(self, after: int) -> ReplaySnapshot:
        """Under ``append_lock``: the events a phone that stored ``after`` is missing, or a reset.

        A reset replays nothing: the phone starts from the present (``latest``, the side chats) and
        pages older messages in with ``history.load``. It is the answer to a phone with nothing yet
        (``after`` 0), to one ahead of this log (a wiped or replaced server), and to one whose missing
        events were pruned for age."""
        latest = await asyncio.to_thread(self.store.latest)
        pruned = await asyncio.to_thread(self.store.pruned_through)
        if latest > 0 and (after == 0 or after > latest or after < pruned):
            chats = [{"chat": chat.id, "title": chat.title, "created_at": chat.created_at}
                     for chat in await asyncio.to_thread(self.store.chats)]
            return ReplaySnapshot(latest, True, [], chats)
        events = await asyncio.to_thread(self.store.events_after, after, until=latest) if latest > after else []
        return ReplaySnapshot(latest, False, events)

    async def load_history(self, params: dict[str, Any], conn: Optional[Connection]) -> dict[str, Any]:
        """``history.load {chat, before, limit}``: write one ``history.page`` into the connection's
        stream. Read and queued under ``append_lock``, so the page reflects every logged event queued
        on the connection before it and none after."""
        chat = self._chat_param(params)
        before, limit = params.get("before"), params.get("limit", HISTORY_PAGE_DEFAULT)
        if isinstance(before, bool) or not isinstance(before, int) or before < 1:
            raise RpcError(INVALID_PARAMS, "before must be a seq (a positive integer)")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= HISTORY_PAGE_MAX:
            raise RpcError(INVALID_PARAMS, f"limit must be from 1 to {HISTORY_PAGE_MAX}")
        if conn is None:
            raise RpcError(INVALID_PARAMS, "history.load needs a connection")
        if self._history_loads >= MAX_HISTORY_LOADS:
            raise RpcError(BUSY, "busy loading history; try again shortly")
        self._history_loads += 1
        try:
            async with self.append_lock:
                messages, older = await asyncio.to_thread(self.store.history, chat, before, limit)
                conn.enqueue(event_frame("history.page", {"chat": chat, "before": before, "messages": messages,
                                                          "next": older}))
        finally:
            self._history_loads -= 1
        return {}

    def agent_info(self) -> dict[str, Any]:
        """``hello``'s ``agent``: the profile as it is now, the avatar with a fresh download link."""
        info: dict[str, Any] = {"name": self._profile_name}
        model = self._configured_model()
        if model:
            info["model"] = model
        info["avatar"] = self._avatar_with_link()
        return info

    # ── the agent's profile ────────────────────────────────────────────────────────────────────

    async def _load_profile(self) -> None:
        store = self.store
        self._profile_name = await asyncio.to_thread(store.meta, "profile_name") or AGENT_NAME
        avatar_id = await asyncio.to_thread(store.meta, "profile_avatar")
        stored = await asyncio.to_thread(store.file, avatar_id) if avatar_id else None
        if stored is not None and os.path.isfile(stored.path):
            self._avatar = stored.description
            self._avatar_digest = await asyncio.to_thread(store.meta, "profile_avatar_digest") or ""
            self._avatar_updated_at = await asyncio.to_thread(store.meta, "profile_avatar_updated_at") or ""
        else:
            self._avatar, self._avatar_digest, self._avatar_updated_at = None, "", ""

    def _avatar_with_link(self) -> Optional[dict[str, Any]]:
        return self._with_links([self._avatar])[0] if self._avatar is not None else None

    def profile_view(self) -> dict[str, Any]:
        """What the agent sees of its profile (``aleph_get_profile``)."""
        avatar = None
        if self._avatar is not None:
            avatar = {"width": self._avatar.get("width"), "height": self._avatar.get("height"),
                      "updated_at": self._avatar_updated_at}
        return {"name": self._profile_name, "avatar": avatar}

    async def update_profile(self, *, name: Optional[str] = None, avatar: Optional[Avatar] = None,
                             reset_avatar: bool = False) -> dict[str, Any]:
        """Apply a validated change (on the gateway's loop) and log ``agent.profile`` if anything
        changed. Returns the profile view plus ``changed``."""
        async with self._profile_lock:
            store = self.store
            changed: list[str] = []
            old_avatar_id = self._avatar.get("id") if self._avatar is not None else None
            if name is not None and name != self._profile_name:
                await asyncio.to_thread(store.set_meta, "profile_name", name)
                self._profile_name = name
                changed.append("name")
            if avatar is not None and avatar.digest != self._avatar_digest:
                file_id = f"f_{uuid.uuid4().hex}"
                key, ciphertext = encrypt_file(avatar.jpeg)
                dest = store.files_dir / f"{file_id}.bin"
                await asyncio.to_thread(dest.write_bytes, ciphertext)
                description = {"id": file_id, "name": "avatar.jpg", "mime": "image/jpeg", "kind": "image",
                               "size": len(avatar.jpeg), "width": avatar.width, "height": avatar.height,
                               "key": b64url(key)}
                await asyncio.to_thread(store.add_file, description, str(dest), None)
                updated_at = now_iso()
                for key_name, value in (("profile_avatar", file_id), ("profile_avatar_digest", avatar.digest),
                                        ("profile_avatar_updated_at", updated_at)):
                    await asyncio.to_thread(store.set_meta, key_name, value)
                self._avatar, self._avatar_digest, self._avatar_updated_at = description, avatar.digest, updated_at
                changed.append("avatar")
            elif reset_avatar and self._avatar is not None:
                for key_name in ("profile_avatar", "profile_avatar_digest", "profile_avatar_updated_at"):
                    await asyncio.to_thread(store.set_meta, key_name, "")
                self._avatar, self._avatar_digest, self._avatar_updated_at = None, "", ""
                changed.append("avatar")
            if "avatar" in changed and old_avatar_id:
                await asyncio.to_thread(store.delete_file, old_avatar_id)
            if changed:
                # Not a message: logged so every device catches up, never pushed.
                await self._log("agent.profile", {"name": self._profile_name, "avatar": self._avatar_with_link(),
                                                  "changed": changed, "at": now_iso()})
                logger.info("Aleph: the agent changed its %s", " and ".join(changed))
            return {**self.profile_view(), "changed": changed}

    @staticmethod
    def _configured_model() -> str:
        try:
            from hermes_cli.config import load_config
            model = (load_config() or {}).get("model") or {}
            if isinstance(model, dict):
                return str(model.get("default") or model.get("model") or "")
            return str(model or "")
        except Exception:
            return ""

    # ── calls ──────────────────────────────────────────────────────────────────────────────────

    async def handle_call(self, method: str, params: dict[str, Any], conn: Optional[Connection] = None) -> Any:
        if method == "push.register":
            if conn is None or not conn.device_id:
                raise RpcError(INVALID_PARAMS, "push.register needs a paired device")
            await self._register_push(conn, params)
            return {}
        if method == "message.send":
            return {"seq": await self._accept_user_message(params)}
        if method == "chat.stop":
            await self._stop(self._chat_param(params))
            return {}
        if method == "chat.create":
            return {"seq": await self._create_chat(params)}
        if method == "chat.delete":
            return {"seq": await self._delete_chat_call(params)}
        if method == "approval.respond":
            await self._respond_approval(params)
            return {}
        if method == "clarify.respond":
            await self._respond_clarify(params)
            return {}
        if method == "file.ticket":
            return await self._file_ticket(params)
        if method == "history.load":
            return await self.load_history(params, conn)
        if method == "activity.list":
            return await self.list_activity(params)
        if method == "reaction.set":
            return {"seq": await self._user_reaction(params)}
        if method in SCHEDULE_CALLS:
            if self._schedule is None:
                raise RpcError(INVALID_PARAMS, "scheduled tasks are not available")
            try:
                return await self._schedule.call(method, params)
            except ScheduleError as exc:
                raise RpcError(INVALID_PARAMS, str(exc)) from exc
        if method == "ping":
            return {}
        raise RpcError(METHOD_NOT_FOUND, f"method not found: {method}")

    async def _accept_user_message(self, params: dict[str, Any]) -> int:
        chat = self._chat_param(params)
        message_id = params.get("id")
        if not isinstance(message_id, str) or not message_id.strip() or len(message_id) > 128:
            raise RpcError(INVALID_PARAMS, "id must be a non-empty string")
        text = params.get("text") or ""
        if not isinstance(text, str) or len(text) > MAX_TEXT_CHARS:
            raise RpcError(INVALID_PARAMS, "text must be a string")
        uploads = params.get("attachments") or []
        if not isinstance(uploads, list) or len(uploads) > MAX_ATTACHMENTS or not all(
                isinstance(item, dict) for item in uploads):
            raise RpcError(INVALID_PARAMS, "attachments must be a list of {ticket, key, name, mime}")
        reply_to = params.get("reply_to")
        if reply_to is not None and not isinstance(reply_to, str):
            raise RpcError(INVALID_PARAMS, "reply_to must be a message id or null")
        location = clean_location(params.get("location"))
        async with self._intake_lock:
            existing = await asyncio.to_thread(self.store.message, message_id)
            if existing is not None:
                if existing.sender != "user" or existing.chat != chat:
                    raise RpcError(INVALID_PARAMS, "id belongs to another message")
                # A resend after a lost answer: answer as the first time; hand it over if the first
                # attempt was logged but never reached the gateway.
                self._dedup.is_duplicate(message_id)
                if not existing.dispatched:
                    await self._dispatch_user_message(chat, message_id, existing.text, existing.attachments,
                                                      reply_to, await self._logged_location(existing.seq))
                return existing.seq
            if not text.strip() and not uploads and location is None:
                raise RpcError(INVALID_PARAMS, "a message needs text, attachments or a location")
            attachments = await self._take_uploads(uploads)
            self._dedup.is_duplicate(message_id)
            payload = {"chat": chat, "id": message_id, "sender": "user", "text": text,
                       "attachments": self._with_links(attachments)}
            if location is not None:
                payload["location"] = location
            quoted = await self._reply_snapshot(chat, reply_to)
            if quoted is not None:
                payload["reply_to"] = quoted
            payload["created_at"] = now_iso()
            event = await self._log("message.new", payload, message={
                "id": message_id, "sender": "user", "kind": "text", "text": text, "attachments": attachments})
            await self._dispatch_user_message(chat, message_id, text, attachments, reply_to, location)
            return event.seq

    async def _reply_snapshot(self, chat: str, reply_to: Any) -> Optional[dict[str, Any]]:
        """The quote a message carries: a snapshot of ``reply_to`` taken now, or None when that id is
        no message of this chat (never sent, deleted, pruned, or in another chat)."""
        if not isinstance(reply_to, str) or not reply_to:
            return None
        quoted = await asyncio.to_thread(self.store.message, reply_to)
        if quoted is None or quoted.chat != chat:
            return None
        location = None if quoted.attachments else await self._logged_location(quoted.seq)
        return replies.snapshot(quoted.id, quoted.sender, quoted.text, quoted.attachments, location)

    async def _logged_location(self, seq: int) -> Optional[dict[str, Any]]:
        """The location a logged message carried (for handing over a resend that never reached the
        gateway)."""
        events = await asyncio.to_thread(self.store.events_after, seq - 1, until=seq)
        return events[0].payload.get("location") if events else None

    async def _dispatch_user_message(self, chat: str, message_id: str, text: str, attachments: list[dict[str, Any]],
                                     reply_to: Optional[str], location: Optional[dict[str, Any]] = None) -> None:
        media_urls, media_types, media_names, inlined, injected = [], [], [], [], []
        message_type = MessageType.TEXT
        for description in attachments:
            stored = await asyncio.to_thread(self.store.file, str(description.get("id")))
            if stored is None or not stored.agent_path:
                continue
            name = str(description.get("name") or "")
            mime = str(description.get("mime") or "application/octet-stream")
            media_urls.append(stored.agent_path)
            media_types.append(mime)
            media_names.append(name)
            content = await asyncio.to_thread(_inline_text, stored.agent_path, name, mime,
                                              str(description.get("kind") or "file"))
            # The gateway words its note by this flag: "included below" only when it really is.
            inlined.append(content is not None)
            if content is not None:
                injected.append(f"[Content of {name or 'the file'}]:\n{content}")
            if message_type == MessageType.TEXT:
                message_type = _media_message_type(str(description.get("kind") or "file"))
        if injected:
            text = "\n\n".join(injected + ([text] if text.strip() else []))
        if location is not None:
            text = location_note(location, text)
            if message_type == MessageType.TEXT:
                message_type = MessageType.LOCATION
        reacted = await self._take_pending_reactions(chat)
        if reacted:
            text = f"{reacted}\n\n{text}" if text.strip() else reacted
        source = self._source(chat)
        event = MessageEvent(source=source, text=text, message_type=message_type, message_id=message_id,
                             media_urls=media_urls, media_types=media_types, media_text_inlined=inlined,
                             media_filenames=media_names)
        if reply_to:
            replied = await asyncio.to_thread(self.store.message, reply_to)
            if replied is not None and replied.chat == chat:
                event.reply_to_message_id = reply_to
                event.reply_to_text = replied.text
                event.reply_to_is_own_message = replied.sender == "agent"
        # Recorded before the hand-off: a turn may start (and send its receipt) inside it.
        self._last_user_message[chat] = message_id
        await asyncio.to_thread(self.store.set_meta, f"last_user_message:{chat}", message_id)
        await self.handle_message(event)
        await asyncio.to_thread(self.store.mark_dispatched, message_id)

    def _source(self, chat: str):
        """The gateway's source for a chat: a DM keyed by its id, so each chat is a session of its own."""
        side = self._chats.get(chat)
        name = (side.title or "Side chat") if side is not None else CHAT_NAME
        return self.build_source(chat_id=chat, chat_name=name, chat_type="dm", user_id=OWNER_ID, user_name=OWNER_NAME)

    async def _command(self, chat: str, command: str) -> None:
        """A slash command of the gateway's own for a chat (not logged as a message)."""
        await self.handle_message(MessageEvent(source=self._source(chat), text=command,
                                               message_type=MessageType.COMMAND,
                                               message_id=f"{command.strip('/')}-{uuid.uuid4().hex}"))

    async def _stop(self, chat: str) -> None:
        """``chat.stop``: the gateway's own ``/stop`` for that chat."""
        await self._command(chat, "/stop")

    # ── chats ──────────────────────────────────────────────────────────────────────────────────

    def _exists(self, chat: str) -> bool:
        return chat == CHAT_ID or chat in self._chats

    def _chat_param(self, params: dict[str, Any], key: str = "chat") -> str:
        """A call's chat: ``main`` or a side chat that exists."""
        chat = params.get(key)
        if not isinstance(chat, str) or not chat:
            raise RpcError(INVALID_PARAMS, f"{key} is required: main or a side chat's id")
        if not self._exists(chat):
            raise RpcError(INVALID_PARAMS, "no chat with that id")
        return chat

    def _outbound_chat(self, chat_id: Any) -> Optional[str]:
        """Where the gateway's send goes, or None for a chat that no longer exists."""
        chat = str(chat_id or CHAT_ID)
        if self._exists(chat):
            return chat
        if chat not in self._deleted_chats:
            logger.warning("Aleph: dropped a send to unknown chat %r", chat)
        return None

    async def _create_chat(self, params: dict[str, Any]) -> int:
        """``chat.create {id, title?}``: open a side chat; the same id again answers as the first time."""
        chat_id, title = params.get("id"), params.get("title")
        if not isinstance(chat_id, str) or not chat_id.strip() or len(chat_id) > MAX_CHAT_ID:
            raise RpcError(INVALID_PARAMS, "id must be a non-empty string")
        if chat_id == CHAT_ID:
            raise RpcError(INVALID_PARAMS, "main always exists")
        if title is not None:
            if not isinstance(title, str):
                raise RpcError(INVALID_PARAMS, "title must be a string or null")
            title = " ".join(title.split())[:MAX_CHAT_TITLE] or None
        async with self._chat_lock, self.append_lock:
            chat, event = await asyncio.to_thread(self.store.create_chat, chat_id, title, now_iso())
            if chat.deleted_seq is not None:
                raise RpcError(INVALID_PARAMS, "that chat was deleted")
            self._chats[chat.id] = chat
            if event is not None:
                self._broadcast(event)
                logger.info("Aleph: opened side chat %s", chat.id)
        return chat.created_seq

    async def _delete_chat_call(self, params: dict[str, Any]) -> int:
        chat_id = params.get("id")
        if chat_id == CHAT_ID:
            raise RpcError(INVALID_PARAMS, "main cannot be deleted")
        if not isinstance(chat_id, str) or not chat_id:
            raise RpcError(INVALID_PARAMS, "id is required")
        known = await asyncio.to_thread(self.store.chat, chat_id)
        if known is None:
            raise RpcError(INVALID_PARAMS, "no chat with that id")
        if known.deleted_seq is not None:
            return known.deleted_seq  # a resend after a lost answer
        return await self.delete_chat(chat_id)

    async def delete_chat(self, chat_id: str) -> int:
        """End a side chat's session (the gateway's ``/new``) and drop it: its events, messages, steps
        and files; logs ``chat.delete``."""
        async with self._chat_lock:
            if chat_id not in self._chats:
                known = await asyncio.to_thread(self.store.chat, chat_id)
                return known.deleted_seq if known is not None and known.deleted_seq is not None else 0
            source = self._source(chat_id)
            self._chats.pop(chat_id)
            self._deleted_chats.add(chat_id)
            self._clear_draft(chat_id)
            self._set_typing(chat_id, active=False)
            self._open_steps.pop(chat_id, None)
            self._activity_status.pop(chat_id, None)
            self._turn_message.pop(chat_id, None)
            self._last_user_message.pop(chat_id, None)
            self._last_receipt.pop(chat_id, None)
            if self._pending_reactions.pop(chat_id, None) is not None:
                await self._save_pending_reactions()
            for cards in (self._approvals, self._clarifies):
                for card_id in [k for k, card in cards.items() if card.get("chat") == chat_id]:
                    cards.pop(card_id, None)
            try:
                await self.handle_message(MessageEvent(source=source, text="/new", message_type=MessageType.COMMAND,
                                                       message_id=f"new-{uuid.uuid4().hex}"))
            except Exception:
                logger.warning("Aleph: ending side chat %s's session failed", chat_id, exc_info=True)
            async with self.append_lock:
                event = await asyncio.to_thread(self.store.delete_chat, chat_id)
                self._broadcast(event)
            logger.info("Aleph: deleted side chat %s", chat_id)
            return event.seq

    async def _sweep_idle_chats(self) -> None:
        """Delete the side chats nobody has written in for ``side_chat_idle_days``, hourly."""
        while True:
            try:
                if self._store is None:
                    return
                cutoff = time.time() - self.side_chat_idle_days * 86400
                for chat_id in await asyncio.to_thread(self.store.idle_chats, cutoff):
                    if chat_id not in self._typing and chat_id not in self._open_steps:  # not mid-turn
                        logger.info("Aleph: side chat %s was idle for %d days", chat_id, self.side_chat_idle_days)
                        await self.delete_chat(chat_id)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.debug("Aleph: idle chat sweep failed", exc_info=True)
            await asyncio.sleep(IDLE_SWEEP_SECONDS)

    def on_session_title(self, source: Any, title: str, title_source: str) -> None:
        """Hermes named a session (on the agent's thread): a side chat's title, logged as ``chat.update``.
        ``main`` has none."""
        chat = str(getattr(source, "chat_id", "") or "")
        title = " ".join(str(title or "").split())[:MAX_CHAT_TITLE]
        if chat and chat != CHAT_ID and title:
            self._on_loop(lambda: asyncio.ensure_future(self._retitle(chat, title)))

    async def _retitle(self, chat_id: str, title: str) -> None:
        async with self._chat_lock:
            chat = self._chats.get(chat_id)
            if chat is None or chat.title == title or self._store is None:
                return
            await asyncio.to_thread(self.store.set_chat_title, chat_id, title)
            self._chats[chat_id] = dataclasses.replace(chat, title=title)
            await self._log("chat.update", {"chat": chat_id, "title": title})

    # ── receipts ───────────────────────────────────────────────────────────────────────────────

    async def on_processing_start(self, event: MessageEvent) -> None:
        """A turn started on the person's input: everything they sent so far is being read."""
        source = getattr(event, "source", None)
        if getattr(source, "platform", None) != self.platform:
            return
        chat = str(getattr(source, "chat_id", "") or CHAT_ID)
        # The steps of this turn answer the newest of the person's messages (none for a cron run).
        internal = getattr(event, "internal", False)
        self._turn_message[chat] = None if internal else self._last_user_message.get(chat)
        if internal or not self._exists(chat) or self._store is None:
            return
        target = self._last_user_message.get(chat)
        if not target or target == self._last_receipt.get(chat):
            return
        if await asyncio.to_thread(self.store.user_message_seq, target) is None:
            return
        self._last_receipt[chat] = target
        await asyncio.to_thread(self.store.set_meta, f"last_receipt:{chat}", target)
        await self._log("receipt", {"chat": chat, "read_up_to": target, "at": now_iso()})

    async def on_processing_complete(self, event: MessageEvent, outcome: Any) -> None:
        source = getattr(event, "source", None)
        if getattr(source, "platform", None) != self.platform:
            return
        chat = str(getattr(source, "chat_id", "") or CHAT_ID)
        self._clear_draft(chat)
        await self._close_open_steps(chat)
        self._set_typing(chat, active=False)

    # ── outbound text ──────────────────────────────────────────────────────────────────────────

    async def send(self, chat_id: str, content: str, reply_to: Optional[str] = None,
                   metadata: Optional[dict[str, Any]] = None) -> SendResult:
        if self._store is None:
            return SendResult(success=False, error="Aleph is not connected", retryable=True)
        meta = metadata or {}
        kind = "tool" if meta.get("_tool_progress") else "system" if meta.get("_interim_send") else "text"
        return await self._send_message(chat_id, content or "", [], kind=kind, reply_to=reply_to)

    async def _send_message(self, chat_id: Any, text: str, attachments: list[dict[str, Any]], *,
                            kind: str = "text", reply_to: Optional[str] = None) -> SendResult:
        chat = self._outbound_chat(chat_id)
        if chat is None:
            # A deleted chat: the reply has nowhere to go (not an error the gateway should retry).
            return SendResult(success=True, message_id=None)
        chat_id = chat
        message_id = str(uuid.uuid4())
        payload: dict[str, Any] = {"chat": chat, "id": message_id, "sender": "agent", "text": text,
                                   "attachments": self._with_links(attachments), "created_at": now_iso()}
        if kind != "text":
            payload["kind"] = kind
        else:
            # The reply's real message replaces its draft on the phone.
            open_draft = self._open_draft.pop(chat_id, None)
            if open_draft is not None:
                payload["draft_id"] = open_draft[0]
            # The gateway anchors every reply to the message that started the turn; answering the
            # person's latest message needs no quote. Anything else it names (an earlier message, one
            # the person sent after the turn began) is quoted.
            if reply_to and str(reply_to) != self._last_user_message.get(chat):
                quoted = await self._reply_snapshot(chat, str(reply_to))
                if quoted is not None:
                    payload["reply_to"] = quoted
        try:
            await self._log("message.new", payload, message={
                "id": message_id, "sender": "agent", "kind": kind, "text": text, "attachments": attachments})
        except Exception as exc:
            logger.error("Aleph: logging an outbound message failed: %s", exc)
            return SendResult(success=False, error=str(exc))
        return SendResult(success=True, message_id=message_id)

    async def edit_message(self, chat_id: str, message_id: str, content: str, *, finalize: bool = False,
                           metadata: Optional[dict[str, Any]] = None) -> SendResult:
        if self._store is None:
            return SendResult(success=False, error="Aleph is not connected", retryable=True)
        stored = await asyncio.to_thread(self.store.message, str(message_id))
        if stored is None or stored.sender != "agent":
            return SendResult(success=False, error="unknown message")
        if stored.text != content:
            await self._log("message.edit", {"chat": stored.chat, "id": stored.id, "text": content})
        return SendResult(success=True, message_id=stored.id)

    async def delete_message(self, chat_id: str, message_id: str) -> bool:
        if self._store is None:
            return False
        stored = await asyncio.to_thread(self.store.message, str(message_id))
        if stored is None or stored.sender != "agent":
            return False
        await self._log("message.delete", {"chat": stored.chat, "id": stored.id})
        return True

    # ── drafts ─────────────────────────────────────────────────────────────────────────────────

    def supports_draft_streaming(self, chat_type: Optional[str] = None, metadata: Optional[dict[str, Any]] = None,
                                 chat_id: Optional[str] = None) -> bool:
        return True

    async def send_draft(self, chat_id: str, draft_id: int, content: str,
                         metadata: Optional[dict[str, Any]] = None) -> SendResult:
        chat = self._outbound_chat(chat_id)
        if chat is None:
            return SendResult(success=True)
        # A segment after a tool call opens with blank lines the final message does not have.
        content = (content or "").lstrip("\n")
        wire_id = self._draft_wire_ids.get(draft_id)
        if wire_id is None:
            # Wire ids are unique across restarts (the consumer's counter starts over).
            wire_id = self._draft_wire_ids[draft_id] = f"{self._draft_prefix}-{draft_id}"
            if len(self._draft_wire_ids) > 512:
                for stale in list(self._draft_wire_ids)[:256]:
                    self._draft_wire_ids.pop(stale, None)
        previous = self._open_draft.get(chat)
        if previous is not None and previous[0] != wire_id:
            self._live("draft.clear", {"chat": chat, "draft_id": previous[0]})
        self._open_draft[chat] = (wire_id, content)
        self._live("draft", {"chat": chat, "draft_id": wire_id, "text": content})
        return SendResult(success=True)

    async def abandon_open_draft(self, chat_id: str, content: str,
                                 metadata: Optional[dict[str, Any]] = None) -> SendResult:
        """The draft's turn died (/stop, /new, an error): the phone drops it."""
        self._clear_draft(str(chat_id or CHAT_ID))
        return SendResult(success=True)

    def _clear_draft(self, chat: str) -> None:
        open_draft = self._open_draft.pop(chat, None)
        if open_draft is not None:
            self._live("draft.clear", {"chat": chat, "draft_id": open_draft[0]})

    # ── typing ─────────────────────────────────────────────────────────────────────────────────

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        chat = self._outbound_chat(chat_id)
        if chat is not None:
            self._set_typing(chat, active=True, status=self._activity_status.get(chat))

    async def stop_typing(self, chat_id: str) -> None:
        self._set_typing(str(chat_id or CHAT_ID), active=False)

    def set_status_text(self, chat_id: str, text: Optional[str]) -> None:
        """Hermes's live status is the Slack form ("is searching …"); Aleph words its status line
        itself from the tool activity (``on_tool_activity``), so this text is not shown."""
        super().set_status_text(chat_id, text)

    # ── reactions ──────────────────────────────────────────────────────────────────────────────

    async def _user_reaction(self, params: dict[str, Any]) -> int:
        """``reaction.set {message_id, emoji|null}``: log it; the agent hears of it with the next message."""
        message_id = params.get("message_id")
        if not isinstance(message_id, str) or not message_id:
            raise RpcError(INVALID_PARAMS, "message_id is required")
        if "emoji" not in params:
            raise RpcError(INVALID_PARAMS, "emoji is required (null takes the reaction back)")
        raw = params.get("emoji")
        if raw is not None and not reactions.is_single_emoji(raw):
            raise RpcError(INVALID_PARAMS, "emoji must be one emoji, or null")
        stored = await asyncio.to_thread(self.store.message, message_id)
        if stored is None:
            raise RpcError(INVALID_PARAMS, "no message with that id")
        event = await self._log("reaction", {"chat": stored.chat, "message_id": message_id, "by": "user",
                                             "emoji": raw, "at": now_iso()})
        if stored.sender == "agent":
            # The latest per message; a retraction drops it.
            pending = self._pending_reactions.setdefault(stored.chat, {})
            pending.pop(message_id, None)
            if raw is not None:
                pending[message_id] = raw
            await self._save_pending_reactions()
        return event.seq

    async def _save_pending_reactions(self) -> None:
        self._pending_reactions = {chat: pending for chat, pending in self._pending_reactions.items() if pending}
        await asyncio.to_thread(self.store.set_meta, "pending_reactions", json.dumps(self._pending_reactions))

    async def _take_pending_reactions(self, chat: str) -> str:
        """The note for the person's reactions in a chat since their last message there, which it
        then clears."""
        pending = self._pending_reactions.pop(chat, None)
        if not pending:
            return ""
        await self._save_pending_reactions()
        reacted = []
        for message_id, emoji in pending.items():
            stored = await asyncio.to_thread(self.store.message, message_id)
            if stored is not None:
                reacted.append((emoji, stored.text, bool(stored.attachments)))
        return reactions.note(reacted)

    async def react(self, emoji: Optional[str], *, chat: str = CHAT_ID, message_id: Optional[str] = None,
                    messages_back: int = 0) -> dict[str, Any]:
        """The agent reacts in a chat (``emoji`` None: takes its reaction back), by default to the
        person's latest message there; ``ValueError`` says what is wrong. Runs on the gateway's loop."""
        if not self._exists(chat):
            raise ValueError("this chat no longer exists")
        if message_id:
            stored = await asyncio.to_thread(self.store.message, message_id)
            if stored is None or stored.chat != chat:
                raise ValueError(f"there is no message {message_id!r} in the conversation")
        else:
            ids = await asyncio.to_thread(self.store.user_message_ids, chat, messages_back + 1)
            if len(ids) <= messages_back:
                raise ValueError("there is no message of the person's to react to yet" if not messages_back
                                 else f"the person has fewer than {messages_back + 1} messages")
            message_id = ids[messages_back]
        await self._log("reaction", {"chat": chat, "message_id": message_id, "by": "agent", "emoji": emoji,
                                     "at": now_iso()})
        return {"message_id": message_id, "emoji": emoji}

    async def _react_on_loop(self, chat_id: Any, emoji: Optional[str], message_id: Optional[str]) -> dict[str, Any]:
        """``send_message(action="react")`` calls from another loop; the log is the gateway loop's."""
        loop = self._loop
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        chat = str(chat_id or CHAT_ID)
        try:
            if loop is None or running is loop:
                result = await self.react(emoji, chat=chat, message_id=message_id)
            else:
                future = asyncio.run_coroutine_threadsafe(self.react(emoji, chat=chat, message_id=message_id), loop)
                result = await asyncio.wrap_future(future)
        except ValueError as exc:
            return {"success": False, "error": str(exc)}
        return {"success": True, **result}

    async def add_reaction(self, chat_id: str, emoji: str, message_id: Optional[str] = None) -> dict[str, Any]:
        """Hermes's ``send_message(action="react")``: react to a message (default: the person's latest)."""
        try:
            cleaned = reactions.clean_emoji(emoji)
        except ValueError as exc:
            return {"success": False, "error": str(exc)}
        if cleaned is None:
            return {"success": False, "error": "emoji is required; use action='unreact' to take a reaction back"}
        return await self._react_on_loop(chat_id, cleaned, message_id)

    async def remove_reaction(self, chat_id: str, message_id: Optional[str] = None) -> dict[str, Any]:
        """``send_message(action="unreact")``: take the agent's reaction back."""
        return await self._react_on_loop(chat_id, None, message_id)

    # ── activity ───────────────────────────────────────────────────────────────────────────────

    def on_tool_activity(self, source: Any, event: dict[str, Any]) -> None:
        """A tool started or finished in a turn this channel answers (the gateway's progress
        callback, on the agent's worker thread)."""
        chat = str(getattr(source, "chat_id", "") or CHAT_ID)
        if not self._exists(chat):
            return
        mode = str(event.get("live_status") or "full")
        tool = str(event.get("tool") or "")
        if event.get("type") == "tool.started":
            described = activity.describe(tool, event.get("args"), with_preview=mode == "full")
            if described is None:
                return  # looking tools up: no step, and the line stays on "Thinking…"
            name, summary, detail = described
            step = activity.new_step(chat, self._turn_message.get(chat), name, summary, detail, now_iso())
            show = mode != "off"
            self._on_loop(lambda: asyncio.ensure_future(self._step_started(chat, tool, step, show)))
        elif event.get("type") == "tool.completed":
            ok = not event.get("is_error")
            self._on_loop(lambda: asyncio.ensure_future(self._step_finished(chat, tool, ok)))

    async def _step_started(self, chat: str, tool: str, step: dict[str, Any], show: bool) -> None:
        if self._store is None or not self._exists(chat):
            return
        self._open_steps.setdefault(chat, []).append((tool, step))
        await asyncio.to_thread(self.store.add_step, step)
        self._live("activity", dict(step))
        if show:
            self._show_status(chat, step["summary"] + "…")

    async def _step_finished(self, chat: str, tool: str, ok: Optional[bool]) -> None:
        open_steps = self._open_steps.get(chat) or []
        # Pair by tool name, oldest first: the progress callback carries no call id.
        index = next((i for i, (raw, _) in enumerate(open_steps) if raw == tool), None)
        if index is None or self._store is None:
            return
        _, step = open_steps.pop(index)
        step.update({"finished_at": now_iso(), "ok": ok})
        await asyncio.to_thread(self.store.update_step, step)
        self._live("activity", dict(step))
        # Back to the most recent step still running, or to null (the phone shows "Thinking…").
        self._show_status(chat, open_steps[-1][1]["summary"] + "…" if open_steps else None)

    async def _close_open_steps(self, chat: str) -> None:
        """The turn ended with steps never reported finished (a stop, an error): they did not work."""
        for raw, _step in list(self._open_steps.get(chat) or []):
            await self._step_finished(chat, raw, False)
        self._open_steps.pop(chat, None)
        self._activity_status.pop(chat, None)

    def _show_status(self, chat: str, status: Optional[str]) -> None:
        self._activity_status[chat] = status
        if chat in self._typing and self._typing.get(chat) != status:
            self._set_typing(chat, active=True, status=status)

    async def list_activity(self, params: dict[str, Any]) -> dict[str, Any]:
        """``activity.list {chat?, before?, limit?}`` -> ``{steps, more}``, newest first; without
        ``chat``, every chat's."""
        chat = self._chat_param(params) if params.get("chat") is not None else None
        before, limit = params.get("before"), params.get("limit", 50)
        if before is not None and not isinstance(before, str):
            raise RpcError(INVALID_PARAMS, "before must be a step id")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 200:
            raise RpcError(INVALID_PARAMS, "limit must be from 1 to 200")
        steps, more = await asyncio.to_thread(self.store.steps, chat=chat, before=before, limit=limit)
        return {"steps": steps, "more": more}

    def _on_loop(self, fn) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            fn()
        else:
            loop.call_soon_threadsafe(fn)

    def _set_typing(self, chat: str, *, active: bool, status: Optional[str] = None) -> None:
        if active:
            if chat in self._typing and self._typing[chat] == status:
                return  # the 2s refresh repeats itself; the phone already shows it
            self._typing[chat] = status
            self._live("typing", self._typing_payload(chat, True, status))
        elif self._typing.pop(chat, "absent") != "absent":
            self._live("typing", self._typing_payload(chat, False))

    @staticmethod
    def _typing_payload(chat: str, active: bool, status: Optional[str] = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"chat": chat, "active": active}
        if active and status:
            payload["status"] = status
        return payload

    # ── approvals ──────────────────────────────────────────────────────────────────────────────

    async def _send_exec_approval_prompt(self, prompt: ExecApprovalPrompt) -> SendResult:
        if self._store is None:
            return SendResult(success=False, error="Aleph is not connected", retryable=True)
        request_id = prompt.request_id or uuid.uuid4().hex
        choices = [{"label": label, "choice": choice, "style": _APPROVAL_STYLES.get(style, "default")}
                   for label, choice, style in prompt.actions]
        chat = self._outbound_chat(prompt.chat_id)
        if chat is None:
            return SendResult(success=False, error="the chat was deleted")
        payload: dict[str, Any] = {"chat": chat, "request_id": request_id, "text": prompt.text, "choices": choices}
        if prompt.command:
            payload["command"] = prompt.command
        if prompt.description:
            payload["reason"] = prompt.description
        self._approvals[request_id] = {"chat": chat, "session_key": prompt.session_key, "choices": prompt.choices,
                                       "card_request_id": prompt.request_id, "answered": False}
        await self._log("approval.request", payload)
        if prompt.request_id:
            self._observe_approval(prompt.session_key, request_id)
        return SendResult(success=True, message_id=request_id)

    def _observe_approval(self, session_key: str, request_id: str) -> None:
        """Log ``approval.resolved`` when the request ends some other way (timeout, ``/approve``)."""
        try:
            from tools.approval import observe_gateway_settle
        except ImportError:
            return
        loop = self._loop

        def settled(reason: str) -> None:  # runs on the agent's thread
            if loop is not None and not loop.is_closed():
                asyncio.run_coroutine_threadsafe(self._approval_settled(request_id, reason), loop)
        if not observe_gateway_settle(session_key, request_id, settled):
            # Already settled before the card went out.
            asyncio.ensure_future(self._approval_settled(request_id, "session_closed"))

    async def _approval_settled(self, request_id: str, reason: str) -> None:
        card = self._approvals.pop(request_id, None)
        if card is None or card.get("answered") or self._store is None:
            return
        by = "timeout" if reason == "timeout" else "other" if reason == "resolved" else "cancelled"
        await self._log("approval.resolved", {"chat": card["chat"], "request_id": request_id, "choice": None, "by": by})

    async def _respond_approval(self, params: dict[str, Any]) -> None:
        request_id, choice = params.get("request_id"), params.get("choice")
        if not isinstance(request_id, str) or not isinstance(choice, str):
            raise RpcError(INVALID_PARAMS, "request_id and choice are required")
        card = self._approvals.get(request_id)
        if card is None:
            raise RpcError(INVALID_PARAMS, "no pending approval with that request_id")
        if choice not in card["choices"]:
            raise RpcError(INVALID_PARAMS, "choice is not one the request offered",
                           {"choices": list(card["choices"])})
        from tools.approval import resolve_gateway_approval_card
        card["answered"] = True
        resolved = resolve_gateway_approval_card(card["session_key"], choice, card["card_request_id"])
        self._approvals.pop(request_id, None)
        chat = card["chat"]
        if resolved:
            await self._log("approval.resolved", {"chat": chat, "request_id": request_id, "choice": choice, "by": "user"})
        else:
            await self._log("approval.resolved", {"chat": chat, "request_id": request_id, "choice": None, "by": "timeout"})

    # ── clarify ────────────────────────────────────────────────────────────────────────────────

    async def send_clarify(self, chat_id: str, question: str, choices: Optional[list], clarify_id: str,
                           session_key: str, metadata: Optional[dict[str, Any]] = None) -> SendResult:
        if self._store is None:
            return SendResult(success=False, error="Aleph is not connected", retryable=True)
        chat = self._outbound_chat(chat_id)
        if chat is None:
            return SendResult(success=False, error="the chat was deleted")
        labels = [str(choice) for choice in (choices or [])]
        entry = None
        try:
            from tools.clarify_gateway import get_pending
            entry = get_pending(clarify_id)
        except ImportError:
            pass
        # The entry is kept to read the answer when the clarify ends some other way.
        self._clarifies[clarify_id] = {"chat": chat, "choices": labels, "answered": False, "entry": entry}
        await self._log("clarify.request", {"chat": chat, "clarify_id": clarify_id, "question": question,
                                            "choices": labels, "allows_text": True})
        return SendResult(success=True, message_id=clarify_id)

    async def retire_clarify_card(self, clarify_id: str, notice: str) -> None:
        """The clarify ended without ``clarify.respond``: a typed answer in the chat, a timeout or a
        reset. Logged so the phone's card stops offering buttons."""
        card = self._clarifies.pop(clarify_id, None)
        if card is None or card.get("answered") or self._store is None:
            return
        answer = getattr(card.get("entry"), "response", None)
        if not isinstance(answer, str) or not answer or answer.startswith("\x00"):
            answer = None  # the SKIPPED / CANCELLED sentinels are not answers
        await self._log("clarify.resolved", {"chat": card["chat"], "clarify_id": clarify_id, "answer": answer,
                                             "by": "user" if answer else "timeout"})

    async def _respond_clarify(self, params: dict[str, Any]) -> None:
        clarify_id = params.get("clarify_id")
        if not isinstance(clarify_id, str):
            raise RpcError(INVALID_PARAMS, "clarify_id is required")
        card = self._clarifies.get(clarify_id)
        if card is None:
            raise RpcError(INVALID_PARAMS, "no pending question with that clarify_id")
        if "choice" in params and params["choice"] is not None:
            index = params["choice"]
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(card["choices"]):
                raise RpcError(INVALID_PARAMS, "choice is out of range")
            answer = card["choices"][index]
        else:
            answer = params.get("text")
            if not isinstance(answer, str) or not answer.strip():
                raise RpcError(INVALID_PARAMS, "choice or text is required")
        from tools.clarify_gateway import resolve_gateway_clarify
        card["answered"] = True
        resolved = resolve_gateway_clarify(clarify_id, answer)
        self._clarifies.pop(clarify_id, None)
        chat = card["chat"]
        if resolved:
            await self._log("clarify.resolved", {"chat": chat, "clarify_id": clarify_id, "answer": answer, "by": "user"})
        else:
            await self._log("clarify.resolved", {"chat": chat, "clarify_id": clarify_id, "answer": None, "by": "timeout"})

    # ── files ──────────────────────────────────────────────────────────────────────────────────

    async def _file_ticket(self, params: dict[str, Any]) -> dict[str, Any]:
        """``file.ticket {size}`` before an upload, ``{id}`` for a fresh download link."""
        if self._tickets is None:
            raise RpcError(INVALID_PARAMS, "not connected")
        if "id" in params:
            file_id = params.get("id")
            stored = await asyncio.to_thread(self.store.file, file_id) if isinstance(file_id, str) else None
            if stored is None or not os.path.isfile(stored.path):
                raise RpcError(INVALID_PARAMS, "no such file")
            ticket = self._tickets.issue_download(stored.id)
        else:
            size = params.get("size")
            limit = upload_limit()
            if isinstance(size, bool) or not isinstance(size, int) or size < 16:
                raise RpcError(INVALID_PARAMS, "size must be the ciphertext's byte count (the file's plus 16)")
            if size > limit + 16:
                raise RpcError(INVALID_PARAMS, f"the file is too large: at most {limit // (1024 * 1024)} MB",
                               {"max_bytes": limit})
            ticket = self._tickets.issue_upload(size)
        return {"ticket": ticket, "url": file_url(ticket)}

    def upload_size(self, ticket: str) -> Optional[int]:
        return self._tickets.upload_size(ticket) if self._tickets is not None else None

    async def accept_upload(self, ticket: str, data: bytes) -> bool:
        if self._tickets is None:
            return False
        return await asyncio.to_thread(self._tickets.accept_upload, ticket, data)

    def redeem_download(self, ticket: str) -> Optional[str]:
        """The ciphertext path a download ticket opens (once), or ``None``."""
        if self._tickets is None or self._store is None:
            return None
        file_id = self._tickets.redeem_download(ticket)
        stored = self.store.file(file_id) if file_id else None
        return stored.path if stored is not None and os.path.isfile(stored.path) else None

    def _with_links(self, attachments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Attachments as a logged ``message.new`` carries them: each with a fresh download link."""
        if self._tickets is None:
            return attachments
        return [{**description, "url": file_url(self._tickets.issue_download(str(description["id"])))}
                for description in attachments]

    async def _take_uploads(self, uploads: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """``message.send`` attachments ``{ticket, key, name, mime}``: decrypt each upload, hand the
        plaintext to the gateway's media cache (what the agent reads) and keep the ciphertext (what
        downloads serve). All are checked before any is spent."""
        if self._tickets is None:
            raise RpcError(INVALID_PARAMS, "not connected")
        opened = []
        for upload in uploads:
            ticket, key = upload.get("ticket"), upload.get("key")
            name, mime = upload.get("name"), upload.get("mime")
            path = self._tickets.uploaded(ticket) if isinstance(ticket, str) else None
            if path is None:
                raise RpcError(INVALID_PARAMS, "unknown or unfinished upload ticket")
            try:
                plaintext = decrypt_file(from_b64url(key) if isinstance(key, str) else b"",
                                         await asyncio.to_thread(path.read_bytes))
            except (ValueError, OSError) as exc:
                raise RpcError(INVALID_PARAMS, f"the upload does not decrypt with its key: {exc}") from exc
            name = Path(name).name.replace("\x00", "").strip()[:200] if isinstance(name, str) else ""
            mime = (mime.split(";", 1)[0].strip().lower() if isinstance(mime, str) else "") or "application/octet-stream"
            opened.append((ticket, str(key), name or "file" + (mimetypes.guess_extension(mime) or ""), mime,
                           path, plaintext, _upload_hints(upload)))
        # Into the media cache first, all of them: a file the gateway refuses (too big, not what its
        # type claims) fails the send before any ticket is spent.
        cached_files = []
        for ticket, key, name, mime, path, plaintext, hints in opened:
            try:
                cached = await cache_media_bytes_async(plaintext, filename=name, mime_type=mime)
            except ValueError as exc:
                raise RpcError(INVALID_PARAMS, f"{name} was refused: {exc}") from exc
            if cached is None:
                raise RpcError(INVALID_PARAMS, f"{name} could not be read")
            cached_files.append(cached)
        attachments = []
        for (ticket, key, name, mime, path, plaintext, hints), cached in zip(opened, cached_files):
            file_id = f"f_{uuid.uuid4().hex}"
            dest = self.store.files_dir / f"{file_id}.bin"
            await asyncio.to_thread(os.replace, path, dest)
            self._tickets.spend_upload(ticket)
            media_type = cached.media_type or mime
            description = self._describe(file_id, name, media_type, _file_kind(cached.kind, media_type, name),
                                         len(plaintext), cached.path, key, hints)
            await asyncio.to_thread(self.store.add_file, description, str(dest), cached.path)
            attachments.append(description)
        return attachments

    @staticmethod
    def _describe(file_id: str, name: str, mime: str, kind: str, size: int, plaintext_path: str,
                  key: str, hints: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """A file description. An image's size is measured; otherwise what the phone said is kept
        (``width``/``height`` for an image or video, ``duration`` for a video or audio)."""
        hints = hints or {}
        description: dict[str, Any] = {"id": file_id, "name": name, "mime": mime, "kind": kind, "size": size}
        if kind in ("image", "video"):
            dims = _image_size(plaintext_path) if kind == "image" else None
            if dims is not None:
                description["width"], description["height"] = dims
            elif "width" in hints and "height" in hints:
                description["width"], description["height"] = hints["width"], hints["height"]
        if kind in ("video", "audio", "voice") and "duration" in hints:
            description["duration"] = hints["duration"]
        description["key"] = key
        return description

    async def _register_agent_file(self, source_path: str, *, kind: str, name: Optional[str] = None) -> dict[str, Any]:
        """Encrypt a file the agent sends into Aleph's store (the original may be a temp/cache file)."""
        src = Path(source_path).expanduser()
        if not src.is_file():
            raise FileNotFoundError(source_path)
        display = name or src.name
        plaintext = await asyncio.to_thread(src.read_bytes)
        key, ciphertext = encrypt_file(plaintext)
        file_id = f"f_{uuid.uuid4().hex}"
        dest = self.store.files_dir / f"{file_id}.bin"
        await asyncio.to_thread(dest.write_bytes, ciphertext)
        mime = mimetypes.guess_type(display)[0] or mimetypes.guess_type(str(src))[0] or "application/octet-stream"
        description = self._describe(file_id, display, mime, kind, len(plaintext), str(src), b64url(key))
        await asyncio.to_thread(self.store.add_file, description, str(dest), None)
        return description

    async def _send_file(self, chat_id: str, path: str, caption: Optional[str], *, kind: str,
                         name: Optional[str] = None, reply_to: Optional[str] = None) -> SendResult:
        if self._store is None:
            return SendResult(success=False, error="Aleph is not connected", retryable=True)
        try:
            description = await self._register_agent_file(path, kind=kind, name=name)
        except (OSError, ValueError) as exc:
            logger.warning("Aleph: cannot attach %s: %s", path, exc)
            return SendResult(success=False, error=f"cannot read {path}: {exc}")
        return await self._send_message(chat_id, caption or "", [description], reply_to=reply_to)

    async def send_image_file(self, chat_id: str, image_path: str, caption: Optional[str] = None,
                              reply_to: Optional[str] = None, metadata: Optional[dict[str, Any]] = None,
                              **kwargs) -> SendResult:
        return await self._send_file(chat_id, image_path, caption, kind="image", reply_to=reply_to)

    async def send_document(self, chat_id: str, file_path: str, caption: Optional[str] = None,
                            file_name: Optional[str] = None, reply_to: Optional[str] = None,
                            metadata: Optional[dict[str, Any]] = None, **kwargs) -> SendResult:
        return await self._send_file(chat_id, file_path, caption, kind="file", name=file_name, reply_to=reply_to)

    async def send_voice(self, chat_id: str, audio_path: str, caption: Optional[str] = None,
                         reply_to: Optional[str] = None, metadata: Optional[dict[str, Any]] = None,
                         **kwargs) -> SendResult:
        return await self._send_file(chat_id, audio_path, caption, kind="voice", reply_to=reply_to)

    async def send_video(self, chat_id: str, video_path: str, caption: Optional[str] = None,
                         reply_to: Optional[str] = None, metadata: Optional[dict[str, Any]] = None,
                         **kwargs) -> SendResult:
        return await self._send_file(chat_id, video_path, caption, kind="video", reply_to=reply_to)

    async def send_files(self, chat_id: str, file_paths: list[str], caption: str = "") -> SendResult:
        """Several files on one message (send_message with media)."""
        if self._store is None:
            return SendResult(success=False, error="Aleph is not connected", retryable=True)
        attachments = []
        for path in file_paths:
            mime = mimetypes.guess_type(path)[0] or ""
            kind = ("image" if mime.startswith("image/") else "video" if mime.startswith("video/")
                    else "audio" if mime.startswith("audio/") else "file")
            try:
                attachments.append(await self._register_agent_file(path, kind=kind))
            except (OSError, ValueError) as exc:
                logger.warning("Aleph: cannot attach %s: %s", path, exc)
        if not attachments and not caption.strip():
            return SendResult(success=False, error="no readable files")
        return await self._send_message(chat_id, caption, attachments)

    # ── misc ───────────────────────────────────────────────────────────────────────────────────

    async def get_chat_info(self, chat_id: str) -> dict[str, Any]:
        chat = str(chat_id or CHAT_ID)
        side = self._chats.get(chat)
        return {"name": (side.title or "Side chat") if side is not None else CHAT_NAME, "type": "dm", "chat_id": chat}

    def format_message(self, content: str) -> str:
        return content  # Markdown goes through untouched; Aleph renders it

