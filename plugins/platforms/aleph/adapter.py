"""Aleph platform adapter: the iOS app as a Hermes messaging channel.

One chat (``main``, a DM) between one person and the agent, spoken over Aleph's own WebSocket
protocol (Aleph repo, ``docs/protocol.md``). Everything that changes the thread is logged with a
per-chat ``seq`` so a phone that was offline catches up; drafts and typing are live only.

Map onto the base adapter:

- inbound ``message.send`` -> ``handle_message`` (idempotent by the phone's message id), echoed as
  ``message.new``;
- ``send`` / ``edit_message`` / ``delete_message`` -> ``message.new`` / ``.edit`` / ``.delete``;
- ``send_draft`` -> live ``draft`` (whole text); the next text ``send`` carries its ``draft_id``;
- ``send_typing`` / ``stop_typing`` / status text -> live ``typing``;
- ``_send_exec_approval_prompt`` / ``send_clarify`` -> ``approval.request`` / ``clarify.request``;
- ``on_processing_start`` -> ``receipt`` (the turn that takes the person's messages has started).

Everything to and from the phone is end-to-end encrypted with Noise (``noise.py``): a device pairs
once by scanning a QR code (``hermes aleph pair``, then the ``/aleph/v1/pair`` handshake) and is known
by its static key from then on.
"""

from __future__ import annotations

import asyncio
import logging
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
    BasePlatformAdapter, ExecApprovalPrompt, MessageEvent, MessageType, SendResult)
from gateway.platforms.helpers import MessageDeduplicator

from plugins.platforms.aleph import noise
from plugins.platforms.aleph.server import (
    INVALID_PARAMS, METHOD_NOT_FOUND, AlephServer, Connection, RpcError, event_frame)
from plugins.platforms.aleph.store import AlephStore, Device, LoggedEvent, load_or_create_server_key

logger = logging.getLogger(__name__)

PLATFORM_NAME = "aleph"
CHAT_ID = "main"
CHAT_NAME = "Aleph"
# Every paired device is the owner's: every inbound message is from this one sender.
OWNER_ID = "owner"
# The app's default name for the agent (hello and the pairing answer); the person may rename it
# on the phone, which then shows that name.
AGENT_NAME = "Aleph"
OWNER_NAME = "Owner"
DEFAULT_PORT = 8650
DEFAULT_HOST = "127.0.0.1"
DEFAULT_HISTORY_LIMIT = 20000
# Generous: Aleph renders long Markdown itself, and drafts stream the whole reply anyway.
MAX_MESSAGE_LENGTH = 32000
MAX_TEXT_CHARS = 64000
# How often the gateway reloads paired keys and the pending pairing written by the CLI (another
# process): a revoked device is cut off, and a new pairing becomes reachable, within this.
DEVICE_POLL_SECONDS = 1.0
LAST_SEEN_INTERVAL = 60.0

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


@dataclass(frozen=True)
class ReplaySnapshot:
    latest: int
    reset: bool
    events: list[LoggedEvent]


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
        self.state_dir = resolve_state_dir(extra)
        self._device_watch: Optional[asyncio.Task] = None
        self._last_seen: dict[str, float] = {}
        self.static: Optional[noise.KeyPair] = None
        # What the handshakes may consult before anyone is authenticated: memory only.
        self._device_keys: dict[bytes, Device] = {}
        self._pending: Optional[tuple[bytes, float]] = None
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
        # Receipt watermark: the newest of the person's messages handed to the gateway.
        self._last_user_message: Optional[str] = None
        self._last_receipt: Optional[str] = None

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
        await self._reload_devices()
        self._last_user_message = await asyncio.to_thread(self._store.meta, "last_user_message")
        self._last_receipt = await asyncio.to_thread(self._store.meta, "last_receipt")
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
        logger.info("Aleph: %d paired device(s)", len(self._device_keys))
        self._device_watch = asyncio.create_task(self._watch_devices())
        self._mark_connected(listener_base=f"http://{self.host}:{port}")
        return True

    async def disconnect(self) -> None:
        server, store, watch = self._server, self._store, self._device_watch
        self._server = self._store = self._device_watch = None
        if watch is not None:
            watch.cancel()
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
                   message: Optional[dict[str, Any]] = None) -> LoggedEvent:
        """Append one logged event and broadcast it, in seq order."""
        async with self.append_lock:
            event = await asyncio.to_thread(self.store.append, event_type, payload, message=message)
            if self._server is not None:
                self._server.broadcast(event_frame(event.type, event.payload, event.seq))
        return event

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
        await self._log("message.new", {"id": message_id, "sender": "agent", "text": text, "attachments": [],
                                        "created_at": now_iso(), "kind": "system"},
                        message={"id": message_id, "sender": "agent", "kind": "system", "text": text,
                                 "attachments": []})
        return {"device_id": device.id, "agent": {"name": self.agent_info()["name"]}}

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

    def _live(self, event_type: str, payload: dict[str, Any]) -> None:
        """Broadcast a live-only event; safe from any thread (status text arrives off-loop)."""
        def emit() -> None:
            if self._server is not None:
                self._server.broadcast(event_frame(event_type, payload))
        self._on_loop(emit)

    def live_state_frames(self) -> list[str]:
        """What a phone connecting mid-turn needs to show the present: typing and the open draft."""
        frames = []
        for chat, status in self._typing.items():
            if chat == self.chat_id:
                payload: dict[str, Any] = {"active": True}
                if status:
                    payload["status"] = status
                frames.append(event_frame("typing", payload))
        open_draft = self._open_draft.get(self.chat_id)
        if open_draft is not None:
            frames.append(event_frame("draft", {"draft_id": open_draft[0], "text": open_draft[1]}))
        return frames

    async def replay_snapshot(self, after: int) -> ReplaySnapshot:
        """Under ``append_lock``: the events a phone that stored ``after`` is missing."""
        latest = await asyncio.to_thread(self.store.latest)
        oldest = await asyncio.to_thread(self.store.oldest)
        reset = False
        start = after
        if after > latest:
            # The phone is ahead of this log (a wiped or replaced server): start over.
            reset, start = True, 0
        elif oldest is not None and after < oldest - 1:
            reset, start = True, 0
        events = await asyncio.to_thread(self.store.events_after, start, until=latest) if latest > start else []
        return ReplaySnapshot(latest, reset, events)

    def agent_info(self) -> dict[str, Any]:
        info: dict[str, Any] = {"name": AGENT_NAME}
        model = self._configured_model()
        if model:
            info["model"] = model
        return info

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
        if method == "message.send":
            return {"seq": await self._accept_user_message(params)}
        if method == "chat.stop":
            await self._stop()
            return {}
        if method == "approval.respond":
            await self._respond_approval(params)
            return {}
        if method == "clarify.respond":
            await self._respond_clarify(params)
            return {}
        if method == "ping":
            return {}
        raise RpcError(METHOD_NOT_FOUND, f"method not found: {method}")

    async def _accept_user_message(self, params: dict[str, Any]) -> int:
        message_id = params.get("id")
        if not isinstance(message_id, str) or not message_id.strip() or len(message_id) > 128:
            raise RpcError(INVALID_PARAMS, "id must be a non-empty string")
        text = params.get("text") or ""
        if not isinstance(text, str) or len(text) > MAX_TEXT_CHARS:
            raise RpcError(INVALID_PARAMS, "text must be a string")
        if params.get("attachments"):
            raise RpcError(INVALID_PARAMS, "attachments are not supported")
        reply_to = params.get("reply_to")
        if reply_to is not None and not isinstance(reply_to, str):
            raise RpcError(INVALID_PARAMS, "reply_to must be a message id or null")
        async with self._intake_lock:
            existing = await asyncio.to_thread(self.store.message, message_id)
            if existing is not None:
                if existing.sender != "user":
                    raise RpcError(INVALID_PARAMS, "id belongs to another message")
                # A resend after a lost answer: answer as the first time; hand it over if the first
                # attempt was logged but never reached the gateway.
                self._dedup.is_duplicate(message_id)
                if not existing.dispatched:
                    await self._dispatch_user_message(message_id, existing.text, reply_to)
                return existing.seq
            if not text.strip():
                raise RpcError(INVALID_PARAMS, "a message needs text")
            self._dedup.is_duplicate(message_id)
            payload = {"id": message_id, "sender": "user", "text": text, "attachments": [], "created_at": now_iso()}
            event = await self._log("message.new", payload, message={
                "id": message_id, "sender": "user", "kind": "text", "text": text, "attachments": []})
            await self._dispatch_user_message(message_id, text, reply_to)
            return event.seq

    async def _dispatch_user_message(self, message_id: str, text: str, reply_to: Optional[str]) -> None:
        source = self.build_source(chat_id=self.chat_id, chat_name=CHAT_NAME, chat_type="dm",
                                   user_id=OWNER_ID, user_name=OWNER_NAME)
        event = MessageEvent(source=source, text=text, message_type=MessageType.TEXT, message_id=message_id)
        if reply_to:
            replied = await asyncio.to_thread(self.store.message, reply_to)
            if replied is not None:
                event.reply_to_message_id = reply_to
                event.reply_to_text = replied.text
                event.reply_to_is_own_message = replied.sender == "agent"
        # Recorded before the hand-off: a turn may start (and send its receipt) inside it.
        self._last_user_message = message_id
        await asyncio.to_thread(self.store.set_meta, "last_user_message", message_id)
        await self.handle_message(event)
        await asyncio.to_thread(self.store.mark_dispatched, message_id)

    async def _stop(self) -> None:
        """``chat.stop``: the gateway's own ``/stop`` for this chat (not logged as a message)."""
        source = self.build_source(chat_id=self.chat_id, chat_name=CHAT_NAME, chat_type="dm",
                                   user_id=OWNER_ID, user_name=OWNER_NAME)
        await self.handle_message(MessageEvent(source=source, text="/stop", message_type=MessageType.COMMAND,
                                               message_id=f"stop-{uuid.uuid4().hex}"))

    # ── receipts ───────────────────────────────────────────────────────────────────────────────

    async def on_processing_start(self, event: MessageEvent) -> None:
        """A turn started on the person's input: everything they sent so far is being read."""
        source = getattr(event, "source", None)
        if getattr(event, "internal", False) or getattr(source, "platform", None) != self.platform:
            return
        if str(getattr(source, "chat_id", "")) != self.chat_id or self._store is None:
            return
        target = self._last_user_message
        if not target or target == self._last_receipt:
            return
        if await asyncio.to_thread(self.store.user_message_seq, target) is None:
            return
        self._last_receipt = target
        await asyncio.to_thread(self.store.set_meta, "last_receipt", target)
        await self._log("receipt", {"read_up_to": target, "at": now_iso()})

    async def on_processing_complete(self, event: MessageEvent, outcome: Any) -> None:
        source = getattr(event, "source", None)
        if getattr(source, "platform", None) != self.platform:
            return
        chat = str(getattr(source, "chat_id", "") or self.chat_id)
        self._clear_draft(chat)
        self._set_typing(chat, active=False)

    # ── outbound text ──────────────────────────────────────────────────────────────────────────

    async def send(self, chat_id: str, content: str, reply_to: Optional[str] = None,
                   metadata: Optional[dict[str, Any]] = None) -> SendResult:
        if self._store is None:
            return SendResult(success=False, error="Aleph is not connected", retryable=True)
        meta = metadata or {}
        kind = "tool" if meta.get("_tool_progress") else "system" if meta.get("_interim_send") else "text"
        return await self._send_message(str(chat_id or self.chat_id), content or "", kind=kind)

    async def _send_message(self, chat_id: str, text: str, *, kind: str = "text") -> SendResult:
        message_id = str(uuid.uuid4())
        payload: dict[str, Any] = {"id": message_id, "sender": "agent", "text": text,
                                   "attachments": [], "created_at": now_iso()}
        if kind != "text":
            payload["kind"] = kind
        else:
            # The reply's real message replaces its draft on the phone.
            open_draft = self._open_draft.pop(chat_id, None)
            if open_draft is not None:
                payload["draft_id"] = open_draft[0]
        try:
            await self._log("message.new", payload, message={
                "id": message_id, "sender": "agent", "kind": kind, "text": text, "attachments": []})
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
            await asyncio.to_thread(self.store.update_message_text, stored.id, content)
            await self._log("message.edit", {"id": stored.id, "text": content})
        return SendResult(success=True, message_id=stored.id)

    async def delete_message(self, chat_id: str, message_id: str) -> bool:
        if self._store is None:
            return False
        stored = await asyncio.to_thread(self.store.message, str(message_id))
        if stored is None or stored.sender != "agent":
            return False
        await asyncio.to_thread(self.store.forget_message, stored.id)
        await self._log("message.delete", {"id": stored.id})
        return True

    # ── drafts ─────────────────────────────────────────────────────────────────────────────────

    def supports_draft_streaming(self, chat_type: Optional[str] = None, metadata: Optional[dict[str, Any]] = None,
                                 chat_id: Optional[str] = None) -> bool:
        return True

    async def send_draft(self, chat_id: str, draft_id: int, content: str,
                         metadata: Optional[dict[str, Any]] = None) -> SendResult:
        chat = str(chat_id or self.chat_id)
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
            self._live("draft.clear", {"draft_id": previous[0]})
        self._open_draft[chat] = (wire_id, content)
        self._live("draft", {"draft_id": wire_id, "text": content})
        return SendResult(success=True)

    async def abandon_open_draft(self, chat_id: str, content: str,
                                 metadata: Optional[dict[str, Any]] = None) -> SendResult:
        """The draft's turn died (/stop, /new, an error): the phone drops it."""
        self._clear_draft(str(chat_id or self.chat_id))
        return SendResult(success=True)

    def _clear_draft(self, chat: str) -> None:
        open_draft = self._open_draft.pop(chat, None)
        if open_draft is not None:
            self._live("draft.clear", {"draft_id": open_draft[0]})

    # ── typing ─────────────────────────────────────────────────────────────────────────────────

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        chat = str(chat_id or self.chat_id)
        status = (getattr(self, "_status_text", None) or {}).get(chat)
        self._set_typing(chat, active=True, status=status)

    async def stop_typing(self, chat_id: str) -> None:
        self._set_typing(str(chat_id or self.chat_id), active=False)

    def set_status_text(self, chat_id: str, text: Optional[str]) -> None:
        """Called from the agent's worker thread: the typing state is only touched on the loop."""
        super().set_status_text(chat_id, text)
        chat, status = str(chat_id), text or None

        def apply() -> None:
            if chat in self._typing and self._typing.get(chat) != status:
                self._set_typing(chat, active=True, status=status)
        self._on_loop(apply)

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
            payload: dict[str, Any] = {"active": True}
            if status:
                payload["status"] = status
            self._live("typing", payload)
        elif self._typing.pop(chat, "absent") != "absent":
            self._live("typing", {"active": False})

    # ── approvals ──────────────────────────────────────────────────────────────────────────────

    async def _send_exec_approval_prompt(self, prompt: ExecApprovalPrompt) -> SendResult:
        if self._store is None:
            return SendResult(success=False, error="Aleph is not connected", retryable=True)
        request_id = prompt.request_id or uuid.uuid4().hex
        choices = [{"label": label, "choice": choice, "style": _APPROVAL_STYLES.get(style, "default")}
                   for label, choice, style in prompt.actions]
        payload: dict[str, Any] = {"request_id": request_id, "text": prompt.text, "choices": choices}
        if prompt.command:
            payload["command"] = prompt.command
        if prompt.description:
            payload["reason"] = prompt.description
        self._approvals[request_id] = {"session_key": prompt.session_key, "choices": prompt.choices,
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
        await self._log("approval.resolved", {"request_id": request_id, "choice": None, "by": by})

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
        if resolved:
            await self._log("approval.resolved", {"request_id": request_id, "choice": choice, "by": "user"})
        else:
            await self._log("approval.resolved", {"request_id": request_id, "choice": None, "by": "timeout"})

    # ── clarify ────────────────────────────────────────────────────────────────────────────────

    async def send_clarify(self, chat_id: str, question: str, choices: Optional[list], clarify_id: str,
                           session_key: str, metadata: Optional[dict[str, Any]] = None) -> SendResult:
        if self._store is None:
            return SendResult(success=False, error="Aleph is not connected", retryable=True)
        labels = [str(choice) for choice in (choices or [])]
        entry = None
        try:
            from tools.clarify_gateway import get_pending
            entry = get_pending(clarify_id)
        except ImportError:
            pass
        # The entry is kept to read the answer when the clarify ends some other way.
        self._clarifies[clarify_id] = {"choices": labels, "answered": False, "entry": entry}
        await self._log("clarify.request", {"clarify_id": clarify_id, "question": question,
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
        await self._log("clarify.resolved", {"clarify_id": clarify_id, "answer": answer,
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
        if resolved:
            await self._log("clarify.resolved", {"clarify_id": clarify_id, "answer": answer, "by": "user"})
        else:
            await self._log("clarify.resolved", {"clarify_id": clarify_id, "answer": None, "by": "timeout"})

    # ── misc ───────────────────────────────────────────────────────────────────────────────────

    async def get_chat_info(self, chat_id: str) -> dict[str, Any]:
        return {"name": CHAT_NAME, "type": "dm", "chat_id": str(chat_id or self.chat_id)}

    def format_message(self, content: str) -> str:
        return content  # Markdown goes through untouched; Aleph renders it

