"""Aleph's listener: the encrypted WebSocket, pairing, and ticketed file transfer.

- ``/aleph/v1/ws``: ``Noise_IK_25519_ChaChaPoly_SHA256`` (prologue ``aleph/1``); afterwards each binary
  message is one JSON-RPC 2.0 frame encrypted with that direction's transport key (up to 16 MB). An
  unknown or revoked key, or a broken handshake, closes with ``4401``. Device keys are looked up in
  memory, never in the database, before the peer is authenticated.
- ``/aleph/v1/pair``: ``Noise_IKpsk1`` with the pending pairing secret as PSK; ``404`` unless a pairing is
  pending, 5 attempts a minute per address, ``4403`` for a wrong secret.
- ``PUT`` / ``GET /aleph/v1/files/<ticket>``: ciphertext only, against a one-time ticket issued over the
  WebSocket (``file.ticket``).

Frames on a connection go out through one queue drained by one writer task, which is also what
encrypts them, so nonces follow the send order and a call's answer, the replay and live events can
never interleave. ``hello`` enqueues its answer, the replay and ``sync.done`` while holding the
adapter's append lock, then subscribes the connection: an event logged meanwhile waits for the lock
and reaches the connection after ``sync.done``, never inside the replay and never twice.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import time
from collections import deque
from typing import TYPE_CHECKING, Any, Optional

from aiohttp import WSMsgType, web

from plugins.platforms.aleph import noise

if TYPE_CHECKING:
    from plugins.platforms.aleph.adapter import AlephAdapter

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = 1
SUPPORTED_PROTOCOLS = [PROTOCOL_VERSION]
BASE_PATH = "/aleph/v1"

# JSON-RPC error codes (docs/protocol.md "Frames").
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INVALID_REQUEST = -32600
PARSE_ERROR = -32700
INTERNAL_ERROR = -32603
NOT_READY = 4001
BUSY = 4002

# WebSocket close codes (4000-4999 is application use).
CLOSE_UNAUTHORIZED = 4401  # unknown or revoked device key, a broken or slow handshake, revoked later
CLOSE_BAD_SECRET = 4403  # pairing: wrong or expired secret
CLOSE_PROTOCOL = 1002  # a transport frame that does not decrypt, or a text frame

HANDSHAKE_SECONDS = 10.0
# A handshake's first message is small (96 bytes to connect, a little more to pair).
MAX_HANDSHAKE_MESSAGE = 4096
# Frames stay under 16 MB; the AEAD tag and a little slack on top.
MAX_FRAME_BYTES = 16 * 1024 * 1024
MAX_WS_MESSAGE = MAX_FRAME_BYTES + 1024

# A connection whose queue grows past this is too slow to keep up: it is closed, and the phone
# reconnects and catches up from its last stored seq.
MAX_QUEUED_FRAMES = 20000
MAX_UPLOAD_BYTES = 200 * 1024 * 1024 + 16

# Pairing attempts allowed from one address per window (protocol "Pairing").
PAIR_ATTEMPTS = 5
PAIR_WINDOW_SECONDS = 60.0
MAX_DEVICE_FIELD = 100


class RpcError(Exception):
    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(message)
        self.code, self.message, self.data = code, message, data


def _frame(obj: dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def event_frame(event_type: str, payload: dict[str, Any], seq: Optional[int] = None) -> str:
    params: dict[str, Any] = {"type": event_type}
    if seq is not None:
        params["seq"] = seq
    params["payload"] = payload
    return _frame({"jsonrpc": "2.0", "method": "event", "params": params})


def client_address(request: web.Request) -> str:
    """The peer's address. Behind the tunnel every request comes from loopback (cloudflared), so
    there, and only there, Cloudflare's ``CF-Connecting-IP`` names the real client."""
    remote = request.remote or "?"
    try:
        loopback = ipaddress.ip_address(remote).is_loopback
    except ValueError:
        loopback = False
    if loopback:
        forwarded = request.headers.get("CF-Connecting-IP", "").strip()
        if forwarded and len(forwarded) <= 64:
            return forwarded
    return remote


class PairRateLimiter:
    """At most ``PAIR_ATTEMPTS`` pairing attempts per address in any ``PAIR_WINDOW_SECONDS``."""

    def __init__(self, attempts: int = PAIR_ATTEMPTS, window: float = PAIR_WINDOW_SECONDS, clock=time.monotonic) -> None:
        self.attempts, self.window, self._clock = attempts, window, clock
        self._seen: dict[str, deque[float]] = {}

    def allow(self, address: str) -> Optional[float]:
        """``None`` if this attempt may go ahead (and counts it), else seconds until one may."""
        now = self._clock()
        if len(self._seen) > 4096:
            for key in [k for k, q in self._seen.items() if not q or now - q[-1] >= self.window]:
                del self._seen[key]
        times = self._seen.setdefault(address, deque())
        while times and now - times[0] >= self.window:
            times.popleft()
        if len(times) >= self.attempts:
            return max(0.0, self.window - (now - times[0]))
        times.append(now)
        return None


class Connection:
    """One phone connection after its handshake: a bounded outbound queue and the writer task that
    encrypts and sends it."""

    def __init__(self, ws: web.WebSocketResponse, peer: str, transport: noise.Transport,
                 device_id: str = "", device_name: str = "") -> None:
        self.ws = ws
        self.peer = peer
        self.transport = transport
        self.device_id = device_id
        self.device_name = device_name
        self.ready = False  # hello answered
        self.client: dict[str, Any] = {}
        self._queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
        self._writer = asyncio.create_task(self._write_loop())

    def enqueue(self, text: str) -> None:
        if self._writer.done():
            return
        if self._queue.qsize() >= MAX_QUEUED_FRAMES:
            logger.warning("Aleph: connection %s fell %d frames behind; closing it", self.peer, MAX_QUEUED_FRAMES)
            self._queue.put_nowait(None)
            return
        self._queue.put_nowait(text)

    def decrypt(self, data: bytes) -> str:
        """One inbound frame. Raises ``NoiseError`` (or ``UnicodeDecodeError``): the connection ends."""
        return self.transport.decrypt(data).decode("utf-8")

    async def _write_loop(self) -> None:
        try:
            while True:
                text = await self._queue.get()
                if text is None:
                    await self.ws.close(code=1013, message=b"too far behind")
                    return
                data = text.encode("utf-8")
                if len(data) > MAX_FRAME_BYTES:
                    logger.error("Aleph: dropping a %d-byte frame to %s (over 16 MB)", len(data), self.peer)
                    continue
                await self.ws.send_bytes(self.transport.encrypt(data))
        except (ConnectionResetError, RuntimeError, asyncio.CancelledError):
            pass
        except Exception:
            logger.debug("Aleph: writer for %s stopped", self.peer, exc_info=True)

    async def close(self, *, code: int = 1000, message: bytes = b"") -> None:
        self._writer.cancel()
        try:
            await self._writer
        except BaseException:
            pass
        if not self.ws.closed:
            await self.ws.close(code=code, message=message)


class AlephServer:
    """The aiohttp application for one adapter."""

    def __init__(self, adapter: AlephAdapter, *, host: str, port: int, static: noise.KeyPair) -> None:
        self.adapter = adapter
        self.host, self.port = host, port
        self.static = static
        self.connections: set[Connection] = set()
        self.pair_limiter = PairRateLimiter()
        self._runner: Optional[web.AppRunner] = None
        app = web.Application(client_max_size=MAX_UPLOAD_BYTES)
        app.router.add_get(f"{BASE_PATH}/ws", self._ws_handler)
        app.router.add_get(f"{BASE_PATH}/pair", self._pair_handler)
        app.router.add_put(f"{BASE_PATH}/files/{{ticket}}", self._upload_handler)
        app.router.add_get(f"{BASE_PATH}/files/{{ticket}}", self._download_handler)
        self.app = app

    # ── lifecycle ──────────────────────────────────────────────────────────────────────────────

    async def start(self) -> None:
        self._runner = web.AppRunner(self.app, access_log=None, handle_signals=False)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.host, self.port, reuse_address=True)
        try:
            await site.start()
        except BaseException:
            await self._runner.cleanup()
            self._runner = None
            raise

    async def stop(self) -> None:
        for conn in list(self.connections):
            await conn.close()
        self.connections.clear()
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    @property
    def bound_port(self) -> Optional[int]:
        """The port actually bound (tests bind port 0)."""
        if self._runner is None:
            return None
        for site in self._runner.sites:
            server = getattr(site, "_server", None)
            for sock in getattr(server, "sockets", None) or ():
                return sock.getsockname()[1]
        return None

    # ── fan-out ────────────────────────────────────────────────────────────────────────────────

    def broadcast(self, text: str) -> None:
        """Queue a frame on every subscribed connection (event-loop thread only)."""
        for conn in list(self.connections):
            if conn.ready:
                conn.enqueue(text)

    async def close_devices(self, keep: set[str]) -> int:
        """Close every connection whose device is not in ``keep`` (it was revoked)."""
        gone = [conn for conn in list(self.connections) if conn.device_id not in keep]
        for conn in gone:
            self.connections.discard(conn)
            logger.info("Aleph: closing %s: device %s was revoked", conn.peer, conn.device_id)
            await conn.close(code=CLOSE_UNAUTHORIZED, message=b"revoked")
        return len(gone)

    # ── handshakes ─────────────────────────────────────────────────────────────────────────────

    @staticmethod
    async def _first_message(ws: web.WebSocketResponse, deadline: float) -> Optional[bytes]:
        """The handshake's first message: one small binary frame before ``deadline``, else ``None``."""
        try:
            msg = await asyncio.wait_for(ws.receive(), max(0.0, deadline - time.monotonic()))
        except TimeoutError:
            return None
        if msg.type != WSMsgType.BINARY or len(msg.data) > MAX_HANDSHAKE_MESSAGE:
            return None
        return bytes(msg.data)

    async def _ws_handler(self, request: web.Request) -> web.StreamResponse:
        address = client_address(request)
        deadline = time.monotonic() + HANDSHAKE_SECONDS
        ws = web.WebSocketResponse(heartbeat=30.0, max_msg_size=MAX_WS_MESSAGE)
        await ws.prepare(request)
        message = await self._first_message(ws, deadline)
        device = None
        handshake = noise.responder(noise.IK, noise.PROLOGUE, self.static)
        if message is not None:
            try:
                handshake.read_message(message)
                device = self.adapter.known_device(handshake.rs)  # in memory: no database yet
            except noise.NoiseError:
                device = None
        if device is None or time.monotonic() > deadline:
            await ws.close(code=CLOSE_UNAUTHORIZED, message=b"unauthorized")
            return ws
        await ws.send_bytes(handshake.write_message(b""))
        conn = Connection(ws, address, handshake.transport(), device.id, device.name)
        self.connections.add(conn)
        await self.adapter.device_seen(device)
        logger.info("Aleph: %s (%s) connected", device.name, device.id)
        try:
            async for msg in ws:
                if msg.type == WSMsgType.BINARY:
                    try:
                        text = conn.decrypt(msg.data)
                    except (noise.NoiseError, UnicodeDecodeError):
                        logger.warning("Aleph: a frame from %s did not decrypt; closing", device.id)
                        await conn.close(code=CLOSE_PROTOCOL, message=b"bad frame")
                        break
                    await self._handle_frame(conn, text)
                elif msg.type == WSMsgType.TEXT:
                    await conn.close(code=CLOSE_PROTOCOL, message=b"binary frames only")
                    break
                elif msg.type in (WSMsgType.ERROR, WSMsgType.CLOSE):
                    break
        finally:
            self.connections.discard(conn)
            await conn.close()
            logger.info("Aleph: %s disconnected", device.name)
        return ws

    async def _pair_handler(self, request: web.Request) -> web.StreamResponse:
        if self.adapter.pending_pairing() is None:  # in memory: no database yet
            raise web.HTTPNotFound()
        address = client_address(request)
        wait = self.pair_limiter.allow(address)
        if wait is not None:
            logger.warning("Aleph: pairing attempts are being rate limited")
            return web.json_response({"error": "too many attempts"}, status=429,
                                     headers={"Retry-After": str(max(1, int(wait + 0.999)))})
        deadline = time.monotonic() + HANDSHAKE_SECONDS
        ws = web.WebSocketResponse(max_msg_size=MAX_HANDSHAKE_MESSAGE)
        await ws.prepare(request)
        message = await self._first_message(ws, deadline)
        pending = self.adapter.pending_pairing()
        if pending is None or message is None:
            await ws.close(code=CLOSE_BAD_SECRET, message=b"no pairing")
            return ws
        secret = pending[0]
        handshake = noise.responder(noise.IKPSK1, noise.PAIR_PROLOGUE, self.static, psk=secret)
        try:
            payload = handshake.read_message(message)
        except noise.NoiseError:
            logger.warning("Aleph: a pairing attempt used the wrong secret")
            await ws.close(code=CLOSE_BAD_SECRET, message=b"wrong secret")
            return ws
        # Authenticated: the payload decrypted under the secret. Only now is it parsed.
        name, model = _device_fields(payload)
        result = await self.adapter.complete_pairing(secret, handshake.rs, name=name, model=model)
        if result is None:  # replaced or spent meanwhile
            await ws.close(code=CLOSE_BAD_SECRET, message=b"wrong secret")
            return ws
        await ws.send_bytes(handshake.write_message(json.dumps(result, separators=(",", ":")).encode()))
        await ws.close()
        return ws

    # ── files ──────────────────────────────────────────────────────────────────────────────────

    async def _upload_handler(self, request: web.Request) -> web.Response:
        ticket = request.match_info["ticket"]
        expected = self.adapter.upload_size(ticket)
        if expected is None:
            raise web.HTTPNotFound()
        if request.content_length is not None and request.content_length != expected:
            return web.json_response({"error": f"expected {expected} bytes"}, status=400)
        try:
            data = await request.read()
        except web.HTTPRequestEntityTooLarge:
            return web.json_response({"error": "file too large"}, status=413)
        if len(data) != expected:
            return web.json_response({"error": f"expected {expected} bytes"}, status=400)
        if not await self.adapter.accept_upload(ticket, data):
            raise web.HTTPNotFound()
        return web.Response(status=204)

    async def _download_handler(self, request: web.Request) -> web.StreamResponse:
        path = self.adapter.redeem_download(request.match_info["ticket"])
        if path is None:
            raise web.HTTPNotFound()
        return web.FileResponse(path, headers={"Content-Type": "application/octet-stream",
                                               "Cache-Control": "no-store"})

    # ── JSON-RPC ───────────────────────────────────────────────────────────────────────────────

    async def _handle_frame(self, conn: Connection, text: str) -> None:
        try:
            frame = json.loads(text)
        except ValueError:
            conn.enqueue(_frame({"jsonrpc": "2.0", "id": None,
                                 "error": {"code": PARSE_ERROR, "message": "parse error"}}))
            return
        if not isinstance(frame, dict) or not isinstance(frame.get("method"), str):
            conn.enqueue(_frame({"jsonrpc": "2.0", "id": frame.get("id") if isinstance(frame, dict) else None,
                                 "error": {"code": INVALID_REQUEST, "message": "invalid request"}}))
            return
        call_id = frame.get("id")
        method = frame["method"]
        params = frame.get("params")
        if params is None:
            params = {}
        try:
            if not isinstance(params, dict):
                raise RpcError(INVALID_PARAMS, "params must be an object")
            if method == "hello":
                # hello answers through the queue itself, ahead of the replay.
                await self._hello(conn, call_id, params)
                return
            if not conn.ready:
                raise RpcError(NOT_READY, "call hello first")
            result = await self.adapter.handle_call(method, params, conn)
        except RpcError as exc:
            if call_id is not None:
                error: dict[str, Any] = {"code": exc.code, "message": exc.message}
                if exc.data is not None:
                    error["data"] = exc.data
                conn.enqueue(_frame({"jsonrpc": "2.0", "id": call_id, "error": error}))
            return
        except Exception:
            logger.exception("Aleph: %s failed", method)
            if call_id is not None:
                conn.enqueue(_frame({"jsonrpc": "2.0", "id": call_id,
                                     "error": {"code": INTERNAL_ERROR, "message": "internal error"}}))
            return
        if call_id is not None:
            conn.enqueue(_frame({"jsonrpc": "2.0", "id": call_id, "result": result}))

    async def _hello(self, conn: Connection, call_id: Any, params: dict[str, Any]) -> None:
        if conn.ready:
            raise RpcError(INVALID_PARAMS, "hello was already called on this connection")
        protocol = params.get("protocol")
        if protocol not in SUPPORTED_PROTOCOLS:
            raise RpcError(INVALID_PARAMS, "unsupported protocol", {"supported": SUPPORTED_PROTOCOLS})
        after = params.get("after", 0)
        if isinstance(after, bool) or not isinstance(after, int) or after < 0:
            raise RpcError(INVALID_PARAMS, "after must be a non-negative integer")
        client = params.get("client")
        conn.client = client if isinstance(client, dict) else {}
        async with self.adapter.append_lock:
            snapshot = await self.adapter.replay_snapshot(after)
            result = {"protocol": PROTOCOL_VERSION, "chat": {"id": self.adapter.chat_id},
                      "agent": self.adapter.agent_info(), "latest": snapshot.latest, "reset": snapshot.reset}
            conn.enqueue(_frame({"jsonrpc": "2.0", "id": call_id, "result": result}))
            for event in snapshot.events:
                conn.enqueue(event_frame(event.type, event.payload, event.seq))
            conn.enqueue(event_frame("sync.done", {"seq": snapshot.latest}))
            conn.ready = True
        for text in self.adapter.live_state_frames():
            conn.enqueue(text)
        logger.info("Aleph: %s caught up from %d to %d%s (%s)", conn.peer, after, snapshot.latest,
                    " after a reset" if snapshot.reset else "", conn.client.get("device") or "unknown device")


def _device_fields(payload: bytes) -> tuple[str, str]:
    """``{"name", "model"}`` from the pairing payload, each optional and at most 100 characters."""
    try:
        fields = json.loads(payload.decode("utf-8")) if payload else {}
    except (ValueError, UnicodeDecodeError):
        fields = {}
    fields = fields if isinstance(fields, dict) else {}
    name, model = fields.get("name"), fields.get("model")
    name = name.strip()[:MAX_DEVICE_FIELD] if isinstance(name, str) else ""
    model = model.strip()[:MAX_DEVICE_FIELD] if isinstance(model, str) else ""
    return name or model or "Aleph", model
