"""A minimal Aleph phone for the plugin's tests: Noise IK handshake, encrypted JSON-RPC frames,
local pairing (Noise IKpsk1) and encrypted file transfer, all through the real listener."""

from __future__ import annotations

import asyncio
import json
import os
import time

import aiohttp

from plugins.platforms.aleph import noise
from plugins.platforms.aleph.files import decrypt_file, encrypt_file
from plugins.platforms.aleph.noise import b64url


def base_url(adapter):
    return f"http://127.0.0.1:{adapter._server.bound_port}"


def url(adapter, path):
    return f"{base_url(adapter)}/aleph/v1{path}"


def mac_headers(adapter, path, *, ts=None, nonce=None, key=None):
    """The upgrade gate's ``X-Aleph-Mac`` for ``path`` (``/aleph/v1/ws`` or ``/aleph/v1/pair``)."""
    from plugins.platforms.aleph.server import mac_key, upgrade_mac
    ts = int(time.time()) if ts is None else ts
    nonce = nonce or b64url(os.urandom(16))
    return {"X-Aleph-Mac": upgrade_mac(key or mac_key(adapter.static.public), ts, nonce, path)}


def pair_device(adapter, name="Test phone", model="iPhone"):
    """Pair a device straight through the store, and tell the running gateway (what a finished
    pairing handshake records). Returns the device and its key pair."""
    keys = noise.KeyPair.generate()
    secret, _ = adapter.store.new_pairing()
    device = adapter.store.complete_pairing(secret, keys.public, name=name, model=model)
    adapter._device_keys[device.public_key] = device
    adapter._pending = None
    return device, keys


def start_pairing(adapter, **kw):
    """What `hermes aleph pair` does, with the gateway picking it up at once instead of on its poll."""
    secret, expires = adapter.store.new_pairing(**kw)
    adapter._pending = (secret, expires)
    return secret


class Phone:
    """Calls with ids, events collected in order; every frame encrypted."""

    def __init__(self, session, ws, transport):
        self.session, self.ws, self.transport = session, ws, transport
        self._next = 0
        self.pending: list = []

    async def send_frame(self, obj):
        await self.ws.send_bytes(self.transport.encrypt(json.dumps(obj).encode()))

    async def call(self, method, params=None):
        self._next += 1
        call_id = self._next
        await self.send_frame({"jsonrpc": "2.0", "id": call_id, "method": method, "params": params or {}})
        while True:
            frame = await self.recv()
            if frame.get("id") == call_id:
                return frame
            self.pending.append(frame)

    async def recv(self, timeout=5.0):
        msg = await asyncio.wait_for(self.ws.receive(), timeout)
        assert msg.type == aiohttp.WSMsgType.BINARY, msg
        return json.loads(self.transport.decrypt(msg.data))

    async def event(self, timeout=5.0):
        """The next event, including any that arrived while waiting for a call's answer."""
        if self.pending:
            return self.pending.pop(0)
        return await self.recv(timeout)

    async def logged(self):
        """The next logged event, skipping live ones (typing refreshes, drafts)."""
        while True:
            frame = await self.event()
            if "seq" in frame["params"]:
                return frame

    async def events_until(self, event_type):
        seen = []
        while True:
            frame = await self.event()
            seen.append(frame["params"])
            if frame["params"]["type"] == event_type:
                return seen

    async def hello(self, after=0):
        answer = await self.call("hello", {"protocol": 1, "after": after,
                                           "client": {"name": "Aleph", "version": "0.1", "device": "test"}})
        replay = await self.events_until("sync.done")
        return answer, replay

    async def upload(self, adapter, data: bytes, *, name: str, mime: str) -> dict:
        """``file.ticket`` + ``PUT``: the ``message.send`` attachment for ``data``."""
        key, ciphertext = encrypt_file(data)
        answer = await self.call("file.ticket", {"size": len(ciphertext)})
        ticket = answer["result"]
        async with self.session.put(base_url(adapter) + ticket["url"], data=ciphertext) as resp:
            assert resp.status == 204, await resp.text()
        return {"ticket": ticket["ticket"], "key": b64url(key), "name": name, "mime": mime}

    async def download(self, adapter, attachment: dict) -> bytes:
        from plugins.platforms.aleph.noise import from_b64url
        async with self.session.get(base_url(adapter) + attachment["url"]) as resp:
            assert resp.status == 200, resp.status
            return decrypt_file(from_b64url(attachment["key"]), await resp.read())

    async def close(self):
        await self.ws.close()
        await self.session.close()


async def handshake(session, adapter, keys, *, server_key=None):
    """Open ``/ws`` and run IK; returns (ws, transport). The server's answer must decrypt."""
    ws = await session.ws_connect(url(adapter, "/ws"), headers=mac_headers(adapter, "/aleph/v1/ws"))
    hs = noise.initiator(noise.IK, noise.PROLOGUE, keys, server_key or adapter.static.public)
    await ws.send_bytes(hs.write_message(b""))
    msg = await asyncio.wait_for(ws.receive(), 5)
    if msg.type != aiohttp.WSMsgType.BINARY:
        await ws.close()
        raise ConnectionRefusedError(ws.close_code)
    hs.read_message(msg.data)
    return ws, hs.transport()


async def connect(adapter, keys) -> Phone:
    session = aiohttp.ClientSession()
    try:
        ws, transport = await handshake(session, adapter, keys)
    except BaseException:
        await session.close()
        raise
    return Phone(session, ws, transport)


async def run_pairing(session, adapter, secret: bytes, keys, payload: dict, *, headers=None,
                      server_key=None):
    """The phone's side of the pairing handshake: ``(answer payload, None)`` or ``(None, close code)``."""
    ws = await session.ws_connect(url(adapter, "/pair"),
                                  headers={**mac_headers(adapter, "/aleph/v1/pair"), **(headers or {})})
    hs = noise.initiator(noise.IKPSK1, noise.PAIR_PROLOGUE, keys, server_key or adapter.static.public, psk=secret)
    await ws.send_bytes(hs.write_message(json.dumps(payload).encode()))
    msg = await asyncio.wait_for(ws.receive(), 5)
    if msg.type != aiohttp.WSMsgType.BINARY:
        code = ws.close_code
        await ws.close()
        return None, code
    answer = json.loads(hs.read_message(msg.data))
    await ws.close()
    return answer, None
