"""What strangers can reach (protocol "Exposure"): the X-Aleph-Mac upgrade gate and the small,
strict handshake before authentication."""

from __future__ import annotations

import asyncio
import os
import time

import aiohttp
import pytest
import pytest_asyncio

from gateway.config import PlatformConfig
from plugins.platforms.aleph import noise
from tests.gateway.aleph_phone import (
    connect, handshake, mac_headers, pair_device, run_pairing, start_pairing, url as _url)



@pytest_asyncio.fixture
async def adapter(tmp_path, monkeypatch):
    from plugins.platforms.aleph import adapter as mod
    monkeypatch.setattr(mod, "DEVICE_POLL_SECONDS", 0.05)
    config = PlatformConfig(enabled=True, extra={"host": "127.0.0.1", "port": 0,
                                                 "state_dir": str(tmp_path / "aleph")})
    instance = mod.AlephAdapter(config)

    async def handler(event):
        return None
    instance.set_message_handler(handler)
    assert await instance.connect()
    yield instance
    await instance.disconnect()


def _pair_direct(adapter, name="Phone"):
    return pair_device(adapter, name=name)


async def _phone(adapter, keys):
    phone = await connect(adapter, keys)
    answer, _ = await phone.hello()
    assert "result" in answer, answer
    return phone
# ── the upgrade gate ─────────────────────────────────────────────────────────────────────────


def test_upgrade_mac_cross_check_vector():
    """The case the phone's tests also compute: server key 32 x 0x07, ts 1794200000, nonce 16 x 0xAB."""
    from tests.gateway.aleph_noise_vectors import build_vectors
    from plugins.platforms.aleph.noise import b64url
    from plugins.platforms.aleph.server import UpgradeGate, mac_key, upgrade_mac
    (vector,) = build_vectors()["upgrade_mac"]
    server_public, nonce = b"\x07" * 32, b64url(b"\xab" * 16)
    assert vector["server_static_public"] == server_public.hex()
    assert vector["mac_key"] == mac_key(server_public).hex()
    assert vector["header"] == upgrade_mac(mac_key(server_public), 1794200000, nonce, "/aleph/v1/ws")
    assert vector["header"].startswith("1794200000.q6urq6urq6urq6urq6urqw.")
    gate = UpgradeGate(server_public, clock=lambda: 1794200000 + 60)
    assert gate.check(vector["header"], "/aleph/v1/ws")


def test_upgrade_gate_rules():
    from plugins.platforms.aleph.noise import b64url
    from plugins.platforms.aleph.server import UpgradeGate, mac_key, upgrade_mac
    server = noise.KeyPair.generate().public
    now = [1_800_000_000.0]
    gate = UpgradeGate(server, clock=lambda: now[0])
    key = mac_key(server)

    def header(ts=None, nonce=None, path="/aleph/v1/ws", with_key=key):
        return upgrade_mac(with_key, int(now[0]) if ts is None else ts, nonce or b64url(os.urandom(16)), path)
    assert gate.check(header(), "/aleph/v1/ws")
    good = header()
    assert gate.check(good, "/aleph/v1/ws")
    assert not gate.check(good, "/aleph/v1/ws")  # a replayed nonce
    for bad in (None, "", "x", "1.2", "a.b.c", header()[:-2], header() + "=", "1" * 200,
                header(ts=int(now[0]) - 121), header(ts=int(now[0]) + 121), header(path="/aleph/v1/pair"),
                header(with_key=mac_key(noise.KeyPair.generate().public)), header(nonce=b64url(b"short"))):
        assert not gate.check(bad, "/aleph/v1/ws"), bad
    assert gate.check(header(ts=int(now[0]) - 120), "/aleph/v1/ws")
    # A nonce is forgotten after 240 seconds (by then its ts is stale anyway).
    fresh = header()
    assert gate.check(fresh, "/aleph/v1/ws")
    now[0] += 241
    assert gate.check(header(), "/aleph/v1/ws")
    assert fresh.split(".")[1] not in gate._seen


@pytest.mark.asyncio
async def test_without_a_valid_mac_the_endpoints_look_like_unknown_paths(adapter):
    start_pairing(adapter)
    async with aiohttp.ClientSession() as session:
        async with session.get(_url(adapter, "/nothing-here")) as resp:
            unknown = (resp.status, await resp.read(), resp.headers.get("Content-Type"))
        assert unknown[0] == 404
        stale = mac_headers(adapter, "/aleph/v1/ws", ts=int(time.time()) - 600)
        wrong_path = mac_headers(adapter, "/aleph/v1/pair")
        for path, headers in (("/ws", {}), ("/ws", stale), ("/ws", wrong_path), ("/pair", {}),
                              ("/pair", {"X-Aleph-Mac": "garbage"})):
            async with session.get(_url(adapter, path), headers={**headers, "Connection": "Upgrade",
                                                                 "Upgrade": "websocket"}) as resp:
                assert (resp.status, await resp.read(), resp.headers.get("Content-Type")) == unknown, path
        # A replayed upgrade header is refused too.
        once = mac_headers(adapter, "/aleph/v1/ws")
        ws = await session.ws_connect(_url(adapter, "/ws"), headers=once)
        await ws.close()
        with pytest.raises(aiohttp.WSServerHandshakeError) as info:
            await session.ws_connect(_url(adapter, "/ws"), headers=once)
        assert info.value.status == 404


@pytest.mark.asyncio
async def test_a_wrong_method_looks_like_an_unknown_path_too(adapter):
    async with aiohttp.ClientSession() as session:
        async with session.get(_url(adapter, "/nothing-here")) as resp:
            unknown = (resp.status, await resp.read(), resp.headers.get("Allow"))
        for method, path in (("POST", "/ws"), ("POST", "/pair"), ("DELETE", "/files/t_x"), ("PATCH", "/files/t_x")):
            async with session.request(method, _url(adapter, path)) as resp:
                assert (resp.status, await resp.read(), resp.headers.get("Allow")) == unknown, (method, path)


# ── the strict handshake ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_first_message_must_have_the_exact_size(adapter):
    _, keys = _pair_direct(adapter)
    async with aiohttp.ClientSession() as session:
        hs = noise.initiator(noise.IK, noise.PROLOGUE, keys, adapter.static.public)
        good = hs.write_message(b"")
        assert len(good) == 96
        for bad in (good[:95], good + b"\x00", os.urandom(96)):
            ws = await session.ws_connect(_url(adapter, "/ws"), headers=mac_headers(adapter, "/aleph/v1/ws"))
            await ws.send_bytes(bad)
            msg = await ws.receive(timeout=3)
            assert msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED) and ws.close_code == 4401
            await ws.close()
        # A text frame first.
        ws = await session.ws_connect(_url(adapter, "/ws"), headers=mac_headers(adapter, "/aleph/v1/ws"))
        await ws.send_str("x" * 96)
        await ws.receive(timeout=3)
        assert ws.close_code == 4401
        await ws.close()
    # Pairing: more than 512 bytes is cut off by the socket itself.
    start_pairing(adapter)
    async with aiohttp.ClientSession() as session:
        ws = await session.ws_connect(_url(adapter, "/pair"), headers=mac_headers(adapter, "/aleph/v1/pair"))
        await ws.send_bytes(os.urandom(600))
        msg = await ws.receive(timeout=3)
        assert msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED)
        await ws.close()
    assert len(adapter.store.devices()) == 1  # only the one paired above
    assert adapter.pending_pairing() is not None  # the oversized attempt did not spend it


@pytest.mark.asyncio
async def test_unauthenticated_connections_are_capped(adapter):
    from plugins.platforms.aleph import server as server_mod
    async with aiohttp.ClientSession() as session:
        async def open_raw(address):
            return await session.ws_connect(_url(adapter, "/ws"), headers={
                **mac_headers(adapter, "/aleph/v1/ws"), "CF-Connecting-IP": address})
        first, second = await open_raw("198.51.100.1"), await open_raw("198.51.100.1")
        with pytest.raises(aiohttp.WSServerHandshakeError) as info:
            await open_raw("198.51.100.1")  # a third from one address
        assert info.value.status == 503
        others = [await open_raw(f"198.51.100.{n}") for n in range(2, 16)]
        assert adapter._server.slots.in_use == server_mod.MAX_UNAUTHENTICATED
        with pytest.raises(aiohttp.WSServerHandshakeError) as info:
            await open_raw("198.51.100.99")  # a 17th in all
        assert info.value.status == 503
        for ws in [first, second, *others]:
            await ws.close()
        for _ in range(100):
            if adapter._server.slots.in_use == 0:
                break
            await asyncio.sleep(0.02)
        assert adapter._server.slots.in_use == 0
        # An authenticated connection does not hold a slot.
        _, keys = _pair_direct(adapter)
        phone = await _phone(adapter, keys)
        assert adapter._server.slots.in_use == 0
        await phone.close()


@pytest.mark.asyncio
async def test_handshakes_never_touch_the_database_before_authentication(adapter, monkeypatch):
    _, keys = _pair_direct(adapter)
    start_pairing(adapter)

    def no_database(*args, **kwargs):
        raise AssertionError("database access before authentication")
    for name in ("device_for_key", "devices", "device", "pending_pairing", "touch_device", "complete_pairing"):
        monkeypatch.setattr(adapter.store, name, no_database)
    async with aiohttp.ClientSession() as session:
        with pytest.raises(ConnectionRefusedError):
            await handshake(session, adapter, noise.KeyPair.generate())  # unknown key
        _, code = await run_pairing(session, adapter, os.urandom(32), noise.KeyPair.generate(), {})
        assert code == 4403  # wrong secret
    monkeypatch.undo()
    phone = await _phone(adapter, keys)  # known: the database is used only after the handshake
    await phone.close()
