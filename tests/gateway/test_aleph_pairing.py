"""Aleph pairing, the server key, connection handshakes, revocation and the `hermes aleph` CLI
(plugins/platforms/aleph), driven through a real listener."""

from __future__ import annotations

import argparse
import asyncio
import os
import stat
import time
from types import SimpleNamespace

import aiohttp
import pytest
import pytest_asyncio

from gateway.config import PlatformConfig
from plugins.platforms.aleph import noise
from tests.gateway.aleph_phone import (
    connect, handshake, mac_headers, pair_device, run_pairing, start_pairing, url as _url)

PUSH_TOKEN = "ab" * 32


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
# ── the server key ───────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_server_key_is_created_once_with_mode_600(adapter):
    path = adapter.state_dir / "server.key"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600 and len(path.read_bytes()) == 32
    from plugins.platforms.aleph.store import load_or_create_server_key
    assert noise.KeyPair.from_private(load_or_create_server_key(adapter.state_dir)) == adapter.static
    await adapter.disconnect()
    assert await adapter.connect()
    assert noise.KeyPair.from_private(path.read_bytes()) == adapter.static


# ── pairing ──────────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_pairing_records_the_key_spends_the_secret_and_tells_the_thread(adapter):
    _other, other_keys = _pair_direct(adapter, "iPad")
    ipad = await _phone(adapter, other_keys)
    secret = start_pairing(adapter)
    keys = noise.KeyPair.generate()
    try:
        async with aiohttp.ClientSession() as session:
            answer, code = await run_pairing(session, adapter, secret, keys,
                                             {"name": "Xin's iPhone", "model": "iPhone18,1"})
            assert code is None
            assert answer["device_id"].startswith("d_") and answer["agent"] == {"name": "Aleph"}
            device = adapter.store.device(answer["device_id"])
            assert device.public_key == keys.public and device.name == "Xin's iPhone"
            assert device.model == "iPhone18,1"
            # Spent: nothing pending, so the endpoint is not there at all.
            with pytest.raises(aiohttp.WSServerHandshakeError) as info:
                await run_pairing(session, adapter, secret, noise.KeyPair.generate(), {})
            assert info.value.status == 404
        # The new device's key now connects.
        phone = await _phone(adapter, keys)
        await phone.close()
        note = (await ipad.logged())["params"]["payload"]
        assert note["kind"] == "system" and note["text"] == "New device paired: Xin's iPhone"
    finally:
        await ipad.close()


@pytest.mark.asyncio
async def test_a_wrong_secret_or_server_key_closes_with_4403(adapter):
    secret = start_pairing(adapter)
    async with aiohttp.ClientSession() as session:
        _, code = await run_pairing(session, adapter, os.urandom(32), noise.KeyPair.generate(), {})
        assert code == 4403
        # A phone that pinned another server key: its MAC fails the gate before any handshake.
        with pytest.raises(aiohttp.WSServerHandshakeError) as info:
            await run_pairing(session, adapter, secret, noise.KeyPair.generate(), {},
                              server_key=noise.KeyPair.generate().public,
                              headers=mac_headers(adapter, "/aleph/v1/pair",
                                                  key=b"\x00" * 32))
        assert info.value.status == 404
        # The secret is still pending after failed attempts.
        answer, code = await run_pairing(session, adapter, secret, noise.KeyPair.generate(), {"model": "iPhone"})
        assert code is None and adapter.store.device(answer["device_id"]).name == "iPhone"
    assert adapter.store.pending_pairing() is None


@pytest.mark.asyncio
async def test_pairing_exists_only_while_one_is_pending_and_works_through_the_tunnel(adapter):
    async def status(headers=None):
        async with aiohttp.ClientSession() as session:
            try:
                await run_pairing(session, adapter, os.urandom(32), noise.KeyPair.generate(), {}, headers=headers)
            except aiohttp.WSServerHandshakeError as exc:
                return exc.status
            return 101
    assert await status() == 404  # nothing pending
    start_pairing(adapter, ttl=-1)
    assert await status() == 404  # expired
    secret = start_pairing(adapter)
    assert await status() == 101  # the handshake runs (and fails on the secret)
    # Pairing goes over the public address: a request through Cloudflare pairs like any other.
    async with aiohttp.ClientSession() as session:
        answer, code = await run_pairing(session, adapter, secret, noise.KeyPair.generate(), {"name": "Away"},
                                         headers={"CF-Connecting-IP": "203.0.113.9", "CF-Ray": "8c1f-NRT"})
    assert code is None and adapter.store.device(answer["device_id"]).name == "Away"


@pytest.mark.asyncio
async def test_a_pairing_started_by_the_cli_becomes_reachable(adapter):
    """`hermes aleph pair` runs in another process: the gateway picks its pairing up from the
    database on its poll, and keeps it in memory for the unauthenticated endpoint."""
    from plugins.platforms.aleph.store import AlephStore
    cli_store = AlephStore(adapter.state_dir)
    secret, _ = cli_store.new_pairing()
    cli_store.close()
    for _ in range(100):
        if adapter.pending_pairing():
            break
        await asyncio.sleep(0.02)
    assert adapter.pending_pairing()[0] == secret
    async with aiohttp.ClientSession() as session:
        answer, code = await run_pairing(session, adapter, secret, noise.KeyPair.generate(), {})
    assert code is None and answer["device_id"]


@pytest.mark.asyncio
async def test_pairing_is_rate_limited_per_address(adapter):
    start_pairing(adapter)
    async with aiohttp.ClientSession() as session:
        for _ in range(5):
            _, code = await run_pairing(session, adapter, os.urandom(32), noise.KeyPair.generate(), {})
            assert code == 4403
        with pytest.raises(aiohttp.WSServerHandshakeError) as info:
            await run_pairing(session, adapter, os.urandom(32), noise.KeyPair.generate(), {})
        assert info.value.status == 429 and 1 <= int(info.value.headers["Retry-After"]) <= 60
        # Behind the tunnel (loopback), Cloudflare's header tells phones apart.
        _, code = await run_pairing(session, adapter, os.urandom(32), noise.KeyPair.generate(), {},
                                    headers={"CF-Connecting-IP": "198.51.100.7"})
        assert code == 4403


def test_the_forwarded_address_is_trusted_only_from_loopback():
    from plugins.platforms.aleph.server import client_address
    req = lambda remote, headers: SimpleNamespace(remote=remote, headers=headers)
    assert client_address(req("127.0.0.1", {"CF-Connecting-IP": "203.0.113.9"})) == "203.0.113.9"
    assert client_address(req("::1", {"CF-Connecting-IP": "203.0.113.8"})) == "203.0.113.8"
    assert client_address(req("192.168.1.30", {"CF-Connecting-IP": "203.0.113.9"})) == "192.168.1.30"
    assert client_address(req("127.0.0.1", {"X-Forwarded-For": "203.0.113.9"})) == "127.0.0.1"


def test_rate_limiter_window_slides():
    from plugins.platforms.aleph.server import PairRateLimiter
    now = [0.0]
    limiter = PairRateLimiter(clock=lambda: now[0])
    assert all(limiter.allow("a") is None for _ in range(5))
    assert limiter.allow("a") == pytest.approx(60)
    now[0] = 59.0
    assert limiter.allow("a") == pytest.approx(1)
    now[0] = 60.0
    assert limiter.allow("a") is None
    assert limiter.allow("b") is None


def test_a_new_pairing_replaces_the_pending_one_and_a_known_key_keeps_its_device(tmp_path):
    from plugins.platforms.aleph.store import AlephStore
    store = AlephStore(tmp_path)
    try:
        first, _ = store.new_pairing()
        second, _ = store.new_pairing()
        key = noise.KeyPair.generate().public
        assert store.complete_pairing(first, key, name="x", model="") is None
        device = store.complete_pairing(second, key, name="old name", model="")
        again, _ = store.new_pairing()
        assert store.complete_pairing(again, key, name="new name", model="m").id == device.id
        assert [d.name for d in store.devices()] == ["new name"]
    finally:
        store.close()


# ── connections ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_slow_handshake_closes_with_4401(adapter, monkeypatch):
    from plugins.platforms.aleph import server as server_mod
    monkeypatch.setattr(server_mod, "HANDSHAKE_SECONDS", 0.2)
    async with aiohttp.ClientSession() as session:
        ws = await session.ws_connect(_url(adapter, "/ws"), headers=mac_headers(adapter, "/aleph/v1/ws"))
        msg = await ws.receive(timeout=3)
        assert msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED) and ws.close_code == 4401
        await ws.close()
    assert server_mod.HANDSHAKE_SECONDS == 0.2
    monkeypatch.undo()
    assert server_mod.HANDSHAKE_SECONDS == 5.0


@pytest.mark.asyncio
async def test_a_frame_that_does_not_decrypt_ends_the_connection(adapter):
    _, keys = _pair_direct(adapter)
    phone = await _phone(adapter, keys)
    try:
        await phone.ws.send_bytes(os.urandom(64))
        msg = await asyncio.wait_for(phone.ws.receive(), 3)
        assert msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED) and phone.ws.close_code == 1002
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_frames_larger_than_64k_go_both_ways(adapter):
    _, keys = _pair_direct(adapter)
    phone = await _phone(adapter, keys)
    try:
        text = "é" * 60_000  # 120 KB of UTF-8, under the 64,000-character text limit
        assert "result" in await phone.call("message.send", {"chat": "main", "id": "big", "text": text})
        echo = (await phone.logged())["params"]["payload"]
        assert echo["text"] == text
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_revoking_closes_connections_and_refuses_the_key(adapter):
    device, keys = _pair_direct(adapter)
    _other, other_keys = _pair_direct(adapter, "iPad")
    phone = await _phone(adapter, keys)
    ipad = await _phone(adapter, other_keys)
    try:
        # What `hermes aleph revoke` does, from its own process: straight to the database.
        from plugins.platforms.aleph.store import AlephStore
        cli_store = AlephStore(adapter.state_dir)
        assert cli_store.revoke_device(device.id)
        cli_store.close()
        msg = await asyncio.wait_for(phone.ws.receive(), 3)
        assert msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED)
        assert phone.ws.close_code == 4401
        async with aiohttp.ClientSession() as session:
            with pytest.raises(ConnectionRefusedError) as info:
                await handshake(session, adapter, keys)
            assert info.value.args[0] == 4401
        assert (await ipad.call("ping"))["result"] == {}  # the other device is untouched
    finally:
        await phone.close()
        await ipad.close()


# ── the CLI ──────────────────────────────────────────────────────────────────────────────────


def _cli(argv):
    from plugins.platforms.aleph import cli
    parser = argparse.ArgumentParser()
    cli.register_cli(parser)
    args = parser.parse_args(argv)
    return args.func(args)


def _link_params(out):
    from urllib.parse import parse_qs, urlsplit
    link = out.split("Link:")[1].split()[0]
    assert link.startswith("aleph://pair?")
    return {k: v[0] for k, v in parse_qs(urlsplit(link).query).items()}


def test_urls_for_the_pairing_link(monkeypatch):
    from plugins.platforms.aleph import cli
    for name in ("ALEPH_PUBLIC_URL", "ALEPH_HOST", "ALEPH_PORT", "ALEPH_ACCESS_ID", "ALEPH_ACCESS_SECRET"):
        monkeypatch.delenv(name, raising=False)
    assert cli.server_url({}) == "http://127.0.0.1:8650"
    assert cli.server_url({"host": "192.168.1.20", "port": 9000}) == "http://192.168.1.20:9000"
    monkeypatch.setattr(cli, "_lan_address", lambda: "192.168.1.44")
    assert cli.server_url({"host": "0.0.0.0"}) == "http://192.168.1.44:8650"
    monkeypatch.setenv("ALEPH_PUBLIC_URL", "https://aleph.xsl.sh/")
    assert cli.server_url({"host": "0.0.0.0"}) == "https://aleph.xsl.sh"
    assert cli.server_url({}, "http://127.0.0.1:8650") == "http://127.0.0.1:8650"
    assert cli.access_token({}) is None
    monkeypatch.setenv("ALEPH_ACCESS_ID", "abc.access")
    assert cli.access_token({}) is None  # both or nothing
    monkeypatch.setenv("ALEPH_ACCESS_SECRET", "s3cr3t/+=")
    assert cli.access_token({}) == ("abc.access", "s3cr3t/+=")
    assert cli.pair_link(server="https://aleph.xsl.sh", key=b"\x00" * 32, secret=b"\xff" * 32) == (
        "aleph://pair?server=https%3A%2F%2Faleph.xsl.sh"
        "&key=AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA&secret=__________________________________________8")
    assert cli.pair_link(server="https://aleph.xsl.sh", key=b"\x00" * 32, secret=b"\xff" * 32,
                         access=("abc.access", "s3cr3t/+=")).endswith(
        "&access_id=abc.access&access_secret=s3cr3t%2F%2B%3D")


def test_cli_pair_devices_and_revoke(monkeypatch, tmp_path, capsys):
    from plugins.platforms.aleph import cli
    from plugins.platforms.aleph.noise import from_b64url
    from plugins.platforms.aleph.store import AlephStore, fingerprint
    monkeypatch.setenv("ALEPH_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("ALEPH_PUBLIC_URL", "https://aleph.xsl.sh")
    for name in ("ALEPH_ACCESS_ID", "ALEPH_ACCESS_SECRET"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(cli, "_extra", dict)
    monkeypatch.setattr(cli, "_listening", lambda host, port: True)
    assert _cli(["devices"]) == 0
    assert "No paired devices" in capsys.readouterr().out
    assert _cli(["pair", "--no-wait"]) == 0
    out = capsys.readouterr().out
    params = _link_params(out)
    assert set(params) == {"server", "key", "secret"} and params["server"] == "https://aleph.xsl.sh"
    server_key = noise.KeyPair.from_private((tmp_path / "server.key").read_bytes())
    assert from_b64url(params["key"]) == server_key.public
    secret = from_b64url(params["secret"])
    assert len(secret) == 32
    gateway_side = AlephStore(tmp_path)
    phone_key = noise.KeyPair.generate().public
    device = gateway_side.complete_pairing(secret, phone_key, name="Xin's iPhone", model="iPhone18,1")
    gateway_side.set_push_token(device.id, PUSH_TOKEN, "production")
    gateway_side.close()
    assert _cli(["devices"]) == 0
    listing = capsys.readouterr().out
    assert device.id in listing and "Xin's iPhone" in listing and "production" in listing
    assert fingerprint(phone_key) in listing
    assert _cli(["revoke", "d_missing"]) == 1
    assert _cli(["revoke", device.id]) == 0
    assert "Revoked Xin's iPhone" in capsys.readouterr().out
    assert _cli(["devices"]) == 0 and "No paired devices" in capsys.readouterr().out


def test_cli_pair_with_an_access_token_and_its_warnings(monkeypatch, tmp_path, capsys):
    from plugins.platforms.aleph import cli
    monkeypatch.setenv("ALEPH_STATE_DIR", str(tmp_path))
    for name in ("ALEPH_PUBLIC_URL", "ALEPH_HOST"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ALEPH_ACCESS_ID", "abc.access")
    monkeypatch.setenv("ALEPH_ACCESS_SECRET", "s3cr3t")
    monkeypatch.setattr(cli, "_extra", dict)
    monkeypatch.setattr(cli, "_listening", lambda host, port: True)
    assert _cli(["pair", "--no-wait"]) == 0
    out = capsys.readouterr().out
    assert "replaces the pending pairing" not in out
    params = _link_params(out)
    assert params["access_id"] == "abc.access" and params["access_secret"] == "s3cr3t"
    assert _cli(["pair", "--no-wait", "--server", "http://192.168.1.20:8650"]) == 0
    out = capsys.readouterr().out
    assert "Warning: this replaces the pending pairing" in out
    assert "the listener binds 127.0.0.1" in out


def test_cli_pair_waits_until_the_phone_pairs(monkeypatch, tmp_path, capsys):
    import threading
    from plugins.platforms.aleph import cli
    from plugins.platforms.aleph.store import AlephStore
    monkeypatch.setenv("ALEPH_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(cli, "_extra", lambda: {"public_url": "https://aleph.xsl.sh"})
    monkeypatch.setattr(cli, "_listening", lambda host, port: True)
    real_sleep = time.sleep
    monkeypatch.setattr(cli.time, "sleep", lambda s: real_sleep(0.02))
    original = AlephStore.new_pairing

    def capture(self, **kw):
        secret, expires = original(self, **kw)
        # The phone pairs a moment later and the gateway records it.
        threading.Timer(0.1, lambda: _complete(secret)).start()
        return secret, expires

    def _complete(secret):
        store = AlephStore(tmp_path)
        try:
            store.complete_pairing(secret, noise.KeyPair.generate().public, name="Aleph", model="iPhone")
        finally:
            store.close()
    monkeypatch.setattr(AlephStore, "new_pairing", capture)
    assert _cli(["pair"]) == 0
    assert "Paired Aleph (iPhone) as d_" in capsys.readouterr().out
