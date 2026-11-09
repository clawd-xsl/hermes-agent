"""Aleph push: push.register, presence, what gets pushed and the APNs sender
(plugins/platforms/aleph/push.py); APNs itself is mocked."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import aiohttp
import pytest
import pytest_asyncio

from gateway.config import PlatformConfig
from gateway.platforms.base import ExecApprovalPrompt
from plugins.platforms.aleph import noise
from tests.gateway.aleph_phone import (
    connect, pair_device, run_pairing, start_pairing)

PUSH_TOKEN = "ab" * 32


class FakeSender:
    """Stands in for ApnsSender: records pushes, answers with a scripted status."""

    def __init__(self):
        self.sent = []
        self.answers = {}  # device token -> PushResult
        self.gate = None  # an asyncio.Event a push waits on, to prove pushes never block

    async def send(self, device_token, environment, payload):
        from plugins.platforms.aleph.push import PushResult
        if self.gate is not None:
            await self.gate.wait()
        self.sent.append((device_token, environment, payload))
        return self.answers.get(device_token, PushResult(200))

    async def close(self):
        pass


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
    instance._push = FakeSender()
    yield instance
    await instance.disconnect()


def _pair_direct(adapter, name="Phone"):
    return pair_device(adapter, name=name)


async def _phone(adapter, keys):
    phone = await connect(adapter, keys)
    answer, _ = await phone.hello()
    assert "result" in answer, answer
    return phone


async def _settle():
    """Let fire-and-forget push tasks run."""
    for _ in range(20):
        await asyncio.sleep(0.01)
# ── push.register and presence ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_push_register_validates_replaces_and_unregisters(adapter):
    device, keys = _pair_direct(adapter)
    phone = await _phone(adapter, keys)
    try:
        bad = [{"token": "not hex", "environment": "production"}, {"token": PUSH_TOKEN},
               {"token": PUSH_TOKEN, "environment": "staging"}, {"token": 42, "environment": "production"}]
        for params in bad:
            assert (await phone.call("push.register", params))["error"]["code"] == -32602
        await phone.call("push.register", {"token": PUSH_TOKEN.upper(), "environment": "development"})
        stored = adapter.store.device(device.id)
        assert (stored.push_token, stored.push_environment) == (PUSH_TOKEN, "development")
        await phone.call("push.register", {"token": "cd" * 32, "environment": "production"})
        stored = adapter.store.device(device.id)
        assert (stored.push_token, stored.push_environment) == ("cd" * 32, "production")
        assert (await phone.call("push.register", {"token": None}))["result"] == {}
        assert adapter.store.device(device.id).push_token is None
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_a_push_token_moves_to_the_device_that_registers_it(adapter):
    old, old_keys = _pair_direct(adapter, "old pairing")
    new, new_keys = _pair_direct(adapter, "new pairing")
    first = await _phone(adapter, old_keys)
    second = await _phone(adapter, new_keys)
    try:
        await first.call("push.register", {"token": PUSH_TOKEN, "environment": "production"})
        await second.call("push.register", {"token": PUSH_TOKEN, "environment": "production"})
        assert adapter.store.device(old.id).push_token is None
        assert adapter.store.device(new.id).push_token == PUSH_TOKEN
    finally:
        await first.close()
        await second.close()


@pytest.mark.asyncio
async def test_presence_tracks_any_active_connection_per_device(adapter):
    device, keys = _pair_direct(adapter)
    app = await _phone(adapter, keys)
    extension = await _phone(adapter, keys)
    try:
        assert adapter._server.active_device_ids() == set()  # connections start inactive
        assert (await app.call("presence", {"active": "yes"}))["error"]["code"] == -32602
        assert (await app.call("presence", {"active": True}))["result"] == {}
        assert adapter._server.active_device_ids() == {device.id}
        await extension.call("presence", {"active": False})
        assert adapter._server.active_device_ids() == {device.id}  # the app's connection still is
        await app.call("presence", {"active": False})
        assert adapter._server.active_device_ids() == set()
        await app.call("presence", {"active": True})
        await app.close()
        await asyncio.sleep(0.1)
        assert adapter._server.active_device_ids() == set()  # a dropped connection is not looking
    finally:
        await extension.close()


@pytest.mark.asyncio
async def test_presence_needs_hello(adapter):
    _, keys = _pair_direct(adapter)
    phone = await connect(adapter, keys)
    try:
        assert (await phone.call("presence", {"active": True}))["error"]["code"] == 4001
    finally:
        await phone.close()


# ── what gets pushed ─────────────────────────────────────────────────────────────────────────


def test_push_body_for_each_logged_event():
    from plugins.platforms.aleph.push import push_body_for
    agent = {"sender": "agent", "text": "hi", "attachments": []}
    assert push_body_for("message.new", agent) == "New message"
    assert push_body_for("message.new", {**agent, "kind": "text"}) == "New message"
    assert push_body_for("message.new", {**agent, "text": "", "attachments": [{"id": "f"}]}) == "New message"
    assert push_body_for("message.new", {**agent, "kind": "tool"}) is None
    assert push_body_for("message.new", {**agent, "kind": "system"}) is None
    assert push_body_for("message.new", {**agent, "sender": "user"}) is None
    assert push_body_for("approval.request", {}) == "Approval needed"
    assert push_body_for("clarify.request", {}) == "Question"
    for quiet in ("message.edit", "message.delete", "receipt", "approval.resolved", "clarify.resolved",
                  "draft", "typing", "sync.done"):
        assert push_body_for(quiet, {}) is None


def test_payload_and_headers_match_the_protocol():
    from plugins.platforms.aleph.push import build_headers, build_payload
    assert build_payload(body="New message", seq=58) == {
        "aps": {"alert": {"body": "New message"}, "sound": "default",
                "mutable-content": 1, "thread-id": "main"},
        "aleph": {"seq": 58},
    }
    headers = build_headers(topic="sh.xsl.aleph", now=1_000_000.0)
    assert headers == {"apns-push-type": "alert", "apns-priority": "10", "apns-topic": "sh.xsl.aleph",
                       "apns-expiration": str(1_000_000 + 86400)}


@pytest.mark.asyncio
async def test_only_notifying_events_push_and_only_to_inactive_devices(adapter):
    _away, away_keys = _pair_direct(adapter, "away")
    _looking, looking_keys = _pair_direct(adapter, "looking")
    _pair_direct(adapter, "never registered")
    phone_away = await _phone(adapter, away_keys)
    phone_looking = await _phone(adapter, looking_keys)
    try:
        await phone_away.call("push.register", {"token": PUSH_TOKEN, "environment": "development"})
        await phone_looking.call("push.register", {"token": "cd" * 32, "environment": "production"})
        await phone_looking.call("presence", {"active": True})
        sender = adapter._push
        await phone_away.call("message.send", {"id": "u1", "text": "hello"})  # user echo: quiet
        await adapter.send("main", "tool row", metadata={"_tool_progress": True})  # quiet
        await adapter.send("main", "Session reset", metadata={"_interim_send": True})  # quiet
        reply = await adapter.send("main", "the reply")
        await adapter.edit_message("main", reply.message_id, "the reply, edited")  # quiet
        await _settle()
        assert len(sender.sent) == 1
        token, environment, payload = sender.sent[0]
        assert (token, environment) == (PUSH_TOKEN, "development")
        assert payload["aps"]["alert"] == {"body": "New message"}
        assert payload["aleph"]["seq"] == adapter.store.message(reply.message_id).seq
        await adapter._send_exec_approval_prompt(ExecApprovalPrompt(
            chat_id="main", session_key="s", text="Run it?",
            actions=[("Once", "once", "primary"), ("Deny", "deny", "danger")],
            command="rm -rf build", description="dangerous", smart_denied=False, request_id=None))
        await adapter.send_clarify("main", "Which one?", ["a", "b"], "c1", "s")
        await _settle()
        assert [p["aps"]["alert"]["body"] for _, _, p in sender.sent[1:]] == ["Approval needed", "Question"]
        # The looking device leaves: now both get pushed.
        await phone_looking.call("presence", {"active": False})
        await adapter.send("main", "another")
        await _settle()
        assert sorted(t for t, _, _ in sender.sent[3:]) == sorted([PUSH_TOKEN, "cd" * 32])
    finally:
        await phone_away.close()
        await phone_looking.close()


@pytest.mark.asyncio
async def test_dead_push_tokens_are_dropped(adapter):
    from plugins.platforms.aleph.push import PushResult
    gone, gone_keys = _pair_direct(adapter, "uninstalled")
    bad, bad_keys = _pair_direct(adapter, "bad token")
    busy, busy_keys = _pair_direct(adapter, "throttled")
    phones = [await _phone(adapter, keys) for keys in (gone_keys, bad_keys, busy_keys)]
    try:
        tokens = ["01" * 32, "02" * 32, "03" * 32]
        for phone, token in zip(phones, tokens):
            await phone.call("push.register", {"token": token, "environment": "production"})
        adapter._push.answers = {tokens[0]: PushResult(410, "Unregistered"),
                                 tokens[1]: PushResult(400, "BadDeviceToken"),
                                 tokens[2]: PushResult(429, "TooManyRequests")}
        await adapter.send("main", "hello")
        await _settle()
        assert adapter.store.device(gone.id).push_token is None
        assert adapter.store.device(bad.id).push_token is None
        assert adapter.store.device(busy.id).push_token == tokens[2]
    finally:
        for phone in phones:
            await phone.close()


@pytest.mark.asyncio
async def test_a_stuck_push_never_holds_up_logging_or_delivery(adapter):
    _, keys = _pair_direct(adapter)
    phone = await _phone(adapter, keys)
    try:
        await phone.call("push.register", {"token": PUSH_TOKEN, "environment": "production"})
        adapter._push.gate = asyncio.Event()  # APNs never answers
        result = await asyncio.wait_for(adapter.send("main", "still delivered"), 1)
        assert result.success
        frame = await phone.event(timeout=2)
        assert frame["params"]["type"] == "message.new" and frame["params"]["payload"]["text"] == "still delivered"
        assert adapter._push_tasks  # the push is still waiting, off to the side
        adapter._push.gate.set()
        await _settle()
        assert len(adapter._push.sent) == 1 and not adapter._push_tasks
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_a_new_pairing_is_pushed_like_an_agent_message(adapter):
    _ipad, ipad_keys = _pair_direct(adapter, "iPad")
    ipad = await _phone(adapter, ipad_keys)
    try:
        await ipad.call("push.register", {"token": PUSH_TOKEN, "environment": "development"})
        secret = start_pairing(adapter)
        async with aiohttp.ClientSession() as session:
            _, code = await run_pairing(session, adapter, secret, noise.KeyPair.generate(), {"name": "Phone"})
        assert code is None
        await _settle()
        assert [p["aps"]["alert"]["body"] for _, _, p in adapter._push.sent] == ["New message"]
    finally:
        await ipad.close()


@pytest.mark.asyncio
async def test_revoking_drops_the_push_token(adapter):
    device, keys = _pair_direct(adapter)
    phone = await _phone(adapter, keys)
    try:
        assert (await phone.call("push.register", {"token": PUSH_TOKEN, "environment": "production"}))["result"] == {}
        from plugins.platforms.aleph.store import AlephStore
        cli_store = AlephStore(adapter.state_dir)
        assert cli_store.revoke_device(device.id)
        cli_store.close()
        assert all(d.push_token != PUSH_TOKEN for d in adapter.store.devices())
    finally:
        await phone.close()


# ── the APNs sender ──────────────────────────────────────────────────────────────────────────


@pytest.fixture
def p8(tmp_path):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    key = ec.generate_private_key(ec.SECP256R1())
    path = tmp_path / "AuthKey_TEST123456.p8"
    path.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                       serialization.NoEncryption()))
    return path, key.public_key()


class FakeHttp:
    def __init__(self, answers):
        self.answers = list(answers)
        self.requests = []

    async def post(self, url, content, headers):
        self.requests.append((url, json.loads(content), dict(headers)))
        status, body = self.answers.pop(0) if self.answers else (200, None)
        return SimpleNamespace(status_code=status, json=lambda: body)

    async def aclose(self):
        pass


def _sender(p8, http, clock):
    from plugins.platforms.aleph.push import ApnsConfig, ApnsSender
    config = ApnsConfig(str(p8[0]), "TEST123456", "PNDKVJR7LY", "sh.xsl.aleph")
    return ApnsSender(config, key_pem=p8[0].read_text(), client_factory=lambda: http, clock=clock)


@pytest.mark.asyncio
async def test_sender_signs_es256_and_picks_the_host_by_environment(p8):
    import jwt
    http = FakeHttp([])
    now = [1_800_000_000.0]
    sender = _sender(p8, http, lambda: now[0])
    payload = {"aps": {}, "aleph": {"seq": 1}}
    assert (await sender.send("aa" * 32, "development", payload)).ok
    assert (await sender.send("bb" * 32, "production", payload)).ok
    (url_dev, body, headers), (url_prod, _, headers_prod) = http.requests
    assert url_dev == f"https://api.sandbox.push.apple.com/3/device/{'aa' * 32}"
    assert url_prod == f"https://api.push.apple.com/3/device/{'bb' * 32}"
    assert body == payload
    assert headers["apns-topic"] == "sh.xsl.aleph" and headers["apns-push-type"] == "alert"
    assert headers["apns-priority"] == "10" and headers["apns-expiration"] == str(int(now[0]) + 86400)
    token = headers["authorization"].removeprefix("bearer ")
    assert jwt.get_unverified_header(token) == {"alg": "ES256", "kid": "TEST123456", "typ": "JWT"}
    claims = jwt.decode(token, p8[1], algorithms=["ES256"], options={"verify_iat": False})
    assert claims == {"iss": "PNDKVJR7LY", "iat": int(now[0])}
    # Reused for 50 minutes, then renewed.
    assert headers_prod["authorization"] == headers["authorization"]
    now[0] += 49 * 60
    await sender.send("aa" * 32, "production", payload)
    assert http.requests[-1][2]["authorization"] == headers["authorization"]
    now[0] += 2 * 60
    await sender.send("aa" * 32, "production", payload)
    assert http.requests[-1][2]["authorization"] != headers["authorization"]


@pytest.mark.asyncio
async def test_sender_renews_an_expired_provider_token_once(p8):
    http = FakeHttp([(403, {"reason": "ExpiredProviderToken"}), (200, None)])
    now = [1_800_000_000.0]
    sender = _sender(p8, http, lambda: now[0])
    sender._token()
    now[0] += 1
    result = await sender.send("aa" * 32, "production", {})
    assert result.ok and len(http.requests) == 2
    assert http.requests[0][2]["authorization"] != http.requests[1][2]["authorization"]
    http.answers = [(400, {"reason": "BadDeviceToken"})]
    result = await sender.send("aa" * 32, "production", {})
    assert (result.status, result.reason, result.token_is_dead) == (400, "BadDeviceToken", True)


def test_push_is_off_without_complete_settings(p8, monkeypatch):
    from plugins.platforms.aleph.push import ApnsConfig, ApnsSender
    assert ApnsSender.create(None) is None
    assert ApnsConfig.from_settings(key_file=str(p8[0]), key_id="K", team_id="", topic="t") is None
    config = ApnsConfig.from_settings(key_file=str(p8[0]), key_id="K", team_id="T", topic="t")
    assert config is not None
    missing = ApnsConfig.from_settings(key_file=str(p8[0].with_name("nope.p8")), key_id="K", team_id="T", topic="t")
    try:
        import h2
    except ImportError:
        assert ApnsSender.create(config) is None  # no HTTP/2 here: off, with a warning
        return
    assert ApnsSender.create(config) is not None
    assert ApnsSender.create(missing) is None


def test_adapter_reads_apns_settings_from_env(monkeypatch, tmp_path):
    from plugins.platforms.aleph.adapter import AlephAdapter
    for name, value in (("ALEPH_APNS_KEY_FILE", "~/keys/AuthKey.p8"), ("ALEPH_APNS_KEY_ID", "ABC123"),
                        ("ALEPH_APNS_TEAM_ID", "PNDKVJR7LY"), ("ALEPH_APNS_TOPIC", "sh.xsl.aleph")):
        monkeypatch.setenv(name, value)
    config = AlephAdapter(PlatformConfig(enabled=True, extra={"state_dir": str(tmp_path)})).apns_config
    assert config.key_id == "ABC123" and config.team_id == "PNDKVJR7LY" and config.topic == "sh.xsl.aleph"
    assert not config.key_file.startswith("~")
    monkeypatch.delenv("ALEPH_APNS_TOPIC")
    assert AlephAdapter(PlatformConfig(enabled=True, extra={"state_dir": str(tmp_path)})).apns_config is None


@pytest.mark.asyncio
async def test_setup_problems_are_reported_once_per_topic_and_keep_the_token(adapter, caplog):
    import logging
    from plugins.platforms.aleph.push import ApnsConfig, PushResult
    adapter.apns_config = ApnsConfig("k.p8", "KEY", "TEAM", "sh.xsl.aleph")
    device, keys = _pair_direct(adapter)
    phone = await _phone(adapter, keys)
    try:
        await phone.call("push.register", {"token": PUSH_TOKEN, "environment": "development"})
        adapter._push.answers = {PUSH_TOKEN: PushResult(400, "TopicDisallowed")}
        with caplog.at_level(logging.DEBUG, logger="plugins.platforms.aleph.adapter"):
            for text in ("one", "two", "three"):
                await adapter.send("main", text)
                await _settle()
            errors = [r for r in caplog.records if r.levelno == logging.ERROR]
            assert len(errors) == 1
            assert "TopicDisallowed" in errors[0].getMessage() and "not registered for push" in errors[0].getMessage()
            assert not [r for r in caplog.records if r.levelno == logging.WARNING and "push" in r.getMessage()]
            assert adapter.store.device(device.id).push_token == PUSH_TOKEN  # not the device's fault
            # Fixed, then broken again: reported again.
            adapter._push.answers = {}
            await adapter.send("main", "works")
            await _settle()
            adapter._push.answers = {PUSH_TOKEN: PushResult(400, "TopicDisallowed")}
            await adapter.send("main", "broken again")
            await _settle()
        assert len([r for r in caplog.records if r.levelno == logging.ERROR]) == 2
    finally:
        await phone.close()


def test_config_problem_reasons():
    from plugins.platforms.aleph.push import PushResult
    assert PushResult(400, "TopicDisallowed").config_problem
    assert PushResult(400, "DeviceTokenNotForTopic").config_problem
    assert PushResult(403, "InvalidProviderToken").config_problem
    assert PushResult(400, "BadDeviceToken").config_problem is None
    assert PushResult(429, "TooManyRequests").config_problem is None
    assert PushResult(200).config_problem is None
