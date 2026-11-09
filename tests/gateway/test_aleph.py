"""The Aleph channel plugin (plugins/platforms/aleph): wire protocol, event log, receipts, cards
and files, driven through a real listener and WebSocket client."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import aiohttp
import pytest
import pytest_asyncio

from gateway.config import PlatformConfig
from gateway.platforms.base import ExecApprovalPrompt
from tests.gateway.aleph_phone import connect, mac_headers, pair_device, url as _url

def _register():
    from plugins.platforms.aleph.register import register
    captured = {}
    tools = []
    register(SimpleNamespace(register_platform=lambda **kw: captured.update(kw),
                             register_cli_command=lambda **kw: captured.setdefault("cli", kw),
                             register_tool=lambda **kw: tools.append(kw)))
    _register.tools = tools
    return captured


@pytest_asyncio.fixture
async def adapter(tmp_path):
    from plugins.platforms.aleph.adapter import AlephAdapter
    config = PlatformConfig(enabled=True, extra={"host": "127.0.0.1", "port": 0,
                                                 "state_dir": str(tmp_path / "aleph")})
    instance = AlephAdapter(config)
    received = []

    async def handler(event):
        received.append(event)
    instance.set_message_handler(handler)
    instance.received = received
    assert await instance.connect()
    instance.device, instance.keys = pair_device(instance)
    yield instance
    await instance.disconnect()


async def _phone(adapter, keys=None):
    return await connect(adapter, keys or adapter.keys)


# ── registration ──────────────────────────────────────────────────────────────────────────────


def test_register_declares_the_channel(monkeypatch):
    from plugins.platforms.aleph import adapter as mod
    captured = _register()
    assert captured["name"] == "aleph" and captured["adapter_factory"] is mod.AlephAdapter
    assert captured["cron_deliver_env_var"] == "ALEPH_HOME_CHANNEL"
    assert captured["required_env"] == ["ALEPH_ENABLED"]
    assert captured["standalone_send"].media and callable(captured["standalone_sender_fn"])
    assert not captured.get("trusted_inbound")
    assert captured["cli"]["name"] == "aleph"
    assert [t["name"] for t in _register.tools] == ["aleph_get_profile", "aleph_set_profile"]
    assert {t["toolset"] for t in _register.tools} == {"aleph"} and all(t["is_async"] for t in _register.tools)
    # No shared secret: an enabled config is a connected one; the env probe needs ALEPH_ENABLED.
    assert captured["config_connected_fn"](PlatformConfig(enabled=True))
    assert not captured["config_connected_fn"](PlatformConfig(enabled=False))
    monkeypatch.delenv("ALEPH_ENABLED", raising=False)
    assert not captured["is_connected"](PlatformConfig(enabled=True))
    monkeypatch.setenv("ALEPH_ENABLED", "true")
    assert captured["is_connected"](PlatformConfig(enabled=True))


def test_env_enablement_seeds_home_owner_and_listener(monkeypatch):
    enable = _register()["env_enablement_fn"]
    monkeypatch.delenv("ALEPH_ENABLED", raising=False)
    monkeypatch.setenv("ALEPH_TOKEN", "a leftover shared secret")
    assert enable() is None
    monkeypatch.setenv("ALEPH_ENABLED", "true")
    monkeypatch.setenv("ALEPH_HOST", "192.168.1.20")
    monkeypatch.setenv("ALEPH_PORT", "9000")
    monkeypatch.setenv("ALEPH_PUBLIC_URL", "https://aleph.example.com")
    monkeypatch.delenv("ALEPH_HOME_CHANNEL", raising=False)
    seed = enable()
    assert "token" not in seed
    assert seed["host"] == "192.168.1.20" and seed["port"] == "9000"
    assert seed["public_url"] == "https://aleph.example.com"
    assert seed["home_channel"] == {"chat_id": "main", "name": "Aleph"}
    assert seed["allow_from"] == ["owner"]
    monkeypatch.setenv("ALEPH_HOME_CHANNEL", "off")
    assert "home_channel" not in enable()


def test_env_only_setup_enables_the_platform_with_main_as_home(monkeypatch, tmp_path):
    """The real config path: ALEPH_ENABLED alone enables the plugin platform with home ``aleph:main``."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("ALEPH_ENABLED", "true")
    monkeypatch.delenv("ALEPH_HOME_CHANNEL", raising=False)
    from gateway.config import GatewayConfig, Platform
    from gateway.config_env import _enable_plugin_platform
    from gateway.platform_registry import PlatformEntry
    captured = _register()
    captured.pop("cli")
    entry = PlatformEntry(source="plugin", **captured)
    config = GatewayConfig()
    _enable_plugin_platform(config, entry)
    platform_config = config.platforms[Platform("aleph")]
    assert platform_config.enabled
    assert platform_config.home_channel.chat_id == "main"
    assert platform_config.extra["allow_from"] == ["owner"]
    assert Platform("aleph") in config.get_connected_platforms()


def test_tool_progress_marker_is_opt_in():
    from plugins.platforms.aleph.adapter import AlephAdapter
    from gateway.platforms.base import BasePlatformAdapter
    assert AlephAdapter.MARKS_TOOL_PROGRESS is True
    assert getattr(BasePlatformAdapter, "MARKS_TOOL_PROGRESS", False) is False


# ── connection and auth ───────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_unknown_keys_and_plain_frames_are_refused(adapter):
    from plugins.platforms.aleph import noise
    from tests.gateway.aleph_phone import handshake
    async with aiohttp.ClientSession() as session:
        with pytest.raises(ConnectionRefusedError) as info:
            await handshake(session, adapter, noise.KeyPair.generate())
        assert info.value.args[0] == 4401
        # A text frame instead of the handshake.
        ws = await session.ws_connect(_url(adapter, "/ws"), headers=mac_headers(adapter, "/aleph/v1/ws"))
        await ws.send_str('{"jsonrpc":"2.0","id":1,"method":"hello","params":{}}')
        msg = await ws.receive(timeout=5)
        assert msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED) and ws.close_code == 4401
        await ws.close()
        # The old endpoints are gone.
        async with session.post(_url(adapter, "/files"), data=b"x") as resp:
            assert resp.status in (404, 405)


@pytest.mark.asyncio
async def test_hello_must_come_first_and_names_supported_protocols(adapter):
    phone = await _phone(adapter)
    try:
        early = await phone.call("message.send", {"id": "m1", "text": "hi"})
        assert early["error"]["code"] == 4001
        wrong = await phone.call("hello", {"protocol": 9, "after": 0})
        assert wrong["error"]["code"] == -32602 and wrong["error"]["data"] == {"supported": [1]}
        answer, replay = await phone.hello()
        assert answer["result"]["chat"] == {"id": "main"} and answer["result"]["latest"] == 0
        assert answer["result"]["reset"] is False and answer["result"]["agent"]["name"] == "Aleph"
        assert replay == [{"type": "sync.done", "payload": {"seq": 0}}]
        unknown = await phone.call("nope")
        assert unknown["error"]["code"] == -32601
        assert (await phone.call("ping"))["result"] == {}
    finally:
        await phone.close()


# ── sending and the log ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_message_send_is_logged_echoed_dispatched_and_idempotent(adapter):
    phone = await _phone(adapter)
    try:
        await phone.hello()
        first = await phone.call("message.send", {"id": "6F1C", "text": "Check the build"})
        assert first["result"] == {"seq": 1}
        echo = await phone.logged()
        assert echo["params"]["type"] == "message.new" and echo["params"]["seq"] == 1
        payload = echo["params"]["payload"]
        assert payload["id"] == "6F1C" and payload["sender"] == "user" and payload["text"] == "Check the build"
        assert payload["attachments"] == [] and "+" in payload["created_at"] or "-" in payload["created_at"][19:]
        again = await phone.call("message.send", {"id": "6F1C", "text": "Check the build"})
        assert again["result"] == {"seq": 1}
        await asyncio.sleep(0.6)  # past the text-batch window
        handed = [event for event in adapter.received if event.message_id == "6F1C"]
        assert len(handed) == 1 and handed[0].text == "Check the build"
        assert handed[0].source.chat_id == "main" and handed[0].source.chat_type == "dm"
        assert handed[0].source.user_id == "owner"
        bad = await phone.call("message.send", {"id": "", "text": "x"})
        assert bad["error"]["code"] == -32602
        empty = await phone.call("message.send", {"id": "m-empty", "text": "  "})
        assert empty["error"]["code"] == -32602
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_reconnect_replays_only_what_the_phone_lacks(adapter):
    for text in ("one", "two", "three"):
        await adapter.send("main", text)
    phone = await _phone(adapter)
    try:
        answer, replay = await phone.hello(after=1)
        assert answer["result"]["latest"] == 3 and answer["result"]["reset"] is False
        assert [(e["seq"], e["payload"]["text"]) for e in replay[:-1]] == [(2, "two"), (3, "three")]
        assert replay[-1] == {"type": "sync.done", "payload": {"seq": 3}}
        await adapter.send("main", "four")
        live = await phone.event()
        assert live["params"]["seq"] == 4 and live["params"]["payload"]["sender"] == "agent"
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_phone_ahead_of_the_log_gets_a_reset(adapter):
    await adapter.send("main", "only")
    phone = await _phone(adapter)
    try:
        answer, replay = await phone.hello(after=50)
        assert answer["result"]["reset"] is True
        assert [e.get("seq") for e in replay] == [1, None]
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_pruned_history_resets_from_the_oldest_event(adapter):
    adapter.store.history_limit = 100
    for i in range(130):
        await adapter.send("main", f"m{i}")
    oldest = adapter.store.oldest()
    assert oldest > 1
    phone = await _phone(adapter)
    try:
        answer, replay = await phone.hello(after=2)
        assert answer["result"]["reset"] is True
        assert replay[0]["seq"] == oldest and replay[-2]["seq"] == 130
        assert [e["seq"] for e in replay[:-1]] == list(range(oldest, 131))
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_event_logged_during_hello_arrives_after_sync_done_exactly_once(adapter):
    for i in range(50):
        await adapter.send("main", f"m{i}")
    phone = await _phone(adapter)
    try:
        hello = asyncio.create_task(phone.hello(after=0))
        racer = asyncio.create_task(adapter.send("main", "racer"))
        answer, replay = await hello
        await racer
        seqs = [e["seq"] for e in replay[:-1]]
        latest = answer["result"]["latest"]
        assert seqs == list(range(1, latest + 1))
        if latest == 50:  # the racer landed after the snapshot: it must follow sync.done
            live = await phone.event()
            assert live["params"]["seq"] == 51 and live["params"]["payload"]["text"] == "racer"
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_two_devices_both_receive_every_event(adapter):
    a, b = await _phone(adapter), await _phone(adapter)
    try:
        await a.hello()
        await b.hello()
        await a.call("message.send", {"id": "x1", "text": "from a"})
        for phone in (a, b):
            frame = await phone.logged()
            assert frame["params"]["payload"]["id"] == "x1"
    finally:
        await a.close()
        await b.close()


# ── agent output ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_draft_then_final_send_carries_the_draft_id(adapter):
    phone = await _phone(adapter)
    try:
        await phone.hello()
        assert adapter.supports_draft_streaming(chat_type="dm")
        await adapter.send_draft("main", 7, "Hel")
        await adapter.send_draft("main", 7, "Hello wor")
        first, second = await phone.event(), await phone.event()
        assert first["params"]["type"] == "draft" and "seq" not in first["params"]
        draft_id = first["params"]["payload"]["draft_id"]
        assert second["params"]["payload"] == {"draft_id": draft_id, "text": "Hello wor"}
        result = await adapter.send("main", "Hello world")
        assert result.success and result.message_id
        final = await phone.event()
        assert final["params"]["type"] == "message.new"
        assert final["params"]["payload"]["draft_id"] == draft_id
        assert final["params"]["payload"]["id"] == result.message_id
        # A segment after a tool call starts with blank lines; the draft drops them.
        await adapter.send_draft("main", 8, "\n\nNext part")
        assert (await phone.event())["params"]["payload"]["text"] == "Next part"
        await adapter.send("main", "Next part")
        await phone.event()
        # A later message has no draft to claim.
        await adapter.send("main", "and more")
        assert "draft_id" not in (await phone.event())["params"]["payload"]
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_abandoned_draft_is_cleared(adapter):
    phone = await _phone(adapter)
    try:
        await phone.hello()
        await adapter.send_draft("main", 3, "partial")
        draft = await phone.event()
        await adapter.abandon_open_draft("main", "partial")
        cleared = await phone.event()
        assert cleared["params"] == {"type": "draft.clear",
                                     "payload": {"draft_id": draft["params"]["payload"]["draft_id"]}}
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_tool_progress_and_gateway_notices_get_their_kinds(adapter):
    phone = await _phone(adapter)
    try:
        await phone.hello()
        await adapter.send_draft("main", 4, "thinking")
        await phone.event()
        await adapter.send("main", "🔍 web_search: \"x\"", metadata={"_tool_progress": True})
        tool = (await phone.event())["params"]["payload"]
        assert tool["kind"] == "tool" and "draft_id" not in tool
        await adapter.send("main", "Gateway restarting", metadata={"_interim_send": True})
        assert (await phone.event())["params"]["payload"]["kind"] == "system"
        await adapter.send("main", "done")
        final = (await phone.event())["params"]["payload"]
        assert "kind" not in final and final["draft_id"]
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_edit_and_delete_agent_messages(adapter):
    phone = await _phone(adapter)
    try:
        await phone.hello()
        sent = await adapter.send("main", "v1")
        await phone.event()
        assert (await adapter.edit_message("main", sent.message_id, "v2")).success
        edit = await phone.event()
        assert edit["params"]["type"] == "message.edit" and edit["params"]["payload"] == {
            "id": sent.message_id, "text": "v2"}
        assert (await adapter.edit_message("main", sent.message_id, "v2")).success  # unchanged: no event
        assert await adapter.delete_message("main", sent.message_id)
        deleted = await phone.event()
        assert deleted["params"]["type"] == "message.delete"
        assert not (await adapter.edit_message("main", "unknown", "x")).success
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_typing_and_status_text_are_live(adapter):
    phone = await _phone(adapter)
    try:
        await phone.hello()
        assert adapter.supports_status_text
        await adapter.send_typing("main")
        await adapter.send_typing("main")  # the refresh repeats: one event
        # Hermes's Slack-style live status ("is searching …") is not what Aleph shows.
        await asyncio.to_thread(adapter.set_status_text, "main", "is searching the web…")
        # The gateway reports the tool from the agent's worker thread; Aleph words the line itself.
        source = SimpleNamespace(chat_id="main")
        await asyncio.to_thread(adapter.on_tool_activity, source, {
            "type": "tool.started", "tool": "web_search", "args": {"query": "flights to Tokyo"}, "live_status": "full"})
        await asyncio.sleep(0.05)
        await asyncio.to_thread(adapter.on_tool_activity, source, {"type": "tool.completed", "tool": "web_search"})
        await asyncio.sleep(0.05)
        await adapter.stop_typing("main")
        frames = [f for f in [(await phone.event())["params"] for _ in range(6)] if f["type"] == "typing"]
        assert frames == [
            {"type": "typing", "payload": {"active": True}},
            {"type": "typing", "payload": {"active": True, "status": "Searching the web for flights to Tokyo…"}},
            {"type": "typing", "payload": {"active": True}},
            {"type": "typing", "payload": {"active": False}},
        ]
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_phone_connecting_mid_turn_sees_typing_and_the_open_draft(adapter):
    await adapter.send_typing("main")
    await adapter.send_draft("main", 9, "so far")
    phone = await _phone(adapter)
    try:
        await phone.hello()
        frames = [(await phone.event())["params"] for _ in range(2)]
        assert frames[0] == {"type": "typing", "payload": {"active": True}}
        assert frames[1]["type"] == "draft" and frames[1]["payload"]["text"] == "so far"
    finally:
        await phone.close()


# ── receipts ──────────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_receipts_follow_turns_including_queued_messages(adapter):
    """The first message starts a turn (receipt a); b and c arrive while it runs and stay at one
    check; the follow-up turn that takes them both starts and receipts c."""
    release = asyncio.Event()
    turns = []

    async def busy(event):
        turns.append(event.text)
        if len(turns) == 1:
            await release.wait()
    adapter.set_message_handler(busy)
    phone = await _phone(adapter)
    try:
        await phone.hello()
        await phone.call("message.send", {"id": "a", "text": "first"})
        frames = await phone.events_until("receipt")
        assert frames[-1]["payload"]["read_up_to"] == "a"
        await phone.call("message.send", {"id": "b", "text": "second"})
        await phone.call("message.send", {"id": "c", "text": "third"})
        echoes = [(await phone.logged())["params"] for _ in range(2)]
        assert [e["payload"]["id"] for e in echoes] == ["b", "c"]
        await asyncio.sleep(0.3)
        assert adapter.store.latest() == 4  # no receipt while the first turn runs
        release.set()
        receipt = (await phone.logged())["params"]
        assert receipt["type"] == "receipt" and receipt["seq"] == 5
        assert receipt["payload"]["read_up_to"] == "c"
        await asyncio.sleep(0.3)
        assert turns[0] == "first" and "second" in turns[1] and "third" in turns[1]
    finally:
        release.set()
        await phone.close()


@pytest.mark.asyncio
async def test_internal_wakes_and_repeat_turns_send_no_receipt(adapter):
    source = SimpleNamespace(platform=adapter.platform, chat_id="main")
    await adapter._accept_user_message({"id": "p0", "text": "hello"})
    await asyncio.sleep(0.2)
    latest = adapter.store.latest()
    assert adapter.store.events_after(latest - 1)[0].type == "receipt"
    await adapter.on_processing_start(SimpleNamespace(internal=False, source=source))
    await adapter.on_processing_start(SimpleNamespace(internal=True, source=source))
    assert adapter.store.latest() == latest


@pytest.mark.asyncio
async def test_receipt_watermark_survives_a_restart(adapter, tmp_path):
    from plugins.platforms.aleph.adapter import AlephAdapter
    await adapter._accept_user_message({"id": "p1", "text": "hello"})
    event = SimpleNamespace(internal=False, source=SimpleNamespace(platform=adapter.platform, chat_id="main"))
    await adapter.on_processing_start(event)
    await adapter.disconnect()
    reborn = AlephAdapter(PlatformConfig(enabled=True, extra=dict(adapter.config.extra)))
    assert await reborn.connect()
    try:
        latest = reborn.store.latest()
        await reborn.on_processing_start(event)
        assert reborn.store.latest() == latest  # already receipted before the restart
    finally:
        await reborn.disconnect()
    await adapter.connect()


@pytest.mark.asyncio
async def test_full_turn_through_the_base_adapter(adapter):
    """handle_message -> processing hooks -> the handler's reply goes out through ``send``."""
    async def reply(event):
        return f"echo: {event.text}"
    adapter.set_message_handler(reply)
    phone = await _phone(adapter)
    try:
        await phone.hello()
        await phone.call("message.send", {"id": "t1", "text": "ping"})
        seen = []
        while not any(e["type"] == "message.new" and e["payload"]["sender"] == "agent" for e in seen):
            seen.append((await phone.logged())["params"])
        types = [e["type"] for e in seen if "seq" in e]
        assert types == ["message.new", "receipt", "message.new"]
        assert seen[-1]["payload"]["text"] == "echo: ping"
    finally:
        await phone.close()


# ── cards ─────────────────────────────────────────────────────────────────────────────────────


def _approval_prompt(request_id="req-1"):
    return ExecApprovalPrompt(chat_id="main", session_key="sk", text="Run `rm -rf build`?",
                              actions=[("Allow once", "once", "primary"), ("Always", "always", ""),
                                       ("Deny", "deny", "danger")],
                              command="rm -rf build", description="deletes files", smart_denied=False,
                              request_id=request_id)


@pytest.mark.asyncio
async def test_approval_card_round_trip(adapter, monkeypatch):
    calls = []
    from tools import approval
    monkeypatch.setattr(approval, "resolve_gateway_approval_card",
                        lambda session_key, choice, request_id: calls.append((session_key, choice, request_id)) or 1)
    monkeypatch.setattr(approval, "observe_gateway_settle", lambda *a: True)
    phone = await _phone(adapter)
    try:
        await phone.hello()
        assert adapter.supports_exec_approval_buttons()
        sent = await adapter._send_exec_approval_prompt(_approval_prompt())
        assert sent.success
        request = (await phone.event())["params"]
        assert request["type"] == "approval.request" and request["seq"] == 1
        assert request["payload"]["choices"] == [
            {"label": "Allow once", "choice": "once", "style": "primary"},
            {"label": "Always", "choice": "always", "style": "default"},
            {"label": "Deny", "choice": "deny", "style": "destructive"}]
        assert request["payload"]["command"] == "rm -rf build" and request["payload"]["reason"] == "deletes files"
        bad = await phone.call("approval.respond", {"request_id": "req-1", "choice": "session"})
        assert bad["error"]["code"] == -32602
        ok = await phone.call("approval.respond", {"request_id": "req-1", "choice": "once"})
        assert ok["result"] == {}
        assert calls == [("sk", "once", "req-1")]
        resolved = (await phone.event())["params"]
        assert resolved["type"] == "approval.resolved"
        assert resolved["payload"] == {"request_id": "req-1", "choice": "once", "by": "user"}
        gone = await phone.call("approval.respond", {"request_id": "req-1", "choice": "once"})
        assert gone["error"]["code"] == -32602
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_approval_settled_elsewhere_is_logged(adapter, monkeypatch):
    observers = []
    from tools import approval
    monkeypatch.setattr(approval, "observe_gateway_settle",
                        lambda session_key, request_id, observer: observers.append(observer) or True)
    phone = await _phone(adapter)
    try:
        await phone.hello()
        await adapter._send_exec_approval_prompt(_approval_prompt("req-2"))
        await phone.event()
        # The observer runs on the agent's thread when the wait times out.
        await asyncio.to_thread(observers[0], "timeout")
        resolved = (await phone.event())["params"]
        assert resolved["payload"] == {"request_id": "req-2", "choice": None, "by": "timeout"}
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_real_approval_queue_resolves_from_the_phone(adapter):
    """Against tools.approval itself: a queued gateway approval is answered by the card."""
    from tools import approval
    from tools.approval_gateway_wait import _ApprovalEntry
    entry = _ApprovalEntry({"request_id": "real-1", "command": "ls"})
    with approval._lock:
        approval._gateway_queues.setdefault("sk-real", []).append(entry)
    phone = await _phone(adapter)
    try:
        await phone.hello()
        prompt = _approval_prompt("real-1")
        prompt.session_key = "sk-real"
        await adapter._send_exec_approval_prompt(prompt)
        await phone.event()
        ok = await phone.call("approval.respond", {"request_id": "real-1", "choice": "deny"})
        assert ok["result"] == {}
        assert entry.event.wait(1) and entry.result == "deny"
        assert (await phone.event())["params"]["payload"]["by"] == "user"
    finally:
        with approval._lock:
            approval._gateway_queues.pop("sk-real", None)
        await phone.close()


@pytest.mark.asyncio
async def test_clarify_card_round_trip(adapter, monkeypatch):
    import tools.clarify_gateway as clarify
    answers = []
    monkeypatch.setattr(clarify, "resolve_gateway_clarify", lambda cid, text: answers.append((cid, text)) or True)
    phone = await _phone(adapter)
    try:
        await phone.hello()
        await adapter.send_clarify("main", "Which env?", ["staging", "prod"], "cl-1", "sk")
        request = (await phone.event())["params"]
        assert request["payload"] == {"clarify_id": "cl-1", "question": "Which env?",
                                      "choices": ["staging", "prod"], "allows_text": True}
        assert (await phone.call("clarify.respond", {"clarify_id": "cl-1", "choice": 5}))["error"]["code"] == -32602
        assert (await phone.call("clarify.respond", {"clarify_id": "cl-1", "choice": 1}))["result"] == {}
        assert answers == [("cl-1", "prod")]
        resolved = (await phone.event())["params"]
        assert resolved["payload"] == {"clarify_id": "cl-1", "answer": "prod", "by": "user"}
        await adapter.send_clarify("main", "Name?", None, "cl-2", "sk")
        await phone.event()
        assert (await phone.call("clarify.respond", {"clarify_id": "cl-2", "text": "Ada"}))["result"] == {}
        assert answers[-1] == ("cl-2", "Ada")
        await phone.event()
        # Ended without a respond (timeout / reset): the card is retired.
        await adapter.send_clarify("main", "Still there?", ["yes"], "cl-3", "sk")
        await phone.event()
        await adapter.retire_clarify_card("cl-3", "expired")
        retired = (await phone.event())["params"]
        assert retired["payload"] == {"clarify_id": "cl-3", "answer": None, "by": "timeout"}
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_chat_stop_dispatches_the_stop_command(adapter):
    phone = await _phone(adapter)
    try:
        await phone.hello()
        assert (await phone.call("chat.stop"))["result"] == {}
        await asyncio.sleep(0.1)
        assert [e.text for e in adapter.received] == ["/stop"]
        assert adapter.store.latest() == 0  # not logged as a message
    finally:
        await phone.close()
