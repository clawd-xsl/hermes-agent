"""Behavior contracts for the persistent signal-ts sidecar transport (plugins/platforms/signal/signal_ts.py
and sidecar/index.mjs)."""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
from pathlib import Path

import pytest

from plugins.platforms.signal.signal_ts import (
    SIDECAR_SCRIPT,
    SignalTsCallError,
    SignalTsSidecar,
    SignalTsStartupError,
    resolve_node_executable,
    runtime_configured,
)


def _write_fake_node(tmp_path: Path) -> Path:
    executable = tmp_path / "fake-node"
    executable.write_text(
        f"""#!{sys.executable}
import json
import os
import sys

def emit(value):
    print(json.dumps(value), flush=True)

emit({{"event": "ready", "account": "+15551234567", "aci": "aci-1", "timestamp": 1}})
for line in sys.stdin:
    request = json.loads(line)
    request_id = request["id"]
    method = request["method"]
    if method == "pid":
        emit({{"id": request_id, "ok": True, "result": os.getpid()}})
    elif method == "emit":
        emit({{
            "event": "envelope",
            "envelope": {{"sourceUuid": "sender", "dataMessage": {{"message": "hello"}}}},
        }})
        emit({{"id": request_id, "ok": True, "result": True}})
    elif method == "fail":
        emit({{
            "id": request_id,
            "ok": False,
            "error": {{"name": "SignalError", "message": "send failed", "code": "429"}},
        }})
    elif method == "shutdown":
        emit({{"id": request_id, "ok": True, "result": True}})
        break
    else:
        emit({{"id": request_id, "ok": True, "result": method}})
"""
    )
    executable.chmod(0o700)
    return executable


def _write_fake_sdk(tmp_path: Path) -> Path:
    sdk_path = tmp_path / "fake-sdk"
    sdk_path.mkdir()
    (sdk_path / "package.json").write_text(
        json.dumps({"type": "module", "main": "index.mjs"})
    )
    (sdk_path / "index.mjs").write_text(
        """
export class SignalTsDecryptionError extends Error {}

const account = {
  account: {
    auth: { username: "user", password: "password" },
    device: { aci: "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa", e164: "+15551234567", deviceId: 2 },
  },
};

export class FileSignalRepository {
  static async open() { return new FileSignalRepository(); }
  async getAccount() { return account; }
  async getRecipientByAci(aci) { return { aci, e164: "+15550000000" }; }
  async getRecipientByE164(e164) { return { aci: "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb", e164 }; }
  async getGroup() { return undefined; }
  async setRecipient() {}
  snapshot() { return { recipients: {} }; }
}

export class SignalTsClient {
  constructor() { this.handlers = new Map(); this.activity = Date.now(); }
  on(name, handler) { this.handlers.set(name, handler); return () => this.handlers.delete(name); }
  async connect() {
    setTimeout(() => this.handlers.get("incoming")?.({ envelope: new Uint8Array(), ack() {} }), 10);
  }
  async disconnect() {}
  getLastTransportActivityAt() { return this.activity; }
  async decryptIncoming() { return {}; }
  async sendMessage(params) {
    if (params.body === "one" && params.bodyRanges?.[0]?.style !== 1) {
      throw new Error("Hermes style was not mapped to a Signal body range");
    }
    return { timestamp: params.timestamp };
  }
  async sendTypingMessage() { return { timestamp: Date.now() }; }
  async sendReactionMessage(params) { return { timestamp: params.timestamp }; }
  async uploadAttachment() { throw new Error("not used"); }
}

export function createLibsignalStores() { return {}; }
export function createSignalLocalAddress() { return {}; }
export function normalizeDecryptedIncomingMessage() {
  return [{
    kind: "sync",
    sender: { serviceId: "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa", deviceId: 1 },
    timestamp: 1700000000000,
    syncMessage: {
      sent: {
        destinationServiceId: "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        timestamp: 1700000000000,
        message: { body: "note from phone", attachments: [], bodyRanges: [] },
      },
    },
  }];
}
export function parseSignalRecipientTarget(value) {
  if (typeof value === "string" && value.startsWith("+")) return { kind: "e164", e164: value };
  return { kind: "aci", aci: value };
}
export function preKeyAuthFromBase64() { return { kind: "access-key", accessKey: new Uint8Array() }; }
export function deriveAccessKeyBase64FromProfileKeyBase64() { return "access"; }
export function base64ToBytes() { return new Uint8Array(); }
export function signalGroupIdFromMasterKey() { return undefined; }
export const signalAttachmentFetch = fetch;
"""
    )
    return sdk_path


def _sidecar(tmp_path: Path, executable: Path, on_envelope, **kwargs) -> SignalTsSidecar:
    sdk_path = tmp_path / "sdk"
    sdk_path.mkdir(exist_ok=True)
    state_path = tmp_path / "state.json"
    state_path.write_text("{}")
    return SignalTsSidecar(
        node_executable=str(executable), sdk_path=sdk_path, state_path=state_path,
        cache_dir=tmp_path / "cache", expected_account="+15551234567", on_envelope=on_envelope,
        startup_timeout=3, call_timeout=3, **kwargs)


def _real_node():
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not available")
    return node


@pytest.mark.asyncio
async def test_one_process_handles_multiple_calls(tmp_path):
    sidecar = _sidecar(tmp_path, _write_fake_node(tmp_path), lambda _envelope: None)
    await sidecar.start()
    try:
        first_pid = await sidecar.call("pid")
        second_pid = await sidecar.call("pid")
        assert first_pid == second_pid == sidecar.process.pid
        assert sidecar.account == "+15551234567"
    finally:
        await sidecar.close()
    assert sidecar.process is None


@pytest.mark.asyncio
async def test_envelopes_go_to_the_callback_without_blocking_replies(tmp_path):
    envelopes = []
    sidecar = _sidecar(tmp_path, _write_fake_node(tmp_path), envelopes.append)
    await sidecar.start()
    try:
        assert await sidecar.call("emit") is True
        assert envelopes == [{"sourceUuid": "sender", "dataMessage": {"message": "hello"}}]
        assert await asyncio.wait_for(sidecar.call("pid"), timeout=1) == sidecar.process.pid
    finally:
        await sidecar.close()


@pytest.mark.asyncio
async def test_structured_sidecar_error_is_preserved(tmp_path):
    sidecar = _sidecar(tmp_path, _write_fake_node(tmp_path), lambda _envelope: None)
    await sidecar.start()
    try:
        with pytest.raises(SignalTsCallError) as captured:
            await sidecar.call("fail")
        assert captured.value.details["code"] == "429"
        assert "send failed" in str(captured.value)
    finally:
        await sidecar.close()


@pytest.mark.asyncio
async def test_exit_resolves_wait_closed_and_fails_pending_calls(tmp_path):
    sidecar = _sidecar(tmp_path, _write_fake_node(tmp_path), lambda _envelope: None)
    await sidecar.start()
    sidecar.process.kill()
    await asyncio.wait_for(sidecar.wait_closed(), timeout=3)
    with pytest.raises(Exception):
        await sidecar.call("pid")
    await sidecar.close()


def test_node_command_is_a_scalar_executable(tmp_path):
    executable = _write_fake_node(tmp_path)
    assert resolve_node_executable(str(executable)) == str(executable)
    assert resolve_node_executable(f"{executable} --flag") is None
    assert resolve_node_executable(str(tmp_path / "missing")) is None
    assert resolve_node_executable("") is None


def test_runtime_configured_requires_state_and_sdk(tmp_path):
    state = tmp_path / "state.json"
    assert runtime_configured({"state_path": str(state), "sdk_path": str(tmp_path)}) is False
    state.write_text("{}")
    assert runtime_configured({"state_path": str(state), "sdk_path": str(tmp_path)}) is True
    assert runtime_configured({"state_path": str(state)}) is False
    assert runtime_configured({"state_path": str(state), "sdk_path": str(tmp_path),
                               "node_command": str(tmp_path / "no-node")}) is False


@pytest.mark.asyncio
async def test_real_sidecar_reports_missing_state_as_config_error(tmp_path):
    """The shipped mjs host must refuse a missing state file with a structured, non-retryable error."""
    sidecar = SignalTsSidecar(
        node_executable=_real_node(), sdk_path=_write_fake_sdk(tmp_path), state_path=tmp_path / "missing.json",
        cache_dir=tmp_path / "cache", expected_account=None, on_envelope=lambda _e: None, startup_timeout=10)
    with pytest.raises(SignalTsStartupError) as captured:
        await sidecar.start()
    assert captured.value.code == "state_missing"
    assert captured.value.retryable is False
    assert "missing.json" in str(captured.value)


@pytest.mark.asyncio
async def test_real_sidecar_reports_account_mismatch(tmp_path):
    state = tmp_path / "state.json"
    state.write_text("{}")
    sidecar = SignalTsSidecar(
        node_executable=_real_node(), sdk_path=_write_fake_sdk(tmp_path), state_path=state,
        cache_dir=tmp_path / "cache", expected_account="+15559999999", on_envelope=lambda _e: None,
        startup_timeout=10)
    with pytest.raises(SignalTsStartupError) as captured:
        await sidecar.start()
    assert captured.value.code == "account_mismatch" and captured.value.retryable is False


@pytest.mark.asyncio
async def test_real_node_sidecar_protocol_and_send_mapping(tmp_path):
    """Exercise the shipped mjs host, not a Python protocol stand-in."""
    assert SIDECAR_SCRIPT.is_file()
    envelopes = []
    state_path = tmp_path / "state.json"
    state_path.write_text("{}")
    sidecar = SignalTsSidecar(
        node_executable=_real_node(), sdk_path=_write_fake_sdk(tmp_path), state_path=state_path,
        cache_dir=tmp_path / "cache", expected_account="+15551234567", on_envelope=envelopes.append,
        startup_timeout=10, call_timeout=5)
    await sidecar.start()
    try:
        process_id = sidecar.process.pid
        first = await sidecar.call("send", {"message": "one", "recipient": ["+15550000000"], "textStyle": "0:3:BOLD"})
        second = await sidecar.call("send", {"message": "two", "recipient": ["+15550000000"]})
        assert first["results"] == [{"type": "SUCCESS"}]
        assert second["results"] == [{"type": "SUCCESS"}]
        assert first["timestamp"] <= second["timestamp"]
        assert sidecar.process.pid == process_id
        for _ in range(100):
            if envelopes:
                break
            await asyncio.sleep(0.02)
        sent = envelopes[0]["syncMessage"]["sentMessage"]
        assert sent["destinationNumber"] == "+15551234567"
        assert sent["message"] == "note from phone"
        with pytest.raises(SignalTsCallError, match="Unsupported"):
            await sidecar.call("bogus")
    finally:
        await sidecar.close()
