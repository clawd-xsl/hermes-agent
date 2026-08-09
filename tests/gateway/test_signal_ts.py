"""Behavior contracts for the persistent signal-ts subprocess transport."""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
from pathlib import Path

import pytest

from gateway.platforms.signal_ts import (
    SignalTsCallError,
    SignalTsSidecar,
    resolve_node_executable,
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


def _sidecar(
    tmp_path: Path,
    executable: Path,
    on_envelope,
) -> SignalTsSidecar:
    sdk_path = tmp_path / "sdk"
    sdk_path.mkdir(exist_ok=True)
    state_path = tmp_path / "state.json"
    state_path.write_text("{}")
    return SignalTsSidecar(
        node_executable=str(executable),
        sdk_path=sdk_path,
        state_path=state_path,
        cache_dir=tmp_path / "cache",
        expected_account="+15551234567",
        on_envelope=on_envelope,
        startup_timeout=3,
        call_timeout=3,
    )


@pytest.mark.asyncio
async def test_one_process_handles_multiple_calls(tmp_path):
    async def on_envelope(_envelope):
        return None

    sidecar = _sidecar(tmp_path, _write_fake_node(tmp_path), on_envelope)
    await sidecar.start()
    try:
        first_pid = await sidecar.call("pid")
        second_pid = await sidecar.call("pid")
        assert first_pid == second_pid
        assert first_pid == sidecar.process.pid
        assert sidecar.account == "+15551234567"
    finally:
        await sidecar.close()


@pytest.mark.asyncio
async def test_inbound_handler_never_blocks_command_responses(tmp_path):
    handler_started = asyncio.Event()
    release_handler = asyncio.Event()

    async def on_envelope(_envelope):
        handler_started.set()
        await release_handler.wait()

    sidecar = _sidecar(tmp_path, _write_fake_node(tmp_path), on_envelope)
    await sidecar.start()
    try:
        await sidecar.call("emit")
        await asyncio.wait_for(handler_started.wait(), timeout=1)
        # The model turn represented by on_envelope is still blocked, but the
        # stdout reader must continue correlating typing/reaction/send calls.
        assert await asyncio.wait_for(sidecar.call("pid"), timeout=1) == sidecar.process.pid
    finally:
        release_handler.set()
        await sidecar.close()


@pytest.mark.asyncio
async def test_structured_sidecar_error_is_preserved(tmp_path):
    async def on_envelope(_envelope):
        return None

    sidecar = _sidecar(tmp_path, _write_fake_node(tmp_path), on_envelope)
    await sidecar.start()
    try:
        with pytest.raises(SignalTsCallError) as captured:
            await sidecar.call("fail")
        assert captured.value.details["code"] == "429"
        assert "send failed" in str(captured.value)
    finally:
        await sidecar.close()


def test_node_command_is_a_scalar_executable(tmp_path):
    executable = _write_fake_node(tmp_path)
    assert resolve_node_executable(str(executable)) == str(executable)
    assert resolve_node_executable(f"{executable} --flag") is None
    assert resolve_node_executable(str(tmp_path / "missing")) is None


@pytest.mark.asyncio
async def test_real_node_sidecar_protocol_and_send_mapping(tmp_path):
    """Exercise the shipped mjs host, not a Python protocol stand-in."""
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not available")

    envelopes = []
    inbound = asyncio.Event()

    async def on_envelope(envelope):
        envelopes.append(envelope)
        inbound.set()

    state_path = tmp_path / "state.json"
    state_path.write_text("{}")
    sidecar = SignalTsSidecar(
        node_executable=node,
        sdk_path=_write_fake_sdk(tmp_path),
        state_path=state_path,
        cache_dir=tmp_path / "cache",
        expected_account="+15551234567",
        on_envelope=on_envelope,
        startup_timeout=3,
        call_timeout=3,
    )
    await sidecar.start()
    try:
        process_id = sidecar.process.pid
        first = await sidecar.call(
            "send",
            {
                "message": "one",
                "recipient": ["+15550000000"],
                "textStyle": "0:3:BOLD",
            },
        )
        second = await sidecar.call(
            "send",
            {"message": "two", "recipient": ["+15550000000"]},
        )
        assert first["results"] == [{"type": "SUCCESS"}]
        assert second["results"] == [{"type": "SUCCESS"}]
        assert first["timestamp"] <= second["timestamp"]
        assert sidecar.process.pid == process_id
        await asyncio.wait_for(inbound.wait(), timeout=1)
        sent = envelopes[0]["syncMessage"]["sentMessage"]
        assert sent["destinationNumber"] == "+15551234567"
        assert sent["message"] == "note from phone"
    finally:
        await sidecar.close()
