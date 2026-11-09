"""Fixed-key vectors for Aleph's two Noise handshakes, for the phone's tests to replay.

Deterministic: every key is derived from a label, so rerunning gives the same bytes. The plugin's
own implementation is checked against the official cacophony vectors first
(``test_noise_matches_the_cacophony_vectors``); these vectors then pin Aleph's exact use of it
(protocol names, prologues, payloads, the PSK) for the Swift side.

    python -m tests.gateway.aleph_noise_vectors <out.json>
"""

from __future__ import annotations

import hashlib
import json
import sys

from plugins.platforms.aleph import noise


def _key(label: str) -> noise.KeyPair:
    return noise.KeyPair.from_private(hashlib.sha256(f"aleph noise vectors: {label}".encode()).digest())


def _vector(protocol: str, prologue: bytes, msg1_payload: bytes, msg2_payload: bytes, transport: list,
            psk: bytes | None = None) -> dict:
    tag = "pair" if psk else "connect"
    init_static, init_ephemeral = _key(f"{tag} phone static"), _key(f"{tag} phone ephemeral")
    resp_static, resp_ephemeral = _key(f"{tag} server static"), _key(f"{tag} server ephemeral")
    i = noise.Handshake(protocol, True, prologue, init_static, rs=resp_static.public, psk=psk, e=init_ephemeral)
    r = noise.Handshake(protocol, False, prologue, resp_static, psk=psk, e=resp_ephemeral)
    msg1 = i.write_message(msg1_payload)
    assert r.read_message(msg1) == msg1_payload
    msg2 = r.write_message(msg2_payload)
    assert i.read_message(msg2) == msg2_payload
    assert i.handshake_hash == r.handshake_hash
    ti, tr = i.transport(), r.transport()
    messages = []
    for sender, plaintext in transport:
        a, b = (ti, tr) if sender == "initiator" else (tr, ti)
        ciphertext = a.encrypt(plaintext)
        assert b.decrypt(ciphertext) == plaintext
        messages.append({"from": sender, "plaintext": plaintext.hex(), "ciphertext": ciphertext.hex()})
    vector = {
        "protocol_name": protocol,
        "prologue": prologue.decode(),
        "prologue_hex": prologue.hex(),
        "init_static": init_static.private.hex(),
        "init_static_public": init_static.public.hex(),
        "init_ephemeral": init_ephemeral.private.hex(),
        "resp_static": resp_static.private.hex(),
        "resp_static_public": resp_static.public.hex(),
        "resp_ephemeral": resp_ephemeral.private.hex(),
    }
    if psk is not None:
        vector["psk"] = psk.hex()
    vector.update({
        "msg1_payload": msg1_payload.hex(),
        "msg1": msg1.hex(),
        "msg2_payload": msg2_payload.hex(),
        "msg2": msg2.hex(),
        "handshake_hash": i.handshake_hash.hex(),
        "transport": messages,
    })
    return vector


def _upgrade_mac_vector() -> dict:
    """The upgrade gate's ``X-Aleph-Mac`` (protocol "Exposure", layer 1) for fixed inputs."""
    from plugins.platforms.aleph.noise import b64url
    from plugins.platforms.aleph.server import mac_key, upgrade_mac
    server_public, ts, nonce, path = b"\x07" * 32, 1794200000, b"\xab" * 16, "/aleph/v1/ws"
    key = mac_key(server_public)
    return {
        "server_static_public": server_public.hex(),
        "mac_key": key.hex(),
        "ts": ts,
        "nonce": nonce.hex(),
        "nonce_b64url": b64url(nonce),
        "path": path,
        "header": upgrade_mac(key, ts, b64url(nonce), path),
    }


def build_vectors() -> dict:
    hello = b'{"jsonrpc":"2.0","id":1,"method":"hello","params":{"protocol":1,"after":0}}'
    answer = b'{"jsonrpc":"2.0","id":1,"result":{"protocol":1,"chat":{"id":"main"},"latest":0,"reset":false}}'
    ping = b'{"jsonrpc":"2.0","id":2,"method":"ping","params":{}}'
    done = b'{"jsonrpc":"2.0","method":"event","params":{"type":"sync.done","payload":{"seq":0}}}'
    return {
        "description": "Aleph protocol 1 Noise vectors (fixed keys, hex). The phone is the initiator, the "
                       "plugin the responder. Transport messages use the Split() keys with empty "
                       "associated data and the Noise nonce, in the order listed.",
        "vectors": [
            _vector(noise.IK, noise.PROLOGUE, b"", b"",
                    [("initiator", hello), ("responder", answer), ("initiator", ping), ("responder", done)]),
            _vector(noise.IKPSK1, noise.PAIR_PROLOGUE, b'{"name":"Aleph","model":"iPhone"}',
                    b'{"device_id":"d_0123456789abcdef","agent":{"name":"Aleph"}}',
                    [("initiator", b"first"), ("responder", b"second"), ("initiator", b"third"),
                     ("responder", b"fourth")],
                    psk=hashlib.sha256(b"aleph noise vectors: pair secret").digest()),
        ],
        "upgrade_mac": [_upgrade_mac_vector()],
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m tests.gateway.aleph_noise_vectors <out.json>")
    with open(sys.argv[1], "w") as handle:
        json.dump(build_vectors(), handle, indent=2)
        handle.write("\n")
