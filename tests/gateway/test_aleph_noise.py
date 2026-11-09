"""Aleph's Noise handshakes (plugins/platforms/aleph/noise.py): the official cacophony vectors,
Aleph's own fixed-key vectors (which the phone's tests replay), and what must fail."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from plugins.platforms.aleph import noise

FIXTURES = Path(__file__).parent / "fixtures"
# ── Noise ────────────────────────────────────────────────────────────────────────────────────


def _replay(vector, *, protocol=None, prologue=None):
    """Run both roles of a vector with its fixed keys; returns the handshake hash."""
    hexb = bytes.fromhex
    protocol = protocol or vector["protocol_name"]
    psk = hexb(vector["psk"]) if vector.get("psk") else (
        hexb(vector["init_psks"][0]) if vector.get("init_psks") else None)
    init_static = noise.KeyPair.from_private(hexb(vector["init_static"]))
    resp_static = noise.KeyPair.from_private(hexb(vector["resp_static"]))
    i = noise.Handshake(protocol, True, prologue if prologue is not None else hexb(vector["init_prologue"]),
                        init_static, rs=resp_static.public, psk=psk,
                        e=noise.KeyPair.from_private(hexb(vector["init_ephemeral"])))
    r = noise.Handshake(protocol, False, prologue if prologue is not None else hexb(vector["resp_prologue"]),
                        resp_static, psk=psk, e=noise.KeyPair.from_private(hexb(vector["resp_ephemeral"])))
    return i, r


def test_noise_matches_the_cacophony_vectors():
    """The official vectors for both patterns (cacophony, public domain): handshake and transport."""
    vectors = json.loads((FIXTURES / "aleph_noise_cacophony.json").read_text())["vectors"]
    assert {v["protocol_name"] for v in vectors} == {noise.IK, noise.IKPSK1}
    for vector in vectors:
        i, r = _replay(vector)
        messages = vector["messages"]
        hexb = bytes.fromhex
        assert i.write_message(hexb(messages[0]["payload"])).hex() == messages[0]["ciphertext"]
        r.read_message(hexb(messages[0]["ciphertext"]))
        assert r.write_message(hexb(messages[1]["payload"])).hex() == messages[1]["ciphertext"]
        assert i.read_message(hexb(messages[1]["ciphertext"])).hex() == messages[1]["payload"]
        assert i.handshake_hash.hex() == r.handshake_hash.hex() == vector["handshake_hash"]
        ti, tr = i.transport(), r.transport()
        for k, message in enumerate(messages[2:]):
            sender, receiver = (ti, tr) if k % 2 == 0 else (tr, ti)
            assert sender.encrypt(hexb(message["payload"])).hex() == message["ciphertext"]
            assert receiver.decrypt(hexb(message["ciphertext"])).hex() == message["payload"]


def test_aleph_vectors_replay():
    """The fixed-key vectors the phone's tests replay (scripts in tests/gateway/aleph_noise_vectors.py)."""
    from tests.gateway.aleph_noise_vectors import build_vectors
    for vector in build_vectors()["vectors"]:
        hexb = bytes.fromhex
        i, r = _replay(vector, prologue=vector["prologue"].encode())
        assert i.write_message(hexb(vector["msg1_payload"])).hex() == vector["msg1"]
        assert r.read_message(hexb(vector["msg1"])).hex() == vector["msg1_payload"]
        assert r.write_message(hexb(vector["msg2_payload"])).hex() == vector["msg2"]
        assert i.read_message(hexb(vector["msg2"])).hex() == vector["msg2_payload"]
        assert i.handshake_hash.hex() == vector["handshake_hash"]
        ti, tr = i.transport(), r.transport()
        for message in vector["transport"]:
            sender, receiver = (ti, tr) if message["from"] == "initiator" else (tr, ti)
            assert sender.encrypt(hexb(message["plaintext"])).hex() == message["ciphertext"]
            assert receiver.decrypt(hexb(message["ciphertext"])).hex() == message["plaintext"]


def test_noise_rejects_tampering_and_wrong_keys():
    server, phone = noise.KeyPair.generate(), noise.KeyPair.generate()
    i = noise.initiator(noise.IK, noise.PROLOGUE, phone, server.public)
    message = bytearray(i.write_message(b""))
    message[40] ^= 1
    with pytest.raises(noise.NoiseError):
        noise.responder(noise.IK, noise.PROLOGUE, server).read_message(bytes(message))
    # A phone that pinned another server key cannot be answered by this one.
    i = noise.initiator(noise.IK, noise.PROLOGUE, phone, noise.KeyPair.generate().public)
    with pytest.raises(noise.NoiseError):
        noise.responder(noise.IK, noise.PROLOGUE, server).read_message(i.write_message(b""))
    # Another prologue (the pairing one) does not complete a connection handshake.
    i = noise.initiator(noise.IK, noise.PAIR_PROLOGUE, phone, server.public)
    with pytest.raises(noise.NoiseError):
        noise.responder(noise.IK, noise.PROLOGUE, server).read_message(i.write_message(b""))
    # Pairing: a wrong secret fails at the server's first decryption.
    i = noise.initiator(noise.IKPSK1, noise.PAIR_PROLOGUE, phone, server.public, psk=b"\x01" * 32)
    with pytest.raises(noise.NoiseError):
        noise.responder(noise.IKPSK1, noise.PAIR_PROLOGUE, server, psk=b"\x02" * 32).read_message(
            i.write_message(b"{}"))


def test_transport_messages_are_not_capped_at_64k_and_must_arrive_in_order():
    server, phone = noise.KeyPair.generate(), noise.KeyPair.generate()
    i = noise.initiator(noise.IK, noise.PROLOGUE, phone, server.public)
    r = noise.responder(noise.IK, noise.PROLOGUE, server)
    r.read_message(i.write_message(b""))
    i.read_message(r.write_message(b""))
    ti, tr = i.transport(), r.transport()
    big = os.urandom(200_000)
    assert tr.decrypt(ti.encrypt(big)) == big
    first, second = ti.encrypt(b"one"), ti.encrypt(b"two")
    with pytest.raises(noise.NoiseError):
        tr.decrypt(second)  # out of order
    assert tr.decrypt(first) == b"one"
