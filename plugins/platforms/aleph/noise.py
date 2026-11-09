"""The Noise handshakes Aleph speaks (https://noiseprotocol.org/noise.html, revision 34).

Two patterns, both with the server as responder and its static key known to the phone in advance:

- ``Noise_IK_25519_ChaChaPoly_SHA256``, prologue ``aleph/1``: every connection.
  ``-> e, es, s, ss`` / ``<- e, ee, se``.
- ``Noise_IKpsk1_25519_ChaChaPoly_SHA256``, prologue ``aleph/1/pair``: pairing, with the pairing
  secret as the PSK. ``-> e, es, s, ss, psk`` / ``<- e, ee, se``.

Written from the spec on the ``cryptography`` package. The initiator role exists for tests and for
the cross-implementation vectors the phone's tests replay; the plugin itself only responds.

After the handshake the two ``CipherState`` s from ``Split()`` encrypt one JSON-RPC frame per
WebSocket message, with empty associated data. Unlike Noise transport messages they are not capped
at 65,535 bytes: the AEAD has no such limit, and the protocol keeps frames under 16 MB.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
from dataclasses import dataclass, field
from typing import Optional

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat, PublicFormat

DHLEN = 32
HASHLEN = 32
TAGLEN = 16
MAX_NONCE = 2 ** 64 - 1

IK = "Noise_IK_25519_ChaChaPoly_SHA256"
IKPSK1 = "Noise_IKpsk1_25519_ChaChaPoly_SHA256"
PROLOGUE = b"aleph/1"
PAIR_PROLOGUE = b"aleph/1/pair"

# Message patterns after the ``<- s`` pre-message.
_PATTERNS = {
    IK: (("e", "es", "s", "ss"), ("e", "ee", "se")),
    IKPSK1: (("e", "es", "s", "ss", "psk"), ("e", "ee", "se")),
}


def b64url(data: bytes) -> str:
    """Keys and secrets in links and payloads: base64url without padding."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def from_b64url(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


class NoiseError(Exception):
    """A handshake or transport message that does not decrypt or is malformed."""


# ── keys ─────────────────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class KeyPair:
    private: bytes
    public: bytes

    @classmethod
    def generate(cls) -> KeyPair:
        return cls.from_private(X25519PrivateKey.generate().private_bytes(
            Encoding.Raw, PrivateFormat.Raw, NoEncryption()))

    @classmethod
    def from_private(cls, private: bytes) -> KeyPair:
        if len(private) != DHLEN:
            raise ValueError("an X25519 private key is 32 bytes")
        key = X25519PrivateKey.from_private_bytes(private)
        return cls(bytes(private), key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw))


def dh(private: bytes, public: bytes) -> bytes:
    try:
        return X25519PrivateKey.from_private_bytes(private).exchange(X25519PublicKey.from_public_bytes(public))
    except ValueError as exc:  # a low-order point gives an all-zero shared secret
        raise NoiseError(f"bad public key: {exc}") from exc


def _hmac(key: bytes, data: bytes) -> bytes:
    return hmac.new(key, data, hashlib.sha256).digest()


def hkdf(chaining_key: bytes, ikm: bytes, outputs: int) -> tuple[bytes, ...]:
    temp = _hmac(chaining_key, ikm)
    out1 = _hmac(temp, b"\x01")
    out2 = _hmac(temp, out1 + b"\x02")
    if outputs == 2:
        return out1, out2
    return out1, out2, _hmac(temp, out2 + b"\x03")


def _nonce(n: int) -> bytes:
    return b"\x00" * 4 + n.to_bytes(8, "little")


# ── CipherState ──────────────────────────────────────────────────────────────────────────────────


class CipherState:
    def __init__(self, key: Optional[bytes] = None) -> None:
        self.k = key
        self.n = 0

    def has_key(self) -> bool:
        return self.k is not None

    def encrypt(self, ad: bytes, plaintext: bytes) -> bytes:
        if self.k is None:
            return plaintext
        if self.n >= MAX_NONCE:
            raise NoiseError("nonce exhausted")
        out = ChaCha20Poly1305(self.k).encrypt(_nonce(self.n), plaintext, ad)
        self.n += 1
        return out

    def decrypt(self, ad: bytes, ciphertext: bytes) -> bytes:
        if self.k is None:
            return ciphertext
        if self.n >= MAX_NONCE:
            raise NoiseError("nonce exhausted")
        try:
            out = ChaCha20Poly1305(self.k).decrypt(_nonce(self.n), ciphertext, ad)
        except InvalidTag as exc:
            raise NoiseError("message does not decrypt") from exc
        self.n += 1  # only after a successful decryption (spec 5.1)
        return out


# ── SymmetricState ───────────────────────────────────────────────────────────────────────────────


class SymmetricState:
    def __init__(self, protocol_name: str) -> None:
        name = protocol_name.encode()
        self.h = name.ljust(HASHLEN, b"\x00") if len(name) <= HASHLEN else hashlib.sha256(name).digest()
        self.ck = self.h
        self.cipher = CipherState()

    def mix_key(self, ikm: bytes) -> None:
        self.ck, temp_k = hkdf(self.ck, ikm, 2)
        self.cipher = CipherState(temp_k[:32])

    def mix_hash(self, data: bytes) -> None:
        self.h = hashlib.sha256(self.h + data).digest()

    def mix_key_and_hash(self, ikm: bytes) -> None:
        self.ck, temp_h, temp_k = hkdf(self.ck, ikm, 3)
        self.mix_hash(temp_h)
        self.cipher = CipherState(temp_k[:32])

    def encrypt_and_hash(self, plaintext: bytes) -> bytes:
        ciphertext = self.cipher.encrypt(self.h, plaintext)
        self.mix_hash(ciphertext)
        return ciphertext

    def decrypt_and_hash(self, ciphertext: bytes) -> bytes:
        plaintext = self.cipher.decrypt(self.h, ciphertext)
        self.mix_hash(ciphertext)
        return plaintext

    def split(self) -> tuple[CipherState, CipherState]:
        k1, k2 = hkdf(self.ck, b"", 2)
        return CipherState(k1[:32]), CipherState(k2[:32])


# ── HandshakeState ───────────────────────────────────────────────────────────────────────────────


@dataclass
class Transport:
    """The result of a finished handshake: one CipherState per direction."""

    send: CipherState
    receive: CipherState
    handshake_hash: bytes
    remote_static: bytes

    def encrypt(self, plaintext: bytes) -> bytes:
        return self.send.encrypt(b"", plaintext)

    def decrypt(self, ciphertext: bytes) -> bytes:
        return self.receive.decrypt(b"", ciphertext)


@dataclass
class Handshake:
    """One side of an IK or IKpsk1 handshake (two messages; the initiator writes first)."""

    protocol_name: str
    initiator: bool
    prologue: bytes
    s: KeyPair
    rs: Optional[bytes] = None  # the responder's static key: required for the initiator
    psk: Optional[bytes] = None
    e: Optional[KeyPair] = None  # fixed ephemeral, for test vectors only
    re: Optional[bytes] = None
    _sym: SymmetricState = field(init=False)
    _step: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if self.protocol_name not in _PATTERNS:
            raise ValueError(f"unsupported protocol {self.protocol_name}")
        if self.initiator and self.rs is None:
            raise ValueError("the initiator must know the responder's static key")
        uses_psk = "psk" in _PATTERNS[self.protocol_name][0]
        if uses_psk and (self.psk is None or len(self.psk) != 32):
            raise ValueError("this pattern needs a 32-byte psk")
        self._psk_mode = uses_psk
        self._sym = SymmetricState(self.protocol_name)
        self._sym.mix_hash(self.prologue)
        # Pre-message ``<- s``.
        self._sym.mix_hash(self.rs if self.initiator else self.s.public)

    @property
    def handshake_hash(self) -> bytes:
        return self._sym.h

    def _local_writes(self) -> bool:
        return (self._step == 0) == self.initiator

    def _dh(self, token: str) -> bytes:
        # A DH token names (initiator's key, responder's key).
        if token == "ee":
            return dh(self.e.private, self.re)
        if token == "ss":
            return dh(self.s.private, self.rs)
        if token == "es":  # initiator's ephemeral, responder's static
            return dh(self.e.private, self.rs) if self.initiator else dh(self.s.private, self.re)
        if token == "se":  # initiator's static, responder's ephemeral
            return dh(self.s.private, self.re) if self.initiator else dh(self.e.private, self.rs)
        raise AssertionError(token)

    def write_message(self, payload: bytes = b"") -> bytes:
        if self._step > 1 or not self._local_writes():
            raise NoiseError("not this side's turn to write")
        out = bytearray()
        for token in _PATTERNS[self.protocol_name][self._step]:
            if token == "e":
                if self.e is None:
                    self.e = KeyPair.generate()
                out += self.e.public
                self._sym.mix_hash(self.e.public)
                if self._psk_mode:
                    self._sym.mix_key(self.e.public)
            elif token == "s":
                out += self._sym.encrypt_and_hash(self.s.public)
            elif token == "psk":
                self._sym.mix_key_and_hash(self.psk)
            else:
                self._sym.mix_key(self._dh(token))
        out += self._sym.encrypt_and_hash(payload)
        self._step += 1
        return bytes(out)

    def read_message(self, message: bytes) -> bytes:
        if self._step > 1 or self._local_writes():
            raise NoiseError("not this side's turn to read")
        data = memoryview(message)
        for token in _PATTERNS[self.protocol_name][self._step]:
            if token == "e":
                if len(data) < DHLEN:
                    raise NoiseError("message too short")
                self.re = bytes(data[:DHLEN])
                data = data[DHLEN:]
                self._sym.mix_hash(self.re)
                if self._psk_mode:
                    self._sym.mix_key(self.re)
            elif token == "s":
                size = DHLEN + (TAGLEN if self._sym.cipher.has_key() else 0)
                if len(data) < size:
                    raise NoiseError("message too short")
                self.rs = self._sym.decrypt_and_hash(bytes(data[:size]))
                data = data[size:]
            elif token == "psk":
                self._sym.mix_key_and_hash(self.psk)
            else:
                self._sym.mix_key(self._dh(token))
        if self._sym.cipher.has_key() and len(data) < TAGLEN:
            raise NoiseError("message too short")
        payload = self._sym.decrypt_and_hash(bytes(data))
        self._step += 1
        return payload

    @property
    def done(self) -> bool:
        return self._step == 2

    def transport(self) -> Transport:
        if not self.done:
            raise NoiseError("the handshake is not finished")
        c1, c2 = self._sym.split()
        send, receive = (c1, c2) if self.initiator else (c2, c1)
        return Transport(send, receive, self._sym.h, self.rs)


def responder(protocol_name: str, prologue: bytes, static: KeyPair, *, psk: Optional[bytes] = None) -> Handshake:
    return Handshake(protocol_name, False, prologue, static, psk=psk)


def initiator(protocol_name: str, prologue: bytes, static: KeyPair, remote_static: bytes, *,
              psk: Optional[bytes] = None) -> Handshake:
    return Handshake(protocol_name, True, prologue, static, rs=remote_static, psk=psk)
