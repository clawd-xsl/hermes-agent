"""Aleph's APNs sender: content-free alerts that wake the phone's notification service extension.

The push says only that something happened (protocol "Push"): a generic body and the event's
``seq``. The extension then fetches the event over the channel itself and rewrites the notice, so
Apple never sees the conversation.

HTTP/2 to ``api.push.apple.com`` (``api.sandbox.push.apple.com`` for development tokens) through
httpx, which needs the ``h2`` package; token auth with an ES256 JWT signed by the team's ``.p8`` key,
reused for up to 50 minutes. Without the key settings, or without ``h2``, push is off and nothing
else changes.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

PRODUCTION_HOST = "https://api.push.apple.com"
SANDBOX_HOST = "https://api.sandbox.push.apple.com"
ENVIRONMENTS = ("development", "production")
# Apple accepts a provider token for an hour and rejects refreshing it more than every 20 minutes.
JWT_MAX_AGE_SECONDS = 50 * 60
EXPIRATION_SECONDS = 24 * 60 * 60

# Logged events that notify, and the generic body each one shows until the extension replaces it.
BODY_MESSAGE = "New message"
BODY_APPROVAL = "Approval needed"
BODY_QUESTION = "Question"


def push_body_for(event_type: str, payload: dict[str, Any]) -> Optional[str]:
    """The generic body for a logged event that notifies, or ``None`` for one that does not.

    Notifies: the agent's text messages (attachment-only ones included), approval requests and
    questions. Not: tool rows, system notices, the person's own messages (their echo), receipts,
    edits, deletions and outcomes. Drafts and typing are never logged, so never reach here.
    """
    if event_type == "message.new":
        if payload.get("sender") != "agent" or (payload.get("kind") or "text") != "text":
            return None
        return BODY_MESSAGE
    if event_type == "approval.request":
        return BODY_APPROVAL
    if event_type == "clarify.request":
        return BODY_QUESTION
    return None


def build_payload(*, body: str, seq: int, chat: str) -> dict[str, Any]:
    """No title: the agent goes by the name the person gave it on the phone, which the server does
    not know; the notification service extension sets it. Notifications group by chat."""
    return {
        "aps": {
            "alert": {"body": body},
            "sound": "default",
            "mutable-content": 1,
            "thread-id": chat,
        },
        "aleph": {"seq": seq, "chat": chat},
    }


def build_headers(*, topic: str, now: Optional[float] = None) -> dict[str, str]:
    return {
        "apns-push-type": "alert",
        "apns-priority": "10",
        "apns-topic": topic,
        "apns-expiration": str(int((now if now is not None else time.time()) + EXPIRATION_SECONDS)),
    }


@dataclass(frozen=True)
class ApnsConfig:
    key_file: str
    key_id: str
    team_id: str
    topic: str

    @classmethod
    def from_settings(cls, *, key_file: str, key_id: str, team_id: str, topic: str) -> Optional[ApnsConfig]:
        """``None`` unless all four are set: push is optional."""
        values = [str(v or "").strip() for v in (key_file, key_id, team_id, topic)]
        if not all(values):
            if any(values):
                logger.warning("Aleph: push is off: set all of ALEPH_APNS_KEY_FILE, ALEPH_APNS_KEY_ID, "
                               "ALEPH_APNS_TEAM_ID and ALEPH_APNS_TOPIC")
            return None
        return cls(str(Path(values[0]).expanduser()), *values[1:])


# APNs reasons that mean the server's setup is wrong, not one device (developer.apple.com,
# "Handling notification responses from APNs"), with what to check.
CONFIG_PROBLEMS = {
    "TopicDisallowed": "the App ID is not registered for push notifications (enable Push in the Apple "
                       "developer account), or ALEPH_APNS_TOPIC is not its bundle id",
    "DeviceTokenNotForTopic": "the device token belongs to another app: check ALEPH_APNS_TOPIC",
    "BadTopic": "ALEPH_APNS_TOPIC is not a valid bundle id",
    "MissingTopic": "ALEPH_APNS_TOPIC is not set",
    "InvalidProviderToken": "Apple rejects the key: check ALEPH_APNS_KEY_ID, ALEPH_APNS_TEAM_ID and "
                            "that the .p8 key is enabled for APNs",
    "MissingProviderToken": "the push carried no provider token",
    "BadCertificateEnvironment": "the key is not enabled for this APNs environment (sandbox/production)",
}


@dataclass(frozen=True)
class PushResult:
    status: int
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.status == 200

    @property
    def config_problem(self) -> Optional[str]:
        """What is wrong with this server's push setup, if Apple's answer says so. Such an answer
        repeats for every push until someone fixes the key or the App ID."""
        return CONFIG_PROBLEMS.get(self.reason) if not self.ok else None

    @property
    def token_is_dead(self) -> bool:
        """Apple says this device token will never work again: stop using it."""
        return self.status == 410 or (self.status == 400 and self.reason == "BadDeviceToken")


class ApnsSender:
    """One HTTP/2 client per APNs host, shared by every push."""

    def __init__(self, config: ApnsConfig, *, key_pem: str,
                 client_factory: Optional[Callable[[], Any]] = None,
                 clock: Callable[[], float] = time.time) -> None:
        self.config = config
        self._key_pem = key_pem
        self._clock = clock
        self._client_factory = client_factory or _http2_client
        self._client: Any = None
        self._jwt: Optional[str] = None
        self._jwt_at = 0.0

    @classmethod
    def create(cls, config: Optional[ApnsConfig]) -> Optional[ApnsSender]:
        """A sender, or ``None`` (with the reason logged) when push cannot work here."""
        if config is None:
            return None
        try:
            import h2
        except ImportError:
            logger.warning("Aleph: push is off: APNs needs HTTP/2; install the h2 package in the Hermes "
                           "environment")
            return None
        try:
            import jwt
        except ImportError:
            logger.warning("Aleph: push is off: PyJWT is missing")
            return None
        try:
            key_pem = Path(config.key_file).read_text()
        except OSError as exc:
            logger.warning("Aleph: push is off: cannot read the APNs key %s: %s", config.key_file, exc)
            return None
        sender = cls(config, key_pem=key_pem)
        try:
            sender._token()
        except Exception as exc:
            logger.warning("Aleph: push is off: the APNs key %s cannot sign: %s", config.key_file, exc)
            return None
        return sender

    def _token(self, *, refresh: bool = False) -> str:
        now = self._clock()
        if refresh or self._jwt is None or now - self._jwt_at >= JWT_MAX_AGE_SECONDS:
            import jwt
            self._jwt = jwt.encode({"iss": self.config.team_id, "iat": int(now)}, self._key_pem,
                                   algorithm="ES256", headers={"kid": self.config.key_id})
            self._jwt_at = now
        return self._jwt

    async def send(self, device_token: str, environment: str, payload: dict[str, Any]) -> PushResult:
        host = SANDBOX_HOST if environment == "development" else PRODUCTION_HOST
        url = f"{host}/3/device/{device_token}"
        body = json.dumps(payload, separators=(",", ":")).encode()
        if self._client is None:
            self._client = self._client_factory()
        result = await self._post(url, body, self._token())
        if result.status == 403 and result.reason == "ExpiredProviderToken":
            result = await self._post(url, body, self._token(refresh=True))
        return result

    async def _post(self, url: str, body: bytes, token: str) -> PushResult:
        headers = build_headers(topic=self.config.topic, now=self._clock())
        headers["authorization"] = f"bearer {token}"
        headers["content-type"] = "application/json"
        response = await self._client.post(url, content=body, headers=headers)
        reason = ""
        if response.status_code != 200:
            try:
                reason = str((response.json() or {}).get("reason") or "")
            except Exception:
                reason = ""
        return PushResult(response.status_code, reason)

    async def close(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            await client.aclose()


def _http2_client():
    import httpx
    # trust_env: a machine that reaches the internet through HTTPS_PROXY pushes through it too.
    return httpx.AsyncClient(http2=True, timeout=httpx.Timeout(15.0, connect=10.0))
