"""Integration tests for the generic webhook platform adapter.

These tests exercise end-to-end flows through the webhook adapter:
1. GitHub PR webhook → agent MessageEvent created
2. Skills config injects skill content into the prompt
3. Cross-platform delivery routes to a mock Telegram adapter
4. GitHub comment delivery invokes ``gh`` CLI (mocked subprocess)
"""

import asyncio
import hashlib
import hmac
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import (
    GatewayConfig,
    Platform,
    PlatformConfig,
)
from gateway.platforms.base import MessageEvent, ProcessingOutcome, SendResult
from gateway.platforms.webhook import WebhookAdapter, _INSECURE_NO_AUTH
from gateway.main_session import MainSessionEnqueueResult
from gateway.session import SessionSource


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adapter(routes, **extra_kw) -> WebhookAdapter:
    """Create a WebhookAdapter with the given routes."""
    extra = {"host": "0.0.0.0", "port": 0, "routes": routes}
    extra.update(extra_kw)
    config = PlatformConfig(enabled=True, extra=extra)
    return WebhookAdapter(config)


def _create_app(adapter: WebhookAdapter) -> web.Application:
    """Build the aiohttp Application from the adapter."""
    app = web.Application()
    app.router.add_get("/health", adapter._handle_health)
    app.router.add_post("/webhooks/{route_name}", adapter._handle_webhook)
    return app


def _github_signature(body: bytes, secret: str) -> str:
    """Compute X-Hub-Signature-256 for *body* using *secret*."""
    return "sha256=" + hmac.new(
        secret.encode(), body, hashlib.sha256
    ).hexdigest()


# A realistic GitHub pull_request event payload (trimmed)
GITHUB_PR_PAYLOAD = {
    "action": "opened",
    "number": 42,
    "pull_request": {
        "title": "Add webhook adapter",
        "body": "This PR adds a generic webhook platform adapter.",
        "html_url": "https://github.com/org/repo/pull/42",
        "user": {"login": "contributor"},
        "head": {"ref": "feature/webhooks"},
        "base": {"ref": "main"},
    },
    "repository": {
        "full_name": "org/repo",
        "html_url": "https://github.com/org/repo",
    },
    "sender": {"login": "contributor"},
}


# ===================================================================
# Test 1: GitHub PR webhook triggers agent
# ===================================================================

class TestGitHubPRWebhook:

    @pytest.mark.asyncio
    async def test_github_pr_webhook_triggers_agent(self):
        """POST with a realistic GitHub PR payload should:
        1. Return 202 Accepted
        2. Call handle_message with a MessageEvent
        3. The event text contains the rendered prompt
        4. The event source has chat_type 'webhook'
        """
        secret = "gh-webhook-test-secret"
        routes = {
            "github-pr": {
                "secret": secret,
                "events": ["pull_request"],
                "prompt": (
                    "Review PR #{number} by {sender.login}: "
                    "{pull_request.title}\n\n{pull_request.body}"
                ),
                "deliver": "log",
            }
        }
        adapter = _make_adapter(routes)

        captured_events: list[MessageEvent] = []

        async def _capture(event: MessageEvent):
            captured_events.append(event)

        adapter.handle_message = _capture

        app = _create_app(adapter)
        body = json.dumps(GITHUB_PR_PAYLOAD).encode()
        sig = _github_signature(body, secret)

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/webhooks/github-pr",
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "X-GitHub-Event": "pull_request",
                    "X-Hub-Signature-256": sig,
                    "X-GitHub-Delivery": "gh-delivery-001",
                },
            )
            assert resp.status == 202
            data = await resp.json()
            assert data["status"] == "accepted"
            assert data["route"] == "github-pr"
            assert data["event"] == "pull_request"
            assert data["delivery_id"] == "gh-delivery-001"

        # Let the asyncio.create_task fire
        await asyncio.sleep(0.05)

        assert len(captured_events) == 1
        event = captured_events[0]
        assert "Review PR #42 by contributor" in event.text
        assert "Add webhook adapter" in event.text
        assert event.source.chat_type == "webhook"
        assert event.source.platform == Platform.WEBHOOK
        assert "github-pr" in event.source.chat_id
        assert event.source.parent_chat_id == "webhook-hooks"
        assert event.message_id == "gh-delivery-001"


class TestMainSessionWebhook:

    @pytest.mark.asyncio
    async def test_main_route_enters_home_fifo_without_webhook_session(self):
        routes = {
            "personal-alert": {
                "secret": _INSECURE_NO_AUTH,
                "prompt": "Please handle: {message}",
                "session": "main",
                "filter_before_main": False,
            }
        }
        adapter = _make_adapter(routes)
        runner = MagicMock()
        adapter.gateway_runner = runner
        adapter.handle_message = AsyncMock()
        home_source = SessionSource(
            platform=Platform.SIGNAL,
            chat_id="+15551234567",
            chat_type="dm",
            user_id="+15551234567",
        )

        with patch(
            "gateway.main_session.resolve_main_session_source",
            return_value=home_source,
        ) as resolve, patch(
            "gateway.main_session.enqueue_main_session_turn",
            new=AsyncMock(
                return_value=MainSessionEnqueueResult(
                    session_key="agent:main:signal:dm:+15551234567",
                    platform="signal",
                    chat_id="+15551234567",
                    queued=True,
                )
            ),
        ) as enqueue:
            async with TestClient(TestServer(_create_app(adapter))) as cli:
                resp = await cli.post(
                    "/webhooks/personal-alert",
                    json={"message": "Build finished"},
                    headers={"X-GitHub-Delivery": "main-hook-001"},
                )
                data = await resp.json()

        assert resp.status == 202
        assert data["session"] == "main"
        assert data["queued"] is True
        resolve.assert_called_once_with(runner, profile=None)
        queued = enqueue.await_args.kwargs
        assert queued["source"] == home_source
        assert "Build finished" in queued["text"]
        assert queued["metadata"]["trigger"] == "webhook"
        adapter.handle_message.assert_not_awaited()
        assert adapter._delivery_info == {}

    @pytest.mark.asyncio
    async def test_main_route_reviews_then_enqueues_only_the_handoff(self):
        routes = {
            "personal-alert": {
                "secret": _INSECURE_NO_AUTH,
                "prompt": "Build notification: {message}",
                "session": "main",
            }
        }
        adapter = _make_adapter(routes)
        runner = MagicMock()
        adapter.gateway_runner = runner
        adapter._end_webhook_session = AsyncMock()
        captured_events: list[MessageEvent] = []

        async def _capture(event: MessageEvent):
            captured_events.append(event)

        adapter.handle_message = _capture
        home_source = SessionSource(
            platform=Platform.SIGNAL,
            chat_id="+15551234567",
            chat_type="dm",
            user_id="+15551234567",
        )
        async def _enqueue_and_complete(*args, **kwargs):
            await kwargs["processing_complete"](ProcessingOutcome.SUCCESS)
            return MainSessionEnqueueResult(
                session_key="agent:main:signal:dm:+15551234567",
                platform="signal",
                chat_id="+15551234567",
                queued=True,
            )

        enqueue = AsyncMock(side_effect=_enqueue_and_complete)

        with patch(
            "gateway.main_session.resolve_main_session_source",
            return_value=home_source,
        ), patch(
            "gateway.main_session.enqueue_main_session_turn",
            new=enqueue,
        ):
            async with TestClient(TestServer(_create_app(adapter))) as cli:
                resp = await cli.post(
                    "/webhooks/personal-alert",
                    json={"message": "Build finished successfully"},
                    headers={"X-GitHub-Delivery": "review-main-001"},
                )
                data = await resp.json()

            await asyncio.sleep(0.05)
            assert resp.status == 202
            assert data["session"] == "main"
            assert data["reviewing"] is True
            assert data["durable"] is True
            assert enqueue.await_count == 0
            assert len(captured_events) == 1

            review_event = captured_events[0]
            assert review_event.source.chat_type == "webhook_review"
            assert review_event.source.chat_id.startswith("webhook-review:")
            assert review_event.source.parent_chat_id == "webhook-hooks"
            assert review_event.internal is True
            assert review_event.allow_gateway_control is False
            assert "reply with exactly NO_REPLY" in review_event.text
            assert "Build finished successfully" in review_event.text

            # Non-final status output is swallowed and cannot wake main.
            await adapter.send(
                review_event.source.chat_id,
                "provider switched",
                metadata={"non_conversational": True},
            )
            assert enqueue.await_count == 0

            handoff = (
                "CI build finished successfully; this unblocks deployment. "
                "Check the release queue next."
            )
            await adapter.send(
                review_event.source.chat_id,
                handoff,
                metadata={"notify": True},
            )
            await adapter.on_processing_complete(
                review_event, ProcessingOutcome.SUCCESS
            )

            # The completion hook durably writes pending_main; the outbox owns
            # the enqueue so HTTP/reviewer lifecycles never carry an in-memory
            # only handoff. Give that independently scheduled stage a turn.
            for _ in range(50):
                if enqueue.await_count:
                    break
                await asyncio.sleep(0.01)

        enqueue.assert_awaited_once()
        queued = enqueue.await_args.kwargs
        assert queued["source"] == home_source
        assert handoff in queued["text"]
        assert "Build notification:" not in queued["text"]
        assert queued["metadata"]["filtered"] is True
        assert queued["raw_message"] is None
        adapter._end_webhook_session.assert_awaited_once_with(
            review_event, review_event.source.chat_id
        )

    @pytest.mark.asyncio
    async def test_main_route_no_reply_never_enters_main(self):
        adapter = _make_adapter({
            "personal-alert": {
                "secret": _INSECURE_NO_AUTH,
                "prompt": "{message}",
                "session": "main",
            }
        })
        adapter.gateway_runner = MagicMock()
        adapter._end_webhook_session = AsyncMock()
        home_source = SessionSource(
            platform=Platform.SIGNAL,
            chat_id="+15551234567",
            chat_type="dm",
            user_id="+15551234567",
        )
        captured_events: list[MessageEvent] = []

        async def _capture(event: MessageEvent):
            captured_events.append(event)

        adapter.handle_message = _capture
        enqueue = AsyncMock()
        with patch(
            "gateway.main_session.resolve_main_session_source",
            return_value=home_source,
        ), patch(
            "gateway.main_session.enqueue_main_session_turn",
            new=enqueue,
        ):
            async with TestClient(TestServer(_create_app(adapter))) as cli:
                resp = await cli.post(
                    "/webhooks/personal-alert",
                    json={"message": "routine heartbeat"},
                    headers={"X-GitHub-Delivery": "review-main-silent-001"},
                )
            await asyncio.sleep(0.05)
            assert resp.status == 202
            review_event = captured_events[0]
            # The normal gateway silence filter suppresses NO_REPLY before send,
            # so an empty authoritative capture is the expected completion state.
            await adapter.on_processing_complete(
                review_event, ProcessingOutcome.SUCCESS
            )

        enqueue.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_main_review_returns_503_when_durable_accept_fails(self):
        adapter = _make_adapter({
            "personal-alert": {
                "secret": _INSECURE_NO_AUTH,
                "prompt": "{message}",
                "session": "main",
            }
        })
        adapter.gateway_runner = MagicMock()
        adapter.handle_message = AsyncMock()
        home_source = SessionSource(
            platform=Platform.SIGNAL,
            chat_id="+15551234567",
            chat_type="dm",
            user_id="+15551234567",
        )

        with patch(
            "gateway.main_session.resolve_main_session_source",
            return_value=home_source,
        ), patch.object(
            adapter._review_store,
            "get",
            return_value=None,
        ), patch.object(
            adapter._review_store,
            "accept",
            side_effect=OSError("state.db unavailable"),
        ) as accept:
            async with TestClient(TestServer(_create_app(adapter))) as cli:
                first = await cli.post(
                    "/webhooks/personal-alert",
                    json={"message": "do not lose me"},
                    headers={"X-GitHub-Delivery": "durable-fail-001"},
                )
                second = await cli.post(
                    "/webhooks/personal-alert",
                    json={"message": "do not lose me"},
                    headers={"X-GitHub-Delivery": "durable-fail-001"},
                )

        assert first.status == 503
        assert second.status == 503
        assert accept.call_count == 2
        adapter.handle_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_main_route_failure_is_retryable(self):
        adapter = _make_adapter({
            "personal-alert": {
                "secret": _INSECURE_NO_AUTH,
                "prompt": "{message}",
                "session": "main",
            }
        })
        adapter.gateway_runner = MagicMock()

        with patch(
            "gateway.main_session.resolve_main_session_source",
            side_effect=RuntimeError("gateway down"),
        ):
            async with TestClient(TestServer(_create_app(adapter))) as cli:
                first = await cli.post(
                    "/webhooks/personal-alert",
                    json={"message": "retry me"},
                    headers={"X-GitHub-Delivery": "retry-main-001"},
                )
                second = await cli.post(
                    "/webhooks/personal-alert",
                    json={"message": "retry me"},
                    headers={"X-GitHub-Delivery": "retry-main-001"},
                )

        assert first.status == 503
        assert second.status == 503


# ===================================================================
# Test 2: Skills injected into prompt
# ===================================================================

class TestSkillsInjection:

    @pytest.mark.asyncio
    async def test_skills_injected_into_prompt(self):
        """When a route has skills: [code-review], the adapter should
        call build_skill_invocation_message() and use its output as the
        prompt instead of the raw template render."""
        routes = {
            "pr-review": {
                "secret": _INSECURE_NO_AUTH,
                "events": ["pull_request"],
                "prompt": "Review this PR: {pull_request.title}",
                "skills": ["code-review"],
            }
        }
        adapter = _make_adapter(routes)

        captured_events: list[MessageEvent] = []

        async def _capture(event: MessageEvent):
            captured_events.append(event)

        adapter.handle_message = _capture

        skill_content = (
            "You are a code reviewer. Review the following:\n"
            "Review this PR: Add webhook adapter"
        )

        # The imports are lazy (inside the handler), so patch the source module
        with patch(
            "agent.skill_commands.build_skill_invocation_message",
            return_value=skill_content,
        ) as mock_build, patch(
            "agent.skill_commands.get_skill_commands",
            return_value={"/code-review": {"name": "code-review"}},
        ):
            app = _create_app(adapter)
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post(
                    "/webhooks/pr-review",
                    json=GITHUB_PR_PAYLOAD,
                    headers={
                        "X-GitHub-Event": "pull_request",
                        "X-GitHub-Delivery": "skill-test-001",
                    },
                )
                assert resp.status == 202

            await asyncio.sleep(0.05)

            assert len(captured_events) == 1
            event = captured_events[0]
            # The prompt should be the skill content, not the raw template
            assert "You are a code reviewer" in event.text
            mock_build.assert_called_once()


# ===================================================================
# Test 3: Cross-platform delivery (webhook → Telegram)
# ===================================================================

class TestCrossPlatformDelivery:

    @pytest.mark.asyncio
    async def test_cross_platform_delivery(self):
        """When deliver='telegram', the response is routed to the
        Telegram adapter via gateway_runner.adapters."""
        routes = {
            "alerts": {
                "secret": _INSECURE_NO_AUTH,
                "prompt": "Alert: {message}",
                "deliver": "telegram",
                "deliver_extra": {"chat_id": "12345"},
            }
        }
        adapter = _make_adapter(routes)
        adapter.handle_message = AsyncMock()

        # Set up a mock gateway runner with a mock Telegram adapter
        mock_tg_adapter = AsyncMock()
        mock_tg_adapter.send = AsyncMock(return_value=SendResult(success=True))

        mock_runner = MagicMock()
        mock_runner.adapters = {Platform.TELEGRAM: mock_tg_adapter}
        mock_runner.config = GatewayConfig(
            platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake")}
        )
        adapter.gateway_runner = mock_runner

        # First, simulate a webhook POST to set up delivery_info
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/webhooks/alerts",
                json={"message": "Server is on fire!"},
                headers={"X-GitHub-Delivery": "alert-001"},
            )
            assert resp.status == 202

        # The adapter should have stored delivery info
        chat_id = "webhook:alerts:alert-001"
        assert chat_id in adapter._delivery_info

        # Now call send() as if the agent has finished
        result = await adapter.send(chat_id, "I've acknowledged the alert.")

        assert result.success is True
        mock_tg_adapter.send.assert_awaited_once_with(
            "12345", "I've acknowledged the alert.", metadata=None
        )
        # Delivery info is retained after send() so interim status messages
        # don't strand the final response (TTL-based cleanup happens on POST).
        assert chat_id in adapter._delivery_info


# ===================================================================
# Test 4: GitHub comment delivery via gh CLI
# ===================================================================

class TestGitHubCommentDelivery:

    @pytest.mark.asyncio
    async def test_github_comment_delivery(self):
        """When deliver='github_comment', the adapter invokes
        ``gh pr comment`` via subprocess.run (mocked)."""
        routes = {
            "pr-bot": {
                "secret": _INSECURE_NO_AUTH,
                "prompt": "Review: {pull_request.title}",
                "deliver": "github_comment",
                "deliver_extra": {
                    "repo": "{repository.full_name}",
                    "pr_number": "{number}",
                },
            }
        }
        adapter = _make_adapter(routes)
        adapter.handle_message = AsyncMock()

        # POST a webhook to set up delivery info
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/webhooks/pr-bot",
                json=GITHUB_PR_PAYLOAD,
                headers={
                    "X-GitHub-Event": "pull_request",
                    "X-GitHub-Delivery": "gh-comment-001",
                },
            )
            assert resp.status == 202

        chat_id = "webhook:pr-bot:gh-comment-001"
        assert chat_id in adapter._delivery_info

        # Verify deliver_extra was rendered with payload data
        delivery = adapter._delivery_info[chat_id]
        assert delivery["deliver_extra"]["repo"] == "org/repo"
        assert delivery["deliver_extra"]["pr_number"] == "42"

        # Mock subprocess.run and call send()
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Comment posted"
        mock_result.stderr = ""

        with patch(
            "gateway.platforms.webhook.subprocess.run",
            return_value=mock_result,
        ) as mock_run:
            result = await adapter.send(
                chat_id, "LGTM! The code looks great."
            )

        assert result.success is True
        mock_run.assert_called_once_with(
            [
                "gh", "pr", "comment", "42",
                "--repo", "org/repo",
                "--body", "LGTM! The code looks great.",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        # Delivery info is retained after send() so interim status messages
        # don't strand the final response (TTL-based cleanup happens on POST).
        assert chat_id in adapter._delivery_info
