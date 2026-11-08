"""Webhook ``session: main`` routes: reviewed (default) and direct.

POSTs go through the real durable admission (``fixtures/webhook_route_authority``); only execution
is held, so the reviewer's reply is driven through ``send()`` exactly as the turn would deliver it.
"""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.main_session import MainSessionEnqueueResult, MainSessionUnavailable
from gateway.platforms import webhook_main
from gateway.platforms.webhook import WebhookAdapter, _INSECURE_NO_AUTH

_ACCEPTED = MainSessionEnqueueResult(session_key="agent:main:signal:dm:+1", platform="signal", chat_id="+1",
                                     queued=False, durable=True)


def _adapter(routes, **extra) -> WebhookAdapter:
    return WebhookAdapter(PlatformConfig(enabled=True, extra={"host": "127.0.0.1", "port": 0, "routes": routes,
                                                              **extra}))


def _app(adapter):
    from tests.gateway.fixtures.webhook_route_authority import mount_authority
    app = web.Application()
    mount_authority(app, adapter)
    app.router.add_post("/webhooks/{route_name}", adapter._handle_webhook)
    return app


def _route(**kw):
    return {"secret": _INSECURE_NO_AUTH, "prompt": "Mail from {sender}: {subject}", **kw}


async def _post(cli, route, delivery, **payload):
    return await cli.post(f"/webhooks/{route}", data=json.dumps(payload or {"sender": "a", "subject": "s"}),
                          headers={"Content-Type": "application/json", "X-GitHub-Delivery": delivery,
                                   "X-GitHub-Event": "message"})


class TestRouteValidation:
    def test_unknown_session_mode_refuses_the_route(self):
        adapter = _adapter({"bad": _route(session="fork")})
        with pytest.raises(ValueError, match="Route 'bad'.*Invalid session mode"):
            adapter._validate_route("bad", adapter._routes["bad"])

    def test_dynamic_route_with_unknown_mode_is_skipped(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "webhook_subscriptions.json").write_text(json.dumps({
            "ok": {"secret": "s3cret-value", "prompt": "x", "session": "main"},
            "bad": {"secret": "s3cret-value", "prompt": "x", "session": "fork"}}))
        adapter = _adapter({})
        adapter._reload_dynamic_routes()
        assert set(adapter._dynamic_routes) == {"ok"}

    def test_platform_defaults_apply_and_routes_override_them(self):
        adapter = _adapter({"a": _route(), "b": _route(session="isolated", filter_before_main="false")},
                           session="main")
        assert adapter._main_mode(adapter._routes["a"]) == "review"
        assert adapter._main_mode(adapter._routes["b"]) is None
        assert webhook_main.route_reviews_before_main(adapter, adapter._routes["b"]) is False


class TestReviewedMainRoute:
    @pytest.mark.asyncio
    async def test_event_is_admitted_as_a_zero_tool_reviewer_aimed_at_main(self, monkeypatch):
        adapter = _adapter({"mail": _route(session="main", skills=["triage"])})
        applied = []
        monkeypatch.setattr(adapter, "_apply_skills", lambda text, skills: applied.append(text) or f"SKILL:{text}")
        events = []
        adapter.handle_message = AsyncMock(side_effect=events.append)
        async with TestClient(TestServer(_app(adapter))) as cli:
            resp = await _post(cli, "mail", "d-1", sender="boss", subject="Contract")
            assert resp.status == 202
            assert (await resp.json())["session"] == "main" and (await resp.json())["reviewing"] is True
        (event,) = events
        assert event.source.chat_type == webhook_main.REVIEW_CHAT_TYPE
        assert "reply with exactly NO_REPLY" in event.text and "Mail from boss: Contract" in event.text
        assert event.text.rstrip().endswith('"Sent by the owner.\\n\\nNO_REPLY".')
        assert applied == []  # skills are for the handoff that wakes main, never the reviewer
        assert adapter.toolsets_for_source(event.source) == []
        assert adapter._delivery_info[event.source.chat_id]["deliver"] == webhook_main.MAIN_DESTINATION

    @pytest.mark.asyncio
    async def test_final_handoff_wakes_main_and_no_reply_or_interim_does_not(self, monkeypatch):
        adapter = _adapter({"mail": _route(session="main", skills=["triage"])})
        monkeypatch.setattr(adapter, "_apply_skills", lambda text, skills: f"SKILL[{','.join(skills)}]:{text}")
        enqueue = AsyncMock(return_value=_ACCEPTED)
        monkeypatch.setattr(webhook_main, "enqueue_main", enqueue)
        events = []
        adapter.handle_message = AsyncMock(side_effect=events.append)
        async with TestClient(TestServer(_app(adapter))) as cli:
            assert (await _post(cli, "mail", "d-2")).status == 202
            chat_id = events[0].source.chat_id
            assert (await adapter.send(chat_id, "NO_REPLY", metadata={"notify": True})).success
            assert (await adapter.send(chat_id, "⚠️ falling back to another model")).success
            enqueue.assert_not_awaited()
            result = await adapter.send(chat_id, "x" * 3000, metadata={"notify": True})
        assert result.success
        text = enqueue.await_args.kwargs["text"]
        assert text.startswith("SKILL[triage]:[External event: mail / message]\n\n")
        assert len(text.split("\n\n", 1)[1]) == webhook_main.MAX_MAIN_HANDOFF_CHARS and text.endswith("…")

    @pytest.mark.asyncio
    async def test_a_failed_handoff_fails_the_send(self, monkeypatch):
        adapter = _adapter({"mail": _route(session="main")})
        monkeypatch.setattr(webhook_main, "enqueue_main", AsyncMock(side_effect=MainSessionUnavailable("down")))
        events = []
        adapter.handle_message = AsyncMock(side_effect=events.append)
        async with TestClient(TestServer(_app(adapter))) as cli:
            assert (await _post(cli, "mail", "d-3")).status == 202
            chat_id = events[0].source.chat_id
            result = await adapter.send(chat_id, "It matters", metadata={"notify": True})
        assert result.success is False and "Main conversation unavailable" in result.error


class TestDirectMainRoute:
    @pytest.mark.asyncio
    async def test_trusted_route_commits_the_rendered_event_straight_into_main(self, monkeypatch):
        adapter = _adapter({"ci": _route(session="main", filter_before_main=False, skills=["ops"])})
        monkeypatch.setattr(adapter, "_apply_skills", lambda text, skills: f"SKILL:{text}")
        enqueue = AsyncMock(return_value=_ACCEPTED)
        monkeypatch.setattr(webhook_main, "enqueue_main", enqueue)
        adapter.handle_message = AsyncMock()
        async with TestClient(TestServer(_app(adapter))) as cli:
            resp = await _post(cli, "ci", "d-4", sender="ci", subject="red build")
            assert resp.status == 202 and (await resp.json())["reviewing"] is False
            assert (await _post(cli, "ci", "d-4")).status == 200  # provider retry is a duplicate
        adapter.handle_message.assert_not_awaited()  # no isolated run, no reviewer
        assert enqueue.await_count == 1
        assert enqueue.await_args.kwargs["text"] == "[External event: ci / message]\n\nSKILL:Mail from ci: red build"

    @pytest.mark.asyncio
    async def test_main_outage_is_retryable(self, monkeypatch):
        adapter = _adapter({"ci": _route(session="main", filter_before_main=False)})
        enqueue = AsyncMock(side_effect=[MainSessionUnavailable("down"), _ACCEPTED])
        monkeypatch.setattr(webhook_main, "enqueue_main", enqueue)
        async with TestClient(TestServer(_app(adapter))) as cli:
            assert (await _post(cli, "ci", "d-5")).status == 503
            assert (await _post(cli, "ci", "d-5")).status == 202
        assert enqueue.await_count == 2


def test_an_explicit_empty_toolset_override_means_zero_tools(monkeypatch):
    """A reviewer's [] must not fall back to the platform's default toolsets."""
    from gateway.run import GatewayRunner
    runner = object.__new__(GatewayRunner)
    reviewer = SimpleNamespace(toolsets_for_source=lambda source: [])
    monkeypatch.setattr(runner, "_delivery_adapter_for", lambda source: reviewer, raising=False)
    tools = runner._resolve_enabled_toolsets_for_source({}, SimpleNamespace(), "webhook")
    assert "terminal" not in tools and "web" not in tools
