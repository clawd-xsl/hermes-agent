"""Behavior contract for the webhook ``retain_sessions`` option.

Each webhook delivery runs in a one-shot session (isolated route run or a
main-route reviewer). With ``retain_sessions: false`` that session is deleted
from state.db once the run succeeds; failed runs are kept for debugging, and
the default keeps every session (ended, as before).

These tests post through the real ``_handle_webhook`` route dispatch and the
real ``handle_message`` → ``on_processing_complete`` pipeline against a real
``SessionStore``/``SessionDB``; only the runner-side message handler is
stubbed, which is the seam the live gateway injects.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.main_session import MainSessionEnqueueResult
from gateway.platforms.base import MessageEvent
from gateway.platforms.webhook import WebhookAdapter, _INSECURE_NO_AUTH
from gateway.session import SessionSource, SessionStore


class _Runner:
    def __init__(self, store: SessionStore, sessions_dir):
        self.session_store = store
        self._session_db = store._db
        self.config = SimpleNamespace(sessions_dir=sessions_dir)

    def _session_key_for_source(self, source: SessionSource) -> str:
        return self.session_store._generate_session_key(source)


def _setup(tmp_path, routes, **extra):
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    store = SessionStore(
        sessions_dir=sessions_dir,
        config=GatewayConfig(platforms={Platform.WEBHOOK: PlatformConfig(enabled=True)}),
    )
    adapter = WebhookAdapter(
        PlatformConfig(
            enabled=True,
            extra={"host": "127.0.0.1", "port": 0, "routes": routes, **extra},
        )
    )
    adapter.gateway_runner = _Runner(store, sessions_dir)
    adapter.config.typing_indicator = False
    return adapter, store


def _install_handler(adapter, store, created, response="", fail=False):
    async def _handler(event: MessageEvent):
        entry = store.get_or_create_session(event.source)
        created.append(entry.session_id)
        if fail:
            raise RuntimeError("agent run failed")
        return response

    adapter._message_handler = _handler


async def _post(adapter, route, delivery_id):
    app = web.Application()
    app.router.add_post("/webhooks/{route_name}", adapter._handle_webhook)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            f"/webhooks/{route}",
            json={"message": "disk almost full"},
            headers={"X-GitHub-Delivery": delivery_id},
        )
        assert resp.status == 202
    deadline = asyncio.get_event_loop().time() + 5
    while adapter._background_tasks and asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(0.02)
    await asyncio.sleep(0.05)


def _route(**kw):
    return {"secret": _INSECURE_NO_AUTH, "prompt": "Alert: {message}", "deliver": "log", **kw}


@pytest.mark.asyncio
async def test_default_keeps_the_ended_session(tmp_path):
    adapter, store = _setup(tmp_path, {"alerts": _route()})
    created: list[str] = []
    _install_handler(adapter, store, created)

    await _post(adapter, "alerts", "keep-001")

    row = store._db.get_session(created[0])
    assert row is not None and row["end_reason"] == "webhook_complete"
    store._db.close()


@pytest.mark.asyncio
async def test_platform_default_false_deletes_successful_session(tmp_path):
    adapter, store = _setup(tmp_path, {"alerts": _route()}, retain_sessions=False)
    created: list[str] = []
    _install_handler(adapter, store, created)

    await _post(adapter, "alerts", "drop-001")

    assert created and store._db.get_session(created[0]) is None
    store._db.close()


@pytest.mark.asyncio
async def test_route_override_wins_over_platform_default(tmp_path):
    adapter, store = _setup(
        tmp_path,
        {"keep": _route(retain_sessions=True), "drop": _route(retain_sessions=False)},
        retain_sessions=False,
    )
    created: list[str] = []
    _install_handler(adapter, store, created)

    await _post(adapter, "keep", "route-keep-001")
    await _post(adapter, "drop", "route-drop-001")

    assert store._db.get_session(created[0]) is not None
    assert store._db.get_session(created[1]) is None
    store._db.close()


@pytest.mark.asyncio
async def test_failed_run_is_kept_even_when_not_retaining(tmp_path):
    adapter, store = _setup(tmp_path, {"alerts": _route(retain_sessions=False)})
    created: list[str] = []
    _install_handler(adapter, store, created, fail=True)

    await _post(adapter, "alerts", "fail-001")

    assert created and store._db.get_session(created[0]) is not None
    store._db.close()


@pytest.mark.asyncio
async def test_main_review_session_deleted_after_handoff_enqueued(tmp_path):
    adapter, store = _setup(
        tmp_path, {"mail": _route(session="main", retain_sessions=False)}
    )
    created: list[str] = []
    _install_handler(
        adapter, store, created, response="Disk on prod-1 is 97% full; clean up logs."
    )
    home = SessionSource(
        platform=Platform.SIGNAL, chat_id="+15551234567", chat_type="dm", user_id="+15551234567"
    )
    deleted_before_enqueue = {}

    async def _enqueue(runner, **kwargs):
        deleted_before_enqueue["value"] = store._db.get_session(created[0]) is None
        return MainSessionEnqueueResult(
            session_key="agent:main:signal:dm:+15551234567",
            platform="signal",
            chat_id="+15551234567",
            queued=False,
        )

    enqueue = AsyncMock(side_effect=_enqueue)
    with patch("gateway.main_session.resolve_main_session_source", return_value=home), patch(
        "gateway.main_session.enqueue_main_session_turn", new=enqueue
    ):
        await _post(adapter, "mail", "review-drop-001")

    enqueue.assert_awaited_once()
    assert "Disk on prod-1" in enqueue.await_args.kwargs["text"]
    assert deleted_before_enqueue["value"] is False
    assert store._db.get_session(created[0]) is None
    store._db.close()
