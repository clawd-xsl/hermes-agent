"""``retain_sessions``: drop a settled one-shot webhook session, never a failed one or main."""
from types import SimpleNamespace

from gateway.config import PlatformConfig
from gateway.platforms import webhook_main
from gateway.platforms.webhook import WebhookAdapter, _INSECURE_NO_AUTH


def _adapter(routes, **extra) -> WebhookAdapter:
    return WebhookAdapter(PlatformConfig(enabled=True, extra={"host": "127.0.0.1", "port": 0, "routes": routes,
                                                              **extra}))


def _route(**kw):
    return {"secret": _INSECURE_NO_AUTH, "prompt": "Mail from {sender}: {subject}", **kw}


class TestRetainSessions:
    class _DB:
        def __init__(self):
            self.deleted = []

        def get_compression_tip(self, sid):
            return "tip-" + sid

        def delete_session(self, sid, **kwargs):
            self.deleted.append((sid, kwargs.get("include_compression_chain")))
            return True

    def _settle(self, adapter, monkeypatch, *, outcome="completed", chat_route="mail", failed=False):
        import asyncio
        from gateway.platforms.webhook import _WebhookDeliveryIdentity
        db = self._DB()
        chat_id = _WebhookDeliveryIdentity.from_parts(None, chat_route, "d").session_chat_id
        if failed:
            adapter._failed_main_handoffs.add(chat_id)
        receipt = SimpleNamespace(ref=SimpleNamespace(session_id="s1"), status="terminal", outcome=outcome)
        asyncio.run(webhook_main.drop_unretained_session(adapter, SimpleNamespace(db=db), receipt, chat_id))
        return db.deleted

    def test_default_keeps_the_session(self, monkeypatch):
        assert self._settle(_adapter({"mail": _route()}), monkeypatch) == []

    def test_false_deletes_a_successful_delivery_with_its_compression_chain(self, monkeypatch):
        assert self._settle(_adapter({"mail": _route(retain_sessions=False)}), monkeypatch) == [("s1", True)]

    def test_platform_default_false_is_inherited_and_failed_runs_are_kept(self, monkeypatch):
        adapter = _adapter({"mail": _route()}, retain_sessions="false")
        assert self._settle(adapter, monkeypatch) == [("s1", True)]
        assert self._settle(adapter, monkeypatch, outcome="failed") == []
        assert self._settle(adapter, monkeypatch, failed=True) == []
        assert not adapter._failed_main_handoffs


def test_a_failed_handoff_marks_the_reviewer_session_kept(monkeypatch):
    import asyncio
    from unittest.mock import AsyncMock
    from gateway.main_session import MainSessionUnavailable
    adapter = _adapter({"mail": _route(session="main")})
    monkeypatch.setattr(webhook_main, "enqueue_main", AsyncMock(side_effect=MainSessionUnavailable("down")))
    delivery = {"deliver": webhook_main.MAIN_DESTINATION, "route": "mail", "deliver_extra": {"event_type": "e"}}
    result = asyncio.run(webhook_main.handoff_to_main(adapter, "chat-1", "matters", delivery, {"notify": True}))
    assert result.success is False and "chat-1" in adapter._failed_main_handoffs


def test_platform_default_is_inherited_by_routes():
    adapter = _adapter({"a": _route(), "b": _route(retain_sessions=True)}, retain_sessions=False)
    assert webhook_main.route_retains_sessions(adapter, adapter._routes["a"]) is False
    assert webhook_main.route_retains_sessions(adapter, adapter._routes["b"]) is True


def test_a_real_session_row_and_its_messages_are_deleted(tmp_path):
    import asyncio
    from gateway.platforms.webhook import _WebhookDeliveryIdentity
    from hermes_state import SessionDB
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("hook-1", source="webhook")
        db.append_message("hook-1", "user", "external event")
        adapter = _adapter({"mail": _route(retain_sessions=False)})
        chat_id = _WebhookDeliveryIdentity.from_parts(None, "mail", "d").session_chat_id
        receipt = SimpleNamespace(ref=SimpleNamespace(session_id="hook-1"), status="terminal", outcome="completed")
        asyncio.run(webhook_main.drop_unretained_session(adapter, SimpleNamespace(db=db), receipt, chat_id))
        assert db.get_session("hook-1") is None
    finally:
        db.close()
