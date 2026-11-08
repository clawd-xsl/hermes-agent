"""`hermes webhook subscribe --session/--[no-]filter-before-main` and the dashboard webhook API."""
import pytest

from hermes_cli.webhook import _load_subscriptions, webhook_command
from tests.hermes_cli.test_dashboard_admin_endpoints import _client
from tests.hermes_cli.test_webhook_cli import _make_args


@pytest.fixture()
def _webhooks_on(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("hermes_cli.webhook._is_webhook_enabled", lambda: True)


def test_subscribe_stores_and_shows_the_main_session_mode(_webhooks_on, capsys):
    webhook_command(_make_args(webhook_action="subscribe", name="mail", session="main",
                               filter_before_main=None))
    route = _load_subscriptions()["mail"]
    assert route["session"] == "main" and "filter_before_main" not in route
    assert "Session: main (reviewed)" in capsys.readouterr().out
    webhook_command(_make_args(webhook_action="subscribe", name="ci", session="main", filter_before_main=False))
    assert _load_subscriptions()["ci"]["filter_before_main"] is False
    webhook_command(_make_args(webhook_action="list"))
    out = capsys.readouterr().out
    assert "Session: main (direct)" in out and "Session: main (reviewed)" in out


def test_plain_subscribe_keeps_the_platform_default(_webhooks_on, capsys):
    webhook_command(_make_args(webhook_action="subscribe", name="plain"))
    assert "session" not in _load_subscriptions()["plain"]
    assert "Session: default" in capsys.readouterr().out


def test_parser_offers_only_the_two_modes():
    import argparse
    from hermes_cli.subcommands.webhook import build_webhook_parser
    parser = argparse.ArgumentParser(prog="hermes")
    build_webhook_parser(parser.add_subparsers(dest="command"), cmd_webhook=lambda args: None)
    args = parser.parse_args(["webhook", "subscribe", "x", "--session", "main", "--no-filter-before-main"])
    assert (args.session, args.filter_before_main) == ("main", False)
    with pytest.raises(SystemExit):
        parser.parse_args(["webhook", "subscribe", "x", "--session", "fork"])


class TestDashboardWebhookSession:
    @pytest.fixture(autouse=True)
    def _setup(self, _isolate_hermes_home):
        from hermes_cli.config import load_config, save_config
        cfg = load_config()
        cfg.setdefault("platforms", {})["webhook"] = {"enabled": True, "extra": {"host": "0.0.0.0", "port": 8644}}
        save_config(cfg)
        self.client, _ = _client()

    def test_create_main_session_webhook(self):
        r = self.client.post("/api/webhooks", json={"name": "context-hook", "deliver": "log", "session": "main",
                                                    "filter_before_main": False})
        assert r.status_code == 200
        assert r.json()["session"] == "main" and r.json()["filter_before_main"] is False
        (sub,) = self.client.get("/api/webhooks").json()["subscriptions"]
        assert (sub["session"], sub["filter_before_main"]) == ("main", False)

    def test_default_route_reports_no_session_and_reviewed(self):
        assert self.client.post("/api/webhooks", json={"name": "plain"}).status_code == 200
        (sub,) = self.client.get("/api/webhooks").json()["subscriptions"]
        assert (sub["session"], sub["filter_before_main"]) == (None, True)

    def test_rejects_an_invalid_session(self):
        assert self.client.post("/api/webhooks", json={"name": "bad", "session": "fork"}).status_code == 400
