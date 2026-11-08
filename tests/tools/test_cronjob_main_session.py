"""The cronjob tool surface for ``session: main`` and the durable origin identity it relies on."""
import json

import pytest

from tools.cronjob_tools import cronjob


@pytest.fixture(autouse=True)
def _cron_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    from gateway.session_context import clear_session_vars, set_session_vars
    tokens = set_session_vars()
    yield
    clear_session_vars(tokens)


def test_registered_handler_creates_lists_and_clears_a_main_job():
    from cron.jobs import get_job
    from tools.registry import registry

    created = json.loads(registry.dispatch("cronjob_manage", {
        "action": "create", "schedule": "every 1h", "prompt": "Use our conversation", "session": "main"}))
    assert created["success"] is True
    assert get_job(created["job_id"])["session"] == "main"
    listing = json.loads(registry.dispatch("cronjob_manage", {"action": "list"}))
    assert next(j for j in listing["jobs"] if j["job_id"] == created["job_id"])["session"] == "main"
    updated = json.loads(registry.dispatch("cronjob_manage", {
        "action": "update", "job_id": created["job_id"], "session": ""}))
    assert updated["success"] is True
    assert "session" not in get_job(created["job_id"])


def test_invalid_mode_is_a_tool_error():
    result = json.loads(cronjob(action="create", schedule="every 1h", prompt="x", session="fork"))
    assert result["success"] is False and "Invalid session mode" in result["error"]


def test_schema_offers_only_the_two_modes():
    from tools.cronjob_tools import CRONJOB_SCHEMA
    assert CRONJOB_SCHEMA["parameters"]["properties"]["session"]["enum"] == ["isolated", "main"]


def test_gateway_origin_keeps_the_durable_routing_identity():
    from cron.jobs import get_job
    from gateway.session_context import set_session_vars

    set_session_vars(platform="slack", chat_id="D123", chat_type="dm", chat_name="Assistant DM", user_id="U123",
                     user_name="Owner", scope_id="T123", session_key="agent:work:slack:dm:T123:D123",
                     profile="work")
    created = json.loads(cronjob(action="create", prompt="x", schedule="every 2m"))
    assert get_job(created["job_id"])["origin"] == {
        "platform": "slack", "chat_id": "D123", "chat_name": "Assistant DM", "chat_type": "dm",
        "thread_id": None, "user_id": "U123", "user_name": "Owner", "scope_id": "T123",
        "session_key": "agent:work:slack:dm:T123:D123", "profile": "work"}
