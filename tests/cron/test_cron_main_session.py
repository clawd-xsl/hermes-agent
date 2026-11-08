"""Cron ``session: main`` — a fire becomes a real turn in the gateway home conversation.

Covers the job field (create/update doors, the no_agent invariant, the cron.session default),
the fire on the owner execution seam (pre-agent gates kept, no isolated agent built, the
scheduler delivers nothing itself), and the one-shot turn-runtime release at session finalize.
"""
import asyncio
import threading
from types import SimpleNamespace

import pytest

from cron.jobs import create_job, get_job, update_job


@pytest.fixture()
def tmp_cron_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    return tmp_path


class TestSessionField:
    def test_main_is_normalized_and_persisted(self, tmp_cron_dir):
        job = create_job(prompt="Use our conversation", schedule="every 1h", session="MAIN")
        assert job["session"] == "main"
        assert get_job(job["id"])["session"] == "main"

    def test_absent_when_the_job_follows_the_config_default(self, tmp_cron_dir):
        assert "session" not in create_job(prompt="Use the default", schedule="every 1h")

    def test_rejects_an_unknown_mode(self, tmp_cron_dir):
        with pytest.raises(ValueError, match="Invalid session mode"):
            create_job(prompt="Bad mode", schedule="every 1h", session="fork")

    def test_main_needs_an_agent_turn(self, tmp_cron_dir):
        scripts = tmp_cron_dir / "scripts"
        scripts.mkdir()
        (scripts / "watch.py").write_text("print('x')\n")
        with pytest.raises(ValueError, match="cannot be combined with no_agent"):
            create_job(prompt="", schedule="every 1h", script="watch.py", no_agent=True, session="main")
        job = create_job(prompt="", schedule="every 1h", script="watch.py", no_agent=True)
        with pytest.raises(ValueError, match="cannot be combined with no_agent"):
            update_job(job["id"], {"session": "main"})

    def test_update_sets_and_clears_the_override(self, tmp_cron_dir):
        job = create_job(prompt="Original", schedule="every 1h")
        assert update_job(job["id"], {"session": "main"})["session"] == "main"
        assert "session" not in update_job(job["id"], {"session": ""})
        with pytest.raises(ValueError, match="Invalid session mode"):
            update_job(job["id"], {"session": "fork"})

    def test_session_is_an_authored_definition_field(self):
        from cron.job_definition import JOB_DEFINITION_FIELDS
        assert "session" in JOB_DEFINITION_FIELDS


def test_mode_precedence_per_job_then_config_and_no_agent_stays_isolated():
    from cron.scheduler_main_session import cron_session_mode
    cfg = {"cron": {"session": "main"}}
    assert cron_session_mode({"id": "a"}, cfg) == "main"
    assert cron_session_mode({"id": "b", "session": "isolated"}, cfg) == "isolated"
    assert cron_session_mode({"id": "c", "no_agent": True}, cfg) == "isolated"
    assert cron_session_mode({"id": "d"}, {}) == "isolated"
    assert cron_session_mode({"id": "e", "session": "main"}, {}) == "main"


def test_main_prompt_drops_the_isolated_delivery_markers():
    from cron.scheduler_prompt import _build_job_prompt
    job = {"id": "j", "prompt": "Check the inbox"}
    main = _build_job_prompt(job, main_session=True)
    isolated = _build_job_prompt(job)
    assert "existing main conversation" in main and "[SILENT]" not in main and "[CRON_FAILURE]" not in main
    assert "NEVER create or update a cron job" in main
    assert "[SILENT]" in isolated
    assert main.endswith("Check the inbox") and isolated.endswith("Check the inbox")


@pytest.fixture()
def owner_loop():
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    thread.join(5)
    loop.close()


def _fire(cron_owner, monkeypatch, owner_loop, job, admit):
    import cron.scheduler as scheduler
    import cron.scheduler_main_session as main_session

    def no_agent(*a, **k):
        raise AssertionError("a main-session fire must never build an isolated cron agent")
    monkeypatch.setattr(scheduler, "_construct_cron_agent", no_agent)
    monkeypatch.setattr(main_session, "_owner", lambda: (SimpleNamespace(runner=None), owner_loop))
    monkeypatch.setattr(main_session, "_admit", admit)
    return scheduler.run_job(job)


def test_fire_commits_the_prepared_prompt_to_main_and_delivers_nothing(cron_owner, monkeypatch, owner_loop):
    from cron.scheduler import SILENT_MARKER
    seen = []

    async def admit(authority, job, text):
        seen.append(text)
        return SimpleNamespace(session_key="agent:main:signal:dm:+1", platform="signal", chat_id="+1",
                               queued=True, durable=True)
    job = {"id": "mainjob00001", "name": "inbox", "prompt": "Check the inbox", "session": "main"}
    success, doc, final, error = _fire(cron_owner, monkeypatch, owner_loop, job, admit)
    assert (success, final, error) == (True, SILENT_MARKER, None)
    assert "**Session:** main" in doc and "agent:main:signal:dm:+1" in doc and "queued" in doc
    (text,) = seen
    assert text.startswith("[Scheduled task: inbox]")
    assert "existing main conversation" in text and text.endswith("Check the inbox")


def test_fire_not_accepted_by_main_is_a_failed_run(cron_owner, monkeypatch, owner_loop):
    async def admit(authority, job, text):
        from gateway.main_session import MainSessionUnavailable
        raise MainSessionUnavailable("session: main needs one connected platform with home_channel configured")
    job = {"id": "mainjob00002", "name": "inbox", "prompt": "Check the inbox", "session": "main"}
    success, doc, final, error = _fire(cron_owner, monkeypatch, owner_loop, job, admit)
    assert success is False and final == ""
    assert "home_channel" in error and "home_channel" in doc


def test_config_default_main_routes_an_unmarked_job(cron_owner, monkeypatch, owner_loop):
    import cron.scheduler as scheduler
    monkeypatch.setattr("cron.scheduler_main_session._user_config", lambda: {"cron": {"session": "main"}})
    seen = []

    async def admit(authority, job, text):
        seen.append(job["id"])
        return SimpleNamespace(session_key="k", platform="signal", chat_id="+1", queued=False, durable=False)
    job = {"id": "mainjob00003", "name": "brief", "prompt": "Morning brief"}
    assert _fire(cron_owner, monkeypatch, owner_loop, job, admit)[0] is True
    assert seen == ["mainjob00003"]


def test_a_hand_edited_unknown_mode_fails_the_fire(cron_owner, monkeypatch, owner_loop):
    async def admit(authority, job, text):
        raise AssertionError("must not reach main")
    job = {"id": "mainjob00004", "name": "odd", "prompt": "x", "session": "fork"}
    success, _doc, final, error = _fire(cron_owner, monkeypatch, owner_loop, job, admit)
    assert success is False and "Invalid session mode" in error


def test_config_default_is_read_from_the_profile_config_file(tmp_path, monkeypatch):
    from cron.scheduler_main_session import cron_session_mode
    monkeypatch.setattr("cron.scheduler._get_hermes_home", lambda: tmp_path)
    assert cron_session_mode({"id": "a"}) == "isolated"  # no config.yaml at all
    (tmp_path / "config.yaml").write_text("cron:\n  session: main\n")
    assert cron_session_mode({"id": "a"}) == "main"


def test_fire_outside_a_session_owner_fails_closed(monkeypatch):
    from cron.scheduler_main_session import run_main_session_fire
    success, _doc, final, error = run_main_session_fire({"id": "j"}, "j", "job", "prompt")
    assert success is False and final == "" and "live gateway" in error


def test_admit_resolves_the_owning_profiles_home_and_commits_as_cron_main(monkeypatch, tmp_path):
    import cron.scheduler_main_session as main_session
    import gateway.main_session as gms
    seen = {}

    def resolve(runner, *, origin=None, profile=None):
        seen["resolve"] = (origin, profile)
        return "home-source"

    async def enqueue(runner, *, source, text, producer, metadata=None):
        seen["enqueue"] = (source, text, producer, metadata)
        return "receipt"
    monkeypatch.setattr(gms, "resolve_main_session_source", resolve)
    monkeypatch.setattr(gms, "enqueue_main_session_turn", enqueue)
    runner = SimpleNamespace(config=SimpleNamespace(multiplex_profiles=False),
                             _served_profile_for_home=lambda home: "work")
    authority = SimpleNamespace(runner=runner, db=SimpleNamespace(db_path=tmp_path / "state.db"), profile_id="p")
    job = {"id": "j1", "origin": {"platform": "signal", "chat_id": "+1", "session_key": "k"}}
    assert asyncio.run(main_session._admit(authority, job, "[Scheduled task: x]\n\ny")) == "receipt"
    assert seen["resolve"] == ({"platform": "signal", "chat_id": "+1", "session_key": "k"}, "work")
    assert seen["enqueue"] == ("home-source", "[Scheduled task: x]\n\ny", gms.CRON_PRODUCER,
                               {"trigger": "cron", "job_id": "j1"})
