"""Scheduled tasks on Aleph (plugins/platforms/aleph/schedule.py): Hermes's cron jobs as the phone
sees them, schedule.list/pause/resume/run/delete, and schedule.changed for any writer."""

from __future__ import annotations

import asyncio

import pytest
import pytest_asyncio

from gateway.config import PlatformConfig
from plugins.platforms.aleph import schedule
from tests.gateway.aleph_phone import connect, pair_device


def _job(**kw):
    base = {"id": "j1", "name": "Daily", "prompt": "Say hi", "enabled": True, "state": "scheduled",
            "schedule": {"kind": "cron", "expr": "0 10 * * *"}, "deliver": "aleph",
            "next_run_at": "2026-11-11T10:00:00-05:00", "repeat": {"times": None, "completed": 0}}
    base.update(kw)
    return base


class TestProject:
    def test_cron_task_with_the_agent_zone(self, monkeypatch):
        import hermes_time
        monkeypatch.setattr(hermes_time, "get_timezone_name", lambda: "America/New_York")
        task = schedule.project(_job())
        assert task == {"id": "j1", "name": "Daily", "prompt": "Say hi",
                        "schedule": {"kind": "cron", "expr": "0 10 * * *", "timezone": "America/New_York"},
                        "state": "scheduled", "next_run_at": "2026-11-11T10:00:00-05:00",
                        "last_run_at": None, "last_status": None, "last_error": None, "deliver": ["aleph"]}

    def test_a_pinned_zone_wins(self):
        assert schedule.project(_job(timezone="Asia/Tokyo"))["schedule"]["timezone"] == "Asia/Tokyo"

    def test_interval_and_once(self):
        assert schedule.project(_job(schedule={"kind": "interval", "minutes": 20160}))["schedule"] == {
            "kind": "interval", "minutes": 20160}
        once = schedule.project(_job(schedule={"kind": "once", "run_at": "2026-12-01T09:00:00-05:00"},
                                     repeat={"times": 1, "completed": 0}))
        assert once["schedule"] == {"kind": "once", "at": "2026-12-01T09:00:00-05:00"}
        assert "repeat" not in once

    def test_deliver_is_platform_names(self):
        job = _job(deliver="signal:+15551234567, origin,local,telegram:42,signal",
                   origin={"platform": "Discord", "chat_id": "c1"})
        assert schedule.project(job)["deliver"] == ["signal", "discord", "telegram"]
        assert schedule.project(_job(deliver="local"))["deliver"] == []
        assert schedule.project(_job(deliver=["aleph", "origin"], origin=None))["deliver"] == ["aleph"]

    def test_state_running_paused_and_next_run(self):
        assert schedule.project(_job(live_execution={"id": "e1"}))["state"] == "running"
        paused = schedule.project(_job(enabled=False, state="paused"))
        assert paused["state"] == "paused" and paused["next_run_at"] is None
        assert schedule.project(_job(state="completed"))["next_run_at"] is None

    def test_last_run_and_its_error_on_one_line(self):
        failed = schedule.project(_job(last_run_at="2026-11-10T10:00:00-05:00", last_status="error",
                                       last_error="boom\n" + "x" * 400))
        assert failed["last_status"] == "error" and failed["last_run_at"] == "2026-11-10T10:00:00-05:00"
        assert "\n" not in failed["last_error"] and len(failed["last_error"]) == schedule.MAX_ERROR_CHARS
        late = schedule.project(_job(last_run_at="t", last_status="delivery_failed", last_error=None,
                                     last_delivery_error="signal: not linked"))
        assert (late["last_status"], late["last_error"]) == ("delivery_failed", "signal: not linked")
        assert schedule.project(_job(last_run_at="t", last_status="ok"))["last_error"] is None

    def test_secrets_in_an_error_are_redacted(self):
        task = schedule.project(_job(last_run_at="t", last_status="error",
                                     last_error="401 with key sk-ant-api03-" + "a" * 40))
        assert "a" * 40 not in task["last_error"]

    def test_repeat_and_long_prompts(self):
        task = schedule.project(_job(repeat={"times": 5, "completed": 3}, prompt="p" * 5000))
        assert task["repeat"] == {"times": 5, "done": 3}
        assert len(task["prompt"]) == schedule.MAX_PROMPT_CHARS

    def test_order_soonest_then_name_then_stopped_last(self):
        tasks = [schedule.project(j) for j in (
            _job(id="c", name="b", next_run_at="2026-11-11T10:00:00-05:00"),
            _job(id="p", name="a", enabled=False, state="paused"),
            _job(id="b", name="a", next_run_at="2026-11-11T15:00:00+00:00"),
            _job(id="a", name="z", next_run_at="2026-11-11T09:00:00-05:00"))]
        assert [t["id"] for t in sorted(tasks, key=schedule._order)] == ["a", "b", "c", "p"]


class TestStore:
    """The real job store under the test's HERMES_HOME."""

    def test_list_pause_resume_delete(self):
        from cron.jobs import create_job, get_job
        daily = create_job(prompt="Morning summary", schedule="0 10 * * *", name="Daily", deliver="aleph")
        create_job(prompt="Check", schedule="every 30m", name="Often", deliver="local")
        names = [t["name"] for t in schedule.list_tasks()]
        assert names == ["Often", "Daily"]

        assert schedule.pause(daily["id"])["state"] == "paused"
        assert schedule.pause(daily["id"])["state"] == "paused"  # again: as it is
        assert [t["name"] for t in schedule.list_tasks()][-1] == "Daily"
        resumed = schedule.resume(daily["id"])
        assert resumed["state"] == "scheduled" and resumed["next_run_at"]

        schedule.delete(daily["id"])
        assert get_job(daily["id"]) is None
        with pytest.raises(schedule.ScheduleError) as exc:
            schedule.delete(daily["id"])
        assert exc.value.unknown

    def test_unknown_ids(self):
        for action in (schedule.pause, schedule.resume, schedule.task):
            with pytest.raises(schedule.ScheduleError) as exc:
                action("nope")
            assert exc.value.unknown

    def test_run_is_an_extra_fire_that_keeps_the_schedule(self):
        from cron.jobs import create_job, get_job
        job = create_job(prompt="Weekly", schedule="0 20 * * 0", name="Weekly")
        before = get_job(job["id"])
        started = []
        task = schedule.run(job["id"], started.append)
        assert task["state"] == "running"
        assert [claimed["id"] for claimed in started] == [job["id"]]
        after = get_job(job["id"])
        assert after["next_run_at"] == before["next_run_at"]
        assert after["enabled"] and after["state"] == "scheduled"

    def test_run_refuses_paused_and_finished_tasks(self):
        from cron.jobs import create_job, pause_job, update_job
        paused = create_job(prompt="x", schedule="every 5m", name="p")
        pause_job(paused["id"])
        with pytest.raises(schedule.ScheduleError, match="resume it first"):
            schedule.run(paused["id"], lambda job: None)
        done = create_job(prompt="x", schedule="every 5m", name="d")
        update_job(done["id"], {"state": "completed", "enabled": False, "next_run_at": None})
        for action in (lambda i: schedule.run(i, lambda job: None), schedule.resume, schedule.pause):
            with pytest.raises(schedule.ScheduleError, match="finished"):
                action(done["id"])


@pytest_asyncio.fixture
async def adapter(tmp_path, monkeypatch):
    from plugins.platforms.aleph.adapter import AlephAdapter
    monkeypatch.setattr(schedule, "POLL_SECONDS", 0.05)
    config = PlatformConfig(enabled=True, extra={"host": "127.0.0.1", "port": 0,
                                                 "state_dir": str(tmp_path / "aleph")})
    instance = AlephAdapter(config)
    assert await instance.connect()
    instance.device, instance.keys = pair_device(instance)
    yield instance
    await instance.disconnect()


async def _changed(phone, timeout=3.0):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        frame = await phone.event(max(0.05, deadline - loop.time()))
        if frame["params"]["type"] == "schedule.changed":
            return frame["params"]


@pytest.mark.asyncio
async def test_the_phone_lists_and_pauses_and_hears_outside_changes(adapter, monkeypatch):
    from cron.jobs import create_job
    job = await asyncio.to_thread(create_job, prompt="Morning", schedule="0 10 * * *", name="Daily",
                                  deliver="aleph")
    await adapter._schedule.check()  # the watch has read the store once
    phone = await connect(adapter, adapter.keys)
    try:
        await phone.hello()
        answer = await phone.call("schedule.list")
        assert [t["id"] for t in answer["result"]["tasks"]] == [job["id"]]

        paused = await phone.call("schedule.pause", {"id": job["id"]})
        assert paused["result"]["task"]["state"] == "paused"
        assert (await _changed(phone))["type"] == "schedule.changed"

        unknown = await phone.call("schedule.run", {"id": "nope"})
        assert unknown["error"]["code"] == -32602
        refused = await phone.call("schedule.run", {"id": job["id"]})
        assert refused["error"]["code"] == -32602 and "resume" in refused["error"]["message"]

        # Another writer (the agent's cronjob tool, the CLI) adds a task.
        await asyncio.to_thread(create_job, prompt="Later", schedule="every 2h", name="Later")
        await _changed(phone)
        listed = await phone.call("schedule.list")
        assert {t["name"] for t in listed["result"]["tasks"]} == {"Daily", "Later"}

        assert (await phone.call("schedule.delete", {"id": job["id"]}))["result"] == {}
    finally:
        await phone.close()
