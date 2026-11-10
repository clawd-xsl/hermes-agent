"""The agent's scheduled tasks on Aleph (protocol "Scheduled tasks"): Hermes's cron jobs
(``cron/jobs.py``) as the phone shows them, the four actions the person takes on one directly, and
a watch that tells the phones when any writer (the agent's ``cronjob`` tool, the CLI, the
scheduler) changed them.

Store reads and writes run on a small executor of their own, so a component that fills the shared
default executor cannot stall the agent page.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

MAX_PROMPT_CHARS = 4000
MAX_ERROR_CHARS = 300
POLL_SECONDS = 2.0
# Without any write, a run's owner can still die (its process killed): re-read now and then.
RECHECK_SECONDS = 30.0
_UNREAD = object()


class ScheduleError(Exception):
    """A call the store refused; ``unknown`` when no task has that id."""

    def __init__(self, message: str, *, unknown: bool = False) -> None:
        super().__init__(message)
        self.unknown = unknown


# ── the task as the phone sees it ─────────────────────────────────────────────────────────────

def _one_line(text: Any, limit: int) -> Optional[str]:
    if not isinstance(text, str) or not text.strip():
        return None
    from agent.redact import redact_sensitive_text
    text = " ".join(redact_sensitive_text(text, redact_url_credentials=True).split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _zone_name(job: dict[str, Any]) -> Optional[str]:
    """The zone a cron expression is read in: the job's pin, else the agent's (its configured zone,
    else the machine's)."""
    from cron.jobs_time import zone_for_job
    zone = zone_for_job(job)
    if zone is not None:
        return zone.key
    try:
        from hermes_time import get_timezone_name
        name = get_timezone_name()
    except Exception:
        name = ""
    return name or os.environ.get("TZ") or _machine_zone()


def _machine_zone() -> Optional[str]:
    try:
        target = os.path.realpath("/etc/localtime")
    except OSError:
        return None
    marker = "zoneinfo/"
    return target.split(marker, 1)[1] if marker in target else None


def _schedule(job: dict[str, Any]) -> dict[str, Any]:
    schedule = job.get("schedule") if isinstance(job.get("schedule"), dict) else {}
    kind = schedule.get("kind")
    if kind == "cron":
        return {"kind": "cron", "expr": schedule.get("expr"), "timezone": _zone_name(job)}
    if kind == "interval":
        return {"kind": "interval", "minutes": schedule.get("minutes")}
    if kind == "once":
        return {"kind": "once", "at": schedule.get("run_at")}
    return {"kind": str(kind or "unknown")}


def _deliver(job: dict[str, Any]) -> list[str]:
    """Where the result goes, as platform names: no chat ids, ``origin`` replaced by the platform the
    task came from, ``local`` (kept on the server) left out."""
    raw = job.get("deliver")
    targets = raw if isinstance(raw, list) else str(raw or "").split(",")
    names: list[str] = []
    for target in targets:
        name = str(target or "").strip().split(":", 1)[0].strip().lower()
        if name == "origin":
            origin = job.get("origin")
            name = str(origin.get("platform") or "").strip().lower() if isinstance(origin, dict) else ""
        if name and name != "local" and name not in names:
            names.append(name)
    return names


def _last(job: dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
    status = job.get("last_status")
    if not status or not job.get("last_run_at"):
        return None, None
    if status == "ok":
        return "ok", None
    if status == "delivery_failed":
        return "delivery_failed", _one_line(job.get("last_delivery_error"), MAX_ERROR_CHARS)
    reason = job.get("last_error") or ("The run was stopped" if status == "cancelled" else None)
    return "error", _one_line(reason, MAX_ERROR_CHARS)


def project(job: dict[str, Any]) -> dict[str, Any]:
    """One job record (``list_jobs``/``get_job`` shape) as a task."""
    state = job.get("state")
    if job.get("live_execution"):
        state = "running"
    elif state not in ("scheduled", "paused", "completed", "error"):
        state = "scheduled" if job.get("enabled", True) else "paused"
    last_status, last_error = _last(job)
    prompt = job.get("prompt") if isinstance(job.get("prompt"), str) else ""
    task: dict[str, Any] = {
        "id": job["id"],
        "name": job.get("name") or job["id"],
        "prompt": prompt[:MAX_PROMPT_CHARS],
        "schedule": _schedule(job),
        "state": state,
        "next_run_at": job.get("next_run_at") if state in ("scheduled", "running") else None,
        "last_run_at": job.get("last_run_at") if last_status else None,
        "last_status": last_status,
        "last_error": last_error,
        "deliver": _deliver(job),
    }
    repeat = job.get("repeat") if isinstance(job.get("repeat"), dict) else {}
    times = repeat.get("times")
    if isinstance(times, int) and times > 0 and task["schedule"]["kind"] != "once":
        task["repeat"] = {"times": times, "done": int(repeat.get("completed") or 0)}
    return task


def _order(task: dict[str, Any]) -> tuple:
    from datetime import datetime
    upcoming = task["next_run_at"]
    try:
        instant = datetime.fromisoformat(upcoming).timestamp() if upcoming else 0.0
    except (TypeError, ValueError):
        instant, upcoming = 0.0, None
    return (upcoming is None, instant, task["name"].casefold(), task["id"])


# ── the store, off the shared executor ────────────────────────────────────────────────────────

def list_tasks() -> list[dict[str, Any]]:
    from cron.jobs import list_jobs
    return sorted((project(job) for job in list_jobs(include_disabled=True)), key=_order)


def _job(job_id: str) -> dict[str, Any]:
    from cron.executions import latest_executions, live_executions
    from cron.jobs import get_job
    job = get_job(job_id) if isinstance(job_id, str) and job_id else None
    if job is None:
        raise ScheduleError("no task with that id", unknown=True)
    job["latest_execution"] = latest_executions([job_id]).get(job_id)
    job["live_execution"] = live_executions([job_id]).get(job_id)
    return job


def task(job_id: str) -> dict[str, Any]:
    return project(_job(job_id))


def pause(job_id: str) -> dict[str, Any]:
    from cron.jobs import pause_job
    current = task(job_id)
    if current["state"] == "paused":
        return current
    if current["state"] == "completed":
        raise ScheduleError("a finished task cannot be paused")
    _store_call(pause_job, job_id, "paused from Aleph")
    return task(job_id)


def resume(job_id: str) -> dict[str, Any]:
    from cron.jobs import resume_job
    current = task(job_id)
    if current["state"] in ("scheduled", "running"):
        return current
    if current["state"] == "completed":
        raise ScheduleError("a finished task cannot be resumed")
    _store_call(resume_job, job_id)
    return task(job_id)


def delete(job_id: str) -> None:
    from cron.jobs import remove_job
    task(job_id)
    if not _store_call(remove_job, job_id):
        raise ScheduleError("no task with that id", unknown=True)


def run(job_id: str, start: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    """Fire the task once now, as the agent's ``cronjob(action="run")`` does: an extra fire that
    leaves the schedule, the repeat count and the state alone. ``start`` runs the claimed job in the
    background. A task already running is returned as it is; a paused or finished one is refused,
    since Hermes fires only a runnable job (resuming it first is the person's call)."""
    current = task(job_id)
    if current["state"] == "running":
        return current
    if current["state"] == "paused":
        raise ScheduleError("a paused task cannot run; resume it first")
    if current["state"] == "completed":
        raise ScheduleError("a finished task cannot run")
    from tools.cronjob_tools import _claim_for_manual_run
    claimed, refusal = _claim_for_manual_run(job_id, "Aleph run")
    if refusal is not None:
        refreshed = task(job_id)
        if refreshed["state"] == "running":
            return refreshed
        raise ScheduleError(str(refusal.get("error") or "the task could not run"))
    start(claimed)
    return {**task(job_id), "state": "running"}


def _store_call(fn, *args):
    try:
        return fn(*args)
    except ValueError as exc:  # Hermes's refusals (a one-shot whose time passed, a terminal job)
        raise ScheduleError(str(exc)) from exc


def run_claimed(job: dict[str, Any]) -> None:
    """The background body of ``run``: Hermes's shared manual-run body (delivery on the gateway's
    loop, in-flight dedupe, the run recorded on the job)."""
    from tools.cronjob_tools import _run_claimed_job
    try:
        result = _run_claimed_job(job)
        if not result.get("success"):
            logger.info("Aleph: scheduled task %s ran from the phone and failed: %s",
                        job.get("id"), result.get("error"))
    except Exception:
        logger.exception("Aleph: scheduled task %s could not run", job.get("id"))


# ── the watch ─────────────────────────────────────────────────────────────────────────────────

def _stamp() -> tuple:
    """What changes whenever any process writes the job store or the executions ledger."""
    from cron.jobs import _current_cron_store
    from hermes_constants import get_hermes_home
    store = _current_cron_store()
    ledger = Path(get_hermes_home()).resolve() / "cron" / "executions.db"
    stamps = []
    for path in (store.jobs_file, ledger, ledger.with_name(ledger.name + "-wal")):
        try:
            info = os.stat(path)
            stamps.append((info.st_mtime_ns, info.st_size))
        except OSError:
            stamps.append(None)
    return tuple(stamps)


def fingerprint(tasks: list[dict[str, Any]]) -> str:
    return hashlib.sha256(json.dumps(tasks, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class Schedule:
    """The adapter's handle on the tasks: calls on its own executor, and a watch that calls
    ``changed`` once per settled change (a burst of writes inside one poll is one change)."""

    def __init__(self, changed: Callable[[], None]) -> None:
        self._changed = changed
        self._executor: Optional[ThreadPoolExecutor] = None
        self._watch: Optional[asyncio.Task] = None
        self._seen: Optional[str] = None

    def start(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="aleph-schedule")
        self._watch = asyncio.get_running_loop().create_task(self._watch_store())

    async def stop(self) -> None:
        watch, self._watch = self._watch, None
        if watch is not None:
            watch.cancel()
        executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

    async def _off_loop(self, fn, *args):
        if self._executor is None:
            raise ScheduleError("scheduled tasks are not available")
        return await asyncio.get_running_loop().run_in_executor(self._executor, fn, *args)

    async def call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "schedule.list":
            return {"tasks": await self._off_loop(list_tasks)}
        job_id = params.get("id")
        if not isinstance(job_id, str) or not job_id.strip():
            raise ScheduleError("id must be a task's id", unknown=True)
        if method == "schedule.delete":
            await self._off_loop(delete, job_id)
            result: dict[str, Any] = {}
        elif method == "schedule.run":
            result = {"task": await self._off_loop(run, job_id, self._start_run)}
        else:
            action = {"schedule.pause": pause, "schedule.resume": resume}[method]
            result = {"task": await self._off_loop(action, job_id)}
        await self.check()
        return result

    @staticmethod
    def _start_run(job: dict[str, Any]) -> None:
        threading.Thread(target=run_claimed, args=(job,), name=f"aleph-cron-{job.get('id')}",
                         daemon=True).start()

    async def check(self) -> None:
        """Re-read the tasks and report a change since the last read."""
        try:
            current = fingerprint(await self._off_loop(list_tasks))
        except Exception:
            logger.debug("Aleph: could not read the scheduled tasks", exc_info=True)
            return
        if self._seen is not None and current != self._seen:
            self._changed()
        self._seen = current

    async def _watch_store(self) -> None:
        stamp: Any = _UNREAD
        loop = asyncio.get_running_loop()
        last_read = 0.0
        while True:
            try:
                now_stamp = await self._off_loop(_stamp)
            except Exception:
                now_stamp = None
            if now_stamp != stamp or loop.time() - last_read >= RECHECK_SECONDS:
                stamp, last_read = now_stamp, loop.time()
                await self.check()
            await asyncio.sleep(POLL_SECONDS)

