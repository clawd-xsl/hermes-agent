"""Cron ``session: main``: a fire becomes a real turn in the gateway home conversation.

The scheduler keeps every pre-agent gate (monitor, wake-gate script, empty payload, injection
scan) and builds the prompt as usual (``_prepare_job_prompt``); instead of constructing an
isolated cron agent it commits that prompt to the home conversation's FIFO
(``gateway.main_session``). Acceptance is the scheduler-side completion point: the live gateway
turn owns model execution, streaming, persistence and delivery, so the fire's own answer is the
silence marker and the scheduler delivers nothing itself.

Agent fires always execute on the session owner (``run_canonical_job`` → ``gateway.session_cron``),
so the gateway runner and its loop are in this process here.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_ACCEPT_TIMEOUT_SECONDS = 30
#: Run-side job flag: this fire was committed to the main conversation.
MAIN_FIRE_FLAG = "_main_session_fire"


def _user_config() -> dict:
    """config.yaml the way the fire's own config load reads it (``_load_cron_job_config``): never
    creating or refusing on a profile home. A corrupt file reads as empty here; the fire's
    ``require_parseable_user_config`` gate refuses it right after."""
    from cron.scheduler import _get_hermes_home
    from hermes_cli.config_backend import config_exists
    from hermes_cli.config_effective import load_user_config_effective
    path = _get_hermes_home() / "config.yaml"
    try:
        return load_user_config_effective(path) if config_exists(str(path)) else {}
    except Exception:
        logger.warning("cron.session: config.yaml unreadable; the fire's config gate decides", exc_info=True)
        return {}


def cron_session_mode(job: dict, cfg: Optional[dict] = None) -> str:
    """``main`` or ``isolated`` for one fire: per-job ``session``, then ``cron.session`` (default
    isolated). A ``no_agent`` job never creates an agent turn, so a global ``main`` default does not
    change its script-only semantics (an explicit ``no_agent`` + ``main`` job is refused at create).
    Raises ValueError for a hand-edited unknown mode."""
    from gateway.main_session import normalize_session_mode
    if job.get("no_agent"):
        return "isolated"
    per_job = job.get("session")
    if per_job is not None and str(per_job).strip():
        return normalize_session_mode(per_job)
    cron_cfg = (_user_config() if cfg is None else cfg).get("cron")
    return normalize_session_mode(cron_cfg.get("session") if isinstance(cron_cfg, dict) else None)


def prepare_fire(job: dict, job_id: str, job_name: str, extra_prompt, cancel_event, prepare):
    """``run_job``'s ``(early_result, prompt)``: ``prepare`` is ``_prepare_job_prompt`` (every pre-agent
    gate) told which conversation the fire runs in. A main fire's result is its commit to the home
    conversation, so ``run_job`` builds no agent; an unknown hand-edited mode fails the fire."""
    from hermes_cli.config_backend import ConfigBackendUnavailable
    try:
        main = cron_session_mode(job) == "main"
    except (ValueError, ConfigBackendUnavailable) as exc:
        # An unknown hand-edited mode, or the config plane being down (a SystemExit on purpose,
        # never served as defaults): this fire fails, exactly like run_job's own config read.
        error = f"{type(exc).__name__}: {exc!s}"
        logger.error("Job '%s': %s", job_id, error)
        return (False, f"# Cron Job: {job_name} (FAILED)\n\n{error}\n", "", error), None
    early, prompt = prepare(job, job_id, job_name, extra_prompt, cancel_event, main)
    if early is None and main:
        early = run_main_session_fire(job, job_id, job_name, prompt)
        if early[0]:
            # Rides back to the firing scheduler (gateway.session_cron.RUN_JOB_FLAGS): the fire's own
            # cron session never held its turn, so bookkeeping drops it (scheduler_authority).
            job[MAIN_FIRE_FLAG] = True
    return early, prompt


def _doc(job_name: str, job_id: str, lines: list[str]) -> str:
    from cron.scheduler import _hermes_now
    head = [f"# Cron Job: {job_name}", "", f"**Job ID:** {job_id}",
            f"**Run Time:** {_hermes_now().strftime('%Y-%m-%d %H:%M:%S')}", "**Session:** main"]
    return "\n".join(head + lines) + "\n"


def _owner():
    """``(authority, loop)`` of the session owner running this fire, or None outside one."""
    from gateway.session_cron import current_execution, owner_for_home
    execution = current_execution()
    if execution is None:
        return None
    authority = execution[0]
    owner = owner_for_home(Path(authority.db.db_path).resolve().parent)
    return (authority, owner[1]) if owner is not None and owner[0] is authority else None


async def _admit(authority, job: dict, text: str):
    from gateway.main_session import CRON_PRODUCER, enqueue_main_session_turn, resolve_main_session_source
    from gateway.session_authorities import owner_scope
    runner = authority.runner
    origin = job.get("origin") if isinstance(job.get("origin"), dict) else None
    with owner_scope(authority):  # the loop task does not inherit the fire thread's profile scope
        profile = runner._served_profile_for_home(Path(authority.db.db_path).resolve().parent)
        source = await asyncio.to_thread(resolve_main_session_source, runner, origin=origin,
                                         profile=profile or (origin or {}).get("profile"))
        return await enqueue_main_session_turn(runner, source=source, text=text, producer=CRON_PRODUCER,
                                               metadata={"trigger": "cron", "job_id": str(job.get("id") or "")})


def run_main_session_fire(job: dict, job_id: str, job_name: str, prompt: str):
    """Commit the prepared prompt as a main-conversation turn; the ``run_job`` result tuple."""
    from cron.scheduler import SILENT_MARKER
    try:
        owner = _owner()
        if owner is None:
            raise RuntimeError("session: main requires the live gateway that owns this cron fire")
        authority, loop = owner
        future = asyncio.run_coroutine_threadsafe(
            _admit(authority, job, f"[Scheduled task: {job_name}]\n\n{prompt}"), loop)
        try:
            accepted = future.result(timeout=_ACCEPT_TIMEOUT_SECONDS)
        except FutureTimeout:
            future.cancel()
            raise RuntimeError(f"session: main turn was not accepted within {_ACCEPT_TIMEOUT_SECONDS}s") from None
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        logger.warning("Main-session cron job '%s' was not accepted: %s", job_id, error, exc_info=True)
        return False, _doc(job_name, job_id, ["", "## Error", "", f"```\n{error}\n```"]), "", error
    logger.info("Job '%s' accepted into the main conversation %s", job_id, accepted.session_key)
    doc = _doc(job_name, job_id, [
        f"**Target:** {accepted.platform}:{accepted.chat_id}", f"**Session Key:** `{accepted.session_key}`",
        f"**Status:** {'queued' if accepted.queued else 'started'}", "",
        "The scheduled instruction was accepted as a real turn in the main conversation. Its "
        "response is handled by the normal gateway delivery path."])
    # The silence marker: the main turn delivers its own answer; the scheduler must not.
    return True, doc, SILENT_MARKER, None
