"""Execution-mode invariants every cron job definition must satisfy.

``create_job`` and ``update_job`` (``cron/jobs.py``) both run these, so an update cannot
bypass a rule the create door enforces.
"""
from typing import Any, Optional

from hermes_constants import display_hermes_home

NO_AGENT_WITHOUT_SCRIPT_ERROR = (
    "no_agent=True requires a script — with no agent and no script "
    "there is nothing for the job to run."
)

MAIN_SESSION_NO_AGENT_ERROR = (
    "session='main' runs an agent turn in the gateway home conversation and cannot be "
    "combined with no_agent=True."
)

SCRIPT_IS_COMMAND_LINE_ERROR = (
    "script must name a file under {scripts_dir} (e.g. 'watchdog.sh'), not a shell command line: "
    "{script!r} has no such file. Write the command into a script there and pass its filename."
)


def _validate_job_mode_invariants(
    monitor_script: Optional[str],
    monitor_url: Optional[str],
    no_agent: bool,
    script: Optional[str],
    session: Optional[str] = None,
) -> None:
    """Execution-mode invariants shared by create_job and update_job (no bypass via the update
    door)."""
    if no_agent and session == "main":
        raise ValueError(MAIN_SESSION_NO_AGENT_ERROR)
    if monitor_script and monitor_url:
        raise ValueError(
            "monitor_script and monitor_url are mutually exclusive — a job "
            "can only have one monitor source.")
    if (monitor_script or monitor_url) and no_agent:
        raise ValueError(
            "monitor_script/monitor_url cannot be combined with no_agent=True — "
            "the whole point of a monitor job is to suppress or wake the AGENT "
            "based on source changes. Use a plain no_agent script job instead.")
    if no_agent and not script:
        raise ValueError(NO_AGENT_WITHOUT_SCRIPT_ERROR)
    if script and _script_is_command_line(script):
        raise ValueError(SCRIPT_IS_COMMAND_LINE_ERROR.format(
            scripts_dir=display_hermes_home() + "/scripts/", script=script))


def normalize_job_session(value: Any) -> Optional[str]:
    """Per-job ``session``: ``main`` | ``isolated``; blank/None = inherit ``cron.session``."""
    if value is None or str(value).strip() == "":
        return None
    from gateway.main_session import normalize_session_mode
    return normalize_session_mode(value)


def _normalize_catch_up(value: Any) -> Optional[bool]:
    """``catch_up``: True runs one catch-up for missed slots, False skips them, None = default."""
    if value is None or isinstance(value, bool):
        return value
    raise ValueError("catch_up must be true, false, or null (use the default).")


def validate_schedule_knobs(
    schedule: dict, timezone: Optional[str], grace_minutes: Optional[int],
) -> None:
    """Per-job schedule knobs (``timezone``/``grace_minutes``) are validated at BOTH doors —
    ``create_job`` and ``update_job`` — so an update cannot store a schedule knob a create
    would reject.

    Raises ValueError for an unknown IANA zone, and for ``grace_minutes`` on a schedule that has
    no wall clock to be late for (``once``/``interval``): the catch-up window only means
    something to a recurring cron cadence, so accepting it silently would suggest an effect it
    cannot have.
    """
    from cron.jobs_time import normalize_grace_minutes, normalize_timezone

    normalize_timezone(timezone)
    minutes = normalize_grace_minutes(grace_minutes)
    if minutes is None:
        return
    kind = (schedule or {}).get("kind")
    if kind != "cron":
        raise ValueError(
            f"grace_minutes only applies to cron schedules (a fixed wall clock that can be "
            f"missed); this job's schedule kind is {kind!r}.")


def _script_is_command_line(script: str) -> bool:
    """A ``script`` with whitespace that resolves to no file is a command line (``echo hi``),
    not a path; run every tick it would deliver "Script not found" as the payload. A plain
    missing filename stays creatable (``hermes cron doctor`` reports it)."""
    if not any(ch.isspace() for ch in script.strip()):
        return False
    from cron.lifecycle_guard import _resolve_script_path
    path = _resolve_script_path(script)
    return path is None or not path.is_file()
