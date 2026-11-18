"""The processes the projects run (the backends' CLIs, git): each in its own process group, so its
children (a tool's commands, MCP servers) end with it."""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal

STREAM_LIMIT = 32 * 1024 * 1024  # one line of a backend's stdout: a Write or a diff comes whole in one
CLOSE_SECONDS = 5.0  # SIGTERM, then SIGKILL
# Pushing and cloning private repositories is the agents' own work: the git token is the one secret
# they keep.
_GIT_TOKENS = ("GITHUB_TOKEN", "GH_TOKEN")


def environment(home: str, **extra: str) -> dict[str, str]:
    """A backend's environment: the gateway's own with Hermes's secrets scrubbed (bots, providers,
    remote compute), the git token kept, and ``HOME`` the home the CLIs are signed in under."""
    from tools.environments.local import hermes_subprocess_env

    env = hermes_subprocess_env()
    env.update({key: os.environ[key] for key in _GIT_TOKENS if os.environ.get(key)})
    return {**env, "HOME": home, **extra}


def signal_group(proc: asyncio.subprocess.Process, signum: int) -> None:
    """Signal the process's group (``start_new_session``). One whose last member is exiting answers
    EPERM on macOS, not ESRCH."""
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(proc.pid, signum)


async def kill(proc: asyncio.subprocess.Process) -> None:
    signal_group(proc, signal.SIGKILL)
    await proc.wait()


async def end(proc: asyncio.subprocess.Process) -> None:
    """SIGTERM to the process's group, SIGKILL after ``CLOSE_SECONDS``."""
    if proc.returncode is not None:
        return
    if proc.stdin is not None:
        proc.stdin.close()
    signal_group(proc, signal.SIGTERM)
    try:
        await asyncio.wait_for(proc.wait(), CLOSE_SECONDS)
    except TimeoutError:
        await kill(proc)
