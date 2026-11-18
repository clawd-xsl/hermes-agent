"""A scripted coding-agent backend for the projects tests: what it reports to ``backends.list`` is set
by the test, and its sessions record what they are given and emit what the test tells them to."""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import Any, Callable, Optional

from gateway.config import PlatformConfig

from plugins.platforms.aleph.projects.backend import Accepted, BackendInfo, Event, Exit, Model, SessionSpec, Sink

@contextlib.asynccontextmanager
async def project_adapter(tmp_path):
    """A connected adapter with a projects root holding ``web``, a paired device (``adapter.device``,
    ``adapter.keys``), a ``claude`` backend that runs and a ``codex`` one that is not installed."""
    from plugins.platforms.aleph.adapter import AlephAdapter
    from tests.gateway.aleph_phone import pair_device
    root = tmp_path / "projects"
    (root / "web").mkdir(parents=True)
    adapter = AlephAdapter(PlatformConfig(enabled=True, extra={
        "host": "127.0.0.1", "port": 0, "state_dir": str(tmp_path / "aleph"), "projects_root": str(root)}))

    async def handler(event):
        pass
    adapter.set_message_handler(handler)
    adapter.project_backends.update(claude=FakeBackend(available()),
                                    codex=FakeBackend(unavailable("not_installed", "codex is not on PATH")))
    assert await adapter.connect()
    adapter.root = adapter.projects.root
    adapter.device, adapter.keys = pair_device(adapter)
    try:
        yield adapter
    finally:
        await adapter.disconnect()


async def until(condition: Callable[[], Any], timeout: float = 3.0) -> Any:
    """Wait for what the sessions do on their own tasks: ``condition``'s first truthy value."""
    deadline = time.monotonic() + timeout
    while not (value := condition()):
        assert time.monotonic() < deadline, "timed out"
        await asyncio.sleep(0.005)
    return value


MODELS = (Model("opus", "Opus", ("low", "medium", "high"), "high"), Model("sonnet", "Sonnet", (), None))


def available(backend_id: str = "claude", name: str = "Claude Code", **fields: Any) -> BackendInfo:
    """A backend that can run, with ``MODELS`` and the ``ask`` and ``auto_review`` modes."""
    return BackendInfo(**{"id": backend_id, "name": name, "available": True, "models": MODELS,
                          "default_model": "opus", "approvals": ("ask", "auto_review"), **fields})


def unavailable(reason: str, message: str, backend_id: str = "codex", name: str = "Codex") -> BackendInfo:
    return BackendInfo(backend_id, name, False, reason, message)


class FakeSession:
    """One thread's session: ``sent`` records each input, accepted into a turn at once unless
    ``accepting`` is off; ``emit`` plays an event to the runtime."""

    def __init__(self, spec: SessionSpec, sink: Sink) -> None:
        self.spec, self.sink = spec, sink
        self.sent: list[str] = []
        self.keys: list[str] = []
        self.accepting = True
        self.answers: list[tuple[str, Any]] = []
        self.interrupted = 0
        self.closed = False
        self.idle = True
        self._session_id: Optional[str] = spec.resume or f"session-{spec.thread}"

    @property
    def session_id(self) -> Optional[str]:
        return self._session_id

    def emit(self, event: Event) -> None:
        self.sink(event)

    async def send(self, key: str, text: str) -> None:
        self.sent.append(text)
        self.keys.append(key)
        if self.accepting:
            self.sink(Accepted(key))

    async def interrupt(self) -> None:
        self.interrupted += 1

    async def answer(self, ask: str, decision: Any) -> None:
        self.answers.append((ask, decision))

    async def apply(self, spec: SessionSpec) -> None:
        self.spec = spec

    async def close(self) -> None:
        self.closed = True
        self.sink(Exit())

    def idle_ok(self) -> bool:
        return self.idle


class FakeBackend:
    """``info`` is what ``probe`` answers (an exception is raised); ``probes`` counts the asks. ``fail``,
    when set, is raised by the next ``open``."""

    def __init__(self, info: BackendInfo | Exception) -> None:
        self.info = info
        self.asks_to_leave_plan = True  # off: plays a backend that only ends a plan-mode turn with its plan
        self.id = info.id if isinstance(info, BackendInfo) else "broken"
        self.name = info.name if isinstance(info, BackendInfo) else "Broken"
        self.probes = 0
        self.sessions: list[FakeSession] = []
        self.fail: Optional[Exception] = None

    async def probe(self) -> BackendInfo:
        self.probes += 1
        if isinstance(self.info, Exception):
            raise self.info
        return self.info

    async def open(self, spec: SessionSpec, sink: Sink) -> FakeSession:
        if self.fail is not None:
            failure, self.fail = self.fail, None
            raise failure
        session = FakeSession(spec, sink)
        self.sessions.append(session)
        return session

    async def stop(self) -> None:
        pass
