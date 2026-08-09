"""Persistent ``signal-ts`` subprocess transport for the Signal adapter.

The Node process owns the Signal socket and all libsignal state.  Python keeps
only a small newline-delimited JSON control plane so gateway message handling
never depends on signal-cli's HTTP/SSE + JSON-RPC bridge.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
import shutil
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

_MAX_PROTOCOL_LINE_BYTES = 1024 * 1024


class SignalTsError(RuntimeError):
    """Base error raised by the persistent Signal runtime."""


class SignalTsProcessError(SignalTsError):
    """The sidecar process exited or violated its protocol."""


class SignalTsCallError(SignalTsError):
    """A signal-ts operation failed in the sidecar."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.details = details or {}


def expand_runtime_path(value: str) -> Path:
    """Expand a configured runtime path without resolving missing parents."""
    return Path(os.path.expandvars(os.path.expanduser(value))).resolve()


def resolve_node_executable(command: str) -> str | None:
    """Resolve the configured scalar Node executable."""
    raw = str(command or "").strip()
    if not raw:
        return None
    # This is deliberately a scalar executable, not a shell command.  Rejecting
    # flags keeps startup deterministic and avoids an accidental shell surface.
    if len(shlex.split(raw)) != 1:
        return None
    expanded = os.path.expandvars(os.path.expanduser(raw))
    if os.path.isabs(expanded):
        return expanded if os.path.isfile(expanded) and os.access(expanded, os.X_OK) else None
    return shutil.which(expanded)


class SignalTsSidecar:
    """One persistent signal-ts process and its multiplexed JSONL requests."""

    def __init__(
        self,
        *,
        node_executable: str,
        sdk_path: Path,
        state_path: Path,
        cache_dir: Path,
        expected_account: str | None,
        on_envelope: Callable[[dict[str, Any]], Awaitable[None]],
        startup_timeout: float = 30.0,
        call_timeout: float = 30.0,
    ) -> None:
        self.node_executable = node_executable
        self.sdk_path = sdk_path
        self.state_path = state_path
        self.cache_dir = cache_dir
        self.expected_account = expected_account
        self.on_envelope = on_envelope
        self.startup_timeout = startup_timeout
        self.call_timeout = call_timeout

        self.process: asyncio.subprocess.Process | None = None
        self.account: str | None = None
        self.aci: str | None = None
        self.last_transport_activity_at: float = 0.0

        self._ready: asyncio.Future[None] | None = None
        self._closed: asyncio.Future[None] | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._write_lock = asyncio.Lock()
        self._pending: dict[str, asyncio.Future[Any]] = {}
        self._event_tasks: set[asyncio.Task[None]] = set()
        self._closing = False
        self._stderr_tail: list[str] = []

    @property
    def running(self) -> bool:
        return bool(self.process and self.process.returncode is None and not self._closing)

    async def start(self) -> None:
        if self.running:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        loop = asyncio.get_running_loop()
        self._ready = loop.create_future()
        self._closed = loop.create_future()
        self._closing = False

        sidecar_script = Path(__file__).with_name("signal_ts_sidecar.mjs")
        args = [
            self.node_executable,
            str(sidecar_script),
            "--sdk-path",
            str(self.sdk_path),
            "--state-path",
            str(self.state_path),
            "--cache-dir",
            str(self.cache_dir),
        ]
        if self.expected_account:
            args.extend(["--expected-account", self.expected_account])

        try:
            self.process = await asyncio.create_subprocess_exec(
                *args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=_MAX_PROTOCOL_LINE_BYTES,
            )
        except OSError as exc:
            raise SignalTsProcessError(f"Could not start signal-ts runtime: {exc}") from exc

        self._reader_task = asyncio.create_task(self._read_stdout(), name="signal-ts-stdout")
        self._stderr_task = asyncio.create_task(self._read_stderr(), name="signal-ts-stderr")
        try:
            await asyncio.wait_for(asyncio.shield(self._ready), timeout=self.startup_timeout)
        except Exception:
            await self.close(force=True)
            detail = f": {self._stderr_tail[-1]}" if self._stderr_tail else ""
            raise SignalTsProcessError(f"signal-ts runtime did not become ready{detail}")

    async def call(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> Any:
        if not self.running or not self.process or not self.process.stdin:
            raise SignalTsProcessError("signal-ts runtime is not connected")
        request_id = uuid.uuid4().hex
        future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        payload = json.dumps(
            {"id": request_id, "method": method, "params": params or {}},
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8") + b"\n"
        try:
            async with self._write_lock:
                if not self.running or not self.process.stdin:
                    raise SignalTsProcessError("signal-ts runtime stopped before request write")
                self.process.stdin.write(payload)
                await self.process.stdin.drain()
            return await asyncio.wait_for(
                asyncio.shield(future),
                timeout=self.call_timeout if timeout is None else timeout,
            )
        except asyncio.TimeoutError as exc:
            raise SignalTsCallError(f"signal-ts {method} timed out") from exc
        finally:
            self._pending.pop(request_id, None)

    async def wait_closed(self) -> None:
        if self._closed:
            await asyncio.shield(self._closed)

    async def close(self, *, force: bool = False) -> None:
        if self._closing:
            if self._closed:
                await asyncio.shield(self._closed)
            return
        process = self.process
        if process and process.returncode is None and not force:
            try:
                await self.call("shutdown", timeout=2.0)
            except Exception:
                pass
        self._closing = True
        if process and process.returncode is None:
            try:
                await asyncio.wait_for(process.wait(), timeout=3.0)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()

        current = asyncio.current_task()
        for task in (self._reader_task, self._stderr_task):
            if task and task is not current and not task.done():
                task.cancel()
        await asyncio.gather(
            *(
                task
                for task in (self._reader_task, self._stderr_task)
                if task and task is not current
            ),
            return_exceptions=True,
        )
        if self._event_tasks:
            await asyncio.gather(*tuple(self._event_tasks), return_exceptions=True)
        self.process = None
        self._fail_pending(SignalTsProcessError("signal-ts runtime closed"))
        if self._closed and not self._closed.done():
            self._closed.set_result(None)

    async def _read_stdout(self) -> None:
        assert self.process and self.process.stdout
        failure: Exception | None = None
        try:
            while True:
                line = await self.process.stdout.readline()
                if not line:
                    break
                try:
                    record = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    raise SignalTsProcessError("signal-ts emitted invalid JSON") from exc
                if not isinstance(record, dict):
                    raise SignalTsProcessError("signal-ts emitted a non-object record")
                self._handle_record(record)
            returncode = await self.process.wait()
            if not self._closing:
                detail = f": {self._stderr_tail[-1]}" if self._stderr_tail else ""
                failure = SignalTsProcessError(
                    f"signal-ts runtime exited with status {returncode}{detail}"
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            failure = exc
        finally:
            if failure:
                self._fail_pending(failure)
                if self._ready and not self._ready.done():
                    self._ready.set_exception(failure)
            if self._closed and not self._closed.done():
                self._closed.set_result(None)

    async def _read_stderr(self) -> None:
        assert self.process and self.process.stderr
        try:
            while True:
                line = await self.process.stderr.readline()
                if not line:
                    return
                message = line.decode("utf-8", "replace").rstrip()
                if not message:
                    continue
                self._stderr_tail.append(message[-1000:])
                del self._stderr_tail[:-20]
                logger.info("signal-ts: %s", message)
        except asyncio.CancelledError:
            raise

    def _handle_record(self, record: dict[str, Any]) -> None:
        request_id = record.get("id")
        if isinstance(request_id, str):
            future = self._pending.get(request_id)
            if not future or future.done():
                return
            if record.get("ok") is True:
                future.set_result(record.get("result"))
            else:
                error = record.get("error")
                details = error if isinstance(error, dict) else {}
                message = str(details.get("message") or error or "unknown signal-ts error")
                future.set_exception(SignalTsCallError(message, details=details))
            return

        event = record.get("event")
        if event == "ready":
            self.account = str(record.get("account") or "") or None
            self.aci = str(record.get("aci") or "") or None
            self.last_transport_activity_at = float(record.get("timestamp") or 0) / 1000.0
            if self._ready and not self._ready.done():
                self._ready.set_result(None)
            return
        if event == "transport":
            self.last_transport_activity_at = float(record.get("timestamp") or 0) / 1000.0
            return
        if event == "envelope" and isinstance(record.get("envelope"), dict):
            task = asyncio.create_task(self.on_envelope(record["envelope"]))
            self._event_tasks.add(task)
            task.add_done_callback(self._event_done)
            return
        if event == "log":
            logger.info("signal-ts: %s", record.get("message", ""))

    def _event_done(self, task: asyncio.Task[None]) -> None:
        self._event_tasks.discard(task)
        if task.cancelled():
            return
        error = task.exception()
        if error:
            logger.error(
                "Signal: failed to handle signal-ts envelope",
                exc_info=(type(error), error, error.__traceback__),
            )

    def _fail_pending(self, error: Exception) -> None:
        for future in tuple(self._pending.values()):
            if not future.done():
                future.set_exception(error)


__all__ = [
    "SignalTsCallError",
    "SignalTsError",
    "SignalTsProcessError",
    "SignalTsSidecar",
    "expand_runtime_path",
    "resolve_node_executable",
]
