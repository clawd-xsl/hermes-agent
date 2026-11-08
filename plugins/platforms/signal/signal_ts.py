"""Persistent ``signal-ts`` sidecar for the Signal adapter.

One Node process (``sidecar/index.mjs``) embeds the signal-ts SDK, owns the single authenticated
Signal socket and every libsignal state mutation for the lifetime of the gateway. Python speaks
newline-delimited JSON to it over stdio: requests ``{"id", "method", "params"}`` get
``{"id", "ok", "result" | "error"}`` replies; unsolicited records are events (``ready``,
``transport``, ``envelope``, ``fatal``, ``log``). stdout is protocol-only; diagnostics go to stderr.

Envelope events are handed to a synchronous callback: the adapter queues them and handles them one
at a time on its own task, so a slow inbound handler can never stall command replies on stdout.
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
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

SIDECAR_SCRIPT = Path(__file__).resolve().parent / "sidecar" / "index.mjs"
_MAX_PROTOCOL_LINE_BYTES = 1024 * 1024
# Sidecar exit status for configuration errors (EX_CONFIG): restarting cannot fix them.
SIDECAR_CONFIG_EXIT = 78


class SignalTsError(RuntimeError):
    """Base error raised by the persistent Signal runtime."""


class SignalTsProcessError(SignalTsError):
    """The sidecar process exited or violated its protocol."""


class SignalTsStartupError(SignalTsProcessError):
    """The sidecar never became ready. ``retryable`` is False for configuration errors
    (missing/unlinked state, account mismatch, missing SDK, Node too old)."""

    def __init__(self, message: str, *, code: str = "startup_failed", retryable: bool = True):
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class SignalTsCallError(SignalTsError):
    """A signal-ts operation failed inside the sidecar; ``details`` is its structured error."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.details = details or {}


def expand_runtime_path(value: str) -> Path:
    """Expand ``~`` / ``$VARS`` in a configured runtime path and make it absolute."""
    return Path(os.path.expandvars(os.path.expanduser(str(value or "").strip()))).resolve()


def resolve_node_executable(command: str | None) -> str | None:
    """Resolve a configured Node executable. It is a scalar executable, never a shell command:
    a value with arguments is rejected so startup stays deterministic and shell-free."""
    raw = str(command or "").strip()
    if not raw:
        return None
    try:
        parts = shlex.split(raw)
    except ValueError:
        return None
    if len(parts) != 1:
        return None
    expanded = os.path.expandvars(os.path.expanduser(raw))
    if os.path.isabs(expanded):
        return expanded if os.path.isfile(expanded) and os.access(expanded, os.X_OK) else None
    return shutil.which(expanded)


def default_node_executable() -> str | None:
    """Node when ``node_command`` is unset: ``node`` on PATH, else an already-installed
    Hermes-managed Node. Never installs anything (callers may ``pm.ensure("node")``)."""
    found = shutil.which("node")
    if found:
        return found
    try:
        import pm
        installed = pm.installed_package("node")
        binary = getattr(installed, "binary", None) if installed is not None else None
        return str(binary) if binary else None
    except Exception:
        return None


def runtime_settings(extra: dict | None) -> dict[str, Any]:
    """Normalized runtime keys from ``platforms.signal.extra`` (paths expanded, blanks kept blank)."""
    extra = extra if isinstance(extra, dict) else {}
    state = str(extra.get("state_path") or "").strip()
    sdk = str(extra.get("sdk_path") or "").strip()
    return {
        "state_path": expand_runtime_path(state) if state else None,
        "sdk_path": expand_runtime_path(sdk) if sdk else None,
        "node_command": str(extra.get("node_command") or "").strip(),
    }


def runtime_configured(extra: dict | None) -> bool:
    """Whether the direct runtime can start from this config: the state file exists, the SDK path
    exists, and an explicitly configured ``node_command`` resolves (an unset one falls back to
    PATH / the managed Node at connect time)."""
    settings = runtime_settings(extra)
    if settings["state_path"] is None or settings["sdk_path"] is None:
        return False
    if not settings["state_path"].is_file() or not settings["sdk_path"].exists():
        return False
    return not settings["node_command"] or resolve_node_executable(settings["node_command"]) is not None


class SignalTsSidecar:
    """One persistent signal-ts process and its multiplexed NDJSON requests."""

    def __init__(
        self,
        *,
        node_executable: str,
        sdk_path: Path,
        state_path: Path,
        cache_dir: Path,
        expected_account: str | None,
        on_envelope: Callable[[dict[str, Any]], None],
        startup_timeout: float = 30.0,
        call_timeout: float = 30.0,
        script_path: Path = SIDECAR_SCRIPT,
    ) -> None:
        self.node_executable = node_executable
        self.sdk_path = sdk_path
        self.state_path = state_path
        self.cache_dir = cache_dir
        self.expected_account = expected_account
        self.on_envelope = on_envelope
        self.startup_timeout = startup_timeout
        self.call_timeout = call_timeout
        self.script_path = script_path

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
        self._closing = False
        self._stderr_tail: list[str] = []
        self._fatal: Optional[dict[str, Any]] = None

    @property
    def running(self) -> bool:
        return bool(self.process and self.process.returncode is None and not self._closing)

    def _args(self) -> list[str]:
        args = [
            # The SDK's SQLite use triggers Node's ExperimentalWarning on every start.
            self.node_executable, "--disable-warning=ExperimentalWarning", str(self.script_path),
            "--sdk-path", str(self.sdk_path),
            "--state-path", str(self.state_path),
            "--cache-dir", str(self.cache_dir),
        ]
        if self.expected_account:
            args.extend(["--expected-account", self.expected_account])
        return args

    async def start(self) -> None:
        if self.running:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        loop = asyncio.get_running_loop()
        self._ready = loop.create_future()
        self._closed = loop.create_future()
        self._closing = False
        self._fatal = None
        try:
            self.process = await asyncio.create_subprocess_exec(
                *self._args(),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=_MAX_PROTOCOL_LINE_BYTES,
            )
        except OSError as exc:
            raise SignalTsStartupError(f"could not start signal-ts runtime: {exc}",
                                       code="spawn_failed", retryable=False) from exc

        self._reader_task = asyncio.create_task(self._read_stdout(), name="signal-ts-stdout")
        self._stderr_task = asyncio.create_task(self._read_stderr(), name="signal-ts-stderr")
        try:
            await asyncio.wait_for(asyncio.shield(self._ready), timeout=self.startup_timeout)
        except BaseException as exc:
            if self.process.returncode is not None and self._stderr_task is not None:
                # The process died on its own: let stderr drain so the error names the cause.
                await asyncio.wait({self._stderr_task}, timeout=1.0)
            await self.close(force=True)
            if isinstance(exc, asyncio.CancelledError):
                raise
            raise self._startup_error(exc) from exc

    def _startup_error(self, exc: BaseException) -> SignalTsStartupError:
        """Describe why startup failed: the sidecar's structured ``fatal`` record wins, then its
        last stderr line, then the timeout."""
        if self._fatal:
            code = str(self._fatal.get("code") or "startup_failed")
            return SignalTsStartupError(
                str(self._fatal.get("message") or code), code=code,
                retryable=bool(self._fatal.get("retryable", True)))
        if isinstance(exc, asyncio.TimeoutError):
            detail = f": {self._stderr_tail[-1]}" if self._stderr_tail else ""
            return SignalTsStartupError(
                f"signal-ts runtime did not become ready within {self.startup_timeout:.0f}s{detail}")
        detail = f": {self._stderr_tail[-1]}" if self._stderr_tail else ""
        return SignalTsStartupError(f"signal-ts runtime did not become ready ({exc}){detail}")

    async def call(self, method: str, params: dict[str, Any] | None = None, *,
                   timeout: float | None = None) -> Any:
        if not self.running or not self.process or not self.process.stdin:
            raise SignalTsProcessError("signal-ts runtime is not connected")
        request_id = uuid.uuid4().hex
        future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        payload = json.dumps(
            {"id": request_id, "method": method, "params": params or {}},
            ensure_ascii=False, separators=(",", ":"),
        ).encode("utf-8") + b"\n"
        try:
            async with self._write_lock:
                if not self.running or not self.process.stdin:
                    raise SignalTsProcessError("signal-ts runtime stopped before request write")
                self.process.stdin.write(payload)
                await self.process.stdin.drain()
            return await asyncio.wait_for(
                asyncio.shield(future), timeout=self.call_timeout if timeout is None else timeout)
        except asyncio.TimeoutError as exc:
            raise SignalTsCallError(f"signal-ts {method} timed out", details={"code": "timeout"}) from exc
        except (BrokenPipeError, ConnectionResetError) as exc:
            raise SignalTsProcessError(f"signal-ts runtime pipe closed during {method}") from exc
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
                await self.call("shutdown", timeout=5.0)
            except Exception:
                pass
        self._closing = True
        if process and process.returncode is None:
            if force:
                process.kill()
            try:
                await asyncio.wait_for(process.wait(), timeout=3.0)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()

        current = asyncio.current_task()
        tasks = [t for t in (self._reader_task, self._stderr_task) if t and t is not current]
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
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
                failure = SignalTsProcessError(f"signal-ts runtime exited with status {returncode}{detail}")
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
        while True:
            line = await self.process.stderr.readline()
            if not line:
                return
            message = line.decode("utf-8", "replace").rstrip()
            if not message:
                continue
            self._stderr_tail.append(message[-1000:])
            del self._stderr_tail[:-20]
            if message.startswith("[debug]"):
                logger.debug("signal-ts: %s", message)
            elif message.startswith(("[warn]", "[error]")):
                logger.warning("signal-ts: %s", message)
            else:
                logger.info("signal-ts: %s", message)

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
        elif event == "transport":
            self.last_transport_activity_at = float(record.get("timestamp") or 0) / 1000.0
        elif event == "envelope" and isinstance(record.get("envelope"), dict):
            try:
                self.on_envelope(record["envelope"])
            except Exception:
                logger.exception("Signal: failed to queue signal-ts envelope")
        elif event == "fatal":
            self._fatal = record
            logger.error("signal-ts: %s", record.get("message") or record.get("code"))
        elif event == "log":
            logger.info("signal-ts: %s", record.get("message", ""))

    def _fail_pending(self, error: Exception) -> None:
        for future in tuple(self._pending.values()):
            if not future.done():
                future.set_exception(error)


__all__ = [
    "SIDECAR_SCRIPT", "SignalTsCallError", "SignalTsError", "SignalTsProcessError", "SignalTsSidecar",
    "SignalTsStartupError", "default_node_executable", "expand_runtime_path", "resolve_node_executable",
    "runtime_configured", "runtime_settings",
]
