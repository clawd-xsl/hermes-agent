"""A log record formatted while the vault registry is locked must not deadlock.

GC finalizers run at arbitrary allocations. One that logs (aiohttp's "Unclosed client session")
inside ``redact_registered_vault_values`` re-enters it on the same thread through
``RedactingFormatter``; with a plain Lock the thread waited on itself and froze the gateway.
"""

import io
import logging
import threading

import pytest

from agent import redact


class _LoggingBucket(dict):
    """A registry bucket whose iteration logs once, standing in for a finalizer that logs."""

    def __init__(self, logger, *args):
        super().__init__(*args)
        self._logger = logger
        self._fired = False

    def __iter__(self):
        if not self._fired:
            self._fired = True
            self._logger.error("Unclosed client session hunter2")
        return super().__iter__()


@pytest.fixture
def scoped_registry(monkeypatch):
    monkeypatch.setattr(redact, "_vault_scope", lambda: "profile-a")
    monkeypatch.setattr(redact, "_VAULT_REDACTION_VALUES", {})


def _run_with_timeout(fn, timeout=5.0):
    result = {}

    def target():
        result["value"] = fn()

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout)
    assert not thread.is_alive(), "vault redaction deadlocked on re-entry"
    return result["value"]


def test_logging_inside_the_vault_lock_does_not_deadlock(scoped_registry):
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(redact.RedactingFormatter("%(message)s"))
    logger = logging.getLogger("test.redact.vault_reentry")
    logger.propagate = False
    logger.addHandler(handler)
    try:
        redact._VAULT_REDACTION_VALUES["profile-a"] = _LoggingBucket(logger, {"hunter2": None})
        out = _run_with_timeout(lambda: redact.redact_registered_vault_values("pw is hunter2"))
    finally:
        logger.removeHandler(handler)

    assert out == "pw is «redacted-vault-secret»"
    # The nested record was formatted, and scrubbed, by the same registry.
    assert "hunter2" not in stream.getvalue()
    assert "«redacted-vault-secret»" in stream.getvalue()


def test_register_and_clear_compute_the_scope_outside_the_lock(monkeypatch, scoped_registry):
    held = []

    def scope():
        # A scope lookup that logs or allocates must not run with the lock held.
        held.append(redact._VAULT_REDACTION_LOCK._is_owned())
        return "profile-a"

    monkeypatch.setattr(redact, "_vault_scope", scope)
    redact.register_vault_redaction_value("s3cret-value")
    assert redact.redact_registered_vault_values("x s3cret-value y") == "x «redacted-vault-secret» y"
    redact.clear_vault_redaction_values()
    assert redact.redact_registered_vault_values("s3cret-value") == "s3cret-value"
    assert held and not any(held)
