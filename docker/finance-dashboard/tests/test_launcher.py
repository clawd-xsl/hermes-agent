"""Finance overlay bootstrap contracts, without Docker or bank API calls."""
import importlib.util
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("host,port", [(None, None), ("0.0.0.0", "8767")])
def test_launch_uses_runtime_home_and_bounded_logs(tmp_path, monkeypatch, host, port):
    path = Path(__file__).resolve().parents[1] / "rootfs/opt/hermes/docker/finance-dashboard-launch.py"
    spec = importlib.util.spec_from_file_location("finance_launch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []
    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace(run=lambda *a, **k: calls.append((a, k))))
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    for key, value in (("DASHBOARD_HOST", host), ("DASHBOARD_PORT", port)):
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)
    root = logging.getLogger()
    old_handlers, old_level = root.handlers[:], root.level
    # basicConfig(force=True) closes existing handlers; detach test-owned ones first.
    root.handlers = []
    try:
        module.main()
        assert calls == [(("app:app",), {"host": host or "127.0.0.1", "port": int(port or "8766"), "log_config": None})]
        handler = root.handlers[0]
        assert handler.maxBytes == 2_000_000
        assert handler.backupCount == 3
        logging.getLogger("dashboard_v2.data").warning("diagnostic marker")
        handler.flush()
        assert "diagnostic marker" in (tmp_path / "logs/finance-dashboard.log").read_text()
    finally:
        for handler in root.handlers:
            handler.close()
        root.handlers = old_handlers
        root.setLevel(old_level)
