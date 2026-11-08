"""Run the sidecar's authenticated-prekey node tests (prekeys.test.mjs) under ``node --test``."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_authenticated_prekey_node_suite():
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not available")
    suite = Path(__file__).with_name("prekeys.test.mjs")
    proc = subprocess.run([node, "--test", str(suite)], capture_output=True, text=True, timeout=120, check=False)
    assert proc.returncode == 0, proc.stdout + proc.stderr
