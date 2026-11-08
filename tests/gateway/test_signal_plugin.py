"""Signal ships as a bundled platform plugin backed by the signal-ts sidecar: the registry entry
carries the requirements probe, config gate, live-adapter sender, chunking cap and setup wizard,
and declares no signal-cli env credentials or env allowlists."""

import sys
from types import SimpleNamespace

from gateway.config import PlatformConfig


def _register():
    from plugins.platforms.signal.register import register
    captured = {}
    register(SimpleNamespace(register_platform=lambda **kw: captured.update(kw)))
    return captured


def test_register_declares_the_signal_ts_wiring(tmp_path):
    from plugins.platforms.signal import adapter

    captured = _register()
    assert captured["name"] == "signal" and captured["label"] == "Signal" and captured["emoji"] == "📡"
    assert captured["adapter_factory"] is adapter.SignalAdapter
    assert captured["max_message_length"] == adapter.MAX_MESSAGE_LENGTH == 8000
    assert captured["cron_deliver_env_var"] == "SIGNAL_HOME_CHANNEL"
    assert captured["pii_safe"] is True and captured["allow_update_command"] is True
    assert callable(captured["standalone_sender_fn"]) and callable(captured["setup_fn"])
    # Access is config-owned: no env allowlist / allow-all / credential names.
    for key in ("allowed_users_env", "allow_all_env", "own_policy_env"):
        assert not captured.get(key)
    assert captured["required_env"] == [] and captured["dashboard_env_vars"] == ()
    assert captured["env_enable_credentials"] == ()
    assert "signal-ts" in captured["dashboard_description"]
    policy = captured["standalone_send"]
    assert policy.media and policy.text_direct and policy.empty_list_between_chunks

    validate = captured["validate_config"]
    assert validate(PlatformConfig(enabled=True)) is False
    assert validate(PlatformConfig(enabled=True, extra={"http_url": "http://127.0.0.1:1", "account": "+15550001"})) is False
    state = tmp_path / "state.json"
    state.write_text("{}")
    extra = {"state_path": str(state), "sdk_path": str(tmp_path), "node_command": sys.executable}
    assert validate(PlatformConfig(enabled=True, extra=extra)) is True
    assert captured["config_connected_fn"](PlatformConfig(enabled=True, extra=extra)) is True


def test_is_connected_reads_the_profile_config_yaml(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    from plugins.platforms.signal.setup import write_signal_yaml_config

    is_connected = _register()["is_connected"]
    assert is_connected(PlatformConfig(enabled=True)) is False
    state = tmp_path / "state.json"
    state.write_text("{}")
    write_signal_yaml_config({"state_path": str(state), "sdk_path": str(tmp_path)})
    assert is_connected(PlatformConfig(enabled=True)) is True


def test_sidecar_files_ship_with_the_package():
    import tomllib
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    data = tomllib.loads((root / "pyproject.toml").read_text())["tool"]["setuptools"]["package-data"]["plugins"]
    assert "platforms/signal/sidecar/*.mjs" in data
    sidecar = root / "plugins" / "platforms" / "signal" / "sidecar"
    assert (sidecar / "index.mjs").is_file() and (sidecar / "prekeys.mjs").is_file()
