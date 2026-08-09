"""Signal direct-runtime setup and status contracts."""

from __future__ import annotations

import sys
from pathlib import Path


def _make_runtime_files(root: Path) -> tuple[Path, Path]:
    sdk = root / "signal-ts"
    (sdk / "dist").mkdir(parents=True)
    (sdk / "package.json").write_text(
        '{"name":"@openclaw/signal-ts","type":"module","main":"dist/index.js"}',
        encoding="utf-8",
    )
    (sdk / "dist" / "index.js").write_text("export {};\n", encoding="utf-8")
    state = root / "signal-state.json"
    state.write_text("{}\n", encoding="utf-8")
    return sdk, state


def test_signal_runtime_writer_round_trips_config_and_status(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    import hermes_cli.gateway as gateway_cli
    from gateway.config import Platform, load_gateway_config
    from hermes_cli.config import read_raw_config

    sdk, state = _make_runtime_files(tmp_path)
    gateway_cli._write_signal_runtime_config(
        {
            "node_command": sys.executable,
            "sdk_path": str(sdk),
            "state_path": str(state),
            "account": "+15551234567",
            "allow_from": ["+15551234567"],
            "group_allow_from": [],
        },
        home_channel="+15551234567",
    )

    raw = read_raw_config()
    signal = raw["platforms"]["signal"]
    assert signal["enabled"] is True
    assert signal["extra"]["sdk_path"] == str(sdk)
    assert signal["home_channel"]["chat_id"] == "+15551234567"

    loaded = load_gateway_config()
    assert loaded._is_platform_connected(
        Platform.SIGNAL, loaded.platforms[Platform.SIGNAL]
    )
    assert gateway_cli._platform_status({"key": "signal"}) == "configured"


def test_signal_status_is_partial_when_files_are_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    import hermes_cli.gateway as gateway_cli

    gateway_cli._write_signal_runtime_config(
        {
            "node_command": sys.executable,
            "sdk_path": str(tmp_path / "missing-sdk"),
            "state_path": str(tmp_path / "missing-state.json"),
        }
    )

    assert gateway_cli._platform_status({"key": "signal"}) == "partially configured"


def test_signal_setup_persists_config_without_signal_env(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    import hermes_cli.gateway as gateway_cli
    from hermes_cli.config import load_env, read_raw_config

    sdk, state = _make_runtime_files(tmp_path)
    answers = {
        "Node executable": sys.executable,
        "signal-ts SDK path": str(sdk),
        "Signal state file": str(state),
        "Expected Signal number": "+15551234567",
        "Home conversation": "+15551234567",
        "Allowed users": "+15551234567",
    }

    def fake_prompt(question, *args, **kwargs):
        for prefix, answer in answers.items():
            if prefix in question:
                return answer
        raise AssertionError(f"unexpected prompt: {question}")

    monkeypatch.setattr(gateway_cli, "prompt", fake_prompt)
    monkeypatch.setattr(gateway_cli, "prompt_yes_no", lambda *args, **kwargs: False)

    gateway_cli._setup_signal()

    signal = read_raw_config()["platforms"]["signal"]
    assert signal["extra"]["account"] == "+15551234567"
    assert signal["extra"]["allow_from"] == ["+15551234567"]
    assert signal["extra"]["state_path"] == str(state.resolve())
    assert signal["extra"]["sdk_path"] == str(sdk.resolve())
    assert signal["home_channel"]["chat_id"] == "+15551234567"
    assert not any(key.startswith("SIGNAL_") for key in load_env())


def test_dashboard_signal_card_has_no_obsolete_env_bridge():
    from hermes_cli.web_server import _build_catalog_entry

    entry = _build_catalog_entry("signal")
    assert entry["env_vars"] == ()
    assert entry["required_env"] == ()
    assert "signal-ts" in entry["description"]
    assert "REST bridge" not in entry["description"]


def test_stale_signal_cli_env_does_not_configure_signal(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setenv("SIGNAL_HTTP_URL", "http://127.0.0.1:8080")
    monkeypatch.setenv("SIGNAL_ACCOUNT", "+15551234567")
    from gateway.config import GatewayConfig, Platform, _apply_env_overrides

    config = GatewayConfig()
    _apply_env_overrides(config)
    assert Platform.SIGNAL not in config.platforms
