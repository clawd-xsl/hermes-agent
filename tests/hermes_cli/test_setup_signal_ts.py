"""Signal signal-ts setup wizard, status and dashboard contracts."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


def _make_runtime_files(root: Path) -> tuple[Path, Path]:
    sdk = root / "signal-ts"
    (sdk / "dist").mkdir(parents=True)
    (sdk / "package.json").write_text('{"name":"@openclaw/signal-ts","type":"module","main":"dist/index.js"}',
                                      encoding="utf-8")
    (sdk / "dist" / "index.js").write_text("export {};\n", encoding="utf-8")
    state = root / "signal-state.json"
    state.write_text("{}\n", encoding="utf-8")
    return sdk, state


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _signal_entry():
    from gateway.platform_registry import platform_registry
    from hermes_cli.plugins import discover_plugins
    discover_plugins()
    return platform_registry.get("signal")


def _run_wizard(monkeypatch, answers):
    import hermes_cli.gateway as gateway_cli
    from plugins.platforms.signal.setup import interactive_setup

    def fake_prompt(question, *args, **kwargs):
        for prefix, answer in answers.items():
            if prefix in question:
                return answer(kwargs) if callable(answer) else answer
        raise AssertionError(f"unexpected prompt: {question}")

    monkeypatch.setattr(gateway_cli, "prompt", fake_prompt)
    monkeypatch.setattr(gateway_cli, "prompt_yes_no", lambda *args, **kwargs: False)
    interactive_setup()


def test_writer_round_trips_config_and_status(tmp_path, hermes_home):
    from gateway.config import Platform, load_gateway_config
    from hermes_cli.config import read_raw_config
    from hermes_cli.gateway_setup_wizard import _platform_status
    from plugins.platforms.signal.setup import write_signal_yaml_config

    sdk, state = _make_runtime_files(tmp_path)
    write_signal_yaml_config(
        {"node_command": sys.executable, "sdk_path": str(sdk), "state_path": str(state),
         "account": "+15551234567", "allow_from": ["+15550000001"], "group_allow_from": []},
        home_channel="+15550000001")

    signal = read_raw_config()["platforms"]["signal"]
    assert signal["enabled"] is True
    assert signal["extra"]["sdk_path"] == str(sdk)
    assert signal["home_channel"]["chat_id"] == "+15550000001"

    loaded = load_gateway_config()
    assert loaded._is_platform_connected(Platform.SIGNAL, loaded.platforms[Platform.SIGNAL])
    assert Platform.SIGNAL in loaded.get_connected_platforms()
    assert loaded.platforms[Platform.SIGNAL].home_channel.chat_id == "+15550000001"
    entry = _signal_entry()
    assert _platform_status({"key": "signal", "_registry_entry": entry}) == "configured"


def test_status_is_not_configured_when_files_are_missing(tmp_path, hermes_home):
    from gateway.config import Platform, load_gateway_config
    from hermes_cli.gateway_setup_wizard import _platform_status
    from plugins.platforms.signal.setup import write_signal_yaml_config

    write_signal_yaml_config({"node_command": sys.executable, "sdk_path": str(tmp_path / "missing-sdk"),
                              "state_path": str(tmp_path / "missing-state.json")})
    assert _platform_status({"key": "signal", "_registry_entry": _signal_entry()}) == "not configured"
    assert Platform.SIGNAL not in load_gateway_config().get_connected_platforms()


def test_setup_persists_config_without_signal_env(tmp_path, hermes_home, monkeypatch):
    from hermes_cli.config import load_env, read_raw_config

    sdk, state = _make_runtime_files(tmp_path)
    _run_wizard(monkeypatch, {
        "Node executable": sys.executable, "signal-ts SDK path": str(sdk), "Signal state file": str(state),
        "Expected Signal number": "+15551234567", "Home conversation": "+15550000001",
        "Allowed users": "+15550000001, +15550000002",
    })

    signal = read_raw_config()["platforms"]["signal"]
    assert signal["extra"]["account"] == "+15551234567"
    assert signal["extra"]["allow_from"] == ["+15550000001", "+15550000002"]
    assert signal["extra"]["group_allow_from"] == []
    assert signal["extra"]["state_path"] == str(state.resolve())
    assert signal["extra"]["sdk_path"] == str(sdk.resolve())
    assert signal["home_channel"]["chat_id"] == "+15550000001"
    assert not any(key.startswith("SIGNAL_") for key in load_env())


def test_setup_does_not_default_bot_account_as_home_or_allowed_user(tmp_path, hermes_home, monkeypatch):
    from hermes_cli.config import read_raw_config

    sdk, state = _make_runtime_files(tmp_path)
    _run_wizard(monkeypatch, {
        "Node executable": sys.executable, "signal-ts SDK path": str(sdk), "Signal state file": str(state),
        "Expected Signal number": "+15551234567",
        # Accept whatever default the wizard offers.
        "Home conversation": lambda kw: kw.get("default", ""), "Allowed users": lambda kw: kw.get("default", ""),
    })

    signal = read_raw_config()["platforms"]["signal"]
    assert signal["extra"]["account"] == "+15551234567"
    assert signal["extra"]["allow_from"] == []
    assert "home_channel" not in signal


def test_setup_refuses_a_missing_state_file(tmp_path, hermes_home, monkeypatch):
    from hermes_cli.config import read_raw_config

    sdk, _ = _make_runtime_files(tmp_path)
    _run_wizard(monkeypatch, {"Node executable": sys.executable, "signal-ts SDK path": str(sdk),
                              "Signal state file": str(tmp_path / "nope.json")})
    assert "signal" not in (read_raw_config().get("platforms") or {})


def test_setup_refuses_an_unbuilt_sdk(tmp_path, hermes_home, monkeypatch):
    from hermes_cli.config import read_raw_config

    (tmp_path / "raw-sdk").mkdir()
    (tmp_path / "raw-sdk" / "package.json").write_text("{}")
    _run_wizard(monkeypatch, {"Node executable": sys.executable, "signal-ts SDK path": str(tmp_path / "raw-sdk")})
    assert "signal" not in (read_raw_config().get("platforms") or {})


def test_dashboard_signal_card_has_no_obsolete_env_bridge():
    from hermes_cli.web_server_messaging import _build_catalog_entry

    entry = _build_catalog_entry("signal", _signal_entry())
    assert entry["env_vars"] == ()
    assert entry["required_env"] == ()
    assert "signal-ts" in entry["description"]
    assert "REST bridge" not in entry["description"]


def test_stale_signal_cli_env_does_not_configure_signal(hermes_home, monkeypatch):
    monkeypatch.setenv("SIGNAL_HTTP_URL", "http://127.0.0.1:8080")
    monkeypatch.setenv("SIGNAL_ACCOUNT", "+15551234567")
    from gateway.config import GatewayConfig, Platform, _apply_env_overrides

    config = GatewayConfig()
    _apply_env_overrides(config)
    assert Platform.SIGNAL not in config.platforms


def test_dump_reports_config_only_signal(tmp_path, hermes_home):
    from hermes_cli.dump import _config_only_platforms
    from plugins.platforms.signal.setup import write_signal_yaml_config

    assert _config_only_platforms() == []
    sdk, state = _make_runtime_files(tmp_path)
    write_signal_yaml_config({"sdk_path": str(sdk), "state_path": str(state)})
    assert _config_only_platforms() == ["signal"]
