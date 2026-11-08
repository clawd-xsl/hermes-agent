"""Interactive ``hermes gateway setup`` wizard for Signal's signal-ts runtime, plus the config.yaml
reader/writer it shares with status checks. Helpers resolve late through the ``hermes_cli.gateway``
facade so the plugin stays importable outside the CLI and the seams tests patch there keep
intercepting. Everything is written to ``platforms.signal`` in config.yaml; nothing goes to .env."""

import shutil
from pathlib import Path


def _gw():
    from hermes_cli import gateway  # late: CLI-only dependency
    return gateway


def read_signal_yaml_config() -> dict:
    """``platforms.signal`` from the profile's raw config.yaml (``gateway.platforms`` first, the
    top-level ``platforms`` section wins), flattened: ``extra`` keys plus ``home_channel``."""
    from hermes_cli.config import read_raw_config
    raw = read_raw_config()
    merged: dict = {}
    gateway_section = raw.get("gateway")
    sources = [gateway_section.get("platforms") if isinstance(gateway_section, dict) else None,
               raw.get("platforms")]
    for platforms in sources:
        signal_config = platforms.get("signal") if isinstance(platforms, dict) else None
        if not isinstance(signal_config, dict):
            continue
        merged.update({key: value for key, value in signal_config.items() if key != "extra"})
        if isinstance(extra := signal_config.get("extra"), dict):
            merged.update(extra)
    return merged


def write_signal_yaml_config(values: dict, *, home_channel: str | None = None) -> None:
    """Enable Signal and merge ``values`` into ``platforms.signal.extra`` (plus an optional home)."""
    from hermes_cli.config import read_raw_config, save_config
    config = read_raw_config()
    platforms = config.get("platforms")
    if not isinstance(platforms, dict):
        platforms = config["platforms"] = {}
    signal_config = platforms.get("signal")
    if not isinstance(signal_config, dict):
        signal_config = platforms["signal"] = {}
    signal_config["enabled"] = True
    extra = signal_config.get("extra")
    if not isinstance(extra, dict):
        extra = signal_config["extra"] = {}
    extra.update(values)
    if home_channel:
        signal_config["home_channel"] = {"platform": "signal", "chat_id": home_channel, "name": "Signal home"}
    save_config(config)


def _csv(value) -> str:
    return ",".join(str(v) for v in value) if isinstance(value, (list, tuple)) else str(value or "")


def _split_csv(text: str) -> list[str]:
    return [value.strip() for value in (text or "").split(",") if value.strip()]


def _prompt_runtime(existing: dict):
    """Node executable, SDK checkout and state file; None (after an error line) when invalid."""
    from plugins.platforms.signal.signal_ts import resolve_node_executable
    gw = _gw()
    default_node = str(existing.get("node_command") or shutil.which("node") or "node")
    node_command = gw.prompt("  Node executable (>= 22.19)", default=default_node).strip() or default_node
    resolved_node = resolve_node_executable(node_command)
    if not resolved_node:
        gw.print_error(f"  Node executable is not runnable: {node_command}")
        return None

    sibling_sdk = Path(gw.PROJECT_ROOT).parent / "signal-ts"
    default_sdk = str(existing.get("sdk_path") or (sibling_sdk if sibling_sdk.exists() else ""))
    print()
    gw.print_info("  Path to the signal-ts SDK checkout (the directory with package.json).")
    sdk_path = Path(gw.prompt("  signal-ts SDK path", default=default_sdk).strip()).expanduser()
    if not (sdk_path / "package.json").is_file() or not (sdk_path / "dist" / "index.js").is_file():
        gw.print_error("  SDK path must contain package.json and a built dist/index.js.")
        gw.print_info("  Build it with: pnpm install --frozen-lockfile && pnpm build")
        return None

    default_state = str(existing.get("state_path") or (gw.get_hermes_home() / "signal-ts" / "default.json"))
    print()
    gw.print_info("  Path to the linked-device state file created by signal-ts when you linked it.")
    state_path = Path(gw.prompt("  Signal state file", default=default_state).strip()).expanduser()
    if not state_path.is_file():
        gw.print_error(f"  Signal state file does not exist: {state_path}")
        gw.print_info("  Link this device with signal-ts first (see the Signal guide).")
        return None
    return resolved_node, sdk_path.resolve(), state_path.resolve()


def _prompt_groups(existing: dict) -> list[str]:
    gw = _gw()
    existing_groups = existing.get("group_allow_from") or []
    print()
    if not gw.prompt_yes_no("  Enable group messaging? (disabled by default)", bool(existing_groups)):
        return []
    print()
    gw.print_info("  Enter group IDs to allow, or * for all groups.")
    return _split_csv(gw.prompt("  Allowed group IDs", default=_csv(existing_groups) or "*"))


def interactive_setup() -> None:
    """``hermes gateway setup`` flow for Signal (registered as the plugin's ``setup_fn``)."""
    gw = _gw()
    gw.print_setup_header("📡 Signal")
    existing = read_signal_yaml_config()
    if existing.get("state_path") and existing.get("sdk_path"):
        print()
        gw.print_success("Signal is already configured.")
        if not gw.prompt_yes_no("  Reconfigure Signal?", False):
            return

    gw._print_info_lines(
        "  Hermes runs signal-ts as a linked device inside the gateway (no signal-cli daemon or Java).",
        "  The SDK checkout must be built once: pnpm install --frozen-lockfile && pnpm build",
    )
    runtime = _prompt_runtime(existing)
    if runtime is None:
        return
    node_command, sdk_path, state_path = runtime

    account = gw.prompt("  Expected Signal number (E.164, optional)",
                        default=str(existing.get("account") or "")).strip()
    existing_home = existing.get("home_channel")
    existing_home_id = str(existing_home.get("chat_id") or "") if isinstance(existing_home, dict) else ""
    # Never default the home or the allowlist to the bot's own number: that would deliver cron
    # output to Note to Self and silently authorize only the bot itself.
    home_channel = gw.prompt("  Home conversation for cron/notifications (optional)",
                             default=existing_home_id).strip()
    print()
    gw.print_info("  Enter phone numbers or ACIs allowed to use this assistant (empty = pairing mode).")
    allowed = _split_csv(gw.prompt("  Allowed users (comma-separated)", default=_csv(existing.get("allow_from"))))
    if not allowed:
        gw.print_warning("  No allowlist saved; unknown DMs will use Hermes pairing mode.")
    groups = _prompt_groups(existing)

    values = {"node_command": node_command, "sdk_path": str(sdk_path), "state_path": str(state_path),
              "allow_from": allowed, "group_allow_from": groups}
    if account:
        values["account"] = account
    write_signal_yaml_config(values, home_channel=home_channel or None)

    print()
    gw.print_success("Signal configured!")
    gw._print_info_lines(
        f"  Runtime: {node_command} + {sdk_path}", f"  State: {state_path}",
        f"  Account: {account or 'read from the state file at startup'}",
        f"  DM auth: {'allowlist' if allowed else 'pairing'}", f"  Groups: {'enabled' if groups else 'disabled'}",
        f"  Home: {home_channel or 'set later with /sethome'}",
    )
