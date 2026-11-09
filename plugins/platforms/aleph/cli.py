"""``hermes aleph ...``: pair the Aleph app, list paired devices, revoke one.

These run in their own process and work on the same state as the gateway (``$HERMES_HOME/aleph``):
a pairing secret written here is used by the gateway's ``/aleph/v1/pair`` handshake, and a revoked
device's open connections are closed by the gateway within a couple of seconds.
"""

from __future__ import annotations

import argparse
import io
import socket
import sys
import time
from datetime import datetime
from typing import Any, Optional
from urllib.parse import quote


def register_cli(parser: argparse.ArgumentParser) -> None:
    subs = parser.add_subparsers(dest="aleph_command", required=False)
    p_pair = subs.add_parser("pair", help="Show the QR code that pairs the Aleph app")
    p_pair.add_argument("--server", default=None,
                        help="Base URL the phone uses, pairing included (default: ALEPH_PUBLIC_URL, else "
                             "http://<LAN address>:<port>)")
    p_pair.add_argument("--no-wait", action="store_true",
                        help="Print the link and exit instead of waiting for the phone to pair")
    subs.add_parser("devices", help="List paired devices")
    p_revoke = subs.add_parser("revoke", help="Revoke a paired device")
    p_revoke.add_argument("device_id", help="The device's id, from `hermes aleph devices`")
    parser.set_defaults(func=dispatch)


def dispatch(args: argparse.Namespace) -> int:
    command = getattr(args, "aleph_command", None) or "devices"
    handler = {"pair": _cmd_pair, "devices": _cmd_devices, "revoke": _cmd_revoke}.get(command)
    if handler is None:
        print(f"unknown subcommand: {command}", file=sys.stderr)
        return 2
    return handler(args)


# ── settings ───────────────────────────────────────────────────────────────────────────────────

def _extra() -> dict[str, Any]:
    """``platforms.aleph.extra`` from config.yaml (env vars still win, via ``setting``)."""
    try:
        from hermes_cli.config import load_config
        platforms = (load_config() or {}).get("platforms") or {}
        extra = (platforms.get("aleph") or {}).get("extra") or {}
        return extra if isinstance(extra, dict) else {}
    except Exception:
        return {}


def _state_dir(extra: dict[str, Any]):
    from plugins.platforms.aleph.adapter import resolve_state_dir
    return resolve_state_dir(extra)


def _open_store(extra: dict[str, Any]):
    from plugins.platforms.aleph.store import AlephStore
    return AlephStore(_state_dir(extra))


def _listen_address(extra: dict[str, Any]) -> tuple[str, int]:
    from plugins.platforms.aleph.adapter import DEFAULT_HOST, DEFAULT_PORT, setting
    host = setting(extra, "host", "ALEPH_HOST", DEFAULT_HOST) or DEFAULT_HOST
    try:
        port = int(setting(extra, "port", "ALEPH_PORT", DEFAULT_PORT) or DEFAULT_PORT)
    except ValueError:
        port = DEFAULT_PORT
    return host, port


def server_url(extra: dict[str, Any], override: Optional[str] = None) -> str:
    """The base URL the phone pairs at and uses from then on: ``ALEPH_PUBLIC_URL`` (the tunnel), else
    the listener's address on the home network."""
    from plugins.platforms.aleph.adapter import setting
    public = (override or setting(extra, "public_url", "ALEPH_PUBLIC_URL")).strip().rstrip("/")
    if public:
        return public
    host, port = _listen_address(extra)
    if host in ("0.0.0.0", "::", ""):
        host = _lan_address() or "127.0.0.1"
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"http://{host}:{port}"


def pair_link(*, server: str, key: bytes, secret: bytes) -> str:
    from plugins.platforms.aleph.noise import b64url
    return f"aleph://pair?server={quote(server, safe='')}&key={b64url(key)}&secret={b64url(secret)}"


def _lan_address() -> Optional[str]:
    """This machine's address on its default route (no packet is sent)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("192.0.2.1", 9))
            return sock.getsockname()[0]
    except OSError:
        return None


def _listening(host: str, port: int) -> bool:
    target = "127.0.0.1" if host in ("0.0.0.0", "") else "::1" if host == "::" else host
    try:
        with socket.create_connection((target, port), timeout=0.5):
            return True
    except OSError:
        return False


def _qr(text: str) -> str:
    try:
        import qrcode  # type: ignore[import-untyped]
    except ImportError:
        return ""
    qr = qrcode.QRCode(version=None, error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=1, border=2)
    qr.add_data(text)
    qr.make(fit=True)
    buf = io.StringIO()
    qr.print_ascii(out=buf, invert=True)
    return buf.getvalue()


def _when(ts: Optional[float]) -> str:
    if not ts:
        return "never"
    return datetime.fromtimestamp(ts).astimezone().strftime("%Y-%m-%d %H:%M")


def _clock(ts: float) -> str:
    return datetime.fromtimestamp(ts).astimezone().strftime("%H:%M")


# ── commands ───────────────────────────────────────────────────────────────────────────────────

def _cmd_pair(args: argparse.Namespace) -> int:
    from plugins.platforms.aleph.noise import KeyPair
    from plugins.platforms.aleph.store import load_or_create_server_key
    extra = _extra()
    store = _open_store(extra)
    try:
        server_key = KeyPair.from_private(load_or_create_server_key(_state_dir(extra)))
        server = server_url(extra, args.server)
        before = {device.id for device in store.devices()}
        pending = store.pending_pairing()
        if pending is not None:
            # One pairing at a time: a phone still pairing with the old link will be refused.
            print(f"\n  Warning: this replaces the pending pairing (valid until {_clock(pending[1])}); "
                  "its link stops working now.")
        secret, expires_at = store.new_pairing()
        link = pair_link(server=server, key=server_key.public, secret=secret)
        print()
        qr = _qr(link)
        if qr:
            print(qr)
        else:
            print("  (No QR code: the qrcode package is not installed. Paste the link into Aleph instead.)\n")
        print(f"  Server: {server}")
        print(f"  Link:   {link}")
        print(f"  Valid until {_clock(expires_at)} (10 minutes), for one device. Anyone who sees this code "
              "can pair, so show it only to the phone.\n")
        host, port = _listen_address(extra)
        if host in ("127.0.0.1", "::1", "localhost") and server.startswith("http://") \
                and "127.0.0.1" not in server and "localhost" not in server:
            print(f"  Warning: the listener binds {host}, so {server} cannot reach it. Set ALEPH_PUBLIC_URL to "
                  "the tunnel's address, or ALEPH_HOST=0.0.0.0 to pair over the home network.\n")
        if not _listening(host, port):
            print(f"  Warning: nothing is listening on {host}:{port}. Start the gateway (`hermes gateway run`) "
                  "with Aleph enabled before the phone pairs.\n")
        if args.no_wait:
            return 0
        print("  Waiting for the phone… (Ctrl-C stops waiting; the link stays valid)")
        try:
            while True:
                fresh = [device for device in store.devices() if device.id not in before]
                if fresh:
                    device = fresh[-1]
                    print(f"\n  Paired {device.name}" + (f" ({device.model})" if device.model else "")
                          + f" as {device.id}.")
                    return 0
                if store.pending_pairing() is None:
                    print("\n  The link expired or was replaced. Run `hermes aleph pair` again.")
                    return 1
                time.sleep(1.0)
        except KeyboardInterrupt:
            print()
            return 0
    finally:
        store.close()


def _cmd_devices(args: argparse.Namespace) -> int:
    from plugins.platforms.aleph.store import fingerprint
    store = _open_store(_extra())
    try:
        devices = store.devices()
    finally:
        store.close()
    if not devices:
        print("No paired devices. Pair one with `hermes aleph pair`.")
        return 0
    rows = [("ID", "NAME", "MODEL", "KEY", "PAIRED", "LAST SEEN")]
    for d in devices:
        rows.append((d.id, d.name, d.model or "-", fingerprint(d.public_key), _when(d.created_at),
                     _when(d.last_seen_at)))
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    for row in rows:
        print("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip())
    return 0


def _cmd_revoke(args: argparse.Namespace) -> int:
    store = _open_store(_extra())
    try:
        device = store.device(args.device_id)
        if device is None or not store.revoke_device(args.device_id):
            print(f"No paired device {args.device_id}. See `hermes aleph devices`.", file=sys.stderr)
            return 1
    finally:
        store.close()
    print(f"Revoked {device.name} ({device.id}). Its key no longer connects, and a "
          "running gateway closes its connections within a few seconds.")
    return 0
