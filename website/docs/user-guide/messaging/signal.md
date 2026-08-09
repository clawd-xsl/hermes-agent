---
title: "Signal"
description: "Run Hermes on Signal with one persistent signal-ts linked-device connection"
---

# Signal setup

Hermes uses [`clawd-xsl/signal-ts`](https://github.com/clawd-xsl/signal-ts)
directly. A single persistent Node sidecar owns the authenticated Signal socket,
libsignal session state, attachment transfers, and reconnects for the lifetime of
the gateway. There is no Java process, HTTP/SSE bridge, or per-message RPC daemon.

This integration is designed for one Signal account and one personal Hermes
agent. Voice and video calls are intentionally out of scope; voice-note and
other file attachments still work as normal messages.

## Requirements

- Node.js 22 or newer
- A built `clawd-xsl/signal-ts` checkout
- A durable signal-ts linked-device state file
- Hermes and signal-ts running as the same OS user, with exclusive access to
  that state file

Build the SDK once:

```bash
git clone git@github.com:clawd-xsl/signal-ts.git
cd signal-ts
pnpm install --frozen-lockfile
pnpm build
```

### Create or reuse linked-device state

For an OpenClaw migration, reuse the existing signal-ts state file; the
migration tool can point Hermes at it without re-linking the Signal account.

For a new linked device, the signal-ts checkout includes an opt-in QR link flow:

```bash
SIGNAL_TS_E2E=1 \
SIGNAL_TS_E2E_LINK_DEVICE=1 \
SIGNAL_TS_E2E_LINK_OUTPUT_FILE=/secure/path/hermes-signal-state.json \
pnpm test:e2e -- src/e2e/live-link-device.e2e.test.ts
```

Scan the printed `sgnl://linkdevice` URL in Signal under **Settings → Linked
devices**. Treat the resulting JSON file like a credential: keep it private,
back it up securely, and never commit it.

## Configure Hermes

Run:

```bash
hermes setup gateway
```

Choose **Signal**, then provide:

- the exact Node executable;
- the built signal-ts checkout directory;
- the durable linked-device state file;
- an optional expected E.164 account number, used to reject the wrong state;
- the DM sender allowlist and optional group allowlist;
- the home conversation for cron results and proactive notifications.

Signal settings are behavioral configuration and live in `~/.hermes/config.yaml`,
not `.env`. The setup wizard writes the equivalent of:

```yaml
platforms:
  signal:
    enabled: true
    home_channel:
      platform: signal
      chat_id: "+15551234567"
      name: "Signal home"
    extra:
      node_command: /usr/local/bin/node
      sdk_path: /opt/signal-ts
      state_path: /secure/hermes-signal-state.json
      account: "+15551234567"
      allow_from:
        - "+15551234567"
      group_allow_from: []
      require_mention: false
      ignore_stories: true
      reactions: true
```

An empty `allow_from` enables Hermes' normal DM pairing flow. A non-empty list
is enforced inside the Signal adapter before reactions or agent work. Groups
are disabled when `group_allow_from` is empty; use explicit group IDs or `"*"`.

Start or restart the gateway:

```bash
hermes gateway restart
```

Use `/sethome` from the desired Signal conversation if you want to change the
cron/notification destination later.

## Runtime behavior

- One persistent Signal socket and one durable state owner are reused for all
  messages, which avoids signal-cli RPC startup and transport latency.
- Incoming messages are decrypted and acknowledged in the Node sidecar. Agent
  turns run asynchronously, so a long response does not block typing,
  reactions, sends, or receipt processing.
- Direct messages, Note to Self, replies/quotes, edits received from other
  clients, reactions, typing indicators, formatted text, and attachments are
  normalized into Hermes' existing gateway message contract.
- If the socket exits, Hermes restarts the sidecar with bounded exponential
  backoff while retaining the same state file. Only one gateway process may own
  an account at a time; Hermes uses a scoped account lock to prevent duplicate
  listeners.
- Signal cannot edit an already-sent message, so Hermes uses its native
  no-edit streaming behavior and sends final chunks without fake edit calls.

## Troubleshooting

| Symptom | Check |
|---|---|
| Signal shows “partially configured” | Verify `state_path`, `sdk_path`, and `node_command` point to real local files. |
| `dist/index.js` is missing | Run `pnpm install --frozen-lockfile && pnpm build` in the signal-ts checkout. |
| State/account mismatch | Correct `extra.account`, or select the state file for that Signal number. |
| Gateway repeatedly reconnects | Inspect `~/.hermes/logs/gateway.log`; make sure no other process owns the same linked device/state file. |
| Unknown DMs receive pairing codes | Set `extra.allow_from` to the one account allowed to use this personal assistant. |
| Group messages are ignored | Add the Signal group ID to `extra.group_allow_from`; enable `require_mention` if desired. |

The dashboard shows Signal connection state but does not edit these paths as
environment variables. Use `hermes setup gateway` or edit `config.yaml`.
