---
sidebar_position: 6
title: "Signal"
description: "Run Hermes on Signal through one persistent signal-ts linked-device connection"
---

# Signal Setup

Hermes connects to Signal with [signal-ts](https://github.com/clawd-xsl/signal-ts), a TypeScript Signal
client built on `@signalapp/libsignal-client`. The gateway starts one persistent Node sidecar that acts as
a **linked device** of your Signal account: it owns the authenticated Signal socket, the libsignal session
state, attachment transfers and reconnects for the gateway's whole lifetime. There is no signal-cli daemon,
no Java process and no HTTP/SSE bridge.

The integration is built for one Signal account serving a personal Hermes agent. Voice and video calls are
out of scope; voice notes and other files work as normal attachments.

---

## Prerequisites

- **Node.js 22.19 or newer.** Hermes uses `node` from `PATH`, or the Node it manages itself, unless you set
  `node_command`.
- **A built signal-ts checkout** (the package is private, so it is not on npm).
- **A linked-device state file** created by signal-ts. Treat it like a password: anyone holding it can read
  and send messages as your linked device.
- Hermes and the state file on the same machine, owned by the user that runs the gateway. Only one process
  may use a state file at a time.

### 1. Build the SDK

```bash
git clone git@github.com:clawd-xsl/signal-ts.git
cd signal-ts
pnpm install --frozen-lockfile
pnpm build
```

Cloning it next to your Hermes checkout (`../signal-ts`) lets the setup wizard find it automatically.

### 2. Link a device

If you already have signal-ts state (for example from OpenClaw), reuse that file; you do not need to link
again.

To link a new device, run the SDK's opt-in QR link flow from the signal-ts checkout:

```bash
SIGNAL_TS_E2E=1 \
SIGNAL_TS_E2E_LINK_DEVICE=1 \
SIGNAL_TS_E2E_LINK_DEVICE_NAME="Hermes" \
SIGNAL_TS_E2E_LINK_OUTPUT_FILE="$HOME/.hermes/signal-ts/default.json" \
pnpm test:e2e -- src/e2e/live-link-device.e2e.test.ts
```

It prints an `sgnl://linkdevice` URL. Open Signal on your phone, go to **Settings → Linked devices → Link new
device**, and scan that URL as a QR code (any QR generator works). The state file is written when linking
finishes. Keep it private (`chmod 600`), back it up securely, and never commit it.

---

## Configure Hermes

```bash
hermes gateway setup
```

Choose **Signal** and answer the prompts:

| Prompt | What to enter |
|---|---|
| Node executable | A Node 22.19+ binary (defaults to `node` on `PATH`) |
| signal-ts SDK path | The built checkout (the directory holding `package.json` and `dist/index.js`) |
| Signal state file | The linked-device state file from step 2 |
| Expected Signal number | Optional E.164 number; startup refuses a state file for a different account |
| Home conversation | Optional chat for cron results and notifications (a number, ACI or `group:<id>`) |
| Allowed users | Numbers or ACIs allowed to talk to the agent; leave empty for pairing mode |
| Group messaging | Off by default; enter group IDs or `*` |

The wizard never pre-fills the home conversation or the allowlist with the linked account's own number.

Signal is configured **only in `config.yaml`**; nothing goes into `.env`. The wizard writes:

```yaml
platforms:
  signal:
    enabled: true
    home_channel:
      platform: signal
      chat_id: "+15550001111"
      name: "Signal home"
    extra:
      node_command: /usr/local/bin/node        # optional: default is node on PATH / Hermes' Node
      sdk_path: /home/me/signal-ts
      state_path: /home/me/.hermes/signal-ts/default.json
      account: "+15559990000"                   # optional: validated against the state file
      allow_from:                               # empty = pairing mode
        - "+15550001111"
      group_allow_from: []                      # empty = groups disabled; "*" = all groups
      require_mention: false                    # groups: only answer when @mentioned
      ignore_stories: true
      reactions: true                           # 👀 while working, then ✅ / ❌
      startup_timeout_seconds: 30
      call_timeout_seconds: 30
```

Then start or restart the gateway:

```bash
hermes gateway restart
```

`hermes status` and `hermes gateway setup` show Signal as configured once the state file and SDK path exist.

### Access control

- **`allow_from` set**: a strict allowlist enforced by the adapter before any reaction or agent work. It
  applies to group messages too: in an allowed group only listed senders are answered. `"*"` admits
  everyone.
- **`allow_from` empty**: pairing mode. An unknown DM receives Hermes' pairing code, which the operator
  approves with `hermes pairing approve signal CODE`. Unknown contacts never get a 👀 reaction.
- **`group_allow_from`**: group IDs the agent may answer in; empty disables groups.

Use `/sethome` in a Signal chat to make it the home conversation later.

### Sending from tools and cron

`send_message` and cron `deliver: signal` go through the running gateway's Signal connection, because a
second client would fight the gateway over the linked device's protocol state. Run scheduled jobs inside
the gateway (the default). A send from a separate process without a running gateway fails with an explicit
error.

---

## How it works

- The sidecar (`plugins/platforms/signal/sidecar/index.mjs`) talks newline-delimited JSON to the adapter
  over stdio. It decrypts and acknowledges inbound envelopes and normalizes them into Hermes' gateway
  message shape: DMs, Note to Self, groups, quotes/replies, edits, mentions and attachments.
- Inbound messages are handled one at a time in arrival order, on their own task, so typing indicators,
  reactions and sends never wait behind an inbound message.
- Markdown bold, italic, strikethrough and code become native Signal text styles; long replies are split at
  8000 characters.
- Attachments up to 100 MB, up to 32 per message, paced by a rate-limit scheduler that follows the server's
  retry hints.
- Ordinary sends fetch recipient prekeys over the account's own authenticated connection, so recipients who
  restrict sealed-sender access still receive messages.
- If the connection drops, the sidecar exits and Hermes starts a fresh one with backoff (1s, doubling to
  10s), reusing the same state file. A scoped lock per account (or per state file) stops a second gateway
  from using the same linked device.
- Signal cannot edit sent messages, so streaming replies are delivered without edit calls.

---

## Troubleshooting

| Symptom | Check |
|---|---|
| Signal shows "not configured" | `state_path` and `sdk_path` must point at existing files; an explicit `node_command` must be runnable. |
| `signal-ts SDK is not built` | Run `pnpm install --frozen-lockfile && pnpm build` in the checkout. |
| `state file does not exist` / `no linked account` | Link a device (step 2) or point `state_path` at the right file. |
| `does not match configured account` | Fix `extra.account`, or select the state file for that number. |
| `requires Node >= 22.19` | Install a newer Node or set `node_command` to one. |
| `rejected this linked device` | The device was unlinked on the phone; link it again. |
| Gateway keeps reconnecting | Check `~/.hermes/logs/gateway.log` and make sure no other process (OpenClaw, another gateway) uses the same state file. |
| Unknown DMs get pairing codes | Set `extra.allow_from`. |
| Group messages are ignored | Add the group ID to `extra.group_allow_from`; with `allow_from` set, the sender must be listed too. |

Configuration errors (missing state, unlinked device, account mismatch, old Node) stop the adapter without
retrying; network errors are retried.

## Migrating from signal-cli

The signal-cli adapter and its `SIGNAL_HTTP_URL`, `SIGNAL_ACCOUNT`, `SIGNAL_ALLOWED_USERS`,
`SIGNAL_GROUP_ALLOWED_USERS`, `SIGNAL_ALLOW_ALL_USERS`, `SIGNAL_REQUIRE_MENTION`, `SIGNAL_IGNORE_STORIES` and
`SIGNAL_REACTIONS` variables are gone; they are ignored if still present. Link a signal-ts device, run
`hermes gateway setup`, move your allowlists to `allow_from` / `group_allow_from`, and stop the signal-cli
daemon. `SIGNAL_HOME_CHANNEL` (written by `/sethome`) still works as a mirror of the configured home channel.
