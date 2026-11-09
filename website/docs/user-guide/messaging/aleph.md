# Aleph

Aleph is a personal iOS app that talks to Hermes as a messaging channel, the way Telegram or Signal do, but
directly: the plugin (`plugins/platforms/aleph/`) runs its own listener and the phone connects to it. There
is one conversation, chat `main`, and it is meant to be the gateway's **home channel**, so `session: main`
cron jobs and webhooks land in it.

What the channel gives the phone:

- a durable, replayable history: every message, edit, deletion, receipt and card is logged with a `seq`, and a
  phone that was offline catches up from the last `seq` it stored;
- streaming drafts of the reply, typing with a status line ("is running …"), and read receipts (two checks
  once a turn that includes the person's messages starts);
- tool progress as compact rows, gateway notices as system notes;
- approval cards for dangerous commands and question cards for the `clarify` tool;
- attachments both ways;
- end-to-end encryption: everything between the phone and the plugin, files included, is ciphertext to
  the network, the tunnel and Cloudflare;
- push notifications through APNs that carry no content: the app's notification extension fetches the
  real text over the channel itself.

The wire protocol is specified in the Aleph repository (`docs/protocol.md`).

## Set it up

1. Enable the channel in the profile's `.env` (`~/.hermes/.env`, or `$HERMES_HOME/.env`):

   ```bash
   ALEPH_ENABLED=true
   ALEPH_HOST=127.0.0.1                   # behind a tunnel (the default); 0.0.0.0 to use the home network
   ALEPH_PORT=8650                        # the default
   ALEPH_PUBLIC_URL=https://aleph.xsl.sh  # what the phone dials, pairing included; see "Reach it from anywhere"
   ```

2. Turn on streaming in `config.yaml` (drafts need the global switch; Aleph streams by default once it is on):

   ```yaml
   streaming:
     enabled: true
   ```

3. Start the gateway:

   ```bash
   hermes gateway run
   ```

   The log shows `Aleph: listening on ws://127.0.0.1:8650/aleph/v1/ws` and `✓ aleph connected`. The first
   start creates the server's key, `$HERMES_HOME/aleph/server.key` (mode 600).

4. Pair the phone (next section).

The paired devices are the conversation's one sender (user id `owner`), so no allowlist is needed. If you set
`GATEWAY_ALLOWED_USERS` for other platforms, add `owner` to it, or the gateway will refuse the phone's
messages.

## Encryption

Everything between the phone and the plugin is encrypted end to end with the
[Noise protocol](https://noiseprotocol.org/noise.html). TLS on the public name is only the transport;
Cloudflare and anything on the way see ciphertext, sizes and timing. The Hermes machine sees plaintext,
since the agent runs there.

- The server has one static X25519 key (`server.key`). Each device has its own, made on the phone at
  pairing; the server stores only the public half, which is what identifies the device. There are no
  tokens or passwords.
- Each connection to `/aleph/v1/ws` starts with a `Noise_IK_25519_ChaChaPoly_SHA256` handshake (prologue
  `aleph/1`). An unknown or revoked device key, or a handshake not finished within 10 seconds, closes with
  `4401`. After it, each binary WebSocket message is one JSON-RPC frame, encrypted with ChaCha20-Poly1305.
  The phone accepts only the server key it pinned at pairing.
- Files are encrypted with a key of their own, which travels only inside the encrypted channel; the HTTP
  side moves ciphertext against one-time tickets.

## Pair a device

A device pairs once, by scanning a QR code. As when linking a device in Signal, the code is the proof: it
carries a one-time secret that only someone looking at it can know, and the secret never travels over the
network. So pairing goes over the same public address as everything else.

```bash
hermes aleph pair
```

shows a QR code and the link it encodes:

```
aleph://pair?server=https%3A%2F%2Faleph.xsl.sh&key=<server key>&secret=<secret>
```

Scan it in Aleph. The phone pairs at `<server>/aleph/v1/pair` with a `Noise_IKpsk1_25519_ChaChaPoly_SHA256`
handshake (prologue `aleph/1/pair`) that uses the secret as its pre-shared key, and uses `server` from then on.

- `server` is `ALEPH_PUBLIC_URL`, or `http://<this machine's LAN address>:<port>` without it; `--server`
  overrides it for one pairing.
- `key` is the server's public key, which the phone pins. `secret` is 32 random bytes, single use, valid for
  ten minutes; a wrong one closes the handshake with `4403`. Running the command again replaces the pending
  pairing, with a warning. The command waits until the phone has paired and names it.
- **Anyone who sees the code can pair**, so show it only to the phone, and only while pairing.
- `/aleph/v1/pair` answers `404` unless a pairing is pending, so most of the time there is nothing to reach.
  It allows 5 attempts a minute per address (behind the tunnel, Cloudflare's `CF-Connecting-IP`, trusted
  only on loopback connections).
- Each pairing posts "New device paired: <name>" into the thread and pushes it, so a pairing nobody expected
  is noticed at once.

```bash
hermes aleph devices          # id, name, model, key fingerprint, when paired, last seen, push environment
hermes aleph revoke d_3f9a…   # the device's key no longer connects
```

Revoking closes the device's open connections within a couple of seconds (WebSocket close code `4401`) and
drops its push token.

## Reach it from anywhere

The phone must reach the agent away from home, so publish the listener through an outbound tunnel. With
[Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/), the
Hermes machine dials out: nothing is port-forwarded, and the name gets a real certificate.

```
phone ──wss──▶ aleph.xsl.sh ──tunnel──▶ cloudflared (this machine) ──▶ 127.0.0.1:8650
        └─ TLS, ends at Cloudflare ─┘
        └──────── Noise, ends in the plugin ─────────────────────────────────┘
```

Create a tunnel (`cloudflared tunnel create hermes`, then `cloudflared tunnel route dns hermes aleph.xsl.sh`)
and send the name, and only the name, to the plugin's listener in `~/.cloudflared/config.yml`:

```yaml
tunnel: hermes
credentials-file: /home/you/.cloudflared/<tunnel-id>.json

ingress:
  - hostname: aleph.xsl.sh
    service: http://127.0.0.1:8650
  - service: http_status:404
```

The listener serves only `/aleph/v1/ws`, `/aleph/v1/pair` and `/aleph/v1/files/`.
Hermes's dashboard and its gateway WebSocket (`/api/ws`) run on other ports and are never published.

Run it with `cloudflared tunnel run hermes` (or install it as a service: `cloudflared service install`), set
`ALEPH_PUBLIC_URL=https://aleph.xsl.sh`, restart the gateway, and pair. WebSockets need no extra settings.

For development on the home network, `ALEPH_HOST=0.0.0.0` (or the LAN address) without `ALEPH_PUBLIC_URL`
puts `http://<LAN address>:8650` in the link; plain `ws://` is for the home network only.

## Push notifications

Push needs the APNs auth key (`.p8`) of the Apple developer team that signs the app. Keep it on the Hermes
machine only, outside any repository, and point the plugin at it:

```bash
ALEPH_APNS_KEY_FILE=/home/you/.hermes/secrets/AuthKey_ABC123DEFG.p8
ALEPH_APNS_KEY_ID=ABC123DEFG
ALEPH_APNS_TEAM_ID=PNDKVJR7LY
ALEPH_APNS_TOPIC=sh.xsl.aleph
```

Without all four, the channel works and pushes nothing; the gateway log says `push on (sh.xsl.aleph)` or
`push off`. APNs speaks only HTTP/2, which comes from the `h2` package in the `messaging` extra.

- The phone registers its APNs token after each connection (`push.register`, per device) and says when the
  person is looking at the app (`presence`). A device with the app in the foreground gets no push.
- Pushed: the agent's messages (files included), approval requests and questions. Not pushed: tool rows,
  system notices, the person's own messages, drafts and typing.
- The push carries only a generic line ("New message", "Approval needed", "Question") and the event's
  `seq`, and no title: the agent goes by the name given to it on the phone, which the server does not
  know, so the app's notification extension sets the title. Debug builds use the APNs sandbox, release builds production; Apple calls go straight from this
  machine, not through the tunnel.
- A token Apple rejects (`410`, or `400 BadDeviceToken`) is dropped. Pushing never delays delivery over the
  WebSocket.

## Home channel

The plugin registers `aleph:main` as its home channel. `session: main` needs exactly one connected platform
with a home channel, so when another platform (Signal, Telegram) also has one, either remove that platform's
home (`SIGNAL_HOME_CHANNEL`, `platforms.<name>.home_channel`) or keep it and set `ALEPH_HOME_CHANNEL=off`.
With `off`, a `deliver=aleph` job needs the explicit target `aleph:main`.

## Settings

| Variable | Default | Meaning |
| --- | --- | --- |
| `ALEPH_ENABLED` | (off) | `true` turns the channel on (or `platforms.aleph.enabled: true` in `config.yaml`) |
| `ALEPH_HOST` | `127.0.0.1` | Address to bind: loopback behind a tunnel, `0.0.0.0` (or a LAN address) for the home network |
| `ALEPH_PORT` | `8650` | Port to bind |
| `ALEPH_PUBLIC_URL` | `http://<LAN address>:<port>` | Base URL the phone uses, pairing included; goes into pairing links |
| `ALEPH_APNS_KEY_FILE` | | Path to the APNs `.p8` key |
| `ALEPH_APNS_KEY_ID` | | The key's id |
| `ALEPH_APNS_TEAM_ID` | | Apple developer team id |
| `ALEPH_APNS_TOPIC` | | The app's bundle id |
| `ALEPH_HOME_CHANNEL` | `main` | Home channel; `off` leaves the home to another platform |
| `ALEPH_HISTORY_LIMIT` | `20000` | Logged events kept for catching up; a phone further behind gets `reset: true` |
| `ALEPH_STATE_DIR` | `$HERMES_HOME/aleph` | The server key (`server.key`), the event log and paired devices (`aleph.db`), and encrypted files |

The same keys can be set under `platforms.aleph.extra` in `config.yaml` (`host`, `port`, `public_url`,
`apns_key_file`, `apns_key_id`, `apns_team_id`, `apns_topic`, `history_limit`, `state_dir`); environment
variables win.

## Endpoints

- `GET /aleph/v1/ws`: the WebSocket. Binary messages only: the Noise IK handshake, then one encrypted
  JSON-RPC 2.0 frame per message. `hello {protocol: 1, after}` first; the server replays what the phone
  lacks, sends `sync.done`, then live events. Also `push.register`, `presence` and `file.ticket`.
- `GET /aleph/v1/pair`: the pairing WebSocket (Noise IKpsk1), there only while a pairing is pending.
- `PUT /aleph/v1/files/<ticket>`: upload ciphertext of exactly the size the ticket was issued for (`204`).
- `GET /aleph/v1/files/<ticket>`: download ciphertext; each link works once.

No request carries credentials: connections authenticate inside the Noise handshake, and file transfers by
their one-time tickets.
