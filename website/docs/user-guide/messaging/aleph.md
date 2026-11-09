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
- the agent's activity out of the thread: one line saying what it is doing, and a history of its steps;
- gateway notices as system notes;
- approval cards for dangerous commands and question cards for the `clarify` tool;
- attachments both ways, and shared locations (they reach the agent as a location pin with a map link);
- end-to-end encryption: everything between the phone and the plugin, files included, is ciphertext to
  the network, the tunnel and Cloudflare;
- push notifications through APNs that carry no content: the app's notification extension fetches the
  real text over the channel itself.
- the agent's own name and avatar, which only the agent changes, with the `aleph` tools.

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
  `aleph/1`). An unknown or revoked device key, or a handshake not finished within 5 seconds, closes with
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

## What strangers can reach

The public address is reachable by anyone, and the two WebSockets must answer before anyone is
authenticated. The plugin keeps that surface small:

1. **Upgrade gate.** Every upgrade to `/aleph/v1/ws` and `/aleph/v1/pair` carries
   `X-Aleph-Mac: <ts>.<nonce>.<mac>`, an HMAC-SHA256 keyed by `SHA256("aleph/1/mac" ‖ server public key)`
   over `<ts>.<nonce>.<path>`. A missing or malformed header, a `ts` more than 120 seconds off, a nonce
   seen in the last 240 seconds or a wrong MAC gets a bare `404`, the same as a path that does not exist.
   The server key is only in pairing codes and on paired phones (WireGuard's `mac1` idea).
2. **A small, strict handshake.** At most 16 unauthenticated connections in all and 2 per address. The
   first message must be binary and exactly 96 bytes (`/ws`) or at most 512 (`/pair`), within 5 seconds;
   anything else closes before any cryptography. Until the handshake succeeds nothing is parsed, nothing
   is read from the database (device keys and the pending pairing are held in memory), and nothing the
   peer sent is logged.
3. **An edge gate (optional).** Behind Cloudflare, the hostname can also sit behind
   [Cloudflare Access](https://developers.cloudflare.com/cloudflare-one/identity/service-tokens/) with a
   service token, so requests without it never reach this machine:
   - In Zero Trust → Access, create a service token, and a self-hosted application for `aleph.xsl.sh`
     with one policy: action **Service Auth**, include **Service Token** → that token.
   - Put the token on the Hermes machine: `ALEPH_ACCESS_ID=<client id>` and
     `ALEPH_ACCESS_SECRET=<client secret>`. `hermes aleph pair` then adds `access_id` and `access_secret` to
     the link, and the phone sends them on every request as `CF-Access-Client-Id` /
     `CF-Access-Client-Secret`.
   - The plugin never checks them; Cloudflare does. Without Cloudflare, leave them out: layers 1 and 2
     still hold.

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

The listener serves only `/aleph/v1/ws`, `/aleph/v1/pair` and `/aleph/v1/files/`; Hermes's dashboard and its
gateway WebSocket (`/api/ws`) run on other ports and are never published. Route the whole name rather than
path rules: with path rules, the plugin's paths would answer with the plugin's `404` and every other path
with cloudflared's, and a scanner could tell which paths exist. With one rule, every path a stranger tries
(an unknown one, `/aleph/v1/ws` without a valid `X-Aleph-Mac`, a wrong method) gets the same bare `404`.

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

## Activity

What the agent does between a message and its reply is not part of the conversation, so Hermes's
tool-progress messages are off on Aleph (`display.platforms.aleph.tool_progress: off` by default; set it
back to `all` to have them in the thread), and so are the "still working" notices
(`long_running_notifications`). Instead:

- the phone shows one line under the agent's name while it works, such as "Searching the web for flights
  to Tokyo…", from Hermes's own tool labels, and nothing between tools;
- each tool call is a step, sent to connected phones as it starts and finishes and kept on the server
  (the most recent 500) for the history the phone shows; `activity.list` pages through them. A step
  carries what the tool was given in full (the command, the query, the file and its text, up to 2000
  characters with its line breaks), with secrets redacted by Hermes's redaction patterns.

A call through the `tool_call` bridge is shown as the tool it runs; looking tools up (`tool_search`,
`tool_describe`) is no step, and the line stays on "Thinking…". Aleph's own tools have their own words
("Reacting to a message", "Changing name or avatar").

`display.platforms.aleph.live_status` controls the wording: `full` (the default) names what the tool
was given, `verb` only what it does ("Searching the web") and leaves the arguments out of the steps
too, `off` shows no line. Steps are recorded either way.

## Attachments and locations

The person's files reach the agent through the gateway's media cache, like on any channel: images go to
the model natively when it has vision, voice notes are transcribed, other files are noted with their paths.
The phone converts camera formats such as HEIC to JPEG first; videos and other files go as they are, with
the width, height and duration the phone knows, so every device can lay them out before downloading.

An upload may be up to 200 MB, but never more than the gateway accepts for inbound media
(`gateway.max_inbound_media_bytes` in `config.yaml`, 128 MB by default); raise that to allow longer videos.
A shared location reaches the agent as a location message: the place and address the phone looked up,
the coordinates and an Apple Maps link, then any text sent with it.

## The agent's name and avatar

On Aleph the agent has a name (`Aleph` until it picks one) and an avatar (none until it sets one; the
app then draws its mark). Only the agent changes them: the app has no editor, so the person asks in the
conversation, and the agent uses its `aleph` tools. The server keeps the profile and is the one truth for
every device; `hello` carries it, and each change is logged as an `agent.profile` event (not pushed), so
every device updates at once and one that was away catches up.

The `aleph` toolset is on by default and offered only where the gateway runs a connected Aleph channel:

- `aleph_get_profile` reads the name and avatar.
- `aleph_set_profile` changes the name (1 to 32 characters, one line), the avatar, or both; or goes back to
  the default avatar with `reset_avatar`. `avatar` is an absolute path on this machine or an `https` URL,
  such as what an image-generation tool returned, to a PNG, JPEG, HEIC, WebP or GIF of at most 20 MB.

The avatar is turned upright, cropped to its centered square, scaled down to 512 pixels, flattened onto
white and stored as a JPEG without metadata, encrypted like any file. URLs are fetched only over `https`
(redirects included) and never from a private or internal address. Setting what is already there changes
nothing; an invalid request changes nothing and tells the agent what to fix.

The tools are defined as MCP tools (revision 2025-06-18, with title, output schema and annotations) and
answer with an MCP `CallToolResult` as JSON; in Hermes the input schema is the tool's parameters.

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
| `ALEPH_ACCESS_ID` | | Optional: a Cloudflare Access service token's client id, put into pairing links |
| `ALEPH_ACCESS_SECRET` | | Optional: that token's client secret |
| `ALEPH_APNS_KEY_FILE` | | Path to the APNs `.p8` key |
| `ALEPH_APNS_KEY_ID` | | The key's id |
| `ALEPH_APNS_TEAM_ID` | | Apple developer team id |
| `ALEPH_APNS_TOPIC` | | The app's bundle id |
| `ALEPH_HOME_CHANNEL` | `main` | Home channel; `off` leaves the home to another platform |
| `ALEPH_HISTORY_LIMIT` | `20000` | Logged events kept for catching up; a phone further behind gets `reset: true` |
| `ALEPH_STATE_DIR` | `$HERMES_HOME/aleph` | The server key (`server.key`), the event log and paired devices (`aleph.db`), and encrypted files |

The same keys can be set under `platforms.aleph.extra` in `config.yaml` (`host`, `port`, `public_url`, `access_id`, `access_secret`,
`apns_key_file`, `apns_key_id`, `apns_team_id`, `apns_topic`, `history_limit`, `state_dir`); environment
variables win.

## Endpoints

- `GET /aleph/v1/ws`: the WebSocket. Binary messages only: the Noise IK handshake, then one encrypted
  JSON-RPC 2.0 frame per message. `hello {protocol: 1, after}` first; the server replays what the phone
  lacks, sends `sync.done`, then live events. Also `push.register`, `presence` and `file.ticket`.
- `GET /aleph/v1/pair`: the pairing WebSocket (Noise IKpsk1), there only while a pairing is pending.
- `PUT /aleph/v1/files/<ticket>`: upload ciphertext of exactly the size the ticket was issued for (`204`).
- `GET /aleph/v1/files/<ticket>`: download ciphertext; each link works once.

No request carries credentials: the two WebSocket upgrades carry only the `X-Aleph-Mac` filter,
connections authenticate inside the Noise handshake, and file transfers by their one-time tickets.
