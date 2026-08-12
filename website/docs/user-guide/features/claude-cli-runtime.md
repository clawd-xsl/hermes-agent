---
title: Persistent Claude CLI Runtime
sidebar_label: Persistent Claude CLI Runtime
---

# Persistent Claude CLI Runtime

Hermes can hand Anthropic turns to one long-lived Claude Code process instead
of creating an HTTP request loop itself. Claude owns the native conversation,
prompt cache, and compaction. Hermes still owns the visible session, messaging
channels, memory, skills, approvals, scheduled delivery, and every configured
tool.

The process stays alive across turns. This avoids repeated Node startup,
credential loading, MCP discovery, and system-prompt ingestion, which is most
noticeable for an always-on single-user assistant.

## Enable it

Install and log in to Claude Code, and make sure Hermes includes its MCP extra:

```bash
npm install -g @anthropic-ai/claude-code
claude login
pip install 'hermes-agent[mcp]'
```

Then edit `~/.hermes/config.yaml`:

```yaml
model:
  provider: "anthropic"
  default: "claude-sonnet-4-6"
  anthropic_runtime: "claude_cli"
  claude_cli:
    command: "claude"
    turn_timeout_seconds: 600
    # Optional: may consume additional usage credits when available.
    fast_mode: false
```

Start a new Hermes session after changing the runtime. Existing sessions keep
their launch-time model toolset and system prompt so their prompt cache remains
stable.

Claude Code authenticates from its own logged-in keychain, or from its official
`CLAUDE_CODE_OAUTH_TOKEN` headless override when that variable is explicitly
exported for the Hermes process. Hermes deliberately does not copy
`ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, HTTP endpoints, or credential pools
into the child. Fallback providers still use their normal Hermes credentials,
but Anthropic API-key rotation and HTTP credential-pool recovery do not apply
while the active runtime is `claude_cli`.

## Capability ownership

The child is an inference and native-history backend, not a second personal
agent. Hermes disables Claude Code features that would create a competing
memory, tool, scheduler, or extension state machine. Their user-visible
outcomes remain owned as follows:

| Claude Code capability disabled in the child | Owner on this runtime |
| --- | --- |
| Auto memory | Hermes memory, external memory-provider prefetch/sync, and the normal periodic background memory review |
| Background tasks | Hermes `delegate_task`, background-agent surfaces, and background memory/skill self-improvement review |
| Bundled skills and skill slash commands | Hermes skills, skill commands, and `skill_manage` |
| `CLAUDE.md` discovery | Hermes' stable identity/context-file prompt (`SOUL.md`, `AGENTS.md`, memory, and enabled skill catalog) |
| Claude cron | Hermes cron; a `wake_main` job enters the real main session with its complete main-agent context |
| Claude hooks and settings sources | Hermes lifecycle hooks, request/tool middleware, approvals, and `config.yaml` |
| Claude plugins, marketplace auto-install, and plugin MCP servers | Hermes plugins and its MCP/tool registry |
| Native Bash, file tools, and Chrome | Hermes terminal, file, and browser tools with the same approvals and guardrails as other providers |
| Prompt suggestions, updater, and nonessential traffic | Host UI or operator-owned maintenance; disabled in the child to avoid IM noise and startup/network latency |

Claude Fast Mode is the exception: it is enabled only when the user selects
Hermes `/fast` or `model.claude_cli.fast_mode`, because it can consume
additional credits.

## Context and history

On the first native turn, Hermes passes the complete effective system prompt
(identity, context files, memory, and skill catalog) once. If the Hermes
session already has a transcript, its complete prior history is handed over as
data before the current user message. Later turns send only the new user input.

Hermes stores the Claude session binding under its profile runtime directory
and resumes it after a gateway restart. If Claude has removed that native
session, Hermes retries the same user turn once with a fresh binding and a full
handoff; the user does not need to resend the message.

Claude's native history is authoritative for context-window management, so
Hermes does not build a second, divergent session summary, proactively prune,
or micro-compact only its local mirror. Those operations would report a smaller
Hermes transcript while leaving the physical context sent to Claude unchanged.
Automatic compaction therefore uses Claude's native implementation. The normal
`compression.enabled` setting is still authoritative: when it is `false`,
Hermes starts the child with native automatic compaction disabled. An explicit
Hermes `/compress [focus]` command is forwarded to Claude as native `/compact`
and succeeds only after Claude emits a real compaction boundary. `/compress
here N` adds an instruction to keep the most recent N exchanges in full detail
while compacting older native context. Hermes deliberately retains the complete
visible transcript for search and crash recovery, so message counts in the UI
do not shrink even though Claude's live model context does.

Manual `/compress` remains available when automatic compression is disabled.
Hermes temporarily restarts the same resumable native session with compaction
enabled for that command, observes the real boundary, then restores the disabled
automatic policy before the next user turn. A configured external
`ContextEngine` keeps its own selection and compression policy; after a
committed rewrite Hermes rebuilds the native binding from that authoritative
transcript.

Hermes still runs its normal completed-turn continuity work:
external memory-provider sync, context-engine `on_turn_complete`, and periodic
background memory/skill review. Review forks use an isolated native Claude
session and cannot enter the foreground conversation history. The normal
`memory`, `session_search`, `skill_manage`, and `delegate_task` tools remain
available directly during the main turn.

If Claude Code reaches a terminal runtime error or keeps returning empty
content after Hermes' bounded recovery attempts, Hermes continues the same
already-prepared turn on the configured `fallback_providers` chain. It does not
rerun turn setup or append the user's message twice.

The same recovery contract covers model stop reasons. `max_tokens` triggers a
bounded continuation and assembles the complete answer; `refusal` is treated as
a deterministic content-policy block, tries a configured fallback once, and
otherwise returns an actionable refusal without empty-response retries. Every
native model iteration updates ContextEngine usage and emits request observer
hooks, including empty/refused responses. Auxiliary Hermes calls (titles,
summaries, plugin LLM calls, vision, and similar work) run in isolated Claude
children. JSON response formats from those calls, and a main-agent
`request_overrides.response_format`, are forwarded to Claude's `--json-schema`
validation instead of being reduced to a prompt-only hint. Other
provider-specific API body overrides that have no Claude Code equivalent are
reported as ignored controls in the turn result.

An explicit `/moa` turn retains Hermes' normal per-model-step advisor refresh:
fresh guidance is attached at each MCP tool-result boundary as request-only
context. Because Claude's stateful history would otherwise retain those
private advisor blocks after Hermes has discarded them, the child remains warm
for the complete MoA tool loop and is retired at that turn boundary. The next
ordinary turn cold-bootstraps from the exact durable Hermes transcript.

## Tool and security model

Claude starts with only `ToolSearch` plus an MCP server named `hermes`. That
server exposes the exact tool schema enabled on the live Hermes session. Calls
return through a loopback-only authenticated bridge and execute in Hermes'
standard sequential tool executor, preserving:

- toolset/session scoping;
- request middleware and plugin hooks;
- approvals and tool guardrails;
- checkpoints before file mutations or destructive commands;
- result-size budgets, progress callbacks, and mutation verification;
- live agent tools such as memory, skills, session search, todo, and delegation.

Tool intent is persisted before execution and its result immediately after.
Claude's stdout event and its parallel MCP request may arrive in either order;
the bridge reconciles them by native tool id and call signature so the
transcript contains one durable tool pair without allowing a side effect to
overtake persistence. If Claude reuses an id inside one batch, Hermes applies
the same deterministic `_d2`, `_d3`, ... repair as the standard agent loop so
later results cannot overwrite earlier calls.

The Claude subprocess is not given Hermes' Anthropic HTTP API credentials. An
explicit `CLAUDE_CODE_OAUTH_TOKEN` is passed through only as Claude Code's own
documented headless authentication mechanism. The child cannot use Claude's
native Bash/file tools on this path; terminal and file work goes through the
same Hermes policies as every other provider.

## Streaming and latency diagnostics

Text deltas stream through the normal Hermes callback, including intermediate
commentary around tool calls. Only the final assistant message becomes the turn
result; an empty terminal result falls back to the last complete streamed
assistant message.

Each result includes `claude_session_reuse` (`cold_miss`, `native_resume`,
`warm_hit`, or `resume_recovery`) and `claude_latency_ms` timing fields for
process startup, first parsed record, first text, total turn time, and process
age. These are available to gateway/session diagnostics without being inserted
into model context.

Hermes maps `model.max_tokens` to Claude Code's native output cap, translates
Hermes' total API-attempt ceiling to Claude's retry count, and applies
`providers.anthropic.request_timeout_seconds` (or the model-specific
`timeout_seconds`) to each child provider request. The separate
`model.claude_cli.turn_timeout_seconds` bounds the complete private tool loop;
when omitted, its default grows as needed to accommodate an explicitly longer
provider request timeout. These startup-scoped values are part of the
warm-child compatibility signature, so changing one retires the old process
instead of silently leaving it on stale limits. Reasoning effort is mapped to
Claude's native `--effort` and `--thinking` controls.

Claude Code still does not expose arbitrary API-wire controls such as
`temperature`, custom `extra_body` fields, or non-fast HTTP service tiers on
this persistent stream-json transport. For supported subscriptions and models,
Hermes' normal `/fast on` setting maps directly to Claude Code's native Fast
Mode. You can also set `model.claude_cli.fast_mode: true` as a runtime-wide
default. Fast Mode is never enabled implicitly because it can consume
additional credits. Unsupported controls are reported in the turn result
rather than silently claimed as applied.

There is one hard transport boundary: Claude Code constructs the private
follow-up requests inside its own tool loop. Hermes can observe every inner
request/response boundary, account its usage, run `agent:step`, and control all
tool execution, but an LLM request/execution middleware or ContextEngine cannot
rewrite each private inner request after the first one. Its request mutation
and context selection apply to the outer logical native turn; per-iteration
`pre_api_request`/`post_api_request` hooks are observer-only. Use the direct
`anthropic_messages` runtime for a plugin whose correctness requires mutating
every provider request.

## Disable it

Remove `model.anthropic_runtime` (or set it to `auto`) and start a new session.
Hermes will return to its native Anthropic Messages transport.
