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
```

Start a new Hermes session after changing the runtime. Existing sessions keep
their launch-time model toolset and system prompt so their prompt cache remains
stable.

## Context and history

On the first native turn, Hermes passes the complete effective system prompt
(identity, context files, memory, and skill catalog) once. If the Hermes
session already has a transcript, its complete prior history is handed over as
data before the current user message. Later turns send only the new user input.

Hermes stores the Claude session binding under its profile runtime directory
and resumes it after a gateway restart. If Claude has removed that native
session, Hermes retries the same user turn once with a fresh binding and a full
handoff; the user does not need to resend the message.

Claude's native history is authoritative for context-window management. Hermes
therefore does not run its own summary compression or background memory/skill
review on this runtime. The normal `memory`, `session_search`, `skill_manage`,
and `delegate_task` tools remain available directly during the main turn.

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

The Claude subprocess is not given Anthropic API tokens from Hermes' environment
and cannot use Claude's native Bash/file tools on this path. Terminal and file
work goes through the same Hermes policies as every other provider.

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

## Disable it

Remove `model.anthropic_runtime` (or set it to `auto`) and start a new session.
Hermes will return to its native Anthropic Messages transport.

