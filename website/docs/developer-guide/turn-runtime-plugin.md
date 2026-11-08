---
title: Turn Runtime Plugins
sidebar_label: Turn runtime plugins
---

# Turn Runtime Plugins

A turn runtime **owns a whole turn's model/tool loop** in an external process — a long-lived CLI
child, an app server — instead of Hermes issuing one provider request per model step. The built-in
example is the Codex app-server runtime (`model.openai_runtime: codex_app_server`); plugins add
others through `ctx.register_turn_runtime()`.

A runtime is keyed by the `api_mode` it owns. An agent whose `api_mode` names a runtime hands every
prepared turn to the runtime, and every place that would otherwise act on Hermes' local mirror of the
conversation asks the runtime first:

| Seam | Runtime method | Default |
| --- | --- | --- |
| Runtime resolution (`hermes_cli.runtime_provider`) | `resolve_runtime(provider=, api_mode=, model_cfg=)` → overrides or None | None |
| Agent construction | `init_client(agent)` → True when the runtime owns inference (no HTTP client) | False |
| The turn (`agent/conversation_loop.py`) | `run_turn(agent, turn)` → result, or None after activating a fallback | abstract |
| Interrupt (any thread) | `request_interrupt(agent)` | no-op |
| Mid-turn steer / redirect | `request_steer(agent, text)` → True/False, or None to use Hermes' steer buffer | None |
| Teardown | `release(agent, hard=)` — `close()` is hard, cache eviction soft | no-op |
| Compaction ownership | `owns_compaction(agent)`, `native_auto_compaction(agent)` | True, True |
| Compaction (`compress_context`, `/compress`) | `compact(agent, messages, system_message, force=, focus_topic=, verbatim_tail=, trigger=, ...)` | None (skip) |
| Attached images | `image_input_mode(agent)` → `"text"`, `"native"` or None | None |

Core keeps everything around the turn: the turn context (system prompt, memory prefetch, plugin
context), the user row, session persistence, finalization, delivery and gateway session hygiene (which
skips a plugin runtime: its native context is not the transcript mirror).

## Selection

`resolve_runtime` runs **before** the credential ladder for the requested provider (a login-only
setup has no key for the ladder to find) and again after it for an implicit (`auto`) request. Return
the keys AIAgent construction needs (`command`, `args`, ...); `api_mode` is always set to the
runtime's name. Explicit `--api-key` / `--base-url` keep the provider's own HTTP route.

## Fallback

A failed turn may activate a configured fallback provider (`agent._try_activate_fallback()`), update
`turn.api_call_count` and return None: the generic loop retries the same prepared turn. The reverse
also works — when a generic-loop fallback activates a runtime's `api_mode`, the loop hands the
prepared turn to `run_turn` with `turn.continuation = True`. Returning None without moving the agent
off the runtime is an error.

## Constraints

Runtimes receive live `AIAgent` objects, so they run in-process only (refused under
`plugins.isolation: host`). Built-in names (`codex_app_server`) cannot be registered by a plugin.
