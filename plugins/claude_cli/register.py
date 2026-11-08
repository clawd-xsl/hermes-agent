"""Plugin entry point: register the ``claude_cli`` turn runtime."""

from __future__ import annotations

from plugins.claude_cli.provider import ClaudeCliTurnRuntime


def register(ctx) -> None:
    ctx.register_turn_runtime(ClaudeCliTurnRuntime())
