"""Persistent Claude Code turn runtime (``model.anthropic_runtime: claude_cli``).

Hermes hands Anthropic turns to one long-lived Claude Code process per session over Claude Code's
bidirectional stream-json protocol: Claude owns the native conversation, prompt cache and
compaction; Hermes keeps the session, channels, memory, skills, approvals, scheduling and every
tool (exposed to Claude through a private MCP server). See ``provider.py``.

Absolute import on purpose: the plugin loader imports this package as ``hermes_plugins.claude_cli``,
while core and tests import ``plugins.claude_cli``. One canonical module tree keeps a single
process-wide session pool.
"""

from plugins.claude_cli.register import register

__all__ = ["register"]
