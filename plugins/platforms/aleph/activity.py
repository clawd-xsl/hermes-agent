"""The agent's activity on Aleph: the steps it takes between a message and its reply (protocol
"Activity"). Not part of the conversation: the phone shows the current step as one line under the
agent's name (``typing.status``) and the steps as a history (``activity`` events, ``activity.list``).

The wording is Hermes's own tool labels (``agent.display.build_tool_label``): a phrase without a
subject in sentence case ("Searching the web for flights to Tokyo"), the form Aleph shows under the
agent's name. Hermes's ``build_status_phrase`` is the Slack form that follows a display name
("is searching the web …").
"""

from __future__ import annotations

import secrets
from typing import Any, Optional

MAX_SUMMARY_CHARS = 80
MAX_DETAIL_CHARS = 2000
# Hermes's bridge tools (``tool_search``): ``tool_call`` runs other tools, which are what the person
# cares about; looking tools up is not something the agent did for the person, so it is no step.
BRIDGE_CALL = "tool_call"
BRIDGE_LOOKUPS = frozenset({"tool_search", "tool_describe"})
# Aleph's own tools, in the same wording as Hermes's curated verbs.
OWN_VERBS = {
    "aleph_react": "Reacting to a message",
    "aleph_set_profile": "Changing name or avatar",
    "aleph_get_profile": "Checking its profile",
}
# Where the full argument is more than one value: the arguments shown one after another.
_DETAIL_PARTS = {
    "write_file": ("path", "content"),
    "patch": ("path", "old_string", "new_string"),
}


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _clip_block(text: str, limit: int) -> str:
    """Like ``_clip`` but keeping line breaks (commands, code, file contents)."""
    text = text.strip("\n").rstrip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def full_argument(tool: str, args: dict[str, Any]) -> Optional[str]:
    """What the tool was given, in full: its primary argument, or for a write or a patch the path
    then the text. Secrets are redacted (Hermes's display redaction of the arguments, then its
    secret patterns over the text)."""
    from agent.display import _FALLBACK_PREVIEW_KEYS, _PRIMARY_ARGS, redact_tool_args_for_display
    args = redact_tool_args_for_display(tool, args) or args
    keys = _DETAIL_PARTS.get(tool)
    if keys is None:
        key = _PRIMARY_ARGS.get(tool) or next((k for k in _FALLBACK_PREVIEW_KEYS if k in args), None)
        keys = (key,) if key else ()
    parts = []
    for key in keys:
        value = args.get(key)
        if isinstance(value, list):
            value = "\n".join(str(item) for item in value)
        if value not in (None, ""):
            parts.append(str(value))
    return _redact("\n\n".join(parts)) or None


def _redact(text: str) -> str:
    from agent.redact import redact_sensitive_text
    return redact_sensitive_text(text, redact_url_credentials=True)


def _unwrap(tool: str, args: Optional[dict[str, Any]]) -> tuple[str, Optional[dict[str, Any]], int]:
    """The tool a ``tool_call`` bridge call runs (its first), its arguments, and how many more."""
    if tool != BRIDGE_CALL or not isinstance(args, dict):
        return tool, args, 0
    calls = args.get("calls")
    calls = calls if isinstance(calls, list) and calls else [args]
    first = calls[0] if isinstance(calls[0], dict) else {}
    name = first.get("name")
    inner = first.get("arguments")
    if not isinstance(name, str) or not name:
        return tool, args, 0
    return name, inner if isinstance(inner, dict) else {}, len(calls) - 1


def describe(tool: str, args: Optional[dict[str, Any]], *,
             with_preview: bool = True) -> Optional[tuple[str, str, Optional[str]]]:
    """``(tool, summary, detail)`` for one tool call: the tool that really ran, the phrase shown for
    it (at most 80 characters), and what it was given in full (at most 2000), or None for a call
    that is no step (looking tools up)."""
    from agent.display import build_tool_label, get_tool_verb
    name, call_args, more = _unwrap(tool, args)
    if name in BRIDGE_LOOKUPS:
        return None
    call_args = call_args or {}
    summary = OWN_VERBS.get(name)
    if summary is None and get_tool_verb(name):
        summary = build_tool_label(name, call_args if with_preview else {}, max_len=MAX_SUMMARY_CHARS)
    if not summary:
        summary = f"Using {name}"  # a tool without a curated verb: its label would be the bare preview
    if more:
        summary = f"{summary} (+{more} more)"
    detail = full_argument(name, call_args) if with_preview and call_args else None
    return name, _clip(_redact(summary), MAX_SUMMARY_CHARS), _clip_block(detail, MAX_DETAIL_CHARS) if detail else None


def new_step(turn: Optional[str], tool: str, summary: str, detail: Optional[str], started_at: str) -> dict[str, Any]:
    step: dict[str, Any] = {"id": f"a_{secrets.token_hex(8)}", "turn": turn, "tool": tool, "summary": summary}
    if detail:
        step["detail"] = detail
    step.update({"started_at": started_at, "finished_at": None, "ok": None})
    return step
