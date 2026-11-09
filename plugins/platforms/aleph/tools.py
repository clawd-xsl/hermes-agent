"""The ``aleph`` toolset: tools the agent calls to act on the Aleph app (protocol "Tools").

Each tool is defined the way MCP defines one (revision 2025-06-18: ``name``, ``title``,
``description``, ``inputSchema``, ``outputSchema``, ``annotations``) and answers with a
``CallToolResult``: one text block the agent can relay plus ``structuredContent`` matching
``outputSchema``, or ``isError: true`` with a text block that says what to fix. Inside Hermes they
are plugin tools: the input schema is the tool's parameters and the result is the
``CallToolResult`` as JSON.

They act on the running gateway's Aleph adapter, so they are offered only in a process where that
adapter is connected (``check_fn``), and run on the gateway's loop.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from plugins.platforms.aleph.profile import AVATAR_SIDE, MAX_NAME_CHARS, ProfileError, clean_name, load_avatar

logger = logging.getLogger(__name__)

TOOLSET = "aleph"

_AVATAR_INFO = {
    "type": ["object", "null"],
    "description": "The avatar as stored, or null for the app's drawn default.",
    "properties": {
        "width": {"type": "integer", "description": "Pixels."},
        "height": {"type": "integer", "description": "Pixels."},
        "updated_at": {"type": "string", "description": "When it was set, ISO 8601."},
    },
    "required": ["width", "height", "updated_at"],
    "additionalProperties": False,
}

GET_PROFILE = {
    "name": "aleph_get_profile",
    "title": "Get your Aleph profile",
    "description": (
        "Read your own name and avatar as the person sees them in the Aleph app (their iOS app for "
        "talking to you). Use it before changing your profile, or when the person asks what you are "
        "called there or whether you have a picture. avatar is null when you have none and the app "
        "draws its default mark."),
    "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    "outputSchema": {
        "type": "object",
        "properties": {"name": {"type": "string"}, "avatar": _AVATAR_INFO},
        "required": ["name", "avatar"],
        "additionalProperties": False,
    },
    "annotations": {"readOnlyHint": True, "openWorldHint": False},
}

SET_PROFILE = {
    "name": "aleph_set_profile",
    "title": "Change your Aleph name or avatar",
    "description": (
        "Change your own name, avatar, or both in the Aleph app. Only you can change them: the app has "
        "no editor, so when the person asks for a new name or picture, use this tool. Every device "
        "shows the change at once, with a note in the conversation.\n"
        f"- name: 1 to {MAX_NAME_CHARS} characters, one line.\n"
        "- avatar: an absolute file path on this machine or an https URL, to a PNG, JPEG, HEIC, WebP or "
        "GIF image (first frame) of at most 20 MB. A picture made with an image tool works: pass the "
        "path or URL it returned. The image is turned upright, cropped to its centered square and "
        f"scaled down to {AVATAR_SIDE} pixels, so prefer a square picture with the subject centered.\n"
        "- reset_avatar: true goes back to the app's default mark. Do not combine it with avatar.\n"
        "Give at least one. Setting what is already there changes nothing (changed is empty). On an "
        "error (isError true) nothing was changed: fix what the message says and call again."),
    "inputSchema": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": f"Your new name, 1 to {MAX_NAME_CHARS} characters, one line."},
            "avatar": {"type": "string",
                       "description": "Absolute path or https URL of the new avatar image."},
            "reset_avatar": {"type": "boolean", "description": "true to go back to the default avatar."},
        },
        "additionalProperties": False,
    },
    "outputSchema": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "avatar": _AVATAR_INFO,
            "changed": {"type": "array", "items": {"type": "string", "enum": ["name", "avatar"]},
                        "description": "What actually changed; empty when everything was already so."},
        },
        "required": ["name", "avatar", "changed"],
        "additionalProperties": False,
    },
    "annotations": {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
}

REACT = {
    "name": "aleph_react",
    "title": "React to a message in Aleph",
    "description": (
        "React to one of the person's messages in Aleph with a single emoji, the way they would tap back "
        "in Messages. Reach for it when a reaction is what a person would do: something funny gets a 😂, "
        "warmth gets a ❤️, a plan you're on board with gets a 👍. Then carry on with whatever the message "
        "needs; if the reaction says it all, it can be the whole reply. Use it like a person: occasionally, "
        "when felt, not on every message and never as a status signal. Never narrate it ('I reacted "
        "with…'): the emoji on the bubble is the point. It goes on the person's latest message unless you "
        "say otherwise; messages_back reaches an earlier one (1 = the one before the latest), for when "
        "something lands late. One reaction per message: another emoji replaces yours, an empty string "
        "takes it back."),
    "inputSchema": {
        "type": "object",
        "properties": {
            "emoji": {"type": "string",
                      "description": "One emoji (e.g. '❤️', '😂', '👍'), or an empty string to take your reaction back."},
            "messages_back": {"type": "integer", "minimum": 0,
                              "description": "Optional. 0 (the default) is the person's latest message, 1 the one before, and so on."},
            "message_id": {"type": "string",
                           "description": "Optional. A specific message's id, instead of messages_back."},
        },
        "required": ["emoji"],
        "additionalProperties": False,
    },
    "outputSchema": {
        "type": "object",
        "properties": {
            "message_id": {"type": "string", "description": "The message reacted to."},
            "emoji": {"type": ["string", "null"], "description": "Your reaction on it now; null when taken back."},
        },
        "required": ["message_id", "emoji"],
        "additionalProperties": False,
    },
    "annotations": {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
}

TOOLS = (GET_PROFILE, SET_PROFILE, REACT)


def hermes_schema(tool: dict[str, Any]) -> dict[str, Any]:
    """The tool as Hermes registers it: the MCP input schema becomes the parameters."""
    return {"name": tool["name"], "description": tool["description"], "parameters": tool["inputSchema"]}


def ok(text: str, structured: dict[str, Any]) -> str:
    return json.dumps({"content": [{"type": "text", "text": text}], "structuredContent": structured},
                      ensure_ascii=False)


def error(text: str) -> str:
    return json.dumps({"content": [{"type": "text", "text": text}], "isError": True}, ensure_ascii=False)


# ── the live adapter ───────────────────────────────────────────────────────────────────────────


def _live() -> tuple[Any, Any]:
    from gateway.config import Platform
    from tools.send_message_senders import _live_adapter
    runner, adapter = _live_adapter(Platform("aleph"))
    if adapter is None or getattr(adapter, "_store", None) is None:
        return runner, None
    return runner, adapter


def _session_chat() -> str:
    """The Aleph chat this turn is in (a side chat, or ``main`` for any other origin)."""
    from gateway.session_context import get_session_env
    if get_session_env("HERMES_SESSION_PLATFORM", "") == "aleph":
        return get_session_env("HERMES_SESSION_CHAT_ID", "") or "main"
    return "main"


def aleph_connected() -> bool:
    """``check_fn``: the tools work only where the gateway runs a connected Aleph adapter."""
    try:
        return _live()[1] is not None
    except Exception:
        return False


async def _on_gateway(runner: Any, make_coro) -> Any:
    from tools.send_message_tool import _dispatch_on_gateway_loop
    return await _dispatch_on_gateway_loop(runner, make_coro, "aleph tool: failed to schedule on the gateway loop")


def _not_running() -> str:
    return error("The Aleph channel is not running in this process, so the profile cannot be read or changed "
                 "here. It works in conversations that go through the gateway with Aleph enabled.")


# ── handlers ───────────────────────────────────────────────────────────────────────────────────


def _describe(profile: dict[str, Any]) -> str:
    avatar = profile["avatar"]
    picture = (f"an avatar ({avatar['width']}x{avatar['height']}, set {avatar['updated_at']})" if avatar
               else "no avatar (the app shows its default mark)")
    return f"Your name in Aleph is {profile['name']!r}, with {picture}."


async def get_profile(args: dict[str, Any], **_: Any) -> str:
    runner, adapter = _live()
    if adapter is None:
        return _not_running()

    async def read():
        return adapter.profile_view()
    profile = await _on_gateway(runner, read)
    return ok(_describe(profile), profile)


async def set_profile(args: dict[str, Any], **_: Any) -> str:
    args = args or {}
    unknown = sorted(set(args) - {"name", "avatar", "reset_avatar"})
    if unknown:
        return error(f"Unknown argument(s): {', '.join(unknown)}. Use name, avatar or reset_avatar.")
    name_arg, avatar_arg, reset = args.get("name"), args.get("avatar"), args.get("reset_avatar")
    if reset is not None and not isinstance(reset, bool):
        return error("reset_avatar must be true or false.")
    if name_arg is None and avatar_arg is None and not reset:
        return error("Nothing to change: give name, avatar, or reset_avatar: true.")
    if avatar_arg is not None and reset:
        return error("Give either avatar or reset_avatar, not both.")
    runner, adapter = _live()
    if adapter is None:
        return _not_running()
    try:
        name: Optional[str] = clean_name(name_arg) if name_arg is not None else None
        avatar = await load_avatar(avatar_arg) if avatar_arg is not None else None
    except ProfileError as exc:
        return error(f"{exc} Nothing was changed.")

    async def apply():
        return await adapter.update_profile(name=name, avatar=avatar, reset_avatar=bool(reset))
    try:
        result = await _on_gateway(runner, apply)
    except Exception as exc:
        logger.warning("Aleph: applying a profile change failed", exc_info=True)
        return error(f"The change could not be applied ({exc.__class__.__name__}: {exc}). Nothing was changed.")
    if isinstance(result, dict) and "error" in result and "changed" not in result:
        return error(f"The change could not be applied: {result['error']}. Nothing was changed.")
    changed = result["changed"]
    if not changed:
        text = "Nothing changed: that is already your profile. " + _describe(result)
    else:
        what = " and ".join("name" if c == "name" else "avatar" for c in changed)
        text = f"Changed your {what}; the person sees it now. " + _describe(result)
    return ok(text, result)


async def react(args: dict[str, Any], **_: Any) -> str:
    from plugins.platforms.aleph.reactions import clean_emoji
    args = args or {}
    unknown = sorted(set(args) - {"emoji", "messages_back", "message_id"})
    if unknown:
        return error(f"Unknown argument(s): {', '.join(unknown)}. Use emoji, and messages_back or message_id.")
    if "emoji" not in args:
        return error("emoji is required: one emoji, or an empty string to take your reaction back.")
    try:
        emoji = clean_emoji(args.get("emoji"))
    except ValueError as exc:
        return error(f"{exc}. Give exactly one emoji, such as ❤️ or 👍.")
    back, message_id = args.get("messages_back"), args.get("message_id")
    if back is not None and (isinstance(back, bool) or not isinstance(back, int) or back < 0):
        return error("messages_back must be 0 or more.")
    if message_id is not None and (not isinstance(message_id, str) or not message_id):
        return error("message_id must be a message id.")
    if message_id and back:
        return error("Give either message_id or messages_back, not both.")
    runner, adapter = _live()
    if adapter is None:
        return _not_running()
    chat = _session_chat()

    async def apply():
        return await adapter.react(emoji, chat=chat, message_id=message_id, messages_back=back or 0)
    try:
        result = await _on_gateway(runner, apply)
    except ValueError as exc:
        return error(f"Could not react: {exc}.")
    except Exception as exc:
        logger.warning("Aleph: reacting failed", exc_info=True)
        return error(f"Could not react ({exc.__class__.__name__}: {exc}).")
    if isinstance(result, dict) and "error" in result and "message_id" not in result:
        return error(f"Could not react: {result['error']}.")
    text = "Reaction removed." if result["emoji"] is None else f"Reacted {result['emoji']}."
    return ok(text, result)


def register_tools(ctx) -> None:
    handlers = {"aleph_get_profile": get_profile, "aleph_set_profile": set_profile, "aleph_react": react}
    for tool in TOOLS:
        ctx.register_tool(name=tool["name"], toolset=TOOLSET, schema=hermes_schema(tool),
                          handler=handlers[tool["name"]], check_fn=aleph_connected, is_async=True,
                          description=tool["title"], emoji="ℵ")
