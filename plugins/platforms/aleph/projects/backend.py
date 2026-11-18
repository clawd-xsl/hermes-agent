"""What a coding agent is to the projects runtime, whichever it is (Claude Code, Codex).

A ``Backend`` says what it can run here (``probe``) and opens one ``BackendSession`` per thread. A
session reports what happens as backend-neutral events, already in the protocol's shape (an item's
kind, ACP tool kind, title and input), through the ``Sink`` it was opened with; the runtime gives
items their wire ids, logs them and keeps the policy that is no backend's (holding a silent reply,
one plan per turn).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, Protocol

APPROVALS = ("ask", "auto_review", "allow_all")
MAX_TITLE = 200  # an item's title, one line
MAX_INPUT = 4000  # an item's input
_RECOMMENDED = re.compile(r"\s*\(recommended\)\s*$", re.IGNORECASE)


def clip(text: str, limit: int = MAX_INPUT) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def title(text: Any) -> str:
    """An item's title: one line, at most ``MAX_TITLE``."""
    return clip(" ".join(str(text or "").split()), MAX_TITLE)


def plain_options(labels: Iterable[str]) -> tuple[tuple[str, ...], Optional[int]]:
    """A question's option labels without the agent's "(Recommended)" mark, and the index of the one
    that carried it."""
    labels = list(labels)
    recommended = next((index for index, label in enumerate(labels) if _RECOMMENDED.search(label)), None)
    return tuple(_RECOMMENDED.sub("", label) for label in labels), recommended


@dataclass(frozen=True)
class Model:
    id: str
    name: str
    efforts: tuple[str, ...] = ()  # the model's own levels, in order
    default_effort: Optional[str] = None

    def wire(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "efforts": list(self.efforts), "default_effort": self.default_effort}


@dataclass(frozen=True)
class BackendInfo:
    """``backends.list``'s entry for one backend: whether a turn can run now, and its choices."""
    id: str
    name: str
    available: bool
    reason: Optional[str] = None  # not_installed, signed_out or error, when not available
    message: Optional[str] = None  # why, on one line
    models: tuple[Model, ...] = ()
    default_model: Optional[str] = None
    approvals: tuple[str, ...] = ()

    def model(self, model_id: Optional[str]) -> Optional[Model]:
        """The model ``model_id`` names, null meaning the default."""
        model_id = model_id or self.default_model
        return next((model for model in self.models if model.id == model_id), None)

    def wire(self) -> dict[str, Any]:
        entry: dict[str, Any] = {"id": self.id, "name": self.name, "available": self.available,
                                 "reason": self.reason}
        if not self.available:
            entry["message"] = self.message
        entry.update(models=[model.wire() for model in self.models], default_model=self.default_model,
                     approvals=list(self.approvals))
        return entry


@dataclass(frozen=True)
class McpServer:
    """A stdio MCP server the session's agent gets, under ``name``: ``command`` and ``args`` start it.
    ``env`` holds its credentials: it goes into the server's environment, never onto a command line
    (Claude Code: the CLI's own environment, named by ``${VAR}`` in ``--mcp-config``)."""
    name: str
    command: str
    args: tuple[str, ...]
    env: dict[str, str]
    tools: tuple[str, ...]  # its tools, which the agent calls without asking the person


@dataclass(frozen=True)
class SessionSpec:
    """What a thread's session runs with: its folder, its effective settings, and the coordination
    it is part of (fixed when its process starts)."""
    thread: str
    project: str
    folder: str
    top_level: bool
    model: Optional[str]
    effort: Optional[str]
    approval: str
    plan: bool
    resume: Optional[str] = None  # the backend's own session id, to resume
    instructions: str = ""  # added to the backend's own system prompt
    mcp: Optional[McpServer] = None  # the coordination tools (``tag.SERVER``)
    # The project's shared memory: outside its folder, yet the agent writes there without asking.
    memory: Optional[str] = None
    # Where Claude Code's Stop hook (``child.py stop``, the path in ``child.STOP_ENV``) writes what the
    # session still runs in the background, for ``idle_ok``.
    stop_report: Optional[str] = None


@dataclass(frozen=True)
class Call:
    """An MCP tool call as the session's MCP config names it, with its arguments whole (an item's
    ``input`` may be cut)."""
    server: str
    tool: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class Patch:
    """One file an edit changes, from a backend that has the diff itself (Codex): ``diff`` is its
    unified diff from the first ``@@`` on, an added or deleted file's included. Paths as the item's
    ``files`` have them."""
    path: str
    diff: str
    status: str = "modified"  # added, modified, deleted or renamed
    old_path: Optional[str] = None


@dataclass
class Item:
    """One step of a turn as the backend reports it: once when it starts, again when it ends (with
    ``status`` no longer ``in_progress``)."""
    key: str  # the backend's own id for it, never on the wire
    kind: str
    status: str = "in_progress"
    fields: dict[str, Any] = field(default_factory=dict)  # the kind's protocol fields: tool, title, text, …
    call: Optional[Call] = None  # an MCP tool call's, at least on its start
    # An edit's change, on its start, for its diff: the backend's own (``patches``), or the tool's input
    # whole (``arguments``: Claude Code's Edit, MultiEdit or Write), applied to the file on disk.
    patches: tuple[Patch, ...] = ()
    arguments: Optional[dict[str, Any]] = None


@dataclass(frozen=True)
class Delta:
    """A running item grew: its text or plan so far, or new command output (``output``: only what is
    new, line feeds as the command wrote them)."""
    key: str
    text: Optional[str] = None
    steps: Optional[list[dict[str, Any]]] = None
    output: Optional[str] = None


# What the log shows for a secret answer, in its message and on its settled card.
SECRET = "••••••"


@dataclass(frozen=True)
class Question:
    """One question of a ``question`` ask, its options plain labels (no "(Recommended)" mark)."""
    question: str
    options: tuple[str, ...] = ()
    multi_select: bool = False
    recommended: Optional[int] = None  # the index of the option the agent suggests
    id: Optional[str] = None  # the backend's own id for it, when its answers are keyed by id (Codex)
    secret: bool = False  # its answer (a password, a key) only reaches the backend: never logged or kept


@dataclass(frozen=True)
class Ask:
    """The backend waits on the person: an approval (``kind`` command, edit, tool, permission or plan)
    or ``question`` (a series of questions answered together)."""
    key: str
    kind: str
    item: Optional[str] = None  # the item it holds, by its backend key
    text: str = ""  # the reason the backend gave; for a tool, the tool's name
    command: Optional[str] = None
    cwd: Optional[str] = None
    files: tuple[dict[str, Any], ...] = ()  # an edit's {path, old_path?, added, removed}
    questions: tuple[Question, ...] = ()
    # What a session grant of this ask covers, in the backend's own rule shape: a later ask with the
    # same ``grant`` is covered by it (Codex: the exact command; Claude Code: its first word; a tool:
    # its name). None: a session grant of it covers nothing more. Edits are matched by their files.
    grant: Optional[str] = None


@dataclass(frozen=True)
class Approve:
    """Allow what an ask holds. ``session``: the backend's own grant for its session too (Codex's
    ``acceptForSession``, a Claude Code session rule). For a plan: leave plan mode and start, in the
    thread's approval mode."""
    session: bool = False


@dataclass(frozen=True)
class Deny:
    """Refuse what an ask holds, or decline its questions. ``message``: what the person typed instead,
    for the agent to do (Claude Code: the deny's message; Codex: decline, then steer the text)."""
    message: Optional[str] = None


@dataclass(frozen=True)
class Answers:
    """A ``question`` ask answered: per question, in order, the picked labels or the typed text (one
    string), empty when the person skipped it."""
    answers: tuple[tuple[str, ...], ...]


Decision = Approve | Deny | Answers


@dataclass(frozen=True)
class Accepted:
    """The backend took the input ``send`` gave under ``key`` into a turn: the running one, or one it
    starts with it (a steer that missed its turn, which the backend runs as its own next turn)."""
    key: str


@dataclass(frozen=True)
class TurnEnd:
    status: str  # completed, interrupted or failed
    error: Optional[str] = None
    usage: Optional[dict[str, int]] = None  # {used, window}: the context after the turn


@dataclass(frozen=True)
class Exit:
    """The session's process ended, by ``close`` or by itself; ``error`` is why, as one sentence,
    when the backend knows."""
    error: Optional[str] = None


Event = Item | Delta | Ask | Accepted | TurnEnd | Exit
Sink = Callable[[Event], None]


class BackendSession(Protocol):
    """One thread's live session with its backend."""

    @property
    def session_id(self) -> Optional[str]:
        """The backend's own id for the session, once it has one: what resumes it."""

    async def send(self, key: str, text: str) -> None:
        """Start a turn with ``text``, or hand it to the running turn (steer); ``Accepted(key)`` follows
        once it is in a turn. Returns once the backend has it, not when a turn takes it."""

    async def interrupt(self) -> None:
        """End the running turn; it ends ``interrupted``."""

    async def answer(self, ask: str, decision: Decision) -> None:
        """Answer an ``Ask`` by its key, in the backend's own terms."""

    async def apply(self, spec: SessionSpec) -> None:
        """Changed settings, from the next turn on."""

    async def close(self) -> None:
        """End the session's process; the session itself stays resumable."""

    def idle_ok(self) -> bool:
        """Whether nothing of the session runs in the background, so closing it while idle loses nothing."""


class Backend(Protocol):
    id: str
    name: str
    # Whether the backend asks itself to leave plan mode (Claude Code's ExitPlanMode, an ``Ask`` of kind
    # ``plan``). One that only ends a plan-mode turn with its plan (Codex) gets the server's own ask.
    asks_to_leave_plan: bool

    async def probe(self) -> BackendInfo:
        """Whether a turn can run here now, and with which models, efforts and approval modes."""

    async def open(self, spec: SessionSpec, sink: Sink) -> BackendSession:
        """Start (or resume) a thread's session; what happens in it goes to ``sink``, on the loop."""

    async def stop(self) -> None:
        """The server stops, its sessions closed: end what the backend shares between them (Codex's
        app-server). The next use starts it again."""
