"""A project's requests (protocol "Projects: Requests"): what its backends wait on the person for, as
the chat's approval and question cards naming ``project`` and ``thread``.

One ``RequestDesk`` per project holds its open requests, its Always Allow rules (kept in the store:
they last as long as the project), and a timer per request: unanswered, it is declined (fail closed). It speaks only the neutral decisions of
``backend.py``; each backend says them in its own wire format. A Codex-style backend, which ends a
plan-mode turn with its plan and asks nothing, gets the server's own plan request here, answered by
starting the next turn.
"""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from plugins.platforms.aleph.projects.backend import (
    SECRET, Answers, Approve, Ask, BackendSession, Decision, Deny, rule_covers, rule_words)
from plugins.platforms.aleph.projects.service import now_iso
from plugins.platforms.aleph.server import INVALID_PARAMS, RpcError

if TYPE_CHECKING:
    from plugins.platforms.aleph.projects.service import Projects
    from plugins.platforms.aleph.projects.session import Session

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 3600  # seconds a request waits before it is declined
IMPLEMENT = "Implement the plan."
_ALLOW = {"label": "Allow", "choice": "once", "style": "primary"}
_ALWAYS = {"label": "Always Allow", "choice": "always", "style": "default"}
_DENY = {"label": "Don't Allow", "choice": "deny", "style": "destructive"}
_PLAN_CHOICES = ({"label": "Start Coding", "choice": "once", "style": "primary"},
                 {"label": "Keep Planning", "choice": "deny", "style": "default"})
_EDITS = "*"  # the edit rule: any file in the project's folder


def typed_answer(labels: list[str], multi: bool, text: str) -> tuple[tuple[str, ...], dict[str, Any]]:
    """A typed answer to a question: what the agent gets, and the card's ``{answer, choices}``. A
    label typed as it is, or its number from 1, is that choice; anything else is the text itself."""
    typed = text.strip()
    index = next((i for i, label in enumerate(labels) if label.casefold() == typed.casefold()), None)
    if index is None and typed.isdigit() and 1 <= int(typed) <= len(labels):
        index = int(typed) - 1
    if index is None:
        return (text,), {"answer": text, "choices": None}
    return (labels[index],), {"answer": labels[index], "choices": [index]}


@dataclass(eq=False)
class _Request:
    """An open request: an ask of a thread's backend, or the server's own plan request (no
    ``backend``). ``id`` is the card on show: its ``request_id``, or the current question's
    ``clarify_id`` in a series."""
    id: str
    session: Session
    ask: Ask
    backend: Optional[BackendSession]  # the session the ask came from: it alone can take the answer
    timer: Optional[asyncio.TimerHandle] = None
    index: int = 0  # the question on show
    answers: list[tuple[str, ...]] = field(default_factory=list)

    @property
    def thread(self) -> str:
        return self.session.thread

    @property
    def question(self) -> bool:
        return self.ask.kind == "question"


def _rule(ask: Ask) -> Optional[str]:
    """What Always Allow of ``ask`` would keep; None when it offers no Always Allow."""
    if ask.kind == "edit":
        return _EDITS
    return ask.rule if ask.kind in ("command", "tool") else None


class RequestDesk:
    """One project's open requests and Always Allow rules."""

    def __init__(self, projects: Projects, project: str) -> None:
        self._projects = projects
        self.project = project
        self._open: dict[str, _Request] = {}
        self._rules: Optional[set[tuple[str, str]]] = None  # (kind, rule), read from the store at first need
        self._tasks: set[asyncio.Task] = set()

    def _scoped(self, request: _Request, **payload: Any) -> dict[str, Any]:
        return {"project": self.project, "thread": request.thread, **payload}

    async def _log(self, event_type: str, payload: dict[str, Any]) -> None:
        await self._projects.adapter._log(event_type, payload)

    def waits(self, thread: str) -> bool:
        return any(request.thread == thread for request in self._open.values())

    def asks_secret(self, thread: str) -> bool:
        """Whether the question on show in ``thread`` takes a secret, which a typed message answers."""
        return any(request.thread == thread and request.question and request.ask.questions[request.index].secret
                   for request in self._open.values())

    def has(self, card_id: str) -> bool:
        return card_id in self._open

    # ── asking ─────────────────────────────────────────────────────────────────────────────────

    async def ask(self, session: Session, ask: Ask, *, server: bool = False) -> None:
        """Under the session's gate: a card for ``ask``, unless an Always Allow rule covers it. Never
        waits for the person; ``server``: the server's own plan request, answered by the server."""
        backend = None if server else session.backend
        if backend is not None and await self._covered(ask):
            await self._answer(backend, ask, Approve(session=True))
            return
        request = _Request(_card_id("q" if ask.kind == "question" else "r"), session, ask, backend)
        self._open[request.id] = request
        request.timer = asyncio.get_running_loop().call_later(
            self._projects.adapter.request_timeout, lambda: self._spawn(self._expire(request)))
        if request.question:
            await self._show_question(request)
            return
        payload = self._scoped(request, request_id=request.id, kind=ask.kind, item=session.item_id(ask.item),
                               text=ask.text)
        if ask.command is not None:
            payload.update(command=ask.command, cwd=ask.cwd)
        if ask.kind == "edit":
            payload["files"] = list(ask.files)
        payload["choices"] = list(_choices(ask))
        if _rule(ask) is not None:
            payload["always_scope"] = await self._scope_line(ask)
        payload["created_at"] = now_iso()
        await self._log("approval.request", payload)

    async def _show_question(self, request: _Request) -> None:
        questions = request.ask.questions
        question = questions[request.index]
        payload = self._scoped(request, clarify_id=request.id, question=question.question,
                               choices=list(question.options), multi_select=question.multi_select and bool(question.options))
        if question.recommended is not None:
            payload["recommended"] = question.recommended
        if question.secret:
            payload["secret"] = True
        if len(questions) > 1:
            payload.update(index=request.index, count=len(questions))
        payload["created_at"] = now_iso()
        await self._log("clarify.request", payload)

    async def _scope_line(self, ask: Ask) -> str:
        """``always_scope``: what Always Allow would cover, in a line under the choices."""
        project = await asyncio.to_thread(self._projects.tables.project, self.project)
        where = project.payload.get("name") or "this project"
        what = "edits" if ask.kind == "edit" else rule_words(ask.rule) if ask.kind == "command" else ask.text or ask.rule
        return f"Won't ask again for {what} in {where}"

    async def _known_rules(self) -> set[tuple[str, str]]:
        if self._rules is None:
            self._rules = await asyncio.to_thread(self._projects.tables.rules, self.project)
        return self._rules

    async def _covered(self, ask: Ask) -> bool:
        """Whether an Always Allow rule of the project covers ``ask``: a command its rule matches, the
        same tool, an edit of files all in the folder."""
        rules = await self._known_rules()
        if ask.kind == "edit":
            return ("edit", _EDITS) in rules and bool(ask.files) and await self._in_folder(ask.files)
        if ask.kind == "command" and ask.command is not None:
            return any(kind == "command" and rule_covers(rule, ask.command) for kind, rule in rules)
        return ask.kind == "tool" and ask.rule is not None and ("tool", ask.rule) in rules

    async def _in_folder(self, files: tuple[dict[str, Any], ...]) -> bool:
        project = await asyncio.to_thread(self._projects.tables.project, self.project)
        folder = os.path.realpath(self._projects.folder(project))

        def inside(path: Any) -> bool:
            real = os.path.realpath(os.path.join(folder, str(path)))
            return real.startswith(folder.rstrip(os.sep) + os.sep)
        return all(inside(one.get("path")) and (not one.get("old_path") or inside(one["old_path"])) for one in files)

    async def _keep_rule(self, ask: Ask) -> None:
        rule = _rule(ask)
        if rule is None:
            return
        await self._known_rules()
        self._rules.add((ask.kind, rule))
        await asyncio.to_thread(self._projects.tables.add_rule, self.project, ask.kind, rule)

    # ── answering ──────────────────────────────────────────────────────────────────────────────

    def choices(self, request_id: str) -> Optional[list[str]]:
        request = self._open.get(request_id)
        if request is None or request.question:
            return None
        return [choice["choice"] for choice in _choices(request.ask)]

    async def respond(self, request_id: str, choice: str) -> None:
        """``approval.respond`` for one of this project's approvals."""
        offered = self.choices(request_id)
        if offered is None:
            raise RpcError(INVALID_PARAMS, "no pending approval with that request_id")
        if choice not in offered:
            raise RpcError(INVALID_PARAMS, "choice is not one the request offered", {"choices": offered})
        request = self._take(request_id)
        ask, session = request.ask, request.session
        if choice == "deny":
            session.decline(ask.item)  # before anything awaits: the item may end meanwhile
            await self._log("approval.resolved", self._scoped(request, request_id=request.id, choice=choice, by="user"))
            await self._answer(request.backend, ask, Deny())
            return
        await self._log("approval.resolved", self._scoped(request, request_id=request.id, choice=choice, by="user"))
        if ask.kind == "plan":
            await self._projects.leave_plan(request.thread)
        if choice == "always":
            await self._keep_rule(ask)
        if request.backend is not None:
            await self._answer(request.backend, ask, Approve(session=choice == "always"))
        elif ask.kind == "plan":
            try:
                await session.submit(IMPLEMENT, cause="user")
            except Exception as exc:
                logger.warning("Aleph: thread %s could not start the plan: %s", request.thread, exc)

    def question(self, clarify_id: str) -> Optional[tuple[list[str], bool]]:
        """The labels of the question on show as ``clarify_id``, and whether it takes several."""
        request = self._open.get(clarify_id)
        if request is None or not request.question:
            return None
        question = request.ask.questions[request.index]
        return list(question.options), question.multi_select and bool(question.options)

    async def pick(self, clarify_id: str, picks: Optional[list[int]], message_id: Optional[str]) -> None:
        """``clarify.respond``, its picks checked against ``question``: the labels picked (logged as the
        person's message), or None for a skip."""
        request = self._open[clarify_id]
        labels = list(request.ask.questions[request.index].options)
        if picks is None:
            await self._answered(request, (), {"answer": None, "choices": None})
            return
        answer = tuple(labels[i] for i in picks)
        outcome = {"answer": ", ".join(answer), "choices": list(picks)}
        event = await self._projects.adapter._log_answer(self._scoped(request), message_id, outcome["answer"])
        await self._answered(request, answer, outcome, (message_id, event.seq))

    async def typed(self, session: Session, text: str, message: tuple[str, int]) -> bool:
        """Under the session's gate, the person's ``message`` (id, seq) just logged in ``session``'s
        thread as ``text``: it answers the thread's open question, or declines its open approvals,
        the text going to the agent as what to do instead. True when it was taken so (it starts no
        turn); a request of another thread is not touched."""
        mine = [request for request in self._open.values() if request.thread == session.thread]
        question = next((request for request in mine if request.question), None)
        if question is not None:
            labels, multi = self.question(question.id)
            answer, outcome = typed_answer(labels, multi, text)
            if question.ask.questions[question.index].secret:
                outcome = {"answer": SECRET, "choices": None}
            await self._answered(question, answer, outcome, message)
            return True
        took = False
        for request in mine:
            self._take(request.id)
            await self._log("approval.resolved", self._scoped(request, request_id=request.id, choice="deny", by="user"))
            session.decline(request.ask.item)
            if request.backend is not None:
                # The text goes once, with the first refusal: the agent reads it as what to do instead.
                await self._answer(request.backend, request.ask, Deny(None if took else text))
                took = True
        if took:
            await session.read(*message)
        return took

    async def _answered(self, request: _Request, answer: tuple[str, ...], outcome: dict[str, Any],
                        message: Optional[tuple[str, int]] = None) -> None:
        """The question on show answered (by ``message``, unless skipped): its card settles, and the
        next of the series shows, or the backend gets every answer together."""
        if self._open.get(request.id) is not request:
            return  # settled meanwhile (it timed out while the answer was being logged)
        await self._log("clarify.resolved", self._scoped(request, clarify_id=request.id, **outcome, by="user"))
        if message is not None:
            await request.session.read(*message)
        request.answers.append(answer)
        if request.index + 1 < len(request.ask.questions):
            del self._open[request.id]
            request.id, request.index = _card_id("q"), request.index + 1
            self._open[request.id] = request
            await self._show_question(request)
            return
        self._take(request.id)
        await self._answer(request.backend, request.ask, Answers(tuple(request.answers)))

    async def _answer(self, backend: Optional[BackendSession], ask: Ask, decision: Decision) -> None:
        if backend is None:
            return
        try:
            await backend.answer(ask.key, decision)
        except Exception as exc:
            # Its process is going: its exit ends the turn, and the turn's end settles the rest.
            logger.warning("Aleph: answering %s %s failed: %s", ask.kind, ask.key, exc)

    # ── settling without the person ────────────────────────────────────────────────────────────

    def _take(self, card_id: str) -> _Request:
        request = self._open.pop(card_id)
        if request.timer is not None:
            request.timer.cancel()
        self._projects.bus.wake(request.thread)  # what was held for the person may go now
        return request

    async def _expire(self, request: _Request) -> None:
        """Unanswered for the whole timeout: declined, fail closed. The server's own plan request
        leaves the thread planning."""
        if self._open.get(request.id) is not request:
            return
        self._take(request.id)
        request.session.decline(request.ask.item)
        await self._settle(request, "timeout")
        await self._answer(request.backend, request.ask, Deny())

    async def _settle(self, request: _Request, by: str) -> None:
        if request.question:
            await self._log("clarify.resolved", self._scoped(request, clarify_id=request.id, answer=None,
                                                             choices=None, by="timeout"))
        else:
            await self._log("approval.resolved", self._scoped(request, request_id=request.id, choice=None, by=by))

    async def cancel(self, thread: str, *, backend_only: bool = False) -> None:
        """The requests of ``thread`` no one can answer any more (its turn or process ended, or it
        closed): an approval settles ``cancelled``, a question as a chat's does, ``timeout``.
        ``backend_only``: the server's own plan request stays, needing no process."""
        for request in [request for request in self._open.values() if request.thread == thread
                        and not (backend_only and request.backend is None)]:
            self._take(request.id)
            await self._settle(request, "cancelled")

    def stop(self) -> None:
        """The server stops or the project goes: forget everything; the next start settles the cards."""
        for request in self._open.values():
            if request.timer is not None:
                request.timer.cancel()
        self._open.clear()
        for task in self._tasks:
            task.cancel()

    def _spawn(self, coroutine) -> None:
        task = asyncio.create_task(coroutine)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)


def _choices(ask: Ask) -> tuple[dict[str, str], ...]:
    if ask.kind == "plan":
        return _PLAN_CHOICES
    return (_ALLOW, _ALWAYS, _DENY) if _rule(ask) is not None else (_ALLOW, _DENY)


def _card_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(8)}"
