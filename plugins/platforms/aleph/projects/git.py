"""A project's folder as git sees it: its changes (``changes.get``, ``project.changes``), a changed
file's diff (``diff.get``) and an edit item's diff.

Every git runs as an asyncio subprocess with a timeout, under ``LC_ALL=C`` (its messages are matched
in English), with prompts off and ``--no-optional-locks``: looking at a working tree must never take
the index lock from under the agent's own git running in it.
"""

from __future__ import annotations

import asyncio
import difflib
import os
import re
import signal
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from plugins.platforms.aleph.projects.backend import Patch
from plugins.platforms.aleph.projects.process import signal_group

GIT_SECONDS = 30.0
MAX_FILES = 2000
MAX_DIFF_LINES = 20000
MAX_DIFF_BYTES = 2 * 1024 * 1024
MAX_LINE_CHARS = 10000
MAX_EDIT_BYTES = 8 * 1024 * 1024  # a file larger is not read to show an edit of it
CONTEXT = 3
FULL_CONTEXT = 1_000_000  # the whole file: a longer one is past MAX_DIFF_LINES anyway
_ENV = {"LC_ALL": "C", "GIT_TERMINAL_PROMPT": "0"}
# What makes git's diff output its own, whatever the user's config says.
_DIFF = ("--no-color", "--no-ext-diff", "--no-textconv")
_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_STATUS = {"A": "added", "C": "added", "D": "deleted", "R": "renamed"}
_COUNTED: dict[tuple[str, int, int], tuple[int, bool]] = {}


class GitError(Exception):
    """git failed or ran out of time: the message is the line that says why, ``stderr`` all it said."""

    def __init__(self, message: str, *, stderr: str = "", timed_out: bool = False) -> None:
        super().__init__(message)
        self.stderr = stderr
        self.timed_out = timed_out


class NotGit(Exception):
    """The folder is no git working tree."""


# ── running git ───────────────────────────────────────────────────────────────────────────────


def _why(stderr: str) -> str:
    """The first line of git's (or ssh's) that says what went wrong; what follows it is git's
    generic afterword ("Could not read from remote repository.")."""
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    said = next((line for line in lines if line.lower().startswith(("fatal:", "error:", "ssh:"))), None)
    if said is None:
        return lines[-1] if lines else "git failed"
    return said.split(":", 1)[1].strip() if not said.lower().startswith("ssh:") else said


async def _run(argv: list[str], *, env: Optional[dict[str, str]] = None, timeout: float = GIT_SECONDS,
               limit: Optional[int] = None, ok: tuple[int, ...] = (0,)) -> Optional[bytes]:
    """``argv``'s output; None when it passed ``limit`` bytes (it is stopped then)."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv, env={**os.environ, **_ENV, **(env or {})}, stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, start_new_session=True)
    except OSError as exc:
        raise GitError(f"git cannot run: {exc}") from exc

    async def read() -> Optional[bytes]:
        chunks, size = [], 0
        while chunk := await proc.stdout.read(1 << 16):
            size += len(chunk)
            if limit is not None and size > limit:
                signal_group(proc, signal.SIGKILL)
                return None
            chunks.append(chunk)
        return b"".join(chunks)

    async def finish() -> tuple[Optional[bytes], bytes]:
        out, err = await asyncio.gather(read(), proc.stderr.read())
        await proc.wait()
        return out, err

    try:
        out, err = await asyncio.wait_for(finish(), timeout)
    except TimeoutError:
        signal_group(proc, signal.SIGKILL)
        await proc.wait()
        raise GitError(f"git took longer than {timeout:.0f} seconds", timed_out=True) from None
    except asyncio.CancelledError:
        signal_group(proc, signal.SIGKILL)
        raise
    if out is not None and proc.returncode not in ok:
        stderr = err.decode("utf-8", "replace")
        raise GitError(_why(stderr), stderr=stderr)
    return out


async def _git(folder: str, *args: str, **kwargs: Any) -> Optional[bytes]:
    # Paths given are literal (a file may be named like a glob).
    return await _run(["git", "--no-optional-locks", "--literal-pathspecs", "-C", folder, *args], **kwargs)


async def _text(folder: str, *args: str, ok: tuple[int, ...] = (0,)) -> str:
    return (await _git(folder, *args, ok=ok) or b"").decode("utf-8", "surrogateescape").strip()


# ── changes ───────────────────────────────────────────────────────────────────────────────────


@dataclass
class _Change:
    path: str
    status: str
    old_path: Optional[str] = None
    added: int = 0
    removed: int = 0
    binary: bool = False
    untracked: bool = False

    def wire(self) -> dict[str, Any]:
        return {"path": self.path, "old_path": self.old_path, "status": self.status, "added": self.added,
                "removed": self.removed, "binary": self.binary}


async def _repo(folder: str) -> None:
    try:
        inside = await _text(folder, "rev-parse", "--is-inside-work-tree")
    except GitError as exc:
        raise NotGit(str(exc)) from exc
    if inside != "true":
        raise NotGit(f"{folder} is no git working tree")


async def _head(folder: str) -> str:
    """``HEAD``, or the empty tree on a branch with no commit yet."""
    head = await _text(folder, "rev-parse", "-q", "--verify", "HEAD^{commit}", ok=(0, 1))
    return head or await _text(folder, "hash-object", "-t", "tree", "--stdin")


async def _base(folder: str) -> Optional[tuple[str, str]]:
    """``(name, merge-base)``: where the branch left the remote's default branch, else ``main`` or
    ``master``; None when there is none (no such branch, or no common history)."""
    remote = await _text(folder, "symbolic-ref", "-q", "refs/remotes/origin/HEAD", ok=(0, 1))
    for ref in [*([remote] if remote else []), "refs/heads/main", "refs/heads/master"]:
        merge_base = await _text(folder, "merge-base", "HEAD", ref, ok=(0, 1, 128))
        if merge_base:
            return re.sub(r"^refs/(?:heads|remotes/[^/]+)/", "", ref), merge_base
    return None


def _split(out: Optional[bytes]) -> list[str]:
    return [os.fsdecode(token) for token in (out or b"").split(b"\0") if token]


def _counted(path: str) -> tuple[int, bool]:
    """``(lines, binary)`` of an untracked file, as git counts a new file's lines; remembered while the
    file keeps its size and time, since a running turn is looked at every 2 s."""
    try:
        info = os.lstat(path)
    except OSError:
        return 0, False
    if not stat.S_ISREG(info.st_mode):
        return (1, False) if stat.S_ISLNK(info.st_mode) else (0, False)
    key = (path, info.st_size, info.st_mtime_ns)
    if key not in _COUNTED:
        lines, last = 0, b"\n"
        try:
            with open(path, "rb") as file:
                if b"\0" in file.read(8000):  # git's own test for a binary file
                    return 0, True
                file.seek(0)
                while chunk := file.read(1 << 20):
                    lines += chunk.count(b"\n")
                    last = chunk[-1:]
        except OSError:
            return 0, False
        if len(_COUNTED) > 20000:
            _COUNTED.clear()
        _COUNTED[key] = (lines + (last != b"\n"), False)
    return _COUNTED[key]


async def _changes(folder: str, rev: str) -> list[_Change]:
    """The working tree against ``rev``, staged or not, untracked files as added; by path."""
    statuses, numstat, untracked = await asyncio.gather(
        _git(folder, "diff", rev, "--name-status", "-z", "-M", "--relative", *_DIFF),
        _git(folder, "diff", rev, "--numstat", "-z", "-M", "--relative", *_DIFF),
        _git(folder, "ls-files", "--others", "--exclude-standard", "-z"))
    changes: dict[str, _Change] = {}
    tokens = _split(statuses)
    index = 0
    while index < len(tokens):
        code = tokens[index]
        if code[:1] in ("R", "C"):
            old, path = tokens[index + 1], tokens[index + 2]
            index += 3
        else:
            old, path = None, tokens[index + 1]
            index += 2
        status = _STATUS.get(code[:1], "modified")
        changes[path] = _Change(path, status, old if status == "renamed" else None)
    tokens = _split(numstat)
    index = 0
    while index < len(tokens):
        added, removed, path = tokens[index].split("\t", 2)
        index += 1
        if not path:  # a rename: its old and new paths follow
            path = tokens[index + 1]
            index += 2
        change = changes.get(path)
        if change is not None:
            change.binary = added == "-"
            change.added, change.removed = (0, 0) if change.binary else (int(added), int(removed))
    paths = [path for path in _split(untracked) if path not in changes]
    counts = await asyncio.to_thread(lambda: [_counted(os.path.join(folder, path)) for path in paths])
    for path, (lines, binary) in zip(paths, counts):
        changes[path] = _Change(path, "added", added=lines, binary=binary, untracked=True)
    return [changes[path] for path in sorted(changes)]


async def _scope(folder: str, scope: str) -> tuple[Optional[tuple[str, str]], Optional[str]]:
    """``(base, rev)``: the branch's base, and what ``scope`` compares the working tree with."""
    await _repo(folder)
    base = await _base(folder)
    rev = await _head(folder) if scope == "uncommitted" else base[1] if base else None
    return base, rev


async def changes(folder: str, scope: str) -> dict[str, Any]:
    """``changes.get``'s answer; ``NotGit`` for a folder that is no repository."""
    base, rev = await _scope(folder, scope)
    found = await _changes(folder, rev) if rev else []
    return {"scope": scope, "base": base[0] if base else None,
            "files": [change.wire() for change in found[:MAX_FILES]], "more": len(found) > MAX_FILES,
            "added": sum(change.added for change in found), "removed": sum(change.removed for change in found)}


async def summary(folder: str) -> dict[str, Any]:
    """``project.changes``'s fields: the uncommitted changes counted, or ``git`` false."""
    try:
        await _repo(folder)
    except NotGit:
        return {"git": False, "branch": None, "files": None, "added": None, "removed": None}
    branch = await _text(folder, "symbolic-ref", "-q", "--short", "HEAD", ok=(0, 1))
    found = await _changes(folder, await _head(folder))
    return {"git": True, "branch": branch or None, "files": len(found),
            "added": sum(change.added for change in found), "removed": sum(change.removed for change in found)}


# ── diffs ─────────────────────────────────────────────────────────────────────────────────────


def hunks(diff: str) -> tuple[list[dict[str, Any]], bool]:
    """A unified diff's hunks, each line keeping its leading `` ``, ``+`` or ``-`` and cut at
    ``MAX_LINE_CHARS``, and whether git called it binary."""
    found: list[dict[str, Any]] = []
    binary = False
    for line in diff.split("\n"):
        match = _HUNK.match(line) if line.startswith("@@") else None
        if match is not None:
            old_start, old_lines, new_start, new_lines = match.groups()
            found.append({"old_start": int(old_start), "old_lines": int(old_lines or 1), "new_start": int(new_start),
                          "new_lines": int(new_lines or 1), "lines": []})
        elif not found:
            binary = binary or line.startswith(("Binary files ", "GIT binary patch"))
        elif line[:1] in (" ", "+", "-"):
            found[-1]["lines"].append(line[:MAX_LINE_CHARS])
    return found, binary


def _file(path: str, status: str, old_path: Optional[str], diff: Optional[str], *, binary: bool = False,
          counts: Optional[tuple[int, int]] = None) -> dict[str, Any]:
    """``diff.get``'s entry for one file; ``diff`` None: past the byte limit. Counts come from the
    hunks unless given (from git's own count)."""
    found, binary = hunks(diff) if diff is not None and not binary else ([], binary)
    lines = [line for hunk in found for line in hunk["lines"]]
    if counts is None:
        counts = (0, 0) if binary or diff is None else (sum(line[0] == "+" for line in lines),
                                                        sum(line[0] == "-" for line in lines))
    too_large = not binary and (diff is None or len(lines) > MAX_DIFF_LINES)
    return {"path": path, "old_path": old_path, "status": status, "binary": binary, "added": counts[0],
            "removed": counts[1], "too_large": too_large, "hunks": [] if too_large else found}


async def file_diff(folder: str, scope: str, path: str, full: bool) -> Optional[dict[str, Any]]:
    """``diff.get``'s entry for a file of ``changes.get``; None when it is not among them."""
    _, rev = await _scope(folder, scope)
    change = next((change for change in (await _changes(folder, rev) if rev else []) if change.path == path), None)
    if change is None:
        return None
    context = f"-U{FULL_CONTEXT if full else CONTEXT}"
    if change.binary:
        out: Optional[bytes] = b""
    elif change.untracked:
        out = await _git(folder, "diff", "--no-index", *_DIFF, context, "--", os.devnull, path,
                         limit=MAX_DIFF_BYTES, ok=(0, 1))
    else:
        out = await _git(folder, "diff", rev, "-M", "--relative", *_DIFF, context, "--",
                         *([change.old_path] if change.old_path else []), path, limit=MAX_DIFF_BYTES)
    return _file(path, change.status, change.old_path, None if out is None else out.decode("utf-8", "replace"),
                 binary=change.binary, counts=(change.added, change.removed))


def _lines(text: str) -> list[str]:
    lines = text.split("\n")
    return lines[:-1] if lines[-1] == "" else lines


def _replaced(text: str, old: Any, new: Any, every: bool) -> Optional[str]:
    if not isinstance(old, str) or not isinstance(new, str):
        return None
    if not old:  # Claude Code's Edit with nothing to replace makes a new file
        return new if not text else None
    if old not in text:
        return None
    return text.replace(old, new) if every else text.replace(old, new, 1)


def edit_files(folder: str, arguments: Optional[dict[str, Any]], patches: tuple[Patch, ...]) -> list[dict[str, Any]]:
    """Blocking. ``diff.get``'s files for an edit as it starts: the backend's own ``patches``, else
    Claude Code's Edit, MultiEdit or Write ``arguments`` applied to the file on disk (an edit that no
    longer applies shows its own strings). Empty for any other tool's input."""
    if patches:
        return [_file(patch.path, patch.status, patch.old_path, patch.diff) for patch in patches]
    target = (arguments or {}).get("file_path")
    if not isinstance(target, str) or not target:
        return []
    real = target if os.path.isabs(target) else os.path.join(folder, target)
    inside = os.path.normpath(real).startswith(os.path.normpath(folder).rstrip(os.sep) + os.sep)
    path = os.path.relpath(real, folder) if inside else real
    try:
        with open(real, "rb") as file:
            data = file.read(MAX_EDIT_BYTES + 1)
        exists = True
    except FileNotFoundError:
        data, exists = b"", False
    except OSError:
        return []
    status = "modified" if exists else "added"
    if len(data) > MAX_EDIT_BYTES:
        return [_file(path, status, None, None)]
    if b"\0" in data[:8000]:
        return [_file(path, status, None, None, binary=True)]
    before = data.decode("utf-8", "replace")
    if isinstance(arguments.get("content"), str):  # Write
        after: Optional[str] = arguments["content"]
    else:
        edits = arguments.get("edits") if isinstance(arguments.get("edits"), list) else [arguments]
        after = before
        for edit in edits:
            edit = edit if isinstance(edit, dict) else {}
            after = _replaced(after, edit.get("old_string"), edit.get("new_string"), bool(edit.get("replace_all")))
            if after is None:
                break
        if after is None:
            before = "\n".join(str(edit.get("old_string") or "") for edit in edits if isinstance(edit, dict))
            after = "\n".join(str(edit.get("new_string") or "") for edit in edits if isinstance(edit, dict))
    diff = "\n".join(difflib.unified_diff(_lines(before), _lines(after), n=CONTEXT, lineterm=""))
    return [_file(path, status, None, diff)]


