"""What a command that only looks does, as an ``execute`` item's ``actions``: each file it reads,
folder it lists and search it runs, in order, or None when any part of it may do something else.
Codex's app-server classifies its own commands (``commandActions``); Claude Code's are classified here,
over the same commands.
"""

from __future__ import annotations

import os
import re
import shlex
from typing import Any, Callable, Iterable, Optional

Relative = Callable[[str], str]

_CODEX = {"read": "read", "listFiles": "list", "search": "search"}
_JOINS = frozenset({"&&", "||", ";", "|"})
# On a pipe's later stages these only shape what the first one printed.
_FILTERS = frozenset({"head", "tail", "wc", "sort", "uniq", "cut", "tr", "nl", "column"})
_READERS = frozenset({"cat", "head", "tail", "nl"})
# Options whose value is the next word, by command (a value attached with "=" is one word).
_VALUED = {
    "head": {"-n", "-c"}, "tail": {"-n", "-c"}, "nl": {"-b", "-s", "-w", "-v", "-i"},
    "tree": {"-L", "-I", "-P", "--filelimit"},
    "rg": {"-e", "--regexp", "-f", "--file", "-g", "--glob", "--iglob", "-t", "--type", "-T", "--type-not",
           "-A", "--after-context", "-B", "--before-context", "-C", "--context", "-m", "--max-count",
           "-d", "--max-depth", "-M", "--max-columns", "-j", "--threads", "-E", "--encoding"},
    "grep": {"-e", "--regexp", "-f", "--file", "-A", "--after-context", "-B", "--before-context", "-C",
             "--context", "-m", "--max-count", "--include", "--exclude", "--exclude-dir"},
}
_PATTERNS = {"-e", "--regexp"}
_PATTERN_FILES = {"-f", "--file"}
# find's tests that name what it looks for, and its actions that do more than print.
_FIND_QUERIES = frozenset({"-name", "-iname", "-path", "-ipath", "-regex", "-iregex"})
_FIND_ACTIONS = frozenset({"-exec", "-execdir", "-ok", "-okdir", "-delete", "-fprint", "-fprint0", "-fprintf",
                           "-fls"})
# sed -n with nothing but a print of lines: "40p", "1,40p", "$p".
_SED_PRINT = re.compile(r"(\d+|\$)(,(\d+|\$))?p")


def from_codex(actions: Any, relative: Relative) -> Optional[list[dict[str, str]]]:
    """Codex's ``commandActions`` as the wire has them; None when any is unknown."""
    shown = []
    for action in actions or ():
        kind = _CODEX.get(action.get("type"))
        if kind is None:
            return None
        shown.append(_action(kind, relative, action.get("name"), action.get("path"), action.get("query")))
    return shown or None


def classify(command: str, relative: Relative) -> Optional[list[dict[str, str]]]:
    """``command``'s actions, when every part of it only reads, lists or searches."""
    # Command substitution runs something else; a line break starts another command.
    if "`" in command or "$(" in command or "\n" in command:
        return None
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        words = list(lexer)
    except ValueError:
        return None
    actions: list[dict[str, str]] = []
    part: list[str] = []
    piped = False
    for word in [*words, ";"]:
        if word not in _JOINS:
            if word and set(word) <= set("();<>&|"):  # a redirection, a subshell, a background job
                return None
            part.append(word)
            continue
        if part:
            found = _part(part, piped, relative)
            if found is None:
                return None
            actions += found
        part, piped = [], word == "|"
    return actions or None


def _part(words: list[str], piped: bool, relative: Relative) -> Optional[list[dict[str, str]]]:
    name, arguments = os.path.basename(words[0]), words[1:]
    if name == "cd" or (piped and name in _FILTERS):
        return []
    if name == "find":
        return _find(arguments, relative)
    options, operands = _split(name, arguments)
    if options is None:
        return None
    flags = {option for option, _ in options}
    if name in _READERS or name == "sed":
        if name == "sed":
            if flags != {"-n"} or not operands or not _SED_PRINT.fullmatch(operands[0]):
                return None
            operands = operands[1:]
        if not operands or "-" in operands:
            return None  # it reads what is piped in
        return [_action("read", relative, os.path.basename(path), path) for path in operands]
    if name in ("ls", "tree"):
        return [_action("list", relative, path=path) for path in operands] or [_action("list", relative)]
    if name in ("rg", "grep"):
        if "--files" in flags:
            return [_action("list", relative, path=path) for path in operands] or [_action("list", relative)]
        patterns = [value for option, value in options if option in _PATTERNS]
        if not patterns and not flags & _PATTERN_FILES:
            if not operands:
                return None
            patterns, operands = [operands[0]], operands[1:]
        query = patterns[0] if len(patterns) == 1 else None
        return [_action("search", relative, path=operands[0] if len(operands) == 1 else None, query=query)]
    return None


def _split(name: str, arguments: list[str]) -> tuple[Optional[list[tuple[str, Optional[str]]]], list[str]]:
    """A simple command's options in order (each with its value, if it takes one) and its operands."""
    valued = _VALUED.get(name, set())
    options: list[tuple[str, Optional[str]]] = []
    operands: list[str] = []
    words = iter(arguments)
    for word in words:
        if word == "--":
            operands += words
        elif word.startswith("-") and word != "-":
            if word in valued:
                value = next(words, None)
                if value is None:
                    return None, []
                options.append((word, value))
            else:
                options.append((word.split("=", 1)[0], None))
        else:
            operands.append(word)
    return options, operands


def _find(arguments: list[str], relative: Relative) -> Optional[list[dict[str, str]]]:
    start = next((index for index, word in enumerate(arguments) if word.startswith(("-", "(", "!"))), len(arguments))
    paths, tests = arguments[:start], arguments[start:]
    if any(test in _FIND_ACTIONS for test in tests):
        return None
    query = next((tests[index + 1] for index, test in enumerate(tests[:-1]) if test in _FIND_QUERIES), None)
    path = paths[0] if len(paths) == 1 else None
    return [_action("search" if query else "list", relative, path=path, query=query)]


def _action(kind: str, relative: Relative, name: Optional[str] = None, path: Optional[str] = None,
            query: Optional[str] = None) -> dict[str, str]:
    fields: Iterable[tuple[str, Optional[str]]] = (
        ("name", name), ("path", relative(path) if path and os.path.isabs(path) else path), ("query", query))
    return {"kind": kind, **{key: value for key, value in fields if value}}
