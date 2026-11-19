"""What a command that only looks does (plugins/platforms/aleph/projects/actions.py): an ``execute``
item's ``actions``, classified for Claude Code, Codex's as its app-server gives them."""

from __future__ import annotations

from plugins.platforms.aleph.projects.actions import classify, from_codex


def _relative(path):
    return path.removeprefix("/work/")


def test_a_command_that_only_reads_lists_or_searches_says_what_in_order():
    for command, expected in (
        ("ls -la", [{"kind": "list"}]),
        ("tree -L 2 docs", [{"kind": "list", "path": "docs"}]),
        ("cat /work/a.py b.py", [{"kind": "read", "name": "a.py", "path": "a.py"},
                                 {"kind": "read", "name": "b.py", "path": "b.py"}]),
        ("sed -n '1,40p' notes.txt", [{"kind": "read", "name": "notes.txt", "path": "notes.txt"}]),
        ("head -n 5 README.md", [{"kind": "read", "name": "README.md", "path": "README.md"}]),
        ('rg -n "def main" src | head -20', [{"kind": "search", "path": "src", "query": "def main"}]),
        ("cd src && grep -rn -A 3 TODO .", [{"kind": "search", "path": ".", "query": "TODO"}]),
        ("grep -e a -e b f", [{"kind": "search", "path": "f"}]),
        ("rg --files src", [{"kind": "list", "path": "src"}]),
        ('find . -name "*.py" -type f', [{"kind": "search", "path": ".", "query": "*.py"}]),
        ("find src -type d", [{"kind": "list", "path": "src"}]),
        ("cat setup.py | grep version", [{"kind": "read", "name": "setup.py", "path": "setup.py"},
                                         {"kind": "search", "query": "version"}]),
    ):
        assert classify(command, _relative) == expected, command


def test_a_command_that_may_do_anything_else_has_none():
    for command in ("rm -rf build", "ls > out", "rg foo 2>/dev/null", "cat $(ls)", "cat `ls`", "echo hi",
                    "git status", "sed -i s/a/b/ f", "sed -n 's/a/b/w out' f", "find . -name x -delete",
                    "find . -exec rm {} ;", "cat", "ls | xargs rm", "ls && rm x", "FOO=1 cat a", "ls &",
                    "cat a\nrm b", "grep", "cat 'unclosed"):
        assert classify(command, _relative) is None, command


def test_codexs_own_actions_are_forwarded_unless_any_is_unknown():
    assert from_codex([{"type": "read", "command": "cat a.py", "name": "a.py", "path": "/work/a.py"},
                       {"type": "listFiles", "command": "ls", "path": None},
                       {"type": "search", "command": "rg x src", "query": "x", "path": "src"}], _relative) == [
        {"kind": "read", "name": "a.py", "path": "a.py"}, {"kind": "list"},
        {"kind": "search", "path": "src", "query": "x"}]
    assert from_codex([{"type": "listFiles", "command": "ls", "path": None},
                       {"type": "unknown", "command": "rm x"}], _relative) is None
    assert from_codex([], _relative) is None and from_codex(None, _relative) is None
