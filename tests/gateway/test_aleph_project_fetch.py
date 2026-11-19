"""Projects on Aleph, what is fetched on demand: a project's changes against HEAD or its base, a
changed file's diff within its limits, an edit's diff kept from before its approval, a command's
output paged and followed, and ``project.changes`` sent when the working tree's summary changed."""

from __future__ import annotations

import asyncio
import os
import subprocess

import pytest
import pytest_asyncio

from plugins.platforms.aleph.projects import output
from plugins.platforms.aleph.projects.backend import Ask, Delta, Item, Patch, TurnEnd
from plugins.platforms.aleph.server import RpcError
from tests.gateway.aleph_backend import project_adapter, until
from tests.gateway.aleph_phone import connect

PROJECT = "C41A0000-0000-4000-8000-000000000001"
OTHER = "C41A0000-0000-4000-8000-000000000002"


def git(folder, *args):
    return subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", "-C", str(folder),
                           *args], check=True, capture_output=True, text=True).stdout


def lines(count, change=None):
    return "".join(f"{change.get(n, n) if change else n}\n" for n in range(1, count + 1))


@pytest_asyncio.fixture
async def adapter(tmp_path, monkeypatch):
    # The machine's own git config stays out: the server's git and the test's run alike.
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    async with project_adapter(tmp_path) as instance:
        web = instance.root / "web"
        git(web, "init", "-q", "-b", "main")
        (web / "count.txt").write_text(lines(20))
        (web / "old.txt").write_text("moved\n")
        git(web, "add", ".")
        git(web, "commit", "-q", "-m", "first")
        instance.web = web
        await instance.handle_call("project.create", {
            "id": PROJECT, "name": "web", "folder": str(web), "backend": "claude", "model": None,
            "effort": None, "approval": "ask"})
        yield instance


async def _error(adapter, method, params):
    with pytest.raises(RpcError) as caught:
        await adapter.handle_call(method, params)
    return caught.value


async def _frames_until(phone, done):
    frames = []
    while not frames or not done(frames[-1]):
        frames.append((await phone.event())["params"])
    return frames


def _changes(files):
    return lambda frame: frame["type"] == "project.changes" and frame["payload"]["files"] == files


async def _started(adapter):
    """The top level's session, a turn running."""
    await adapter.handle_call("message.send", {"id": "m-1", "thread": PROJECT, "text": "Go"})
    [session] = adapter.projects.backends["claude"].sessions
    return session


def _logged(adapter, event_type):
    return [event.payload for event in adapter.store.events_after(0) if event.type == event_type]


# ── changes and diffs ─────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_changes_compare_the_working_tree_with_head_or_with_the_branchs_base(adapter):
    web = adapter.web
    git(web, "switch", "-q", "-c", "feature")
    (web / "count.txt").write_text(lines(20, {1: "one"}))
    git(web, "commit", "-q", "-am", "feature")
    (web / "count.txt").write_text(lines(20, {1: "one", 20: "twenty"}))
    git(web, "mv", "old.txt", "new.txt")
    (web / "notes.md").write_text("a\nb")  # untracked, the last line unended

    uncommitted = await adapter.handle_call("changes.get", {"project": PROJECT, "scope": "uncommitted"})
    assert uncommitted == {"scope": "uncommitted", "base": "main", "more": False, "added": 3, "removed": 1, "files": [
        {"path": "count.txt", "old_path": None, "status": "modified", "added": 1, "removed": 1, "binary": False},
        {"path": "new.txt", "old_path": "old.txt", "status": "renamed", "added": 0, "removed": 0, "binary": False},
        {"path": "notes.md", "old_path": None, "status": "added", "added": 2, "removed": 0, "binary": False}]}
    base = await adapter.handle_call("changes.get", {"project": PROJECT, "scope": "base"})
    assert base["base"] == "main" and (base["files"][0]["added"], base["files"][0]["removed"]) == (2, 2)

    # A folder that is no repository has no changes to show.
    (adapter.root / "plain").mkdir()
    await adapter.handle_call("project.create", {
        "id": OTHER, "name": "plain", "folder": str(adapter.root / "plain"), "backend": "claude",
        "model": None, "effort": None, "approval": "ask"})
    error = await _error(adapter, "changes.get", {"project": OTHER, "scope": "uncommitted"})
    assert error.data["reason"] == "not_git"


@pytest.mark.asyncio
async def test_a_files_diff_keeps_gits_lines_within_its_limits(adapter):
    web = adapter.web
    (web / "count.txt").write_text(lines(20, {10: "ten", 11: "x" * 12000}))
    (web / "huge.txt").write_text(lines(20001))
    (web / "image.png").write_bytes(b"\x89PNG\0\0\0data")

    def diff(path, **fields):
        return adapter.handle_call("diff.get", {"project": PROJECT, "scope": "uncommitted", "path": path, **fields})
    [changed] = (await diff("count.txt"))["files"]
    assert (changed["status"], changed["added"], changed["removed"], changed["too_large"]) == ("modified", 2, 2, False)
    [hunk] = changed["hunks"]
    assert (hunk["old_start"], hunk["old_lines"], hunk["new_start"], hunk["new_lines"]) == (7, 8, 7, 8)
    assert hunk["lines"][:5] == [" 7", " 8", " 9", "-10", "-11"] and hunk["lines"][-3:] == [" 12", " 13", " 14"]
    assert hunk["lines"][6] == "+" + "x" * 9999  # a line past 10,000 characters is cut
    [whole] = (await diff("count.txt", full=True))["files"][0]["hunks"]
    assert whole["old_start"] == 1 and len(whole["lines"]) == 22

    [huge] = (await diff("huge.txt"))["files"]
    assert huge["too_large"] and huge["hunks"] == [] and huge["added"] == 20001
    [image] = (await diff("image.png"))["files"]
    assert image["binary"] and image["hunks"] == [] and not image["too_large"]
    error = await _error(adapter, "diff.get", {"project": PROJECT, "scope": "uncommitted", "path": "old.txt"})
    assert error.code == -32602


# ── edits and the working tree's summary ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_an_edits_diff_is_kept_before_its_approval_and_changes_go_live_when_they_change(adapter):
    web = adapter.web
    phone = await connect(adapter, adapter.keys)
    try:
        await phone.hello()
        await _frames_until(phone, _changes(0))  # the clean tree's summary, as last seen
        session = await _started(adapter)
        session.emit(Item("toolu_1", "tool", fields={"tool": "edit", "title": "count.txt", "input": "…"}, arguments={
            "file_path": str(web / "count.txt"), "old_string": "5\n", "new_string": "five\n"}))
        session.emit(Ask("ask_1", "edit", item="toolu_1", files=({"path": "count.txt", "added": 1, "removed": 1},)))
        [request] = await until(lambda: _logged(adapter, "approval.request"))
        [kept] = (await adapter.handle_call("diff.get", {"item": request["item"]}))["files"]
        assert kept["path"] == "count.txt" and kept["hunks"][0]["lines"][3:5] == ["-5", "+five"]
        await adapter.handle_call("approval.respond", {"request_id": request["request_id"], "choice": "once"})

        # The tool ran: its item's end has the working tree looked at.
        (web / "count.txt").write_text(lines(20, {5: "five"}))
        session.emit(Item("toolu_1", "tool", "completed"))
        frames = await _frames_until(phone, _changes(1))
        assert frames[-1]["payload"] == {"project": PROJECT, "git": True, "branch": "main", "files": 1, "added": 1,
                                         "removed": 1}
        [item] = [payload for payload in _logged(adapter, "item.completed") if payload["id"] == request["item"]]
        assert item["files"] == [{"path": "count.txt", "added": 1, "removed": 1}]

        # A backend's own diff is kept as it gave it; a turn's end looks at the tree too.
        session.emit(Item("call_2", "tool", fields={"tool": "edit", "title": "new.py", "input": "…"}, patches=(
            Patch("new.py", "@@ -0,0 +1,2 @@\n+a\n+b\n", "added"),)))
        [patched] = await until(lambda: [p for p in _logged(adapter, "item.started") if p.get("title") == "new.py"])
        assert (await adapter.handle_call("diff.get", {"item": patched["id"]}))["files"][0]["hunks"][0]["lines"] == [
            "+a", "+b"]
        (web / "new.py").write_text("a\nb\n")
        session.emit(TurnEnd("completed"))
        await _frames_until(phone, _changes(2))
    finally:
        await phone.close()
    # A phone catching up gets each project's summary after sync.done.
    other = await connect(adapter, adapter.keys)
    try:
        await other.hello(after=adapter.store.latest())
        assert (await other.event())["params"]["payload"]["files"] == 2
    finally:
        await other.close()


# ── output ────────────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_commands_output_is_paged_and_followed_until_it_ends_or_the_phone_goes(adapter, monkeypatch):
    monkeypatch.setattr(output, "KEEP_BYTES", 8)
    monkeypatch.setattr(output, "REWRITE_BYTES", 16)
    session = await _started(adapter)
    session.emit(Item("toolu_1", "tool", fields={"tool": "execute", "title": "make", "input": "make"}))
    session.emit(Delta("toolu_1", output="a\nb\nc"))
    [item] = await until(lambda: _logged(adapter, "item.started"))
    await until(lambda: (adapter.projects.state_dir / PROJECT / f"{item['id']}.out").exists())
    phone, gone = await connect(adapter, adapter.keys), await connect(adapter, adapter.keys)
    try:
        await phone.hello(after=adapter.store.latest())
        await gone.hello(after=adapter.store.latest())
        page = (await phone.call("output.get", {"item": item["id"], "before": None, "limit": 2, "follow": True}))
        assert page["result"] == {"start": 1, "lines": ["b", "c"], "total": 3, "running": True}
        older = await phone.call("output.get", {"item": item["id"], "before": 1, "limit": 2})
        assert older["result"]["start"] == 0 and older["result"]["lines"] == ["a"]
        await gone.call("output.get", {"item": item["id"], "before": None, "limit": 1, "follow": True})
        await gone.close()
        await until(lambda: len(adapter.projects.outputs._following) == 1)  # the closed one's follow ended

        # New output replaces from the partial last line on, sent again whole.
        session.emit(Delta("toolu_1", output="d\ne\n"))
        frames = await _frames_until(phone, lambda frame: frame["type"] == "output.delta")
        assert frames[-1]["payload"] == {"item": item["id"], "start": 2, "lines": ["cd", "e"]}
        # Past its limit the output keeps only its last lines; start says what was dropped.
        session.emit(Delta("toolu_1", output="f\ng\nh\ni\n"))
        session.emit(Item("toolu_1", "tool", "completed", {"exit_code": 0}))
        [done] = await until(lambda: _logged(adapter, "item.completed"))
        assert done["output_lines"] == 8
        last = (await phone.call("output.get", {"item": item["id"], "before": None, "limit": 10, "follow": True}))
        assert last["result"] == {"start": 4, "lines": ["f", "g", "h", "i"], "total": 8, "running": False}
        assert not adapter.projects.outputs._following  # following ended with the command
    finally:
        await phone.close()
