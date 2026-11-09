"""The agent's activity on Aleph (plugins/platforms/aleph/activity.py): the status line, activity steps
as live events, activity.list, and the gateway hook that reports tool starts and finishes."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
import pytest_asyncio

from gateway.config import Platform, PlatformConfig
from plugins.platforms.aleph.activity import describe
from tests.gateway.aleph_phone import connect, pair_device


@pytest_asyncio.fixture
async def adapter(tmp_path):
    from plugins.platforms.aleph.adapter import AlephAdapter
    config = PlatformConfig(enabled=True, extra={"host": "127.0.0.1", "port": 0,
                                                 "state_dir": str(tmp_path / "aleph")})
    instance = AlephAdapter(config)

    async def handler(event):
        return None
    instance.set_message_handler(handler)
    assert await instance.connect()
    instance.device, instance.keys = pair_device(instance)
    yield instance
    await instance.disconnect()


SOURCE = SimpleNamespace(chat_id="main", platform=Platform("aleph"))


def _turn(message_id="m-1", internal=False):
    return SimpleNamespace(source=SOURCE, internal=internal, message_id=message_id)


async def _tool(adapter, event_type, tool, args=None, **extra):
    """What the gateway's progress callback does, from the agent's worker thread."""
    await asyncio.to_thread(adapter.on_tool_activity, SOURCE, {"type": event_type, "tool": tool, "args": args,
                                                               "live_status": extra.pop("mode", "full"), **extra})
    for _ in range(20):
        await asyncio.sleep(0.005)


async def _live(phone, event_type="activity", count=1):
    out = []
    while len(out) < count:
        frame = (await phone.event())["params"]
        if frame["type"] == event_type:
            out.append(frame["payload"])
    return out


# ── wording ───────────────────────────────────────────────────────────────────────────────────


def test_the_phrase_has_no_subject_and_starts_with_a_capital():
    assert describe("web_search", {"query": "flights to Tokyo"}) == (
        "web_search", "Searching the web for flights to Tokyo", "flights to Tokyo")
    assert describe("terminal", {"command": "ls -la ~/Code"})[1:] == ("Running ls -la ~/Code", "ls -la ~/Code")
    assert describe("read_file", {"path": "/Users/x/notes.md"})[1] == "Reading notes.md"
    assert describe("skills_list", {})[1:] == ("Listing skills", None)


def test_tools_without_a_verb_say_which_tool_and_bridge_calls_are_unwrapped():
    assert describe("some_mcp_tool", {"query": "x"}) == ("some_mcp_tool", "Using some_mcp_tool", "x")
    assert describe("aleph_set_profile", {"name": "Iris"}) == ("aleph_set_profile", "Changing name or avatar", "Iris")
    bridged = {"calls": [{"name": "web_search", "arguments": {"query": "weather in Paris"}},
                         {"name": "read_file", "arguments": {"path": "/tmp/a.md"}}]}
    assert describe("tool_call", bridged) == (
        "web_search", "Searching the web for weather in Paris (+1 more)", "weather in Paris")
    assert describe("tool_call", {"name": "aleph_react", "arguments": {"emoji": "👍"}})[:2] == (
        "aleph_react", "Reacting to a message")
    # Looking tools up is no step.
    assert describe("tool_search", {"query": "react"}) is None
    assert describe("tool_describe", {"name": "aleph_react"}) is None


def test_the_detail_is_the_whole_argument_with_its_lines_and_no_secrets():
    assert describe("terminal", {"command": "cd ~/x\nls -la"})[2] == "cd ~/x\nls -la"
    assert describe("write_file", {"path": "/tmp/a.py", "content": "print(1)\nprint(2)\n"})[2] == (
        "/tmp/a.py\n\nprint(1)\nprint(2)")
    assert describe("patch", {"path": "a.py", "old_string": "x = 1", "new_string": "x = 2"})[2] == (
        "a.py\n\nx = 1\n\nx = 2")
    _tool, summary, detail = describe("terminal", {"command": "export API_KEY=sk-ant-abcdefghijklmnopqrstuvwxyz0123 && run"})
    assert "sk-ant" not in detail and "sk-ant" not in summary and "API_KEY=***" in detail
    assert "hunter2" not in describe("terminal", {"command": "curl https://u:hunter2pass@example.com/x"})[2]


def test_verb_mode_keeps_arguments_out_and_long_phrases_are_clipped():
    assert describe("web_search", {"query": "secret plans"}, with_preview=False) == (
        "web_search", "Searching the web", None)
    _tool, summary, detail = describe("terminal", {"command": "echo " + "x" * 400})
    assert len(summary) <= 80 and summary.endswith("…") and detail == "echo " + "x" * 400
    _tool, _summary, detail = describe("terminal", {"command": "echo " + "x" * 3000})
    assert len(detail) == 2000 and detail.endswith("…")


# ── steps ─────────────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_step_starts_and_finishes_as_live_events_and_drives_the_status_line(adapter):
    phone = await connect(adapter, adapter.keys)
    try:
        await phone.hello()
        adapter._last_user_message["main"] = "m-1"  # the person's message this turn answers
        await adapter.on_processing_start(_turn())
        await adapter.send_typing("main")
        latest = adapter.store.latest()
        await _tool(adapter, "tool.started", "web_search", {"query": "flights to Tokyo"})
        (started,) = await _live(phone)
        assert started["chat"] == "main" and started["turn"] == "m-1" and started["tool"] == "web_search"
        assert started["summary"] == "Searching the web for flights to Tokyo"
        assert started["detail"] == "flights to Tokyo" and started["finished_at"] is None and started["ok"] is None
        (typing,) = await _live(phone, "typing")
        while typing.get("status") is None:
            (typing,) = await _live(phone, "typing")
        assert typing == {"chat": "main", "active": True, "status": "Searching the web for flights to Tokyo…"}
        await _tool(adapter, "tool.completed", "web_search", is_error=True)
        (finished,) = await _live(phone)
        assert finished["id"] == started["id"] and finished["finished_at"] and finished["ok"] is False
        (typing,) = await _live(phone, "typing")
        assert typing == {"chat": "main", "active": True}  # null between tools: the phone shows "Thinking…"
        assert adapter.store.latest() == latest  # steps are not in the event log
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_parallel_steps_pair_by_tool_and_the_line_falls_back_to_the_one_still_running(adapter):
    await adapter.on_processing_start(_turn())
    await adapter.send_typing("main")
    await _tool(adapter, "tool.started", "web_search", {"query": "one"})
    await _tool(adapter, "tool.started", "read_file", {"path": "/tmp/two.md"})
    await _tool(adapter, "tool.started", "web_search", {"query": "three"})
    assert adapter._activity_status["main"] == "Searching the web for three…"
    await _tool(adapter, "tool.completed", "web_search")
    steps, _ = adapter.store.steps(limit=10)
    by_detail = {s["detail"]: s for s in steps}
    assert by_detail["one"]["ok"] is True and by_detail["three"]["finished_at"] is None
    assert adapter._activity_status["main"] == "Searching the web for three…"
    await _tool(adapter, "tool.completed", "web_search")
    assert adapter._activity_status["main"] == "Reading two.md…"
    await _tool(adapter, "tool.completed", "nothing_started")  # no matching step: ignored
    await _tool(adapter, "tool.completed", "read_file")
    assert adapter._activity_status["main"] is None


@pytest.mark.asyncio
async def test_looking_tools_up_is_no_step_and_the_line_stays_on_thinking(adapter):
    await adapter.on_processing_start(_turn())
    await adapter.send_typing("main")
    await _tool(adapter, "tool.started", "tool_search", {"query": "react"})
    assert adapter._activity_status.get("main") is None
    await _tool(adapter, "tool.completed", "tool_search")
    assert adapter.store.steps(limit=5)[0] == []


@pytest.mark.asyncio
async def test_a_turn_that_ends_mid_step_closes_it_as_failed(adapter):
    await adapter.on_processing_start(_turn())
    await _tool(adapter, "tool.started", "terminal", {"command": "sleep 100"})
    await adapter.on_processing_complete(_turn(), None)
    (step,) = adapter.store.steps(limit=5)[0]
    assert step["ok"] is False and step["finished_at"]
    assert not adapter._open_steps.get("main") and adapter._activity_status.get("main") is None


@pytest.mark.asyncio
async def test_status_modes_and_cron_turns(adapter):
    await adapter.on_processing_start(_turn(internal=True))  # a cron run answers no message
    await adapter.send_typing("main")
    await _tool(adapter, "tool.started", "web_search", {"query": "private"}, mode="off")
    assert adapter._activity_status.get("main") is None  # live_status off: no line
    await _tool(adapter, "tool.completed", "web_search")
    await _tool(adapter, "tool.started", "web_search", {"query": "private"}, mode="verb")
    assert adapter._activity_status["main"] == "Searching the web…"
    steps, _ = adapter.store.steps(limit=5)
    assert all(s["turn"] is None for s in steps)
    assert "detail" not in steps[0] and steps[0]["summary"] == "Searching the web"


@pytest.mark.asyncio
async def test_a_phone_connecting_mid_step_sees_it(adapter):
    await adapter.on_processing_start(_turn())
    await adapter.send_typing("main")
    await _tool(adapter, "tool.started", "read_file", {"path": "/tmp/a.md"})
    phone = await connect(adapter, adapter.keys)
    try:
        await phone.hello()
        (step,) = await _live(phone)
        assert step["summary"] == "Reading a.md" and step["finished_at"] is None
    finally:
        await phone.close()


# ── activity.list ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_activity_list_pages_newest_first(adapter):
    for n in range(7):
        await _tool(adapter, "tool.started", "web_search", {"query": f"q{n}"})
        await _tool(adapter, "tool.completed", "web_search")
    phone = await connect(adapter, adapter.keys)
    try:
        await phone.hello()
        first = (await phone.call("activity.list", {"limit": 3}))["result"]
        assert [s["detail"] for s in first["steps"]] == ["q6", "q5", "q4"] and first["more"] is True
        second = (await phone.call("activity.list", {"before": first["steps"][-1]["id"], "limit": 3}))["result"]
        assert [s["detail"] for s in second["steps"]] == ["q3", "q2", "q1"] and second["more"] is True
        last = (await phone.call("activity.list", {"before": second["steps"][-1]["id"]}))["result"]
        assert [s["detail"] for s in last["steps"]] == ["q0"] and last["more"] is False
        assert len((await phone.call("activity.list"))["result"]["steps"]) == 7
        for bad in ({"limit": 0}, {"limit": 201}, {"limit": "5"}, {"before": 3}):
            assert (await phone.call("activity.list", bad))["error"]["code"] == -32602
        assert (await phone.call("activity.list", {"before": "a_unknown"}))["result"] == {"steps": [], "more": False}
    finally:
        await phone.close()


def test_only_the_most_recent_steps_are_kept(tmp_path):
    from plugins.platforms.aleph.store import AlephStore
    store = AlephStore(tmp_path)
    try:
        for n in range(6):
            store.add_step({"id": f"a_{n}", "chat": "main", "turn": None, "tool": "t", "summary": "s", "started_at": "x",
                            "finished_at": None, "ok": None}, keep=4)
        steps, more = store.steps(limit=10)
        assert [s["id"] for s in steps] == ["a_5", "a_4", "a_3", "a_2"] and more is False
    finally:
        store.close()


# ── the gateway side ──────────────────────────────────────────────────────────────────────────


def test_tool_progress_rows_are_off_on_aleph_by_default():
    from gateway.display_config import resolve_display_setting
    from hermes_cli.plugins import discover_plugins
    discover_plugins()  # the platform's registered display defaults
    assert resolve_display_setting({}, "aleph", "tool_progress") == "off"
    assert resolve_display_setting({}, "aleph", "live_status") == "full"
    assert resolve_display_setting({}, "aleph", "long_running_notifications") is False
    assert resolve_display_setting({}, "aleph", "interim_assistant_messages") is True  # the agent's own words stay
    user = {"display": {"platforms": {"aleph": {"tool_progress": "all"}}}}
    assert resolve_display_setting(user, "aleph", "tool_progress") == "all"  # still the person's choice
