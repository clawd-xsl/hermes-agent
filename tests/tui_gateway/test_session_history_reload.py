"""session.history.reload: an idle resumed session picks up turns persisted by
another writer (e.g. the messaging gateway appending Signal turns), while its
own turns and running turns are never clobbered."""

import threading
import types

import pytest

from tui_gateway import server


@pytest.fixture
def db(tmp_path, monkeypatch):
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("s1", source="signal")
    db.append_message("s1", role="user", content="hello from signal")
    db.append_message("s1", role="assistant", content="hi there")
    monkeypatch.setattr(server, "_get_db", lambda: db)
    yield db
    db.close()


@pytest.fixture
def session(db):
    # A cold resume hydrates the model history stamped with durable row ids.
    history = db.get_messages_as_conversation("s1", repair_alternation=True, include_row_ids=True)
    sess = {
        "agent": types.SimpleNamespace(),
        "session_key": "s1",
        "history": list(history),
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "cols": 80,
    }
    server._sessions["sid"] = sess
    yield sess
    server._sessions.pop("sid", None)


def _reload():
    resp = server.handle_request(
        {"id": "r", "method": "session.history.reload", "params": {"session_id": "sid"}}
    )
    assert "error" not in resp, resp
    return resp["result"]


def _texts(result):
    return [m.get("text") for m in result["messages"]]


def _contents(session):
    return [m.get("content") for m in session["history"]]


def test_external_turn_is_reloaded_into_view_and_model_history(db, session):
    assert _reload()["reloaded"] is False  # first call records the baseline
    assert _reload()["reloaded"] is False  # nothing new

    db.append_message("s1", role="user", content="sent later via signal")
    db.append_message("s1", role="assistant", content="answered on signal")

    result = _reload()

    assert result["reloaded"] is True
    assert _texts(result)[-2:] == ["sent later via signal", "answered on signal"]
    # The model-fed history follows too, so the next local turn sees them once.
    assert _contents(session) == ["hello from signal", "hi there", "sent later via signal", "answered on signal"]
    assert _reload()["reloaded"] is False


def test_own_turn_rebaselines_instead_of_reloading(db, session):
    _reload()

    # A local turn persists its rows and bumps history_version on completion.
    user_id = db.append_message("s1", role="user", content="typed in the dashboard")
    reply_id = db.append_message("s1", role="assistant", content="local reply")
    with session["history_lock"]:
        session["history"] = session["history"] + [
            {"role": "user", "content": "typed in the dashboard", "_row_id": user_id},
            {"role": "assistant", "content": "local reply", "_row_id": reply_id},
        ]
        session["history_version"] += 1

    assert _reload()["reloaded"] is False
    assert _reload()["reloaded"] is False
    assert session["history_version"] == 1


def test_running_session_is_never_reloaded(db, session):
    _reload()
    db.append_message("s1", role="user", content="external while busy")
    session["running"] = True

    result = _reload()

    assert result == {"reloaded": False, "reason": "busy"}
    assert "external while busy" not in _contents(session)

    session["running"] = False
    assert _reload()["reloaded"] is True
    assert "external while busy" in _contents(session)


def test_compression_continuation_asks_client_to_resume_tip(db, session, monkeypatch):
    _reload()
    db.append_message("s1", role="user", content="rotated elsewhere")
    monkeypatch.setattr(db, "resolve_resume_session_id", lambda key: "s1-tip")

    result = _reload()

    assert result["reloaded"] is False
    assert result["rotated_to"] == "s1-tip"
