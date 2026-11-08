"""A settled webhook delivery releases what a turn runtime pooled for its one-shot session."""
from types import SimpleNamespace

from gateway.platforms import webhook_main


def test_release_covers_both_sides_of_a_compression_rotation(monkeypatch):
    released = []
    monkeypatch.setattr("agent.turn_runtime_registry.release_runtime_sessions",
                        lambda *ids: released.append(ids) or 0)
    db = SimpleNamespace(get_compression_tip=lambda sid: "tip-" + sid)
    receipt = SimpleNamespace(ref=SimpleNamespace(session_id="s1"))
    webhook_main.release_one_shot_session(SimpleNamespace(db=db), receipt)
    assert released == [("tip-s1", "s1")]
