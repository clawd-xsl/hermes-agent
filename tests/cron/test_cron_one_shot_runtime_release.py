"""A finished cron fire releases what a turn runtime pooled for its one-shot session."""


def test_finalize_releases_both_sides_of_a_compression_rotation(monkeypatch):
    import cron.scheduler as scheduler
    released = []
    monkeypatch.setattr("agent.turn_runtime_registry.release_runtime_sessions",
                        lambda *ids: released.append(ids) or 0)

    class _DB:
        def get_compression_tip(self, session_id):
            return "cron_tip"

        def __getattr__(self, name):
            return lambda *a, **k: None
    monkeypatch.setattr("hermes_state_registry.release_or_close", lambda db: None)
    scheduler._finalize_cron_session(_DB(), None, "j", "job", "cron_orig")
    assert released == [("cron_tip", "cron_orig")]
