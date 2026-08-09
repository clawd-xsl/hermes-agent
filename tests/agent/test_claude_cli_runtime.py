from __future__ import annotations

from types import SimpleNamespace

from agent.claude_cli_runtime import _dedupe_final_projection, _record_usage


class _Compressor:
    def __init__(self):
        self.usage = None

    def update_from_response(self, usage):
        self.usage = usage


class _SessionDB:
    def __init__(self):
        self.calls = []

    def update_token_counts(self, *args, **kwargs):
        self.calls.append((args, kwargs))


def test_usage_accounts_aggregate_turn_but_compressor_tracks_last_native_call():
    db = _SessionDB()
    agent = SimpleNamespace(
        session_api_calls=0,
        session_prompt_tokens=0,
        session_completion_tokens=0,
        session_total_tokens=0,
        session_input_tokens=0,
        session_output_tokens=0,
        session_cache_read_tokens=0,
        session_cache_write_tokens=0,
        session_estimated_cost_usd=0.0,
        context_compressor=_Compressor(),
        _session_db=db,
        _session_db_created=True,
        session_id="session-1",
        model="claude-opus-4-6",
    )

    result = _record_usage(
        agent,
        {
            "input_tokens": 100,
            "cache_read_input_tokens": 200,
            "cache_creation_input_tokens": 10,
            "output_tokens": 30,
        },
        {
            "input_tokens": 5,
            "cache_read_input_tokens": 95,
            "output_tokens": 12,
        },
    )

    assert result["prompt_tokens"] == 310
    assert result["total_tokens"] == 340
    assert result["last_prompt_tokens"] == 100
    assert agent.session_total_tokens == 340
    assert agent.context_compressor.usage == {
        "prompt_tokens": 100,
        "completion_tokens": 12,
        "total_tokens": 112,
    }
    assert db.calls[0][1]["billing_mode"] == "subscription_included"
    assert db.calls[0][1]["api_call_count"] == 1


def test_final_projection_is_not_duplicated():
    projected = [{"role": "assistant", "content": "done"}]
    assert _dedupe_final_projection(projected, "done") is projected
    assert _dedupe_final_projection([], "done") == [
        {"role": "assistant", "content": "done"}
    ]

