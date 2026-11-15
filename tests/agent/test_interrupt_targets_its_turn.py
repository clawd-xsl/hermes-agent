"""An interrupt stops the turn it was sent to, never a later one: a message that arrives while a finished
turn is still winding down must not abort the next turn on the same (cached) agent, which is often the
turn that answers that very message."""

from __future__ import annotations

from unittest.mock import patch

from tests.agent.test_run_agent import _mock_response
from tests.agent.test_turn_user_intervened import _loop_agent


def _run(agent, model_call, text):
    agent._interruptible_api_call = model_call
    with (patch.object(agent, "_flush_messages_to_session_db"), patch.object(agent, "_persist_session"),
          patch.object(agent, "_save_trajectory"), patch.object(agent, "_cleanup_task_resources")):
        return agent.run_conversation(text)


def _reply(text):
    return lambda api_kwargs: _mock_response(content=text, finish_reason="stop")


def test_an_interrupt_after_the_turn_ended_does_not_reach_the_next_turn():
    agent = _loop_agent()
    first = _run(agent, _reply("first answer"), "first question")
    assert first["final_response"] == "first answer"

    assert agent.interrupt("second question") is False  # the turn it aimed at is over
    second = _run(agent, _reply("second answer"), "second question")
    assert second["final_response"] == "second answer" and not second.get("interrupted")
    assert second["api_calls"] >= 1


def test_an_interrupt_mid_turn_stops_that_turn_only():
    agent = _loop_agent()

    def interrupted_call(api_kwargs):
        assert agent.interrupt("never mind") is True
        raise InterruptedError("the interrupt cancelled the request")

    stopped = _run(agent, interrupted_call, "a long task")
    assert stopped.get("interrupted") is True
    after = _run(agent, _reply("fresh answer"), "never mind")
    assert after["final_response"] == "fresh answer" and not after.get("interrupted")
