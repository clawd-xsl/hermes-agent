from __future__ import annotations

import json
import threading

import pytest

from agent.claude_cli_journal import (
    ClaudeCliToolJournal,
    ClaudeCliToolJournalError,
    ToolIntent,
)


def _intent(tool_use_id: str = "tool-1", *, value: str = "hello") -> ToolIntent:
    return ToolIntent(
        tool_use_id=tool_use_id,
        name="send_message",
        arguments={"value": value},
        batch_id="batch-1",
        ordinal=0,
    )


def _result(tool_use_id: str = "tool-1", *, content: str = "sent") -> dict:
    return {
        "role": "tool",
        "name": "send_message",
        "tool_call_id": tool_use_id,
        "content": content,
        "effect_disposition": "performed",
    }


def test_completed_effect_is_replayed_after_restart(tmp_path):
    first = ClaudeCliToolJournal("owner", root=tmp_path)
    assert first.claim_batch([_intent()])[0].disposition == "execute"
    first.complete([_result()])

    recovered = ClaudeCliToolJournal("owner", root=tmp_path)
    claim = recovered.claim_batch([_intent()])[0]

    assert claim.disposition == "replay"
    assert claim.result_row == _result()


def test_running_effect_is_unknown_after_restart_and_never_replayed(tmp_path):
    first = ClaudeCliToolJournal("owner", root=tmp_path)
    assert first.claim_batch([_intent()])[0].disposition == "execute"

    recovered = ClaudeCliToolJournal("owner", root=tmp_path)
    claim = recovered.claim_batch([_intent()])[0]

    assert claim.disposition == "unknown"
    assert claim.result_row is None


def test_concurrent_journal_instances_cannot_both_claim_one_effect(tmp_path):
    first = ClaudeCliToolJournal("owner", root=tmp_path)
    second = ClaudeCliToolJournal("owner", root=tmp_path)
    ready = threading.Barrier(2)
    dispositions = []

    def claim(journal):
        ready.wait()
        dispositions.append(journal.claim_batch([_intent()])[0].disposition)

    threads = [
        threading.Thread(target=claim, args=(journal,))
        for journal in (first, second)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2.0)

    assert all(not thread.is_alive() for thread in threads)
    assert sorted(dispositions) == ["execute", "unknown"]


def test_batch_claim_is_one_durable_transition(tmp_path, monkeypatch):
    journal = ClaudeCliToolJournal("owner", root=tmp_path)
    intents = [_intent("tool-1"), _intent("tool-2", value="world")]
    journal.prepare_batch(intents)
    writes = []
    original = journal._persist
    monkeypatch.setattr(
        journal,
        "_persist",
        lambda: (writes.append(journal.snapshot()), original())[1],
    )

    claims = journal.claim_batch(intents)

    assert [claim.disposition for claim in claims] == ["execute", "execute"]
    assert len(writes) == 1
    assert {row["state"] for row in writes[0].values()} == {"running"}


def test_host_skipped_effect_can_complete_without_running(tmp_path):
    journal = ClaudeCliToolJournal("owner", root=tmp_path)
    journal.prepare_batch([_intent()])

    journal.complete_without_effect([_result(content="skipped")])

    claim = journal.claim_batch([_intent()])[0]
    assert claim.disposition == "replay"
    assert claim.result_row == _result(content="skipped")


def test_reused_tool_id_with_different_input_fails_closed(tmp_path):
    journal = ClaudeCliToolJournal("owner", root=tmp_path)
    journal.prepare_batch([_intent(value="first")])

    with pytest.raises(ClaudeCliToolJournalError, match="different tool intent"):
        journal.prepare_batch([_intent(value="second")])


def test_corrupt_journal_fails_closed(tmp_path):
    journal = ClaudeCliToolJournal("owner", root=tmp_path)
    journal.root.mkdir(parents=True, exist_ok=True)
    journal.path.write_text("not-json", encoding="utf-8")

    with pytest.raises(ClaudeCliToolJournalError, match="unreadable"):
        ClaudeCliToolJournal("owner", root=tmp_path)


def test_reconciled_record_remains_auditable(tmp_path):
    journal = ClaudeCliToolJournal("owner", root=tmp_path)
    journal.claim_batch([_intent()])
    journal.complete([_result()])
    journal.mark_reconciled(["tool-1"])

    on_disk = json.loads(journal.path.read_text(encoding="utf-8"))
    assert on_disk["records"]["tool-1"]["state"] == "reconciled"
    assert on_disk["records"]["tool-1"]["result_row"] == _result()


def test_audit_tail_is_bounded_without_pruning_unresolved_effects(tmp_path):
    journal = ClaudeCliToolJournal("owner", root=tmp_path)
    unresolved = _intent("still-running")
    journal.claim_batch([unresolved])
    for index in range(journal._MAX_RECONCILED_RECORDS + 2):
        intent = _intent(f"done-{index}", value=str(index))
        journal.claim_batch([intent])
        journal.complete([_result(intent.tool_use_id, content=str(index))])
        journal.mark_reconciled([intent.tool_use_id])

    journal.prepare_batch([_intent("new-call")])
    snapshot = journal.snapshot()

    assert snapshot["still-running"]["state"] == "running"
    assert snapshot["new-call"]["state"] == "prepared"
    assert "done-0" not in snapshot
    assert "done-1" not in snapshot
    assert "done-2" in snapshot
