"""Durability contracts for reviewed ``session: main`` webhook receipts."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.main_session import MainSessionEnqueueResult
from gateway.platforms.webhook import WebhookAdapter, _INSECURE_NO_AUTH
from gateway.platforms.base import ProcessingOutcome
from gateway.session import SessionSource
from gateway.webhook_review_store import (
    COMPLETED,
    FILTERED,
    MAIN_ENQUEUING,
    PENDING_MAIN,
    PENDING_REVIEW,
    REVIEWING,
    WebhookReviewStore,
)


def _source() -> SessionSource:
    return SessionSource(
        platform=Platform.SIGNAL,
        chat_id="+15551234567",
        chat_type="dm",
        user_id="+15551234567",
    )


def _accept(store: WebhookReviewStore, *, delivery_id: str = "delivery-1"):
    return store.accept(
        profile="default",
        route="gmail",
        delivery_id=delivery_id,
        event_id=f"webhook:gmail:{delivery_id}",
        event_type="mail.received",
        review_prompt="Review this event",
        skills=["email"],
        main_source=_source().to_dict(),
    )


def _adapter(store: WebhookReviewStore) -> WebhookAdapter:
    config = PlatformConfig(
        enabled=True,
        extra={
            "host": "127.0.0.1",
            "port": 0,
            "routes": {
                "gmail": {
                    "secret": _INSECURE_NO_AUTH,
                    "session": "main",
                    "prompt": "{message}",
                }
            },
        },
    )
    adapter = WebhookAdapter(config)
    adapter._review_store = store
    adapter.gateway_runner = MagicMock()
    return adapter


def test_accept_is_durable_and_idempotent_across_store_instances(tmp_path):
    db_path = tmp_path / "state.db"
    first_store = WebhookReviewStore(db_path)
    first = _accept(first_store)

    second_store = WebhookReviewStore(db_path)
    duplicate = _accept(second_store)

    assert first.created is True
    assert duplicate.created is False
    assert duplicate.receipt["receipt_id"] == first.receipt["receipt_id"]
    assert duplicate.receipt["state"] == PENDING_REVIEW


def test_new_adapter_lifecycle_reclaims_an_interrupted_reviewer(tmp_path):
    store = WebhookReviewStore(tmp_path / "state.db")
    accepted = _accept(store).receipt
    receipt_id = accepted["receipt_id"]
    now = float(accepted["created_at"])

    first_claim = store.claim_due(owner_token="old-adapter", now=now)
    assert first_claim[0]["state"] == REVIEWING
    # The same live owner cannot double-claim its reviewer before lease expiry.
    assert store.claim_due(owner_token="old-adapter", now=now + 1) == []

    recovered = store.claim_due(owner_token="new-adapter", now=now + 1)
    assert recovered[0]["receipt_id"] == receipt_id
    assert recovered[0]["state"] == REVIEWING
    assert recovered[0]["attempts"] == 2


def test_reviewer_handoff_is_persisted_before_main_stage(tmp_path):
    db_path = tmp_path / "state.db"
    store = WebhookReviewStore(db_path)
    accepted = _accept(store).receipt
    receipt_id = accepted["receipt_id"]
    store.claim_due(owner_token="adapter", now=accepted["created_at"])

    assert store.mark_reviewed(
        receipt_id,
        owner_token="adapter",
        handoff="A new important email needs a reply.",
    )
    pending = store.get(receipt_id)
    assert pending["state"] == PENDING_MAIN
    assert pending["handoff"] == "A new important email needs a reply."
    assert pending["attempts"] == 0

    # A fresh adapter lifecycle can resume the admitted handoff without the
    # reviewer or its original in-memory event.
    restarted_store = WebhookReviewStore(db_path)
    main_claim = restarted_store.claim_due(
        owner_token="restarted-adapter", now=pending["next_attempt_at"]
    )
    assert main_claim[0]["state"] == MAIN_ENQUEUING
    assert main_claim[0]["handoff"] == "A new important email needs a reply."
    assert restarted_store.mark_completed(receipt_id, owner_token="restarted-adapter")
    assert restarted_store.get(receipt_id)["state"] == COMPLETED


def test_filtered_receipt_is_terminal_and_dedupable(tmp_path):
    store = WebhookReviewStore(tmp_path / "state.db")
    accepted = _accept(store).receipt
    receipt_id = accepted["receipt_id"]
    store.claim_due(owner_token="adapter", now=accepted["created_at"])
    assert store.mark_filtered(receipt_id, owner_token="adapter")
    assert store.get(receipt_id)["state"] == FILTERED
    assert _accept(store).created is False


@pytest.mark.asyncio
async def test_adapter_uses_gateway_owned_executor_for_store_io(tmp_path):
    store = WebhookReviewStore(tmp_path / "state.db")
    receipt_id = _accept(store).receipt["receipt_id"]
    adapter = _adapter(store)
    submitted = []

    async def _run_blocking(call):
        submitted.append(call)
        return call()

    adapter.gateway_runner._run_in_executor_with_context = _run_blocking

    receipt = await adapter._run_review_store_call(store.get, receipt_id)

    assert receipt["receipt_id"] == receipt_id
    assert len(submitted) == 1


@pytest.mark.asyncio
async def test_new_adapter_worker_restarts_interrupted_review(tmp_path):
    store = WebhookReviewStore(tmp_path / "state.db")
    accepted = _accept(store).receipt
    receipt_id = accepted["receipt_id"]
    store.claim_due(owner_token="dead-adapter", now=accepted["created_at"])

    adapter = _adapter(store)
    adapter._end_webhook_session = AsyncMock()
    captured = []

    async def _capture(event):
        captured.append(event)

    adapter.handle_message = _capture
    adapter._ensure_review_outbox_worker()
    for _ in range(50):
        if captured:
            break
        await asyncio.sleep(0.01)

    assert len(captured) == 1
    event = captured[0]
    assert event.metadata["webhook_receipt_id"] == receipt_id
    assert event.raw_message is None
    recovered = store.get(receipt_id)
    assert recovered["state"] == REVIEWING
    assert recovered["attempts"] == 2

    # Finish the synthetic capture so the test leaves no leased receipt/task.
    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)
    assert store.get(receipt_id)["state"] == FILTERED
    for _ in range(50):
        worker = adapter._review_outbox_worker
        if worker is None:
            break
        await asyncio.sleep(0.01)
    assert adapter._review_outbox_worker is None


@pytest.mark.asyncio
async def test_main_enqueue_failure_returns_handoff_to_durable_pending(tmp_path):
    store = WebhookReviewStore(tmp_path / "state.db")
    accepted = _accept(store).receipt
    receipt_id = accepted["receipt_id"]
    store.claim_due(owner_token="adapter", now=accepted["created_at"])
    store.mark_reviewed(
        receipt_id,
        owner_token="adapter",
        handoff="Production deployment failed and needs attention.",
    )
    pending = store.get(receipt_id)
    main_claim = store.claim_due(owner_token="adapter", now=pending["next_attempt_at"])[
        0
    ]

    adapter = _adapter(store)
    adapter._review_owner_token = "adapter"
    # Drive this one claimed stage directly; the assertion is about the
    # durable transition, not the worker's timing loop.
    adapter._ensure_review_outbox_worker = lambda: None
    enqueue = AsyncMock(side_effect=RuntimeError("main temporarily unavailable"))
    with patch("gateway.main_session.enqueue_main_session_turn", new=enqueue):
        await adapter._process_claimed_review_receipt(main_claim)

    after_failure = store.get(receipt_id)
    assert after_failure["state"] == PENDING_MAIN
    assert (
        after_failure["handoff"] == "Production deployment failed and needs attention."
    )
    assert after_failure["last_error"] == "main temporarily unavailable"

    retry_claim = store.claim_due(
        owner_token="adapter",
        now=after_failure["next_attempt_at"],
    )[0]

    async def _enqueue_then_fail_turn(*args, **kwargs):
        await kwargs["processing_complete"](ProcessingOutcome.FAILURE)
        return MainSessionEnqueueResult(
            session_key="agent:main:signal:dm:+15551234567",
            platform="signal",
            chat_id="+15551234567",
            queued=False,
        )

    turn_failure = AsyncMock(side_effect=_enqueue_then_fail_turn)
    with patch("gateway.main_session.enqueue_main_session_turn", new=turn_failure):
        await adapter._process_claimed_review_receipt(retry_claim)

    after_turn_failure = store.get(receipt_id)
    assert after_turn_failure["state"] == PENDING_MAIN
    assert "main turn ended with outcome" in after_turn_failure["last_error"]

    success_claim = store.claim_due(
        owner_token="adapter",
        now=after_turn_failure["next_attempt_at"],
    )[0]

    async def _enqueue_and_complete(*args, **kwargs):
        await kwargs["processing_complete"](ProcessingOutcome.SUCCESS)
        return MainSessionEnqueueResult(
            session_key="agent:main:signal:dm:+15551234567",
            platform="signal",
            chat_id="+15551234567",
            queued=False,
        )

    success = AsyncMock(side_effect=_enqueue_and_complete)
    with patch("gateway.main_session.enqueue_main_session_turn", new=success):
        await adapter._process_claimed_review_receipt(success_claim)

    assert store.get(receipt_id)["state"] == COMPLETED
    queued = success.await_args.kwargs
    assert "Production deployment failed" in queued["text"]
    assert queued["raw_message"] is None
