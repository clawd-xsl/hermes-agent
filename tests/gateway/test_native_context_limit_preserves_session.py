import pytest

from gateway.run import _should_reset_exhausted_session


@pytest.mark.parametrize("deferred", [False, True])
def test_native_limit_never_authorizes_automatic_reset(deferred):
    assert not _should_reset_exhausted_session({
        "compression_exhausted": True,
        "compression_deferred": deferred,
        "native_history_preserved": True,
    })


def test_local_compression_reset_still_requires_terminal_exhaustion():
    assert _should_reset_exhausted_session({"compression_exhausted": True})
    assert not _should_reset_exhausted_session({"compression_exhausted": True, "compression_deferred": True})
    assert not _should_reset_exhausted_session({"failed": True})
