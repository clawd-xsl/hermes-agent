"""Dashboard cron endpoints carry the per-job ``session: main|isolated`` field."""
import pytest
from fastapi import HTTPException

import hermes_cli.web_models as _web_models
import hermes_cli.web_routers.cron as _rt_cron
import hermes_cli.web_server_cron as _web_server_cron


@pytest.fixture()
def isolated_profiles(tmp_path, monkeypatch):
    from hermes_cli import profiles

    default_home = tmp_path / ".hermes"
    profiles_root = default_home / "profiles"
    worker_home = profiles_root / "worker_alpha"
    for home in (default_home, worker_home):
        (home / "cron").mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text("model: test-model\n", encoding="utf-8")
    monkeypatch.setattr(profiles, "_get_default_hermes_home", lambda: default_home)
    monkeypatch.setattr(profiles, "_get_profiles_root", lambda: profiles_root)
    monkeypatch.setattr(_web_server_cron, "_notify_cron_provider_for_profile", lambda profile: None)
    return {"default": default_home, "worker_alpha": worker_home}


@pytest.mark.asyncio
async def test_create_persists_main_session(isolated_profiles):
    job = await _rt_cron.create_cron_job(
        _web_models.CronJobCreate(prompt="Use the ongoing conversation", schedule="every 1h", session="main"),
        profile="worker_alpha")
    assert job["session"] == "main"


@pytest.mark.asyncio
async def test_update_rejects_an_invalid_session(isolated_profiles):
    job = _web_server_cron._call_cron_for_profile("worker_alpha", "create_job", prompt="x", schedule="every 1h")
    with pytest.raises(HTTPException, match="Invalid session mode"):
        await _rt_cron.update_cron_job(job["id"], _web_models.CronJobUpdate(updates={"session": "fork"}),
                                       profile="worker_alpha")


@pytest.mark.asyncio
async def test_update_null_session_restores_the_config_default(isolated_profiles):
    job = _web_server_cron._call_cron_for_profile("worker_alpha", "create_job", prompt="x", schedule="every 1h",
                                                  session="main")
    updated = await _rt_cron.update_cron_job(job["id"], _web_models.CronJobUpdate(updates={"session": None}),
                                             profile="worker_alpha")
    assert "session" not in updated
