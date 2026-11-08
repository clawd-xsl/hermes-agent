"""`hermes cron create/edit --session` sets, shows and clears the per-job conversation mode."""
import argparse

import pytest

from cron.jobs import get_job, list_jobs
from hermes_cli.cron import cron_command
from hermes_cli.subcommands.cron import build_cron_parser


@pytest.fixture()
def tmp_cron_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    return tmp_path


def _run(*argv):
    parser = argparse.ArgumentParser(prog="hermes")
    build_cron_parser(parser.add_subparsers(dest="command"), cmd_cron=cron_command)
    return cron_command(parser.parse_args(["cron", *argv]))


def test_create_edit_and_default(tmp_cron_dir, capsys):
    assert _run("create", "every 1h", "Use our conversation", "--session", "main") in (0, None)
    assert "Session: main" in capsys.readouterr().out
    (job,) = list_jobs(include_disabled=True)
    assert job["session"] == "main"
    assert _run("edit", job["id"], "--session", "isolated") == 0
    assert get_job(job["id"])["session"] == "isolated"
    assert _run("edit", job["id"], "--session", "default") == 0
    assert "session" not in get_job(job["id"])


def test_unknown_mode_is_refused_by_the_parser(tmp_cron_dir):
    with pytest.raises(SystemExit):
        _run("create", "every 1h", "x", "--session", "fork")
