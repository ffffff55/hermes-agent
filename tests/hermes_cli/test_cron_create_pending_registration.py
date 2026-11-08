"""Saved cron jobs must not be reported as a failed create.

Regression for #150973. ``hermes cron create`` (and ``/cron add``) used to
print ``Failed to create job:`` plus the raw ``CronSchedulerRegistrationError``
and exit 1 after the job was already stored. Re-running create then duplicated
it. The human line is ``user_message()``; the exit code must not say the job
is absent.
"""

from types import SimpleNamespace

import pytest

from hermes_cli import cron as cron_cli


def _create_args(**overrides):
    args = dict(
        schedule="every 2h",
        prompt="probe",
        name="probe-test",
        deliver="local",
        repeat=None,
        skill=None,
        skills=None,
        script=None,
        workdir=None,
        no_agent=False,
    )
    args.update(overrides)
    return SimpleNamespace(**args)


@pytest.fixture()
def tmp_cron_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    return tmp_path


@pytest.fixture()
def failing_external_scheduler(monkeypatch):
    """Registration raises after the job is persisted; provider details stay private."""

    def register_job(job):
        del job
        raise RuntimeError("private callback URL and token")

    monkeypatch.setattr(
        "cron.scheduler_provider.resolve_cron_scheduler",
        lambda: SimpleNamespace(name="chronos", register_job=register_job),
    )


def test_cron_create_pending_registration_is_not_a_total_failure(monkeypatch, capsys):
    monkeypatch.setattr(cron_cli, "_active_cron_provider_name", lambda: "chronos")
    monkeypatch.setattr(
        cron_cli,
        "_cron_api",
        lambda **kwargs: {
            "success": False,
            "error": (
                "Cron job 'abc' was saved, but its first scheduler registration "
                "failed (RuntimeError). Do not create a duplicate."
            ),
            "user_message": (
                "Saved cron job 'probe-test', but couldn't register it with the "
                "external scheduler yet. The job is kept — don't re-create it; "
                "pause/resume or edit it (e.g. via /cron) to retry registration."
            ),
            "job_id": "abc",
            "name": "probe-test",
            "schedule": "every 120m",
            "next_run_at": "2026-11-08T01:37:00+00:00",
            "job": {"enabled": True, "id": "abc"},
            "job_saved": True,
            "scheduler_registered": False,
            "retry_create": False,
        },
    )

    rc = cron_cli.cron_create(_create_args())

    out = capsys.readouterr().out
    assert rc == 0
    assert "Created job: abc" in out
    assert "Saved cron job 'probe-test'" in out
    assert "don't re-create" in out
    assert "Failed to create job" not in out
    assert "RuntimeError" not in out
    assert "Next run: 2026-11-08T01:37:00+00:00" in out


def test_cron_create_keeps_real_failures(monkeypatch, capsys):
    monkeypatch.setattr(
        cron_cli, "_cron_api", lambda **kwargs: {"success": False, "error": "boom"}
    )

    rc = cron_cli.cron_create(_create_args())

    out = capsys.readouterr().out
    assert rc == 1
    assert "Failed to create job: boom" in out
    assert "Created job" not in out


def test_cron_create_reports_saved_job_when_registration_fails(
    tmp_cron_dir, failing_external_scheduler, capsys
):
    from cron.jobs import list_jobs

    rc = cron_cli.cron_create(_create_args())

    out = capsys.readouterr().out
    saved = [job for job in list_jobs(include_disabled=True) if job.get("name") == "probe-test"]
    assert len(saved) == 1
    assert rc == 0
    assert f"Created job: {saved[0]['id']}" in out
    assert "Saved cron job 'probe-test'" in out
    assert "don't re-create" in out
    assert "Failed to create job" not in out
    assert "RuntimeError" not in out
    assert "private callback URL and token" not in out
    assert saved[0].get("next_run_at")
    assert str(saved[0]["next_run_at"]) in out


def test_slash_cron_add_reports_saved_job_when_registration_fails(
    tmp_cron_dir, failing_external_scheduler
):
    from cron.jobs import list_jobs
    from hermes_cli.cli_cron_lines import cron_command_lines

    lines = cron_command_lines('add "every 2h" "probe" --name probe-test')
    text = "\n".join(lines)
    saved = [job for job in list_jobs(include_disabled=True) if job.get("name") == "probe-test"]

    assert len(saved) == 1
    assert "Created job:" in text
    assert saved[0]["id"] in text
    assert "Saved cron job 'probe-test'" in text
    assert "don't re-create" in text
    assert "Failed to create job" not in text
    assert "RuntimeError" not in text
    assert "private callback URL and token" not in text
