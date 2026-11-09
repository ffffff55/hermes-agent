"""Receipt recovery must read the owning profile's secrets, and stop when the job is gone.

Regression for NousResearch/hermes-agent#151918. ``reconcile_pending`` composes
delivery on the ticker thread, which binds the profile home but not a secret
scope. On a multiplexed gateway ``get_secret('SLACK_HOME_CHANNEL')`` (Slack
origin thread staleness) raises ``UnscopedSecretError``, the admission is left
in place, and every tick retries. A deleted job has nothing left to deliver, so
that admission is consumed once instead of deferring forever.
"""
from contextlib import asynccontextmanager
import json
import logging

import pytest

from agent import secret_scope
from cron import delivery_queue, executions, jobs, scheduler_authority
from hermes_cli import gateway_client
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


def _peer(monkeypatch, answer):
    class Peer:
        async def rpc(self, method, **params):
            return answer(method, params)

    @asynccontextmanager
    async def connected():
        yield Peer()

    monkeypatch.setattr(gateway_client, "connect_gateway", connected)


def _multiplex_profile(tmp_path, monkeypatch):
    """Launch home in the process env, a different profile home on disk.

    Multiplex is on and no secret scope is installed — the ticker's state when
    it calls ``reconcile_pending``. The launch env carries a different Slack
    home channel so a scope miss cannot look like the profile value.
    """
    launch = tmp_path / "launch"
    profile = tmp_path / "profiles" / "work"
    launch.mkdir()
    (profile / "cron").mkdir(parents=True)
    (profile / ".env").write_text("SLACK_HOME_CHANNEL=C_PROFILE\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setenv("SLACK_HOME_CHANNEL", "C_LAUNCH")
    secret_scope.set_multiplex_active(True)
    return profile


def _pending_slack_admission(profile, monkeypatch):
    """A dead firer's terminal receipt for a Slack-origin job in ``profile``."""
    origin = {"platform": "slack", "chat_id": "C_PROFILE", "thread_id": "1700000000.000100"}
    token = set_hermes_home_override(str(profile))
    monkeypatch.setattr("cron.delivery_queue.DELIVERY_DB", profile / "cron" / "deliveries.db")
    store = jobs.use_cron_store(profile / "cron")
    store.__enter__()

    job = jobs.create_job(prompt="report", schedule="every 1h", deliver="origin", origin=origin)
    execution = executions.create_execution(job["id"], source="test")
    executions.mark_execution_running(execution["id"])
    with executions._transaction() as conn:
        conn.execute(
            "UPDATE executions SET process_id=?,pid=? WHERE id=?",
            ("old-owner", 999999, execution["id"]),
        )
    params = {"job_id": job["id"], "request_id": execution["id"], "extra_prompt": None}
    journal = scheduler_authority.journal_path(job["id"], execution["id"])
    journal.parent.mkdir(parents=True, exist_ok=True)
    journal.write_text(json.dumps({"params": params, "receipt": None}))
    return token, store, job, execution, journal


def _close_profile(token, store):
    store.__exit__(None, None, None)
    reset_hermes_home_override(token)
    secret_scope.set_multiplex_active(False)


def test_receipt_recovery_reads_profile_slack_home_under_multiplex(tmp_path, monkeypatch):
    """A pending Slack-origin receipt resolves the owning profile's home channel
    and is consumed. Unscoped, the same read raises and the admission stays."""
    profile = _multiplex_profile(tmp_path, monkeypatch)
    token, store, job, execution, journal = _pending_slack_admission(profile, monkeypatch)
    seen = {}

    def answer(method, params):
        return {"status": "terminal", "job": job, "result": [True, "document", "the answer", None]}

    _peer(monkeypatch, answer)

    real_enqueue = delivery_queue.enqueue

    def enqueue(execution_id, queued_job, content, **kwargs):
        # Inside the delivery the recovery path is composing right now.
        seen["secret"] = secret_scope.get_secret("SLACK_HOME_CHANNEL", "")
        seen["scope_home"] = secret_scope.current_secret_scope_home()
        from cron.scheduler_delivery import _env_home_target_chat_id, _resolve_delivery_targets
        seen["home"] = _env_home_target_chat_id("slack")
        seen["targets"] = _resolve_delivery_targets(queued_job)
        return real_enqueue(execution_id, queued_job, content, **kwargs)

    monkeypatch.setattr(delivery_queue, "enqueue", enqueue)
    try:
        assert secret_scope.current_secret_scope() is None
        scheduler_authority.reconcile_pending()
        assert seen["secret"] == "C_PROFILE"
        assert seen["home"] == "C_PROFILE"
        assert seen["scope_home"] == str(journal.parents[2])
        assert seen["targets"][0]["chat_id"] == "C_PROFILE"
        assert seen["targets"][0]["thread_id"] is None  # stale Slack thread dropped via the profile home
        assert not journal.exists()
        saved = executions.get_execution(execution["id"])
        assert saved["status"] == "completed"
        assert saved["delivery_outcome"] == "queued"
        assert secret_scope.current_secret_scope() is None
        with pytest.raises(secret_scope.UnscopedSecretError):
            secret_scope.get_secret("SLACK_HOME_CHANNEL")
    finally:
        _close_profile(token, store)


def test_receipt_recovery_consumes_admission_when_job_is_gone(tmp_path, monkeypatch, caplog):
    """A deleted job's admission is logged once and consumed. The next tick does
    not recover, deliver, or warn again."""
    profile = _multiplex_profile(tmp_path, monkeypatch)
    token, store, job, execution, journal = _pending_slack_admission(profile, monkeypatch)
    jobs.remove_job(job["id"])
    calls = []

    def answer(method, params):
        calls.append(method)
        return {"status": "terminal", "job": job, "result": [True, "document", "the answer", None]}

    _peer(monkeypatch, answer)

    enqueued = []
    monkeypatch.setattr(
        delivery_queue, "enqueue",
        lambda *args, **kwargs: enqueued.append(args) or None,
    )
    try:
        with caplog.at_level(logging.WARNING, logger="cron.scheduler_authority"):
            scheduler_authority.reconcile_pending()
            scheduler_authority.reconcile_pending()
        assert not journal.exists()
        assert calls == []
        assert enqueued == []
        stopped = [
            record for record in caplog.records
            if record.name == "cron.scheduler_authority" and "no longer exists" in record.getMessage()
        ]
        deferred = [record for record in caplog.records if "deferred" in record.getMessage()]
        assert len(stopped) == 1
        assert str(journal) in stopped[0].getMessage()
        assert deferred == []
        # The run is not re-booked against a record that is gone.
        assert executions.get_execution(execution["id"])["status"] == "running"
    finally:
        _close_profile(token, store)
