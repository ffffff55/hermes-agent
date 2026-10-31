"""Due-scan grace gate for one-shot cron jobs.

create_job / update_job / resume_job all reject a one-shot whose run time is
more than ONESHOT_GRACE_SECONDS in the past ("will never fire"), and
``_recoverable_oneshot_run_at`` never recovers such a schedule — but the
due-scan (``_get_due_jobs_locked``) used to dispatch any one-shot whose
*persisted* ``next_run_at`` was in the past, even hours later (gateway down
past the window, host asleep, hand-edited jobs.json). That fired a
wall-clock one-shot late, contradicting the "will never fire" contract.

Regression for #147102: a run_claim or fire_claim left by a dead process
(or past its TTL) used to count as in-flight forever, so the one-shot was
neither fired nor retired and never got a missed-run diagnostic.

These tests pin the due-scan grace gate:
  - beyond grace  -> not due; record retired with a diagnostic
  - within grace  -> still due (legitimate catch-up)
  - beyond grace + live claim -> not due, record kept (mark_job_run can land)
  - beyond grace + dead or TTL-expired claim -> retired with the missed diagnostic
  - beyond grace + outage claim and no fire_claim -> still due (recovery fire)
  - re-triggered  -> due again (the Run button still works)
"""

import socket
import subprocess
import sys

import pytest
from datetime import datetime, timedelta, timezone, UTC

from cron.jobs import (
    FIRE_CLAIM_TTL_SECONDS,
    _oneshot_run_claim_ttl_seconds,
    get_due_jobs,
    load_jobs,
    save_jobs,
    trigger_job,
)

FIXED_NOW = datetime(2026, 6, 22, 12, 0, 0, tzinfo=UTC)


@pytest.fixture()
def cron_store(tmp_path, monkeypatch):
    """Redirect cron storage to a temp dir and pin the clock."""
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    monkeypatch.setattr("cron.jobs._hermes_now", lambda: FIXED_NOW)
    return tmp_path


def _dead_owner() -> str:
    """``host:pid`` of a process that has already exited, on this machine."""
    proc = subprocess.run(
        [sys.executable, "-c", "import os; print(os.getpid())"],
        capture_output=True,
        text=True,
        check=True,
    )
    return f"{socket.gethostname()}:{int(proc.stdout.strip())}"


def _diagnostics(cron_store, jid):
    out_dir = cron_store / "cron" / "output" / jid
    return list(out_dir.glob("*.md")) if out_dir.exists() else []


def _oneshot(jid, run_at_dt, *, completed=0, run_claim=None, fire_claim=None):
    return {
        "id": jid,
        "name": jid,
        "prompt": "x",
        "schedule": {"kind": "once", "run_at": run_at_dt.isoformat()},
        "next_run_at": run_at_dt.isoformat(),
        "last_run_at": None,
        "enabled": True,
        "state": "scheduled",
        "repeat": {"times": 1, "completed": completed},
        "deliver": "local",
        "run_claim": run_claim,
        "fire_claim": fire_claim,
    }


class TestOneShotGraceDueScan:
    def test_stale_beyond_grace_not_due_and_retired_with_diagnostic(self, cron_store):
        stale = _oneshot("stale", FIXED_NOW - timedelta(hours=3))
        save_jobs([stale])

        due = get_due_jobs()

        assert [d["id"] for d in due] == []
        # Record is retired (removed) so it stops being scanned — with a
        # diagnostic left behind so the miss is operator-visible.
        assert [j["id"] for j in load_jobs()] == []
        out_dir = cron_store / "cron" / "output" / "stale"
        diag = list(out_dir.glob("*.md")) if out_dir.exists() else []
        assert diag, "expected a diagnostic file for the retired stale one-shot"

    def test_within_grace_still_due(self, cron_store):
        recent = _oneshot("recent", FIXED_NOW - timedelta(seconds=60))
        save_jobs([recent])

        due = get_due_jobs()
        assert [d["id"] for d in due] == ["recent"]
        # Still stored (dispatch proceeds normally).
        assert [j["id"] for j in load_jobs()] == ["recent"]

    def test_live_run_claim_skips_without_retiring(self, cron_store, monkeypatch):
        # A live run_claim means a run is still in flight elsewhere — do not
        # fire or retire, so mark_job_run can still land.
        missed = []
        monkeypatch.setattr(
            "cron.jobs.record_cron_missed", lambda job: missed.append(job.get("id"))
        )
        claimed = _oneshot(
            "claimed",
            FIXED_NOW - timedelta(hours=3),
            completed=1,
            run_claim={
                "at": (FIXED_NOW - timedelta(seconds=30)).isoformat(),
                "by": "other-host",
            },
        )
        save_jobs([claimed])

        due = get_due_jobs()
        assert [d["id"] for d in due] == []
        assert [j["id"] for j in load_jobs()] == ["claimed"]
        assert _diagnostics(cron_store, "claimed") == []
        assert missed == []

    def test_live_fire_claim_skips_without_retiring(self, cron_store, monkeypatch):
        missed = []
        monkeypatch.setattr(
            "cron.jobs.record_cron_missed", lambda job: missed.append(job.get("id"))
        )
        claimed = _oneshot(
            "firing",
            FIXED_NOW - timedelta(hours=3),
            run_claim={
                "at": (FIXED_NOW - timedelta(
                    seconds=_oneshot_run_claim_ttl_seconds() + 1
                )).isoformat(),
                "by": "other-host",
            },
            fire_claim={
                "at": (FIXED_NOW - timedelta(seconds=30)).isoformat(),
                "by": "other-host",
            },
        )
        save_jobs([claimed])

        due = get_due_jobs()
        assert [d["id"] for d in due] == []
        assert [j["id"] for j in load_jobs()] == ["firing"]
        assert _diagnostics(cron_store, "firing") == []
        assert missed == []

    def test_expired_run_claim_is_retired_with_missed_diagnostic(
        self, cron_store, monkeypatch
    ):
        missed = []
        monkeypatch.setattr(
            "cron.jobs.record_cron_missed", lambda job: missed.append(job.get("id"))
        )
        expired = _oneshot(
            "expired",
            FIXED_NOW - timedelta(hours=3),
            run_claim={
                "at": (FIXED_NOW - timedelta(
                    seconds=_oneshot_run_claim_ttl_seconds() + 1
                )).isoformat(),
                "by": "other-host",
            },
        )
        save_jobs([expired])

        due = get_due_jobs()
        assert [d["id"] for d in due] == []
        assert [j["id"] for j in load_jobs()] == []
        diag = _diagnostics(cron_store, "expired")
        assert len(diag) == 1
        text = diag[0].read_text(encoding="utf-8")
        assert "removed before firing" in text
        assert "removed without producing output" not in text
        assert missed == ["expired"]

    def test_dead_owner_run_claim_is_retired_with_missed_diagnostic(
        self, cron_store, monkeypatch
    ):
        # Fresh timestamp, so only owner death (not the TTL) makes it stale.
        missed = []
        monkeypatch.setattr(
            "cron.jobs.record_cron_missed", lambda job: missed.append(job.get("id"))
        )
        dead = _oneshot(
            "dead",
            FIXED_NOW - timedelta(hours=3),
            completed=1,
            run_claim={
                "at": (FIXED_NOW - timedelta(seconds=30)).isoformat(),
                "by": _dead_owner(),
            },
        )
        save_jobs([dead])

        due = get_due_jobs()
        assert [d["id"] for d in due] == []
        assert [j["id"] for j in load_jobs()] == []
        diag = _diagnostics(cron_store, "dead")
        assert len(diag) == 1
        text = diag[0].read_text(encoding="utf-8")
        assert "removed before firing" in text
        assert "removed without producing output" not in text
        assert missed == ["dead"]

    def test_expired_fire_claim_is_retired_with_missed_diagnostic(
        self, cron_store, monkeypatch
    ):
        missed = []
        monkeypatch.setattr(
            "cron.jobs.record_cron_missed", lambda job: missed.append(job.get("id"))
        )
        expired = _oneshot(
            "expired-fire",
            FIXED_NOW - timedelta(hours=3),
            fire_claim={
                "at": (FIXED_NOW - timedelta(
                    seconds=FIRE_CLAIM_TTL_SECONDS + 1
                )).isoformat(),
                "by": "other-host",
            },
        )
        save_jobs([expired])

        due = get_due_jobs()
        assert [d["id"] for d in due] == []
        assert [j["id"] for j in load_jobs()] == []
        diag = _diagnostics(cron_store, "expired-fire")
        assert len(diag) == 1
        assert "removed before firing" in diag[0].read_text(encoding="utf-8")
        assert missed == ["expired-fire"]

    def test_outage_claim_without_fire_claim_still_fires(self, cron_store, monkeypatch):
        # Store-outage recovery: an outage marker, even with no timestamp or an
        # already-expired one, never started a run. It stays due past grace.
        missed = []
        monkeypatch.setattr(
            "cron.jobs.record_cron_missed", lambda job: missed.append(job.get("id"))
        )
        marker = _oneshot(
            "outage-marker",
            FIXED_NOW - timedelta(hours=3),
            run_claim={"outage": True},
        )
        expired = _oneshot(
            "outage-expired",
            FIXED_NOW - timedelta(hours=3),
            run_claim={
                "at": (FIXED_NOW - timedelta(
                    seconds=_oneshot_run_claim_ttl_seconds() + 1
                )).isoformat(),
                "by": "other-host",
                "outage": True,
            },
        )
        save_jobs([marker, expired])

        due = get_due_jobs()
        assert sorted(d["id"] for d in due) == ["outage-expired", "outage-marker"]
        assert sorted(j["id"] for j in load_jobs()) == ["outage-expired", "outage-marker"]
        assert missed == []

    def test_retriggered_stale_oneshot_is_due(self, cron_store):
        # A user can still explicitly re-run a stale one-shot: trigger_job
        # sets next_run_at=now, which is inside the grace window -> due.
        stale = _oneshot("stale", FIXED_NOW - timedelta(hours=3))
        save_jobs([stale])
        trigger_job("stale")

        due = get_due_jobs()
        assert [d["id"] for d in due] == ["stale"]

    def test_recurring_jobs_unaffected(self, cron_store):
        # The grace gate is once-kind only; a stale recurring job keeps its
        # existing execute-now catch-up behavior.
        interval = {
            "id": "int",
            "name": "int",
            "prompt": "x",
            "schedule": {"kind": "interval", "minutes": 60},
            "next_run_at": (FIXED_NOW - timedelta(hours=2)).isoformat(),
            "last_run_at": None,
            "enabled": True,
            "state": "scheduled",
            "repeat": {"times": None, "completed": 0},
            "deliver": "local",
        }
        save_jobs([interval])
        due = get_due_jobs()
        # Recurring stale handling still executes once now (existing behavior).
        assert [d["id"] for d in due] == ["int"]
