"""Regression for #148885: per-turn browser cleanup must stop the vault CDP supervisor.

``browser_exec`` attaches that supervisor under the bare task id, while a
served-profile gateway turn stores the session (and its creation supervisor)
under ``<task>@<home>``. Cleanup has to stop both before the cloud browser is
released. Outside a served-profile scope the two keys are the same id.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

from hermes_constants import (
    get_hermes_home_override,
    reset_hermes_home_override,
    set_hermes_home_override,
)
from tools import browser_session as bs
from tools import browser_session_lifecycle as lifecycle
from tools.browser_supervisor import SUPERVISOR_REGISTRY
from tools.browser_use_cli import _backend_cache_key

_BB = "bb-1"


def _supervisor(task_id: str, events: list) -> SimpleNamespace:
    return SimpleNamespace(stop=lambda task_id=task_id: events.append(("stop", task_id)))


def _arm(monkeypatch, sessions: dict, supervisor_ids: list[str], events: list) -> dict:
    """Install sessions plus fake CDP supervisors, and record cloud-session close."""
    monkeypatch.setattr(bs, "_active_sessions", sessions)
    monkeypatch.setattr(bs, "_session_last_activity", dict.fromkeys(sessions, 1.0))
    monkeypatch.setattr(bs, "_session_owner_homes", {})
    monkeypatch.setattr(bs, "_cleanup_failures", {})
    registry = {task_id: _supervisor(task_id, events) for task_id in supervisor_ids}
    monkeypatch.setattr(SUPERVISOR_REGISTRY, "_by_task", registry)
    monkeypatch.setattr(SUPERVISOR_REGISTRY, "_home_by_task", {})
    provider = Mock()
    provider.close_session.side_effect = lambda sid: events.append(("close", sid))
    monkeypatch.setattr("tools.browser_session_cloud.get_cloud_provider", lambda: provider)
    return registry


def _cloud() -> dict:
    return {
        "bb_session_id": _BB,
        "cdp_url": "ws://127.0.0.1:9/devtools/browser/x",
        "features": {},
    }


def test_served_profile_cleanup_stops_bare_task_vault_supervisor(tmp_path, monkeypatch):
    """A gateway turn's tagged session and the bare-task vault supervisor both stop
    before the cloud browser is closed. A different task's supervisor stays."""
    home = tmp_path / "served"
    home.mkdir()
    token = set_hermes_home_override(str(home))
    try:
        task = "sup-repro"
        cache_key = _backend_cache_key(task)
        assert cache_key != task and "@" in cache_key
        events: list = []
        sessions = {cache_key: _cloud()}
        registry = _arm(monkeypatch, sessions, [cache_key, task, "other-task"], events)

        lifecycle.cleanup_browser(task)

        assert events == [("stop", cache_key), ("stop", task), ("close", _BB)]
        assert list(registry) == ["other-task"]
        assert sessions == {}
    finally:
        reset_hermes_home_override(token)


def test_cleanup_without_profile_override_leaves_no_supervisor(monkeypatch):
    """Without a served-profile tag the session key is the bare task id, so the one
    supervisor is the vault supervisor and cleanup leaves none."""
    assert get_hermes_home_override() is None
    task = "sup-repro"
    cache_key = _backend_cache_key(task)
    assert cache_key == task
    events: list = []
    sessions = {cache_key: _cloud()}
    registry = _arm(monkeypatch, sessions, [cache_key], events)

    lifecycle.cleanup_browser(task)

    assert events == [("stop", task), ("close", _BB)]
    assert registry == {}
    assert sessions == {}


def test_untagged_sidecar_cleanup_keeps_the_primary_supervisor(monkeypatch):
    """``cleanup_browser`` of a hybrid ``::local`` key reaps only that sidecar.

    The sidecar's owner id is the primary session's supervisor key. Stopping it
    would drop the live browser's CDP supervisor.
    """
    assert get_hermes_home_override() is None
    task = "sup-repro"
    sidecar = task + bs.LOCAL_SUFFIX
    events: list = []
    sessions = {task: _cloud(), sidecar: {"features": {"local": True}}}
    registry = _arm(monkeypatch, sessions, [task, sidecar], events)

    lifecycle.cleanup_browser(sidecar)

    assert events == [("stop", sidecar)]
    assert list(registry) == [task]
    assert list(sessions) == [task]
