"""Opening a session that never wrote a local receipt skips policy refresh (#148456).

Desktop resume of a Telegram (or cron/kanban) session calls ``refresh_resumed_policy``.
Those lanes have no ``gateway.local_policy.v1`` receipt by design. ``local_receipt``
raises ``storage_unavailable`` for that missing key and for a corrupt value; only the
missing key is "this session takes no refresh". A present but unreadable receipt still
surfaces.
"""
import asyncio
import logging
from types import SimpleNamespace

import pytest

from hermes_state import SessionDB
from hermes_state_local import POLICY_PREFIX
from hermes_state_runtime import RuntimeStoreError


def _session(tmp_path, source):
    db = SessionDB(db_path=tmp_path / 'state.db')
    sid = f'{source}-open'
    db.create_session(sid, source=source, session_key=f'agent:main:{source}:dm:1', chat_id='1')
    return db, SimpleNamespace(db=db), SimpleNamespace(session_id=sid), sid


def _warnings(caplog):
    return [record for record in caplog.records if 'keeps its stored policy' in record.getMessage()]


@pytest.mark.parametrize('source', ['telegram', 'cron', 'kanban'])
def test_refresh_skips_a_session_with_no_local_receipt(tmp_path, caplog, source):
    """No receipt means no refresh: ``_prepare`` returns None and resume logs nothing."""
    from gateway.session_policy_refresh import _prepare, commit_refresh, refresh_resumed_policy
    db, authority, ref, sid = _session(tmp_path, source)
    generation = db.get_session(sid)['runtime_generation']
    try:
        assert db.get_meta(POLICY_PREFIX + sid) is None
        assert _prepare(authority, sid) is None
        with caplog.at_level(logging.DEBUG, logger='gateway.session_policy_refresh'):
            resumed = asyncio.run(refresh_resumed_policy(authority, ref))
            committed = asyncio.run(commit_refresh(authority, ref))
        assert resumed is None
        assert committed is False
        assert _warnings(caplog) == []
        assert db.get_session(sid)['runtime_generation'] == generation
        assert db.get_meta(POLICY_PREFIX + sid) is None
    finally:
        db.close()


@pytest.mark.parametrize('payload', ['not-json', '[]'])
def test_a_corrupt_receipt_is_not_swallowed(tmp_path, caplog, payload):
    """A stored receipt that does not parse still raises, and resume warns."""
    from gateway.session_policy_refresh import _prepare, commit_refresh, refresh_resumed_policy
    db, authority, ref, sid = _session(tmp_path, 'telegram')
    db.set_meta(POLICY_PREFIX + sid, payload)
    try:
        with pytest.raises(RuntimeStoreError, match='storage_unavailable') as raised:
            _prepare(authority, sid)
        assert raised.value.reason == 'storage_unavailable'
        assert raised.value.__cause__ is not None
        with pytest.raises(RuntimeStoreError, match='storage_unavailable'):
            asyncio.run(commit_refresh(authority, ref))
        with caplog.at_level(logging.DEBUG, logger='gateway.session_policy_refresh'):
            assert asyncio.run(refresh_resumed_policy(authority, ref)) is None
        warnings = _warnings(caplog)
        assert [(record.levelno, record.getMessage()) for record in warnings] == [
            (logging.WARNING, f'Session {sid} keeps its stored policy on resume: storage_unavailable')]
    finally:
        db.close()
