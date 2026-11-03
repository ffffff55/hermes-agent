"""A legacy chat whose stored cwd is already gone keeps that folder.

Regression for #148238: ``legacy_launch_params`` copies the row's cwd into create
params, and ``build_policy`` used to refuse a missing directory as ``invalid_params``
before the turn gate from 2c27f97cb56 (``workspace_unavailable``) could name the
folder and the way back. An existing directory is unchanged.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_state_runtime import RuntimeStoreError


def _authority(tmp_path, monkeypatch):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(run, '_load_gateway_config', lambda *_a, **_k: {
        'platform_toolsets': {'gui': [], 'cli': [], 'acp': []}})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None)
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    return store, runner


async def _owner(tmp_path, monkeypatch):
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection

    store, runner = _authority(tmp_path, monkeypatch)
    authority = await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id='first')
    native = AuthorityConnection(authority, object(), {'provider': 'local', 'user_id': 'native',
        'profile_id': str(tmp_path), 'instance_id': 'first'})
    return store, authority, native


def test_create_still_requires_an_existing_directory(tmp_path):
    """A new session's cwd check is unchanged: missing is invalid_params, a live directory is kept."""
    from gateway.session_policy import build_policy

    missing = tmp_path / 'gone'
    config = {'platform_toolsets': {'cli': []}}
    with pytest.raises(RuntimeStoreError) as refused:
        build_policy({'source': 'cli', 'cwd': str(missing), 'model': 'm'}, config)
    assert (refused.value.reason, refused.value.field) == ('invalid_params', 'cwd')
    live = tmp_path / 'here'
    live.mkdir()
    policy = build_policy({'source': 'cli', 'cwd': str(live), 'model': 'm'}, config)
    assert policy.cwd == str(live)
    # Adoption may keep a missing absolute path. A relative one is still not a workspace.
    kept = build_policy({'source': 'cli', 'cwd': str(missing), 'model': 'm'}, config, frozen_workspace=True)
    assert kept.cwd == str(missing)
    with pytest.raises(RuntimeStoreError) as relative:
        build_policy({'source': 'cli', 'cwd': 'not/absolute', 'model': 'm'}, config, frozen_workspace=True)
    assert (relative.value.reason, relative.value.field) == ('invalid_params', 'cwd')


@pytest.mark.asyncio
async def test_legacy_resume_keeps_a_missing_cwd_and_the_turn_names_it(tmp_path, monkeypatch):
    """Resume adopts the stored folder even when it is already gone; a turn names it.

    The column cwd and an older ACP row's ``model_config`` cwd are the same copy.
    A relative stored cwd is still a bad create param. A live directory is that directory.
    """
    from gateway.config import Platform
    from gateway.run_turn_runner import TurnRunner
    from agent.error_surface import build_error_surface_from_result
    from hermes_state_local import local_receipt

    gone = tmp_path / 'deleted-project'
    acp_gone = tmp_path / 'deleted-editor'
    live = tmp_path / 'still-here'
    live.mkdir()
    store, authority, native = await _owner(tmp_path, monkeypatch)
    db = store._db
    db.create_session('old-gone', source='desktop', model='old-model', cwd=str(gone))
    db.append_message('old-gone', role='user', content='STILL_HERE')
    db.create_session('old-acp', source='acp', model='old-model', model_config={'cwd': str(acp_gone)})
    db.create_session('old-live', source='desktop', model='old-model', cwd=str(live))
    db.create_session('old-relative', source='desktop', model='old-model', cwd='not/absolute')
    db.create_session('old-empty', source='desktop', model='old-model')
    try:
        resumed = await native.dispatch({'id': 1, 'method': 'session.resume',
                                         'params': {'session_id': 'old-gone'}})
        assert resumed.get('result', {}).get('session_id') == 'old-gone', resumed
        assert [m['content'] for m in resumed['result']['messages']] == ['STILL_HERE']
        policy = authority.runner.adapters[Platform.LOCAL].policies['old-gone']
        assert policy.cwd == str(gone) and not gone.is_dir()
        assert local_receipt(db, 'old-gone')['policy']['cwd'] == str(gone)

        source = authority.sessions['old-gone'].source
        result = TurnRunner(authority.runner, SimpleNamespace(
            source=source, session_id='old-gone', session_key=None)).run_sync()
        assert result['failed'] and not result['completed'] and result['api_calls'] == 0, result
        assert result['failure_reason'] == 'workspace_unavailable' and result['failure_retryable'] is False
        assert str(gone) in result['final_response'], result['final_response']
        surface = build_error_surface_from_result(result)
        assert (surface['layer'], surface['code'], surface['retryable']) == (
            'gateway', 'workspace_unavailable', False)

        acp = await native.dispatch({'id': 2, 'method': 'session.resume',
                                     'params': {'session_id': 'old-acp'}})
        assert acp.get('result', {}).get('session_id') == 'old-acp', acp
        assert authority.runner.adapters[Platform.LOCAL].policies['old-acp'].cwd == str(acp_gone)

        kept = await native.dispatch({'id': 3, 'method': 'session.resume',
                                      'params': {'session_id': 'old-live'}})
        assert kept.get('result', {}).get('session_id') == 'old-live', kept
        assert authority.runner.adapters[Platform.LOCAL].policies['old-live'].cwd == str(live)

        relative = await native.dispatch({'id': 4, 'method': 'session.resume',
                                          'params': {'session_id': 'old-relative'}})
        assert relative['error']['data']['reason'] == 'invalid_params', relative
        assert relative['error']['data']['field'] == 'cwd'
        with pytest.raises(RuntimeStoreError, match='storage_unavailable'):
            local_receipt(db, 'old-relative')

        empty = await native.dispatch({'id': 5, 'method': 'session.resume',
                                       'params': {'session_id': 'old-empty'}})
        assert empty.get('result', {}).get('session_id') == 'old-empty', empty
        fallback = authority.runner.adapters[Platform.LOCAL].policies['old-empty'].cwd
        assert fallback != str(gone) and Path(fallback).is_dir()
    finally:
        await native.close()
        store.close_all_db_handles()
