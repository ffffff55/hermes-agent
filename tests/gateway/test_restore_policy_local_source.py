"""Regression for #148473: a peer-refreshed ``source='local'`` receipt must load.

Peer refresh rewrites interactive creation labels to the ``local`` placeholder
(``STORED_AS_LOCAL``). ``restore_policy`` used to treat that stored value as an
unknown surface, so the receipt became ``storage_unavailable`` and the session
stayed paused. Real corruption (unknown source, platform mismatch, and the
other receipt checks) stays refused.
"""
import pytest


def _receipt(tmp_path):
    from dataclasses import asdict
    from gateway.session_policy import build_policy

    policy = build_policy(
        {'cwd': str(tmp_path), 'model': 'm'},
        {'platform_toolsets': {'cli': ['terminal']}},
    )
    return policy, asdict(policy)


def test_restore_policy_loads_stored_local_source(tmp_path):
    from dataclasses import replace
    from gateway.session_policy import restore_policy

    policy, stored = _receipt(tmp_path)
    assert restore_policy(stored) == policy
    restored = restore_policy({**stored, 'source': 'local', 'platform': 'cli'})
    assert restored == replace(policy, source='local', platform='cli')
    assert restored.source == 'local' and restored.platform == 'cli'


@pytest.mark.parametrize('patch', [
    pytest.param({'source': 'bogus'}, id='bogus-source'),
    pytest.param({'source': 'unknown'}, id='unknown-source'),
    pytest.param({'source': 'cli', 'platform': 'desktop'}, id='platform-mismatch'),
    pytest.param({'edit_approval_policy': 'nonsense'}, id='edit-approval'),
    pytest.param({'safe_mode': True, 'ignore_user_config': False}, id='safe-mode-mismatch'),
    pytest.param({'cwd': 'relative/path'}, id='relative-cwd'),
    pytest.param({'model': ''}, id='empty-model'),
    pytest.param({'model': '   '}, id='blank-model'),
    pytest.param({'model': None}, id='missing-model'),
    pytest.param({'source': 'local', 'cwd': 'relative/path'}, id='local-relative-cwd'),
    pytest.param({'source': 'local', 'model': ''}, id='local-empty-model'),
    pytest.param({'source': 'local', 'edit_approval_policy': 'nonsense'}, id='local-edit-approval'),
    pytest.param(
        {'source': 'local', 'safe_mode': True, 'ignore_user_config': False},
        id='local-safe-mode-mismatch',
    ),
])
def test_restore_policy_still_rejects_corrupt_receipts(tmp_path, patch):
    from gateway.session_policy import restore_policy
    from hermes_state_runtime import RuntimeStoreError

    _policy, stored = _receipt(tmp_path)
    with pytest.raises(RuntimeStoreError, match='storage_unavailable'):
        restore_policy({**stored, **patch})
