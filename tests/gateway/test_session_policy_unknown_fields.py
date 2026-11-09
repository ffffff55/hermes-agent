"""Regression for NousResearch/hermes-agent#151667.

A stored session-policy receipt can name a field the running build's
``LocalSessionPolicy`` does not define (added on the other side of a release,
or since removed). ``restore_policy`` adopts that receipt — known fields
restore, including the required ones — and a genuinely invalid known field
still answers ``storage_unavailable``.
"""
import pytest


def _stored_policy(tmp_path):
    from dataclasses import asdict

    from gateway.session_policy import build_policy

    policy = build_policy({'cwd': str(tmp_path), 'model': 'm'}, {})
    return policy, asdict(policy)


def test_restore_adopts_a_receipt_with_unknown_keys(tmp_path):
    """``fallback_providers`` and any other unknown key restore like a receipt without them."""
    from gateway.session_policy import restore_policy

    policy, stored = _stored_policy(tmp_path)
    extra = {
        **stored,
        'fallback_providers': [{'provider': 'other', 'model': 'elsewhere'}],
        'some_future_field': {'enabled': True},
    }
    adopted = restore_policy(extra)
    assert adopted == restore_policy(stored) == policy
    assert adopted.cwd == stored['cwd']
    assert adopted.model == stored['model']
    assert adopted.source == stored['source']
    assert adopted.platform == stored['platform']
    assert tuple(adopted.toolsets) == tuple(stored['toolsets'])


def test_unknown_key_does_not_skip_validation_of_a_known_field(tmp_path):
    """Dropping an unknown key still validates the fields this build defines.

    The wire reason stays ``storage_unavailable``. The cause is the invalid
    known field, not ``TypeError`` from the extra key.
    """
    from gateway.session_policy import restore_policy
    from hermes_state_runtime import RuntimeStoreError

    _, stored = _stored_policy(tmp_path)
    broken = {**stored, 'fallback_providers': [], 'some_future_field': 1, 'model': '   '}
    with pytest.raises(RuntimeStoreError, match='storage_unavailable') as raised:
        restore_policy(broken)
    assert raised.value.reason == 'storage_unavailable'
    assert type(raised.value.__cause__) is ValueError
