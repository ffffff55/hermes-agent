"""A stored registry pseudo-endpoint is route identity, not a launch URL (#147676).

``legacy_launch_params`` forwarded ``moa://local`` and ``process://...`` as
``base_url``. ``session.resume`` then died in ``_apply_launch_overrides`` with
``invalid_params`` on that field, so a pre-gateway chat could not be reopened.
Migration drops the pseudo URL when a named provider already owns the route.
The launch validator still refuses a non-http(s) URL a caller passes itself.
"""
import json

import pytest


def _row(tmp_path, model_config, **columns):
    row = {
        'source': 'desktop',
        'cwd': str(tmp_path),
        'model': 'DeepReview',
        'model_config': json.dumps(model_config),
    }
    row.update(columns)
    return row


@pytest.mark.parametrize('provider,base_url', [
    ('moa', 'moa://local'),
    ('proc-provider', 'process://proc-provider'),
])
def test_named_provider_pseudo_endpoint_resumes(tmp_path, provider, base_url):
    """The #147676 snippet: stored route and the same billing stamp, then build_policy."""
    from gateway.session_local_migration import legacy_launch_params
    from gateway.session_policy import build_policy

    row = _row(
        tmp_path,
        {'model': 'DeepReview', 'provider': provider, 'base_url': base_url,
         'api_mode': 'chat_completions'},
        billing_base_url=base_url,
    )
    params, api_mode = legacy_launch_params(row)
    assert params['provider'] == provider and params['model'] == 'DeepReview'
    assert 'base_url' not in params
    assert api_mode == 'chat_completions'
    policy = build_policy(params, {})
    assert policy.provider == provider
    assert policy.base_url != base_url


def test_billing_fallback_does_not_restore_a_named_providers_pseudo_endpoint(tmp_path):
    """The stamp is copied when the stored route has no endpoint; a named provider still drops it."""
    from gateway.session_local_migration import legacy_launch_params
    from gateway.session_policy import build_policy

    row = _row(
        tmp_path,
        {'model': 'DeepReview', 'provider': 'moa'},
        billing_base_url='moa://local',
    )
    params, _api_mode = legacy_launch_params(row)
    assert params['provider'] == 'moa'
    assert 'base_url' not in params
    assert build_policy(params, {}).provider == 'moa'


@pytest.mark.parametrize('base_url', [
    'https://api.example.com/v1',
    'HTTPS://api.example.com/v1',
    'http://127.0.0.1:1234/v1',
])
def test_http_base_url_is_forwarded_unchanged(tmp_path, base_url):
    from gateway.session_local_migration import legacy_launch_params
    from gateway.session_policy import build_policy

    row = _row(tmp_path, {'model': 'DeepReview', 'provider': 'openrouter', 'base_url': base_url})
    params, _api_mode = legacy_launch_params(row)
    assert params['provider'] == 'openrouter'
    assert params['base_url'] == base_url
    policy = build_policy(params, {})
    assert policy.provider == 'openrouter'
    assert policy.base_url == base_url


def test_custom_https_billing_endpoint_stays_on_the_route(tmp_path):
    """A bare custom class plus its http(s) endpoint is still the route the billing fallback records."""
    from gateway.session_local_migration import legacy_launch_params

    url = 'https://localhost:1234/v1'
    row = _row(tmp_path, {}, billing_provider='custom', billing_base_url=url)
    params, _api_mode = legacy_launch_params(row)
    assert params['provider'] == 'custom'
    assert params['base_url'] == url


def test_launch_overrides_still_refuse_a_pseudo_base_url(tmp_path):
    from gateway.session_policy import build_policy
    from hermes_state_runtime import RuntimeStoreError

    with pytest.raises(RuntimeStoreError, match='invalid_params') as caught:
        build_policy({'cwd': str(tmp_path), 'provider': 'moa', 'base_url': 'moa://local'}, {})
    assert caught.value.field == 'base_url'
