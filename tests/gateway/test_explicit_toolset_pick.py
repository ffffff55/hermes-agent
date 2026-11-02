"""An explicit ``-t`` pick is the session's toolset set (#147832).

``_get_platform_tools`` auto-unions plugin toolsets missing from
``known_plugin_toolsets`` and every enabled MCP server. ``_resolve_toolsets``
drops the ones the pick did not name. A launch with no pick keeps that union.
"""


def _plugin_keys():
    return {'plugin_extra'}


def _config(tmp_path):
    absent = str(tmp_path / 'absent')
    return {
        'platform_toolsets': {'cli': ['terminal', 'file']},
        'mcp_servers': {
            'extra_srv': {'command': absent},
            'kept_srv': {'command': absent},
        },
    }


def _policy(tmp_path, config, **params):
    from gateway.session_policy import build_policy
    return build_policy({'source': 'cli', 'cwd': str(tmp_path), **params}, config)


def test_explicit_pick_drops_unnamed_plugin_and_mcp_toolsets(tmp_path, monkeypatch):
    """``hermes chat -t session_search`` keeps that toolset and nothing the pick omitted.

    A named plugin and an ``mcp-<server>`` spelling stay; the other enabled server does not.
    """
    monkeypatch.setattr('hermes_cli.tools_config._get_plugin_toolset_keys', _plugin_keys)
    monkeypatch.setattr('hermes_cli.plugins.get_plugin_toolset_keys_nowait', _plugin_keys)
    config = _config(tmp_path)

    bare = set(_policy(tmp_path, config, toolsets=['session_search']).toolsets)
    assert 'session_search' in bare
    assert not {'plugin_extra', 'extra_srv', 'kept_srv', 'terminal', 'file'} & bare

    named = set(_policy(
        tmp_path, config, toolsets=['session_search', 'plugin_extra', 'mcp-kept_srv'],
    ).toolsets)
    assert {'session_search', 'plugin_extra', 'kept_srv'} <= named
    assert 'extra_srv' not in named


def test_no_explicit_pick_keeps_plugin_and_mcp_defaults(tmp_path, monkeypatch):
    """No ``-t`` still inherits the profile list plus auto-unioned plugins and MCP servers."""
    monkeypatch.setattr('hermes_cli.tools_config._get_plugin_toolset_keys', _plugin_keys)
    monkeypatch.setattr('hermes_cli.plugins.get_plugin_toolset_keys_nowait', _plugin_keys)
    picked = set(_policy(tmp_path, _config(tmp_path)).toolsets)
    assert {'terminal', 'file', 'plugin_extra', 'extra_srv', 'kept_srv'} <= picked
