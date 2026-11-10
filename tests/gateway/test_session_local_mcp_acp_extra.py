"""A missing ``[acp]`` extra is a named store refusal, not an uncaught import error.

Regression for #150409: ``validate_servers`` imported ``acp.schema`` unconditionally, so
ACP editor MCP validation without the optional extra surfaced as JSON-RPC ``-32603`` and
the ``ModuleNotFoundError`` only appeared in the gateway log.
"""
import sys

import pytest

from gateway.session_local_mcp import validate_servers
from hermes_state_runtime import RuntimeStoreError


class _MissingAcp:
    """Force the ``acp`` import path to fail the way a missing extra does."""

    def find_spec(self, fullname, path, target=None):
        if fullname == 'acp' or fullname.startswith('acp.'):
            raise ModuleNotFoundError("No module named 'acp'", name='acp')
        # Not our module: fall through (implicit None) so the next finder runs.


def test_a_missing_acp_extra_is_a_named_store_refusal(monkeypatch):
    for name in [key for key in sys.modules if key == 'acp' or key.startswith('acp.')]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(sys, 'meta_path', [_MissingAcp(), *sys.meta_path])

    with pytest.raises(RuntimeStoreError, match='acp_extra_missing') as caught:
        validate_servers([])

    refusal = caught.value
    assert refusal.reason == 'acp_extra_missing'
    assert refusal.field == 'editor.mcp_servers'
    assert refusal.detail == "the '[acp]' extra (agent-client-protocol) is not installed"
    assert isinstance(refusal.__cause__, ModuleNotFoundError)
