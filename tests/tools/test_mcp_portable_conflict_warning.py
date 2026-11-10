"""Same-name native MCP entries are an intended override of a portable server.

Regression for #152315: a long-running gateway reloads MCP config about once a
minute, and the conflict / rejected-native warnings were logged on every load.
"""

import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tools import mcp_tool_config as mcp_config


@pytest.fixture(autouse=True)
def _reset_portable_conflict_dedupe():
    warned = getattr(mcp_config, "_portable_conflict_warned", None)
    if isinstance(warned, set):
        warned.clear()
    yield
    warned = getattr(mcp_config, "_portable_conflict_warned", None)
    if isinstance(warned, set):
        warned.clear()


def _messages(caplog, needle: str) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if needle in record.getMessage()
    ]


def _load(native: dict, portable: dict, caplog):
    manager = SimpleNamespace(
        get_portable_mcp_servers=lambda: {
            name: dict(cfg) for name, cfg in portable.items()
        }
    )
    with (
        patch("hermes_cli.config.load_config", return_value={"mcp_servers": native}),
        patch("hermes_cli.plugins.discover_plugins"),
        patch("hermes_cli.plugins.get_plugin_manager", return_value=manager),
        caplog.at_level(logging.WARNING, logger="tools.mcp_tool"),
    ):
        return mcp_config._load_mcp_config()


def test_native_override_warns_once_per_name_and_keeps_the_native_entry(caplog):
    """Two loads of the same clash log the skip warning once; native config wins."""
    native = {
        "rea": {
            "command": "/usr/local/bin/rea",
            "args": ["mcp"],
            "env": {"REA_HOME": "/opt/rea"},
        }
    }
    portable = {"rea": {"command": "npx", "args": ["-y", "rea-agents", "mcp"]}}

    first = _load(native, portable, caplog)
    second = _load(native, portable, caplog)

    assert first["rea"] == native["rea"]
    assert second["rea"] == native["rea"]
    assert _messages(caplog, "conflicts with native config; skipping") == [
        "Portable MCP server 'rea' conflicts with native config; skipping",
    ]

    other_native = {"other": {"command": "native-other", "args": ["serve"]}}
    other_portable = {"other": {"command": "portable-other", "args": ["serve"]}}
    third = _load(other_native, other_portable, caplog)
    fourth = _load(other_native, other_portable, caplog)

    assert third["other"] == other_native["other"]
    assert fourth["other"] == other_native["other"]
    assert _messages(caplog, "conflicts with native config; skipping") == [
        "Portable MCP server 'rea' conflicts with native config; skipping",
        "Portable MCP server 'other' conflicts with native config; skipping",
    ]


def test_rejected_native_entry_skips_the_portable_server_once(caplog):
    """A malformed native budget stays off, and the skip warning is once per name."""
    native = {"srv": {"command": "x", "spend_limit": -5}}
    portable = {"srv": {"command": "python3", "args": ["s.py"]}}

    assert "srv" not in _load(native, portable, caplog)
    assert "srv" not in _load(native, portable, caplog)
    assert _messages(
        caplog, "skipped: the native entry of that name was rejected"
    ) == [
        "Portable MCP server 'srv' skipped: the native entry of that name was rejected",
    ]
