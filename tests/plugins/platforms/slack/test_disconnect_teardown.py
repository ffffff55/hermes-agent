"""Slack disconnect releases the platform lock when teardown fails.

Regression for #147845: ``client.close()`` raising, or cancellation while
``chat.stopStream`` is in flight, skipped ``_release_platform_lock()`` and
left the Slack app token locked.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.slack.adapter import SlackAdapter


def _adapter() -> SlackAdapter:
    adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-fake"))
    adapter._platform_lock_scope = "slack-app-token"
    adapter._platform_lock_identity = "xapp-fake"
    adapter._running = True
    return adapter


@pytest.mark.asyncio
async def test_disconnect_releases_lock_when_client_close_raises():
    """A raising workspace client must not skip other clients or the lock."""
    adapter = _adapter()
    primary = MagicMock()
    primary.close.side_effect = RuntimeError("workspace client close failed")
    secondary = MagicMock()
    secondary.close = AsyncMock()
    adapter._app = SimpleNamespace(client=primary)
    adapter._team_clients = {"T2": secondary}
    adapter._team_bot_user_ids = {"T2": "U_BOT"}
    adapter._channel_team = {"C1": "T2"}
    adapter._app_token = "xapp-fake"
    adapter._bot_user_id = "U_BOT"

    with patch("gateway.status_scoped_locks.release_scoped_lock") as release:
        await adapter.disconnect()

    secondary.close.assert_awaited_once()
    release.assert_called_once_with("slack-app-token", "xapp-fake")
    assert adapter._platform_lock_identity is None
    assert adapter._app is None
    assert adapter._team_clients == {}
    assert adapter._team_bot_user_ids == {}
    assert adapter._channel_team == {}
    assert adapter._bot_user_id is None
    assert adapter._app_token is None


@pytest.mark.asyncio
async def test_disconnect_releases_lock_when_seal_is_cancelled():
    """Cancellation inside chat.stopStream still finishes cleanup and the lock."""
    adapter = _adapter()
    started = asyncio.Event()
    closed: list[bool] = []

    class _Client:
        async def chat_stopStream(self, **kwargs):
            started.set()
            await asyncio.Event().wait()

        async def close(self):
            closed.append(True)

    client = _Client()
    adapter._app = SimpleNamespace(client=client)
    adapter._get_client = lambda *_args, **_kwargs: client
    adapter._active_streams = {
        ("T1", "C1", "111.000"): {"ts": "111.000", "sent": "partial"},
    }

    with patch("gateway.status_scoped_locks.release_scoped_lock") as release:
        task = asyncio.create_task(adapter.disconnect())
        await asyncio.wait_for(started.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert closed == [True]
    release.assert_called_once_with("slack-app-token", "xapp-fake")
    assert adapter._platform_lock_identity is None
    assert adapter._active_streams == {}
    assert adapter._app is None
