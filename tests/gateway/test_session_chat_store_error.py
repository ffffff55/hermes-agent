"""POST /api/sessions/{id}/chat maps a chat-platform store refusal to 4xx.

Regression for #150153: ``admit_api_steps`` raises ``RuntimeStoreError``
(``permission_denied``) when the session source is a chat platform. ``/v1/runs``
already answers that with ``store_error_response``; this door was letting it
escape as HTTP 500. The foreign-session attach request in that issue is not
part of this contract — the session stays on its platform and nothing is admitted.
"""

import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from hermes_state_runtime import list_session_admissions


@pytest.mark.asyncio
async def test_session_chat_refuses_chat_platform_session_as_client_error(api, owner):
    owner.db.create_session("tg-chat", source="telegram")
    app = web.Application()
    app.router.add_post("/api/sessions/{session_id}/chat", api._handle_session_chat)
    async with TestClient(TestServer(app)) as client:
        response = await client.post("/api/sessions/tg-chat/chat", json={"message": "hello"})
        raw = await response.text()
    # A 500 from the uncaught RuntimeStoreError is text/plain, so status is
    # checked before JSON parsing.
    assert response.status == 409, (response.status, raw)
    body = json.loads(raw)
    assert body["error"]["code"] == "permission_denied"
    assert body["error"]["message"] == "permission_denied"
    assert owner.db.get_session("tg-chat")["source"] == "telegram"
    assert list_session_admissions(owner.db, session_id="tg-chat", pending_only=False) == []
