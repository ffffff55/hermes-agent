"""CORS preflight must list methods the API server actually serves.

Regression for #150279: ``PATCH /api/sessions/{id}`` is a documented session
rename, but the preflight ``Access-Control-Allow-Methods`` omitted ``PATCH``,
so browsers blocked the request before it was sent.
"""

import pytest
from aiohttp.test_utils import TestClient, TestServer

from tests.gateway.test_api_server import _create_app, _make_adapter


def _allow_methods(headers) -> set[str]:
    return {part.strip() for part in headers.get("Access-Control-Allow-Methods", "").split(",") if part.strip()}


@pytest.mark.asyncio
async def test_session_patch_preflight_lists_patch():
    """OPTIONS for PATCH /api/sessions/{id} advertises PATCH to the browser."""
    adapter = _make_adapter(cors_origins=["http://localhost:3000"])
    app = _create_app(adapter)
    app.router.add_route("PATCH", "/api/sessions/{session_id}", adapter._handle_patch_session)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.options(
            "/api/sessions/sess_1",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "PATCH",
                "Access-Control-Request-Headers": "content-type,authorization",
            },
        )
    assert resp.status == 200
    assert resp.headers.get("Access-Control-Allow-Origin") == "http://localhost:3000"
    assert "PATCH" in _allow_methods(resp.headers)
