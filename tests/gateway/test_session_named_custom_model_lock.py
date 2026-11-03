"""A confirmed model lock on a named custom provider survives the session chat path.

Regression for #148601 (#116070 / #116752). ``POST /api/sessions/{id}/chat/stream``
runs through the gateway turn runner, which resolves ``providers.<name>`` to the
runtime id ``custom``. The lock must compare that resolved identity. The direct
``_create_agent`` path already stamps it; this test does not mock the turn runner
or ``_run_agent``.
"""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from hermes_constants import get_hermes_home


def _events(body):
    frames = []
    for block in body.split("\n\n"):
        lines = block.splitlines()
        name = next((line[7:] for line in lines if line.startswith("event: ")), None)
        data = next((line[6:] for line in lines if line.startswith("data: ")), None)
        if name and data:
            frames.append((name, json.loads(data)))
    return frames


def _turn_runner():
    """Enough of a gateway runner for ``TurnRunner.run_sync`` to build the agent for real.

    Session runtime resolution stays a stub (the profile default). ``prepare_api_runtime``
    then applies the admitted lock through the real provider catalog.
    """
    from gateway.run_turn_prepare import GatewayTurnPrepareMixin

    runner = MagicMock()
    runner.session_authority = None
    runner.session_authorities = None
    runner.config = SimpleNamespace(streaming=None)
    runner._provider_routing = {}
    runner._agent_cache_lock = None
    runner._agent_cache = None
    runner._session_db = None
    runner._prefill_messages = None
    runner._service_tier = None
    runner._ephemeral_system_prompt = ""
    runner.session_store = SimpleNamespace(_entries={})
    runner._delivery_adapter_for.return_value = None
    runner._resolve_session_agent_runtime.return_value = (
        "model-a",
        {"provider": "openrouter", "api_key": "sk-global", "base_url": "https://openrouter.example/v1",
         "api_mode": "chat_completions"},
    )
    runner._resolve_session_reasoning_config.return_value = None
    runner._resolve_session_service_tier.return_value = None
    runner._agent_config_signature.return_value = ("sig",)
    runner._extract_cache_busting_config.return_value = {}
    runner._refresh_fallback_model.return_value = "fallback-model"
    runner._consume_pending_native_image_paths.return_value = []
    runner._consume_pending_turn_sidecar_notes.return_value = []
    runner._get_system_prompt_for_channel.return_value = ""
    runner._resolve_turn_agent_config = GatewayTurnPrepareMixin._resolve_turn_agent_config.__get__(runner)
    return runner


class _LockedTurnAgent:
    """The model call, and nothing else: the turn runner still constructs this agent."""

    def __init__(self, **kwargs):
        self.provider = kwargs.get("provider") or ""
        self.model = kwargs.get("model") or ""
        self.base_url = kwargs.get("base_url") or ""
        self.session_id = kwargs.get("session_id")
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_total_tokens = 0
        self.tools = []

    def run_conversation(self, *_args, **_kwargs):
        return {"final_response": "hello from model-b", "messages": [], "completed": True, "api_calls": 1}


def _named_gateway_config(monkeypatch):
    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        "model:\n"
        "  default: model-a\n"
        "  provider: openrouter\n"
        "providers:\n"
        "  my-gateway:\n"
        "    base_url: https://gateway.example/v1\n"
        "    key_env: MY_GATEWAY_KEY\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("MY_GATEWAY_KEY", "sk-gateway-test")


@pytest.mark.asyncio
async def test_session_chat_stream_confirms_named_custom_provider_lock(api, owner, monkeypatch):
    """``require_model_lock`` on ``providers.my-gateway`` completes as runtime provider ``custom``."""
    from gateway.run_turn_runner import TurnRunner
    from gateway.turn_context import TurnContext

    _named_gateway_config(monkeypatch)
    owner.db.create_session("s1", source="api_server")
    built = {}

    class Agent(_LockedTurnAgent):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            built.update(kwargs)

    async def handle(event):
        ctx = TurnContext(
            source=event.source, message=event.text if isinstance(event.text, str) else "hi",
            history=[], session_id=event.source.chat_id, session_key="api:" + event.source.chat_id,
            user_config={}, AIAgent=Agent, resolve_display_setting=lambda *_a, **_k: None,
            _run_still_current=lambda: True, _hooks_ref=SimpleNamespace(loaded_hooks=False),
            enabled_toolsets=[], disabled_toolsets=[],
        )
        result = TurnRunner(_turn_runner(), ctx).run_sync()
        return result.get("final_response") or ""

    owner.runner._handle_message = handle
    app = web.Application()
    app.router.add_post("/api/sessions/{session_id}/chat/stream", api._handle_session_chat_stream)
    async with TestClient(TestServer(app)) as client:
        response = await client.post(
            "/api/sessions/s1/chat/stream",
            json={"message": "hi", "model": "model-b", "provider": "my-gateway", "require_model_lock": True},
        )
        body = await response.text()

    assert response.status == 200, body
    frames = _events(body)
    names = [name for name, _payload in frames]
    assert "run.completed" in names, body
    assert "run.failed" not in names, body
    completed = next(payload for name, payload in frames if name == "run.completed")
    runtime = completed["runtime"]
    assert runtime["provider"] == "custom"
    assert runtime["model"] == "model-b"
    assert runtime["route_source"] == "raw_request"
    assert runtime["model_lock"] == "confirmed"
    assert runtime["requested"]["provider"] == "my-gateway"
    assert runtime["requested"]["model"] == "model-b"
    assert built["provider"] == "custom"
    assert built["model"] == "model-b"
    assert built["base_url"] == "https://gateway.example/v1"
    assert "confirmed model lock runtime mismatch" not in body
