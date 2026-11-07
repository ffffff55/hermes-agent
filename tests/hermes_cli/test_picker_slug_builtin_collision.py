"""Regression for #150823.

A ``providers:`` entry whose key is a canonical built-in id (``ai-gateway``)
was listed with that bare slug. Selecting it stored ``provider: ai-gateway``,
and runtime resolution then picked the built-in — which demands
``AI_GATEWAY_API_KEY`` — instead of the user's endpoint. ``custom:ai-gateway``
already reaches the entry, and the payload already lists that spelling among
the row's aliases. The slug the picker persists has to be that identity.

A key no built-in claims keeps the bare spelling clients already store.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from hermes_cli.inventory import build_model_options_payload, load_picker_context


def _offline_picker(monkeypatch) -> None:
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda *a, **k: {})
    monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {})
    monkeypatch.setattr(
        "hermes_cli.model_switch_providers._discover_endpoint_models",
        lambda *a, **k: (None, False),
    )
    monkeypatch.setattr("hermes_cli.inventory._prewarm_pricing_async", lambda *a, **k: None)


def _payload(cfg: dict) -> dict:
    with patch("hermes_cli.config.load_config", return_value=cfg):
        ctx = load_picker_context()
    return build_model_options_payload(ctx)


def _named(payload: dict, name: str) -> list[dict]:
    return [row for row in payload["providers"] if row.get("name") == name]


_USER_PROVIDERS = {
    "ai-gateway": {
        "name": "My Gateway",
        "api": "https://gateway.example.com/v1",
        "transport": "codex_responses",
        "model": "gateway-model",
        "discover_models": False,
    },
    "local-lab": {
        "name": "Local Lab",
        "api": "http://127.0.0.1:9/v1",
        "model": "lab-model",
        "discover_models": False,
    },
}


@pytest.fixture
def picker_payload(monkeypatch):
    _offline_picker(monkeypatch)
    return _payload({"model": {}, "providers": _USER_PROVIDERS, "custom_providers": []})


def test_colliding_user_provider_slug_is_custom_prefixed(picker_payload):
    rows = _named(picker_payload, "My Gateway")
    assert len(rows) == 1
    assert rows[0]["slug"] == "custom:ai-gateway"
    assert rows[0]["is_user_defined"] is True


def test_non_colliding_user_provider_keeps_bare_slug(picker_payload):
    rows = _named(picker_payload, "Local Lab")
    assert len(rows) == 1
    assert rows[0]["slug"] == "local-lab"
    assert rows[0]["is_user_defined"] is True
