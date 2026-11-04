"""Regression for #148477.

When the pinned vision provider cannot build a client, auto rescue must use the
live main runtime's model. Carrying the unavailable provider's ``gpt-*`` model
onto Anthropic keeps ``temperature`` and turns ``reasoning_effort`` into budget
thinking, which the API rejects with 400.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

_VISION_GPT = "gpt-6.1-sol"
_RUNTIME_CLAUDE = "claude-opus-5-5"


def _write_config(text: str) -> None:
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(text, encoding="utf-8")


def _anthropic_message(api_kwargs: dict):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text="ok")],
        stop_reason="end_turn",
        usage=None,
        model=api_kwargs.get("model"),
    )


@pytest.fixture
def _aux_cache():
    from agent.auxiliary_client import _client_cache, _client_cache_lock

    with _client_cache_lock:
        _client_cache.clear()
    yield
    with _client_cache_lock:
        _client_cache.clear()


def _call_vision(monkeypatch, captured: list, *, temperature: float = 0.1):
    from agent.auxiliary_client import call_llm, reset_runtime_main, set_runtime_main

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    def _capture(_client, api_kwargs, **_kwargs):
        captured.append(dict(api_kwargs))
        return _anthropic_message(api_kwargs)

    monkeypatch.setattr("agent.anthropic_adapter.create_anthropic_message", _capture)
    token = set_runtime_main(
        "anthropic", _RUNTIME_CLAUDE, api_mode="anthropic_messages", api_key="sk-ant-test",
    )
    try:
        return call_llm(
            task="vision",
            messages=[{"role": "user", "content": "describe this image"}],
            temperature=temperature,
        )
    finally:
        reset_runtime_main(token)


def _assert_no_temperature_budget_mismatch(api_kwargs: dict) -> None:
    thinking = api_kwargs.get("thinking") or {}
    budget = isinstance(thinking, dict) and "budget_tokens" in thinking
    temperature = api_kwargs.get("temperature")
    assert not (budget and temperature is not None and temperature != 1), api_kwargs


class TestVisionAutoRescueDropsPinnedModel:
    def test_unavailable_pinned_provider_uses_anthropic_runtime_model(
        self, monkeypatch, _aux_cache, caplog,
    ):
        """Codex vision is pinned and has no credentials; the live runtime is Anthropic.

        Auto rescue must send the runtime Claude model, not the pinned gpt-* model,
        and must not pair a non-1 temperature with budget thinking.
        """
        _write_config(f"""
model:
  provider: openai-codex
  default: {_VISION_GPT}
  supports_vision: true
auxiliary:
  vision:
    provider: openai-codex
    model: {_VISION_GPT}
    reasoning_effort: medium
""")
        captured: list = []
        with caplog.at_level("WARNING", logger="agent.auxiliary_client"):
            _call_vision(monkeypatch, captured)

        assert "falling back to auto vision backends" in caplog.text
        assert captured, "auto rescue did not reach the Anthropic Messages adapter"
        api_kwargs = captured[0]
        assert api_kwargs["model"] == _RUNTIME_CLAUDE
        assert not str(api_kwargs["model"]).lower().startswith("gpt")
        thinking = api_kwargs.get("thinking") or {}
        assert thinking.get("type") == "adaptive"
        assert "budget_tokens" not in thinking
        assert api_kwargs.get("output_config", {}).get("effort") == "medium"
        assert "temperature" not in api_kwargs
        _assert_no_temperature_budget_mismatch(api_kwargs)

    def test_auto_vision_keeps_its_own_configured_model(self, monkeypatch, _aux_cache):
        """``provider: auto`` plus an explicit model is the user's choice, not a foreign pin."""
        configured = "claude-sonnet-4-6"
        _write_config(f"""
model:
  provider: anthropic
  default: {_RUNTIME_CLAUDE}
  supports_vision: true
auxiliary:
  vision:
    provider: auto
    model: {configured}
    reasoning_effort: medium
""")
        captured: list = []
        _call_vision(monkeypatch, captured)
        assert captured, "vision auto route did not reach the Anthropic Messages adapter"
        assert captured[0]["model"] == configured
