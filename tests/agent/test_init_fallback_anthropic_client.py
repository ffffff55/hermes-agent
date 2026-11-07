"""Regression for #150445: init-time fallback onto anthropic_messages must
build the Anthropic client before the primary-runtime snapshot.

A codex_responses primary selects ``_init_openai_client``. When that primary
has no credentials, the fallback chain can flip ``api_mode`` to
``anthropic_messages`` without ever calling ``_init_anthropic_client``. The
snapshot then read ``_anthropic_api_key`` and ``AIAgent`` init died with
AttributeError, which a managed worker surfaced only as a bare failure.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.primary_runtime_snapshot import _snapshot_primary_runtime
from run_agent import AIAgent


def _make_tool_defs():
    return [{"type": "function", "function": {"name": "web_search",
             "description": "search", "parameters": {"type": "object", "properties": {}}}}]


def _mock_client(api_key="sk-ant-api03-testkey123456", base_url="https://api.anthropic.com"):
    client = MagicMock()
    client.api_key = api_key
    client.base_url = base_url
    client._default_headers = None
    return client


def test_codex_init_fallback_to_anthropic_builds_client_and_snapshot():
    """openai-codex with no primary credentials falls back to anthropic.

    Before the fix, init raised AttributeError: 'AIAgent' object has no
    attribute '_anthropic_api_key'. After it, the Anthropic client exists
    and the snapshot carries the fallback key.
    """
    fb = _mock_client()

    def fake_resolve(provider, model=None, raw_codex=False,
                     explicit_base_url=None, explicit_api_key=None):
        if provider == "anthropic":
            return fb, "claude-sonnet-4-5"
        return None, None

    with patch("agent.auxiliary_client.resolve_provider_client", side_effect=fake_resolve), \
         patch("model_tools.get_tool_definitions", return_value=_make_tool_defs()), \
         patch("model_tools.check_toolset_requirements", return_value={}), \
         patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()):

        agent = AIAgent(
            provider="openai-codex",
            model="gpt-5.4",
            api_key=None,
            base_url=None,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=[{"provider": "anthropic", "model": "claude-sonnet-4-5"}],
        )

    assert agent.provider == "anthropic"
    assert agent.model == "claude-sonnet-4-5"
    assert agent.api_mode == "anthropic_messages"
    assert agent._fallback_activated is True
    assert agent._anthropic_client is not None
    assert agent._anthropic_api_key == fb.api_key
    assert agent._anthropic_base_url == "https://api.anthropic.com"
    assert agent._is_anthropic_oauth is False
    snapshot = agent._primary_runtime
    assert snapshot["api_mode"] == "anthropic_messages"
    assert snapshot["anthropic_api_key"] == fb.api_key
    assert snapshot["anthropic_base_url"] == "https://api.anthropic.com"
    assert snapshot["is_anthropic_oauth"] is False


def test_snapshot_anthropic_fields_default_when_unset():
    """The snapshot reads the Anthropic fields with defaults, so a mode flip
    that never built the client does not raise AttributeError."""
    agent = SimpleNamespace(
        model="claude-sonnet-4-5",
        provider="anthropic",
        requested_provider="anthropic",
        base_url="https://api.anthropic.com",
        api_mode="anthropic_messages",
        api_key="sk-ant-api03-testkey123456",
        request_overrides={},
        _client_kwargs={},
        _use_prompt_caching=False,
        _use_native_cache_layout=False,
        context_compressor=SimpleNamespace(
            model="claude-sonnet-4-5",
            base_url="https://api.anthropic.com",
            api_key="sk-ant-api03-testkey123456",
            provider="anthropic",
            context_length=200_000,
            threshold_tokens=100_000,
        ),
    )

    _snapshot_primary_runtime(agent)

    assert agent._primary_runtime["anthropic_api_key"] == ""
    assert agent._primary_runtime["anthropic_base_url"] is None
    assert agent._primary_runtime["is_anthropic_oauth"] is False
