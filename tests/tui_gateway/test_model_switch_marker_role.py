"""Tests for _append_model_switch_marker role fix (issue #48338).

The model switch marker must NOT use role="system" because strict providers
(vLLM, Qwen) reject system messages that appear mid-conversation. Using
role="user" is safe — the system prompt is prepended to the API message list,
so a user-role marker can appear at any later position, and the gateway's
sanitize/merge pass already coalesces consecutive user messages.
"""

from __future__ import annotations


from tui_gateway.server import _append_model_switch_marker


class TestAppendModelSwitchMarkerRole:
    """Verify the marker uses role='user', not role='system'."""

    def test_marker_uses_user_role(self) -> None:
        """The history entry must be role='user', not role='system'."""
        session: dict = {"session_key": "test-session", "history": []}
        _append_model_switch_marker(session, model="gpt-4o", provider="openai")
        assert len(session["history"]) == 1
        entry = session["history"][0]
        assert entry["role"] == "user", (
            f"Expected role='user' but got role='{entry['role']}'. "
            "Strict providers (vLLM, Qwen) reject mid-conversation system messages."
        )




class TestModelSwitchMarkerDedup:
    """#65891: only the newest marker is meaningful; older ones must not
    accumulate in the live history and burn context tokens every turn."""

    @staticmethod
    def _markers(session: dict) -> list:
        from tui_gateway.server import _is_model_switch_marker

        return [h for h in session["history"] if _is_model_switch_marker(h)]

    def test_second_switch_replaces_first_marker(self) -> None:
        session: dict = {"session_key": "s", "history": []}
        _append_model_switch_marker(session, model="model-a", provider="p")
        _append_model_switch_marker(session, model="model-b", provider="p")
        markers = self._markers(session)
        assert len(markers) == 1, "a second switch must replace, not stack, the marker"
        assert "model-b" in markers[0]["content"]
        assert "model-a" not in markers[0]["content"]
        # The surviving marker is the last history entry.
        assert session["history"][-1] is markers[0]


    def test_dedup_preserves_real_conversation_turns(self) -> None:
        session: dict = {
            "session_key": "s",
            "history": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ],
        }
        _append_model_switch_marker(session, model="model-a", provider="p")
        _append_model_switch_marker(session, model="model-b", provider="p")
        # Real turns untouched; exactly one marker, appended at the end.
        assert session["history"][0] == {"role": "user", "content": "hello"}
        assert session["history"][1] == {"role": "assistant", "content": "hi"}
        assert len(self._markers(session)) == 1
        assert len(session["history"]) == 3

    def test_prior_marker_between_turns_is_removed(self) -> None:
        # A stale marker not at the tail (a later turn followed it) is still
        # stripped on the next switch.
        session: dict = {
            "session_key": "s",
            "history": [
                {"role": "user", "content": "q1"},
                _make_marker_entry("model-a"),
                {"role": "assistant", "content": "a1"},
            ],
        }
        _append_model_switch_marker(session, model="model-b", provider="p")
        markers = self._markers(session)
        assert len(markers) == 1
        assert "model-b" in markers[0]["content"]
        # The real turns are preserved in order.
        assert [h["content"] for h in session["history"] if not _is_marker(h)] == ["q1", "a1"]

    def test_history_version_increments_once_on_replace(self) -> None:
        session: dict = {"session_key": "s", "history": [], "history_version": 0}
        _append_model_switch_marker(session, model="model-a", provider="p")
        _append_model_switch_marker(session, model="model-b", provider="p")
        assert session["history_version"] == 2  # one increment per switch


def _make_marker_entry(model: str) -> dict:
    from tui_gateway.server import _MODEL_SWITCH_MARKER_PREFIX

    return {"role": "user", "content": f"{_MODEL_SWITCH_MARKER_PREFIX}{model}.]"}


def _is_marker(entry: dict) -> bool:
    from tui_gateway.server import _is_model_switch_marker

    return _is_model_switch_marker(entry)


class TestAgentlessModelSwitchPersists:
    """#149870: a composer pick is ``model --provider X``. That shape does not
    pre-build an agent, so the durable commit used to be skipped entirely.
    Resume, the desktop timeline, and the model's ``[System: ...]`` injection
    all read the row this switch has to write."""

    @staticmethod
    def _switch(monkeypatch, db, session: dict, raw: str, **kwargs):
        from types import SimpleNamespace

        from tui_gateway import server

        result = SimpleNamespace(
            success=True, new_model="gemini-3.8-flash-high", target_provider="easycliproxyapi_ws",
            base_url="http://127.0.0.1:8318/v1", api_key="sk-test", api_mode="chat_completions",
            warning_message="", error_message="", model_info=None, runtime_capabilities=None)
        monkeypatch.setattr("hermes_cli.model_switch.switch_model", lambda **_kw: result)
        monkeypatch.setattr(server, "_get_db", lambda: db)
        monkeypatch.setattr(server, "_config_model_target", lambda: ("deepseek/deepseek-v4.1-flash", "cmdcode2api"))
        # The picker's provider is a configured custom entry on a real profile. This home has
        # no providers.yaml; treat the stored name as routable so resume assembly is what we assert.
        monkeypatch.setattr(server, "_is_routable_provider", lambda _provider: True)
        return server._apply_model_switch("sid", session, raw, confirm_expensive_model=True, **kwargs)

    def test_agentless_provider_pick_writes_marker_and_route(self, tmp_path, monkeypatch) -> None:
        import json

        from hermes_state import SessionDB
        from tui_gateway import server

        key = "local-" + ("a" * 64)
        db = SessionDB(tmp_path / "state.db")
        try:
            db.create_session(
                key, source="local", model="deepseek/deepseek-v4.1-flash",
                model_config={
                    "model": "deepseek/deepseek-v4.1-flash",
                    "provider": "cmdcode2api",
                    "base_url": "http://127.0.0.1:11434/v1",
                    "api_mode": "chat_completions",
                    "follow_profile_config": True,
                    "gateway_runtime": {
                        "provider": "cmdcode2api",
                        "base_url": "http://127.0.0.1:11434/v1",
                        "api_mode": "chat_completions",
                    },
                })
            session = {
                "agent": None,
                "session_key": key,
                "follow_profile_config": True,
                "history": [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}],
            }

            out = self._switch(
                monkeypatch, db, session, "gemini-3.8-flash-high --provider easycliproxyapi_ws",
                persist_override=False)

            assert out["value"] == "gemini-3.8-flash-high"
            assert out["scope"] == "session"
            assert session.get("agent") is None
            assert session["model_override"]["model"] == "gemini-3.8-flash-high"
            assert session["model_override"]["provider"] == "easycliproxyapi_ws"
            marker = session["history"][-1]
            assert marker["role"] == "user"
            assert marker["display_kind"] == "model_switch"
            assert marker["content"].startswith("[System: The active model for this chat has changed to ")
            assert "gemini-3.8-flash-high" in marker["content"]
            assert "via provider easycliproxyapi_ws" in marker["content"]
            assert [h["content"] for h in session["history"][:-1]] == ["hello", "hi"]

            stored = db.get_session(key)
            assert stored["model"] == "gemini-3.8-flash-high"
            config = json.loads(stored["model_config"])
            assert config["provider"] == "easycliproxyapi_ws"
            assert config["base_url"] == "http://127.0.0.1:8318/v1"
            assert config["api_mode"] == "chat_completions"
            assert config["model"] == "gemini-3.8-flash-high"
            # The resume reader prefers nested gateway_runtime over the top-level keys.
            assert config["gateway_runtime"]["provider"] == "easycliproxyapi_ws"
            assert config["gateway_runtime"]["base_url"] == "http://127.0.0.1:8318/v1"
            assert "sk-test" not in stored["model_config"]
            assert config["composer_override_profile"] == {
                "model": "deepseek/deepseek-v4.1-flash", "provider": "cmdcode2api"}
            assert config["follow_profile_config"] is True
            restored = server._stored_session_runtime_overrides(stored)
            assert restored["model_override"]["model"] == "gemini-3.8-flash-high"
            assert restored["model_override"]["provider"] == "easycliproxyapi_ws"
            assert restored["model_override"]["base_url"] == "http://127.0.0.1:8318/v1"
            assert restored["provider_override"] == "easycliproxyapi_ws"

            rows = [m for m in db.get_messages(key) if m.get("display_kind") == "model_switch"]
            assert len(rows) == 1
            assert rows[0]["role"] == "user"
            assert rows[0]["content"] == marker["content"]
        finally:
            db.close()

    def test_agentless_once_does_not_persist(self, tmp_path, monkeypatch) -> None:
        import json

        import pytest

        from hermes_state import SessionDB

        key = "local-" + ("b" * 64)
        db = SessionDB(tmp_path / "state.db")
        try:
            db.create_session(key, source="local", model="deepseek/deepseek-v4.1-flash", model_config={
                "provider": "cmdcode2api",
                "gateway_runtime": {"provider": "cmdcode2api", "base_url": "http://127.0.0.1:11434/v1"},
            })
            session = {"agent": None, "session_key": key, "history": []}
            with pytest.raises(ValueError, match="--once"):
                self._switch(monkeypatch, db, session, "gemini-3.8-flash-high --provider easycliproxyapi_ws --once")
            stored = db.get_session(key)
            assert stored["model"] == "deepseek/deepseek-v4.1-flash"
            assert json.loads(stored["model_config"])["gateway_runtime"]["provider"] == "cmdcode2api"
            assert db.get_messages(key) == []
            assert session["history"] == []
            assert "model_override" not in session
        finally:
            db.close()
