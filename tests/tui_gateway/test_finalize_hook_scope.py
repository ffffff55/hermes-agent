"""
Regression test (fleet H-12): the on_session_finalize hook dispatch in
``_finalize_session`` must run inside the session's bound profile scope.

Out-of-turn finalize dispatch (reaper/Timer/shutdown paths) used to fire
``on_session_finalize`` OUTSIDE the ``_session_profile_runtime_scope`` block
that covers ``on_session_end``. Any plugin finalize hook that reads a
profile-scoped secret at call time (the camofox plugin reads CAMOFOX_USER_ID
to clean up its browser session) then fails closed under multiplex —
``UnscopedSecretError`` from ``agent.secret_scope.get_secret`` — and the
operator sees the "could not read this profile" banner. The warn-once dedup
is per PROCESS lifetime, so the failure recurred after every gateway or
dashboard restart, which is why a restart never cured it.

The test forces multiplex semantics (``is_multiplex_active`` -> True) and
stubs the last hop of the hook chain (``hermes_cli.plugins.invoke_hook``)
with a finalizer that reads a scoped secret; the whole chain above it
(scope binding -> ``_notify_session_boundary`` ->
``hermes_cli.lifecycle.finalize_session``) is real code.
"""

import threading

from unittest.mock import patch


def _make_session(profile_home, session_key="test-scope-key"):
    return {
        "agent": None,
        "history": [],
        "history_lock": threading.Lock(),
        "session_key": session_key,
        "_finalized": False,
        "profile_home": profile_home,
    }


class TestFinalizeHookScope:
    def test_finalize_hook_sees_session_profile_secret(self, tmp_path, monkeypatch):
        # A fake named profile whose .env carries a scoped secret.
        (tmp_path / ".env").write_text("CAMOFOX_USER_ID=finalize-scope-test\n")

        from agent import secret_scope

        # Multiplex on + no launch-body scope: the exact condition under which
        # the live defect raised UnscopedSecretError (out-of-turn, unbound).
        monkeypatch.setattr(secret_scope, "is_multiplex_active", lambda: True)

        seen = {}

        def _fake_plugin_invoke(hook_name, **kwargs):
            assert hook_name == "on_session_finalize"
            try:
                seen["value"] = secret_scope.get_secret("CAMOFOX_USER_ID")
                seen["error"] = None
            except Exception as exc:  # capture, don't mask
                seen["value"] = None
                seen["error"] = repr(exc)
            return []

        with patch("hermes_cli.plugins.invoke_hook", _fake_plugin_invoke):
            from tui_gateway.server import _finalize_session

            _finalize_session(_make_session(str(tmp_path)))

        assert "error" in seen, "on_session_finalize hook was never dispatched"
        assert seen["error"] is None, (
            f"on_session_finalize hook ran outside the session's profile scope: {seen['error']}"
        )
        assert seen["value"] == "finalize-scope-test"
