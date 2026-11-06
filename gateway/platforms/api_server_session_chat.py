"""The synchronous session-chat turn (``POST /api/sessions/{id}/chat``)."""

from __future__ import annotations

from typing import Any


async def run_session_chat_turn(adapter, ctx: dict[str, Any]):
    """Run one session-chat turn, including its one policy-gated re-run.

    Returns ``(result, usage)``, or the HTTP response for a store refusal.
    ``admit_api_steps`` raises ``RuntimeStoreError`` when the session source is a
    chat platform (anything other than ``api_server`` / ``bot_room``). ``/v1/runs``
    maps that through ``store_error_response``; leaving it uncaught on this door
    is HTTP 500 (#150153).
    """
    from gateway.platforms.api_server import _openai_error
    from gateway.platforms.api_server_authority_runs import store_error_response
    from hermes_state_runtime import RuntimeStoreError
    from tools.bot_failure_reasons import RETRY_NONE, result_retry_action

    session_id = ctx["session_id"]
    history = await adapter._conversation_history_for_session(session_id)
    try:
        result, usage = await adapter._run_agent(conversation_history=history, **ctx["run_kwargs"])
        # One policy-gated re-run of a transiently failed turn — the peer-DM transport's half of the
        # retry the local (``tools.bot_mode_dm``) and relayed (``tui_gateway.methods_bot_relay``)
        # delivery lanes already apply (#93091 item 5, #115325). Same policy, same gate: transient
        # classes (429 / 5xx) re-run the SAME session once, a context overflow lets the re-run's
        # pre-API compaction shrink the transcript first, and auth/quota/config/model never re-run. The
        # store is read again first: the failed attempt's turn-start persist left the DM as the
        # transcript's unanswered tail row, and the re-run resumes that row instead of appending a
        # second copy of it. A turn that fails again reaches the peer client exactly as before.
        if result_retry_action(result) != RETRY_NONE:
            history = await adapter._conversation_history_for_session(session_id)
            result, usage = await adapter._run_agent(
                conversation_history=history, resume_unanswered_turn=True, **ctx["run_kwargs"])
    except RuntimeStoreError as exc:
        return store_error_response(_openai_error, exc)
    return result, usage
