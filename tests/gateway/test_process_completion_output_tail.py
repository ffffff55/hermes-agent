"""Process-completion notifications keep a usable output tail.

A payload longer than the notification window is snapped to a line boundary so
the notice never starts mid-line (#23284). The only newline in a single-line
payload is the trailing one; treating that as a snap point used to drop the
whole tail and report "showing last 0 chars" (#152075), which swallowed
``bot_mode_dm --run-delivery`` replies.
"""

from types import SimpleNamespace

from gateway.run_notifications import GatewayNotificationsMixin

# Same window ``_build_process_completion_event`` keeps before the marker.
_TAIL_LIMIT = 2000


def _completion_output(payload: str) -> str:
    session = SimpleNamespace(
        command="echo payload",
        output_buffer=payload,
        exit_code=0,
        task_id="",
        owner_task_id="",
        started_at=None,
        completion_reason="exited",
        termination_source="",
        parent_session_id="",
    )
    event = GatewayNotificationsMixin._build_process_completion_event(
        {}, session, "proc_tail",
    )
    return event["output"]


def _split_truncated(output: str) -> tuple[str, str]:
    marker, sep, tail = output.partition("\n")
    assert sep == "\n", output
    return marker, tail


def test_single_line_payload_keeps_the_tail_window():
    """A single line ending in a newline keeps the ~2000-char window, not 0."""
    suffix = "reply-kept-" + ("z" * 32)
    payload = ("x" * 4500) + suffix + "\n"

    marker, tail = _split_truncated(_completion_output(payload))

    assert "showing last 0 chars" not in marker
    assert len(tail) >= _TAIL_LIMIT - 50
    assert tail == payload[-len(tail):]
    assert tail.rstrip("\n").endswith(suffix)
    reported = marker.removeprefix("[… output truncated — showing last ").removesuffix(" chars]")
    assert reported == str(len(tail))


def test_multiline_payload_snaps_to_a_line_boundary():
    """A cut that lands mid-line still discards the partial line (#23284)."""
    partial = "P" * 2500
    kept = "kept-line-alpha\nkept-line-beta\n"
    payload = partial + "\n" + kept

    marker, tail = _split_truncated(_completion_output(payload))

    assert tail == kept
    assert "P" not in tail
    assert marker == f"[… output truncated — showing last {len(kept)} chars]"
    assert len(kept) < _TAIL_LIMIT


def test_short_output_is_not_truncated():
    """Output at or under the window is returned unchanged."""
    short = "short reply\n"
    assert _completion_output(short) == short

    exact = "e" * _TAIL_LIMIT
    assert _completion_output(exact) == exact
