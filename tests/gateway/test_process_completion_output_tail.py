"""Completion notices keep a nonempty tail when a long single line ends in a newline.

Regression for #151053. ``_build_process_completion_event`` keeps the last
``COMPLETION_OUTPUT_CHARS`` (2000 by default) and snaps to a line boundary.
When that window's only newline is its final character, slicing past it used
to yield an empty string, so the notice said "showing last 0 chars" while the
process output itself was intact.
"""

import re

import pytest

from gateway.run_notifications import GatewayNotificationsMixin
from tools.process_registry import COMPLETION_OUTPUT_CHARS, ProcessSession

_LIMIT = COMPLETION_OUTPUT_CHARS
_WATCHER = {"platform": "telegram", "chat_id": "1", "session_key": "agent:main:test"}
_MARKER = re.compile(r"\[… output truncated — showing last (\d+) chars\]")


def _event(output: str, **session_fields) -> tuple[dict, ProcessSession]:
    session = ProcessSession(
        id="proc_tail",
        command="echo payload",
        output_buffer=output,
        exit_code=0,
        exited=True,
        **session_fields,
    )
    event = GatewayNotificationsMixin._build_process_completion_event(
        _WATCHER, session, session.id,
    )
    return event, session


def _truncated_tail(output: str) -> str:
    marker, sep, tail = output.partition("\n")
    match = _MARKER.fullmatch(marker)
    assert sep == "\n" and match, output[:120]
    shown = int(match.group(1))
    assert shown == len(tail) and shown > 0
    return tail


def test_long_single_line_ending_in_newline_keeps_a_nonempty_tail():
    """The reported shape: one JSON line over the cap, terminated by a newline."""
    prefix, suffix = '{"review":"', '"}\n'
    raw = prefix + ("x" * (2408 - len(prefix) - len(suffix))) + suffix
    assert len(raw) == 2408 and raw.endswith("\n") and "\n" not in raw[:-1]

    event, session = _event(raw)
    tail = _truncated_tail(event["output"])
    assert tail == raw[-_LIMIT:]
    assert tail.endswith("\n")
    assert session.output_buffer == raw


def test_long_single_line_without_newline_keeps_the_capped_tail():
    raw = "y" * (_LIMIT + 25)
    event, session = _event(raw)
    assert _truncated_tail(event["output"]) == raw[-_LIMIT:]
    assert session.output_buffer == raw


def test_multiline_truncation_snaps_to_a_line_boundary():
    dropped = "DROP_THIS_PREFIX\n"
    partial = "m" * (_LIMIT + 50)
    kept = "KEEP_WHOLE\nstill here\n"
    raw = dropped + partial + "\n" + kept

    event, _session = _event(raw)
    tail = _truncated_tail(event["output"])
    assert tail == kept
    assert "DROP_THIS_PREFIX" not in event["output"]
    assert "m" not in tail
    assert len(tail) < _LIMIT


def test_short_output_is_unchanged():
    raw = "hello\nworld\n"
    event, session = _event(raw)
    assert event["output"] == raw
    assert "truncated" not in event["output"]
    assert session.output_buffer == raw


@pytest.mark.parametrize("length", [_LIMIT - 1, _LIMIT, _LIMIT + 1, _LIMIT + 2])
def test_boundary_lengths_around_the_cap(length):
    """Just under, at, and just over the default cap (1999–2002 when the cap is 2000)."""
    raw = "B" * (length - 1) + "\n"
    assert len(raw) == length
    event, session = _event(raw)
    assert session.output_buffer == raw
    if length <= _LIMIT:
        assert event["output"] == raw
        return
    tail = _truncated_tail(event["output"])
    assert tail == raw[-_LIMIT:]
    assert len(tail) == _LIMIT
    assert tail.endswith("\n")
    assert set(tail[:-1]) == {"B"}


def test_requested_completion_capacity_sizes_the_tail():
    """A process that asked for more than the default cap keeps that much of a trailing-newline line."""
    raw = "C" * 4000 + "\n"
    event, session = _event(raw, completion_output_chars=2500)
    tail = _truncated_tail(event["output"])
    assert tail == raw[-2500:]
    assert len(tail) == 2500
    assert session.output_buffer == raw

    whole = "ok\n"
    event, _session = _event(whole, completion_output_chars=2500)
    assert event["output"] == whole
