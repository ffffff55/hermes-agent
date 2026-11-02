"""``function_call_arguments.done`` must not clobber complete argument deltas.

Some Responses backends put a truncated ``arguments`` string on
``function_call_arguments.done`` (``json.loads`` raises ``Expecting ',' delimiter``)
even though the ``function_call_arguments.delta`` chunks already concatenate to
complete JSON. Replacing the accumulation unconditionally makes the settled
tool call unparseable (#148158).

``.done.arguments`` still wins when it is itself complete JSON, and an explicit
``""`` stays the zero-arg call. When neither side parses, the done value is
kept unchanged — it is not repaired.
"""

import json
from types import SimpleNamespace

import pytest

from agent.codex_runtime import _consume_codex_event_stream


def _stream(deltas, done_arguments=None, *, include_done=True, omit_arguments=False):
    events = [
        SimpleNamespace(type="response.created", response=SimpleNamespace(id="resp_1")),
        SimpleNamespace(
            type="response.output_item.added",
            output_index=0,
            item=SimpleNamespace(
                type="function_call",
                id="fc_1",
                call_id="call_1",
                name="read_file",
                arguments="",
            ),
        ),
    ]
    for delta in deltas:
        events.append(
            SimpleNamespace(
                type="response.function_call_arguments.delta",
                item_id="fc_1",
                output_index=0,
                delta=delta,
            )
        )
    if include_done:
        done = SimpleNamespace(
            type="response.function_call_arguments.done",
            item_id="fc_1",
            output_index=0,
        )
        if not omit_arguments:
            done.arguments = done_arguments
        events.append(done)
    events.append(
        SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(id="resp_1", status="completed", output=None),
        )
    )
    return events


def _settled_arguments(events):
    final = _consume_codex_event_stream(events, model="gpt-test")
    calls = [item for item in final.output if getattr(item, "type", "") == "function_call"]
    assert len(calls) == 1
    return calls[0].arguments


def test_valid_done_arguments_replace_accumulated_deltas():
    """A complete ``.done.arguments`` object is authoritative, even when the
    deltas already parsed as a different object."""
    args = _settled_arguments(
        _stream(['{"path": "', 'a.txt"}'], '{"path": "from-done.txt"}')
    )
    assert args == '{"path": "from-done.txt"}'
    assert json.loads(args) == {"path": "from-done.txt"}


def test_valid_empty_object_done_arguments_replace_accumulated_deltas():
    """``{}`` is complete JSON, so it wins over a longer accumulated object."""
    args = _settled_arguments(_stream(['{"path": "a.txt"}'], "{}"))
    assert args == "{}"
    assert json.loads(args) == {}


def test_truncated_done_arguments_keep_valid_accumulated_deltas():
    """The done payload ``{"path": "a.txt"`` is the truncation that raises
    ``Expecting ',' delimiter``. The deltas already form the full object."""
    done = '{"path": "a.txt"'
    with pytest.raises(json.JSONDecodeError, match="Expecting ',' delimiter"):
        json.loads(done)

    args = _settled_arguments(_stream(['{"path": "', 'a.txt"}'], done))
    assert args == '{"path": "a.txt"}'
    assert json.loads(args) == {"path": "a.txt"}


def test_invalid_done_and_invalid_deltas_keep_the_done_value():
    """Neither side parses: keep the done string so the existing error path
    still sees it. Do not repair it into JSON."""
    done = '{"path": "a.txt"'
    args = _settled_arguments(_stream(['{"path": "a.tx'], done))
    assert args == done
    with pytest.raises(json.JSONDecodeError):
        json.loads(args)


def test_explicit_empty_done_arguments_stay_zero_arg():
    """An explicit ``""`` is the zero-arg call, not a truncated payload, so it
    still replaces accumulated deltas and settles as ``{}``."""
    args = _settled_arguments(_stream(['{"path": "a.txt"}'], ""))
    assert args == "{}"


def test_missing_done_arguments_field_keeps_accumulated_deltas():
    """Only a present field is authoritative. Omitting ``arguments`` leaves
    the streamed deltas in place."""
    args = _settled_arguments(
        _stream(['{"path": "a.txt"}'], include_done=True, omit_arguments=True)
    )
    assert args == '{"path": "a.txt"}'
    assert json.loads(args) == {"path": "a.txt"}
