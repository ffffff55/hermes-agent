"""Partial-read write refusals name the condition that blocked a full view.

Regression for #150889: a stamp with ``partial=True`` used to always say the
read was offset/limit pagination, even when the read was partial because the
snapshot was unstable, content was redacted, lines were clamped, or the
character budget cut it. When a write is refused does not change.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from tools import file_state
from tools.file_tools import clear_file_ops_cache, read_file_tool, write_file_tool
from tools.file_tools_read_tracking import _file_metadata, _read_tracker

_PAGE_WORDS = ("pagination", "remaining pages")


@pytest.fixture
def task(request):
    task_id = f"partial-reason-{request.node.name}"
    file_state.get_registry().clear()
    _read_tracker.clear()
    yield task_id
    clear_file_ops_cache(task_id)
    file_state.get_registry().clear()
    _read_tracker.clear()


def _body(text: str, path) -> str:
    return text.replace(str(path), "")


def _stamp(
    path, *, partial: bool, reason: str | None = None, task_id: str = "stamp"
) -> str:
    file_state.get_registry().clear()
    file_state.record_read(task_id, path, partial=partial, partial_reason=reason)
    message = file_state.check_stale(task_id, path)
    assert message is not None
    return message


@pytest.mark.parametrize(
    ("reason", "needle"),
    [
        (file_state.PARTIAL_UNSTABLE, "unstable snapshot"),
        (file_state.PARTIAL_REDACTED, "redacted content"),
        (file_state.PARTIAL_TRUNCATED_LINES, "length limit"),
        (file_state.PARTIAL_CHAR_BUDGET, "character budget"),
        (file_state.PARTIAL_REPEAT, "repeatedly"),
    ],
)
def test_non_pagination_partial_stamp_names_the_condition(tmp_path, reason, needle):
    path = tmp_path / "f.txt"
    path.write_text("alpha\n")
    message = _stamp(path, partial=True, reason=reason)
    body = _body(message, path)
    assert needle in body
    for word in _PAGE_WORDS:
        assert word not in body
    assert "page" not in body


def test_pagination_partial_stamp_still_mentions_pages(tmp_path):
    path = tmp_path / "f.txt"
    path.write_text("alpha\nbeta\n")
    message = _stamp(path, partial=True, reason=file_state.PARTIAL_PAGINATION)
    body = _body(message, path)
    assert "offset/limit pagination" in body
    assert "remaining pages" in body


def test_bare_partial_stamp_keeps_the_pagination_wording(tmp_path):
    """A partial flag with no reason is the historical pagination stamp."""
    path = tmp_path / "f.txt"
    path.write_text("alpha\n")
    message = _stamp(path, partial=True)
    body = _body(message, path)
    assert "offset/limit pagination" in body
    assert "remaining pages" in body


def test_unknown_partial_reason_does_not_invent_pagination(tmp_path):
    path = tmp_path / "f.txt"
    path.write_text("alpha\n")
    message = _stamp(path, partial=True, reason="mystery")
    body = _body(message, path)
    assert "mystery" in body
    assert "page" not in body


def test_full_stamp_is_not_a_partial_refusal(tmp_path):
    path = tmp_path / "f.txt"
    path.write_text("alpha\n")
    file_state.get_registry().clear()
    file_state.record_read("stamp", path, partial=False, partial_reason="redacted")
    assert file_state.check_stale("stamp", path) is None


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            dict(stable=False, window_partial=False, coverage_complete=True),
            file_state.PARTIAL_UNSTABLE,
        ),
        (
            dict(window_partial=True, coverage_complete=False),
            file_state.PARTIAL_PAGINATION,
        ),
        (
            dict(window_partial=True, coverage_complete=False, char_budget=True),
            file_state.PARTIAL_CHAR_BUDGET,
        ),
        (
            dict(
                window_partial=True,
                coverage_complete=False,
                char_budget=True,
                truncated_lines=True,
            ),
            file_state.PARTIAL_TRUNCATED_LINES,
        ),
        (
            dict(window_partial=True, coverage_complete=False, redacted=True),
            file_state.PARTIAL_REDACTED,
        ),
        (dict(over_repeat=True, coverage_complete=False), file_state.PARTIAL_REPEAT),
    ],
)
def test_incomplete_read_reason_prefers_the_blocking_condition(kwargs, expected):
    base = dict(
        stable=True,
        over_repeat=False,
        window_partial=False,
        coverage_complete=False,
        redacted=False,
        truncated_lines=False,
        char_budget=False,
    )
    base.update(kwargs)
    assert file_state.incomplete_read_reason(**base) == expected


def _refused(path, task_id: str, content: str = "replacement\n") -> dict:
    before = path.read_text(encoding="utf-8")
    refused = json.loads(write_file_tool(str(path), content, task_id=task_id))
    assert refused.get("stale_write_blocked"), refused
    assert path.read_text(encoding="utf-8") == before
    return refused


def test_paging_to_eof_still_allows_the_overwrite(tmp_path, task):
    path = tmp_path / "paged.txt"
    path.write_text("one\ntwo\nthree\n", encoding="utf-8")
    first = json.loads(read_file_tool(str(path), offset=1, limit=2, task_id=task))
    assert first.get("truncated") is True
    refused = _refused(path, task)
    assert "remaining pages" in _body(refused["error"], path)
    assert "error" not in json.loads(
        read_file_tool(str(path), offset=3, limit=2, task_id=task)
    )
    written = json.loads(write_file_tool(str(path), "yes\n", task_id=task))
    assert "error" not in written, written
    assert path.read_text(encoding="utf-8") == "yes\n"


def test_redacted_read_refusal_does_not_mention_pages(tmp_path, task):
    path = tmp_path / "secret.txt"
    secret = "ghp_" + "A" * 40
    path.write_text(f"token={secret}\n", encoding="utf-8")
    with patch("agent.redact._REDACT_ENABLED", True):
        read = json.loads(read_file_tool(str(path), task_id=task))
    assert secret not in read["content"]
    assert read.get("truncated") is not True
    refused = _refused(path, task, content="token=cleared\n")
    body = _body(refused["error"], path)
    assert "redacted content" in body
    assert "page" not in body
    assert path.read_text(encoding="utf-8") == f"token={secret}\n"


def test_clamped_line_refusal_does_not_mention_pages(tmp_path, task):
    from tools.tool_output_limits import get_max_line_length

    path = tmp_path / "clamped.txt"
    original = "x" * (get_max_line_length() + 10) + "\nshort\n"
    path.write_text(original, encoding="utf-8")
    read = json.loads(read_file_tool(str(path), task_id=task))
    assert read.get("truncated_lines") is True
    assert read.get("truncated") is not True
    refused = _refused(path, task)
    body = _body(refused["error"], path)
    assert "length limit" in body
    assert "page" not in body
    assert path.read_text(encoding="utf-8") == original


def test_char_budget_refusal_does_not_mention_pages(tmp_path, task):
    path = tmp_path / "budget.txt"
    original = "one\ntwo\nthree\nfour\n"
    path.write_text(original, encoding="utf-8")
    with patch("tools.file_tools._get_max_read_chars", return_value=20):
        read = json.loads(read_file_tool(str(path), task_id=task))
    assert read.get("truncated_by") == "bytes"
    assert "truncated_lines" not in read
    refused = _refused(path, task)
    body = _body(refused["error"], path)
    assert "character budget" in body
    assert "next_offset" in body
    assert "page" not in body
    assert path.read_text(encoding="utf-8") == original


def test_unstable_snapshot_refusal_does_not_mention_pages(tmp_path, task):
    """A complete-looking read (no offset, truncated false) stamped partial
    because metadata moved during the read must not say to read more pages."""
    path = tmp_path / "unstable.txt"
    original = "alpha\nbeta\n"
    path.write_text(original, encoding="utf-8")

    def shifted(resolved: str):
        meta = _file_metadata(resolved)
        if meta is None:
            return None
        dev, ino, size, mtime_ns, ctime_ns = meta
        return dev, ino, size, mtime_ns + 1, ctime_ns

    with patch("tools.file_tools._file_metadata", shifted):
        read = json.loads(read_file_tool(str(path), task_id=task))
    assert "error" not in read, read
    assert read.get("truncated") is not True
    assert "next_offset" not in read
    refused = _refused(path, task)
    body = _body(refused["error"], path)
    assert "unstable snapshot" in body
    assert "page" not in body
    assert path.read_text(encoding="utf-8") == original


def test_stable_full_read_still_allows_overwrite(tmp_path, task):
    path = tmp_path / "full.txt"
    path.write_text("alpha\nbeta\n", encoding="utf-8")
    read = json.loads(read_file_tool(str(path), task_id=task))
    assert "error" not in read, read
    assert file_state.check_stale(task, path) is None
    written = json.loads(write_file_tool(str(path), "gamma\n", task_id=task))
    assert "error" not in written, written
    assert path.read_text(encoding="utf-8") == "gamma\n"
