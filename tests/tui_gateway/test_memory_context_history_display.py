"""History hydration must not paint an unterminated memory fence (#149933).

The live stream already drops a block-boundary ``<memory-context>`` span that never
closes. The persisted assistant row stays raw for replay; Desktop and the TUI read
the display projection (``_history_to_messages``, and the REST twin
``_project_for_display``).
"""

from hermes_cli.web_routers.sessions import _project_for_display
from tui_gateway.server import _history_to_messages

_NOTE = "[System context note - internal memory injection metadata]"
_UNTERMINATED = f"Normal assistant reply.\n\n<memory-context>\n{_NOTE}"
_COMPLETE = "Before.\n\n<memory-context>\nsecret fact\n</memory-context>\n\nAfter."
_INLINE = "The <memory-context> tag name is documented here."
_CODE = "In docs, write `<memory-context>` when you mean the fence."


def test_history_to_messages_strips_unterminated_assistant_fence():
    row = {"role": "assistant", "content": _UNTERMINATED, "api_content": _UNTERMINATED}
    shown = _history_to_messages([row])[0]

    assert shown["text"].startswith("Normal assistant reply.")
    assert _NOTE not in shown["text"]
    assert "memory-context" not in shown["text"].lower()
    assert "api_content" not in shown
    assert row["content"] == _UNTERMINATED
    assert row["api_content"] == _UNTERMINATED


def test_history_to_messages_strips_complete_assistant_fence():
    row = {"role": "assistant", "content": _COMPLETE}
    shown = _history_to_messages([row])[0]

    assert shown["text"].startswith("Before.")
    assert "After." in shown["text"]
    assert "secret fact" not in shown["text"]
    assert "<memory-context>" not in shown["text"]
    assert row["content"] == _COMPLETE


def test_history_to_messages_keeps_inline_and_code_mentions():
    for raw in (_INLINE, _CODE):
        row = {"role": "assistant", "content": raw}
        assert _history_to_messages([row])[0]["text"] == raw
        assert row["content"] == raw


def test_history_to_messages_leaves_user_fence_and_hidden_rows():
    user = {"role": "user", "content": _UNTERMINATED}
    hidden = {"role": "assistant", "display_kind": "hidden", "content": _UNTERMINATED}

    assert _history_to_messages([user])[0]["text"] == _UNTERMINATED
    assert _history_to_messages([hidden]) == []
    assert user["content"] == _UNTERMINATED
    assert hidden["content"] == _UNTERMINATED


def test_history_to_messages_scrubs_joined_assistant_parts():
    parts = [
        {"type": "text", "text": "Visible answer.\n\n"},
        {"type": "text", "text": f"<memory-context>\n{_NOTE}"},
    ]
    row = {"role": "assistant", "content": parts}
    shown = _history_to_messages([row])[0]

    assert shown["text"].startswith("Visible answer.")
    assert _NOTE not in shown["text"]
    assert parts[1]["text"].endswith(_NOTE)


def test_history_to_messages_keeps_reasoning_when_fence_was_the_only_text():
    row = {
        "role": "assistant",
        "content": f"<memory-context>\n{_NOTE}",
        "reasoning": "thought",
    }
    shown = _history_to_messages([row])[0]

    assert _NOTE not in shown["text"]
    assert shown["reasoning"] == "thought"


def test_rest_projection_strips_unterminated_assistant_content():
    row = {"role": "assistant", "content": _UNTERMINATED, "api_content": _UNTERMINATED}
    projected = _project_for_display([row])[0]

    assert projected["content"].startswith("Normal assistant reply.")
    assert _NOTE not in projected["content"]
    assert "memory-context" not in projected["content"].lower()
    assert projected["api_content"] == _UNTERMINATED
    assert row["content"] == _UNTERMINATED
    assert row["api_content"] == _UNTERMINATED


def test_rest_projection_keeps_inline_mention_and_user_fence():
    inline = {"role": "assistant", "content": _INLINE}
    user = {"role": "user", "content": _COMPLETE}
    hidden = {"role": "assistant", "display_kind": "hidden", "content": _UNTERMINATED}

    assert _project_for_display([inline])[0]["content"] == _INLINE
    assert _project_for_display([user])[0]["content"] == _COMPLETE
    assert _project_for_display([hidden])[0]["content"] == _UNTERMINATED
    assert hidden["content"] == _UNTERMINATED


def test_rest_projection_scrubs_assistant_text_part():
    parts = [
        {"type": "text", "text": "Visible answer.\n\n"},
        {"type": "text", "text": f"<memory-context>\n{_NOTE}"},
    ]
    row = {"role": "assistant", "content": parts}
    projected = _project_for_display([row])[0]
    shown = "".join(part["text"] for part in projected["content"])

    assert shown.startswith("Visible answer.")
    assert _NOTE not in shown
    assert parts[1]["text"].endswith(_NOTE)
    assert row["content"] is parts
