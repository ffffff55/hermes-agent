"""Regression: image-part detection must not hash JSON Schema ``type`` values.

Structured tool results can carry JSON Schema unions such as
``{"type": ["string", "null"]}``. Set membership hashes that value and
raises ``TypeError`` before the provider request opens (#148586).
Non-image type values are not image parts; a real image part still is.
"""

import pytest

from agent.client_lifecycle import ClientLifecycleMixin

# JSON Schema ``type`` values that are not native image parts. Lists and
# dicts are unhashable; the others are hashable but not image kinds.
_SCHEMA_TYPES = [
    pytest.param(["string", "null"], id="list"),
    pytest.param([], id="empty-list"),
    pytest.param({"anyOf": [{"type": "string"}]}, id="dictionary"),
    pytest.param(None, id="null"),
    pytest.param(1, id="number"),
    pytest.param("text", id="text"),
]

_IMAGE_PARTS = {
    "messages": {
        "type": "image_url",
        "image_url": {"url": "https://example.invalid/cat.png"},
    },
    "input": {
        "type": "input_image",
        "image_url": "https://example.invalid/cat.png",
    },
}


def _tool_result_request(field, schema_type):
    """Chat ``messages`` or Responses ``input`` carrying one schema-typed tool result."""
    return {
        field: [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "content": {"schema": {"type": schema_type}},
                    }
                ],
            }
        ]
    }


@pytest.mark.parametrize("field", ["messages", "input"])
@pytest.mark.parametrize("schema_type", _SCHEMA_TYPES)
def test_non_image_schema_types_are_not_image_parts(field, schema_type):
    request = _tool_result_request(field, schema_type)

    assert ClientLifecycleMixin._api_kwargs_have_image_parts(request) is False

    request[field][0]["content"].append(dict(_IMAGE_PARTS[field]))
    assert ClientLifecycleMixin._api_kwargs_have_image_parts(request) is True
