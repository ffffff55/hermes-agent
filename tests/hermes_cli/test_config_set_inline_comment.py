"""Regression for #148524: ``hermes config set`` must not corrupt a list key's comment.

A block list whose key carries an inline comment used to rewrite as the comment
on one line, a bare ``[]`` on the next, and the following keys smashed together.
The file no longer parsed. The empty value has to land on the key line, the
comment has to survive, and neighboring keys have to stay on their own lines.
"""

import hermes_yaml as yaml
import pytest


@pytest.fixture
def user_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_MANAGED_DIR", raising=False)
    import hermes_cli.config as cfg
    from hermes_cli import managed_scope

    cfg._LOAD_CONFIG_CACHE.clear()
    cfg._RAW_CONFIG_CACHE.clear()
    managed_scope.invalidate_managed_cache()
    return home


def _set_empty_list(home, text: str) -> str:
    (home / "config.yaml").write_text(text, encoding="utf-8")
    from hermes_cli.config import set_config_value

    set_config_value("kanban.dispatch_profiles", "[]")
    return (home / "config.yaml").read_text(encoding="utf-8")


def _assert_neighbors(text: str, data: dict) -> None:
    kanban = data["kanban"]
    assert kanban["dispatch_profiles"] == []
    assert kanban["dispatch_in_gateway"] is True
    assert kanban["dispatch_interval_seconds"] == 60
    lines = [line.strip() for line in text.splitlines()]
    assert "dispatch_in_gateway: true" in lines
    assert "dispatch_interval_seconds: 60" in lines


def test_config_set_empty_list_keeps_inline_comment_parseable(user_home):
    text = _set_empty_list(
        user_home,
        "kanban:\n"
        "  dispatch_profiles:     # some comment\n"
        "    - mordecai\n"
        "  dispatch_in_gateway: true\n"
        "  dispatch_interval_seconds: 60\n",
    )
    data = yaml.safe_load(text)
    _assert_neighbors(text, data)
    lines = [line.strip() for line in text.splitlines()]
    assert "dispatch_profiles: []  # some comment" in lines


def test_config_set_empty_list_keeps_comment_above_items_parseable(user_home):
    """A comment between the key and the block is the same dump failure."""
    text = _set_empty_list(
        user_home,
        "kanban:\n"
        "  dispatch_profiles:\n"
        "    # why this list\n"
        "    - mordecai\n"
        "  dispatch_in_gateway: true\n"
        "  dispatch_interval_seconds: 60\n",
    )
    data = yaml.safe_load(text)
    _assert_neighbors(text, data)
    assert "# why this list" in text
