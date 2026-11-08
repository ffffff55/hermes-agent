"""Classic CLI ghost text is accepted by Right, Ctrl-E and Ctrl-F.

Regression for NousResearch/hermes-agent#151352: a plain Application never
loads prompt_toolkit's auto-suggest bindings, so those keys fell through to
cursor movement. Overlay navigation that also binds Right keeps priority.
"""

import asyncio

import pytest

from tests.waiting import async_wait_until

_PREFIX = "okay let"
_SUFFIX = " the rest"


@pytest.fixture
def cli(monkeypatch):
    from cli import HermesCLI

    monkeypatch.setenv("HERMES_DEFER_AGENT_STARTUP", "1")
    cli = HermesCLI(model="fixture", provider="openai-compat", api_key="fixture",
                    base_url="http://127.0.0.1:1/v1")
    cli._tui_init_run_state()
    return cli


def _production_key_bindings(cli):
    """The registry ``run()`` installs: layout bindings, then auto-suggest merged ahead of them."""
    from prompt_toolkit.enums import EditingMode

    kb = cli._tui_build_key_bindings()
    layout, style = cli._tui_build_layout(kb)
    app = cli._tui_build_application(layout, kb, style)
    expect = EditingMode.VI if cli._vim_mode else EditingMode.EMACS
    assert app.editing_mode == expect
    return app.key_bindings


def _is_accept(binding) -> bool:
    return getattr(binding.handler, "__name__", "") == "_accept"


def _arm(editor, text, suggestion, *, cursor=None):
    from prompt_toolkit.auto_suggest import Suggestion

    editor.buffer.insert_text(text)
    editor.buffer.cursor_position = len(text) if cursor is None else cursor
    editor.buffer.suggestion = Suggestion(suggestion)


async def _until(predicate, timeout=3.0):
    await async_wait_until(predicate, timeout=timeout, interval=0.01)


def _drive(cli, script):
    """Run the production key registry against a real input area."""
    from prompt_toolkit.application import Application
    from prompt_toolkit.enums import EditingMode
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.layout import Layout
    from prompt_toolkit.output import DummyOutput

    editor = cli._tui_build_input_area()
    cli._input_area = editor
    bindings = _production_key_bindings(cli)
    mode = EditingMode.VI if cli._vim_mode else EditingMode.EMACS

    async def run():
        with create_pipe_input() as pipe:
            app = Application(layout=Layout(editor), key_bindings=bindings, editing_mode=mode,
                              input=pipe, output=DummyOutput())
            painted = asyncio.Event()
            app.after_render += lambda _: painted.set()
            task = asyncio.create_task(app.run_async())
            await asyncio.wait_for(painted.wait(), 3)
            try:
                await script(pipe, editor)
            finally:
                app.exit()
                await task

    asyncio.run(run())


def test_application_registry_includes_accept_keys_behind_overlay_right(cli):
    bindings = _production_key_bindings(cli)
    for key in ("right", "c-e", "c-f"):
        assert any(_is_accept(b) for b in bindings.get_bindings_for_keys((key,))), key
    assert any(getattr(b.handler, "__name__", "") == "_fill"
               for b in bindings.get_bindings_for_keys(("escape", "f")))

    right = bindings.get_bindings_for_keys(("right",))
    accept_at = next(i for i, b in enumerate(right) if _is_accept(b))
    connection = next(b for b in right if b.handler == cli._tui_connection_side)
    assert right.index(connection) > accept_at
    assert not connection.filter()

    cli._connection_state = {"phase": "authorized", "selected": 0}
    assert connection.filter()
    cli._connection_state = None

    cli._backtrack_state = {"entries": [(0, "ALPHA"), (1, "BETA")], "selected": 1}
    assert any(b.filter() for b in right[accept_at + 1:])
    cli._backtrack_state = None
    assert not any(b.filter() for b in right[accept_at + 1:])

    # prompt_toolkit parses "tab" as c-i. Auto-suggest must not take that key:
    # Tab stays the CLI handler (completion, ghost text, and the busy-queue binding).
    tab = bindings.get_bindings_for_keys(("c-i",))
    assert any(b.handler == cli._tui_handle_tab for b in tab)
    assert not any(_is_accept(b) for b in tab)


@pytest.mark.parametrize("chord", ["\x1b[C", "\x1bOC", "\x05", "\x06"],
                         ids=["right-csi", "right-ss3", "c-e", "c-f"])
def test_accept_keys_insert_the_ghost_suggestion(cli, chord):
    async def script(pipe, editor):
        _arm(editor, _PREFIX, _SUFFIX)
        pipe.send_text(chord)
        await _until(lambda: editor.text == _PREFIX + _SUFFIX)
        assert editor.buffer.cursor_position == len(editor.text)

    _drive(cli, script)


def test_right_accepts_the_suggestion_in_vi_insert_mode(cli):
    cli._vim_mode = True

    async def script(pipe, editor):
        _arm(editor, _PREFIX, _SUFFIX)
        pipe.send_text("\x1b[C")
        await _until(lambda: editor.text == _PREFIX + _SUFFIX)

    _drive(cli, script)


def test_alt_f_accepts_the_next_word_of_the_suggestion(cli):
    async def script(pipe, editor):
        _arm(editor, "ok", "continue now")
        pipe.send_text("\x1bf")
        await _until(lambda: editor.text == "okcontinue ")

    _drive(cli, script)


def test_right_moves_the_cursor_when_no_suggestion_can_be_accepted(cli):
    async def script(pipe, editor):
        _arm(editor, _PREFIX, "NO", cursor=4)
        pipe.send_text("\x1b[C")
        await _until(lambda: editor.buffer.cursor_position == 5)
        assert editor.text == _PREFIX

    _drive(cli, script)


def test_connection_nav_right_beats_an_available_suggestion(cli):
    cli._connection_state = {"phase": "authorized", "selected": 0}

    async def script(pipe, editor):
        _arm(editor, _PREFIX, _SUFFIX)
        pipe.send_text("\x1b[C")
        await _until(lambda: cli._connection_state["selected"] == 1)
        assert editor.text == _PREFIX

    _drive(cli, script)


def test_backtrack_right_beats_an_available_suggestion(cli):
    cli._backtrack_state = {"entries": [(0, "ALPHA"), (1, "BETA")], "selected": 1}

    async def script(pipe, editor):
        _arm(editor, _PREFIX, _SUFFIX)
        pipe.send_text("\x1b[C")
        await _until(lambda: cli._backtrack_state["selected"] == 0)
        assert editor.text == _PREFIX
        assert cli._backtrack_state.get("_reveal") is True

    _drive(cli, script)
