"""Regression for #148237.

Without psutil the live-system guard cannot walk the process tree. A child
the guard itself started (a server fixture's ``Popen``) must still be
signalable, and every pid it did not start must still be refused. Allowing
the kill to raise inside teardown aborts cleanup and leaves ``Popen.__exit__``
blocked in ``wait()``.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys

import pytest

_ABSENT = object()


@pytest.fixture(scope="module", autouse=True)
def _psutil_import_fails():
    """Force the guard's no-psutil branch.

    The guard imports psutil when its function fixture starts. This module
    fixture runs first; ``sys.modules['psutil'] is None`` makes that import
    raise, which the guard already treats as psutil absent. A function-scoped
    hide would run too late.
    """
    saved = sys.modules.get("psutil", _ABSENT)
    sys.modules["psutil"] = None
    try:
        yield
    finally:
        if saved is _ABSENT:
            sys.modules.pop("psutil", None)
        else:
            sys.modules["psutil"] = saved


def _unused_pid() -> int:
    """A pid that is not a live process.

    Signal 0 is a liveness probe the guard lets through, so this does not
    depend on the allowlist. ``ProcessLookupError`` is the POSIX "no such
    process" result; an existing pid we may not signal is not unused.
    """
    for candidate in range(1 << 22, (1 << 22) + 64):
        try:
            os.kill(candidate, 0)
        except ProcessLookupError:
            return candidate
        except PermissionError:
            continue
    raise RuntimeError("could not find an unused pid")


def _child_blocking_on_stdin() -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.read()"],
        stdin=subprocess.PIPE,
    )


def _reap(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is None and proc.stdin is not None:
        try:
            proc.stdin.close()
        except OSError:
            pass
    proc.wait(timeout=5)


def test_no_psutil_allows_signal_to_guard_spawned_child():
    """Teardown must be able to signal a child this guard's Popen started."""
    proc = _child_blocking_on_stdin()
    try:
        os.kill(proc.pid, signal.SIGTERM)
        proc.wait(timeout=5)
    finally:
        _reap(proc)
    assert proc.returncode not in (None, 0)


@pytest.mark.platforms("linux")
def test_no_psutil_refuses_foreign_pid():
    with pytest.raises(RuntimeError, match="live-system guard"):
        os.kill(1, signal.SIGTERM)


@pytest.mark.platforms("posix")
def test_no_psutil_branch_refuses_unrecorded_pid():
    """A missing pid is allowlisted once psutil can see it is gone. Refusing
    it is what proves this module is on the no-psutil branch, so the spawned-
    child test cannot pass just because the process-tree walk is still on.
    """
    with pytest.raises(RuntimeError, match="live-system guard"):
        os.kill(_unused_pid(), signal.SIGTERM)
