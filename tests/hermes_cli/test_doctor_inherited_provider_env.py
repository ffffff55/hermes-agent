"""Doctor must not flag missing API keys when a profile inherits them.

Regression for #151049: ``_check_env_file`` scanned only the profile-local
``.env``. A named profile that keeps platform tokens locally and resolves
provider credentials from the shared Hermes-root ``.env`` (or the process
environment) was told to run ``hermes setup`` even though Auth Providers and
API Connectivity could already see the key.
"""

from __future__ import annotations

import hermes_constants
from hermes_cli import doctor as doctor_mod
from hermes_cli import doctor_config

_PLATFORM_ONLY = "TELEGRAM_BOT_TOKEN=000:platform-only\nTELEGRAM_ALLOWED_USERS=1\n"
_SHARED_MARKER = "configured-in-shared-scope"
_PROCESS_MARKER = "configured-in-process-scope"
_CONFIGURE_API_KEYS = "Run 'hermes setup' to configure API keys"


def _clear_provider_hints(monkeypatch) -> None:
    for key in doctor_mod._PROVIDER_ENV_HINTS:
        monkeypatch.delenv(key, raising=False)


def _named_profile(tmp_path, *, shared: str, profile: str):
    """``<tmp>/.hermes/.env`` plus ``profiles/zero/.env``. Returns the profile home."""
    root = tmp_path / ".hermes"
    profile_home = root / "profiles" / "zero"
    profile_home.mkdir(parents=True)
    (root / ".env").write_text(shared, encoding="utf-8")
    (profile_home / ".env").write_text(profile, encoding="utf-8")
    return profile_home


def _check(monkeypatch, profile_home):
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    monkeypatch.setattr(doctor_mod, "HERMES_HOME", profile_home)
    monkeypatch.setattr(hermes_constants, "_default_hermes_root_memo", None)
    return doctor_config._check_env_file(False)


def test_shared_root_provider_key_is_not_flagged(monkeypatch, tmp_path, capsys):
    """Shared ``~/.hermes/.env`` holds the provider key; the profile file does not."""
    _clear_provider_hints(monkeypatch)
    profile_home = _named_profile(
        tmp_path,
        shared=f"MINIMAX_API_KEY={_SHARED_MARKER}\n",
        profile=_PLATFORM_ONLY,
    )

    finding = _check(monkeypatch, profile_home)
    out = capsys.readouterr().out

    assert _CONFIGURE_API_KEYS not in finding.issues
    assert "API key or custom endpoint configured" in out
    assert _SHARED_MARKER not in out
    assert _SHARED_MARKER not in " ".join(finding.issues)


def test_process_env_provider_key_is_not_flagged(monkeypatch, tmp_path, capsys):
    """A provider key exported into the process counts even when neither file has one."""
    _clear_provider_hints(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", _PROCESS_MARKER)
    profile_home = _named_profile(
        tmp_path,
        shared=_PLATFORM_ONLY,
        profile=_PLATFORM_ONLY,
    )

    finding = _check(monkeypatch, profile_home)
    out = capsys.readouterr().out

    assert _CONFIGURE_API_KEYS not in finding.issues
    assert _PROCESS_MARKER not in out


def test_no_provider_key_in_any_scope_still_flags(monkeypatch, tmp_path, capsys):
    """Platform tokens alone, in the shared file and the profile file, still flag."""
    _clear_provider_hints(monkeypatch)
    profile_home = _named_profile(
        tmp_path,
        shared=_PLATFORM_ONLY,
        profile=_PLATFORM_ONLY,
    )

    finding = _check(monkeypatch, profile_home)
    out = capsys.readouterr().out

    assert _CONFIGURE_API_KEYS in finding.issues
    assert "No API key found" in out
