"""Tests for tools/cronjob_job_args.py::_validate_cron_script_path (issue #105761).

Regression coverage: the validator used to accept a script path that pointed
at a file that doesn't exist, only failing later at every cron fire with a
generic "Script not found" error from cron/scheduler_script.py. It also
hardcoded "~/.hermes/scripts/" in its messages even though resolution goes
through get_hermes_home(), which is per-profile.
"""

from hermes_constants import get_hermes_home
from tools.cronjob_job_args import _validate_cron_script_path, _validate_cron_skills


def _plant_skill(directory: str, frontmatter_name: str | None = None) -> None:
    skill_dir = get_hermes_home() / "skills" / directory
    skill_dir.mkdir(parents=True, exist_ok=True)
    shown = frontmatter_name or directory.split("/")[-1]
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {shown}\ndescription: planted {shown}\n---\n\nBody.\n",
        encoding="utf-8",
    )


class TestValidateCronScriptPath:
    def test_missing_script_file_is_rejected_at_creation_time(self):
        error = _validate_cron_script_path("does_not_exist.sh")
        assert error is not None
        assert "not found" in error.lower()

    def test_existing_script_file_passes(self):
        scripts_dir = get_hermes_home() / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        (scripts_dir / "real.sh").write_text("#!/bin/sh\necho hi\n")
        assert _validate_cron_script_path("real.sh") is None

    def test_missing_file_error_names_the_resolved_scripts_dir(self):
        # The resolved dir must appear literally so the message stays correct
        # under profiles, where get_hermes_home() is not the global ~/.hermes.
        scripts_dir = get_hermes_home() / "scripts"
        error = _validate_cron_script_path("missing.py")
        assert str(scripts_dir) in error

    def test_absolute_path_error_names_the_resolved_scripts_dir_not_global_literal(self):
        scripts_dir = get_hermes_home() / "scripts"
        error = _validate_cron_script_path("/etc/passwd")
        assert str(scripts_dir) in error


class TestValidateCronSkills:
    """``--skill`` uses the same lookup cron loads with (directory, relative
    path, frontmatter name, bundle slug) — not a separate name list."""

    def test_missing_name_is_reported(self):
        error = _validate_cron_skills(["nonexistent-skill-xyz"])
        assert error is not None
        assert "Skill not found: nonexistent-skill-xyz" in error

    def test_every_missing_name_is_reported(self):
        _plant_skill("planted-ok")
        # Both misses are named together so one typo in a repeated --skill
        # does not hide the next. The installed name is not part of the error.
        error = _validate_cron_skills(["planted-ok", "no-such-skill-a", "no-such-skill-b"])
        assert error is not None
        assert "no-such-skill-a" in error
        assert "no-such-skill-b" in error
        assert "planted-ok" not in error

    def test_directory_and_frontmatter_names_pass(self):
        _plant_skill("dir-only-slug", frontmatter_name="display-slug")
        assert _validate_cron_skills(["dir-only-slug"]) is None
        assert _validate_cron_skills(["display-slug"]) is None

    def test_category_alias_passes(self):
        _plant_skill("widgets/demo-widget")
        assert _validate_cron_skills(["widgets/demo-widget"]) is None
        assert _validate_cron_skills(["widgets:demo-widget"]) is None

    def test_bundle_slug_passes_without_a_skill_of_that_name(self, monkeypatch):
        monkeypatch.delenv("HERMES_BUNDLES_DIR", raising=False)
        bundles = get_hermes_home() / "skill-bundles"
        bundles.mkdir(parents=True, exist_ok=True)
        (bundles / "daily-pack.yaml").write_text(
            "name: daily-pack\nskills:\n  - not-installed-member\n",
            encoding="utf-8",
        )
        assert _validate_cron_skills(["daily-pack"]) is None

    def test_empty_list_is_not_an_error(self):
        assert _validate_cron_skills(None) is None
        assert _validate_cron_skills([]) is None
