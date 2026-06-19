"""
Tests for core/skills.py and core/prompt_templates.py.
"""
from __future__ import annotations

import os
import tempfile

import pytest


# ============================================================================
# Skills
# ============================================================================

class TestSkills:
    def _write_skill(self, parent_dir: str, skill_name: str, content: str) -> str:
        """
        Skills in subdirs use SKILL.md (load_skills_from_dir recurses subdirs
        with include_root_files=False, which looks for SKILL.md).
        """
        skill_dir = os.path.join(parent_dir, skill_name)
        os.makedirs(skill_dir, exist_ok=True)
        path = os.path.join(skill_dir, "SKILL.md")
        with open(path, "w") as f:
            f.write(content)
        return path

    def test_load_skills_from_dir(self):
        from pi_coding_agent.core.skills import load_skills_from_dir
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_skill(
                tmpdir, "my-skill",
                "---\ndescription: A test skill\n---\nSkill body content.",
            )
            result = load_skills_from_dir(tmpdir, "user")
            assert len(result.skills) == 1
            assert result.skills[0].name == "my-skill"
            assert result.skills[0].description == "A test skill"
            assert result.skills[0].source_info is not None
            assert result.skills[0].source_info.path.endswith("SKILL.md")
            assert result.skills[0].source_info.source == "local"
            assert result.skills[0].source_info.scope == "user"
            assert result.skills[0].source_info.base_dir.endswith("my-skill")

    def test_load_skills_empty_dir(self):
        from pi_coding_agent.core.skills import load_skills_from_dir
        with tempfile.TemporaryDirectory() as tmpdir:
            result = load_skills_from_dir(tmpdir, "user")
            assert len(result.skills) == 0

    def test_load_skills_missing_description(self):
        from pi_coding_agent.core.skills import load_skills_from_dir
        with tempfile.TemporaryDirectory() as tmpdir:
            # Skill file without description in frontmatter (using SKILL.md)
            skill_dir = os.path.join(tmpdir, "no-desc")
            os.makedirs(skill_dir)
            path = os.path.join(skill_dir, "SKILL.md")
            with open(path, "w") as f:
                f.write("# Just a header\nNo frontmatter.")
            result = load_skills_from_dir(tmpdir, "user")
            assert len(result.skills) == 0

    def test_format_skills_for_prompt(self):
        from pi_coding_agent.core.skills import Skill, format_skills_for_prompt
        skills = [
            Skill(name="skill1", description="First skill", file_path="/a/skill1.md",
                  base_dir="/a", source="user"),
            Skill(name="skill2", description="Second skill", file_path="/b/skill2.md",
                  base_dir="/b", source="user"),
        ]
        formatted = format_skills_for_prompt(skills)
        assert "skill1" in formatted
        assert "skill2" in formatted
        assert "First skill" in formatted

    def test_load_skills_nonexistent_dir(self):
        from pi_coding_agent.core.skills import load_skills_from_dir
        result = load_skills_from_dir("/nonexistent/path/here", "user")
        assert len(result.skills) == 0

    def test_system_prompt_includes_skills_without_read_tool(self):
        from pi_coding_agent.core.system_prompt import build_system_prompt

        prompt = build_system_prompt(
            "/tmp/project",
            custom_prompt="Agent identity.",
            selected_tools=["runner"],
            skills=[{"name": "health-ping", "content": "Run /goal clear first."}],
        )

        assert "### health-ping" in prompt
        assert "Run /goal clear first." in prompt

    def test_bundled_skills_dir_resolves(self):
        """The bundled skills directory resolves in both wheel and dev layouts."""
        from pi_coding_agent.bundled_skills import get_bundled_skills_dir, is_enabled

        assert is_enabled() is True
        bundled = get_bundled_skills_dir()
        # Either the wheel-installed location or the dev location must resolve.
        assert bundled is not None
        assert os.path.isdir(bundled)

    def test_load_skills_includes_bundled_agent_build_pattern(self):
        """The bundled agent-build-pattern skill loads by default."""
        from pi_coding_agent.bundled_skills import is_enabled
        if not is_enabled():
            pytest.skip("bundled skills disabled via PI_NO_BUNDLED_SKILLS")

        from pi_coding_agent.core.skills import load_skills

        result = load_skills()
        bundled_skills = [s for s in result.skills if s.source == "bundled"]
        assert any(s.name == "agent-build-pattern" for s in bundled_skills)

    def test_load_skills_respects_no_bundled_env(self):
        """PI_NO_BUNDLED_SKILLS=1 suppresses the bundled skill."""
        from pi_coding_agent.core.skills import load_skills

        old = os.environ.get("PI_NO_BUNDLED_SKILLS")
        os.environ["PI_NO_BUNDLED_SKILLS"] = "1"
        try:
            result = load_skills()
        finally:
            if old is None:
                os.environ.pop("PI_NO_BUNDLED_SKILLS", None)
            else:
                os.environ["PI_NO_BUNDLED_SKILLS"] = old

        bundled_skills = [s for s in result.skills if s.source == "bundled"]
        assert bundled_skills == []

    def test_default_system_prompt_references_bundled_agent_build_skill(self):
        """The default system prompt advertises the bundled agent-build-pattern skill."""
        from pi_coding_agent.bundled_skills import is_enabled
        if not is_enabled():
            pytest.skip("bundled skills disabled via PI_NO_BUNDLED_SKILLS")

        from pi_coding_agent.core.system_prompt import build_system_prompt

        prompt = build_system_prompt("/tmp/project")
        assert "Agent build discipline" in prompt
        assert "agent-build-pattern" in prompt
        assert "bundled_skills/agent-build-pattern/SKILL.md" in prompt


# ============================================================================
# Prompt Templates
# ============================================================================

class TestPromptTemplates:
    def _write_template(self, directory: str, name: str, content: str) -> str:
        path = os.path.join(directory, f"{name}.md")
        with open(path, "w") as f:
            f.write(content)
        return path

    def test_load_prompt_templates(self):
        from pi_coding_agent.core.prompt_templates import (
            LoadPromptTemplatesOptions,
            load_prompt_templates,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_template(
                tmpdir, "my-template",
                "---\ndescription: A test template\n---\nHello $1!",
            )
            opts = LoadPromptTemplatesOptions(
                prompt_paths=[tmpdir], include_defaults=False
            )
            result = load_prompt_templates(opts)
            assert len(result) == 1
            assert result[0].name == "my-template"
            assert "A test template" in result[0].description
            assert result[0].source_info is not None
            assert result[0].source_info.path.endswith("my-template.md")
            assert result[0].source_info.source == "local"
            assert result[0].source_info.scope == "temporary"
            assert result[0].source_info.base_dir == tmpdir

    def test_substitute_args_positional(self):
        from pi_coding_agent.core.prompt_templates import substitute_args
        # substitute_args(content, args: list[str]) - positional $1, $2
        template = "Hello $1, you are $2 years old."
        result = substitute_args(template, ["Alice", "30"])
        assert result == "Hello Alice, you are 30 years old."

    def test_substitute_args_all(self):
        from pi_coding_agent.core.prompt_templates import substitute_args
        template = "Args: $@"
        result = substitute_args(template, ["a", "b", "c"])
        assert "a b c" in result

    def test_substitute_args_missing_position(self):
        from pi_coding_agent.core.prompt_templates import substitute_args
        template = "Hello $1 and $2!"
        result = substitute_args(template, ["Alice"])
        assert "Alice" in result
        assert "$2" not in result  # Missing args become empty string

    def test_expand_prompt_template_match(self):
        from pi_coding_agent.core.prompt_templates import (
            PromptTemplate,
            expand_prompt_template,
        )
        tpl = PromptTemplate(
            name="greet",
            description="Greet someone",
            content="Hello $1!",
            source="user",
            file_path="/tmp/greet.md",
        )
        expanded = expand_prompt_template("/greet World", [tpl])
        assert "Hello World!" in expanded

    def test_expand_prompt_template_no_match(self):
        from pi_coding_agent.core.prompt_templates import (
            PromptTemplate,
            expand_prompt_template,
        )
        tpl = PromptTemplate(
            name="greet",
            description="Greet someone",
            content="Hello $1!",
            source="user",
            file_path="/tmp/greet.md",
        )
        # Not a slash command
        result = expand_prompt_template("Hello World", [tpl])
        assert result == "Hello World"

    def test_parse_command_args(self):
        from pi_coding_agent.core.prompt_templates import parse_command_args
        # Returns all tokens including the command name
        result = parse_command_args("/greet Alice Bob")
        assert "Alice" in result
        assert "Bob" in result

    def test_load_prompt_templates_empty(self):
        from pi_coding_agent.core.prompt_templates import (
            LoadPromptTemplatesOptions,
            load_prompt_templates,
        )
        opts = LoadPromptTemplatesOptions(include_defaults=False)
        result = load_prompt_templates(opts)
        assert isinstance(result, list)
