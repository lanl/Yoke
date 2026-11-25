"""Consistency checks for the agent skills under ``skills/``.

The skills are Markdown instructions for AI agents, so nothing executes them in
CI. These tests catch the ways they silently go stale: malformed frontmatter,
broken ``.claude/skills`` symlinks, repo paths or ``yoke.`` names that no longer
exist, a batch-contract table that no longer matches the datastep source, and
saved machine profiles missing their date or allocation warning.
"""

import ast
import importlib
import inspect
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILLS_DIR = REPO_ROOT / "skills"
CLAUDE_SKILLS_DIR = REPO_ROOT / ".claude" / "skills"
CONTRACTS_FILE = SKILLS_DIR / "shared" / "batch_contracts.md"
MACHINES_DIR = SKILLS_DIR / "yoke-slurm-template" / "machines"

# Skills the plans (dev_plans/agent_skills_*.md) commit to.
EXPECTED_SKILLS = {
    "yoke-new-dataset",
    "yoke-new-training-step",
    "yoke-slurm-template",
    "yoke-new-harness",
    # Phase 2 (dev_plans/agent_skills_phase2.md).
    "yoke-train-my-data",
    "yoke-model",
    "yoke-new-loss",
    "yoke-launch-study",
    "yoke-run-doctor",
    # Remote use through a Nomad MCP server.
    "yoke-finetune-mcp",
}

# Inline code spans (`...`) are where skills name paths and Python objects.
# Fenced code blocks are skipped: they hold illustrative snippets.
CODE_SPAN_RE = re.compile(r"`([^`\n]+)`")
FENCE_RE = re.compile(r"^```.*?^```", re.MULTILINE | re.DOTALL)

# A repo-relative path starts with one of these top-level entries. An optional
# ``:line`` or ``:line-line`` suffix (as in ``file.py:42``) is stripped.
PATH_PREFIXES = (
    "src/",
    "applications/",
    "tests/",
    "skills/",
    "docs/",
    "dev_plans/",
    ".claude/",
)
ROOT_FILES = ("AGENTS.md", "pyproject.toml")
PATH_RE = re.compile(r"^(?P<path>[\w./-]+?)(?::\d+(?:-\d+)?)?$")

# Characters marking a path or name as a pattern or placeholder rather than a
# concrete file (e.g. ``tests/datasets/test_<module>.py``, ``study_###``).
PLACEHOLDER_CHARS = ("<", ">", "*", "#", "{", "}", "...", "@@")

# Dotted Yoke names, e.g. ``yoke.utils.builders.build_adamw``.
YOKE_NAME_RE = re.compile(r"^yoke(?:\.[A-Za-z_]\w*)+$")


def _skill_dirs() -> list[Path]:
    """Return the skill directories (every ``skills/`` child except ``shared``)."""
    return sorted(p for p in SKILLS_DIR.iterdir() if p.is_dir() and p.name != "shared")


def _markdown_files() -> list[Path]:
    """Return every Markdown file under ``skills/``."""
    return sorted(SKILLS_DIR.rglob("*.md"))


def _code_spans(path: Path) -> list[str]:
    """Return the inline code spans of a Markdown file, outside fenced blocks.

    Args:
        path (Path): Markdown file to scan.

    Returns:
        list[str]: Stripped contents of each inline code span.
    """
    text = FENCE_RE.sub("", path.read_text())
    return [m.group(1).strip() for m in CODE_SPAN_RE.finditer(text)]


def _is_placeholder(token: str) -> bool:
    """Return whether a token is a pattern or placeholder, not a concrete name.

    Args:
        token (str): Inline code span contents.

    Returns:
        bool: True if the token contains a placeholder marker.
    """
    return any(ch in token for ch in PLACEHOLDER_CHARS)


def _parse_frontmatter(path: Path) -> dict[str, str]:
    """Parse the simple ``key: value`` YAML frontmatter of a ``SKILL.md``.

    Only single-line ``key: value`` pairs are supported, which is all the skills
    use. This avoids depending on PyYAML, which Yoke does not declare.

    Args:
        path (Path): ``SKILL.md`` file.

    Returns:
        dict[str, str]: Frontmatter keys and values.

    Raises:
        ValueError: If the file does not start with a ``---`` delimited block.
    """
    lines = path.read_text().splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError(f"{path} does not start with '---' frontmatter.")
    meta = {}
    for line in lines[1:]:
        if line.strip() == "---":
            return meta
        key, sep, value = line.partition(":")
        if sep:
            meta[key.strip()] = value.strip()
    raise ValueError(f"{path} frontmatter is not closed with '---'.")


def _resolve_yoke_name(name: str) -> object:
    """Import the longest module prefix of a dotted name, then getattr the rest.

    Args:
        name (str): Dotted name starting with ``yoke.``.

    Returns:
        object: The resolved module, class, function, or attribute.

    Raises:
        ImportError: If no prefix of the name is an importable module.
        AttributeError: If the remainder is not an attribute of that module.
    """
    parts = name.split(".")
    for split in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:split]))
        except ModuleNotFoundError:
            continue
        for attr in parts[split:]:
            obj = getattr(obj, attr)
        return obj
    raise ImportError(f"No importable module prefix in {name!r}.")


def _contract_rows() -> list[tuple[str, str, str]]:
    """Parse the machine-readable table in ``batch_contracts.md``.

    The table sits between ``<!-- contracts:begin -->`` and
    ``<!-- contracts:end -->`` and has columns ``contract | datastep | tuple``.

    Returns:
        list[tuple[str, str, str]]: ``(contract, datastep name, tuple)`` rows,
        with backticks stripped.
    """
    text = CONTRACTS_FILE.read_text()
    block = text.split("<!-- contracts:begin -->")[1].split("<!-- contracts:end -->")[0]
    rows = []
    for line in block.splitlines():
        cells = [c.strip().strip("`") for c in line.strip().strip("|").split("|")]
        # Skip the header and the |---| separator row.
        if len(cells) != 3 or cells[0] in ("contract", "---"):
            continue
        rows.append((cells[0], cells[1], cells[2]))
    return rows


def _unpacked_from_data(func: object) -> list[str]:
    """Return each target unpacked from ``data`` in a function body.

    Finds assignments ``a, b, c = data`` (with or without parentheses) and
    returns their targets normalized to ``"a, b, c"``.

    Args:
        func (object): Function whose source is inspected.

    Returns:
        list[str]: Normalized target lists, in source order.
    """
    tree = ast.parse(inspect.cleandoc("\n" + inspect.getsource(func)))
    found = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Name)
            and node.value.id == "data"
            and isinstance(node.targets[0], ast.Tuple)
        ):
            found.append(", ".join(ast.unparse(e) for e in node.targets[0].elts))
    return found


# ---------------------------------------------------------------------------
# Skill discovery and frontmatter
# ---------------------------------------------------------------------------
def test_expected_skills_present() -> None:
    """Every planned skill exists and every skill directory has a SKILL.md."""
    names = {p.name for p in _skill_dirs()}
    assert EXPECTED_SKILLS <= names
    for skill in _skill_dirs():
        assert (skill / "SKILL.md").is_file(), f"{skill} has no SKILL.md"


@pytest.mark.parametrize("skill", _skill_dirs(), ids=lambda p: p.name)
def test_frontmatter(skill: Path) -> None:
    """SKILL.md frontmatter has a matching name and a non-empty description."""
    meta = _parse_frontmatter(skill / "SKILL.md")
    assert meta.get("name") == skill.name
    assert meta.get("description"), f"{skill.name} has an empty description"


# ---------------------------------------------------------------------------
# Claude Code symlinks
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("skill", _skill_dirs(), ids=lambda p: p.name)
def test_claude_symlink(skill: Path) -> None:
    """Each skill has a relative .claude/skills symlink resolving to it."""
    link = CLAUDE_SKILLS_DIR / skill.name
    assert link.is_symlink(), f"{link} is missing or not a symlink"
    target = link.readlink()
    assert not target.is_absolute(), f"{link} must be a relative symlink"
    assert link.resolve() == skill.resolve()


# ---------------------------------------------------------------------------
# References from skills to the repo
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "md", _markdown_files(), ids=lambda p: str(p.relative_to(SKILLS_DIR))
)
def test_repo_paths_exist(md: Path) -> None:
    """Every concrete repo path named in a skill file exists."""
    missing = []
    for token in _code_spans(md):
        if _is_placeholder(token):
            continue
        if not (token.startswith(PATH_PREFIXES) or token in ROOT_FILES):
            continue
        match = PATH_RE.match(token)
        if match is None:
            # Not a bare path (e.g. a command line); nothing to check.
            continue
        path = match.group("path").rstrip("/")
        if not (REPO_ROOT / path).exists():
            missing.append(token)
    assert not missing, f"{md.relative_to(REPO_ROOT)} names missing paths: {missing}"


@pytest.mark.parametrize(
    "md", _markdown_files(), ids=lambda p: str(p.relative_to(SKILLS_DIR))
)
def test_yoke_names_import(md: Path) -> None:
    """Every dotted ``yoke.`` name in a skill file resolves to a real object."""
    broken = []
    for token in _code_spans(md):
        # Allow a call suffix such as ``yoke.x.f(...)``; check the name only.
        name = token.split("(")[0].strip()
        if _is_placeholder(name) or not YOKE_NAME_RE.match(name):
            continue
        try:
            _resolve_yoke_name(name)
        except (ImportError, AttributeError) as err:
            broken.append(f"{name} ({type(err).__name__})")
    assert not broken, f"{md.relative_to(REPO_ROOT)} names broken objects: {broken}"


# ---------------------------------------------------------------------------
# Batch contracts
# ---------------------------------------------------------------------------
def test_contract_table_not_empty() -> None:
    """The contract table parses and has rows."""
    assert len(_contract_rows()) > 0


@pytest.mark.parametrize(
    ("contract", "datastep", "expected"),
    _contract_rows(),
    ids=[row[1].rsplit(".", 1)[-1] for row in _contract_rows()],
)
def test_contract_matches_source(contract: str, datastep: str, expected: str) -> None:
    """Each table row's tuple is exactly what its datastep unpacks from data."""
    func = _resolve_yoke_name(datastep)
    unpacked = _unpacked_from_data(func)
    assert unpacked, f"{datastep} has no '... = data' unpacking line"
    assert expected in unpacked, (
        f"{datastep} ({contract}) unpacks {unpacked}, table says ({expected})"
    )


def test_all_public_datasteps_listed() -> None:
    """Every public datastep function appears in the contract table."""
    import yoke.utils.training.datastep as pkg

    listed = {row[1] for row in _contract_rows()}
    unlisted = []
    for py in sorted(Path(pkg.__path__[0]).glob("*.py")):
        if py.name == "__init__.py":
            continue
        module = importlib.import_module(f"{pkg.__name__}.{py.stem}")
        for name, obj in inspect.getmembers(module, inspect.isfunction):
            if obj.__module__ != module.__name__ or name.startswith("_"):
                continue
            if not name.endswith(("_datastep", "_datastep_cylex")):
                continue
            if f"{module.__name__}.{name}" not in listed:
                unlisted.append(f"{module.__name__}.{name}")
    assert not unlisted, f"Datasteps missing from batch_contracts.md: {unlisted}"


# ---------------------------------------------------------------------------
# Saved machine profiles
# ---------------------------------------------------------------------------
def _machine_profiles() -> list[Path]:
    """Return saved machine profiles (every machines/*.md except README.md)."""
    if not MACHINES_DIR.is_dir():
        return []
    return sorted(p for p in MACHINES_DIR.glob("*.md") if p.name != "README.md")


@pytest.mark.parametrize("profile", _machine_profiles(), ids=lambda p: p.stem)
def test_machine_profile(profile: Path) -> None:
    """Profiles are dated and keep the volatile-allocations warning."""
    text = profile.read_text()
    assert re.search(r"^last verified: \d{4}-\d{2}-\d{2}\s*$", text, re.MULTILINE)
    assert "## Allocations seen" in text
    assert "re-query" in text.lower()
