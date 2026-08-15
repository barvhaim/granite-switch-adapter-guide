import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINK_PATTERN = re.compile(r"\[[^]]+\]\(([^)]+)\)")


def test_local_markdown_links_resolve():
    markdown_files = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
    missing: list[str] = []

    for markdown_file in markdown_files:
        for target in LINK_PATTERN.findall(markdown_file.read_text(encoding="utf-8")):
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            relative_target = target.split("#", 1)[0]
            if not relative_target:
                continue
            resolved = (markdown_file.parent / relative_target).resolve()
            if not resolved.exists():
                missing.append(f"{markdown_file.relative_to(ROOT)} -> {target}")

    assert not missing, "Missing local Markdown links:\n" + "\n".join(missing)


def test_readme_referenced_scripts_exist():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    referenced = sorted(set(re.findall(r"scripts/[A-Za-z0-9_.-]+\.py", readme)))
    assert referenced
    assert all((ROOT / path).is_file() for path in referenced)
