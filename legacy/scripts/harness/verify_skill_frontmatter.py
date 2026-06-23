from __future__ import annotations

import re
from pathlib import Path

from check_utils import ROOT, fail_with_violations, pass_check


SKILLS_DIR = ROOT / ".agents" / "skills"


def parse_frontmatter(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    match = re.match(r"---\n(.*?)\n---\n", text, flags=re.DOTALL)
    if not match:
        return {}
    data: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            data[key.strip()] = value.strip()
    return data


def main() -> None:
    expected = {
        "v2x-urban-environment-sim",
        "hierarchical-gnn-topology-constructor",
        "avalanche-closed-form-consensus",
        "coupled-loss-pcgrad-gradnorm",
        "scalability-evaluation-harness",
    }
    violations: list[tuple[Path, int, str, str]] = []

    if not SKILLS_DIR.exists():
        violations.append((SKILLS_DIR, 1, "Missing .agents/skills directory", "missing"))
        fail_with_violations("skill frontmatter", violations)

    found = {path.name for path in SKILLS_DIR.iterdir() if path.is_dir()}
    for missing in sorted(expected - found):
        violations.append((SKILLS_DIR / missing, 1, "Missing required skill directory", missing))

    for skill_dir in sorted(SKILLS_DIR.iterdir()):
        if not skill_dir.is_dir():
            continue
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            violations.append((skill_md, 1, "Missing SKILL.md", "missing"))
            continue
        metadata = parse_frontmatter(skill_md)
        if metadata.get("name") != skill_dir.name:
            violations.append((skill_md, 1, "Frontmatter name must match folder name", metadata.get("name", "")))
        if not metadata.get("description"):
            violations.append((skill_md, 1, "Frontmatter description is required", "description"))
        if len(metadata.get("description", "")) < 40:
            violations.append((skill_md, 1, "Description is too vague for reliable triggering", metadata.get("description", "")))

    if violations:
        fail_with_violations("skill frontmatter", violations)
    pass_check("skill frontmatter", "all required skills have name/description metadata")


if __name__ == "__main__":
    main()
