from __future__ import annotations

from pathlib import Path

from check_utils import ROOT, fail_with_violations, pass_check


REQUIRED_SECTIONS = (
    "## Purpose",
    "## Forbidden Shortcuts",
    "## Validation Commands",
    "## Done When",
)

SKILL_REQUIRED_TERMS = {
    "v2x-urban-environment-sim": ("O(N^2)", "feasibility", "O(Nk)"),
    "hierarchical-gnn-topology-constructor": ("same hard-forward", "soft graph", "Bernoulli"),
    "avalanche-closed-form-consensus": ("Monte Carlo", "random sampling", "binomial enumeration", "C_avalanche"),
    "coupled-loss-pcgrad-gradnorm": ("C_avalanche", "D_avalanche", "E_avalanche", "link reliability"),
    "scalability-evaluation-harness": ("10000", "seed", "O(N^2)"),
}


def main() -> None:
    violations: list[tuple[Path, int, str, str]] = []
    skills_dir = ROOT / ".agents" / "skills"

    for skill_dir in sorted(skills_dir.iterdir()):
        if not skill_dir.is_dir():
            continue
        skill_md = skill_dir / "SKILL.md"
        text = skill_md.read_text(encoding="utf-8")
        for section in REQUIRED_SECTIONS:
            if section not in text:
                violations.append((skill_md, 1, f"Missing required section {section}", section))
        for term in SKILL_REQUIRED_TERMS.get(skill_dir.name, ()):
            if term not in text:
                violations.append((skill_md, 1, f"Missing concrete acceptance/guardrail term {term}", term))
        source_map = skill_dir / "references" / "source_map.md"
        if not source_map.exists():
            violations.append((source_map, 1, "Missing per-skill source map reference", "missing"))

    if violations:
        fail_with_violations("skill constraints", violations)
    pass_check("skill constraints", "skills include concrete guardrails and validation commands")


if __name__ == "__main__":
    main()
