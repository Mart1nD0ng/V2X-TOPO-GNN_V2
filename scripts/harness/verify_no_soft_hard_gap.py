from __future__ import annotations

from pathlib import Path

from check_utils import ROOT, fail_with_violations, find_regex_violations, iter_implementation_files, pass_check, todo_check

FORBIDDEN = (
    (r"if\s+self\.training\s*:[\s\S]{0,240}soft", "Do not branch forward topology into training-soft behavior"),
    (r"else\s*:[\s\S]{0,240}threshold", "Do not threshold only outside training"),
    (r"if\s+not\s+self\.training\s*:[\s\S]{0,240}(edge_probs\s*>\s*0\.5|threshold)", "Do not threshold topology only outside training"),
    (r"if\s+mode\s*==\s*['\"](eval|validation|test)['\"][\s\S]{0,240}(edge_probs\s*>\s*0\.5|threshold)", "Do not add eval/test/validation-only topology threshold branches"),
    (r"if\s+split\s*==\s*['\"](eval|validation|test)['\"][\s\S]{0,240}(edge_probs\s*>\s*0\.5|threshold)", "Do not add split-specific topology threshold branches"),
    (r"\beval_threshold\b", "Do not introduce validation-only topology thresholds"),
    (r"\bpostprocess_repair\b", "Topology repair must live in shared forward rules"),
    (r"\bvalidation_only_repair\b", "Topology repair must not be validation-only"),
)


def main() -> None:
    docs = ROOT / "docs" / "TOPOLOGY_CONSTRUCTOR.md"
    violations: list[tuple[Path, int, str, str]] = []
    if not docs.exists():
        violations.append((docs, 1, "Missing topology constructor documentation", "missing"))
    else:
        text = docs.read_text(encoding="utf-8")
        if "same hard-forward rule" not in text:
            violations.append((docs, 1, "Documentation must require the same hard-forward rule", "same hard-forward rule"))

    files = list(iter_implementation_files())
    if files:
        violations.extend(find_regex_violations(files, FORBIDDEN))
    else:
        todo_check("no soft/hard topology gap", "constructor code is not implemented yet")

    if violations:
        fail_with_violations("no soft/hard topology gap", violations)
    pass_check("no soft/hard topology gap", "documentation guardrail present")


if __name__ == "__main__":
    main()
