from __future__ import annotations

from pathlib import Path

from check_utils import ROOT, fail_with_violations, pass_check


REQUIRED_SOURCES = (
    "3GPP TR 37.885",
    "3GPP TR 38.901",
    "Avalanche/Snowball",
    "GradNorm",
    "PCGrad",
    "PyTorch Geometric",
    "DGL",
    "SUMO",
    "5G-LENA",
)

REQUIRED_ADDITIONAL_TERMS = (
    "Sionna",
    "GraphGPS",
)

REQUIRED_FIELDS = (
    "Title:",
    "Version/release or access date:",
    "Official URL:",
    "Implementation facts extracted:",
    "Skill uses it:",
    "Unresolved uncertainty:",
)


def main() -> None:
    path = ROOT / "docs" / "research" / "source_map.md"
    violations: list[tuple[Path, int, str, str]] = []
    if not path.exists():
        violations.append((path, 1, "Missing research source map", "missing"))
        fail_with_violations("research source map", violations)

    text = path.read_text(encoding="utf-8")
    for source in REQUIRED_SOURCES:
        marker = f"## {source}"
        if marker not in text:
            violations.append((path, 1, f"Missing pinned source section {source}", source))
            continue
        start = text.index(marker)
        next_start = text.find("\n## ", start + len(marker))
        section = text[start: next_start if next_start != -1 else len(text)]
        for field in REQUIRED_FIELDS:
            if field not in section:
                violations.append((path, 1, f"Missing field {field} in {source}", field))

    for term in REQUIRED_ADDITIONAL_TERMS:
        if term not in text:
            violations.append((path, 1, f"Missing source-map term {term}", term))

    if violations:
        fail_with_violations("research source map", violations)
    pass_check("research source map", "required sources are pinned with implementation facts and uncertainties")


if __name__ == "__main__":
    main()
