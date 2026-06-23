from __future__ import annotations

from pathlib import Path

from check_utils import ROOT, fail_with_violations, pass_check


REQUIRED_TERMS = (
    "Avalanche/Snowball",
    "not PBFT",
    "Monte Carlo",
    "random sampling",
    "binomial enumeration",
    "training-soft / validation-hard",
    "shared topology layer",
    "Link reliability",
    "SINR",
    "BLER",
    "HARQ",
    "coverage",
    "O(Nk)",
    "O(N^2)",
    "Environment feasibility",
    "before GNN training",
)


def main() -> None:
    path = ROOT / "docs" / "model_contract.md"
    violations: list[tuple[Path, int, str, str]] = []
    if not path.exists():
        violations.append((path, 1, "Missing model contract", "missing"))
        fail_with_violations("model contract", violations)

    text = path.read_text(encoding="utf-8")
    for term in REQUIRED_TERMS:
        if term not in text:
            violations.append((path, 1, f"Missing model-contract term {term}", term))

    config = ROOT / ".codex" / "config.toml"
    if not config.exists():
        violations.append((config, 1, "Missing Codex project config", "missing"))
    else:
        config_text = config.read_text(encoding="utf-8")
        for required in (
            'approval_policy = "on-request"',
            'sandbox_mode = "workspace-write"',
            "project_doc_max_bytes = 65536",
        ):
            if required not in config_text:
                violations.append((config, 1, f"Missing conservative config value {required}", required))

    if violations:
        fail_with_violations("model contract", violations)
    pass_check("model contract", "contract and Codex defaults are present")


if __name__ == "__main__":
    main()
