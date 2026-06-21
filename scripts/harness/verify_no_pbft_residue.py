from __future__ import annotations

from check_utils import fail_with_violations, find_regex_violations, iter_implementation_files, pass_check, todo_check


FORBIDDEN = (
    (r"\bPBFT_N_PHASES\b", "Do not use PBFT phase-count constants in evaluator code"),
    (r"\bpbft\b", "Consensus target is Avalanche/Snowball, not PBFT"),
    (r"\bn_phases\b", "Do not use phase-count consensus objectives"),
    (r"\bthree[_-]?phase[_-]?consensus\b", "Do not use three-phase consensus targets"),
    (r"\bprepare_phase\b", "Do not add prepare-phase consensus objectives"),
    (r"\bcommit_phase\b", "Do not add commit-phase consensus objectives"),
    (r"\bpbft[_-]?(prepare|commit|phase|consensus)\b", "Do not add PBFT-specific consensus variants"),
)


def main() -> None:
    files = list(iter_implementation_files())
    if not files:
        todo_check("no PBFT residue", "implementation directories are not present yet")
        return
    violations = find_regex_violations(files, FORBIDDEN)
    if violations:
        fail_with_violations("no PBFT residue", violations)
    pass_check("no PBFT residue", f"scanned {len(files)} implementation files")


if __name__ == "__main__":
    main()
