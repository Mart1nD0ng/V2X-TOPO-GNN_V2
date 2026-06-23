"""Honesty harness (P0-1): forbid undisclosed mean-field (Q=1) reliability conclusions.

The SSMC mean-field closed form (quenched_quadrature=1) is 22-40x optimistic and gradient-blind to
the query-spread lever (docs/CLOSED_FORM_FIDELITY.md). It must NOT be cited as a reliability
*conclusion* without disclosing the currency. The scanner FAILS a file that cites a reliability F as
a conclusion (remediated_F_final, "F = 0.00x", reliability_target, ...) but never discloses the
evaluator currency (mean-field / quenched / Monte-Carlo / Q=<n> / evaluator_currency / surrogate /
an explicit ``<!-- mean-field-disclosed -->`` opt-out marker).

ENFORCED SURFACE (be honest about the limits — audit defect fix):
  * In the real repo (a .git root) the gate scans GIT-TRACKED docs/**/*.md and README.md ONLY.
    result/ is gitignored, hence NEVER scanned by the gate — the ``evaluator_currency`` tokens the
    result generators emit are self-documentation, NOT gate-enforced protection. To audit the local
    result/ tree manually, run with ``--include-results`` (scans untracked result/**/*.md|json too;
    legacy pre-convention run dirs will be flagged — that is the point of a manual audit).
  * In a non-git root (the contract-test tempdir) every matching file is scanned.
  * This is a KEYWORD-PRESENCE gate, not a value-level check: it verifies a currency token exists in
    the file, NOT that the cited number was produced in the disclosed currency. It deters undisclosed
    claims; it cannot validate them.

This is the documentation-side analogue of the numerical honesty checks (verify_no_link_reliability_loss,
the MC-vs-exact reference). It is wired into invariants-check (-> harness-check -> agent-check).
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from check_utils import ROOT, fail_with_violations, pass_check, read_text, todo_check

# A reliability F cited as a RESULT/conclusion (not a generic mention of the letter F).
_CONCLUSION_PATTERNS = (
    r"remediated_F\w*",
    r"\bF_final\b",
    r"\bF_dec(?:rease)?\b",
    r"\bheadline[\s_]+F\b",
    r"\bF\s*[=:≈]\s*0?\.\d",          # "F = 0.0043", "F: 0.01", "F ≈ 0.02"
    r"\bF_avalanche\w*",
    r"\breliability_target\b",         # the result-JSON / config key form (not prose "reliability target")
    r"\bachieves?\s+F\b",
)
# Disclosure of the evaluator currency anywhere in the file -> the F citation is honest.
_DISCLOSURE_PATTERNS = (
    r"mean[\s_-]?field",
    r"quenched",
    r"monte[\s_-]?carlo",
    r"\bMC\b",
    r"surrogate",
    r"evaluator_currency",
    r"quenched_quadrature",
    r"\bQ\s*=\s*\d",
    r"optimistic",
    r"<!--\s*mean-field-disclosed\s*-->",
)

_CONCLUSION_RE = re.compile("|".join(_CONCLUSION_PATTERNS), re.IGNORECASE)
_DISCLOSURE_RE = re.compile("|".join(_DISCLOSURE_PATTERNS), re.IGNORECASE)
_SKIP_DIRS = {".git", ".venv", "node_modules", "__pycache__", ".agent", ".agents"}


def _candidate_files() -> list[Path]:
    seen: set[Path] = set()
    files: list[Path] = []
    for pattern in ("docs/**/*.md", "**/README.md", "result/**/*.md", "result/**/*.json"):
        for path in ROOT.glob(pattern):
            if not path.is_file():
                continue
            if any(part in _SKIP_DIRS for part in path.relative_to(ROOT).parts):
                continue
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                files.append(path)
    return files


def _tracked_subset(files: list[Path]) -> list[Path] | None:
    """When ROOT is a git repo, keep only tracked files (result/ is gitignored -> excluded), so the
    gate enforces on the committed surface and ignores transient run dirs. None if git is unavailable."""
    if not (ROOT / ".git").exists():
        return None
    try:
        out = subprocess.run(
            ["git", "ls-files"], cwd=ROOT, text=True, capture_output=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    tracked = {(ROOT / line).resolve() for line in out.splitlines() if line.strip()}
    return [p for p in files if p.resolve() in tracked]


def main() -> None:
    include_results = "--include-results" in sys.argv[1:]
    files = _candidate_files()
    tracked = _tracked_subset(files)
    if tracked is not None:
        if include_results:
            # manual-audit mode: tracked surface PLUS the (untracked, gitignored) result/ tree.
            result_files = [p for p in files if p.relative_to(ROOT).parts[:1] == ("result",)]
            seen = {p.resolve() for p in tracked}
            files = tracked + [p for p in result_files if p.resolve() not in seen]
        else:
            files = tracked
    if not files:
        todo_check("no mean-field reliability claim", "no docs/result reliability surface to scan yet")
        return
    violations: list[tuple[Path, int, str, str]] = []
    for path in files:
        text = read_text(path)
        if _DISCLOSURE_RE.search(text):
            continue  # currency disclosed somewhere in the file -> honest
        match = _CONCLUSION_RE.search(text)
        if match is None:
            continue  # no reliability conclusion -> nothing to disclose
        line = text.count("\n", 0, match.start()) + 1
        violations.append(
            (
                path,
                line,
                "Reliability F cited as a conclusion without disclosing the evaluator currency "
                "(mean-field Q=1 is 22-40x optimistic; see docs/CLOSED_FORM_FIDELITY.md). State "
                "mean-field/quenched/MC or add a Q=<n> / evaluator_currency token.",
                match.group(0),
            )
        )
    if violations:
        fail_with_violations("no mean-field reliability claim", violations)
    pass_check("no mean-field reliability claim", f"scanned {len(files)} doc/result files")


if __name__ == "__main__":
    main()
