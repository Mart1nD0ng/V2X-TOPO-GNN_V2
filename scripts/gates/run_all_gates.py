"""One-button acceptance-gate runner (spec §8).

Runs every *implemented* gate, prints a G1-G11 status table, and writes a JSON
evidence file to ``docs/gate_evidence/latest.json``.  Gates not yet implemented are
reported as 🔴 (not started) so the table always shows the full picture.

Usage:
    python scripts/gates/run_all_gates.py            # all implemented gates
    python scripts/gates/run_all_gates.py G2 G3      # a subset
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from _common import GateResult, print_result  # type: ignore  # noqa: E402

# Full gate catalogue.  ``module`` is None for gates not yet implemented.
GATE_CATALOGUE: list[tuple[str, str, str | None]] = [
    ("G1", "shared finite-mixture global F (H1/3.3)", "gate_g1"),
    ("G2", "weighted distinct-peer k-subset policy (3.1)", "gate_g2"),
    ("G3", "exact heterogeneous quorum DP (3.2)", "gate_g3"),
    ("G4", "physics-constrained adaptive topology (H2/3.4)", "gate_g4"),
    ("G5", "rigorous finite-blocklength path (H3/3.5)", "gate_g5"),
    ("G6", "independent D/E objectives (3.6)", "gate_g6"),
    ("G7", "preference-conditioned Pareto model (3.7)", "gate_g7"),
    ("G8", "global-risk emission + stop-gradient (3.8)", "gate_g8"),
    ("G9", "near-linear complexity profiling (H4)", "gate_g9"),
    ("G10", "single mathematical mainline (H5)", "gate_g10"),
    ("G11", "baseline comparison win (ultimate)", "gate_g11"),
]


def _module_exists(name: str | None) -> bool:
    return bool(name) and (HERE / f"{name}.py").exists()


def run_selected(selection: list[str] | None) -> list[GateResult]:
    results: list[GateResult] = []
    for gate, title, module in GATE_CATALOGUE:
        if selection and gate not in selection:
            continue
        if not _module_exists(module):
            results.append(GateResult(gate=gate, title=title, passed=False, notes="not implemented"))
            continue
        mod = importlib.import_module(module)
        try:
            result = mod.run()
        except Exception as exc:  # surface, never silently pass
            result = GateResult(gate=gate, title=title, passed=False, notes=f"exception: {exc!r}")
        results.append(result)
    return results


def main(argv: list[str]) -> int:
    selection = [a.upper() for a in argv] or None
    results = run_selected(selection)

    implemented = [r for r in results if r.notes != "not implemented"]
    for r in implemented:
        print_result(r)

    print(f"\n{'=' * 70}\nGATE SUMMARY\n{'=' * 70}")
    for r in results:
        print(f"  {r.status_icon} {r.gate:4s} {r.title}"
              + (f"   [{r.notes}]" if r.notes in {"not implemented"} else ""))

    out_dir = ROOT / "docs" / "gate_evidence"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        r.gate: {"title": r.title, "passed": r.passed, "evidence": r.evidence, "notes": r.notes}
        for r in results
    }
    (out_dir / "latest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nEvidence written to {(out_dir / 'latest.json').relative_to(ROOT).as_posix()}")

    # Exit nonzero if any *implemented & selected* gate failed.
    failed = [r for r in implemented if not r.passed]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
