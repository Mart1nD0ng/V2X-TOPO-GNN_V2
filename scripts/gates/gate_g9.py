"""G9 (spec H4 / §11.8): end-to-end near-linear complexity profiling -- DISCRIMINATIVE.

Acceptance: the FULL Eq. 56 forward (graph build -> GNN -> pi (G2) -> ell (G4xG5) ->
consensus recurrence (G1) -> D/E objectives (G6)) is APPROXIMATELY LINEAR in ``N`` and
``E`` -- no super-linear (quadratic) blow-up.  Density is held fixed (area ∝ N) so that
``E = O(N)`` is the honest regime.

Per anti-degradation §7 the gate must be DISCRIMINATIVE: a genuinely quadratic implementation
(in time OR memory) must FAIL.  The earlier version was not -- the ~2 s fixed per-call
overhead flattened the full-range log-log slope and a stage-isolated quadratic was diluted by
profiling only ``t_total``.  This version uses checks that survive that overhead:

  * NO N x N TENSOR (deterministic): a TorchDispatch hook records the largest single tensor
    materialised by the real forward; its scaling exponent vs E must be < 1.4.  The honest
    mainline's largest tensor is the quorum-DP cube ``[m*Q,(k+1)^3] = O(E k^3)`` (constant
    65*E), so the exponent is exactly 1.0; ANY ``N x N`` (cdist, broadcast, matmul, einsum,
    tuple-zeros) scales as ``N^2`` -> exponent ~2.  This catches idioms the static grep misses.
  * NO TIME-QUADRATIC (overhead-immune): ``fit_linear_vs_quadratic`` is applied PER STAGE
    (build/gnn/consensus/total); each stage's quadratic contribution at ``E_max`` must be
    < 0.5 of its time.  The quadratic-coefficient fit is immune to the additive overhead that
    flattens the log-log slope, and per-stage isolation defeats the t_total dilution.
  * a per-stage TOP-OF-RANGE (last-3) exponent < 1.6 (a local slope the small-N overhead
    cannot flatten), ``E ~ N`` exponent in [0.85,1.25], a broadened no-N x N grep, and the
    bucket ``total_cells <= 2E`` (D4) -- the inclusion-probability head is now also bucketed,
    so the whole profiled path is ``O(E)`` with no dense ``[N, max_deg]`` allocation.

A figure is written to ``docs/gate_evidence/g9_scaling.png``.  ``tests/test_g9_scaling.py``
includes a discriminativeness regression test that injects a real N x N op and asserts FAIL.
"""

from __future__ import annotations

import sys
from pathlib import Path

from _common import GateResult, grep_repo, main_single, run_pytest  # type: ignore

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from profile_scaling import (  # noqa: E402
    ROOT, fit_exponent, fit_linear_vs_quadratic, local_exponent, make_figure,
    measure_peak_numels, profile_scaling,
)

NS = [200, 400, 800, 1600, 3200, 6400]
STAGES = ("build", "gnn", "consensus", "total")
NXN_PATTERN = (r"cdist|\(\(?\s*(N|num_nodes)\s*,\s*(N|num_nodes)\b|"
               r"unsqueeze\(0\)[^\n]*unsqueeze\(1\)|unsqueeze\(1\)[^\n]*unsqueeze\(0\)|"
               r"torch\.outer|einsum|@\s*\w+\.[tT]\b")


def run() -> GateResult:
    evidence: dict = {}
    res = profile_scaling(NS, reps=3)

    e_exp = fit_exponent(res["N"], res["E"])
    stage_exp = {st: fit_exponent(res["E"], res[f"t_{st}"]) for st in STAGES}
    stage_quad = {st: fit_linear_vs_quadratic(res["E"], res[f"t_{st}"])["quad_contrib_ratio"] for st in STAGES}
    stage_local = {st: local_exponent(res["E"], res[f"t_{st}"]) for st in STAGES}
    cells_ok = all(res["total_cells"][i] <= 2 * res["E"][i] for i in range(len(res["N"])))
    max_cell_ratio = max(res["total_cells"][i] / res["E"][i] for i in range(len(res["N"])))

    # deterministic N x N detector: largest materialised tensor must scale ~linearly in E
    nm = measure_peak_numels(NS[-3:])  # 1600, 3200, 6400
    numel_exp = fit_exponent(nm["E"], nm["max_numel"])
    max_numel_ratio = max(m / e for m, e in zip(nm["max_numel"], nm["E"]))

    nxn_hits = grep_repo(NXN_PATTERN, globs=("src/mainline/*.py",))

    evidence["N range / E range"] = f"{res['N'][0]}-{res['N'][-1]} / {res['E'][0]}-{res['E'][-1]}"
    evidence["E ~ N exponent"] = f"{e_exp:.3f}"
    evidence["full-range t~E exp (build/gnn/cons/total)"] = " / ".join(f"{stage_exp[s]:.2f}" for s in STAGES)
    evidence["top-of-range t~E exp (build/gnn/cons/total)"] = " / ".join(f"{stage_local[s]:.2f}" for s in STAGES)
    evidence["per-stage quad-contrib@Emax (build/gnn/cons/total)"] = " / ".join(f"{stage_quad[s]:.2f}" for s in STAGES)
    evidence["max-tensor numel ~E exp / ratio@Emax (no N×N)"] = f"{numel_exp:.3f} / {max_numel_ratio:.0f}xE"
    evidence["bucket total_cells <= 2E (max ratio)"] = f"{cells_ok} ({max_cell_ratio:.2f})"
    evidence["no N×N static grep (mainline hits)"] = f"{len(nxn_hits)}"

    fig_path = make_figure(res, ROOT / "docs" / "gate_evidence" / "g9_scaling.png")
    fig_ok = fig_path.exists() and fig_path.stat().st_size > 0
    evidence["figure"] = f"{fig_path.relative_to(ROOT).as_posix()} ({fig_path.stat().st_size} B)" if fig_ok else "MISSING"

    # ---- discriminative verdicts (each survives the fixed-overhead regime) ----
    edges_linear = 0.85 <= e_exp <= 1.25
    numel_linear = numel_exp < 1.40                                  # deterministic no-N×N
    quad_not_needed = all(stage_quad[s] < 0.5 for s in STAGES)       # overhead-immune, per-stage
    local_not_superlinear = max(stage_local.values()) < 1.60         # top-of-range slope
    mem_ok = cells_ok and len(nxn_hits) == 0

    tests_ok, tail = run_pytest("tests/test_g9_scaling.py")
    evidence["pytest"] = "passed" if tests_ok else f"FAILED\n{tail}"

    passed = bool(edges_linear and numel_linear and quad_not_needed and local_not_superlinear
                  and mem_ok and fig_ok and tests_ok)
    return GateResult(
        gate="G9",
        title="near-linear complexity profiling (H4): end-to-end runtime/memory ~ linear in N, E",
        passed=passed,
        evidence=evidence,
        notes="Density fixed (area ∝ N) so E=O(N). Discriminative against quadratics: a deterministic "
              "max-tensor-numel scaling guard (no N×N, any idiom), per-stage quadratic-coefficient fits "
              "(overhead-immune), and top-of-range slopes. Honest mainline: largest tensor is the O(E k^3) "
              "quorum cube (numel~E exp 1.0), pi head now bucketed (O(E), no [N,max_deg]); all stages <= linear.",
    )


if __name__ == "__main__":
    sys.exit(main_single(run))
