"""G0 closure -- redesign canonical-path hygiene (ESD spec; constraints #4/#6/#7/#10/#13).

Replaces the OLD project's `test_g10_single_mainline.py` (which asserted the superseded
single-`src/mainline` architecture). The redesign deliberately uses the spec module layout
(`src/{environment,sampling,models,optimization,validation,protocol}`) and REUSES only legacy
*math primitives* (`global_evaluator` padding, `finite_blocklength`, `symmetric_polynomials`,
`quorum_dp`, `topology.receiver_load`) -- the legacy GNN/controls evaluator
(`src/mainline/model.py::evaluate_controls`, tau_proxy/Q=1) is imported by NO redesign module
and is excluded here as frozen-historical (tag `legacy-global-fde-v1`).

This gate is DISCRIMINATIVE on the live redesign surface:
  * #4  no fixed degree cap / candidate truncation (token scan + behavioural dense-graph check);
  * #7  no numeric ``tau_proxy`` (the canonical chain couples tau per round);
  * #6  the comparison / training / oracle layers score via the canonical evaluators
        (`run_consensus_episode` / `run_dynamic_mc`), not a private dynamics re-implementation;
  * #10 the deployable query policies do not read ground truth / peer votes.
Lines bearing the ``# G0-allow`` sentinel are exempt (self-test samples / intentional markers).
"""

from __future__ import annotations

import re
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
CANON_DIRS = ["src/environment", "src/sampling", "src/models", "src/optimization",
              "src/validation", "src/protocol"]
SENTINEL = "G0-allow"

FORBIDDEN = {
    "degree cap / candidate truncation (#4)": (
        r"\.topk\s*\(|\btop_k\b|\btopk\b|degree_cap|max_degree\b|max_?neighbou?rs?\b|"
        r"fixed_fanout|prune_to|sorted\([^)]*\)\s*\[\s*:|\[\s*:\s*(MAX_\w+|cap\w*|fanout)\s*\]"),
    "numeric tau_proxy (#7)": r"tau_proxy\s*[:=]\s*[0-9]",
}


def _canon_files() -> list[Path]:
    out: list[Path] = []
    for d in CANON_DIRS:
        out += [p for p in (ROOT / d).rglob("*.py") if "__pycache__" not in p.parts]
    return sorted(out)


def _scan(pattern: str) -> list[str]:
    rx = re.compile(pattern, re.I)
    hits = []
    for p in _canon_files():
        for i, line in enumerate(p.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            if SENTINEL in line:
                continue
            if rx.search(line):
                hits.append(f"{p.relative_to(ROOT).as_posix()}:{i}")
    return hits


def test_no_forbidden_closures_on_canonical_path():
    hits = {name: _scan(pat) for name, pat in FORBIDDEN.items()}
    assert all(len(v) == 0 for v in hits.values()), hits


def test_behavioural_no_degree_cap():
    """A dense radius cluster -> every node has the FULL neighbourhood (no cap), regardless of
    how a cap would be written (constraint #4)."""
    from src.environment.candidate_graph import build_radius_graph
    N = 30
    pos = torch.zeros(N, 2, dtype=torch.float64)
    pos[:, 0] = torch.linspace(0, 10, N)                  # all mutually within radius 50
    g = build_radius_graph(pos, 50.0)
    deg = torch.bincount(g.src_index, minlength=N)
    assert int(deg.max()) == N - 1 and int(deg.min()) == N - 1


def test_single_canonical_evaluator():
    """The comparison / training / oracle layers must reach a canonical evaluator, not roll their
    own consensus dynamics (constraint #6)."""
    canon = ("run_consensus_episode", "run_dynamic_mc")
    for rel in ("src/optimization/headline.py", "src/optimization/topology_oracle.py",
                "src/optimization/primal_dual.py", "scripts/analysis/scaling_benchmark.py"):
        p = ROOT / rel
        assert p.exists(), rel
        txt = p.read_text(encoding="utf-8", errors="ignore")
        assert any(c in txt for c in canon), f"{rel} does not use a canonical evaluator"


def test_query_policies_do_not_read_truth_or_votes():
    """Constraint #10: the deployable policies' selection logic must not reference ground truth /
    the evidence correctness / peer votes. (The MC reads peers' actual colours to SIMULATE polls,
    but that is the environment, not the policy -- so we scan only the policy modules.)"""
    leak = re.compile(r"ground_truth|\bY_?star\b|init_pref_correct|\.correct\b|peer_colour|"
                      r"evidence\.sample|\.vote\b")
    policy_modules = ["src/sampling/baseline_policies.py", "src/sampling/cdq_query.py",
                      "src/sampling/esp_query.py", "src/sampling/dpp_query.py", "src/models/esd_gnn.py"]
    hits = []
    for rel in policy_modules:
        p = ROOT / rel
        for i, line in enumerate(p.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            if SENTINEL in line:
                continue
            if leak.search(line):
                hits.append(f"{rel}:{i}: {line.strip()[:80]}")
    assert not hits, hits
