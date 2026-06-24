"""G8 -- perfect-link protocol feasibility floors (spec §3.3, the viability gate).

Acceptance: the well-mixed recursion is monotone and matches the independent dynamic MC in a
complete-graph well-mixed setting (grounding the floor model); network floors are correct; the
scan finds feasible params for a solid correct majority at the target N and correctly rejects
the unsolvable 50/50 case (the gate has teeth).
"""

import torch

from src.environment.canonical_episode import ProtocolConfig
from src.environment.evidence_model import EvidenceModel
from src.environment.round_physics import RoundPhysicsConfig
from src.environment.urban_scene import ManhattanScene
from src.sampling import UniformQueryPolicy
from src.validation import run_dynamic_mc
from src.validation.feasibility import (
    network_floors,
    scan_feasibility,
    wellmixed_terminal,
)


def test_wellmixed_monotone_in_beta_and_init():
    # higher beta -> lower per-node wrong probability (exponential safety)
    ws = [wellmixed_terminal(0.8, 5, 3, b, 4 * b)[1] for b in (2, 3, 4, 6)]
    assert all(ws[i + 1] <= ws[i] + 1e-12 for i in range(len(ws) - 1))
    # higher correct majority -> lower wrong, higher correct
    c_lo, w_lo, _ = wellmixed_terminal(0.6, 5, 3, 4, 16)
    c_hi, w_hi, _ = wellmixed_terminal(0.9, 5, 3, 4, 16)
    assert c_hi >= c_lo and w_hi <= w_lo


def test_network_floors_formulas():
    f = network_floors(c=0.999999, w=1e-6, undec=0.0, N=1000)
    # F_wrong = 1-(1-1e-6)^1000 ~ 1e-3
    assert abs(f.F_wrong - (1 - (1 - 1e-6) ** 1000)) < 1e-9
    assert 9.9e-4 < f.F_wrong < 1.01e-3
    # F_deadline = 1 - c^N
    assert abs(f.F_deadline - (1 - 0.999999 ** 1000)) < 1e-9
    assert 0.0 <= f.F_disagree <= 1.0


def test_scan_feasible_for_solid_majority_at_target_N():
    """The viability gate must PASS for a solid correct majority at the target scale."""
    for N in (100, 1000, 10000):
        r = scan_feasibility(0.8, N)
        assert r["feasible"], f"infeasible at N={N} for init=0.8 (would block training)"
        p = r["feasible_params"][0]
        assert p["F_wrong"] <= 1e-3 / 10 and p["F_disagree"] <= 1e-4 / 10 and p["F_deadline"] <= 1e-2 / 10
        assert 2 * p["alpha"] > p["k"]                # strict majority


def test_scan_infeasible_for_tie_has_teeth():
    """A 50/50 split cannot be resolved to the correct side -> the gate must reject it."""
    r = scan_feasibility(0.5, 1000)
    assert not r["feasible"]


def _complete_scene(N, seed, box=30.0):
    g = torch.Generator().manual_seed(seed)
    pos = torch.rand(N, 2, generator=g, dtype=torch.float64) * box
    reg = torch.zeros(N, dtype=torch.long)
    return ManhattanScene(positions=pos, region_of=reg,
                          segment_endpoints=torch.zeros((1, 2, 2), dtype=torch.float64),
                          comm_radius=500.0, int_radius=500.0, block_m=100.0, grid=(1, 1))


def test_wellmixed_recursion_matches_dynamic_mc():
    """Ground the floor model: the well-mixed recursion matches the INDEPENDENT dynamic MC in a
    complete-graph perfect-link setting (representative sampling), at a measurable failure rate."""
    N, k, alpha, beta, r_max = 120, 3, 2, 2, 8
    init_correct = 0.62
    scene = _complete_scene(N, seed=0)
    # p_node = 1 - init_correct so q_i = init_correct (well-mixed, no region bias)
    ev = EvidenceModel(region_of=scene.region_of, p_region=torch.zeros(1, dtype=torch.float64),
                       p_node=torch.full((N,), 1 - init_correct, dtype=torch.float64))
    c_rec, w_rec, u_rec = wellmixed_terminal(init_correct, k, alpha, beta, r_max)
    mc = run_dynamic_mc(scene, ev, UniformQueryPolicy(), ProtocolConfig(k, alpha, beta, r_max),
                        RoundPhysicsConfig(), num_trials=6000,
                        generator=torch.Generator().manual_seed(3), link_override=1.0)
    c_mc = float(mc.decided_correct_freq.mean())
    w_mc = float(mc.decided_wrong_freq.mean())
    # finite-N (with/without replacement) bias ~ k/N ~ 2.5%; require close agreement
    assert abs(c_mc - c_rec) < 0.03, f"c: mc={c_mc:.4f} rec={c_rec:.4f}"
    assert abs(w_mc - w_rec) < 0.03, f"w: mc={w_mc:.4f} rec={w_rec:.4f}"
    # both should see a measurable but minority wrong-rate (sanity that the test is discriminating)
    assert 0.01 < w_rec < 0.4
