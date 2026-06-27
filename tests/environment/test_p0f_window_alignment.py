"""G-P0-PHYSICS P0-F: analytic poll-window ell aligns with the independent discrete model.

Two checks (spec §5.2 / §8.3 / §12):
1. SURROGATE VALIDATION (fast): the analytic soft-budget HARQ success
   (`harq_success_at_budget`, the differentiable mean-field used in the canonical ell) matches
   the INDEPENDENT discrete completion-time reference
   (`harq_success_within_window_discrete`, which integrates the floor attempt-fitting over the
   random M/M/1 sojourn) in the saturated/empty limits and stays close + monotone between — the
   honest surrogate gap.
2. CANONICAL-PATH EFFECT (full physics MC): a smaller poll window Delta_poll raises the
   macrostate deadline-miss probability in the independent dynamic MC headline — the window
   timeout genuinely flows through the SAME canonical physics into the MC judge's basin outcome.
"""

import math

import torch

from src.config.service_profile import ConsensusServiceProfile
from src.environment.candidate_graph import build_candidate_graph
from src.environment.evidence_model import EvidenceModel
from src.environment.interference_graph import build_interference_graph
from src.environment.round_physics import (
    RoundPhysicsConfig,
    harq_success_at_budget,
    harq_success_within_window_discrete,
)
from src.environment.urban_scene import ManhattanScene
from src.environment.canonical_episode import ProtocolConfig
from src.sampling import UniformQueryPolicy
from src.validation import run_dynamic_mc

RT_SLOTS = 2.0           # request_slots + response_slots
SLOT = 1e-3
RT_TIME = RT_SLOTS * SLOT


def _succ_by_m(p1=0.6, M=3):
    # chase-combining: decode-by-m is increasing and concave-ish; use a saturating sequence
    vals = []
    miss = 1.0
    for _ in range(M):
        miss *= (1.0 - p1)        # independent-attempt lower bound (monotone, in [0,1])
        vals.append(torch.tensor([[1.0 - miss]], dtype=torch.float64))
    return vals


def test_discrete_reference_limits():
    sbm = _succ_by_m(p1=0.6, M=3)
    qd = torch.zeros((1, 1), dtype=torch.float64)
    # huge window -> saturates to succ_M
    big = harq_success_within_window_discrete(sbm, qd, RT_TIME, poll_window_s=10.0)
    assert torch.allclose(big, sbm[-1], atol=1e-9)
    # window below one round-trip -> ~0 (floor budget 0)
    tiny = harq_success_within_window_discrete(sbm, qd, RT_TIME, poll_window_s=RT_TIME * 0.5)
    assert float(tiny) < 1e-9


def test_analytic_soft_matches_discrete_in_operating_regime():
    """In the OPERATING regime (budget >= 1 complete round-trip) the differentiable soft
    surrogate tracks the independent discrete model closely; at the saturated/empty ends they
    agree exactly. Below one round-trip the soft surrogate is a documented OPTIMISTIC upper
    bound (any smooth monotone relaxation gives partial credit where the discrete floor — which
    needs a full request+response round-trip — gives 0); the MC/discrete model is the judge."""
    sbm = _succ_by_m(p1=0.5, M=3)
    qd = torch.zeros((1, 1), dtype=torch.float64)
    # operating regime: budget >= 1
    for w in (RT_TIME * 1.0, RT_TIME * 1.5, RT_TIME * 2.0, RT_TIME * 3.0, RT_TIME * 10.0):
        m_win = torch.tensor([[(w / SLOT) / RT_SLOTS]], dtype=torch.float64).clamp(0.0, 3.0)
        soft = harq_success_at_budget(sbm, m_win)
        disc = harq_success_within_window_discrete(sbm, qd, RT_TIME, poll_window_s=w)
        # worst case is half the largest HARQ step at an interval midpoint (here 0.125)
        assert abs(float(soft) - float(disc)) < 0.15
    # exact agreement at the saturated end
    soft_sat = harq_success_at_budget(sbm, torch.tensor([[3.0]], dtype=torch.float64))
    disc_sat = harq_success_within_window_discrete(sbm, qd, RT_TIME, poll_window_s=100.0)
    assert abs(float(soft_sat) - float(disc_sat)) < 1e-9
    # sub-unit regime: soft is an optimistic upper bound over the discrete floor (documented gap)
    for w in (RT_TIME * 0.3, RT_TIME * 0.5, RT_TIME * 0.8):
        m_win = torch.tensor([[(w / SLOT) / RT_SLOTS]], dtype=torch.float64).clamp(0.0, 3.0)
        soft = float(harq_success_at_budget(sbm, m_win))
        disc = float(harq_success_within_window_discrete(sbm, qd, RT_TIME, poll_window_s=w))
        assert soft >= disc - 1e-12 and disc < 1e-9   # discrete floor = 0 below one round-trip


def test_discrete_reference_monotone_in_window_and_queue():
    sbm = _succ_by_m(p1=0.5, M=3)
    qd0 = torch.zeros((1, 1), dtype=torch.float64)
    prev = -1.0
    for w in (RT_TIME * 0.5, RT_TIME, RT_TIME * 2, RT_TIME * 5):
        v = float(harq_success_within_window_discrete(sbm, qd0, RT_TIME, poll_window_s=w))
        assert v >= prev - 1e-12
        prev = v
    # a longer queue wait can only lower the within-window success
    w = RT_TIME * 2.5
    lo_q = harq_success_within_window_discrete(sbm, torch.tensor([[1e-9]], dtype=torch.float64), RT_TIME, poll_window_s=w)
    hi_q = harq_success_within_window_discrete(sbm, torch.tensor([[RT_TIME]], dtype=torch.float64), RT_TIME, poll_window_s=w)
    assert float(hi_q) <= float(lo_q) + 1e-12


def _tiny_scene(node_err=0.05):
    pos = torch.tensor([[0.0, 0.0], [25.0, 0.0], [50.0, 0.0], [75.0, 0.0], [100.0, 0.0]],
                       dtype=torch.float64)
    reg = torch.tensor([0, 0, 0, 0, 0])
    scene = ManhattanScene(positions=pos, region_of=reg,
                           segment_endpoints=torch.zeros((1, 2, 2), dtype=torch.float64),
                           comm_radius=60.0, int_radius=90.0, block_m=100.0, grid=(2, 1))
    ev = EvidenceModel(region_of=reg, p_region=torch.zeros(1, dtype=torch.float64),
                       p_node=torch.full((5,), float(node_err), dtype=torch.float64))
    return scene, ev


def test_smaller_window_raises_mc_deadline_miss():
    """The poll-window timeout flows through the SAME canonical round physics into the
    independent dynamic-MC macrostate headline: shrinking Delta_poll strictly increases the
    deadline-miss probability (fewer polls complete in time -> slower/failed finalisation)."""
    scene, ev = _tiny_scene(node_err=0.05)
    pcfg = ProtocolConfig(k=2, alpha=2, beta=2, r_max=8)
    prof = ConsensusServiceProfile.urban_default().replace(k=2, alpha=2, beta=2, max_poll_epochs=8)
    POL = UniformQueryPolicy()
    # large window: polls complete -> low deadline; small window (< one round-trip): polls time out
    big = RoundPhysicsConfig(poll_window_s=0.05)
    small = RoundPhysicsConfig(poll_window_s=1.2e-3)   # < rt_time = 2ms -> budget < 1
    mc_big = run_dynamic_mc(scene, ev, POL, pcfg, big, num_trials=800,
                            generator=torch.Generator().manual_seed(2), service_profile=prof)
    mc_small = run_dynamic_mc(scene, ev, POL, pcfg, small, num_trials=800,
                              generator=torch.Generator().manual_seed(2), service_profile=prof)
    assert mc_small.basin_F_deadline > mc_big.basin_F_deadline + 0.1
    # and the four outcomes still partition probability
    for mc in (mc_big, mc_small):
        total = mc.basin_P_correct + mc.basin_F_wrong + mc.basin_F_split + mc.basin_F_deadline
        assert abs(total - 1.0) < 1e-9
