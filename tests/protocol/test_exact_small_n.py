"""G1+G6 -- small-N exact joint chain & three-way agreement (spec §8.1 lvl 2, §8).

The exact joint chain is the ground truth. The decisive check: the independent dynamic MC
must reproduce it within its confidence interval (proving the MC is an unbiased estimator of
the TRUE joint process -- the property that makes it a valid judge). The analytic mean-field
may differ at tiny strongly-coupled N (it is exact only in the weak-coupling/large-N limit);
all three are run with the SAME fixed link so the comparison isolates the protocol layer.
"""

import torch

from src.environment.canonical_episode import ProtocolConfig, run_consensus_episode
from src.environment.evidence_model import EvidenceModel
from src.environment.round_physics import RoundPhysicsConfig
from src.environment.urban_scene import ManhattanScene
from src.protocol.exact_small_n import exact_joint_terminal
from src.sampling import UniformQueryPolicy
from src.validation import run_dynamic_mc

PHY = RoundPhysicsConfig()
POL = UniformQueryPolicy()


def _tiny(pos, reg, cr=80.0, ir=120.0):
    G = int(reg.max()) + 1
    return ManhattanScene(positions=pos, region_of=reg,
                          segment_endpoints=torch.zeros((G, 2, 2), dtype=torch.float64),
                          comm_radius=cr, int_radius=ir, block_m=100.0, grid=(2, 1))


def _line3(node_err=0.2):
    pos = torch.tensor([[0.0, 0.0], [35.0, 0.0], [70.0, 0.0]], dtype=torch.float64)
    reg = torch.tensor([0, 0, 0])
    scene = _tiny(pos, reg)
    ev = EvidenceModel(region_of=reg, p_region=torch.zeros(1, dtype=torch.float64),
                       p_node=torch.full((3,), float(node_err), dtype=torch.float64))
    return scene, ev


def test_exact_perfect_allcorrect_is_unity():
    pos = torch.tensor([[0.0, 0.0], [40.0, 0.0]], dtype=torch.float64)
    reg = torch.tensor([0, 0])
    scene = _tiny(pos, reg, cr=60.0, ir=90.0)
    ev = EvidenceModel(region_of=reg, p_region=torch.zeros(1, dtype=torch.float64),
                       p_node=torch.zeros(2, dtype=torch.float64))
    r = exact_joint_terminal(scene, ev, POL, ProtocolConfig(k=1, alpha=1, beta=2, r_max=4),
                             link_reliability=1.0)
    assert abs(r.S_allcorrect - 1.0) < 1e-12
    assert r.F_wrong < 1e-12 and r.F_disagree < 1e-12


def test_exact_probabilities_valid_and_monotone_in_link():
    scene, ev = _line3()
    pcfg = ProtocolConfig(k=2, alpha=2, beta=2, r_max=4)
    prev = -1.0
    for ell in (0.6, 0.8, 1.0):
        r = exact_joint_terminal(scene, ev, POL, pcfg, link_reliability=ell)
        for v in (r.S_allcorrect, r.F_wrong, r.F_disagree):
            assert -1e-12 <= v <= 1.0 + 1e-12
        assert r.S_allcorrect >= prev - 1e-9      # better links -> more all-correct
        prev = r.S_allcorrect


def test_dynamic_mc_matches_exact_joint_chain():
    """THE three-way core: the independent MC reproduces the exact joint terminal within its
    CI -> the MC is an unbiased estimator of the true process (spec §8, validates the judge)."""
    scene, ev = _line3(node_err=0.2)
    pcfg = ProtocolConfig(k=2, alpha=2, beta=2, r_max=4)
    ell = 0.9
    ex = exact_joint_terminal(scene, ev, POL, pcfg, link_reliability=ell)
    mc = run_dynamic_mc(scene, ev, POL, pcfg, PHY, num_trials=40000,
                        generator=torch.Generator().manual_seed(11), link_override=ell)
    assert mc.S_allcorrect_ci[0] <= ex.S_allcorrect <= mc.S_allcorrect_ci[1]
    assert mc.F_wrong_ci[0] <= ex.F_wrong <= mc.F_wrong_ci[1]


def test_analytic_meanfield_gap_documented_but_direction_agrees():
    """The analytic mean-field differs from the exact at tiny strongly-coupled N (expected),
    but must agree on DIRECTION: better links -> more all-correct in BOTH exact and analytic."""
    scene, ev = _line3(node_err=0.2)
    pcfg = ProtocolConfig(k=2, alpha=2, beta=2, r_max=4)
    ex_lo = exact_joint_terminal(scene, ev, POL, pcfg, link_reliability=0.7)
    ex_hi = exact_joint_terminal(scene, ev, POL, pcfg, link_reliability=0.98)
    an_lo = run_consensus_episode(scene, ev, POL, pcfg, PHY, return_trajectory=False, link_override=0.7)
    an_hi = run_consensus_episode(scene, ev, POL, pcfg, PHY, return_trajectory=False, link_override=0.98)
    # direction agrees
    assert ex_hi.S_allcorrect > ex_lo.S_allcorrect
    assert float(an_hi.S_allcorrect) > float(an_lo.S_allcorrect)
    # there IS a real mean-field gap at this tiny coupled N (sanity that we're not trivially equal)
    assert abs(float(an_hi.S_allcorrect) - ex_hi.S_allcorrect) > 0.02


def test_link_override_trace_marks_non_headline():
    scene, ev = _line3()
    an = run_consensus_episode(scene, ev, POL, ProtocolConfig(k=2, alpha=2, beta=2, r_max=4),
                               PHY, return_trajectory=False, link_override=0.9)
    assert an.mechanism_trace["link_override"] == 0.9
    assert an.mechanism_trace["full_physics"] is False
    # headline (no override) is full physics
    an2 = run_consensus_episode(scene, ev, POL, ProtocolConfig(k=2, alpha=2, beta=2, r_max=4),
                                PHY, return_trajectory=False)
    assert an2.mechanism_trace["link_override"] is None
    assert an2.mechanism_trace["full_physics"] is True
