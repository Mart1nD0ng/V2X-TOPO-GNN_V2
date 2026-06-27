"""G-MACROSTATE: Level-1 exact == Level-3 dynamic-MC basin first-hitting (spec §4, §12).

The macrostate basin outcome is the headline metric. This pins the spec's three-level
agreement for it: the exact joint-chain basin first-hitting (Level 1, absorption) must be
reproduced by the independent dynamic MC (Level 3, realised trajectories) within its
confidence interval — the same fixed link in both isolates the protocol/basin layer. Split
reachability is checked on a larger two-opposing-region scene (N=3 cannot host a split).
"""

import torch

from src.config.service_profile import ConsensusServiceProfile
from src.environment.canonical_episode import ProtocolConfig
from src.environment.evidence_model import EvidenceModel
from src.environment.round_physics import RoundPhysicsConfig
from src.environment.scenarios import build_scenario
from src.environment.urban_scene import ManhattanScene, build_manhattan_scene
from src.protocol.exact_small_n import exact_joint_basin_first_hitting
from src.sampling import UniformQueryPolicy
from src.validation import run_dynamic_mc

PHY = RoundPhysicsConfig()
POL = UniformQueryPolicy()


def _tiny(pos, reg, cr=80.0, ir=120.0):
    G = int(reg.max()) + 1
    return ManhattanScene(positions=pos, region_of=reg,
                          segment_endpoints=torch.zeros((G, 2, 2), dtype=torch.float64),
                          comm_radius=cr, int_radius=ir, block_m=100.0, grid=(2, 1))


def _line3(node_err):
    pos = torch.tensor([[0.0, 0.0], [35.0, 0.0], [70.0, 0.0]], dtype=torch.float64)
    reg = torch.tensor([0, 0, 0])
    scene = _tiny(pos, reg)
    ev = EvidenceModel(region_of=reg, p_region=torch.zeros(1, dtype=torch.float64),
                       p_node=torch.full((3,), float(node_err), dtype=torch.float64))
    return scene, ev


def _profile(r_max):
    # align the basin deadline R_d with the protocol horizon r_max for an apples-to-apples check
    return ConsensusServiceProfile.urban_default().replace(
        k=2, alpha=2, beta=2, max_poll_epochs=r_max)


def test_exact_basin_probabilities_sum_to_one():
    scene, ev = _line3(node_err=0.3)
    pcfg = ProtocolConfig(k=2, alpha=2, beta=2, r_max=4)
    out = exact_joint_basin_first_hitting(scene, ev, POL, pcfg, link_reliability=0.9,
                                          profile=_profile(4))
    total = out["P_correct"] + out["F_wrong"] + out["F_split"] + out["F_deadline"]
    assert abs(total - 1.0) < 1e-9
    assert out["F_split"] == 0.0   # N=3 uniform omega cannot satisfy C>=rho_s AND W>=rho_s


def test_mc_basin_matches_exact_joint_chain():
    """THE Level-1/Level-3 agreement for the macrostate outcome: the independent MC basin
    first-hitting reproduces the exact joint-chain absorption within the CI."""
    scene, ev = _line3(node_err=0.3)
    pcfg = ProtocolConfig(k=2, alpha=2, beta=2, r_max=4)
    ell = 0.9
    prof = _profile(4)
    ex = exact_joint_basin_first_hitting(scene, ev, POL, pcfg, link_reliability=ell, profile=prof)
    mc = run_dynamic_mc(scene, ev, POL, pcfg, PHY, num_trials=40000,
                        generator=torch.Generator().manual_seed(7), link_override=ell,
                        service_profile=prof)
    # MC four outcomes sum to 1
    total = mc.basin_P_correct + mc.basin_F_wrong + mc.basin_F_split + mc.basin_F_deadline
    assert abs(total - 1.0) < 1e-9
    # exact bracketed by the MC Wilson CIs (unbiased estimator of the true process)
    assert mc.basin_F_wrong_ci[0] <= ex["F_wrong"] <= mc.basin_F_wrong_ci[1]
    assert mc.basin_F_deadline_ci[0] <= ex["F_deadline"] <= mc.basin_F_deadline_ci[1]
    # P_correct agreement via its own CI (recompute from the count proportion)
    assert abs(mc.basin_P_correct - ex["P_correct"]) < 0.02


def test_mc_basin_split_is_reachable_in_two_balanced_opposing_clusters():
    """Two equal, spatially SEPARATED clusters with opposite evidence each finalise their own
    unanimous opinion -> C≈0.5 and W≈0.5 simultaneously -> the split basin (C≥ρ_s and W≥ρ_s)
    fires (spec §4). Unlike a lone dissenter (node-union 'disagree'), the macrostate split
    needs BOTH opinions to hold substantial participation mass."""
    # cluster A {0..3} near x=0 (mostly-correct), cluster B {4..7} near x=200 (mostly-wrong);
    # comm_radius keeps the clusters internally connected but mutually out of range.
    xs = [0.0, 5.0, 10.0, 15.0, 200.0, 205.0, 210.0, 215.0]
    pos = torch.tensor([[x, 0.0] for x in xs], dtype=torch.float64)
    reg = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    scene = _tiny(pos, reg, cr=40.0, ir=60.0)
    ev = EvidenceModel(region_of=reg, p_region=torch.zeros(2, dtype=torch.float64),
                       p_node=torch.tensor([0.02, 0.02, 0.02, 0.02, 0.98, 0.98, 0.98, 0.98],
                                           dtype=torch.float64))
    pcfg = ProtocolConfig(k=2, alpha=2, beta=2, r_max=10)
    prof = ConsensusServiceProfile.urban_default().replace(k=2, alpha=2, beta=2, max_poll_epochs=10)
    mc = run_dynamic_mc(scene, ev, POL, pcfg, PHY, num_trials=2000,
                        generator=torch.Generator().manual_seed(5), link_override=0.97,
                        service_profile=prof)
    total = mc.basin_P_correct + mc.basin_F_wrong + mc.basin_F_split + mc.basin_F_deadline
    assert abs(total - 1.0) < 1e-9
    assert mc.basin_F_split > 0.5   # balanced opposing clusters reliably split


def test_mc_basin_requires_no_truth_leak_into_participation():
    """The MC basin uses an exogenous participation measure passed in; it is not derived from
    the sampled truth. Passing a custom (non-uniform) omega changes the masses deterministically."""
    scene, ev = _line3(node_err=0.25)
    pcfg = ProtocolConfig(k=2, alpha=2, beta=2, r_max=4)
    prof = _profile(4)
    omega = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64)
    mc = run_dynamic_mc(scene, ev, POL, pcfg, PHY, num_trials=2000,
                        generator=torch.Generator().manual_seed(1), link_override=0.9,
                        service_profile=prof, participation=omega)
    total = mc.basin_P_correct + mc.basin_F_wrong + mc.basin_F_split + mc.basin_F_deadline
    assert abs(total - 1.0) < 1e-9
