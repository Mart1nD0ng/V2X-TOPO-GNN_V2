"""G7 -- effective-sampling-dynamics diagnostics (spec §5).

Acceptance (plan Phase 6): the hand-scenario directions all hold -- symmetric link loss lowers
PROGRESS (not drift); opinion-correlated evidence moves DRIFT; redundant (correlated) peers
lower the effective sample size; a weak cut lowers cross-region MIXING; a hub raises receiver
LOAD.
"""

import torch

from src.mainline.quorum_dp import quorum_decision_probabilities
from src.sampling.effective_dynamics import (
    cross_region_response_mass,
    effective_sample_size,
    progress_drift,
    receiver_load,
    region_response_kernel,
    region_spectral_gap,
    response_conditioned_marginal,
)
from src.environment.evidence_model import EvidenceModel


# ----------------------------------------------------- response-conditioned pi~
def test_response_conditioned_marginal_normalises_and_thins():
    # one source (0) with 3 candidates; equal pi, but ell favours edge 2
    src = torch.tensor([0, 0, 0])
    dst = torch.tensor([1, 2, 3])
    pi = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
    ell = torch.tensor([0.2, 0.2, 0.9], dtype=torch.float64)
    pit = response_conditioned_marginal(src, 4, pi, ell)
    assert abs(float(pit.sum()) - 1.0) < 1e-12         # per-source normalised
    assert pit[2] > pit[0] and pit[2] > pit[1]          # mass shifts to the high-delivery peer


# --------------------------------------------------------- progress / drift
def _quorum(ell, u, v, k=3, alpha=2):
    log_w = torch.zeros(1, k, dtype=torch.float64)
    p_plus = torch.full((1, k), ell * u, dtype=torch.float64)
    p_minus = torch.full((1, k), ell * v, dtype=torch.float64)
    dec = quorum_decision_probabilities(log_w, p_plus, p_minus, k, alpha)
    return dec.h_plus, dec.h_minus


def test_symmetric_loss_lowers_progress_not_drift():
    tau = torch.ones(1, dtype=torch.float64)
    hp_hi, hm_hi = _quorum(0.95, 0.5, 0.5)
    hp_lo, hm_lo = _quorum(0.40, 0.5, 0.5)
    g_hi = progress_drift(hp_hi, hm_hi, tau)
    g_lo = progress_drift(hp_lo, hm_lo, tau)
    assert float(g_lo["progress"]) < float(g_hi["progress"])     # less delivery -> less progress
    assert abs(float(g_hi["drift"])) < 1e-9 and abs(float(g_lo["drift"])) < 1e-9  # symmetric -> no drift


def test_opinion_split_moves_drift():
    tau = torch.ones(1, dtype=torch.float64)
    hp, hm = _quorum(0.9, 0.85, 0.15)                  # peers lean correct
    d = progress_drift(hp, hm, tau)
    assert float(d["drift"]) > 0.05                     # positive (correct-direction) drift
    hp2, hm2 = _quorum(0.9, 0.15, 0.85)                # peers lean wrong
    assert float(progress_drift(hp2, hm2, tau)["drift"]) < -0.05


# --------------------------------------------------------- effective sample size
def test_redundant_peers_lower_effective_sample_size():
    # node 0 polls 3 peers all in region 0 (shared bias -> correlated); node 4 polls 3 peers in
    # three different clean regions (independent). region bias makes region-0 peers correlated.
    region_of = torch.tensor([0, 0, 0, 0, 9, 1, 2, 3])   # nodes 1,2,3 in region 0; 5,6,7 in regions 1,2,3
    p_region = torch.zeros(10, dtype=torch.float64)
    p_region[0] = 0.4                                    # region 0 shared error -> intra-region correlation
    p_node = torch.full((8,), 0.1, dtype=torch.float64)
    ev = EvidenceModel(region_of=region_of, p_region=p_region, p_node=p_node)
    src = torch.tensor([0, 0, 0, 4, 4, 4])
    dst = torch.tensor([1, 2, 3, 5, 6, 7])               # node 0 -> region-0 peers; node 4 -> diverse
    w = torch.ones(6, dtype=torch.float64)
    keff = effective_sample_size(src, dst, 8, w, ev)
    assert keff[4] > keff[0] + 0.2                       # diverse peers -> higher ESS
    assert keff[0] < 3.0                                 # redundant peers -> ESS well below k=3


# --------------------------------------------------------- mixing / weak cut
def test_weak_cut_lowers_cross_region_mass_and_gap():
    # two regions (0,1), 3 nodes each; dense intra-region polling + a SINGLE cross edge
    region_of = torch.tensor([0, 0, 0, 1, 1, 1])
    src = torch.tensor([0, 0, 1, 1, 2, 3, 3, 4, 4, 5, 2])   # last edge 2->3 is the weak cut
    dst = torch.tensor([1, 2, 0, 2, 1, 4, 5, 3, 5, 4, 3])
    pit = torch.ones(src.numel(), dtype=torch.float64)
    cross = float(cross_region_response_mass(src, dst, region_of, pit))
    assert cross < 0.15                                   # weak cut -> little cross-region mass
    P = region_response_kernel(src, dst, region_of, pit, 2)
    gap = float(region_spectral_gap(P))
    # a well-mixed kernel (add a symmetric strong cut) has a larger gap
    src2 = torch.cat([src, torch.tensor([3, 0, 4, 1, 5, 2])])
    dst2 = torch.cat([dst, torch.tensor([0, 3, 1, 4, 2, 5])])
    pit2 = torch.ones(src2.numel(), dtype=torch.float64)
    gap_mixed = float(region_spectral_gap(region_response_kernel(src2, dst2, region_of, pit2, 2)))
    assert gap_mixed > gap


# --------------------------------------------------------- hub load
def test_hub_raises_receiver_load():
    # nodes 1..6 all poll the hub (node 0); node 0 also polls node 1
    N = 7
    src = torch.tensor([1, 2, 3, 4, 5, 6, 0])
    dst = torch.tensor([0, 0, 0, 0, 0, 0, 1])
    pi = torch.ones(src.numel(), dtype=torch.float64)
    tau = torch.ones(N, dtype=torch.float64)
    load = receiver_load(pi, tau, src, dst, N)
    assert float(load[0]) == 6.0                          # hub receives 6 queries
    assert float(load[0]) > float(load[1:].max())         # hub is the most loaded
