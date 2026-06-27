"""G-MACROSTATE: participation-weighted macrostate + basin first-hitting (spec §3-§4).

This is the headline-metric replacement for the legacy node-union ``1-∏(1-w_i)``
(forbidden shortcut #1). The macrostate consensus quantities are participation-weighted
*masses* in [0,1]

    C_r = Σ_i ω_i 1{i correctly finalized},   W_r = Σ_i ω_i 1{i wrongly finalized},
    U_r = 1 − C_r − W_r,

and the run outcome is the FIRST basin hit among correct / wrong / split, else deadline
(spec §4). The basins are disjoint by the profile invariant ρ_s > 1 − ρ_f, the four outcome
probabilities sum to 1, and — crucially — replicating the population under uniform ω leaves
every basin outcome unchanged (no mechanical N-coupling, unlike the node union).
"""

import pytest
import torch

from src.config.service_profile import ConsensusServiceProfile
from src.metrics.basins import basin_label, basins_disjoint
from src.metrics.first_hitting import (
    basin_outcome_probabilities,
    first_hitting_outcome,
)
from src.metrics.macrostate import (
    macrostate_occupancy,
    pairwise_disagreement,
    region_disagreement,
    strict_disagreement,
)

PROFILE = ConsensusServiceProfile.urban_default()  # ρ_f=0.6, ρ_s=0.45, R_d=20


# ----------------------------------------------------------------- macrostate occupancy
def test_occupancy_sums_to_one_and_matches_weights():
    # 5 nodes, states: +1 correct, -1 wrong, 0 undecided
    state = torch.tensor([1, 1, -1, 0, 0])
    omega = torch.full((5,), 1.0 / 5, dtype=torch.float64)
    C, W, U = macrostate_occupancy(state, omega)
    assert torch.isclose(C, torch.tensor(2.0 / 5, dtype=torch.float64))
    assert torch.isclose(W, torch.tensor(1.0 / 5, dtype=torch.float64))
    assert torch.isclose(C + W + U, torch.tensor(1.0, dtype=torch.float64))


def test_occupancy_respects_nonuniform_participation():
    state = torch.tensor([1, -1])
    omega = torch.tensor([0.9, 0.1], dtype=torch.float64)  # node 0 dominates the scope
    C, W, U = macrostate_occupancy(state, omega)
    assert torch.isclose(C, torch.tensor(0.9, dtype=torch.float64))
    assert torch.isclose(W, torch.tensor(0.1, dtype=torch.float64))


def test_occupancy_batched_trajectory():
    # [T=2, R+1=3, N=4] finalization trajectory (absorbing once decided)
    traj = torch.tensor([
        [[0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, -1]],
        [[0, 0, 0, 0], [-1, -1, 0, 0], [-1, -1, -1, 0]],
    ])
    omega = torch.full((4,), 0.25, dtype=torch.float64)
    C, W, U = macrostate_occupancy(traj, omega)
    assert C.shape == (2, 3)
    assert torch.isclose(C[0, 2], torch.tensor(0.5, dtype=torch.float64))   # 2/4 correct
    assert torch.isclose(W[1, 2], torch.tensor(0.75, dtype=torch.float64))  # 3/4 wrong


# ----------------------------------------------------------------- basins (spec §4)
def test_basins_are_disjoint_under_profile_invariant():
    assert basins_disjoint(PROFILE)
    # an overlapping (invalid) threshold pair would not be disjoint — but the profile forbids
    # constructing it, so we test the predicate on the raw masses directly:
    rho_f, rho_s = PROFILE.correct_basin_mass, PROFILE.split_basin_mass
    # no (C, W) can satisfy two basins at once
    grid = torch.linspace(0, 1, 21)
    for c in grid:
        for w in grid:
            if float(c + w) > 1.0 + 1e-9:
                continue
            labels = []
            if c >= rho_f:
                labels.append("correct")
            if w >= rho_f:
                labels.append("wrong")
            if c >= rho_s and w >= rho_s:
                labels.append("split")
            assert len(labels) <= 1, f"overlap at C={float(c)},W={float(w)}: {labels}"


def test_basin_label_classifies_each_basin():
    assert basin_label(torch.tensor(0.7), torch.tensor(0.1), PROFILE) == "correct"
    assert basin_label(torch.tensor(0.1), torch.tensor(0.7), PROFILE) == "wrong"
    assert basin_label(torch.tensor(0.46), torch.tensor(0.46), PROFILE) == "split"
    assert basin_label(torch.tensor(0.3), torch.tensor(0.2), PROFILE) == "none"


# ----------------------------------------------------------------- first hitting (spec §4)
def test_first_hitting_returns_first_basin_and_tau():
    # correct mass crosses ρ_f=0.6 first at epoch 2
    C = torch.tensor([0.0, 0.4, 0.7, 0.9], dtype=torch.float64)
    W = torch.tensor([0.0, 0.1, 0.1, 0.05], dtype=torch.float64)
    fh = first_hitting_outcome(C, W, PROFILE)
    assert fh.outcome == "correct"
    assert fh.tau == 2


def test_first_hitting_deadline_when_no_basin_reached():
    R = PROFILE.max_poll_epochs
    C = torch.full((R + 1,), 0.3, dtype=torch.float64)
    W = torch.full((R + 1,), 0.2, dtype=torch.float64)
    fh = first_hitting_outcome(C, W, PROFILE)
    assert fh.outcome == "deadline"
    assert fh.tau == R + 1


def test_first_hitting_split_before_correct():
    # split (both ≥ ρ_s=0.45) at epoch 1, correct (≥0.6) only later — split wins (it's first)
    C = torch.tensor([0.0, 0.46, 0.62], dtype=torch.float64)
    W = torch.tensor([0.0, 0.46, 0.46], dtype=torch.float64)
    fh = first_hitting_outcome(C, W, PROFILE)
    assert fh.outcome == "split"
    assert fh.tau == 1


def test_outcome_probabilities_sum_to_one():
    # batch of 4 paths exercising each outcome
    R = PROFILE.max_poll_epochs
    def pad(seq):
        t = torch.full((R + 1,), seq[-1], dtype=torch.float64)
        t[: len(seq)] = torch.tensor(seq, dtype=torch.float64)
        return t
    C = torch.stack([pad([0, 0.7]), pad([0, 0.1]), pad([0, 0.46, 0.46]), pad([0.3])])
    W = torch.stack([pad([0, 0.1]), pad([0, 0.7]), pad([0, 0.46, 0.46]), pad([0.2])])
    res = basin_outcome_probabilities(C, W, PROFILE)
    total = res["P_correct"] + res["F_wrong"] + res["F_split"] + res["F_deadline"]
    assert abs(total - 1.0) < 1e-12
    assert res["P_correct"] == 0.25 and res["F_wrong"] == 0.25
    assert res["F_split"] == 0.25 and res["F_deadline"] == 0.25


# ------------------------------------------------- the anti-node-union invariant (plan §5)
def test_population_replication_leaves_basin_outcome_invariant():
    """Under uniform ω, duplicating the population (2N identical copies) must NOT change the
    macrostate masses or the basin outcome — the metric is a participation-weighted FRACTION,
    not a node union. (The legacy 1-(1-p)^N would change with N.)"""
    # a single realised trajectory over N=4 nodes
    base = torch.tensor([
        [0, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 1, 0],
    ])  # [R+1=3, N=4]
    omega_n = torch.full((4,), 0.25, dtype=torch.float64)
    Cn, Wn, _ = macrostate_occupancy(base, omega_n)
    fh_n = first_hitting_outcome(Cn, Wn, PROFILE)

    # replicate every node once -> 2N=8 nodes, uniform ω = 1/8, same per-node states
    rep = base.repeat(1, 2)  # [3, 8]
    omega_2n = torch.full((8,), 1.0 / 8, dtype=torch.float64)
    C2, W2, _ = macrostate_occupancy(rep, omega_2n)
    fh_2n = first_hitting_outcome(C2, W2, PROFILE)

    assert torch.allclose(Cn, C2) and torch.allclose(Wn, W2)
    assert fh_n.outcome == fh_2n.outcome and fh_n.tau == fh_2n.tau


def test_pairwise_disagreement_zero_when_unanimous_max_when_balanced():
    zero = pairwise_disagreement(torch.tensor(0.8), torch.tensor(0.0))
    balanced = pairwise_disagreement(torch.tensor(0.4), torch.tensor(0.4))
    skewed = pairwise_disagreement(torch.tensor(0.7), torch.tensor(0.1))
    assert float(zero) < 1e-9
    assert float(balanced) > float(skewed)        # C=W maximises pairwise disagreement
    assert float(balanced) <= 1.0 + 1e-9


def test_region_disagreement_high_when_regions_oppose():
    # two regions: region 0 all correct, region 1 all wrong -> high regional disagreement
    state = torch.tensor([1, 1, -1, -1])
    region = torch.tensor([0, 0, 1, 1])
    omega = torch.full((4,), 0.25, dtype=torch.float64)
    opposed = region_disagreement(state, omega, region)
    # mixed-but-homogeneous regions -> lower regional disagreement
    state2 = torch.tensor([1, -1, 1, -1])
    homog = region_disagreement(state2, omega, region)
    assert float(opposed) > float(homog)


def test_strict_disagreement_is_a_separate_fixed_n_audit():
    """F_strict = P(∃ decided i,j with Y_i≠Y_j) is retained as a fixed-N safety audit (spec §4),
    distinct from the cross-scale basin outcome."""
    both = torch.tensor([1, -1, 0])      # a correct and a wrong decided node coexist
    only_correct = torch.tensor([1, 1, 0])
    assert bool(strict_disagreement(both))
    assert not bool(strict_disagreement(only_correct))
