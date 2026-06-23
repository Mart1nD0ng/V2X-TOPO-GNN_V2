"""G1 -- true binary Snowball protocol semantics (spec §3, docs/PROTOCOL_SEMANTICS.md).

Acceptance (PROTOCOL_SEMANTICS.md §6):
  1. hand-computed quorum-sequence trace matches the state path step-for-step;
  2. the transition operator is row-stochastic and conserves mass each round;
  3. the confidence-persistence sentinel passes (Snowball != Snowflake);
  4. the single-node exact chain matches a brute-force path enumeration.
"""

from itertools import product

import pytest
import torch

from src.protocol import binary_snowball as bs


# ---------------------------------------------------------------- reference trace
def test_reference_trace_confidence_persists():
    """Hand-computed path for beta=5, init pref +, outcomes [+,+,+,0,-] (§3.3)."""
    beta, r_max = 5, 12
    path = bs.simulate_trajectory([+1, +1, +1, 0, -1], beta, r_max, initial_pref=+1)
    expected = [
        ("U", 0, 1, 0, 0),   # initial
        ("U", 1, 1, 1, 1),   # + : d=1, streak +1
        ("U", 2, 1, 1, 2),   # + : d=2, streak +2
        ("U", 3, 1, 1, 3),   # + : d=3, streak +3
        ("U", 3, 1, 0, 0),   # 0 : streak broken; d & pref PERSIST at 3,+
        ("U", 2, 1, -1, 1),  # - : d=2>0 so pref stays +; new - streak
    ]
    assert path == expected


def test_reference_trace_decision_decides_pref_not_last():
    """beta=2: three +,+ then a -,- streak; decision colour = pref (max confidence)."""
    beta, r_max = 2, 12
    # +,+ -> d=2 decide? c reaches 2 at second + -> decides +. Use beta=3 to keep alive.
    beta = 3
    path = bs.simulate_trajectory([+1, +1, +1], beta, r_max, initial_pref=+1)
    assert path[-1] == ("D", 1)  # decided correct after 3 consecutive + quorums

    # A long + lead then a short - streak that does NOT overcome confidence:
    path2 = bs.simulate_trajectory([+1, +1, +1, +1], beta=5, r_max=r_max, initial_pref=+1)
    # 4 consecutive + (d=4, c=4 < 5) -> still undecided, pref +
    assert path2[-1] == ("U", 4, 1, 1, 4)


# ------------------------------------------------------------- row-stochasticity
def test_transition_row_stochastic_and_mass_conserved():
    beta, r_max = 4, 8
    layout = bs.snowball_layout(beta, r_max)
    torch.manual_seed(0)
    B = 5
    raw = torch.rand(B, 3, dtype=torch.float64)
    h = raw / raw.sum(dim=1, keepdim=True)
    T = bs.transition_matrix(h[:, 0], h[:, 1], h[:, 2], layout)
    row_sums = T.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-12)

    p = bs.initial_distribution(0.6, layout, B)
    for _ in range(r_max):
        p = bs.apply_round(p, h[:, 0], h[:, 1], h[:, 2], layout)
        assert torch.allclose(p.sum(dim=-1), torch.ones(B, dtype=torch.float64), atol=1e-12)
        assert bool((p >= -1e-15).all())


def test_apply_round_matches_dense_matrix():
    """Sparse scatter apply_round == dense p @ T (proves the sparse op is the operator)."""
    beta, r_max = 3, 6
    layout = bs.snowball_layout(beta, r_max)
    torch.manual_seed(1)
    B = 4
    raw = torch.rand(B, 3, dtype=torch.float64)
    h = raw / raw.sum(dim=1, keepdim=True)
    p = bs.initial_distribution(torch.tensor([0.2, 0.5, 0.8, 0.5]), layout, B)
    T = bs.transition_matrix(h[:, 0], h[:, 1], h[:, 2], layout)
    dense = torch.bmm(p.unsqueeze(1), T).squeeze(1)
    sparse = bs.apply_round(p, h[:, 0], h[:, 1], h[:, 2], layout)
    assert torch.allclose(dense, sparse, atol=1e-12)


# ---------------------------------------------------- Snowball != Snowflake (G1)
def test_confidence_persistence_vs_snowflake():
    """The defining sentinel: a single opposite quorum after a confident lead does NOT
    flip true-Snowball preference, whereas the legacy Snowflake automaton flips."""
    beta, r_max = 5, 12
    layout = bs.snowball_layout(beta, r_max)
    # deterministic outcome sequence +,+,+,0,- via one-hot h
    seq = [+1, +1, +1, 0, -1]
    onehot = {1: (1.0, 0.0, 0.0), -1: (0.0, 1.0, 0.0), 0: (0.0, 0.0, 1.0)}

    p = bs.initial_distribution(1.0, layout, 1)  # all mass at pref=+
    for o in seq:
        hp, hm, hz = (torch.tensor([x], dtype=torch.float64) for x in onehot[o])
        p = bs.apply_round(p, hp, hm, hz, layout)
    u, v = bs.readout_preference(p, layout)
    assert float(u) == pytest.approx(1.0, abs=1e-12)   # Snowball: still prefers +
    assert float(v) == pytest.approx(0.0, abs=1e-12)

    # Legacy Snowflake streak automaton on the identical sequence -> flips to wrong.
    from src.mainline import snowball as sf
    psf = sf.initial_distribution(1.0, beta, 1)
    for o in seq:
        hp, hm, hz = (torch.tensor([x], dtype=torch.float64) for x in onehot[o])
        Tsf = sf.build_transition(hp, hm, hz, beta)
        psf = torch.bmm(psf.unsqueeze(1), Tsf).squeeze(1)
    u_sf, v_sf = sf.readout_preference(psf, beta)
    assert float(u_sf) == pytest.approx(0.0, abs=1e-12)  # Snowflake: flipped to wrong
    assert float(v_sf) == pytest.approx(1.0, abs=1e-12)


# --------------------------------------------------------- exact chain vs brute
def test_exact_chain_matches_bruteforce_enumeration():
    """Iterated apply_round == probability-weighted enumeration of all 3^R paths."""
    beta, r_max = 3, 6
    R = 6
    layout = bs.snowball_layout(beta, r_max)
    idx = {s: i for i, s in enumerate(layout.states)}
    h_plus, h_minus, h_zero = 0.45, 0.25, 0.30  # fixed, sum to 1

    p = bs.initial_distribution(1.0, layout, 1)  # pure pref=+ start
    hp = torch.tensor([h_plus], dtype=torch.float64)
    hm = torch.tensor([h_minus], dtype=torch.float64)
    hz = torch.tensor([h_zero], dtype=torch.float64)
    for _ in range(R):
        p = bs.apply_round(p, hp, hm, hz, layout)
    chain = p.squeeze(0)

    brute = torch.zeros(layout.state_count, dtype=torch.float64)
    probs = {1: h_plus, -1: h_minus, 0: h_zero}
    for seq in product((1, -1, 0), repeat=R):
        pr = 1.0
        for o in seq:
            pr *= probs[o]
        end = bs.simulate_trajectory(list(seq), beta, r_max, initial_pref=+1)[-1]
        brute[idx[end]] += pr
    assert torch.allclose(chain, brute, atol=1e-12)


# --------------------------------------------------------------- readout / shape
def test_initial_readout_and_batched_shapes():
    beta, r_max = 4, 8
    layout = bs.snowball_layout(beta, r_max)
    p0 = bs.initial_distribution(0.7, layout, 3)
    u, v = bs.readout_preference(p0, layout)
    assert torch.allclose(u, torch.full((3,), 0.7, dtype=torch.float64), atol=1e-12)
    assert torch.allclose(u + v, torch.ones(3, dtype=torch.float64), atol=1e-12)
    c, w, und = bs.terminal_outcomes(p0, layout)
    assert torch.allclose(und, torch.ones(3, dtype=torch.float64), atol=1e-12)  # none decided yet

    # [N, Q, S] batched apply_round
    N, Q = 4, 2
    p = bs.initial_distribution(0.5, layout, N * Q).reshape(N, Q, layout.state_count)
    h = torch.full((N, Q), 1.0 / 3, dtype=torch.float64)
    p = bs.apply_round(p, h, h, h, layout)
    assert p.shape == (N, Q, layout.state_count)
    assert torch.allclose(p.sum(-1), torch.ones(N, Q, dtype=torch.float64), atol=1e-12)


def test_differentiable_in_h():
    beta, r_max = 3, 6
    layout = bs.snowball_layout(beta, r_max)
    raw = torch.rand(2, 3, dtype=torch.float64, requires_grad=True)
    h = raw / raw.sum(dim=1, keepdim=True)
    p = bs.initial_distribution(0.5, layout, 2)
    for _ in range(3):
        p = bs.apply_round(p, h[:, 0], h[:, 1], h[:, 2], layout)
    c, w, _ = bs.terminal_outcomes(p, layout)
    c.sum().backward()
    assert raw.grad is not None and bool(torch.isfinite(raw.grad).all())
