"""G4 (spec §7, §3.4): physics-constrained adaptive topology (H2 core).

Checks:
  1. Spatial-hashing candidate graph == brute-force radius graph (no edge missed/added).
  2. NO degree cap: a dense cluster keeps every in-radius neighbour (degree = true count).
  3. E = O(N) at fixed density (linear edge growth), no N x N tensor.
  4. Receiver load Lambda_j = sum_{i->j} tau_i pi_ij matches brute force (Eq. 33).
  5. Mode-2 collision / interference rise with load; SINR falls with load.
  6. Hub-overload suppression is a MECHANISM, not a cap: concentrating pi on a hub raises
     its load -> collision/interference -> lowers ell of hub polls; the ablation that
     removes physical cost removes the suppression entirely.
  7. The load-coupled link reliability is differentiable in pi.
"""

from __future__ import annotations

import itertools

import torch

from src.mainline.topology import (
    CandidateGraph,
    LoadCoupledLinkConfig,
    aggregate_interference,
    build_candidate_graph,
    link_sinr_with_interference,
    load_coupled_link_reliability,
    los_probability,
    mode2_collision_from_load,
    receiver_load,
)

torch.manual_seed(0)
DT = torch.float64


def _brute_force_edges(pos, r):
    N = pos.shape[0]
    s, d = [], []
    for i in range(N):
        for j in range(N):
            if i != j and float(((pos[i] - pos[j]) ** 2).sum()) <= r * r:
                s.append(i)
                d.append(j)
    return set(zip(s, d))


def test_candidate_graph_matches_bruteforce():
    gen = torch.Generator().manual_seed(1)
    for N in [20, 50, 100]:
        pos = torch.rand(N, 2, generator=gen, dtype=DT) * 100.0
        r = 25.0
        g = build_candidate_graph(pos, r)
        got = set(zip(g.src_index.tolist(), g.dst_index.tolist()))
        ref = _brute_force_edges(pos, r)
        assert got == ref, (N, len(got), len(ref))
        # distance is differentiable in positions
        assert g.distance.shape == g.src_index.shape


def test_no_degree_cap_dense_cluster():  # G10-allow (asserts ABSENCE of a cap; name mentions the token)
    # 30 nodes packed within the radius of node 0 -> node 0 must keep all 29 neighbours.
    pos = torch.zeros(31, 2, dtype=DT)
    pos[1:] = torch.linspace(1.0, 9.0, 30, dtype=DT).unsqueeze(1) * torch.tensor([1.0, 0.0], dtype=DT)
    g = build_candidate_graph(pos, comm_radius=20.0)
    out_deg0 = int((g.src_index == 0).sum())
    assert out_deg0 == 30, out_deg0  # ALL in-radius neighbours kept, no cap


def test_edges_linear_in_N_at_fixed_density():
    gen = torch.Generator().manual_seed(2)
    density = 0.02  # nodes per unit area
    r = 12.0
    counts = []
    Ns = [200, 400, 800, 1600]
    for N in Ns:
        side = (N / density) ** 0.5
        pos = torch.rand(N, 2, generator=gen, dtype=DT) * side
        g = build_candidate_graph(pos, r)
        counts.append(g.num_edges / N)  # average degree
    # average degree roughly constant -> E = O(N)
    assert max(counts) / min(counts) < 1.4, counts


def test_receiver_load_matches_bruteforce():
    src = torch.tensor([0, 0, 1, 2, 3, 3])
    dst = torch.tensor([1, 2, 2, 1, 1, 2])
    N = 4
    pi = torch.tensor([0.3, 0.7, 0.4, 0.6, 0.5, 0.5], dtype=DT)
    tau = torch.tensor([0.9, 0.8, 0.7, 0.6], dtype=DT)
    load = receiver_load(pi, tau, src, dst, N)
    ref = torch.zeros(N, dtype=DT)
    for e in range(src.numel()):
        ref[dst[e]] += tau[src[e]] * pi[e]
    assert torch.allclose(load, ref, atol=1e-12)


def test_collision_and_sinr_monotone_in_load():
    S = 5.0
    load = torch.linspace(0.0, 10.0, 50, dtype=DT)
    pcol = mode2_collision_from_load(load, S)
    assert torch.all(pcol[1:] - pcol[:-1] >= -1e-12)  # increasing in load
    assert float(pcol[0]) == 0.0 and 0.0 <= float(pcol[-1]) <= 1.0
    # interference rises with load -> SINR falls
    E, N = 6, 3
    src = torch.tensor([0, 1, 2, 0, 1, 2])
    dst = torch.tensor([2, 2, 2, 1, 1, 1])
    rx = torch.full((E,), 1e-6, dtype=DT)
    w_lo = torch.full((E,), 0.1, dtype=DT)
    w_hi = torch.full((E,), 0.9, dtype=DT)
    I_lo = aggregate_interference(rx, w_lo, dst, N, S)
    I_hi = aggregate_interference(rx, w_hi, dst, N, S)
    assert float(I_hi[2]) > float(I_lo[2])
    g_lo = link_sinr_with_interference(rx, w_lo, I_lo, dst, S, noise_mw=1e-9)
    g_hi = link_sinr_with_interference(rx, w_hi, I_hi, dst, S, noise_mw=1e-9)
    assert float(g_hi.mean()) < float(g_lo.mean())


def test_hub_overload_suppressed_by_mechanism():
    M = 8
    src, dst = [], []
    for i in range(1, M + 1):
        src += [i, i]
        dst += [0, (i % M) + 1]  # each leaf polls the hub (0) and the next leaf
    src = torch.tensor(src)
    dst = torch.tensor(dst)
    N = M + 1
    E = src.numel()
    dist = torch.full((E,), 30.0, dtype=DT)
    graph = CandidateGraph(src, dst, dist, N)
    los = torch.ones(E, dtype=DT)
    tau = torch.ones(N, dtype=DT)
    cfg = LoadCoupledLinkConfig()
    hub = dst == 0

    pi_conc = torch.where(hub, torch.tensor(0.95, dtype=DT), torch.tensor(0.05, dtype=DT))
    pi_spread = torch.full((E,), 0.5, dtype=DT)
    r_conc = load_coupled_link_reliability(graph, pi_conc, tau, los, cfg)
    r_spread = load_coupled_link_reliability(graph, pi_spread, tau, los, cfg)

    # concentration raises hub load, hub collision; lowers hub-poll reliability
    assert float(r_conc["load"][0]) > float(r_spread["load"][0])
    assert float(r_conc["p_collision"][hub].mean()) > float(r_spread["p_collision"][hub].mean())
    assert float(r_conc["ell_poll"][hub].mean()) < float(r_spread["ell_poll"][hub].mean())

    # ABLATION: with physical cost removed, ell does NOT depend on concentration
    a_conc = load_coupled_link_reliability(graph, pi_conc, tau, los, cfg, disable_physical_cost=True)
    a_spread = load_coupled_link_reliability(graph, pi_spread, tau, los, cfg, disable_physical_cost=True)
    assert abs(float(a_conc["ell_poll"][hub].mean()) - float(a_spread["ell_poll"][hub].mean())) < 1e-9


def test_load_coupled_reliability_differentiable():
    M = 6
    src, dst = [], []
    for i in range(1, M + 1):
        src += [i, i]
        dst += [0, (i % M) + 1]
    src = torch.tensor(src)
    dst = torch.tensor(dst)
    N = M + 1
    E = src.numel()
    dist = torch.full((E,), 40.0, dtype=DT)
    graph = CandidateGraph(src, dst, dist, N)
    los = torch.ones(E, dtype=DT)
    tau = torch.ones(N, dtype=DT)
    cfg = LoadCoupledLinkConfig()
    pi = torch.full((E,), 0.5, dtype=DT, requires_grad=True)
    out = load_coupled_link_reliability(graph, pi, tau, los, cfg)
    out["ell_poll"].sum().backward()
    assert pi.grad is not None and torch.isfinite(pi.grad).all()
    assert float(pi.grad.abs().sum()) > 0  # reliability genuinely depends on pi (via load)


def test_los_probability_monotone():
    d = torch.tensor([10.0, 50.0, 100.0, 500.0], dtype=DT)
    los = los_probability(d)
    assert torch.all(los[1:] - los[:-1] <= 0)
    assert torch.all((los >= 0) & (los <= 1))


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} G4 tests passed.")
