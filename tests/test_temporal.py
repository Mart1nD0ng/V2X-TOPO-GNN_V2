"""G-TEMPORAL (S17): the temporal-memory primitive + temporal correlated-evidence sequence and
their Mechanism Identifiability Contract (C1-C5).

The headline stays the macrostate basin first-hitting (legacy global-risk emission does NOT enter);
the memory is an auxiliary, causal, observable-driven, differentiable signal feeding the CDQ 2.0
query. Here we lock the contract: causality (no future leak, C2), differentiability (C3),
memory-off == static mainline (C5), matched-marginal-in-time (C1/C4), persistence scoping (C1),
observable truth-independence (C2).
"""

import statistics

import torch

from src.environment import build_manhattan_scene
from src.environment.temporal_sequence import TemporalCorrelationSequence
from src.metrics.temporal_memory import causal_ema, estimate_quality, no_memory


def _scene(seed=0):
    return build_manhattan_scene(3, 3, 3, block_m=120.0, comm_radius=95.0, int_radius=150.0,
                                 generator=torch.Generator().manual_seed(seed))


# ---------------------------------------------------------------- memory primitive: C2/C3/C5
def test_causal_ema_matches_recurrence():
    x = torch.randn(6, 4, dtype=torch.float64)
    rho = 0.4
    m = causal_ema(x, rho)
    ref = [x[0]]
    for t in range(1, 6):
        ref.append((1 - rho) * ref[-1] + rho * x[t])
    assert torch.allclose(m, torch.stack(ref), atol=1e-12)


def test_causal_ema_no_future_leak():
    """C2: m_t depends ONLY on x_{<=t}; a future x_{t+1..} has EXACTLY zero gradient into m_t."""
    x = torch.randn(8, 3, dtype=torch.float64, requires_grad=True)
    m = causal_ema(x, 0.5)
    # gradient of an early memory entry w.r.t. all inputs: future rows must be exactly 0
    g = torch.autograd.grad(m[3].sum(), x, retain_graph=True)[0]
    assert torch.count_nonzero(g[4:]) == 0          # x_{>3} does not influence m_3 (no future leak)
    assert torch.count_nonzero(g[:4]) > 0           # x_{<=3} does
    # and perturbing a future value leaves m_3 bit-identical
    x2 = x.detach().clone(); x2[5] += 10.0
    assert torch.allclose(causal_ema(x2, 0.5)[3], m[3].detach(), atol=1e-12)


def test_causal_ema_differentiable_in_x_and_rho():
    """C3: smooth in x and rho (the loss on the memory reaches the CDQ params; no detach/.item())."""
    x = torch.randn(5, 2, dtype=torch.float64, requires_grad=True)
    rho = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    (causal_ema(x, rho) ** 2).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert rho.grad is not None and torch.isfinite(rho.grad) and float(rho.grad.abs()) > 0


def test_memory_off_recovers_current_epoch():
    """C5: rho=1 => m_t = x_t (memory OFF), and no_memory is the current-epoch baseline."""
    x = torch.randn(5, 3, dtype=torch.float64)
    assert torch.allclose(causal_ema(x, 1.0), x, atol=1e-12)
    assert torch.equal(no_memory(x), x)


# ---------------------------------------------------------------- temporal env: C1/C2/C4
def test_matched_marginal_in_time():
    """C1/C4: every epoch has the SAME marginal q_i (only the correlated band / its persistence
    changes), for BOTH persistence regimes -- so a marginal-only policy is blind to the structure."""
    sc = _scene()
    for persistence in (0.0, 1.0):
        seq = TemporalCorrelationSequence(sc, T=8, persistence=persistence, base_node_err=0.35,
                                          corr_strength=0.3, seed=1)
        q0 = seq.model(0).correct_observation_prob()
        for t in range(1, 8):
            assert torch.allclose(seq.model(t).correct_observation_prob(), q0, atol=1e-12)


def test_persistence_controls_active_band_schedule():
    """C1: persistence=1 keeps the active band constant; persistence=0 reshuffles it."""
    sc = _scene()
    persistent = TemporalCorrelationSequence(sc, T=30, persistence=1.0, seed=2)
    iid = TemporalCorrelationSequence(sc, T=30, persistence=0.0, seed=2)
    p_oh = persistent.true_active_onehot()
    assert int(p_oh.argmax(dim=1).unique().numel()) == 1          # one band, always
    iid_oh = iid.true_active_onehot()
    assert int(iid_oh.argmax(dim=1).unique().numel()) > 1         # multiple bands over time


def test_observable_proxy_is_truth_independent():
    """C2: the observable proxy is a per-band MEASUREMENT (schedule + noise), NOT the sampled truth.
    Sampling the evidence (drawing Y*/bits) does not change the proxy."""
    sc = _scene()
    seq = TemporalCorrelationSequence(sc, T=5, persistence=0.7, seed=3)
    p1 = seq.proxy_sequence()
    _ = seq.model(0).sample(64, generator=torch.Generator().manual_seed(99))   # draw truth
    p2 = seq.proxy_sequence()
    assert torch.equal(p1, p2)                                    # proxy independent of any truth draw
    assert p1.shape == (5, seq.n_sensor)


def test_memory_driven_policy_runs_in_mc_and_eta_zero_is_static():
    """Integration + C5: a memory-driven CDQ 2.0 diversity policy runs in the independent dynamic MC
    (basins sum to 1), and at eta=0 it reduces to the static ESP mainline (memory cannot matter)."""
    import torch.nn.functional as F
    from src.config.service_profile import ConsensusServiceProfile
    from src.environment import ProtocolConfig, RoundPhysicsConfig
    from src.metrics.participation import uniform_participation
    from src.sampling import DistanceQueryPolicy
    from src.sampling.cdq2_wiring import CDQ2Policy
    from src.validation import run_dynamic_mc

    sc = _scene()
    seq = TemporalCorrelationSequence(sc, T=4, persistence=1.0, seed=4)
    mem = causal_ema(seq.proxy_sequence(), 0.3)
    model = seq.model(0)
    n = seq.n_sensor
    w = (2.0 * mem[0].clamp_min(0.0).sqrt())

    def div(graph):
        b = model.sensor_of[graph.dst_index]
        return torch.cat([w[b].unsqueeze(-1), F.one_hot(b, n).to(torch.float64)], dim=-1)

    proto = ProtocolConfig(k=3, alpha=2, beta=3, r_max=8)
    phy = RoundPhysicsConfig(subchannels=10, slots_per_window=40)
    prof = ConsensusServiceProfile.urban_default().replace(k=3, alpha=2, beta=3, max_poll_epochs=8)
    omega = uniform_participation(sc.num_nodes)
    base = DistanceQueryPolicy(beta_per_m=0.05)
    args = dict(num_trials=300, generator=torch.Generator().manual_seed(0), link_override=0.85,
                service_profile=prof, participation=omega)
    mc = run_dynamic_mc(sc, model, CDQ2Policy(base, r=n + 1, eta=6.0, diversity=div), proto, phy, **args)
    assert abs((mc.basin_P_correct + mc.basin_F_wrong + mc.basin_F_split + mc.basin_F_deadline) - 1.0) < 1e-9
    # eta=0: the memory-driven CDQ2 collapses to ESP -> same basin outcomes (same CRN)
    esp = run_dynamic_mc(sc, model, base, proto, phy, **args)
    cdq2_0 = run_dynamic_mc(sc, model, CDQ2Policy(base, r=n + 1, eta=0.0, diversity=div), proto, phy, **args)
    assert abs(esp.basin_P_correct - cdq2_0.basin_P_correct) < 0.04


def test_memory_estimate_scoped_to_persistence():
    """C1 scoping: a memory (EMA of the observable proxy) tracks the persistent correlated band
    well, but under iid-in-time it cannot beat the marginal mean -- so memory only helps when the
    structure persists (matched-marginal-in-time isolates this; the marginals are identical)."""
    sc = _scene()
    q_persist, q_iid = [], []
    for seed in range(6):
        sp = TemporalCorrelationSequence(sc, T=24, persistence=1.0, obs_noise=0.4, seed=seed)
        si = TemporalCorrelationSequence(sc, T=24, persistence=0.0, obs_noise=0.4, seed=seed)
        q_persist.append(float(estimate_quality(causal_ema(sp.proxy_sequence(), 0.3),
                                                 sp.true_active_onehot())))
        q_iid.append(float(estimate_quality(causal_ema(si.proxy_sequence(), 0.3),
                                             si.true_active_onehot())))
    assert statistics.mean(q_persist) > statistics.mean(q_iid) + 0.2
    assert statistics.mean(q_persist) > 0.3                       # the memory genuinely tracks structure
