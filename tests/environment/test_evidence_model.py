"""G2 -- correlated-evidence environment (spec §6, plan Phase 3).

Acceptance (spec §11 G2):
  * region correlation matches theory;
  * zero-correlation control recovers independence;
  * (policy-cannot-read-truth is enforced at the model interface, Phase 9; here the
    sampler keeps truth and observables in separate fields).
"""

import torch

from src.environment.evidence_model import EvidenceModel, pairwise_correlation_theory


def _empirical_corr(c: torch.Tensor, i: int, j: int) -> float:
    ci = c[:, i].to(torch.float64)
    cj = c[:, j].to(torch.float64)
    vi, vj = ci.var(unbiased=False), cj.var(unbiased=False)
    if float(vi) <= 0 or float(vj) <= 0:
        return 0.0
    cov = ((ci - ci.mean()) * (cj - cj.mean())).mean()
    return float(cov / (vi * vj) ** 0.5)


def _model():
    # 6 nodes, 2 regions: nodes 0,1,2 in region 0; nodes 3,4,5 in region 1
    region_of = torch.tensor([0, 0, 0, 1, 1, 1])
    p_region = torch.tensor([0.30, 0.10], dtype=torch.float64)
    p_node = torch.tensor([0.05, 0.20, 0.10, 0.15, 0.05, 0.25], dtype=torch.float64)
    return EvidenceModel(region_of=region_of, p_region=p_region, p_node=p_node)


def test_marginal_matches_empirical():
    m = _model()
    g = torch.Generator().manual_seed(0)
    s = m.sample(400_000, generator=g)
    emp = s.correct.to(torch.float64).mean(dim=0)
    assert torch.allclose(emp, m.correct_observation_prob(), atol=5e-3)


def test_empirical_correlation_matches_theory():
    m = _model()
    g = torch.Generator().manual_seed(1)
    s = m.sample(600_000, generator=g)
    # same region (0,1): positive; cross region (0,3): ~0
    for (i, j) in [(0, 1), (0, 2), (1, 2), (3, 4), (0, 3), (2, 5), (1, 4)]:
        theo = pairwise_correlation_theory(m, i, j)
        emp = _empirical_corr(s.correct, i, j)
        assert abs(emp - theo) < 0.01, f"pair {(i,j)}: emp={emp:.4f} theo={theo:.4f}"
    # qualitative: same-region strictly positive, cross-region zero
    assert pairwise_correlation_theory(m, 0, 1) > 0.05
    assert pairwise_correlation_theory(m, 0, 3) == 0.0


def test_zero_correlation_control_recovers_independence():
    # p_region = 0 -> B_g = 0 deterministically -> C_i = ¬E_i independent across nodes
    region_of = torch.tensor([0, 0, 1, 1])
    m = EvidenceModel(
        region_of=region_of,
        p_region=torch.zeros(2, dtype=torch.float64),
        p_node=torch.tensor([0.1, 0.2, 0.15, 0.25], dtype=torch.float64),
    )
    for i in range(4):
        for j in range(4):
            if i != j:
                assert pairwise_correlation_theory(m, i, j) == 0.0
    g = torch.Generator().manual_seed(2)
    s = m.sample(300_000, generator=g)
    assert abs(_empirical_corr(s.correct, 0, 1)) < 0.01  # same region, but no shared bias


def test_analytic_scenarios_consistent_with_marginal():
    m = _model()
    omega, init_cp = m.analytic_scenarios()  # [Q], [N, Q]
    assert abs(float(omega.sum()) - 1.0) < 1e-12
    assert bool((init_cp >= -1e-12).all()) and bool((init_cp <= 1 + 1e-12).all())
    # weighted marginal over scenarios == q_i
    marg = (init_cp * omega.unsqueeze(0)).sum(dim=1)  # [N]
    assert torch.allclose(marg, m.correct_observation_prob(), atol=1e-12)


def test_scenario_decomposition_reproduces_pairwise_correlation_exactly():
    """The shared-latent decomposition implies the exact theoretical correlation
    (conditional independence given Z): Cov = sum_r w_r p_ir p_jr - q_i q_j."""
    m = _model()
    omega, init_cp = m.analytic_scenarios()
    q = m.correct_observation_prob()
    for (i, j) in [(0, 1), (1, 2), (3, 5), (0, 3), (2, 4)]:
        e_cc = float((omega * init_cp[i] * init_cp[j]).sum())
        qi, qj = float(q[i]), float(q[j])
        vi, vj = qi * (1 - qi), qj * (1 - qj)
        corr = 0.0 if vi <= 0 or vj <= 0 else (e_cc - qi * qj) / (vi * vj) ** 0.5
        assert abs(corr - pairwise_correlation_theory(m, i, j)) < 1e-12


def test_analytic_scenarios_refuses_too_many_regions():
    region_of = torch.arange(20)
    m = EvidenceModel(
        region_of=region_of,
        p_region=torch.full((20,), 0.1, dtype=torch.float64),
        p_node=torch.full((20,), 0.1, dtype=torch.float64),
    )
    try:
        m.analytic_scenarios(max_scenarios=1 << 10)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_truth_and_observables_separated():
    """The sample keeps TRUTH-derived fields explicit; region structure (observable) is
    distinct from C_i / Y* (not policy-visible). Guards against truth leaking via the
    same field a policy would consume."""
    m = _model()
    s = m.sample(8, generator=torch.Generator().manual_seed(3))
    assert s.correct.dtype == torch.bool          # truth-derived
    assert s.region_bits.shape == (8, m.num_regions)
    # observable region assignment is a property of the model, independent of any sample
    assert m.region_of.shape == (m.num_nodes,)
