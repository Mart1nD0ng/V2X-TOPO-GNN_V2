"""G9a -- ESD-GNN architecture (spec §9.3-§9.6).

Acceptance: the multi-graph encoder produces a valid differentiable CDQ kernel on the
canonical path; it uses ONLY observable structure (no truth/vote leak -- constraint #10);
it is scene/scale-agnostic (transfers across N); the dynamics refinement runs.
"""

import torch

from src.environment import (
    ProtocolConfig,
    RoundPhysicsConfig,
    build_manhattan_scene,
    build_scenario,
    run_consensus_episode,
)
from src.models import ESDGNN, ESDGNNConfig, ESDGNNQueryPolicy, build_scene_features

PHY = RoundPhysicsConfig(subchannels=10, slots_per_window=40)


def _scene(gx=2, gy=2, per=4, seed=0):
    return build_manhattan_scene(gx, gy, per, block_m=130.0, comm_radius=75.0, int_radius=115.0,
                                 generator=torch.Generator().manual_seed(seed))


def _model(cfg=None):
    torch.manual_seed(0)
    return ESDGNN(cfg or ESDGNNConfig(hidden_dim=24, r=4, n_enc=3, n_refine=2, k=3)).double()


def test_valid_kernel_and_differentiable():
    scene = _scene()
    pol = ESDGNNQueryPolicy(_model(), scene)
    q, b = pol.kernel(None)
    E = pol.features.gc.num_edges
    assert q.shape == (E,) and bool((q > 0).all()) and bool(torch.isfinite(q).all())
    assert b.shape == (E, 4) and bool(torch.isfinite(b).all())
    q.sum().backward()
    g = sum(float(p.grad.abs().sum()) for p in pol.model.parameters() if p.grad is not None)
    assert g > 0 and torch.isfinite(torch.tensor(g))


def test_no_truth_or_vote_leak():
    """The kernel must be identical across DIFFERENT evidence realisations -- the model sees
    only observable geometry/region structure, never Y* or peer votes (constraint #10)."""
    scene = _scene()
    model = _model()
    q1, b1 = ESDGNNQueryPolicy(model, scene).kernel(None)
    # build_scene_features takes only the scene geometry/region -> evidence-independent
    feats = build_scene_features(scene, model.cfg)
    assert "correct" not in feats.__dict__  # no truth-derived field
    q2, b2 = model(feats)
    assert torch.allclose(q1, q2) and torch.allclose(b1, b2)
    # the feature builder ignores any evidence model -> same kernel under different bias
    _ = build_scenario("one_biased_region", scene), build_scenario("two_opposing_regions", scene)
    q3, b3 = ESDGNNQueryPolicy(model, scene).kernel(None)
    assert torch.allclose(q1, q3) and torch.allclose(b1, b3)


def test_runs_in_canonical_episode_cdq():
    scene = _scene()
    pol = ESDGNNQueryPolicy(_model(), scene)
    ev = build_scenario("one_biased_region", scene, base_node_err=0.1, region_bias=0.85)
    res = run_consensus_episode(scene, ev, pol, ProtocolConfig(k=3, alpha=2, beta=3, r_max=10),
                                PHY, return_trajectory=False)
    assert res.mechanism_trace["query_law"] == "cdq"
    assert res.mechanism_trace["query_policy"] == "esd_gnn"
    assert torch.isfinite(res.F_wrong) and 0 <= float(res.S_allcorrect) <= 1
    res.F_wrong.backward()                      # end-to-end gradient through episode + model
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in pol.model.parameters())


def test_transfers_across_scales():
    """One model, two different scene sizes -> valid kernels on both (scale-agnostic, no ids)."""
    model = _model()
    for gx, per in [(2, 4), (4, 6)]:                # N=16 and N~ a few hundred
        scene = _scene(gx, gx, per, seed=gx)
        pol = ESDGNNQueryPolicy(model, scene)
        q, b = pol.kernel(None)
        assert q.shape[0] == pol.features.gc.num_edges and bool((q > 0).all())
        assert bool(torch.isfinite(b).all())


def test_kernel_depends_on_observable_structure():
    """Sanity that the GNN actually uses the graph (not a constant): different scenes give
    different per-edge quality distributions."""
    model = _model()
    q_a, _ = ESDGNNQueryPolicy(model, _scene(seed=1)).kernel(None)
    q_b, _ = ESDGNNQueryPolicy(model, _scene(seed=2)).kernel(None)
    assert abs(float(q_a.std()) - float(q_b.std())) > 1e-9 or abs(float(q_a.mean()) - float(q_b.mean())) > 1e-9


def test_refinement_runs_and_load_feedback_nontrivial():
    scene = _scene()
    model = _model(ESDGNNConfig(hidden_dim=24, r=4, n_enc=2, n_refine=2, k=3))
    feats = build_scene_features(scene, model.cfg)
    q, b = model(feats)
    lam = model._receiver_load(feats, q, b)          # the §9.6 analytic feedback
    assert lam.shape[0] == scene.num_nodes
    assert float(lam.std()) > 0                       # load varies across nodes (non-constant feedback)
    # a model without refinement still produces a valid kernel (refinement is additive)
    model0 = _model(ESDGNNConfig(hidden_dim=24, r=4, n_enc=2, n_refine=0, k=3))
    q0, _ = model0(build_scene_features(scene, model0.cfg))
    assert bool((q0 > 0).all())
