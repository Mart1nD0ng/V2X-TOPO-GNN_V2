"""G9c -- ESD-GNN mechanism ablation flags (runtime activation; spec §10.2).

Acceptance: each ablation switch (no-CDQ / no-region / no-interference / no-refinement) is a
RUNTIME-ACTIVE change -- it actually alters the model output and the canonical-path query law
(not merely an accepted config field, constraint #13). The full model and every ablation
produce valid differentiable kernels.
"""

import torch

from src.environment import build_manhattan_scene
from src.models import ESDGNN, ESDGNNConfig, ESDGNNQueryPolicy, build_scene_features


def _scene(seed=0):
    return build_manhattan_scene(3, 3, 4, block_m=120.0, comm_radius=95.0, int_radius=140.0,
                                 generator=torch.Generator().manual_seed(seed))


def _kernel(cfg, scene):
    torch.manual_seed(0)
    model = ESDGNN(cfg).double()
    feats = build_scene_features(scene, cfg)
    return model, feats, model(feats)


def test_no_cdq_switches_query_law_to_esp():
    scene = _scene()
    full = ESDGNNQueryPolicy(ESDGNN(ESDGNNConfig(k=3)).double(), scene)
    nocdq = ESDGNNQueryPolicy(ESDGNN(ESDGNNConfig(k=3, use_cdq=False)).double(), scene)
    assert full.query_law == "cdq" and nocdq.query_law == "esp"
    # the ESP ablation exposes log_weights (diagonal kernel); the CDQ one exposes a kernel
    lw = nocdq.log_weights(None)
    assert lw.shape[0] == nocdq.features.gc.num_edges and torch.isfinite(lw).all()
    q, b = full.kernel(None)
    assert bool((q > 0).all()) and b.shape[1] == full.model.cfg.r


def test_each_ablation_flag_is_runtime_active():
    """Disabling region / interference / refinement must CHANGE the model output (the mechanism
    is genuinely wired, not an ignored config field)."""
    scene = _scene()
    base = ESDGNNConfig(hidden_dim=24, r=4, n_enc=3, n_refine=2, k=3)
    _, _, (q_full, _) = _kernel(base, scene)
    for cfg in (ESDGNNConfig(hidden_dim=24, r=4, n_enc=3, n_refine=2, k=3, use_region=False),
                ESDGNNConfig(hidden_dim=24, r=4, n_enc=3, n_refine=2, k=3, use_interference=False),
                ESDGNNConfig(hidden_dim=24, r=4, n_enc=3, n_refine=0, k=3)):
        _, _, (q_abl, _) = _kernel(cfg, scene)
        # same init seed + same data, but a disabled channel changes the quality output
        assert not torch.allclose(q_full, q_abl), "ablation flag did not change the model output"


def test_all_variants_produce_valid_differentiable_output():
    scene = _scene()
    for cfg in (ESDGNNConfig(k=3),
                ESDGNNConfig(k=3, use_cdq=False),
                ESDGNNConfig(k=3, use_region=False),
                ESDGNNConfig(k=3, use_interference=False),
                ESDGNNConfig(k=3, n_refine=0)):
        torch.manual_seed(0)
        model = ESDGNN(cfg).double()
        feats = build_scene_features(scene, cfg)
        q, b = model(feats)
        assert bool((q > 0).all()) and bool(torch.isfinite(b).all())
        q.sum().backward()
        assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
