"""G-MACROSTATE / Phase 1: exogenous normalized participation measure ω (spec §2).

ω_i ≥ 0, Σ_i ω_i = 1, EXOGENOUS (from the application/scene, never the query policy),
and not derived from simulator-only truth. Two admissible rules: uniform (1/N) and an
application-weighted soft scope K_app(d_i, …) (spec §2). The headline reports both
(uniform + application sensitivity).
"""

import inspect

import pytest
import torch

from src.environment.urban_scene import build_manhattan_scene
from src.metrics.participation import (
    application_participation,
    participation_measure,
    uniform_participation,
)


def _scene(n_seg_veh=3, gx=3, gy=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    return build_manhattan_scene(gx, gy, n_seg_veh, generator=g)


def test_uniform_sums_to_one_and_is_flat():
    w = uniform_participation(17, dtype=torch.float64)
    assert torch.isclose(w.sum(), torch.tensor(1.0, dtype=torch.float64), atol=1e-12)
    assert torch.allclose(w, torch.full_like(w, 1.0 / 17))
    assert bool((w >= 0).all())


def test_application_normalized_and_nonneg():
    scene = _scene()
    event = scene.positions.mean(dim=0)
    w = application_participation(scene, event_xy=event, length_scale=50.0)
    assert torch.isclose(w.sum(), torch.tensor(1.0, dtype=w.dtype), atol=1e-12)
    assert bool((w >= 0).all())
    assert w.numel() == scene.num_nodes


def test_application_weights_closer_vehicles_more():
    scene = _scene()
    event = scene.positions[0]
    w = application_participation(scene, event_xy=event, length_scale=40.0)
    d = (scene.positions - event).norm(dim=1)
    # monotone: the closest vehicle to the event has strictly more mass than the farthest
    assert w[d.argmin()] > w[d.argmax()]


def test_application_is_deterministic():
    scene = _scene()
    event = scene.positions.mean(dim=0)
    w1 = application_participation(scene, event_xy=event, length_scale=50.0)
    w2 = application_participation(scene, event_xy=event, length_scale=50.0)
    assert torch.equal(w1, w2)


def test_participation_dispatch_matches_rules():
    scene = _scene()
    event = scene.positions.mean(dim=0)
    wu = participation_measure(scene, "uniform")
    assert torch.allclose(wu, uniform_participation(scene.num_nodes, dtype=wu.dtype))
    wa = participation_measure(scene, "application", event_xy=event, length_scale=50.0)
    assert torch.allclose(wa, application_participation(scene, event_xy=event, length_scale=50.0))


def test_participation_is_not_a_gradient_target():
    # ω is exogenous: it must never carry policy gradient (constraint: policy cannot modify ω).
    scene = _scene()
    event = scene.positions.mean(dim=0)
    for w in (participation_measure(scene, "uniform"),
              participation_measure(scene, "application", event_xy=event, length_scale=50.0)):
        assert not w.requires_grad


def test_participation_signature_takes_no_policy():
    # structural guarantee: the measure is a pure function of (scene, rule, app-config);
    # there is no way for a query policy/model to enter it.
    params = set(inspect.signature(participation_measure).parameters)
    assert not ({"policy", "model", "query_policy", "evidence", "truth"} & params)


def test_unknown_rule_raises():
    scene = _scene()
    with pytest.raises(ValueError, match="participation|rule|uniform|application"):
        participation_measure(scene, "from_policy")
