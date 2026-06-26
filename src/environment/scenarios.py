"""Named correlated-evidence scenarios (spec §6.3, plan Phase 3).

Builds an :class:`~src.environment.evidence_model.EvidenceModel` over a given
:class:`~src.environment.urban_scene.ManhattanScene` for each studied evidence regime.
These are *evidence-level* scenarios (who observes correctly and how correlated); the
geometric scenarios that change the graph itself (weak cut, hub congestion) are produced
by geometry edits and are added with the diagnostics that need them (G7) -- they are
listed here as ``GEOMETRIC_SCENARIOS`` so the catalogue is explicit, not silently missing.

Scenarios (correctness frame: ``+`` = aligned with ``Y*``):

* ``all_correct``  -- perfect-evidence control: ``p_region=p_node=0`` ⇒ ``q_i=1`` for all.
* ``iid``          -- independent node errors only: ``p_region=0`` ⇒ zero pairwise
                      correlation; ``q_i = 1 - p_node``.
* ``one_biased_region`` -- a single region suffers a shared error (``p_region`` high) ⇒
                      its vehicles mostly start wrong and are positively correlated; all
                      other regions are clean.
* ``two_opposing_regions`` -- the scene splits by median-x into two halves with opposite
                      initial opinions (one clean, one strongly biased) ⇒ two correlated
                      opinion clusters.
"""

from __future__ import annotations

import torch

from .evidence_model import EvidenceModel
from .urban_scene import ManhattanScene

__all__ = ["EVIDENCE_SCENARIOS", "GEOMETRIC_SCENARIOS", "build_scenario"]

EVIDENCE_SCENARIOS = ("all_correct", "iid", "one_biased_region", "two_opposing_regions")
GEOMETRIC_SCENARIOS = ("weak_cut", "hub_congestion")  # produced by geometry edits (G7)


def _segment_midpoints(scene: ManhattanScene) -> torch.Tensor:
    return scene.segment_endpoints.mean(dim=1)  # [G, 2]


def build_scenario(
    name: str,
    scene: ManhattanScene,
    *,
    base_node_err: float = 0.10,
    region_bias: float = 0.85,
    dtype: torch.dtype = torch.float64,
) -> EvidenceModel:
    """Build the evidence model for a named scenario over ``scene``.

    Args:
        name: one of :data:`EVIDENCE_SCENARIOS`.
        base_node_err: per-node independent error ``p_i`` (clean regions).
        region_bias: shared region error ``p_g`` applied to biased regions.
    """
    G = scene.num_regions
    region_of = scene.region_of
    p_node = torch.full((scene.num_nodes,), float(base_node_err), dtype=dtype)
    p_region = torch.zeros(G, dtype=dtype)

    if name == "all_correct":
        p_node = torch.zeros(scene.num_nodes, dtype=dtype)
        # p_region already 0
    elif name == "iid":
        pass  # p_region 0, p_node = base
    elif name == "one_biased_region":
        p_region[0] = float(region_bias)
    elif name == "two_opposing_regions":
        mids = _segment_midpoints(scene)
        median_x = mids[:, 0].median()
        right = mids[:, 0] > median_x
        p_region = torch.where(right, torch.full((G,), float(region_bias), dtype=dtype),
                               torch.zeros(G, dtype=dtype))
    else:
        raise ValueError(
            f"unknown scenario {name!r}; evidence scenarios are {EVIDENCE_SCENARIOS} "
            f"(geometric scenarios {GEOMETRIC_SCENARIOS} are built by geometry edits)"
        )
    return EvidenceModel(region_of=region_of, p_region=p_region, p_node=p_node)
