"""Global-risk emission and the temporal emission feedback (spec §3.8, Eqs. 58-59 -- G8).

The legacy design emitted a *per-node marginal confidence* as the inter-frame feature -- a
heuristic scalar with no global meaning.  The redesign replaces it with the **node's
contribution to the global risk** of the shared-mixture joint (the H1 object of G1):

    r_{ir}(t) = -log c_{ir}(t)                                                   (Eq. 58)

so that, per scenario ``r``, the global risk decomposes EXACTLY as a sum of node
contributions

    -log S_r(t) = sum_{i in H} r_{ir}(t),     S_r(t) = prod_{i in H} c_{ir}(t),

i.e. the emission is literally a summand of ``-log S_C``'s per-scenario log-likelihood,
which is the G1 loss.  The next-frame feature is the scenario-averaged, stop-gradient,
normalised, clipped risk

    e_i^t = clip( sg[ sum_r rho_r r_{ir}(t) ] / r_max , 0, 1 )                   (Eq. 59)

with ``rho`` the scenario posterior (Eq. 14) and ``sg`` the stop-gradient.  ``e_i^t`` feeds
the next temporal frame, aligning the recurrent emission with the global ``F``.

eps convention (decision D9, consistent with the G1 D4 fix).  We use the SAME multiplicative
floor as the evaluator, ``r_{ir} = -log(max(c_{ir}, eps))``, not the additive ``-log(c+eps)``
written literally in Eq. 58.  This (a) makes the per-scenario identity ``-log S_r = sum_i
r_{ir}`` hold to machine precision against the mainline ``S_r`` (which already floors with
``clamp_min(eps)``, D4), and (b) keeps risk non-negative (``c in [0,1] => r >= 0``), whereas
``-log(c+eps)`` is slightly negative at ``c -> 1``.  The two conventions agree to ``O(eps)``
for ``c >> eps`` (the normal regime); in the floored tail ``c <= eps`` they differ by up to
``log 2``, but there ``S_C`` is already in the total-failure floor and ``loss_F = -log S_C``
matches the mainline by construction (D4).  The choice is a numerical-consistency convention,
not a change of the modelled quantity; the literal additive form would in fact break the
``-log S_r`` identity against the mainline (err ~5e-6 >> the gate's 1e-12 tolerance).

Aggregate vs per-scenario.  The exact tie ``sum_{i in H} (r_max e_i) = -log S_C = loss_F``
holds only for ``Q=1``.  For ``Q>1`` the posterior-averaged aggregate equals
``E_rho[-log S_r]`` which by Jensen is ``>= -log S_C`` (a cross-entropy-like quantity); the
*per-scenario* identity ``-log S_r = sum_i r_{ir}`` and the ``S_C`` reconstruction remain
exact for any ``Q``.

THE BOUNDED-SCALAR CLAIM (spec §3.8, explicitly forbidden to *assert* without evidence).
One must NOT claim that, because ``e_i^t`` is a bounded scalar in ``[0,1]``, the recurrent
*hidden state* of a temporal model is automatically constrained.  :func:`hidden_state_boundedness_ablation`
is the mechanism experiment that tests this; its honest finding (see G8 gate / decision D9)
is that the claim is **falsified**: a bounded input does not bound the hidden state of an
arbitrary recurrence -- only a contractive / gated recurrence does.  What IS exactly true is
the weaker set of properties this module verifies: ``e in [0,1]`` by construction, the
stop-gradient cuts the backward risk path, and the emission is monotonically aligned with the
global ``F``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from .model import OperatingPointConfig, PreferenceConditionedTopologyGNN, model_operating_point
from .topology import CandidateGraph

__all__ = [
    "EmissionConfig",
    "global_risk_contribution",
    "neg_log_S_r",
    "risk_emission",
    "ScalarEmissionRecurrentModel",
    "RecurrenceState",
    "recurrence_step",
    "hidden_state_boundedness_ablation",
]


@dataclass(frozen=True)
class EmissionConfig:
    """Parameters of the Eq. 58-59 risk emission."""

    eps: float = 1e-6
    # r_max normalises the emission to [0,1].  Default = the theoretical maximum per-node
    # risk -log(eps): since e_i = (sum_r rho_r r_ir)/r_max is a posterior average of terms
    # each <= -log(eps), e_i <= 1 BEFORE the clip (the clip is then a numerical safety net,
    # not the thing that creates the bound).
    r_max: float | None = None

    def resolved_r_max(self) -> float:
        return -math.log(self.eps) if self.r_max is None else float(self.r_max)


def global_risk_contribution(
    c_ir: torch.Tensor, scenario_posterior: torch.Tensor, *, eps: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    """Eq. 58 risk contributions and their scenario-posterior average.

    Args:
        c_ir: ``[N, Q]`` terminal ``P(correct | Z=r)`` (G1 ``GlobalConsensusResult.c_ir``).
        scenario_posterior: ``[Q]`` posterior ``rho_r`` (Eq. 14).

    Returns:
        ``r_ir`` ``[N, Q]`` = ``-log(max(c_ir, eps))`` (>= 0), and the per-node risk
        ``node_risk`` ``[N]`` = ``sum_r rho_r r_ir`` (the Eq. 59 inner quantity, pre-sg).
    """
    if c_ir.ndim != 2:
        raise ValueError("c_ir must be [N, Q]")
    rho = scenario_posterior.reshape(-1)
    if rho.numel() != c_ir.shape[1]:
        raise ValueError("scenario_posterior must have Q entries")
    r_ir = -torch.log(c_ir.clamp_min(eps))
    node_risk = (rho.reshape(1, -1) * r_ir).sum(dim=1)
    return r_ir, node_risk


def neg_log_S_r(
    c_ir: torch.Tensor, *, eligible_mask: torch.Tensor | None = None, eps: float = 1e-6
) -> torch.Tensor:
    """Per-scenario global risk ``-log S_r(t) = sum_{i in H} r_{ir}(t)`` ``[Q]``.

    Exactly matches the mainline ``log S_r`` (same multiplicative floor, D4) so the §3.8
    decomposition identity holds to machine precision.
    """
    r_ir = -torch.log(c_ir.clamp_min(eps))
    if eligible_mask is not None:
        r_ir = r_ir[eligible_mask.to(torch.bool)]
    return r_ir.sum(dim=0)


def risk_emission(
    c_ir: torch.Tensor,
    scenario_posterior: torch.Tensor,
    *,
    eps: float = 1e-6,
    r_max: float | None = None,
) -> torch.Tensor:
    """Eq. 59 next-frame emission ``e_i = clip(sg[sum_r rho_r r_ir]/r_max, 0, 1)`` ``[N]``.

    The ``detach()`` is the stop-gradient ``sg``: the emission is a *feature* of the next
    frame, never a differentiable path back into the consensus that produced it.  The result
    is in ``[0,1]`` by construction.
    """
    _, node_risk = global_risk_contribution(c_ir, scenario_posterior, eps=eps)
    denom = -math.log(eps) if r_max is None else float(r_max)
    return (node_risk.detach() / denom).clamp(0.0, 1.0)


# --------------------------------------------------------------------------------------------
# Production temporal model: scalar emission feedback (faithful to Eq. 59).
#
# The ONLY quantity carried between frames is the bounded emission scalar e_i^{t-1} in [0,1],
# appended as a node-feature channel; the GNN is otherwise stateless.  Hence the inter-frame
# channel is provably bounded -- the honest, narrow version of the §3.8 design.  The emission
# is stop-gradient, so there is no back-propagation-through-time along the risk channel: each
# frame's loss trains that frame's forward pass given a detached emission input.
# --------------------------------------------------------------------------------------------


class ScalarEmissionRecurrentModel(nn.Module):
    """Temporal model with bounded scalar emission feedback (spec §3.8).

    Wraps a :class:`PreferenceConditionedTopologyGNN` whose node features are augmented with
    one extra channel carrying the previous frame's emission ``e_i^{t-1}``.
    """

    def __init__(self, static_node_dim: int, edge_dim: int, *, hidden: int = 32,
                 layers: int = 2, pref_dim: int = 3):
        super().__init__()
        self.static_node_dim = static_node_dim
        self.gnn = PreferenceConditionedTopologyGNN(
            static_node_dim + 1, edge_dim, hidden=hidden, layers=layers, pref_dim=pref_dim)

    def forward(
        self,
        graph: CandidateGraph,
        static_node_feat: torch.Tensor,
        edge_feat: torch.Tensor,
        lam: torch.Tensor,
        cfg: OperatingPointConfig,
        *,
        frames: int,
        emission_cfg: EmissionConfig | None = None,
        use_emission: bool = True,
    ) -> dict:
        """Run ``frames`` temporal frames; return per-frame operating points and emissions.

        ``use_emission=False`` zeroes the emission channel every frame (ablation control: a
        model that ignores the feedback), keeping the architecture identical.
        """
        if emission_cfg is None:
            emission_cfg = EmissionConfig()
        N = graph.num_nodes
        dtype = static_node_feat.dtype
        e_prev = torch.zeros(N, dtype=dtype)
        ops: list[dict] = []
        emissions: list[torch.Tensor] = []
        for _ in range(frames):
            chan = e_prev.unsqueeze(-1) if use_emission else torch.zeros(N, 1, dtype=dtype)
            nf = torch.cat([static_node_feat, chan], dim=-1)
            out = model_operating_point(self.gnn, graph, nf, edge_feat, lam, cfg)
            e_t = risk_emission(out["c_ir"], out["scenario_posterior"],
                                eps=emission_cfg.eps, r_max=emission_cfg.r_max)
            ops.append(out)
            emissions.append(e_t)
            e_prev = e_t  # detached already (stop-gradient): no BPTT through the risk channel
        return {"ops": ops, "emissions": emissions}


# --------------------------------------------------------------------------------------------
# Mechanism ablation for the bounded-scalar claim (spec §3.8: "must NOT claim a bounded scalar
# auto-constrains all hidden state -- verify or falsify by ablation, then correct").
#
# A *richer* recurrence carries a full per-node hidden VECTOR H_i^t in R^d, driven by the SAME
# bounded emission e_i^t in [0,1].  Varying only the recurrence cell shows whether the bounded
# input bounds the state.  This is the contrast the spec warns against conflating with the
# faithful scalar-feedback model above.
# --------------------------------------------------------------------------------------------


@dataclass
class RecurrenceState:
    H: torch.Tensor  # [N, d] hidden state


def recurrence_step(
    state: RecurrenceState,
    emission: torch.Tensor,  # [N] bounded in [0,1]
    proj: torch.Tensor,      # [1, d] fixed input projection
    *,
    kind: str,
    rho_scale: float = 1.3,
    gru: nn.GRUCell | None = None,
) -> RecurrenceState:
    """One recurrence step ``H^t = Cell(H^{t-1}, B e^t)`` with a bounded emission input.

    ``kind``:
      - ``"expansive"``  : ``H^t = rho_scale * H^{t-1} + B e^t`` with ``rho_scale > 1``
        (spectral radius > 1; non-contractive).
      - ``"contractive"``: same with ``rho_scale < 1`` (set via ``rho_scale``).
      - ``"gru"``        : a GRU cell (tanh-gated; state bounded by construction).
    The driving input ``B e^t`` is identical across kinds and bounded (``e in [0,1]``).
    """
    u = emission.reshape(-1, 1) * proj  # [N, d], bounded since e in [0,1]
    if kind == "gru":
        assert gru is not None
        H = gru(u, state.H)
    elif kind in ("expansive", "contractive"):
        H = rho_scale * state.H + u
    else:
        raise ValueError(f"unknown cell kind {kind!r}")
    return RecurrenceState(H=H)


def hidden_state_boundedness_ablation(
    emission_trajectory: list[torch.Tensor] | torch.Tensor,
    *,
    hidden_dim: int = 16,
    kinds: tuple[str, ...] = ("gru", "contractive", "expansive"),
    rho_scales: dict | None = None,
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
) -> dict:
    """Feed the SAME bounded emission sequence into a recurrent hidden state under different
    cells; record ``max_i ||H_i^t||`` per frame and the overall growth ratio.

    Returns a dict ``kind -> {"norms": [T], "growth_ratio": ||H^T|| / ||H^1||}``.  A bounded
    input that *auto-constrained* the state would keep every growth ratio ~O(1); the expansive
    cell's ratio blowing up is the falsification.
    """
    if isinstance(emission_trajectory, torch.Tensor):
        emission_trajectory = [emission_trajectory[t] for t in range(emission_trajectory.shape[0])]
    # all emissions must be bounded in [0,1] -- this is the hypothesis's premise
    for e in emission_trajectory:
        assert float(e.min()) >= -1e-12 and float(e.max()) <= 1.0 + 1e-9, "emission not in [0,1]"
    N = int(emission_trajectory[0].numel())
    rho_scales = rho_scales or {"contractive": 0.5, "expansive": 1.3}

    gen = torch.Generator().manual_seed(seed)
    proj = torch.randn(1, hidden_dim, generator=gen, dtype=dtype)
    # fixed GRU cell with frozen random weights (deterministic, seeded)
    gru = nn.GRUCell(hidden_dim, hidden_dim).to(dtype)
    with torch.no_grad():
        for p in gru.parameters():
            p.copy_(torch.randn(p.shape, generator=gen, dtype=dtype))
    for p in gru.parameters():
        p.requires_grad_(False)

    out: dict = {}
    for kind in kinds:
        H0 = torch.ones(N, hidden_dim, dtype=dtype)  # identical nonzero init
        state = RecurrenceState(H=H0)
        norms: list[float] = []
        rho = rho_scales.get(kind, 1.3)
        with torch.no_grad():
            for e in emission_trajectory:
                state = recurrence_step(state, e, proj, kind=kind, rho_scale=rho, gru=gru)
                norms.append(float(state.H.norm(dim=1).max()))
        ratio = norms[-1] / (norms[0] + 1e-12)
        out[kind] = {"norms": norms, "growth_ratio": ratio}
    return out
