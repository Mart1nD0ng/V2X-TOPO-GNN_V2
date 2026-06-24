"""Independent dynamic Monte-Carlo judge (spec §8.1 level 3, §8.3).

A genuine round-by-round FORWARD simulation of the consensus process -- the independent
external validator of the analytic episode. Per trial it:

1. samples the evidence realisation (region bits ``B_g``, node errors ``E_i``) -> each
   node's initial preference colour (``+`` if its observation is correct, else ``-``);
2. each round, for every still-undecided node, samples a ``k``-subset of its candidate
   peers from the SAME query policy (the exact ancestral ESP sampler);
3. samples each poll's success ~ ``Bern(ell_poll)`` (the fading-marginalised request AND
   response delivery from the round physics) and reads the polled peer's **actual current
   colour** (its true preference / decided colour -- NOT a marginal);
4. forms the ternary quorum (``+`` if >= alpha correct votes, ``-`` if >= alpha wrong,
   else no quorum) and advances the node's **true binary-Snowball** counters
   (``d, pref, last, c, decided``) exactly as ``src.protocol.binary_snowball._step``.

It then reports empirical agreement-safety / validity / all-correct frequencies (with
confidence intervals), per-node decided frequencies and finalisation latency.

Independence (constraint #8). The MC NEVER reads the analytic terminal marginals
``c_ir/w_ir`` nor the analytic shared-latent decomposition; it samples the joint forward
process and so captures the inter-node correlations the analytic mean-field approximates.
What it legitimately SHARES with the analytic episode is the *system definition*: the same
query policy and the same physical link model. ``Bern(ell_poll)`` is the exact
fading-marginalised poll outcome (``P(success|H)=ell(H)`` integrated over fading gives a
Bernoulli with mean ``ell``), so fading need not be sampled separately.

Physics fidelity knob. ``physics_per_trial=False`` (default) evaluates ``ell`` once per
round at the MC's OWN empirical mean active mass (a within-MC mean-field over the sampled
states -- still independent of the analytic, and it isolates the *protocol* mean-field when
comparing to the analytic). ``physics_per_trial=True`` evaluates ``ell`` per trial from
each trial's sampled active set, additionally exposing physics nonlinearity (slower).

Complexity: ``O(R · num_trials · (N · max_deg + E_phys))``; no ``N x N`` tensor.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.mainline.global_evaluator import build_source_padding
from src.protocol.binary_snowball import snowball_layout  # for r_max/beta validation parity
from src.sampling.esp_query import edge_inclusion_probabilities

from src.environment.candidate_graph import build_candidate_graph
from src.environment.canonical_episode import ProtocolConfig
from src.environment.evidence_model import EvidenceModel
from src.environment.interference_graph import build_interference_graph
from src.environment.round_physics import RoundPhysicsConfig, edge_geometry, round_physics
from src.environment.urban_scene import ManhattanScene

__all__ = ["DynamicMCResult", "run_dynamic_mc"]


@dataclass(frozen=True)
class DynamicMCResult:
    F_disagree: float
    F_wrong: float
    S_allcorrect: float
    F_disagree_ci: tuple[float, float]
    F_wrong_ci: tuple[float, float]
    S_allcorrect_ci: tuple[float, float]
    decided_correct_freq: torch.Tensor   # [N]
    decided_wrong_freq: torch.Tensor     # [N]
    undecided_freq: torch.Tensor         # [N]
    mean_rounds_to_decide: float         # over finalised (node,trial)
    mean_finalisation_time: float        # mean wall-clock to all-eligible-correct (finished trials)
    finished_fraction: float             # fraction of trials with all eligible decided correct
    num_trials: int


def _wilson_ci(freq: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = freq
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def run_dynamic_mc(
    scene: ManhattanScene,
    evidence: EvidenceModel,
    query_policy,
    protocol_cfg: ProtocolConfig,
    phy_cfg: RoundPhysicsConfig,
    *,
    num_trials: int = 4000,
    generator: torch.Generator | None = None,
    eligible_mask: torch.Tensor | None = None,
    physics_per_trial: bool = False,
    dtype: torch.dtype = torch.float64,
) -> DynamicMCResult:
    """Run the independent dynamic Monte-Carlo (spec §8). See module docstring."""
    if evidence.num_nodes != scene.num_nodes:
        raise ValueError("evidence and scene must have the same N")
    k, alpha, beta, r_max = protocol_cfg.k, protocol_cfg.alpha, protocol_cfg.beta, protocol_cfg.r_max
    snowball_layout(beta, r_max)  # validates (beta, r_max) the same way the analytic does
    N = scene.num_nodes
    T = int(num_trials)
    device = scene.positions.device
    if generator is None:
        generator = torch.Generator(device=device)

    gc = build_candidate_graph(scene.positions, scene.comm_radius)
    gi = build_interference_graph(scene.positions, scene.int_radius)
    geom_c = edge_geometry(gc, phy_cfg)
    geom_i = edge_geometry(gi, phy_cfg)

    # query policy -> log-weights -> inclusion pi (same policy as deployment/analytic, #3)
    log_weights = query_policy.log_weights(gc).to(device=device, dtype=dtype)
    pi = edge_inclusion_probabilities(gc.src_index, gc.dst_index, N, log_weights, k)

    # source padding for per-(trial,node) subset sampling
    pad = build_source_padding(gc.src_index, gc.dst_index, N)
    nmax = pad.max_deg
    if bool(torch.any(pad.out_degree < k).cpu()):
        raise ValueError("a source has out-degree < k; apply the §7.2 shortage protocol upstream")
    slot_edge = pad.slot_edge                              # [N, nmax] edge id (0 where invalid)
    slot_mask = pad.slot_mask                              # [N, nmax] bool
    slot_dst = gc.dst_index[slot_edge]                    # [N, nmax] peer node id
    neg = torch.full((), float("-inf"), dtype=dtype, device=device)
    slot_logw = torch.where(slot_mask, log_weights[slot_edge], neg)  # [N, nmax]

    if eligible_mask is None:
        eligible_mask = torch.ones(N, dtype=torch.bool, device=device)
    elig = eligible_mask.to(device=device, dtype=torch.bool)

    # ---- sample evidence per trial (independent of the analytic decomposition) ----
    ev = evidence.sample(T, generator=generator, device=device)   # correct [T, N]
    pref = torch.where(ev.correct, 1, -1).to(torch.int64)         # [T, N] current colour
    d = torch.zeros((T, N), dtype=torch.int64, device=device)
    last = torch.zeros((T, N), dtype=torch.int64, device=device)
    cnt = torch.zeros((T, N), dtype=torch.int64, device=device)
    decided = torch.zeros((T, N), dtype=torch.int64, device=device)   # 0 / +1 / -1
    rounds_to_decide = torch.full((T, N), r_max + 1, dtype=torch.int64, device=device)
    cumulative_time = torch.zeros(T, dtype=dtype, device=device)

    from src.protocol.binary_snowball import PLUS, MINUS  # +1, -1

    for t_round in range(1, r_max + 1):
        active = decided == 0                              # [T, N] bool

        # ---- physics: ell_poll for this round ----
        if physics_per_trial:
            active_phys = active.to(dtype).transpose(0, 1)            # [N, T]
        else:
            active_phys = active.to(dtype).mean(dim=0, keepdim=True).transpose(0, 1)  # [N, 1] mean active
        phys = round_physics(gc, gi, pi, active_phys, phy_cfg, geom_comm=geom_c, geom_int=geom_i)
        ell = phys.ell_poll                                # [E, Bphys]
        ell_slot = ell[slot_edge]                          # [N, nmax, Bphys]
        if physics_per_trial:
            ell_slot = ell_slot.permute(2, 0, 1)           # [T, N, nmax]
        else:
            ell_slot = ell_slot.squeeze(-1).unsqueeze(0)   # [1, N, nmax] -> broadcast

        # ---- sample k-subsets for every (trial, node) from the SAME ESP sampler ----
        chosen = _sample_subsets(slot_logw, slot_mask, k, T, generator)   # [T, N, nmax] bool

        # ---- poll outcomes: success ~ Bern(ell), read peer's ACTUAL colour ----
        peer_colour = pref[:, slot_dst]                    # [T, N, nmax] in {+1,-1}
        u = torch.rand((T, N, nmax), generator=generator, device=device, dtype=dtype)
        responded = chosen & slot_mask.unsqueeze(0) & (u < ell_slot)
        votes_plus = (responded & (peer_colour == PLUS)).sum(dim=-1)    # [T, N]
        votes_minus = (responded & (peer_colour == MINUS)).sum(dim=-1)  # [T, N]

        # ---- ternary quorum (strict majority 2alpha>k -> mutually exclusive) ----
        o = torch.zeros((T, N), dtype=torch.int64, device=device)
        o = torch.where(votes_plus >= alpha, torch.ones_like(o), o)
        o = torch.where(votes_minus >= alpha, -torch.ones_like(o), o)

        # ---- advance TRUE binary-Snowball counters (vectorised _step) ----
        o0 = active & (o == 0)
        onz = active & (o != 0)
        # no-quorum: break streak, confidence persists
        last = torch.where(o0, torch.zeros_like(last), last)
        cnt = torch.where(o0, torch.zeros_like(cnt), cnt)
        # quorum: update confidence, preference, streak
        d_new = (d + o).clamp(-r_max, r_max)
        d = torch.where(onz, d_new, d)
        pref_dir = torch.where(d > 0, torch.ones_like(pref),
                               torch.where(d < 0, -torch.ones_like(pref), pref))
        pref = torch.where(onz, pref_dir, pref)
        cnt_inc = torch.where(last == o, cnt + 1, torch.ones_like(cnt))
        cnt = torch.where(onz, cnt_inc, cnt)
        last = torch.where(onz, o, last)
        newly = onz & (cnt >= beta) & (decided == 0)
        decided = torch.where(newly, pref, decided)
        rounds_to_decide = torch.where(newly, torch.full_like(rounds_to_decide, t_round), rounds_to_decide)

        # ---- wall-clock: trials not yet all-eligible-correct accrue this round's duration ----
        all_correct_now = ((decided == 1) | ~elig.unsqueeze(0)).all(dim=1)   # [T]
        running = ~all_correct_now
        # network round duration = slowest active node's tau (shared physics -> scalar/round)
        tau = phys.tau                                       # [N, Bphys]
        round_dur = tau.max()                                # conservative scalar duration
        cumulative_time = cumulative_time + running.to(dtype) * round_dur

    # ---- terminal statistics ----
    dec = decided
    elig_b = elig.unsqueeze(0)
    is_corr = dec == 1
    is_wrong = dec == -1
    all_correct = (is_corr | ~elig_b).all(dim=1)                       # [T]
    any_wrong = (is_wrong & elig_b).any(dim=1)
    any_corr = (is_corr & elig_b).any(dim=1)
    disagree = any_wrong & any_corr                                     # both colours decided

    S_allcorrect = float(all_correct.to(dtype).mean())
    F_wrong = float(any_wrong.to(dtype).mean())
    F_disagree = float(disagree.to(dtype).mean())

    decided_correct_freq = (is_corr.to(dtype)).mean(dim=0)             # [N]
    decided_wrong_freq = (is_wrong.to(dtype)).mean(dim=0)
    undecided_freq = ((dec == 0).to(dtype)).mean(dim=0)

    finalised = rounds_to_decide <= r_max
    mean_rounds = float(rounds_to_decide[finalised].to(dtype).mean()) if bool(finalised.any()) else float("nan")
    finished = all_correct
    mean_time = float(cumulative_time[finished].mean()) if bool(finished.any()) else float("nan")

    return DynamicMCResult(
        F_disagree=F_disagree, F_wrong=F_wrong, S_allcorrect=S_allcorrect,
        F_disagree_ci=_wilson_ci(F_disagree, T), F_wrong_ci=_wilson_ci(F_wrong, T),
        S_allcorrect_ci=_wilson_ci(S_allcorrect, T),
        decided_correct_freq=decided_correct_freq, decided_wrong_freq=decided_wrong_freq,
        undecided_freq=undecided_freq, mean_rounds_to_decide=mean_rounds,
        mean_finalisation_time=mean_time, finished_fraction=float(finished.to(dtype).mean()),
        num_trials=T,
    )


def _sample_subsets(slot_logw: torch.Tensor, slot_mask: torch.Tensor, k: int, T: int,
                    generator: torch.Generator) -> torch.Tensor:
    """Sample a ``k``-subset per (trial, node) with the exact ESP ancestral sampler.

    Returns ``[T, N, nmax]`` boolean (exactly ``k`` True per (trial,node) row over valid
    slots). The same policy weights are reused across trials; only the random draws differ.
    """
    from src.mainline.symmetric_polynomials import sample_k_subset
    N, nmax = slot_logw.shape
    lw = slot_logw.unsqueeze(0).expand(T, N, nmax).reshape(T * N, nmax)
    mk = slot_mask.unsqueeze(0).expand(T, N, nmax).reshape(T * N, nmax)
    chosen = sample_k_subset(lw, k, mask=mk, generator=generator)      # [T*N, nmax]
    return chosen.reshape(T, N, nmax)
