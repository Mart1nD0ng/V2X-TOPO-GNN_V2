"""Exact small-``N`` joint protocol chain (spec §8.1 level 2 -- the ground truth).

For a tiny network this enumerates the EXACT joint Markov chain of the true binary-Snowball
process and returns the exact terminal reliability. It is the ground truth against which
the analytic episode (mean-field) and the independent dynamic MC are checked (the three-way
agreement of spec §8): the MC must match this within its confidence interval (it samples
the same true process), while the analytic may differ by its mean-field error.

The joint state is ``x = (x_1, ..., x_N)`` with ``x_i`` a per-node binary-Snowball state
(``src.protocol.binary_snowball`` layout, ``S`` states each). The chain is
TIME-HOMOGENEOUS under a fixed per-edge link reliability ``ell`` (we isolate the protocol
+ correlation layer; the physics is validated separately by the round-physics / FBL tests):
given the joint state ``x`` every node's current answer colour is fixed, so each node ``i``'s
ternary quorum distribution ``(h_i^+, h_i^-, h_i^0)`` is computed EXACTLY by the §5 quorum
DP with the polled peers' colours as ``0/1`` indicators, and -- because nodes poll
independently given ``x`` -- the joint transition factorises into a product of the per-node
Snowball transitions. The exact next distribution is therefore the ``p[x]``-weighted sum of
Kronecker products of per-node transition rows.

Exactness boundary. Exact for the defined process under a FIXED ``ell`` (ideal/fixed link)
and the finite horizon ``R_max``; feasible only for tiny ``N`` (cost ``O(R_max · S^{2N})``).
It deliberately holds the physics at a constant ``ell`` so the comparison isolates the
protocol/correlation approximation of the analytic episode -- run the analytic and MC with
``link_override=ell`` for an apples-to-apples three-way check.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import torch

from src.environment.candidate_graph import build_candidate_graph
from src.environment.canonical_episode import ProtocolConfig
from src.environment.evidence_model import EvidenceModel
from src.environment.urban_scene import ManhattanScene
from src.sampling.esp_query import edge_inclusion_probabilities
from src.mainline.quorum_dp import quorum_decision_probabilities

from .binary_snowball import snowball_layout, transition_matrix

__all__ = ["ExactJointResult", "exact_joint_terminal", "exact_joint_basin_first_hitting"]

_MAX_JOINT = 200_000  # cap on S^N


@dataclass(frozen=True)
class ExactJointResult:
    F_disagree: float
    F_wrong: float
    S_allcorrect: float
    num_joint_states: int


def _build_joint_chain(scene, evidence, query_policy, protocol_cfg, link_reliability, max_scenarios):
    """Shared exact-joint-chain setup: initial distribution + a one-round ``step(p)`` closure.

    Returns ``(p_init, step, decode_decided, N, layout)`` where ``decode_decided(flat) ->
    (per-node decided code in {+1 correct, -1 wrong, 0 undecided})`` and ``step`` advances the
    exact joint distribution one round (the ``p[x]``-weighted Kronecker product of per-node
    binary-Snowball transition rows under fixed link ``ell``). Isolates the protocol/correlation
    layer at a constant ``ell`` (the round physics is validated separately).
    """
    if not (0.0 <= float(link_reliability) <= 1.0):
        raise ValueError("link_reliability must be in [0, 1]")
    k, alpha, beta, r_max = protocol_cfg.k, protocol_cfg.alpha, protocol_cfg.beta, protocol_cfg.r_max
    N = scene.num_nodes
    layout = snowball_layout(beta, r_max)
    S = layout.state_count
    if S ** N > _MAX_JOINT:
        raise ValueError(f"S^N = {S}^{N} exceeds {_MAX_JOINT}; use smaller N / beta / r_max")

    gc = build_candidate_graph(scene.positions, scene.comm_radius)
    log_weights = query_policy.log_weights(gc).to(dtype=torch.float64)
    neighbours: list[list[tuple[int, float]]] = [[] for _ in range(N)]
    for e in range(gc.num_edges):
        neighbours[int(gc.src_index[e])].append((int(gc.dst_index[e]), float(log_weights[e])))
    for i in range(N):
        if len(neighbours[i]) < k:
            raise ValueError(f"node {i} has out-degree {len(neighbours[i])} < k={k} (§7.2 shortage)")

    is_plus = np.zeros(S, dtype=bool)
    for s in layout.pref_plus_idx:
        is_plus[s] = True
    dec_plus, dec_minus = layout.decided_plus, layout.decided_minus
    init_plus, init_minus = layout.initial_plus, layout.initial_minus
    ell = float(link_reliability)
    strides = [S ** (N - 1 - i) for i in range(N)]

    # ---- initial joint distribution over colour configs (exact, region-correlated) ----
    omega_s, init_cp = evidence.analytic_scenarios(max_scenarios=max_scenarios)
    omega_s = omega_s.to(torch.float64).numpy()
    init_cp = init_cp.to(torch.float64).numpy()
    p = np.zeros(S ** N, dtype=np.float64)
    for colours in product((1, 0), repeat=N):
        prob = 0.0
        for r in range(omega_s.shape[0]):
            f = omega_s[r]
            for i in range(N):
                f *= init_cp[i, r] if colours[i] == 1 else (1.0 - init_cp[i, r])
            prob += f
        if prob <= 0.0:
            continue
        idx = sum((init_plus if colours[i] == 1 else init_minus) * strides[i] for i in range(N))
        p[idx] += prob

    row_cache: dict[tuple[int, tuple[int, ...]], np.ndarray] = {}
    nb_index = [[nb for nb, _ in neighbours[i]] for i in range(N)]

    def node_transition(i: int, nb_colours: tuple[int, ...]) -> np.ndarray:
        key = (i, nb_colours)
        cached = row_cache.get(key)
        if cached is not None:
            return cached
        deg = len(neighbours[i])
        lw = torch.tensor([[neighbours[i][j][1] for j in range(deg)]], dtype=torch.float64)
        u = torch.tensor([[float(nb_colours[j]) for j in range(deg)]], dtype=torch.float64)
        p_plus = ell * u
        p_minus = ell * (1.0 - u)
        dec = quorum_decision_probabilities(lw, p_plus, p_minus, k, alpha)
        Tm = transition_matrix(dec.h_plus, dec.h_minus, dec.h_zero, layout)[0].numpy()
        row_cache[key] = Tm
        return Tm

    def step(p_cur: np.ndarray) -> np.ndarray:
        p_next = np.zeros_like(p_cur)
        nz = np.nonzero(p_cur > 1e-18)[0]
        for flat in nz:
            mass = p_cur[flat]
            rem = int(flat)
            xs = []
            for i in range(N):
                xs.append(rem // strides[i])
                rem = rem % strides[i]
            rows = []
            for i in range(N):
                nb_colours = tuple(int(is_plus[xs[nb]]) for nb in nb_index[i])
                rows.append(node_transition(i, nb_colours)[xs[i]])
            outer = rows[0] * mass
            for i in range(1, N):
                outer = np.multiply.outer(outer, rows[i]).reshape(-1)
            p_next += outer
        return p_next

    def decode_decided(flat: int) -> np.ndarray:
        rem = int(flat)
        out = np.zeros(N, dtype=np.int64)
        for i in range(N):
            xi = rem // strides[i]
            rem = rem % strides[i]
            if xi == dec_plus:
                out[i] = 1
            elif xi == dec_minus:
                out[i] = -1
        return out

    return p, step, decode_decided, N, layout


def exact_joint_basin_first_hitting(
    scene: ManhattanScene,
    evidence: EvidenceModel,
    query_policy,
    protocol_cfg: ProtocolConfig,
    *,
    link_reliability: float,
    profile,
    omega: torch.Tensor | None = None,
    max_scenarios: int = 1 << 16,
) -> dict:
    """EXACT participation-weighted macrostate basin first-hitting probabilities (spec §4, §12 L1).

    Power-iterates the exact joint chain; at each epoch ``r = 0 … R_d`` the live mass whose
    macrostate ``(C_r, W_r)`` has JUST entered a basin is absorbed into that outcome (basins are
    monotone-absorbing in this decided-mass-only process, and disjoint, so the FIRST hit is
    unambiguous). Mass surviving past ``R_d`` is a deadline miss. The four probabilities sum to 1
    exactly (mass conservation). This is the Level-1 ground truth the dynamic MC (Level 3) must
    match within its confidence interval.
    """
    from src.metrics.basins import CORRECT, WRONG, SPLIT

    p, step, decode_decided, N, _ = _build_joint_chain(
        scene, evidence, query_policy, protocol_cfg, link_reliability, max_scenarios)

    if omega is None:
        w = np.full(N, 1.0 / N, dtype=np.float64)
    else:
        w = omega.to(torch.float64).cpu().numpy().reshape(N)
    rho_f = profile.correct_basin_mass
    rho_s = profile.split_basin_mass
    R_d = profile.max_poll_epochs

    absorbed = {CORRECT: 0.0, WRONG: 0.0, SPLIT: 0.0}
    tau_correct_mass = 0.0  # sum of mass*tau over correct hits (for mean T_confirm in epochs)

    for r in range(R_d + 1):
        nz = np.nonzero(p > 1e-18)[0]
        for flat in nz:
            mass = float(p[flat])
            dec = decode_decided(int(flat))
            C = float(w[dec == 1].sum())
            W = float(w[dec == -1].sum())
            in_c = C >= rho_f
            in_w = W >= rho_f
            in_s = (C >= rho_s) and (W >= rho_s)
            if in_c:
                absorbed[CORRECT] += mass
                tau_correct_mass += mass * r
                p[flat] = 0.0
            elif in_w:
                absorbed[WRONG] += mass
                p[flat] = 0.0
            elif in_s:
                absorbed[SPLIT] += mass
                p[flat] = 0.0
        if r < R_d:
            p = step(p)

    deadline = float(p[p > 1e-18].sum())
    pc = absorbed[CORRECT]
    return {
        "P_correct": pc,
        "F_wrong": absorbed[WRONG],
        "F_split": absorbed[SPLIT],
        "F_deadline": deadline,
        "tau_correct_mean": (tau_correct_mass / pc) if pc > 1e-15 else float("nan"),
    }


def exact_joint_terminal(
    scene: ManhattanScene,
    evidence: EvidenceModel,
    query_policy,
    protocol_cfg: ProtocolConfig,
    *,
    link_reliability: float,
    eligible_mask: torch.Tensor | None = None,
    max_scenarios: int = 1 << 16,
) -> ExactJointResult:
    """Exact terminal reliability of the joint chain under a fixed link ``ell`` (see module).

    Compare with ``run_consensus_episode(..., link_override=link_reliability)`` and
    ``run_dynamic_mc(..., link_override=link_reliability)`` for the three-way agreement.
    """
    if not (0.0 <= float(link_reliability) <= 1.0):
        raise ValueError("link_reliability must be in [0, 1]")
    k, alpha, beta, r_max = protocol_cfg.k, protocol_cfg.alpha, protocol_cfg.beta, protocol_cfg.r_max
    N = scene.num_nodes
    layout = snowball_layout(beta, r_max)
    S = layout.state_count
    if S ** N > _MAX_JOINT:
        raise ValueError(f"S^N = {S}^{N} exceeds {_MAX_JOINT}; use smaller N / beta / r_max")

    gc = build_candidate_graph(scene.positions, scene.comm_radius)
    log_weights = query_policy.log_weights(gc).to(dtype=torch.float64)
    # per-source neighbour adjacency (dst node + log-weight)
    neighbours: list[list[tuple[int, float]]] = [[] for _ in range(N)]
    for e in range(gc.num_edges):
        neighbours[int(gc.src_index[e])].append((int(gc.dst_index[e]), float(log_weights[e])))
    for i in range(N):
        if len(neighbours[i]) < k:
            raise ValueError(f"node {i} has out-degree {len(neighbours[i])} < k={k} (§7.2 shortage)")

    is_plus = np.zeros(S, dtype=bool)
    for s in layout.pref_plus_idx:
        is_plus[s] = True
    dec_plus, dec_minus = layout.decided_plus, layout.decided_minus
    init_plus, init_minus = layout.initial_plus, layout.initial_minus
    ell = float(link_reliability)

    # ---- initial joint distribution over colour configs (exact, with region correlation) ----
    omega, init_cp = evidence.analytic_scenarios(max_scenarios=max_scenarios)
    omega = omega.to(torch.float64).numpy()
    init_cp = init_cp.to(torch.float64).numpy()                     # [N, Q]
    p = np.zeros(S ** N, dtype=np.float64)
    strides = [S ** (N - 1 - i) for i in range(N)]
    for colours in product((1, 0), repeat=N):                      # 1 = correct(+), 0 = wrong(-)
        # P(colour config) = sum_r omega_r prod_i (init_cp if + else 1-init_cp)
        prob = 0.0
        for r in range(omega.shape[0]):
            f = omega[r]
            for i in range(N):
                f *= init_cp[i, r] if colours[i] == 1 else (1.0 - init_cp[i, r])
            prob += f
        if prob <= 0.0:
            continue
        idx = sum((init_plus if colours[i] == 1 else init_minus) * strides[i] for i in range(N))
        p[idx] += prob

    # ---- per-node transition rows cached by (node, neighbour-colour-config) ----
    row_cache: dict[tuple[int, tuple[int, ...]], np.ndarray] = {}

    def node_transition(i: int, nb_colours: tuple[int, ...]) -> np.ndarray:
        key = (i, nb_colours)
        cached = row_cache.get(key)
        if cached is not None:
            return cached
        deg = len(neighbours[i])
        lw = torch.tensor([[neighbours[i][j][1] for j in range(deg)]], dtype=torch.float64)
        u = torch.tensor([[float(nb_colours[j]) for j in range(deg)]], dtype=torch.float64)  # peer + indicator
        p_plus = ell * u
        p_minus = ell * (1.0 - u)
        dec = quorum_decision_probabilities(lw, p_plus, p_minus, k, alpha)
        T = transition_matrix(dec.h_plus, dec.h_minus, dec.h_zero, layout)[0].numpy()  # [S, S]
        row_cache[key] = T
        return T

    # ---- power-iterate the exact joint chain R_max rounds ----
    nb_index = [[nb for nb, _ in neighbours[i]] for i in range(N)]
    for _ in range(r_max):
        p_next = np.zeros_like(p)
        nz = np.nonzero(p > 1e-18)[0]
        for flat in nz:
            mass = p[flat]
            # decode joint state
            xs = []
            rem = int(flat)
            for i in range(N):
                xs.append(rem // strides[i])
                rem = rem % strides[i]
            # per-node next-state row (identity if decided -> absorbing already in T)
            rows = []
            for i in range(N):
                nb_colours = tuple(int(is_plus[xs[nb]]) for nb in nb_index[i])
                T = node_transition(i, nb_colours)
                rows.append(T[xs[i]])                               # [S]
            # joint next = mass * kron(rows...)
            outer = rows[0] * mass
            for i in range(1, N):
                outer = np.multiply.outer(outer, rows[i]).reshape(-1)
            p_next += outer
        p = p_next

    # ---- exact terminal reliability over the eligible set ----
    if eligible_mask is None:
        elig = [True] * N
    else:
        elig = [bool(x) for x in eligible_mask.tolist()]
    S_all = F_wrong = F_dis = 0.0
    nz = np.nonzero(p > 1e-18)[0]
    for flat in nz:
        mass = float(p[flat])
        rem = int(flat)
        any_plus = any_minus = False
        all_plus = True
        for i in range(N):
            xi = rem // strides[i]
            rem = rem % strides[i]
            if not elig[i]:
                continue
            if xi == dec_plus:
                any_plus = True
            elif xi == dec_minus:
                any_minus = True
                all_plus = False
            else:  # undecided
                all_plus = False
        if all_plus:
            S_all += mass
        if any_minus:
            F_wrong += mass
        if any_plus and any_minus:
            F_dis += mass

    return ExactJointResult(F_disagree=F_dis, F_wrong=F_wrong, S_allcorrect=S_all,
                            num_joint_states=int(S ** N))
