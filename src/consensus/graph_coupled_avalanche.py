from __future__ import annotations

from typing import Any

import torch

import numpy as np

from .avalanche_closed_form import PROBABILITY_TOLERANCE, quorum_success_probability
from .topology_query_support import compute_topology_query_support


def _validate_parameters(k: int, alpha: int, beta: int, rounds: int, eps: float, temperature: float) -> None:
    if k < 1:
        raise ValueError("k must be positive")
    if alpha < 1 or alpha > k:
        raise ValueError("alpha must satisfy 1 <= alpha <= k")
    if 2 * alpha <= k:
        raise ValueError("alpha must be a strict majority: 2 * alpha > k")
    if beta < 1:
        raise ValueError("beta must be positive")
    if rounds < 0:
        raise ValueError("rounds must be nonnegative")
    if eps < 0.0 or eps >= 0.5:
        raise ValueError("eps must satisfy 0 <= eps < 0.5")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")


def _as_node_probability(name: str, value: torch.Tensor, num_nodes: int, reference: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    result = value.to(dtype=reference.dtype, device=reference.device).reshape(-1)
    if result.numel() != num_nodes:
        raise ValueError(f"{name} must contain num_nodes values")
    detached = result.detach()
    if bool(torch.any(~torch.isfinite(detached)).cpu()):
        raise ValueError(f"{name} must contain only finite probabilities")
    if bool(torch.any(detached < -PROBABILITY_TOLERANCE).cpu()) or bool(
        torch.any(detached > 1.0 + PROBABILITY_TOLERANCE).cpu()
    ):
        raise ValueError(f"{name} must be in [0, 1]")
    return torch.clamp(result, 0.0, 1.0)


def _checked_probability(name: str, value: torch.Tensor) -> torch.Tensor:
    detached = value.detach()
    if bool(torch.any(detached < -PROBABILITY_TOLERANCE).cpu()) or bool(
        torch.any(detached > 1.0 + PROBABILITY_TOLERANCE).cpu()
    ):
        raise ValueError(f"{name} must be in [0, 1]")
    return torch.clamp(value, 0.0, 1.0)


def _coupled_query_support(
    *,
    num_nodes: int,
    src_index: torch.Tensor,
    dst_index: torch.Tensor,
    normalized_query_weight: torch.Tensor,
    link_success: torch.Tensor,
    correct_preference: torch.Tensor,
    wrong_preference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    base = correct_preference.new_zeros((num_nodes,))
    response_weight = normalized_query_weight * link_success
    correct_terms = response_weight * correct_preference[dst_index]
    wrong_terms = response_weight * wrong_preference[dst_index]
    p_correct = base.index_add(0, src_index, correct_terms)
    p_wrong = base.index_add(0, src_index, wrong_terms)
    support_sum = p_correct + p_wrong
    if bool(torch.any(support_sum.detach() > 1.0 + PROBABILITY_TOLERANCE).cpu()):
        raise ValueError("topology-coupled query support exceeds one")
    return _checked_probability("p_correct_query", p_correct), _checked_probability("p_wrong_query", p_wrong)


def _state_mass(
    correct_states: torch.Tensor,
    wrong_states: torch.Tensor,
    correct_absorbed: torch.Tensor,
    wrong_absorbed: torch.Tensor,
    undecided_neutral: torch.Tensor,
) -> torch.Tensor:
    return (
        correct_states.sum(dim=1)
        + wrong_states.sum(dim=1)
        + correct_absorbed
        + wrong_absorbed
        + undecided_neutral
    )


def _transition_states(
    *,
    correct_states: torch.Tensor,
    wrong_states: torch.Tensor,
    correct_absorbed: torch.Tensor,
    wrong_absorbed: torch.Tensor,
    undecided_neutral: torch.Tensor,
    h_plus: torch.Tensor,
    h_minus: torch.Tensor,
    h_zero: torch.Tensor,
    beta: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    correct_sum = correct_states.sum(dim=1)
    wrong_sum = wrong_states.sum(dim=1)
    if beta == 1:
        transient_mass = correct_sum + wrong_sum + undecided_neutral
        next_correct_absorbed = correct_absorbed + h_plus * transient_mass
        next_wrong_absorbed = wrong_absorbed + h_minus * transient_mass
        next_correct = correct_states.new_zeros(correct_states.shape)
        next_wrong = wrong_states.new_zeros(wrong_states.shape)
        next_correct[:, 0] = h_zero * correct_sum
        next_wrong[:, 0] = h_zero * wrong_sum
        next_neutral = h_zero * undecided_neutral
        return next_correct, next_wrong, next_correct_absorbed, next_wrong_absorbed, next_neutral

    next_correct = correct_states.new_zeros(correct_states.shape)
    next_wrong = wrong_states.new_zeros(wrong_states.shape)
    next_correct_absorbed = correct_absorbed + h_plus * correct_states[:, beta - 1]
    next_wrong_absorbed = wrong_absorbed + h_minus * wrong_states[:, beta - 1]

    next_correct[:, 0] = h_zero * correct_sum
    next_wrong[:, 0] = h_zero * wrong_sum
    next_neutral = h_zero * undecided_neutral

    next_correct[:, 1] = h_plus * correct_states[:, 0] + h_plus * wrong_sum + h_plus * undecided_neutral
    next_wrong[:, 1] = h_minus * wrong_states[:, 0] + h_minus * correct_sum + h_minus * undecided_neutral
    for count in range(2, beta):
        next_correct[:, count] = h_plus * correct_states[:, count - 1]
        next_wrong[:, count] = h_minus * wrong_states[:, count - 1]
    return next_correct, next_wrong, next_correct_absorbed, next_wrong_absorbed, next_neutral


def _gauss_hermite_normal(num_points: int, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Gauss-Hermite nodes/weights for the STANDARD-NORMAL measure: E_{z~N(0,1)}[f(z)] ~= sum_q w_q f(z_q),
    with sum_q w_q = 1. Deterministic, no sampling."""
    nodes, weights = np.polynomial.hermite_e.hermegauss(int(num_points))
    weights = weights / weights.sum()
    return (
        torch.as_tensor(nodes, dtype=dtype, device=device),
        torch.as_tensor(weights, dtype=dtype, device=device),
    )


def _network_fraction_tail(
    p_correct: torch.Tensor, tau: float, var_floor: float, reference: torch.Tensor
) -> torch.Tensor:
    """MACRO graph-level reliability: ``P[ network fraction-correct < tau ]`` — the probability that
    fewer than ``tau`` of the nodes reach correct consensus. With ``Y = (1/N) sum_i 1{i correct}``,
    ``E[Y] = (1/N) sum p_i`` and (independent-decision) ``Var[Y] = (1/N^2) sum p_i(1-p_i)``; the tail is
    the Normal (Poisson-binomial) moment-match ``P[Y<tau] = 0.5*erfc((E[Y]-tau)/sqrt(2 Var[Y]))``.

    This is a JOINT GRAPH-LEVEL quantity (the whole-network correct-fraction tail), not a per-node
    marginal average nor a single round — satisfying the macro requirement. Differentiable; a variance
    floor keeps the gradient finite when the fraction concentrates at large N. The independent-decision
    variance omits cross-node decision correlation, so the tail is sharper than the simulation when whole
    clusters co-fail (the mean E[Y] is faithful; see docs/CLOSED_FORM_FIDELITY.md).
    """
    n = int(p_correct.numel())
    if n == 0:
        return reference.new_tensor(0.0)
    p = torch.clamp(p_correct.reshape(-1), 0.0, 1.0)
    mean_frac = p.mean()
    var_frac = torch.clamp((p * (1.0 - p)).sum() / float(n * n), min=float(var_floor))
    z = (mean_frac - float(tau)) / torch.sqrt(2.0 * var_frac)
    return torch.clamp(0.5 * torch.erfc(z), 0.0, 1.0)


def _evaluate_quenched_avalanche(
    *,
    num_nodes: int,
    support: Any,
    link_values: torch.Tensor,
    initial_correct: torch.Tensor,
    initial_wrong: torch.Tensor,
    k: int,
    alpha: int,
    beta: int,
    rounds: int,
    quenched_quadrature: int,
    network_tail_tau: float | None,
    network_tail_var_floor: float,
    eps: float,
    temperature: float,
    reference: torch.Tensor,
) -> dict[str, Any]:
    """SSMC quenched-disorder closed form. Each node carries ``Q = quenched_quadrature`` PERSISTENT
    Gauss-Hermite copies along the principal axis of the FROZEN seed-disorder covariance of its
    realized support ``(c_correct, c_wrong)``:

        s_c  = sum_j rw_ij^2 * u_c_j (1 - u_c_j)        (correct-support variance)
        s_w  = sum_j rw_ij^2 * u_w_j (1 - u_w_j)        (wrong-support variance)
        s_cw = -sum_j rw_ij^2 * u_c_j * u_w_j           (correct/wrong anti-covariance)

    The copy means evolve (cross-copy weighted mean -> the corrected, lower marginal); the disorder
    loadings stay frozen at the seed level, so an unlucky neighbourhood draw PERSISTS across all
    rounds (the quenched effect the mean-field/annealed closure erases). ``Q = 1`` reduces to the
    mean-field evaluator. Fully analytic and differentiable, no sampling; the network reliability is
    the disorder-averaged joint outcome (a macro quantity), not a per-node marginal.
    """
    Q = int(quenched_quadrature)
    src = support.src_index
    dst = support.dst_index
    rw = support.normalized_query_weight * link_values  # response weight per edge
    base = reference.new_zeros((num_nodes,))
    if num_nodes == 0:
        zero = reference.new_tensor(0.0)
        empty = reference.new_zeros((0,))
        return {
            "query_support": support,
            "node_p_correct_decision": empty,
            "node_p_wrong_decision": empty,
            "node_p_undecided": empty,
            "node_expected_rounds": empty,
            "C_avalanche_node_mean": zero,
            "D_avalanche_rounds_mean": zero,
            "graph_metrics": {},
            "quenched_quadrature": Q,
        }

    ic = initial_correct
    iw = initial_wrong
    rw2 = rw * rw
    s_c = base.index_add(0, src, rw2 * ic[dst] * (1.0 - ic[dst]))
    s_w = base.index_add(0, src, rw2 * iw[dst] * (1.0 - iw[dst]))
    s_cw = base.index_add(0, src, -rw2 * ic[dst] * iw[dst])
    trace = s_c + s_w
    det = s_c * s_w - s_cw * s_cw
    # NaN-safe sqrt guards (audit/roadmap fix): when the 2x2 covariance degenerates (link success
    # -> 0 on a node's whole query set, e.g. sparse/low-density cells), trace^2-4det, the
    # eigenvector norm, and lam1 all hit EXACTLY 0; sqrt(0) has an infinite backward gradient and
    # clamp's zero-gradient multiplies it into inf*0 = NaN, poisoning the whole model in one step.
    # clamp_min(_SQRT_EPS) keeps the forward shift at sqrt(1e-24)=1e-12 (orders below every test
    # tolerance, exactly 0-cells only) while making the backward finite.
    _SQRT_EPS = 1e-24
    disc = torch.clamp(trace * trace - 4.0 * det, min=_SQRT_EPS).sqrt()
    lam1 = 0.5 * (trace + disc)  # principal (largest) eigenvalue of the 2x2 covariance
    # principal eigenvector, chosen from the two algebraic forms by larger norm (numerically robust)
    v1_c, v1_w = s_cw, lam1 - s_c
    v2_c, v2_w = lam1 - s_w, s_cw
    use_first = (v1_c * v1_c + v1_w * v1_w) >= (v2_c * v2_c + v2_w * v2_w)
    e_c = torch.where(use_first, v1_c, v2_c)
    e_w = torch.where(use_first, v1_w, v2_w)
    norm = torch.clamp(e_c * e_c + e_w * e_w, min=_SQRT_EPS).sqrt()
    degenerate = norm <= 1e-12
    e_c = torch.where(degenerate, torch.ones_like(e_c), e_c) / torch.clamp(norm, min=1e-18)
    e_w = torch.where(degenerate, torch.zeros_like(e_w), e_w) / torch.clamp(norm, min=1e-18)
    std1 = torch.clamp(lam1, min=_SQRT_EPS).sqrt()
    load_c = std1 * e_c  # [N] persistent disorder loading on correct support
    load_w = std1 * e_w

    z, w = _gauss_hermite_normal(Q, reference.dtype, reference.device)  # [Q]
    w_row = w.unsqueeze(0)  # [1, Q]
    off_c = load_c.unsqueeze(1) * z.unsqueeze(0)  # [N, Q]  frozen support offsets
    off_w = load_w.unsqueeze(1) * z.unsqueeze(0)

    rep = Q
    correct_states = reference.new_zeros((num_nodes * Q, beta))
    wrong_states = reference.new_zeros((num_nodes * Q, beta))
    correct_states[:, 0] = ic.repeat_interleave(rep)
    wrong_states[:, 0] = iw.repeat_interleave(rep)
    correct_absorbed = reference.new_zeros((num_nodes * Q,))
    wrong_absorbed = reference.new_zeros((num_nodes * Q,))
    undecided_neutral = torch.clamp(1.0 - ic.repeat_interleave(rep) - iw.repeat_interleave(rep), 0.0, 1.0)
    expected_rounds = reference.new_zeros((num_nodes * Q,))

    for _ in range(rounds):
        p_correct_copy = (correct_absorbed + correct_states.sum(dim=1)).reshape(num_nodes, Q)
        p_wrong_copy = (wrong_absorbed + wrong_states.sum(dim=1)).reshape(num_nodes, Q)
        m_correct = (p_correct_copy * w_row).sum(dim=1)  # corrected marginals [N]
        m_wrong = (p_wrong_copy * w_row).sum(dim=1)
        mu_correct = base.index_add(0, src, rw * m_correct[dst])
        mu_wrong = base.index_add(0, src, rw * m_wrong[dst])
        c_correct = torch.clamp(mu_correct.unsqueeze(1) + off_c, eps, 1.0 - eps)  # [N, Q]
        c_wrong = torch.clamp(mu_wrong.unsqueeze(1) + off_w, eps, 1.0 - eps)
        c_wrong = torch.minimum(c_wrong, torch.clamp(1.0 - c_correct, eps, 1.0 - eps))
        h_plus = quorum_success_probability(c_correct.reshape(-1), k=k, alpha=alpha, eps=eps, temperature=temperature)
        h_minus = quorum_success_probability(c_wrong.reshape(-1), k=k, alpha=alpha, eps=eps, temperature=temperature)
        h_minus = torch.minimum(h_minus, torch.clamp(1.0 - h_plus, 0.0, 1.0))
        h_zero = torch.clamp(1.0 - h_plus - h_minus, 0.0, 1.0)
        transient_mass = correct_states.sum(dim=1) + wrong_states.sum(dim=1) + undecided_neutral
        expected_rounds = expected_rounds + transient_mass
        correct_states, wrong_states, correct_absorbed, wrong_absorbed, undecided_neutral = _transition_states(
            correct_states=correct_states,
            wrong_states=wrong_states,
            correct_absorbed=correct_absorbed,
            wrong_absorbed=wrong_absorbed,
            undecided_neutral=undecided_neutral,
            h_plus=h_plus,
            h_minus=h_minus,
            h_zero=h_zero,
            beta=beta,
        )

    correct_node = (correct_absorbed.reshape(num_nodes, Q) * w_row).sum(dim=1)
    wrong_node = (wrong_absorbed.reshape(num_nodes, Q) * w_row).sum(dim=1)
    undecided_copy = (correct_states.sum(dim=1) + wrong_states.sum(dim=1) + undecided_neutral).reshape(num_nodes, Q)
    undecided_node = torch.clamp((undecided_copy * w_row).sum(dim=1), 0.0, 1.0)
    expected_node = torch.clamp((expected_rounds.reshape(num_nodes, Q) * w_row).sum(dim=1), 0.0, float(rounds))

    metric_dtype = reference.dtype
    graph_metrics = {
        "node_mean_correct_decision": correct_node.mean(),
        "node_min_correct_decision": correct_node.min(),
        "node_10pct_correct_decision": torch.quantile(correct_node, 0.10),
        "node_mean_wrong_decision": wrong_node.mean(),
        "undecided_mean": undecided_node.mean(),
        "expected_rounds_mean": expected_node.mean(),
        "isolated_node_count": support.diagnostics["isolated_node_count"],
        "unique_out_degree_lt_k_count": (
            support.diagnostics["unique_out_degree"] < float(k)
        ).to(dtype=metric_dtype).sum(),
        "effective_unique_peer_degree_lt_k_count": (
            support.effective_unique_peer_degree < float(k)
        ).to(dtype=metric_dtype).sum(),
    }
    result: dict[str, Any] = {
        "query_support": support,
        "node_p_correct_decision": correct_node,
        "node_p_wrong_decision": wrong_node,
        "node_p_undecided": undecided_node,
        "node_expected_rounds": expected_node,
        "C_avalanche_node_mean": correct_node.mean(),
        "D_avalanche_rounds_mean": expected_node.mean(),
        "graph_metrics": graph_metrics,
        "quenched_quadrature": Q,
    }
    if network_tail_tau is not None:
        tail = _network_fraction_tail(correct_node, network_tail_tau, network_tail_var_floor, reference)
        result["F_network_tail"] = tail
        graph_metrics["F_network_tail"] = tail
    return result


def evaluate_graph_coupled_avalanche(
    *,
    num_nodes: int,
    link_success: torch.Tensor,
    initial_correct_preference: torch.Tensor,
    initial_wrong_preference: torch.Tensor,
    k: int,
    alpha: int,
    beta: int,
    rounds: int,
    edge_index: torch.Tensor | None = None,
    src_index: torch.Tensor | None = None,
    dst_index: torch.Tensor | None = None,
    query_weight: torch.Tensor | None = None,
    topology_weight: torch.Tensor | None = None,
    allow_self_loops: bool = False,
    allow_multi_edges: bool = False,
    query_support_backend: str = "legacy",
    diagnostics_mode: str = "full",
    eps: float = 1e-6,
    temperature: float = 1.0,
    quenched_quadrature: int = 1,
    network_tail_tau: float | None = None,
    network_tail_var_floor: float = 1e-6,
    return_history: bool = False,
) -> dict[str, Any]:
    _validate_parameters(k=k, alpha=alpha, beta=beta, rounds=rounds, eps=eps, temperature=temperature)
    if num_nodes < 0:
        raise ValueError("num_nodes must be nonnegative")
    if not isinstance(initial_correct_preference, torch.Tensor):
        raise TypeError("initial_correct_preference must be a torch.Tensor")
    reference = initial_correct_preference
    initial_correct = _as_node_probability(
        "initial_correct_preference",
        initial_correct_preference,
        num_nodes,
        reference,
    )
    initial_wrong = _as_node_probability(
        "initial_wrong_preference",
        initial_wrong_preference,
        num_nodes,
        reference,
    )
    if bool(torch.any((initial_correct + initial_wrong).detach() > 1.0 + PROBABILITY_TOLERANCE).cpu()):
        raise ValueError("initial_correct_preference + initial_wrong_preference must be <= 1")

    support = compute_topology_query_support(
        num_nodes=num_nodes,
        edge_index=edge_index,
        src_index=src_index,
        dst_index=dst_index,
        query_weight=query_weight,
        topology_weight=topology_weight,
        link_success=link_success,
        node_correct_preference=initial_correct,
        node_wrong_preference=initial_wrong,
        allow_self_loops=allow_self_loops,
        allow_multi_edges=allow_multi_edges,
        query_support_backend=query_support_backend,
        diagnostics_mode=diagnostics_mode,
    )
    link_values = link_success.to(dtype=reference.dtype, device=reference.device).reshape(-1)

    if quenched_quadrature is not None and int(quenched_quadrature) > 1:
        if return_history:
            raise ValueError("return_history is not supported when quenched_quadrature > 1")
        return _evaluate_quenched_avalanche(
            num_nodes=num_nodes,
            support=support,
            link_values=link_values,
            initial_correct=initial_correct,
            initial_wrong=initial_wrong,
            k=k,
            alpha=alpha,
            beta=beta,
            rounds=rounds,
            quenched_quadrature=int(quenched_quadrature),
            network_tail_tau=network_tail_tau,
            network_tail_var_floor=network_tail_var_floor,
            eps=eps,
            temperature=temperature,
            reference=reference,
        )

    correct_states = reference.new_zeros((num_nodes, beta))
    wrong_states = reference.new_zeros((num_nodes, beta))
    correct_states[:, 0] = initial_correct
    wrong_states[:, 0] = initial_wrong
    correct_absorbed = reference.new_zeros((num_nodes,))
    wrong_absorbed = reference.new_zeros((num_nodes,))
    undecided_neutral = _checked_probability("initial_neutral_preference", 1.0 - initial_correct - initial_wrong)
    expected_rounds = reference.new_zeros((num_nodes,))

    correct_history: list[torch.Tensor] = []
    wrong_history: list[torch.Tensor] = []
    h_plus_history: list[torch.Tensor] = []
    h_minus_history: list[torch.Tensor] = []
    mass_error_history: list[torch.Tensor] = []

    for _ in range(rounds):
        correct_preference = correct_absorbed + correct_states.sum(dim=1)
        wrong_preference = wrong_absorbed + wrong_states.sum(dim=1)
        p_correct_query, p_wrong_query = _coupled_query_support(
            num_nodes=num_nodes,
            src_index=support.src_index,
            dst_index=support.dst_index,
            normalized_query_weight=support.normalized_query_weight,
            link_success=link_values,
            correct_preference=correct_preference,
            wrong_preference=wrong_preference,
        )
        h_plus = quorum_success_probability(p_correct_query, k=k, alpha=alpha, eps=eps, temperature=temperature)
        h_minus = quorum_success_probability(p_wrong_query, k=k, alpha=alpha, eps=eps, temperature=temperature)
        h_zero_raw = 1.0 - h_plus - h_minus
        if bool(torch.any(h_zero_raw.detach() < -PROBABILITY_TOLERANCE).cpu()):
            raise ValueError("round quorum probabilities exceed one")
        h_zero = torch.clamp(h_zero_raw, 0.0, 1.0)

        transient_mass = correct_states.sum(dim=1) + wrong_states.sum(dim=1) + undecided_neutral
        expected_rounds = expected_rounds + transient_mass

        if return_history:
            correct_history.append(correct_preference)
            wrong_history.append(wrong_preference)
            h_plus_history.append(h_plus)
            h_minus_history.append(h_minus)

        correct_states, wrong_states, correct_absorbed, wrong_absorbed, undecided_neutral = _transition_states(
            correct_states=correct_states,
            wrong_states=wrong_states,
            correct_absorbed=correct_absorbed,
            wrong_absorbed=wrong_absorbed,
            undecided_neutral=undecided_neutral,
            h_plus=h_plus,
            h_minus=h_minus,
            h_zero=h_zero,
            beta=beta,
        )
        if return_history:
            mass_error_history.append(
                torch.abs(
                    _state_mass(
                        correct_states,
                        wrong_states,
                        correct_absorbed,
                        wrong_absorbed,
                        undecided_neutral,
                    )
                    - 1.0
                )
            )

    p_undecided = torch.clamp(
        correct_states.sum(dim=1) + wrong_states.sum(dim=1) + undecided_neutral,
        0.0,
        1.0,
    )
    graph_metrics = {
        "node_mean_correct_decision": correct_absorbed.mean() if num_nodes else reference.new_tensor(0.0),
        "node_min_correct_decision": correct_absorbed.min() if num_nodes else reference.new_tensor(0.0),
        "node_10pct_correct_decision": torch.quantile(correct_absorbed, 0.10) if num_nodes else reference.new_tensor(0.0),
        "node_mean_wrong_decision": wrong_absorbed.mean() if num_nodes else reference.new_tensor(0.0),
        "undecided_mean": p_undecided.mean() if num_nodes else reference.new_tensor(0.0),
        "expected_rounds_mean": expected_rounds.mean() if num_nodes else reference.new_tensor(0.0),
        "isolated_node_count": support.diagnostics["isolated_node_count"],
        "unique_out_degree_lt_k_count": (
            support.diagnostics["unique_out_degree"] < float(k)
        ).to(dtype=reference.dtype).sum(),
        "effective_unique_peer_degree_lt_k_count": (
            support.effective_unique_peer_degree < float(k)
        ).to(dtype=reference.dtype).sum(),
    }
    result: dict[str, Any] = {
        "query_support": support,
        "node_p_correct_decision": correct_absorbed,
        "node_p_wrong_decision": wrong_absorbed,
        "node_p_undecided": p_undecided,
        "node_expected_rounds": torch.clamp(expected_rounds, 0.0, float(rounds)),
        "C_avalanche_node_mean": correct_absorbed.mean() if num_nodes else reference.new_tensor(0.0),
        "D_avalanche_rounds_mean": expected_rounds.mean() if num_nodes else reference.new_tensor(0.0),
        "graph_metrics": graph_metrics,
    }
    if network_tail_tau is not None:
        tail = _network_fraction_tail(correct_absorbed, network_tail_tau, network_tail_var_floor, reference)
        result["F_network_tail"] = tail
        graph_metrics["F_network_tail"] = tail
    if return_history:
        empty_history = reference.new_zeros((0, num_nodes))
        result.update(
            {
                "preference_correct_history": torch.stack(correct_history) if correct_history else empty_history,
                "preference_wrong_history": torch.stack(wrong_history) if wrong_history else empty_history,
                "h_plus_history": torch.stack(h_plus_history) if h_plus_history else empty_history,
                "h_minus_history": torch.stack(h_minus_history) if h_minus_history else empty_history,
                "state_mass_error_history": torch.stack(mass_error_history) if mass_error_history else empty_history,
            }
        )
    return result
