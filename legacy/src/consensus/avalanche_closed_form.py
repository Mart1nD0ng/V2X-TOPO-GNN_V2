from __future__ import annotations

from typing import Any

import numpy as np
import scipy.special
import torch

PROBABILITY_TOLERANCE = 1e-6


class _RegularizedBetaInc(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, a_value: float, b_value: float) -> torch.Tensor:
        x_cpu = x.detach().to(dtype=torch.float64, device="cpu").numpy()
        y_cpu = scipy.special.betainc(float(a_value), float(b_value), x_cpu)
        result = torch.as_tensor(y_cpu, dtype=x.dtype, device=x.device)
        ctx.save_for_backward(x)
        ctx.a_value = float(a_value)
        ctx.b_value = float(b_value)
        return result

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        (x,) = ctx.saved_tensors
        a_tensor = torch.as_tensor(ctx.a_value, dtype=x.dtype, device=x.device)
        b_tensor = torch.as_tensor(ctx.b_value, dtype=x.dtype, device=x.device)
        log_pdf = (
            (a_tensor - 1.0) * torch.log(x)
            + (b_tensor - 1.0) * torch.log1p(-x)
            - torch.lgamma(a_tensor)
            - torch.lgamma(b_tensor)
            + torch.lgamma(a_tensor + b_tensor)
        )
        return grad_output * torch.exp(log_pdf), None, None


def avalanche_state_count(beta: int) -> int:
    if beta < 1:
        raise ValueError("beta must be positive")
    return 2 * int(beta) + 2


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
    if not np.isfinite(eps) or eps < 0.0 or eps >= 0.5:
        raise ValueError("eps must satisfy 0 <= eps < 0.5")
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be positive")


def _validate_probability_tensor(name: str, value: torch.Tensor, tolerance: float = PROBABILITY_TOLERANCE) -> None:
    if not torch.is_floating_point(value):
        raise ValueError(f"{name} must use a floating-point dtype")
    if value.numel() == 0:
        raise ValueError(f"{name} must not be empty")
    value_detached = value.detach()
    if bool(torch.any(~torch.isfinite(value_detached)).cpu()):
        raise ValueError(f"{name} must contain only finite probabilities")
    if bool(torch.any(value_detached < -tolerance).cpu()) or bool(torch.any(value_detached > 1.0 + tolerance).cpu()):
        raise ValueError(f"{name} must be in [0, 1]")


def _as_tensor_like(value: torch.Tensor | float, reference: torch.Tensor) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(dtype=reference.dtype, device=reference.device)
    return torch.as_tensor(value, dtype=reference.dtype, device=reference.device)


def _stable_probability(x: torch.Tensor, eps: float, temperature: float) -> torch.Tensor:
    x_clamped = torch.clamp(x, 0.0, 1.0)
    if temperature != 1.0:
        inner = torch.clamp(x_clamped, eps, 1.0 - eps)
        x_clamped = torch.sigmoid(torch.logit(inner) / temperature)
    return eps + (1.0 - 2.0 * eps) * x_clamped


def quorum_success_probability(
    x: torch.Tensor,
    *,
    k: int,
    alpha: int,
    eps: float = 1e-6,
    temperature: float = 1.0,
) -> torch.Tensor:
    _validate_parameters(k=k, alpha=alpha, beta=1, rounds=0, eps=eps, temperature=temperature)
    _validate_probability_tensor("x", x)
    x_eff = _stable_probability(x, eps=eps, temperature=temperature)
    return _RegularizedBetaInc.apply(x_eff, float(alpha), float(k - alpha + 1))


def betabinomial_upper_tail(
    mu: torch.Tensor,
    rho: torch.Tensor,
    *,
    k: int,
    alpha: int,
    eps: float = 1e-6,
    rho_floor: float = 1e-5,
    rho_max: float = 0.999,
) -> torch.Tensor:
    """SSMC Layer-A quorum: ``P[X >= alpha]`` for ``X ~ Beta-Binomial(k, a, b)`` whose
    ``k`` exchangeable votes have mean ``mu`` and intra-class correlation ``rho``
    (so ``Var(vote-fraction) = mu(1-mu)*rho``).

    This is exactly ``E_{c ~ Beta(mu, rho)}[ I_c(alpha, k-alpha+1) ]`` — the per-draw quorum
    ``Binomial(k, c)`` analytically integrated over the *realized-support* distribution
    ``c`` rather than evaluated at its mean. As ``rho -> 0`` it converges to the Binomial
    tail ``I_mu(alpha, k-alpha+1)`` (today's mean-field evaluator); as ``rho > 0`` it is
    over-dispersed and reproduces the trajectory-variance pessimism the mean-field erases.

    Pure-torch log-space (``lgamma``/``logsumexp``): differentiable in BOTH ``mu`` and
    ``rho``, avoiding the ``_RegularizedBetaInc`` parameter-gradient gap. No sampling.
    """
    if k < 1:
        raise ValueError("k must be positive")
    if alpha < 1 or alpha > k:
        raise ValueError("alpha must satisfy 1 <= alpha <= k")
    if not (0.0 <= rho_floor < rho_max < 1.0):
        raise ValueError("require 0 <= rho_floor < rho_max < 1")
    mu_c = torch.clamp(mu, eps, 1.0 - eps)
    rho_c = torch.clamp(rho, rho_floor, rho_max)
    concentration = (1.0 - rho_c) / rho_c  # a + b
    a = mu_c * concentration
    b = (1.0 - mu_c) * concentration
    counts = torch.arange(alpha, k + 1, dtype=mu_c.dtype, device=mu_c.device)
    counts = counts.reshape(*([1] * mu_c.ndim), -1)  # [..., M] broadcast over mu
    a_e = a.unsqueeze(-1)
    b_e = b.unsqueeze(-1)
    k_t = torch.as_tensor(float(k), dtype=mu_c.dtype, device=mu_c.device)
    log_choose = torch.lgamma(k_t + 1.0) - torch.lgamma(counts + 1.0) - torch.lgamma(k_t - counts + 1.0)
    log_pmf = (
        log_choose
        + torch.lgamma(counts + a_e)
        + torch.lgamma(k_t - counts + b_e)
        - torch.lgamma(k_t + a_e + b_e)
        - torch.lgamma(a_e)
        - torch.lgamma(b_e)
        + torch.lgamma(a_e + b_e)
    )
    tail = torch.exp(torch.logsumexp(log_pmf, dim=-1))
    return torch.clamp(tail, 0.0, 1.0)


def _build_transition_matrix(h_plus: torch.Tensor, h_minus: torch.Tensor, beta: int, eps: float) -> torch.Tensor:
    batch_count = int(h_plus.shape[0])
    state_count = avalanche_state_count(beta)
    correct_abs = 2 * beta
    wrong_abs = 2 * beta + 1
    h_zero_raw = 1.0 - h_plus - h_minus
    if bool(torch.any(h_zero_raw < -eps).detach().cpu()):
        raise ValueError("h_plus + h_minus exceeds 1; use compatible query probabilities and alpha")
    h_zero = torch.clamp(h_zero_raw, 0.0, 1.0)
    matrix = h_plus.new_zeros((batch_count, state_count, state_count))

    for count in range(beta):
        plus_state = count
        minus_state = beta + count
        plus_success_target = correct_abs if count + 1 >= beta else count + 1
        minus_success_target = wrong_abs if count + 1 >= beta else beta + count + 1
        switch_to_minus_target = wrong_abs if beta == 1 else beta + 1
        switch_to_plus_target = correct_abs if beta == 1 else 1

        matrix[:, plus_state, plus_success_target] = h_plus
        matrix[:, plus_state, switch_to_minus_target] = h_minus
        matrix[:, plus_state, 0] = h_zero

        matrix[:, minus_state, minus_success_target] = h_minus
        matrix[:, minus_state, switch_to_plus_target] = h_plus
        matrix[:, minus_state, beta] = h_zero

    matrix[:, correct_abs, correct_abs] = 1.0
    matrix[:, wrong_abs, wrong_abs] = 1.0
    return matrix


def _initial_distribution(
    initial_correct_preference: torch.Tensor | float,
    reference: torch.Tensor,
    beta: int,
) -> torch.Tensor:
    initial = _as_tensor_like(initial_correct_preference, reference).reshape(-1)
    if initial.numel() == 1 and reference.numel() != 1:
        initial = initial.expand(reference.numel())
    if initial.numel() != reference.numel():
        raise ValueError("initial_correct_preference must be scalar or match p_correct_query shape")
    _validate_probability_tensor("initial_correct_preference", initial)
    initial = torch.clamp(initial, 0.0, 1.0)
    state_count = avalanche_state_count(beta)
    pi0 = reference.new_zeros((reference.numel(), state_count))
    pi0[:, 0] = initial
    pi0[:, beta] = 1.0 - initial
    return pi0


def _expected_rounds(pi0: torch.Tensor, matrix: torch.Tensor, beta: int, rounds: int) -> torch.Tensor:
    transient_count = 2 * beta
    if rounds == 0:
        return pi0.new_zeros((pi0.shape[0],))
    q_matrix = matrix[:, :transient_count, :transient_count]
    dist = pi0[:, :transient_count]
    expected = pi0.new_zeros((pi0.shape[0],))
    for _ in range(rounds):
        expected = expected + dist.sum(dim=1)
        dist = (dist.unsqueeze(1) @ q_matrix).squeeze(1)
    return expected


def evaluate_avalanche_closed_form(
    p_correct_query: torch.Tensor,
    p_wrong_query: torch.Tensor | None = None,
    *,
    k: int,
    alpha: int,
    beta: int,
    rounds: int,
    initial_correct_preference: torch.Tensor | float = 1.0,
    eps: float = 1e-6,
    temperature: float = 1.0,
) -> dict[str, torch.Tensor]:
    _validate_parameters(k=k, alpha=alpha, beta=beta, rounds=rounds, eps=eps, temperature=temperature)
    if not isinstance(p_correct_query, torch.Tensor):
        raise TypeError("p_correct_query must be a torch.Tensor")
    original_shape = p_correct_query.shape
    p_correct = p_correct_query.reshape(-1)
    _validate_probability_tensor("p_correct_query", p_correct)
    if p_wrong_query is None:
        p_wrong = 1.0 - p_correct
    else:
        if not isinstance(p_wrong_query, torch.Tensor):
            raise TypeError("p_wrong_query must be a torch.Tensor when provided")
        p_wrong_raw = p_wrong_query.to(dtype=p_correct_query.dtype, device=p_correct_query.device)
        if p_wrong_raw.numel() == 1 and p_correct.numel() != 1:
            p_wrong = p_wrong_raw.reshape(-1).expand_as(p_correct)
        elif p_wrong_raw.shape == p_correct_query.shape:
            p_wrong = p_wrong_raw.reshape(-1)
        else:
            raise ValueError("p_wrong_query must be scalar-like or match p_correct_query shape")
        _validate_probability_tensor("p_wrong_query", p_wrong)
        query_sum = p_correct + p_wrong
        if bool(torch.any(query_sum.detach() > 1.0 + PROBABILITY_TOLERANCE).cpu()):
            raise ValueError("p_correct_query + p_wrong_query must be <= 1")

    if p_wrong_query is None:
        if p_wrong.numel() == 1 and p_correct.numel() != 1:
            p_wrong = p_wrong.expand_as(p_correct)

    h_plus = quorum_success_probability(p_correct, k=k, alpha=alpha, eps=eps, temperature=temperature)
    h_minus = quorum_success_probability(p_wrong, k=k, alpha=alpha, eps=eps, temperature=temperature)
    matrix = _build_transition_matrix(h_plus, h_minus, beta=beta, eps=eps)
    pi0 = _initial_distribution(initial_correct_preference, p_correct, beta=beta)

    matrix_power = torch.linalg.matrix_power(matrix, rounds)
    pi_t = (pi0.unsqueeze(1) @ matrix_power).squeeze(1)
    correct_abs = 2 * beta
    wrong_abs = 2 * beta + 1
    p_correct_decision = pi_t[:, correct_abs]
    p_wrong_decision = pi_t[:, wrong_abs]
    p_undecided = torch.clamp(1.0 - p_correct_decision - p_wrong_decision, 0.0, 1.0)
    expected_rounds = torch.clamp(_expected_rounds(pi0, matrix, beta=beta, rounds=rounds), 0.0, float(rounds))

    def restore(value: torch.Tensor) -> torch.Tensor:
        return value.reshape(original_shape)

    p_correct_node = restore(p_correct_decision)
    expected_node = restore(expected_rounds)
    return {
        "h_plus": restore(h_plus),
        "h_minus": restore(h_minus),
        "p_correct_decision": p_correct_node,
        "p_wrong_decision": restore(p_wrong_decision),
        "p_undecided": restore(p_undecided),
        "expected_rounds": expected_node,
        "C_avalanche_node_mean": p_correct_node.mean(),
        "D_avalanche_rounds_mean": expected_node.mean(),
    }
