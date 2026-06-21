from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class CoupledLossConfig:
    reliability_failure_target: float = 1e-5
    reliability_tail_failure_target: float = 1e-5
    reliability_tau: float = 0.5
    delay_target_rounds: float = 5.0
    delay_p90_target_rounds: float = 8.0
    delay_tau: float = 1.0
    energy_target_j: float = 1e-3
    energy_p90_target_j: float = 2e-3
    energy_tau: float = 1e-3
    lambda_reliability_tail: float = 1.0
    lambda_delay_tail: float = 0.5
    lambda_energy_tail: float = 0.5
    reliability_tail_mode: str = "max"
    reliability_soft_tail_tau: float = 0.01
    use_reliability_gate: bool = True
    min_reliability_weight_when_satisfied: float = 0.05
    weight_reliability: float = 1.0
    weight_delay: float = 1.0
    weight_energy: float = 1.0
    eps: float = 1e-12
    # P1 remediation: scale-invariant backward loss.
    # Every component (L_R/L_D/L_E) is a mean over N nodes, which injects an
    # explicit 1/N factor into the per-edge gradient. As N grows from 500 to
    # 10000 the realised update collapses faster than 1/N^2 (R1 scale-law audit),
    # so production-scale training stalls. When ``scale_invariant_backward`` is
    # enabled the loss returns a separate ``effective_backward_loss`` =
    # raw_total_loss * (N / scale_reference_node_count) that cancels the mean's
    # 1/N. The RAW ``total_loss`` and raw C/D/E metrics are reported unchanged;
    # only the scalar used for ``.backward()`` is rescaled (R3 raw/effective
    # contract).
    scale_invariant_backward: bool = False
    scale_reference_node_count: float = 100.0

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None) -> "CoupledLossConfig":
        data = data or {}
        return cls(
            reliability_failure_target=float(data.get("reliability_failure_target", cls.reliability_failure_target)),
            reliability_tail_failure_target=float(
                data.get("reliability_tail_failure_target", cls.reliability_tail_failure_target)
            ),
            reliability_tau=float(data.get("reliability_tau", cls.reliability_tau)),
            delay_target_rounds=float(data.get("delay_target_rounds", cls.delay_target_rounds)),
            delay_p90_target_rounds=float(data.get("delay_p90_target_rounds", cls.delay_p90_target_rounds)),
            delay_tau=float(data.get("delay_tau", cls.delay_tau)),
            energy_target_j=float(data.get("energy_target_j", cls.energy_target_j)),
            energy_p90_target_j=float(data.get("energy_p90_target_j", cls.energy_p90_target_j)),
            energy_tau=float(data.get("energy_tau", cls.energy_tau)),
            lambda_reliability_tail=float(data.get("lambda_reliability_tail", cls.lambda_reliability_tail)),
            lambda_delay_tail=float(data.get("lambda_delay_tail", cls.lambda_delay_tail)),
            lambda_energy_tail=float(data.get("lambda_energy_tail", cls.lambda_energy_tail)),
            reliability_tail_mode=str(data.get("reliability_tail_mode", cls.reliability_tail_mode)),
            reliability_soft_tail_tau=float(data.get("reliability_soft_tail_tau", cls.reliability_soft_tail_tau)),
            use_reliability_gate=bool(data.get("use_reliability_gate", cls.use_reliability_gate)),
            min_reliability_weight_when_satisfied=float(
                data.get("min_reliability_weight_when_satisfied", cls.min_reliability_weight_when_satisfied)
            ),
            weight_reliability=float(data.get("weight_reliability", cls.weight_reliability)),
            weight_delay=float(data.get("weight_delay", cls.weight_delay)),
            weight_energy=float(data.get("weight_energy", cls.weight_energy)),
            eps=float(data.get("eps", cls.eps)),
            scale_invariant_backward=bool(data.get("scale_invariant_backward", cls.scale_invariant_backward)),
            scale_reference_node_count=float(
                data.get("scale_reference_node_count", cls.scale_reference_node_count)
            ),
        )


def _first_tensor(values: Mapping[str, Any]) -> torch.Tensor:
    for value in values.values():
        if isinstance(value, torch.Tensor):
            return value
    return torch.tensor(0.0, dtype=torch.float64)


def _as_tensor(value: Any, reference: torch.Tensor) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(dtype=reference.dtype, device=reference.device)
    return torch.as_tensor(value, dtype=reference.dtype, device=reference.device)


def _scalar_metric(values: Mapping[str, Any], keys: tuple[str, ...], reference: torch.Tensor) -> torch.Tensor:
    for key in keys:
        if key in values:
            tensor = _as_tensor(values[key], reference)
            return tensor.mean() if tensor.ndim else tensor
    raise KeyError(f"missing required metric; expected one of {keys}")


def _validate_probability(name: str, value: torch.Tensor, tol: float = 1e-8) -> torch.Tensor:
    if bool(torch.any(~torch.isfinite(value.detach())).cpu()):
        raise ValueError(f"{name} must contain only finite probabilities")
    if bool(torch.any(value.detach() < -tol).cpu()) or bool(torch.any(value.detach() > 1.0 + tol).cpu()):
        raise ValueError(f"{name} must be in [0, 1]")
    return torch.clamp(value, 0.0, 1.0)


def _node_failure(values: Mapping[str, Any], reference: torch.Tensor) -> torch.Tensor:
    if "node_failure_probability" in values:
        return _validate_probability("node_failure_probability", _as_tensor(values["node_failure_probability"], reference).reshape(-1))
    if "node_p_wrong_decision" in values and "node_p_undecided" in values:
        wrong = _validate_probability("node_p_wrong_decision", _as_tensor(values["node_p_wrong_decision"], reference).reshape(-1))
        undecided = _validate_probability("node_p_undecided", _as_tensor(values["node_p_undecided"], reference).reshape(-1))
        if wrong.numel() != undecided.numel():
            raise ValueError("node_p_wrong_decision and node_p_undecided must have the same shape")
        failure = wrong + undecided
        if bool(torch.any(failure.detach() > 1.0 + 1e-8).cpu()):
            raise ValueError("node_p_wrong_decision + node_p_undecided must be <= 1")
        return torch.clamp(failure, 0.0, 1.0)
    if "node_p_correct_decision" in values:
        correct = _validate_probability("node_p_correct_decision", _as_tensor(values["node_p_correct_decision"], reference).reshape(-1))
        return torch.clamp(1.0 - correct, 0.0, 1.0)
    if "F_avalanche_node_mean" in values:
        return _validate_probability("F_avalanche_node_mean", _as_tensor(values["F_avalanche_node_mean"], reference).reshape(-1))
    if "C_avalanche_node_mean" in values:
        correct = _validate_probability("C_avalanche_node_mean", _as_tensor(values["C_avalanche_node_mean"], reference).reshape(-1))
        return torch.clamp(1.0 - correct, 0.0, 1.0)
    raise KeyError("evaluator output must provide node failure, decision probabilities, F, or C")


def _smooth_max_tail(failure: torch.Tensor, reference: torch.Tensor, tau_value: float) -> torch.Tensor:
    if failure.numel() == 0:
        return reference.new_tensor(0.0)
    tau = reference.new_tensor(tau_value)
    return tau * torch.logsumexp(failure / tau, dim=0)


def _tail_failure(
    values: Mapping[str, Any],
    failure: torch.Tensor,
    config: CoupledLossConfig,
    reference: torch.Tensor,
) -> torch.Tensor:
    mode = config.reliability_tail_mode
    if mode == "p90":
        if "F_avalanche_node_p90" in values:
            return _validate_probability("F_avalanche_node_p90", _as_tensor(values["F_avalanche_node_p90"], reference))
        return torch.quantile(failure, 0.90) if failure.numel() else reference.new_tensor(0.0)
    if mode == "softmax_tail":
        return _smooth_max_tail(failure, reference, config.reliability_soft_tail_tau)
    if mode == "softmean_tail":
        if "F_avalanche_node_softmax_tail" in values:
            return _validate_probability(
                "F_avalanche_node_softmax_tail",
                _as_tensor(values["F_avalanche_node_softmax_tail"], reference),
            )
        if failure.numel() == 0:
            return reference.new_tensor(0.0)
        tau = reference.new_tensor(config.reliability_soft_tail_tau)
        return tau * (
            torch.logsumexp(failure / tau, dim=0)
            - torch.log(reference.new_tensor(float(max(failure.numel(), 1))))
        )
    if mode == "max":
        if "F_avalanche_node_max" in values:
            return _validate_probability("F_avalanche_node_max", _as_tensor(values["F_avalanche_node_max"], reference))
        return failure.max() if failure.numel() else reference.new_tensor(0.0)
    raise ValueError("reliability_tail_mode must be one of: p90, softmax_tail, softmean_tail, max")


def _positive_config(config: CoupledLossConfig) -> None:
    positive_fields = {
        "reliability_failure_target": config.reliability_failure_target,
        "reliability_tail_failure_target": config.reliability_tail_failure_target,
        "reliability_tau": config.reliability_tau,
        "delay_tau": config.delay_tau,
        "energy_tau": config.energy_tau,
        "reliability_soft_tail_tau": config.reliability_soft_tail_tau,
        "eps": config.eps,
    }
    for name, value in positive_fields.items():
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")
    if config.min_reliability_weight_when_satisfied < 0.0:
        raise ValueError("min_reliability_weight_when_satisfied must be nonnegative")
    if config.scale_reference_node_count <= 0.0:
        raise ValueError("scale_reference_node_count must be positive")


def compute_coupled_loss(
    evaluator_output: Mapping[str, Any],
    config: CoupledLossConfig | Mapping[str, object] | None = None,
) -> dict[str, Any]:
    cfg = config if isinstance(config, CoupledLossConfig) else CoupledLossConfig.from_mapping(config)
    _positive_config(cfg)
    reference = _first_tensor(evaluator_output)
    failure = _node_failure(evaluator_output, reference)
    failure_mean = failure.mean() if failure.numel() else reference.new_tensor(0.0)
    failure_tail = _tail_failure(evaluator_output, failure, cfg, reference)
    conservative_tail = _smooth_max_tail(failure, reference, cfg.reliability_soft_tail_tau)
    eps = reference.new_tensor(cfg.eps)

    l_r_mean = F.softplus(
        (torch.log(failure_mean + eps) - torch.log(reference.new_tensor(cfg.reliability_failure_target)))
        / reference.new_tensor(cfg.reliability_tau)
    )
    l_r_tail = F.softplus(
        (torch.log(failure_tail + eps) - torch.log(reference.new_tensor(cfg.reliability_tail_failure_target)))
        / reference.new_tensor(cfg.reliability_tau)
    )
    l_r = l_r_mean + cfg.lambda_reliability_tail * l_r_tail

    d_mean = _scalar_metric(evaluator_output, ("D_avalanche_rounds_mean", "D_avalanche"), reference)
    d_p90 = _scalar_metric(evaluator_output, ("D_avalanche_rounds_p90", "D_avalanche_rounds_mean", "D_avalanche"), reference)
    l_d_mean = F.softplus((d_mean - reference.new_tensor(cfg.delay_target_rounds)) / reference.new_tensor(cfg.delay_tau))
    l_d_tail = F.softplus((d_p90 - reference.new_tensor(cfg.delay_p90_target_rounds)) / reference.new_tensor(cfg.delay_tau))
    l_d = l_d_mean + cfg.lambda_delay_tail * l_d_tail

    e_mean = _scalar_metric(evaluator_output, ("E_consensus_node_mean", "E_avalanche"), reference)
    e_p90 = _scalar_metric(evaluator_output, ("E_consensus_node_p90", "E_consensus_node_mean", "E_avalanche"), reference)
    l_e_mean = F.softplus((e_mean - reference.new_tensor(cfg.energy_target_j)) / reference.new_tensor(cfg.energy_tau))
    l_e_tail = F.softplus((e_p90 - reference.new_tensor(cfg.energy_p90_target_j)) / reference.new_tensor(cfg.energy_tau))
    l_e = l_e_mean + cfg.lambda_energy_tail * l_e_tail

    mean_satisfied = bool((failure_mean.detach() < cfg.reliability_failure_target).cpu())
    tail_satisfied = bool((failure_tail.detach() < cfg.reliability_tail_failure_target).cpu())
    reliability_should_weaken = mean_satisfied and tail_satisfied
    reliability_weight_value = cfg.weight_reliability
    if cfg.use_reliability_gate and reliability_should_weaken:
        reliability_weight_value = cfg.weight_reliability * cfg.min_reliability_weight_when_satisfied
    weights = {
        "R": reference.new_tensor(reliability_weight_value),
        "D": reference.new_tensor(cfg.weight_delay),
        "E": reference.new_tensor(cfg.weight_energy),
    }
    weighted_l_r = weights["R"] * l_r
    weighted_l_d = weights["D"] * l_d
    weighted_l_e = weights["E"] * l_e
    total = weighted_l_r + weighted_l_d + weighted_l_e
    reliability_nines_mean = -torch.log10(failure_mean + eps)
    reliability_nines_tail = -torch.log10(failure_tail + eps)
    # P1: scale-invariant backward scalar. RAW total_loss stays untouched for
    # reporting; effective_backward_loss is what callers should .backward() on.
    node_count = int(failure.numel())
    if cfg.scale_invariant_backward and node_count > 0:
        scale_multiplier_value = float(node_count) / float(cfg.scale_reference_node_count)
    else:
        scale_multiplier_value = 1.0
    scale_multiplier = reference.new_tensor(scale_multiplier_value)
    effective_backward_loss = total * scale_multiplier
    return {
        "total_loss": total,
        "effective_backward_loss": effective_backward_loss,
        "scale_invariant_backward": bool(cfg.scale_invariant_backward),
        "scale_backward_multiplier": scale_multiplier,
        "scale_reference_node_count": float(cfg.scale_reference_node_count),
        "node_count": node_count,
        "L_R": l_r,
        "L_D": l_d,
        "L_E": l_e,
        "weighted_L_R": weighted_l_r,
        "weighted_L_D": weighted_l_d,
        "weighted_L_E": weighted_l_e,
        "L_R_mean": l_r_mean,
        "L_R_tail": l_r_tail,
        "L_D_mean": l_d_mean,
        "L_D_tail": l_d_tail,
        "L_E_mean": l_e_mean,
        "L_E_tail": l_e_tail,
        "F_mean": failure_mean,
        "F_tail": failure_tail,
        "F_tail_conservative": conservative_tail,
        "reliability_nines_mean": reliability_nines_mean,
        "reliability_nines_tail": reliability_nines_tail,
        "reliability_loss_should_weaken": reliability_should_weaken,
        "weights": weights,
        "diagnostics": {
            "reliability_tail_mode": cfg.reliability_tail_mode,
            "reliability_gate_mode": "mean_and_tail" if cfg.use_reliability_gate else "disabled",
            "F_tail_conservative": conservative_tail,
            "mean_reliability_satisfied": mean_satisfied,
            "tail_reliability_satisfied": tail_satisfied,
            "mean_failure_satisfied": mean_satisfied,
            "tail_failure_satisfied": tail_satisfied,
            "use_reliability_gate": cfg.use_reliability_gate,
            "delay_mean": d_mean,
            "delay_tail": d_p90,
            "energy_mean": e_mean,
            "energy_tail": e_p90,
        },
    }
