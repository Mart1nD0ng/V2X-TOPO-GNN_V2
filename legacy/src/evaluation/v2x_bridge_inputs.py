from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from src.consensus.avalanche_closed_form import PROBABILITY_TOLERANCE


@dataclass(frozen=True)
class V2XBridgeInputs:
    num_nodes: int
    src_index: torch.Tensor
    dst_index: torch.Tensor
    topology_weight: torch.Tensor
    distance_m: torch.Tensor
    los_flag: torch.Tensor
    node_initial_correct: torch.Tensor
    node_initial_wrong: torch.Tensor

    def as_evaluation_kwargs(self) -> dict[str, torch.Tensor | int]:
        return {
            "num_nodes": self.num_nodes,
            "src_index": self.src_index,
            "dst_index": self.dst_index,
            "topology_weight": self.topology_weight,
            "distance_m": self.distance_m,
            "los_flag": self.los_flag,
            "node_initial_correct": self.node_initial_correct,
            "node_initial_wrong": self.node_initial_wrong,
        }


def _get_field(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _array_field(obj: Any, names: tuple[str, ...]) -> Any:
    for name in names:
        value = _get_field(obj, name)
        if value is not None:
            return value
    return None


def _to_long_tensor(name: str, values: Any, device: torch.device | str | None) -> torch.Tensor:
    if values is None:
        raise ValueError(f"{name} is required")
    tensor = torch.as_tensor(values, dtype=torch.long, device=device).reshape(-1)
    return tensor


def _to_float_tensor(name: str, values: Any, dtype: torch.dtype, device: torch.device | str | None) -> torch.Tensor:
    if values is None:
        raise ValueError(f"{name} is required")
    tensor = torch.as_tensor(values, dtype=dtype, device=device).reshape(-1)
    if bool(torch.any(~torch.isfinite(tensor.detach())).cpu()):
        raise ValueError(f"{name} must contain only finite values")
    return tensor


def _node_probability(name: str, values: Any, num_nodes: int, dtype: torch.dtype, device: torch.device | str | None) -> torch.Tensor:
    tensor = _to_float_tensor(name, values, dtype, device)
    if tensor.numel() != num_nodes:
        raise ValueError(f"{name} must contain num_nodes values")
    if bool(torch.any(tensor.detach() < -PROBABILITY_TOLERANCE).cpu()) or bool(
        torch.any(tensor.detach() > 1.0 + PROBABILITY_TOLERANCE).cpu()
    ):
        raise ValueError(f"{name} must be in [0, 1]")
    return torch.clamp(tensor, 0.0, 1.0)


def _candidate_los_lookup(candidate_graph: Any) -> dict[tuple[int, int], bool]:
    if candidate_graph is None:
        return {}
    src_values = _array_field(candidate_graph, ("source", "src_index", "src"))
    dst_values = _array_field(candidate_graph, ("target", "dst_index", "dst"))
    los_values = _array_field(candidate_graph, ("los_flag", "los"))
    if src_values is None or dst_values is None or los_values is None:
        return {}
    lookup: dict[tuple[int, int], bool] = {}
    for src, dst, los in zip(np.asarray(src_values), np.asarray(dst_values), np.asarray(los_values)):
        lookup[(int(src), int(dst))] = bool(los)
    return lookup


def _los_from_topology_or_candidate(topology: Any, candidate_graph: Any, src: torch.Tensor, dst: torch.Tensor) -> np.ndarray:
    los_values = _array_field(topology, ("los_flag", "los"))
    if los_values is not None:
        return np.asarray(los_values, dtype=bool)
    lookup = _candidate_los_lookup(candidate_graph)
    if lookup:
        return np.asarray(
            [lookup.get((int(src_value), int(dst_value)), True) for src_value, dst_value in zip(src.cpu(), dst.cpu())],
            dtype=bool,
        )
    return np.ones(int(src.numel()), dtype=bool)


def build_v2x_bridge_inputs(
    *,
    topology: Any | None = None,
    candidate_graph: Any | None = None,
    num_nodes: int | None = None,
    src_index: Any | None = None,
    dst_index: Any | None = None,
    topology_weight: Any | None = None,
    distance_m: Any | None = None,
    los_flag: Any | None = None,
    node_initial_correct: Any,
    node_initial_wrong: Any,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> V2XBridgeInputs:
    """Convert M1 sparse environment/topology outputs into M3 evaluator tensors."""

    source_obj = topology if topology is not None else candidate_graph
    if source_obj is None and (src_index is None or dst_index is None):
        raise ValueError("topology, candidate_graph, or explicit src/dst edge arrays are required")

    inferred_num_nodes = num_nodes
    if inferred_num_nodes is None:
        inferred = _get_field(source_obj, "num_nodes")
        if inferred is None:
            raise ValueError("num_nodes is required when it cannot be inferred")
        inferred_num_nodes = int(inferred)
    if inferred_num_nodes < 0:
        raise ValueError("num_nodes must be nonnegative")

    raw_src = src_index if src_index is not None else _array_field(source_obj, ("source", "src_index", "src"))
    raw_dst = dst_index if dst_index is not None else _array_field(source_obj, ("target", "dst_index", "dst"))
    src = _to_long_tensor("src_index", raw_src, device)
    dst = _to_long_tensor("dst_index", raw_dst, device)
    if src.numel() != dst.numel():
        raise ValueError("src_index and dst_index must have the same edge count")
    if src.numel():
        if bool(torch.any(src < 0).cpu()) or bool(torch.any(src >= inferred_num_nodes).cpu()):
            raise ValueError("src_index contains node ids outside [0, num_nodes)")
        if bool(torch.any(dst < 0).cpu()) or bool(torch.any(dst >= inferred_num_nodes).cpu()):
            raise ValueError("dst_index contains node ids outside [0, num_nodes)")

    raw_distance = distance_m if distance_m is not None else _array_field(source_obj, ("distance_m", "distance"))
    distance = _to_float_tensor("distance_m", raw_distance, dtype, device)
    if distance.numel() != src.numel():
        raise ValueError("distance_m must have one value per edge")
    if bool(torch.any(distance.detach() < 0.0).cpu()):
        raise ValueError("distance_m must be nonnegative")

    raw_weight = topology_weight
    if raw_weight is None:
        raw_weight = _array_field(source_obj, ("topology_weight", "query_weight", "success_probability", "channel_score"))
    if raw_weight is None:
        raw_weight = np.ones(int(src.numel()), dtype=float)
    weight = _to_float_tensor("topology_weight", raw_weight, dtype, device)
    if weight.numel() != src.numel():
        raise ValueError("topology_weight must have one value per edge")
    if bool(torch.any(weight.detach() < 0.0).cpu()):
        raise ValueError("topology_weight must be nonnegative")

    raw_los = los_flag if los_flag is not None else _los_from_topology_or_candidate(source_obj, candidate_graph, src, dst)
    los = _to_float_tensor("los_flag", raw_los, dtype, device)
    if los.numel() != src.numel():
        raise ValueError("los_flag must have one value per edge")

    correct = _node_probability("node_initial_correct", node_initial_correct, inferred_num_nodes, dtype, device)
    if node_initial_wrong is None:
        raise ValueError("node_initial_wrong is required for V2X evaluation paths")
    wrong = _node_probability("node_initial_wrong", node_initial_wrong, inferred_num_nodes, dtype, device)
    if bool(torch.any((correct + wrong).detach() > 1.0 + PROBABILITY_TOLERANCE).cpu()):
        raise ValueError("node_initial_correct + node_initial_wrong must be <= 1")

    return V2XBridgeInputs(
        num_nodes=inferred_num_nodes,
        src_index=src,
        dst_index=dst,
        topology_weight=weight,
        distance_m=distance,
        los_flag=los,
        node_initial_correct=correct,
        node_initial_wrong=wrong,
    )


m1_outputs_to_v2x_bridge_inputs = build_v2x_bridge_inputs
