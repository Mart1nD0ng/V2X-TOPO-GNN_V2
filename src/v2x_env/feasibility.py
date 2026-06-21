from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .baselines import (
    Topology,
    degree_capped_greedy,
    knn_by_channel_score,
    knn_by_distance,
    mst_backbone_plus_augmentation,
)
from .candidate_graph import CandidateGraph, CandidateGraphConfig, build_candidate_graph
from .channel_model import ChannelConfig
from .vehicle_snapshot import generate_vehicle_snapshot


class _UnionFind:
    def __init__(self, count: int) -> None:
        self.parent = list(range(count))
        self.size = [1] * count

    def find(self, value: int) -> int:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            parent = self.parent[value]
            self.parent[value] = root
            value = parent
        return root

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.size[left_root] < self.size[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.size[left_root] += self.size[right_root]

    def giant_ratio(self) -> float:
        if not self.parent:
            return 0.0
        counts: dict[int, int] = defaultdict(int)
        for value in range(len(self.parent)):
            counts[self.find(value)] += 1
        return max(counts.values()) / float(len(self.parent))


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text)
    except Exception:
        loaded = json.loads(text)
    if not isinstance(loaded, dict):
        raise ValueError(f"config must contain a mapping: {path}")
    return loaded


def stable_config_hash(config: Mapping[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def giant_component_ratio(num_nodes: int, source: np.ndarray, target: np.ndarray) -> float:
    uf = _UnionFind(num_nodes)
    for src, dst in zip(source, target):
        uf.union(int(src), int(dst))
    return uf.giant_ratio()


def _quantiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"p10": 0.0, "p50": 0.0, "p90": 0.0}
    q10, q50, q90 = np.quantile(values, [0.10, 0.50, 0.90])
    return {"p10": float(q10), "p50": float(q50), "p90": float(q90)}


def _query_success_proxy(topology: Topology) -> float:
    if topology.num_nodes == 0 or topology.edge_count == 0:
        return 0.0
    grouped: list[list[float]] = [[] for _ in range(topology.num_nodes)]
    for src, dst, prob in zip(topology.source, topology.target, topology.success_probability):
        probability = float(np.clip(prob, 0.0, 1.0))
        grouped[int(src)].append(probability)
        grouped[int(dst)].append(probability)
    node_values = []
    for values in grouped:
        if not values:
            node_values.append(0.0)
            continue
        miss_probability = 1.0
        for value in values:
            miss_probability *= 1.0 - value
        node_values.append(1.0 - miss_probability)
    return float(np.mean(node_values))


def candidate_metrics(graph: CandidateGraph) -> dict[str, Any]:
    outgoing = graph.outgoing_counts()
    return {
        "candidate_edge_count": graph.edge_count,
        "average_candidate_degree": float(np.mean(outgoing)) if outgoing.size else 0.0,
        "max_candidate_degree": int(np.max(outgoing)) if outgoing.size else 0,
        "candidate_giant_component_ratio": giant_component_ratio(graph.num_nodes, graph.source, graph.target),
        "candidate_link_success_quantiles": _quantiles(graph.success_probability),
        **graph.diagnostics,
    }


def _undirected_pairs(source: np.ndarray, target: np.ndarray) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for src, dst in zip(source, target):
        left = int(src)
        right = int(dst)
        if left == right:
            continue
        pairs.add((left, right) if left < right else (right, left))
    return pairs


def _reciprocity_ratio(source: np.ndarray, target: np.ndarray) -> float:
    directed = {(int(src), int(dst)) for src, dst in zip(source, target) if int(src) != int(dst)}
    if not directed:
        return 0.0
    reciprocated = sum(1 for src, dst in directed if (dst, src) in directed)
    return float(reciprocated / len(directed))


def topology_metrics(topology: Topology) -> dict[str, Any]:
    out_counts = np.bincount(topology.source, minlength=topology.num_nodes) if topology.num_nodes else np.asarray([], dtype=int)
    pairs = _undirected_pairs(topology.source, topology.target)
    directed_edge_count = topology.edge_count
    undirected_pair_count = len(pairs)
    legacy_average_degree = 2.0 * directed_edge_count / max(topology.num_nodes, 1)
    undirected_average_degree = 2.0 * undirected_pair_count / max(topology.num_nodes, 1)
    return {
        "edge_count": directed_edge_count,
        "average_degree": float(legacy_average_degree),
        "directed_edge_count": directed_edge_count,
        "mean_out_degree": float(np.mean(out_counts)) if out_counts.size else 0.0,
        "max_out_degree": int(np.max(out_counts)) if out_counts.size else 0,
        "undirected_pair_count": undirected_pair_count,
        "undirected_average_degree": float(undirected_average_degree),
        "reciprocity_ratio": _reciprocity_ratio(topology.source, topology.target),
        "giant_component_ratio": giant_component_ratio(topology.num_nodes, topology.source, topology.target),
        "link_success_quantiles": _quantiles(topology.success_probability),
        "query_success_proxy": _query_success_proxy(topology),
    }


def _baseline_suite(graph: CandidateGraph, config: Mapping[str, Any]) -> list[Topology]:
    base_cfg = config.get("baselines", {})
    if not isinstance(base_cfg, Mapping):
        base_cfg = {}
    low_knn_values = base_cfg.get("low_knn_k", [1, 2])
    if not isinstance(low_knn_values, list):
        low_knn_values = [low_knn_values]
    knn_k = int(base_cfg.get("knn_k", 4))
    greedy_degree = int(base_cfg.get("greedy_max_degree", 8))
    augment_k = int(base_cfg.get("mst_augment_k", 2))
    baselines = [
        knn_by_distance(graph, k=int(k), name=f"low_knn_distance_k{int(k)}") for k in low_knn_values
    ]
    baselines.extend([
        knn_by_distance(graph, k=knn_k),
        knn_by_channel_score(graph, k=knn_k),
        degree_capped_greedy(graph, max_degree=greedy_degree),
        mst_backbone_plus_augmentation(graph, max_degree=max(greedy_degree, 4), augment_k=augment_k),
    ])
    return baselines


def classify_environment(candidate: Mapping[str, Any], baselines: Mapping[str, Mapping[str, Any]], config: Mapping[str, Any]) -> tuple[str, list[str]]:
    thresholds = config.get("classification", {})
    if not isinstance(thresholds, Mapping):
        thresholds = {}
    min_candidate_giant = float(thresholds.get("min_candidate_giant_component_ratio", 0.90))
    min_baseline_giant = float(thresholds.get("min_baseline_giant_component_ratio", 0.85))
    min_proxy = float(thresholds.get("min_query_success_proxy", 0.40))
    too_easy_proxy = float(thresholds.get("too_easy_query_success_proxy", 0.985))
    low_degree_easy_proxy = float(thresholds.get("too_easy_low_degree_proxy", 0.96))
    min_avg_candidate_degree = float(thresholds.get("min_avg_candidate_degree", 4.0))
    max_avg_candidate_degree = float(thresholds.get("max_avg_candidate_degree", 24.0))

    reasons: list[str] = []
    candidate_giant = float(candidate["candidate_giant_component_ratio"])
    avg_candidate_degree = float(candidate["average_candidate_degree"])
    if candidate_giant < min_candidate_giant:
        reasons.append(f"candidate graph giant component {candidate_giant:.3f} below {min_candidate_giant:.3f}")
    if avg_candidate_degree < min_avg_candidate_degree:
        reasons.append(f"average candidate degree {avg_candidate_degree:.2f} below {min_avg_candidate_degree:.2f}")
    if avg_candidate_degree > max_avg_candidate_degree:
        reasons.append(f"average candidate degree {avg_candidate_degree:.2f} above {max_avg_candidate_degree:.2f}")

    high_baselines = {
        name: metrics for name, metrics in baselines.items() if not name.startswith("low_knn_")
    } or dict(baselines)
    low_baselines = {
        name: metrics for name, metrics in baselines.items() if name.startswith("low_knn_")
    }

    best_name = None
    best_proxy = -1.0
    best_giant = -1.0
    for name, metrics in high_baselines.items():
        proxy = float(metrics["query_success_proxy"])
        giant = float(metrics["giant_component_ratio"])
        if giant > best_giant or (giant == best_giant and proxy > best_proxy):
            best_name = name
            best_proxy = proxy
            best_giant = giant
    if best_giant < min_baseline_giant:
        reasons.append(f"best baseline giant component {best_giant:.3f} below {min_baseline_giant:.3f}")
    if best_proxy < min_proxy:
        reasons.append(f"best query_success_proxy {best_proxy:.3f} below {min_proxy:.3f}")
    if reasons:
        return "too_hard", reasons

    low_saturated = [
        (name, float(metrics["query_success_proxy"]))
        for name, metrics in low_baselines.items()
        if float(metrics["query_success_proxy"]) >= low_degree_easy_proxy
        and float(metrics["giant_component_ratio"]) >= min_baseline_giant
    ]
    if best_proxy >= too_easy_proxy and low_saturated:
        low_name, low_proxy = max(low_saturated, key=lambda item: item[1])
        return "too_easy", [
            f"{best_name} query_success_proxy {best_proxy:.3f} and {low_name} {low_proxy:.3f} indicate low-budget saturation"
        ]
    low_note = "no low-budget KNN saturated"
    if low_baselines:
        strongest_low = max(float(metrics["query_success_proxy"]) for metrics in low_baselines.values())
        low_note = f"strongest low-budget query_success_proxy {strongest_low:.3f}"
    return "usable", [f"{best_name} is connected enough while {low_note}; best proxy {best_proxy:.3f}"]


def evaluate_environment_config(config: Mapping[str, Any]) -> dict[str, Any]:
    snapshot_cfg = dict(config.get("snapshot", {}))
    snapshot = generate_vehicle_snapshot(snapshot_cfg)
    channel_cfg = ChannelConfig.from_mapping(config.get("channel", {}))
    candidate_cfg = CandidateGraphConfig.from_mapping(config.get("candidate_graph", {}))
    graph = build_candidate_graph(snapshot, channel_cfg, candidate_cfg)
    candidate = candidate_metrics(graph)
    baseline_results = {topology.name: topology_metrics(topology) for topology in _baseline_suite(graph, config)}
    classification, reasons = classify_environment(candidate, baseline_results, config)
    return {
        "config_hash": stable_config_hash(config),
        "seed": int(snapshot["seed"]),
        "node_count": int(np.asarray(snapshot["node_id"]).size),
        "vehicle_count": int(np.sum(np.asarray(snapshot["node_type"], dtype=object) == "vehicle")),
        "rsu_count": int(np.sum(np.asarray(snapshot["node_type"], dtype=object) == "rsu")),
        "candidate_graph": candidate,
        "baselines": baseline_results,
        "classification": classification,
        "classification_reasons": reasons,
        "selected_config": _selected_config_summary(config),
    }


def _selected_config_summary(config: Mapping[str, Any]) -> dict[str, Any]:
    channel = config.get("channel", {})
    candidate = config.get("candidate_graph", {})
    snapshot = config.get("snapshot", {})
    return {
        "seed": snapshot.get("seed"),
        "vehicle_count": snapshot.get("vehicle_count", snapshot.get("explicit_vehicle_count")),
        "tx_power_dbm": channel.get("tx_power_dbm"),
        "mcs_threshold_db": channel.get("mcs_threshold_db"),
        "bandwidth_mhz": channel.get("bandwidth_mhz"),
        "noise_dbm": channel.get("noise_dbm"),
        "interference_proxy_dbm": channel.get("interference_proxy_dbm"),
        "carrier_frequency_ghz": channel.get("carrier_frequency_ghz"),
        "nlos_penalty_db": channel.get("nlos_penalty_db"),
        "candidate_radius_m": candidate.get("radius_m"),
        "max_candidates_per_node": candidate.get("max_candidates_per_node"),
    }


def combine_seed_reports(reports: list[dict[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    classifications = [report["classification"] for report in reports]
    usable_count = classifications.count("usable")
    too_easy_count = classifications.count("too_easy")
    too_hard_count = classifications.count("too_hard")
    candidate_degrees = [report["candidate_graph"]["average_candidate_degree"] for report in reports]
    candidate_giants = [report["candidate_graph"]["candidate_giant_component_ratio"] for report in reports]
    cap_hit_ratios = [report["candidate_graph"]["cap_hit_ratio"] for report in reports]
    pair_checks_per_node = [report["candidate_graph"]["candidate_pair_checks_per_node"] for report in reports]
    build_times = [report["candidate_graph"]["build_wall_time_ms"] for report in reports]
    best_proxy = []
    best_giant = []
    best_mean_out_degree = []
    best_undirected_average_degree = []
    low_budget_proxy = []
    low_budget_giant = []
    for report in reports:
        baseline_values = list(report["baselines"].values())
        best_metrics = max(
            baseline_values,
            key=lambda item: (float(item["giant_component_ratio"]), float(item["query_success_proxy"])),
        )
        best_proxy.append(float(best_metrics["query_success_proxy"]))
        best_giant.append(float(best_metrics["giant_component_ratio"]))
        best_mean_out_degree.append(float(best_metrics["mean_out_degree"]))
        best_undirected_average_degree.append(float(best_metrics["undirected_average_degree"]))
        low_values = [
            metrics for name, metrics in report["baselines"].items() if name.startswith("low_knn_")
        ]
        if low_values:
            strongest_low = max(
                low_values,
                key=lambda item: (float(item["giant_component_ratio"]), float(item["query_success_proxy"])),
            )
            low_budget_proxy.append(float(strongest_low["query_success_proxy"]))
            low_budget_giant.append(float(strongest_low["giant_component_ratio"]))
    if usable_count == len(reports):
        classification = "usable"
    elif too_hard_count:
        classification = "too_hard"
    elif too_easy_count:
        classification = "too_easy"
    else:
        classification = "too_hard"
    limits = config.get("classification", {})
    if not isinstance(limits, Mapping):
        limits = {}
    too_easy_limit = float(limits.get("too_easy_query_success_proxy", 0.985))
    mean_best_proxy = float(np.mean(best_proxy)) if best_proxy else 0.0
    mean_cap_hit_ratio = float(np.mean(cap_hit_ratios)) if cap_hit_ratios else 0.0
    return {
        "config_hash": stable_config_hash(config),
        "classification": classification,
        "full_config": config,
        "seed_reports": reports,
        "summary": {
            "seeds": [report["seed"] for report in reports],
            "usable_seed_count": usable_count,
            "too_hard_seed_count": too_hard_count,
            "too_easy_seed_count": too_easy_count,
            "mean_average_candidate_degree": float(np.mean(candidate_degrees)) if candidate_degrees else 0.0,
            "mean_candidate_giant_component_ratio": float(np.mean(candidate_giants)) if candidate_giants else 0.0,
            "mean_best_query_success_proxy": mean_best_proxy,
            "mean_best_baseline_giant_component_ratio": float(np.mean(best_giant)) if best_giant else 0.0,
            "mean_cap_hit_ratio": mean_cap_hit_ratio,
            "mean_candidate_pair_checks_per_node": float(np.mean(pair_checks_per_node)) if pair_checks_per_node else 0.0,
            "mean_candidate_build_wall_time_ms": float(np.mean(build_times)) if build_times else 0.0,
            "mean_best_baseline_out_degree": float(np.mean(best_mean_out_degree)) if best_mean_out_degree else 0.0,
            "mean_best_baseline_undirected_average_degree": (
                float(np.mean(best_undirected_average_degree)) if best_undirected_average_degree else 0.0
            ),
            "strongest_low_budget_query_success_proxy": float(np.mean(low_budget_proxy)) if low_budget_proxy else 0.0,
            "strongest_low_budget_giant_component_ratio": float(np.mean(low_budget_giant)) if low_budget_giant else 0.0,
            "best_query_success_proxy_exceeds_too_easy_threshold": bool(mean_best_proxy >= too_easy_limit),
            "cap_hit_warning": bool(mean_cap_hit_ratio >= 0.95),
        },
        "selected_config": _selected_config_summary(config),
    }
