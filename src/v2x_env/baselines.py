from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .candidate_graph import CandidateGraph


@dataclass(frozen=True)
class Topology:
    name: str
    num_nodes: int
    source: np.ndarray
    target: np.ndarray
    success_probability: np.ndarray
    distance_m: np.ndarray

    @property
    def edge_count(self) -> int:
        return int(self.source.size)


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

    def union(self, left: int, right: int) -> bool:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return False
        if self.size[left_root] < self.size[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.size[left_root] += self.size[right_root]
        return True


def _topology_from_edge_indices(name: str, graph: CandidateGraph, edge_indices: Iterable[int]) -> Topology:
    indices = np.asarray(list(edge_indices), dtype=int)
    if indices.size == 0:
        return Topology(
            name=name,
            num_nodes=graph.num_nodes,
            source=np.asarray([], dtype=int),
            target=np.asarray([], dtype=int),
            success_probability=np.asarray([], dtype=float),
            distance_m=np.asarray([], dtype=float),
        )
    return Topology(
        name=name,
        num_nodes=graph.num_nodes,
        source=graph.source[indices],
        target=graph.target[indices],
        success_probability=graph.success_probability[indices],
        distance_m=graph.distance_m[indices],
    )


def knn_by_distance(graph: CandidateGraph, k: int = 4, name: str | None = None) -> Topology:
    grouped: dict[int, list[int]] = defaultdict(list)
    for edge_idx, src in enumerate(graph.source):
        grouped[int(src)].append(edge_idx)
    selected: list[int] = []
    for edge_indices in grouped.values():
        edge_indices.sort(key=lambda idx: float(graph.distance_m[idx]))
        selected.extend(edge_indices[:k])
    return _topology_from_edge_indices(name or "knn_distance", graph, selected)


def knn_by_channel_score(graph: CandidateGraph, k: int = 4, name: str | None = None) -> Topology:
    grouped: dict[int, list[int]] = defaultdict(list)
    for edge_idx, src in enumerate(graph.source):
        grouped[int(src)].append(edge_idx)
    selected: list[int] = []
    for edge_indices in grouped.values():
        edge_indices.sort(key=lambda idx: (-float(graph.channel_score[idx]), float(graph.distance_m[idx])))
        selected.extend(edge_indices[:k])
    return _topology_from_edge_indices(name or "knn_channel", graph, selected)


def degree_capped_greedy(graph: CandidateGraph, max_degree: int = 8) -> Topology:
    order = sorted(range(graph.edge_count), key=lambda idx: (-float(graph.channel_score[idx]), float(graph.distance_m[idx])))
    degree = np.zeros(graph.num_nodes, dtype=int)
    selected: list[int] = []
    seen_pairs: set[tuple[int, int]] = set()
    for edge_idx in order:
        src = int(graph.source[edge_idx])
        dst = int(graph.target[edge_idx])
        pair = (src, dst) if src < dst else (dst, src)
        if pair in seen_pairs:
            continue
        if degree[src] >= max_degree or degree[dst] >= max_degree:
            continue
        seen_pairs.add(pair)
        degree[src] += 1
        degree[dst] += 1
        selected.append(edge_idx)
    return _topology_from_edge_indices("degree_capped_greedy", graph, selected)


def mst_backbone_plus_augmentation(graph: CandidateGraph, max_degree: int = 10, augment_k: int = 2) -> Topology:
    order = sorted(range(graph.edge_count), key=lambda idx: (float(graph.distance_m[idx]), -float(graph.channel_score[idx])))
    degree = np.zeros(graph.num_nodes, dtype=int)
    uf = _UnionFind(graph.num_nodes)
    selected: list[int] = []
    seen_pairs: set[tuple[int, int]] = set()
    for edge_idx in order:
        src = int(graph.source[edge_idx])
        dst = int(graph.target[edge_idx])
        pair = (src, dst) if src < dst else (dst, src)
        if pair in seen_pairs:
            continue
        if degree[src] >= max_degree or degree[dst] >= max_degree:
            continue
        if uf.union(src, dst):
            seen_pairs.add(pair)
            degree[src] += 1
            degree[dst] += 1
            selected.append(edge_idx)

    grouped: dict[int, list[int]] = defaultdict(list)
    for edge_idx, src in enumerate(graph.source):
        grouped[int(src)].append(edge_idx)
    for edge_indices in grouped.values():
        edge_indices.sort(key=lambda idx: (-float(graph.channel_score[idx]), float(graph.distance_m[idx])))
        added = 0
        for edge_idx in edge_indices:
            src = int(graph.source[edge_idx])
            dst = int(graph.target[edge_idx])
            pair = (src, dst) if src < dst else (dst, src)
            if pair in seen_pairs:
                continue
            if degree[src] >= max_degree or degree[dst] >= max_degree:
                continue
            seen_pairs.add(pair)
            degree[src] += 1
            degree[dst] += 1
            selected.append(edge_idx)
            added += 1
            if added >= augment_k:
                break
    return _topology_from_edge_indices("mst_backbone_augmented", graph, selected)
