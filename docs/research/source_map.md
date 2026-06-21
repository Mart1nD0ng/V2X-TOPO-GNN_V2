# Research Source Map

This map pins the external sources that define the repository's standards,
simulator, topology-learning, consensus, and optimization assumptions. It is a
source-to-implementation index, not a claim that every optional feature is active
in the production training path.

## 3GPP TR 37.885

Title: Study on evaluation methodology of new Vehicle-to-Everything (V2X) use cases for LTE and NR.
Version/release or access date: 3GPP dynamic report, accessed 2026-06-16.
Official URL: https://www.3gpp.org/DynaReport/37885.htm
Implementation facts extracted: V2X evaluation must be scenario-driven, with vehicle density, communication range, and use-case conditions treated as controlled operating variables.
Skill uses it: v2x-urban-environment-sim.
Unresolved uncertainty: The current repository uses a sparse internal simulator rather than a full 3GPP-calibrated system simulator.

## 3GPP TR 38.901

Title: Study on channel model for frequencies from 0.5 to 100 GHz.
Version/release or access date: 3GPP dynamic report, accessed 2026-06-16.
Official URL: https://www.3gpp.org/dynareport/38901.htm
Implementation facts extracted: Channel effects should be exposed through environment and evaluator parameters; LOS/NLOS, path loss, and interference approximations are diagnostics or inputs, not direct optimization losses.
Skill uses it: v2x-urban-environment-sim.
Unresolved uncertainty: Richer 38.901-calibrated PHY remains a validation extension after production training is stable.

## Avalanche/Snowball

Title: Snowflake to Avalanche: A Novel Metastable Consensus Protocol Family for Cryptocurrencies.
Version/release or access date: arXiv 1906.08936, accessed 2026-06-16.
Official URL: https://arxiv.org/abs/1906.08936
Implementation facts extracted: The implemented consensus target is the Snow family and Avalanche/Snowball finality behavior, not PBFT prepare/commit phases.
Skill uses it: avalanche-closed-form-consensus.
Unresolved uncertainty: Repository evaluators use closed-form differentiable approximations rather than Monte Carlo network simulation.

## GradNorm

Title: GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks.
Version/release or access date: ICML/PMLR 2018, accessed 2026-06-16.
Official URL: https://proceedings.mlr.press/v80/chen18a.html
Implementation facts extracted: Adaptive objective balancing should react to gradient magnitudes and task training rates rather than static scalar weights alone.
Skill uses it: coupled-loss-pcgrad-gradnorm.
Unresolved uncertainty: GradNorm is a governance option; it does not by itself prove that every deployment cell improves.

## PCGrad

Title: Gradient Surgery for Multi-Task Learning.
Version/release or access date: NeurIPS 2020, accessed 2026-06-16.
Official URL: https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html
Implementation facts extracted: Conflicting task gradients can be projected to reduce negative transfer, but projection is only justified when directional conflict is observed.
Skill uses it: coupled-loss-pcgrad-gradnorm.
Unresolved uncertainty: In this project, PCGrad must stay subordinate to measured gradient cosine diagnostics.

## PyTorch Geometric

Title: PyTorch Geometric documentation.
Version/release or access date: Documentation homepage, accessed 2026-06-16.
Official URL: https://pytorch-geometric.readthedocs.io/
Implementation facts extracted: Sparse graph batching and message passing patterns motivate O(Nk) candidate processing rather than dense O(N^2) tensors.
Skill uses it: hierarchical-gnn-topology-constructor.
Unresolved uncertainty: The current production path uses repository-native layers; PyG is a reference pattern, not a hard dependency.

## DGL

Title: Deep Graph Library documentation.
Version/release or access date: Documentation/project homepage, accessed 2026-06-16.
Official URL: https://www.dgl.ai/
Implementation facts extracted: Graph learning systems should preserve sparse graph abstractions and avoid materializing dense adjacency where candidate degree is bounded.
Skill uses it: hierarchical-gnn-topology-constructor.
Unresolved uncertainty: DGL is a cross-check for graph-system design choices, not the runtime backend.

## SUMO

Title: Eclipse SUMO - Simulation of Urban MObility.
Version/release or access date: Documentation/project homepage, accessed 2026-06-16.
Official URL: https://eclipse.dev/sumo/
Implementation facts extracted: Mobility and traffic scenarios should be reproducible, parameterized, and separable from the topology constructor.
Skill uses it: v2x-urban-environment-sim.
Unresolved uncertainty: SUMO/ns-3 co-simulation is a validation extension, not required before current production training.

## 5G-LENA

Title: 5G-LENA NR module for ns-3.
Version/release or access date: Project documentation, accessed 2026-06-16.
Official URL: https://5g-lena.cttc.es/
Implementation facts extracted: Full NR simulation should be treated as an external validation harness; the repository's differentiable evaluator must remain lightweight enough for training.
Skill uses it: v2x-urban-environment-sim.
Unresolved uncertainty: 5G-LENA calibration is not yet wired into the default training loop.

## Sionna

Title: Sionna documentation.
Version/release or access date: NVLabs documentation, accessed 2026-06-16.
Official URL: https://nvlabs.github.io/sionna/index.html
Implementation facts extracted: Differentiable wireless blocks are a reference for future PHY realism, especially when gradients through channel approximations are needed.
Skill uses it: v2x-urban-environment-sim.
Unresolved uncertainty: Sionna-style differentiable channel modeling is deferred until the topology constructor and production training path are stable.

## GraphGPS

Title: Recipe for a General, Powerful, Scalable Graph Transformer / GraphGPS.
Version/release or access date: arXiv 2205.12454 and project repository, accessed 2026-06-16.
Official URL: https://github.com/rampasek/GraphGPS
Implementation facts extracted: Scalable graph transformers combine local message passing with global context while preserving sparse graph computation.
Skill uses it: hierarchical-gnn-topology-constructor.
Unresolved uncertainty: GraphGPS informs architecture boundaries; the current model remains a purpose-built topology constructor.
