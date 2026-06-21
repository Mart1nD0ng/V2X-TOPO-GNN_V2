# Model Contract

This repository targets scalable 5G NR-V2X topology construction with a
hierarchical GNN, analytic Avalanche/Snowball reliability, and a coupled
consensus-delay-energy training objective. The model is a topology constructor,
not an independent Bernoulli edge classifier.

## Consensus Contract

- The target consensus model is Avalanche/Snowball, not PBFT.
- Reliability evaluation must remain analytic and differentiable. Monte Carlo,
  random sampling, and binomial enumeration are not allowed in the evaluator used
  for training or deterministic checks.
- Link reliability, SINR, BLER, HARQ, and coverage are internal physical-layer
  variables or diagnostics. They are not direct loss terms.

## Topology Contract

- Training, validation, and deployment use the same hard-forward topology
  construction rule. The obsolete training-soft / validation-hard split is not
  the deployed contract.
- A straight-through surrogate may be used only for backward gradients; the
  forward topology must stay byte-identical to the hard construction rule.
- The shared topology layer is the only component that turns candidate scores
  into the deployed graph.

## Scale And Environment Contract

- Candidate graph and interference approximations must stay sparse and scale
  approximately O(Nk), not O(N^2), for 10k-node scenarios.
- Environment feasibility must pass before GNN training is treated as valid.
