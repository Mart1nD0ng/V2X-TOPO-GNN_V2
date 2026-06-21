# Evaluator Model Audit

## Paper-Derived URLLC Link Contract

The uploaded paper, *Optimizing Resource Allocation in URLLC for Real-Time
Wireless Control Systems*, uses a finite-blocklength URLLC channel model where
link reliability is coupled to SNR, payload, allocated bandwidth, and
transmission duration through the Q function.

For this project, the relevant simplified contract is:

```text
n = N * B0 * T
C = log(1 + SINR)
epsilon = Q((n*C - payload + 0.5*log(n)) / sqrt(n))
link_success = 1 - epsilon
```

where `N * B0` is allocated bandwidth, `T` is the single-hop transmission time,
and `payload` is the packet size in nats. Longer `T` or more bandwidth can reduce
packet error, but they increase latency and transmission energy. This is the
required coupling that was missing from the previous evaluator.

## Current-State Finding

Before this change, evaluator link success was a sigmoid of SINR only:

```text
link_success = sigmoid((SINR_dB - threshold_dB) / transition_width_dB)
```

This made communication delay and energy mostly downstream of Avalanche rounds
and a constant packet-duration energy proxy. The D/E ablation result under
`result/de_ablation_v1/` therefore correctly reported D/E as weakly coupled and
almost inert: edge selection could improve F, while D/E changed only slightly.

## Implemented Correction

`src/evaluation/v2x_consensus_bridge.py` now supports an opt-in finite-blocklength
Q-function reliability path:

- `finite_blocklength_reliability`
- `payload_bits`
- `resource_block_count`
- `subcarrier_spacing_hz`
- `single_hop_delay_s`

The production config enables it and also enables receiver-load-aware
interference:

```yaml
physical:
  interference_density_coupling_db: 10.0
  finite_blocklength_reliability: true
  payload_bits: 100.0
  resource_block_count: 4.0
  single_hop_delay_s: 0.001
```

The selected topology now affects receiver in-load, which changes interference,
which changes finite-blocklength packet error, which changes Avalanche rounds and
therefore delay and energy.

## Production-Scale D/E Result

The load-aware finite-blocklength evaluator passes production-scale training
readiness at 2000 and 10000 nodes. It also exposes a more specific D/E finding:
at the current `small_realistic` Avalanche profile, D/E are near their protocol
lower bounds, not freely optimizable objectives.

For `k=5`, `beta=5`, `single_hop_delay_s=0.001`, and the default energy proxy,
the best possible expected consensus delay is five rounds and the per-node
energy floor is approximately:

```text
E_min = beta * k * single_hop_delay_s * (P_tx + P_rx + P_proc)
      ~= 1.37e-2 J
```

The full `make de-ablation` run reaches `D=5.78` rounds and `E=1.59e-2 J`,
both within about 20% of these lower bounds. The old production targets
(`D=1 round`, `E=1e-4 J`) were physically impossible for this protocol setting,
so they have been replaced with protocol-aware targets (`D=5`, `E=0.014 J`).

The remaining limitation is structural: with fixed active degree and normalized
query weights, `E_consensus_node_mean` is mostly `D_avalanche_rounds_mean` times
a constant per-round query energy. Energy is therefore not independently
controllable unless future topology construction can change active degree,
resource allocation, transmit power, packet duration, or route/hop structure.
