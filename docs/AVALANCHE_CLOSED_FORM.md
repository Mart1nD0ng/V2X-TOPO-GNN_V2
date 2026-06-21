# Avalanche Closed Form

M2/M2.1 implements a deterministic differentiable Snowball/Avalanche communication finality evaluator. It models repeated subsampled voting for valid transactions under externally supplied query support probabilities. It does not implement a GNN, topology constructor, training loss, energy objective, or full adversarial Avalanche network safety proof.

## Public API

`src/consensus/avalanche_closed_form.py` exposes:

```python
evaluate_avalanche_closed_form(
    p_correct_query,
    p_wrong_query=None,
    *,
    k,
    alpha,
    beta,
    rounds,
    initial_correct_preference=1.0,
    eps=1e-6,
    temperature=1.0,
)
```

Returned tensors:

- `h_plus`
- `h_minus`
- `p_correct_decision`
- `p_wrong_decision`
- `p_undecided`
- `expected_rounds`
- `C_avalanche_node_mean`
- `D_avalanche_rounds_mean`

Energy is intentionally absent in M2 and belongs to later coupled evaluator/loss phases.

## Parameters

- `k`: validators queried per poll.
- `alpha`: quorum threshold for a sufficient majority. The project evaluator requires a strict majority, `2 * alpha > k`.
- `beta`: consecutive successful quorum threshold.
- `rounds`: maximum polling rounds before timeout.
- `initial_correct_preference`: initial probability mass assigned to the correct-preference chain.
- `eps`: deterministic probability stabilization value.
- `temperature`: optional evaluator configuration used identically in all modes.

## Quorum Probability

For query support probability `x`, the single-round quorum probability is:

```text
H(x; k, alpha) = I_x(alpha, k - alpha + 1)
```

`I_x(a,b)` is the regularized incomplete beta function. The implementation uses `scipy.special.betainc` for the forward value and a custom PyTorch autograd rule for the derivative.

No binomial tail summation, scipy binomial PMF/CDF, random sampling, or simulation-based reliability is used.

M2/M2.1 uses SciPy's CPU implementation for the forward incomplete-beta call. This is acceptable for the current correctness harness, but a GPU-native or vectorized replacement may be needed before high-throughput training.

## Input Probability Contract

Raw upstream query probabilities are validated before stabilization. `p_correct_query` and provided `p_wrong_query` must be finite floating-point tensors in `[0, 1]`, within numerical tolerance. When `p_wrong_query` is provided:

```text
p_correct_query + p_wrong_query <= 1
```

The remaining probability mass is the no-quorum/no-decision outcome for a single polling round. If `p_wrong_query` is omitted, the evaluator uses `1 - p_correct_query`.

## Gradient Formula

The derivative is the beta density:

```text
d/dx I_x(a,b) = x^(a-1) * (1-x)^(b-1) / B(a,b)
```

It is computed in log-domain:

```text
log_pdf = (a-1) log(x)
        + (b-1) log1p(-x)
        - lgamma(a) - lgamma(b) + lgamma(a+b)
```

Inputs are stabilized through the same deterministic rule in training, validation, and deployment:

```text
x_eff = eps + (1 - 2 eps) * clamp_or_temperature_smooth(x)
```

There is no train-only smoothing.

## Topology Sensitivity And Assumptions

M2.1 is static single-node finality under fixed node-level iid query-slot probabilities. For node `i`, the evaluator assumes each of the `k` query slots has the same probability of returning correct support, wrong support, or no quorum-producing response during a round.

M2.2 adds a deterministic sparse topology bridge:

```text
q_ij = topology_weight_ij / sum_j topology_weight_ij
p_correct_query_i = sum_j q_ij link_success_ij u_j
p_wrong_query_i   = sum_j q_ij link_success_ij v_j
p_link_response_i    = sum_j q_ij link_success_ij
p_link_no_response_i = 1 - p_link_response_i
p_neutral_query_i    = p_link_response_i - p_correct_query_i - p_wrong_query_i
p_no_support_query_i = 1 - p_correct_query_i - p_wrong_query_i
```

where `u_j` is node `j`'s correct-preference probability and `v_j` is its wrong-preference probability. V2X topology paths pass both `p_correct_query` and explicit `p_wrong_query` into the closed-form evaluator so no-response mass is not silently converted into wrong support.

Topology query graphs are simple directed peer-query graphs by default. Self-loops and duplicate directed `(src, dst)` peer edges are rejected unless explicitly enabled for diagnostics. If multi-edges are enabled, the bridge keeps query-support semantics over edge slots but reports duplicate counts and separate unique-peer diagnostics. Edge-slot diagnostics such as `out_degree` and `effective_query_degree` must not be interpreted as hard distinct-peer capacity when multi-edges are present.

`p_no_response` is kept only as a backward-compatible alias for `p_no_support_query`. New code should distinguish link no-response (`p_link_no_response`) from a successful query to a neutral peer (`p_neutral_query`). The identity is:

```text
p_no_support_query_i = p_link_no_response_i + p_neutral_query_i
```

Self-loops are rejected by default because a V2X/Avalanche peer query should not query the source node as its own peer. They can be allowed only through an explicit diagnostic path, which reports `self_loop_count`. Multi-edges are separately controlled and report `duplicate_edge_count`.

If deployment samples `k` independent query slots from row distribution `q_i` with replacement, the incomplete-beta quorum formula is exact for the mapped node-level support probability. If deployment queries `k` distinct heterogeneous peers without replacement, the current formula is a mean-field approximation; a Poisson-binomial or saddlepoint extension is needed to model heterogeneous peer probabilities exactly.

The bridge reports edge-slot effective query degree, `1 / sum_j q_ij^2`, and unique-peer effective degree after aggregating duplicate edges by `(src, dst)`. K-related hard distinct-peer feasibility warnings should use `unique_out_degree`, `positive_unique_out_degree`, and `effective_unique_peer_degree`. Effective unique-peer degree below `k` means the iid with-replacement approximation may overstate what hard distinct-peer querying can achieve.

M2.2 is a static one-hop topology-to-query bridge. It does not model time-varying preference propagation across the graph; it only maps a fixed topology, fixed link-success probabilities, and fixed node preference probabilities into one-shot per-round support probabilities.

## Graph-Coupled Mean-Field Recurrence

M2.3 adds a graph-coupled mean-field recurrence over node marginal Snowball states. Each node maintains marginal mass over:

```text
C_0..C_{beta-1}
W_0..W_{beta-1}
C_abs
W_abs
U
```

At each round, current marginal preferences are:

```text
u_i(t) = C_abs_i(t) + sum_r C_{i,r}(t)
v_i(t) = W_abs_i(t) + sum_r W_{i,r}(t)
```

The sparse topology bridge maps neighbor marginals into round-specific support:

```text
c_i(t) = sum_j q_ij link_success_ij u_j(t)
w_i(t) = sum_j q_ij link_success_ij v_j(t)
```

Then the same incomplete-beta quorum rule gives `h_plus_i(t)` and `h_minus_i(t)`, and each node updates its local Snowball marginal state. The expected-rounds diagnostic adds transient mass before each round.

M2.3 is deterministic and differentiable, and it captures topology propagation effects such as weak cuts and bridge bottlenecks better than M2.2. It is still not an exact full joint Avalanche safety proof: the exact joint process over all node states grows exponentially in `num_nodes`, so M2.3 tracks marginals and uses a mean-field closure.

## Small-N Exact Joint Reference

M2.4 adds a test-only exact joint Snowball reference under `tests/consensus/reference_exact_joint_snowball.py` and an explicit comparison utility under `scripts/analysis/compare_mean_field_exact_snowball.py`. This is not a production evaluator and not a training path.

The exact reference enumerates each node's local state space:

```text
C_0..C_{beta-1}, W_0..W_{beta-1}, C_abs, W_abs, U
```

and then enumerates the Cartesian product of local states. It is therefore limited to small cases:

```text
num_nodes <= 5
beta <= 3
rounds <= 6
```

Larger inputs raise `ValueError`. The reference still computes quorum probabilities through the same regularized incomplete-beta function; it does not enumerate quorum tails.

The purpose of M2.4 is to quantify where M2.3 agrees with the exact joint recurrence and where the mean-field closure introduces a measurable approximation gap. Deterministic disconnected components and small deterministic sanity cases should match closely. Shared uncertain parents or nonlinear quorum settings can produce a finite gap, which is expected and should be reported rather than hidden.

## V2X Evaluation Bridge

M3 keeps Avalanche math in the graph-coupled evaluator and adds a separate V2X bridge under `src/evaluation/v2x_consensus_bridge.py`. The bridge maps sparse V2X topology edges and channel proxy diagnostics into `link_success`, then calls M2.3 with explicit initial correct and wrong preferences.

The bridge returns node-wise C/D outputs and graph quantiles:

```text
node_p_correct_decision
node_p_wrong_decision
node_p_undecided
node_expected_rounds
C_avalanche_node_mean / min / p10
D_avalanche_rounds_mean / p90
```

It also returns a consensus energy proxy E from expected query rounds and row-normalized sparse query weights. This proxy is deterministic and differentiable, but it is not a full NR sidelink power model and not an objective by itself. Channel terms such as link success and SINR are diagnostics used by the evaluator bridge, not direct optimization terms.

M3.1 calibrates this bridge in the failure domain. The evaluator reports:

```text
F_i = P_wrong_i + P_undecided_i
reliability_nines_i = -log10(F_i + eps)
```

High `C_avalanche` is a desirable outcome. If `F` is far below the configured target, the case is classified as `above_target_high_reliability`; this means future reliability pressure should weaken and delay/energy metrics should become more important. This is diagnostic preparation for M4 and does not add a training objective.

M3.2 makes that diagnostic tail-aware. The global weakening flag is true only when both mean failure and tail failure are classified as `above_target_high_reliability`. A smooth logsumexp failure tail is also reported as a metric for future M4 barrier design. This remains evaluator calibration only; no GNN training or coupled loss is implemented here.

## M54 Evaluator Profiling

M54 adds a diagnostic profiler around the existing V2X evaluator path. It does
not replace `evaluate_v2x_graph_consensus`; instead it reconstructs the same
sparse channel proxy, topology query-support bridge, graph-coupled Avalanche
recurrence, energy proxy, failure-domain metrics, and diagnostics with timing
hooks.

The graph-coupled section reports recurrence time, per-round mean/max time,
round count, beta, k, alpha, node count, and active edge count. Query-support
profiling reports row normalization, sparse index accumulation, support
probability computation, zero-row handling, and effective-degree diagnostics.
Reliability state is reported from failure probability `F`; a scale smoke can
pass computationally while still being `below_target` in reliability.

M54 remains forward-only. It does not add a backward pass, change the
closed-form recurrence, add dense node-by-node graph tensors, or promote
channel/link diagnostics into direct loss terms.

## M55 Query-Support Optimization Design

M55 is report-only and does not change the closed-form evaluator. It documents
safe future optimization options for the sparse topology query-support bridge
used before graph-coupled Avalanche recurrence.

Any future query-support fast path must preserve:

- explicit `p_correct_query` and `p_wrong_query` computation,
- `p_correct_query + p_wrong_query <= 1` validation,
- neutral and no-support probability identities,
- isolated zero-row behavior,
- gradients through topology weights, link success, and node preferences,
- diagnostics when explicitly enabled,
- no Monte Carlo, no random sampling, no binomial enumeration, and no dense
  NxN graph tensor allocation.

M55 can recommend implementation work such as fused sparse reductions,
CSR-style row pointers, or diagnostics gating, but it does not authorize 10k
backward/training behavior by itself.

## M55.1 Query Support Backend

The graph-coupled evaluator now exposes `query_support_backend="legacy" |
"fused_fast"` and `diagnostics_mode="full" | "lite" | "off"`. These options
change the sparse implementation and diagnostic workload only. They do not
change the Avalanche closed-form recurrence, quorum probabilities, absorbing
chain semantics, or the explicit `p_wrong_query` path.

Legacy/full and fused_fast/full must agree on `p_correct_query`,
`p_wrong_query`, `p_link_response`, `p_neutral_query`, `p_no_support_query`,
response aliases, and enabled diagnostics within tolerance. Lite/off modes
can reduce diagnostic work, but they cannot alter probability tensors or
gradient paths through topology weights, link success, and node preferences.

## Absorbing Chain

The evaluator builds an absorbing Markov chain with `2 * beta + 2` states:

```text
(+, 0..beta-1)    current preference correct, consecutive success count
(-, 0..beta-1)    current preference wrong, consecutive success count
C_abs             absorbed correct decision
W_abs             absorbed wrong decision
```

Transitions from `(+ , c)`:

```text
h_plus  -> (+, c+1) or C_abs when c+1 >= beta
h_minus -> (-, 1) or W_abs when beta == 1
h0      -> (+, 0)
```

Transitions from `(- , c)`:

```text
h_minus -> (-, c+1) or W_abs when c+1 >= beta
h_plus  -> (+, 1) or C_abs when beta == 1
h0      -> (-, 0)
```

where:

```text
h0 = 1 - h_plus - h_minus
```

The implementation rejects incompatible query inputs where `h_plus + h_minus` exceeds one beyond numerical tolerance.

`h0` represents no quorum in that polling round. A no-quorum round resets the current confidence streak to zero and leaves the current preference unchanged, so timeout with no absorption remains undecided.

Decision probabilities are computed by matrix power:

```text
pi_T = pi0 @ M^rounds
p_correct_decision = pi_T[C_abs]
p_wrong_decision = pi_T[W_abs]
p_undecided = 1 - p_correct_decision - p_wrong_decision
```

Expected truncated rounds use a finite-horizon transient recurrence. This avoids singular inverse failures when the transient chain has no absorption:

```text
dist_0 = pi0_Q
expected = 0
for t in 0..R-1:
    expected += sum(dist_t)
    dist_{t+1} = dist_t @ Q
```

## Tests

The consensus tests verify:

- `H(x;k,alpha)` is monotonic.
- `H` matches `scipy.special.betainc`.
- finite-difference and autograd gradients agree.
- decision probabilities move in the expected direction.
- probabilities sum to one.
- expected rounds stay between zero and `rounds`.
- harness checks still reject prohibited implementation patterns.

## Non-Claims

This evaluator is a differentiable communication finality model for valid transactions. It does not claim full adversarial safety, validator economic security, mempool behavior, DAG/Snowman implementation fidelity, or 5G sidelink PHY/MAC correctness.
