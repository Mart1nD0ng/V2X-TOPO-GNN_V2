# Model Architecture Upgrade — Design Document

This document is the **design contract** for the next model overhaul: replacing the current
"GNN → final-GRU" stack with a literature-grounded, three-axis coupled architecture. It is written
to be implemented incrementally and ablated stage-by-stage, and to be citable in the thesis.

It complements `docs/REMODEL_RESULTS.md` (the evidence chain for the evaluator/objective overhaul #1–#5)
and `docs/TEMPORAL_MODEL_DESIGN.md` (the temporal-model justification). Where those documents fixed the
**training signal**, this one fixes the **function approximator** that consumes it.

> **Status:** S1 **IMPLEMENTED** (Axis A); S2/S3 designed. Implementation is staged S1 → S2 → S3 (§6).
> Every recommendation below names (a) the paper it comes from, (b) the exact file/class it lands in,
> (c) value/cost, and (d) why it is *not* the naive version.
>
> **S1 landed (Axis A).** `src/models/hierarchical_gnn.py` now carries all five Axis-A mechanisms as
> **opt-in, byte-identical-when-off** flags (`attention_heads`, `gcnii_alpha`/`gcnii_lambda`, `jk_mode`,
> `channel_recalibration`/`se_reduction`) plus the `apply_dropedge` utility (train-only, wired into
> `training_smoke.py` via `dropedge_prob`). Threaded through `TemporalGNNScorer` and the training config.
> Tests: `tests/models/test_structural_encoder_upgrades.py` (20 tests) — byte-identical-when-off,
> per-mechanism effect/finiteness/gradient-connectivity, empty-graph safety, and a measured
> **GCNII anti-over-smoothing** result (deep L=12 directional distinctiveness: legacy 0.59 vs GCNII 0.93;
> legacy decays with depth, GCNII stays high). Full mainline suite green (403 passed).
>
> **S1 ablation (F_MC, train-quenched → MC-evaluated; `docs/S1_ABLATION_RESULTS.md`).** At the deployed
> depth 2 every mechanism is **neutral** (all within seed noise of baseline 0.375 — no over-smoothing to
> fix at shallow depth). The decisive result is the **depth sweep**: the legacy additive-residual baseline
> **over-smooths catastrophically** (F_MC 0.38→0.59 from depth 2→8, collapsing on 2/3 seeds), while **GCNII
> and JK stay depth-flat ~0.37–0.39** (≈ 0.22 F_MC better at depth 8). So Axis-A buys nothing at the
> shallow operating point but is the **architectural prerequisite for depth** — i.e. for S2's deeper
> recurrent stacks. Init caveat: these mechanisms require `xavier` (the legacy `deterministic` ramp is
> pathological for them). Carry **GCNII + JK** into S2 as the depth-robust backbone.

---

## 0. Why now — and the keystone dependency

Two facts make an architecture upgrade timely and *justified* (it was not, before):

1. **The evaluator is now faithful (#1).** The quenched closed form replaced the ~22–40× optimistic
   mean-field. A more expressive model trained against a *wrong* evaluator just reward-hacks the error;
   against the corrected evaluator, expressiveness can finally pay off.
2. **Temporal modelling: OPEN, not "now helps" (this bullet's earlier claim is RETRACTED).** The per-frame
   ceiling probe (`docs/TEMPORAL_MODEL_DESIGN.md`) historically concluded "temporal cannot beat memoryless"
   under the broken mean-field evaluator. An interim "4-arm × 3-seed stream" experiment under the corrected
   evaluator suggested a ~12% GRU win — but that result has **no reproducible artifact** (no script / json /
   figure survives; `grep '4-arm'` finds only prose) and was **superseded** by the capacity-matched S2/B1
   ablation (`docs/S2_B1_ABLATION_RESULTS.md`, `result/s2_ablation*`): temporal recurrence is
   **neutral-to-negative on F** at 10/20 dB and **reverses but noisily at 30 dB**. All F here are
   **quenched-evaluator** (train Q=11 / eval Q=21), not Monte-Carlo. The honest status: in this
   fully-observable, deterministic-transition, per-frame-objective sim, memory has no theoretical edge
   (`run_environment_forensics.py` measures markov-prediction error ≈ 0), so the temporal question only
   becomes live once the environment carries real hidden state (P1-1: AR(1) shadow, churn, history-dependent
   objective). graph_gru / carried state therefore stay **opt-in and OFF by default**.

---

## 1. Diagnosis — the current model in the literature's coordinate system

Read from `src/models/hierarchical_gnn.py`, `src/models/temporal_scorer.py`, `src/topology/construction.py`.

| Component | Current implementation | Literature name | Weakness |
|---|---|---|---|
| Message passing | `SparseMessagePassingBlock`: **mean**-aggregate (`index_add`/counts) → `update_mlp` → **residual** `node_embedding + updated` | Naive MLP-MPNN, **mean aggregation** | Mean is a low-pass filter → over-smoothing; *for an edge scorer this is acute*: if `h_src ≈ h_dst`, `edge_score_head([h_src‖h_dst‖e])` loses discriminability |
| Multi-scale | `region_context`: per-region **mean-pool** → `region_update` MLP | One hand-defined hierarchy level | Coarse scale is *prescribed*, not *learned* |
| Temporal | `TemporalGNNScorer`: `encode_nodes` → per-node `GRUCell(H,H)` → `edge_head([state[src]‖state[dst]‖e])` | **Textbook "GNN → GRU" (encode-then-recur)** | The exact naive stack being replaced; GRU is **per-node independent** (state does not flow over the topology); docstring self-admits "**node identity must be stable across frames**" → breaks under vehicle churn |
| Domain / regime | none | no conditioning | one model cannot serve the whole density × SINR × scenario envelope |
| Channels | none | no channel recalibration | #3 added heterogeneous channels (reliability, dF/dic) but nothing reweights them per regime |

**Three real problems** (consolidated, not five):
1. **Structural axis** — mean-pool MPNN, over-smoothing, non-attentional.
2. **Temporal axis** — naive encode-then-recur, and **fragile to node churn** (N varies as vehicles enter/leave; per-node recurrence assumes a fixed population).
3. **Domain axis** — absent. And for a **surrogate-trained** constructor the dominant domain gap is
   **surrogate-vs-MC** (objective mismatch), not urban-vs-highway.

What is *already* right and must be preserved: **permutation equivariance** + **inductive sparse message
passing** (no fixed-N assumption) + **channel-as-graph** edge features (the #2 NR-V2X PHY: path loss / SINR /
mode-2 collision feed `edge_features`). These match the wireless-GNN canon (Shen et al. JSAC 2021; Eisen &
Ribeiro REGNN, TSP 2020) and the graphon-transferability justification for density generalization
(Ruiz/Gama/Ribeiro, NeurIPS 2020). **Do not break them.**

---

## 2. Design principle — decouple three axes, then COUPLE them

The naive GNN→LSTM does structure-then-time **serially** and ignores domain entirely. The design splits into
three axes and **couples** them explicitly:

```
              ┌─────────── domain axis  z = (density, SINR, load, scenario) ──────────┐
              │                  FiLM(γ,β) modulates every layer                       │
              ▼                                                                        ▼
node/edge → [STRUCTURAL: GATv2 attention + GCNII initial-residual + JK fusion + DropEdge + ECA channel-recal]
              │  per-layer hidden ↓                                                     │
              ▼                                                                        │
            [TEMPORAL: graph-coupled GRU (state diffuses along the CURRENT topology) + id-keyed memory]
              ▼                                                                  ← slow reliability state
            edge head → row-softmax top-k constructor (interface unchanged) → differentiable reliability (#1)
```

The interface to `TopologyConstructionLayer` and the quenched evaluator is **unchanged** — this is a swap of
the function approximator, not of the training contract.

---

## 3. Axis A — Structural encoder (deep, attentional, anti-over-smoothing)

Lands in `src/models/hierarchical_gnn.py` (`SparseMessagePassingBlock`, `HierarchicalGNNScorer`).

### A1 — GATv2 dynamic attention (replace mean aggregation) — **highest single-point value**
- **Paper:** Brody, Alon, Yahav, *How Attentive are Graph Attention Networks?* (GATv2), ICLR 2022.
- **Mechanism:** GAT computes **static** attention (the neighbour ranking is identical for every query node).
  GATv2 moves the linear layer outside the nonlinearity: `e_ij = aᵀ·LeakyReLU(W[h_i‖h_j])` → **query-conditioned (dynamic)** attention, same parameter cost.
- **Why not naive:** topology construction **is** a query-conditioned ranking problem (vehicle *i*'s best
  partner depends on *i*'s own state). This is exactly GAT's static-attention failure and GATv2's fix. The
  attention logit and our `edge_score` are *the same quantity* — they can be tied / auxiliary-supervised.
- **Landing:** the aggregation segment of `SparseMessagePassingBlock.forward` (the mean `index_add` → an
  attention-weighted `index_add` keyed by `softmax_j(e_ij)` over each dst-node's incoming edges).
- **Cost:** low; parameter count comparable.

### A2 — GCNII initial-residual + identity mapping (for depth)
- **Paper:** Chen et al., *Simple and Deep GCN* (GCNII), ICML 2020.
- **Mechanism:** `H^(l+1) = σ( ((1−α)ÂH^(l) + α·H⁰)·((1−β_l)I + β_l W^(l)) )`, α≈0.1, β_l = log(λ/l+1).
- **Why not the existing residual:** we already have a *previous-layer* residual (`node_embedding + updated`),
  but it does not stop deep over-smoothing or weight degeneration. The **initial residual to H⁰** keeps each
  vehicle's *own* features (position/velocity/ic/#3-reliability) alive to the final layer — necessary for the
  edge head to stay discriminative.
- **Landing:** add an α-mix of the encoder output `H⁰` inside `SparseMessagePassingBlock`.
- **Cost:** low; benefit only material when `message_layers > 3`.

### A3 — Jumping-Knowledge fusion (degree heterogeneity)
- **Paper:** Xu et al., *Jumping Knowledge Networks* (JKNet), ICML 2018.
- **Mechanism:** concat/max the per-layer outputs into the head → each node selects its own effective depth.
- **Why for V2X:** the graph is severely degree-heterogeneous (an RSU/hub vehicle over-smooths after 1–2 hops;
  an edge vehicle needs more). JK lets each node contribute the right-scale embedding.
- **Landing:** `edge_score_head` input becomes a concat of all `message_blocks` outputs instead of only the last.
- **Cost:** low (concat).

### A4 — DropEdge (near-free regularizer + augmentation)
- **Paper:** Rong et al., *DropEdge*, ICLR 2020.
- **Mechanism:** randomly drop a fraction of candidate edges each training step.
- **Why apt here:** our candidate edge set is already a noisy, mobility-derived set — dropping edges is both
  augmentation and robustness to topology noise. **Train-only** (off at eval/deploy → deployment-faithful).
- **Landing:** sample `edge_mask` in the training harness (`src/training/training_smoke.py`).
- **Cost:** ~0.

### A5 — ECA / SE channel recalibration ("特征通道重建")
- **Papers:** Wang et al., *ECA-Net*, CVPR 2020; Hu et al., *Squeeze-and-Excitation*, CVPR 2018.
- **Mechanism:** "squeeze" = global pool **over nodes** → per-channel descriptor; "excite" = per-channel
  sigmoid gate. ECA uses a parameter-light 1-D conv over channels (no SE bottleneck FC).
- **Why for us:** #3 introduced heterogeneous channels (position vs velocity vs ic vs carried-reliability vs
  dF/dic) whose **relevance varies by regime** (information-routing at 0 dB vs load-coupling at 20 dB). A
  channel gate lets the model emphasize the right channels per frame. Our hidden dim is small → **prefer ECA**.
- **Landing:** one ECA block before `edge_score_head`.
- **Cost:** very low.

---

## 4. Axis B — Temporal coupling (the actual replacement for GNN→LSTM)

Lands in `src/models/temporal_scorer.py` (`TemporalGNNScorer`). The current model is the textbook naive version
(`new_state = self.cell(node_embedding, state)` is per-node and topology-blind). Three principled paths,
ranked, with the **node-churn** caveat flagged.

### B1 — (primary) Graph-coupled recurrence — DCRNN / AGCRN
- **Papers:** Li et al., *DCRNN*, ICLR 2018; Bai et al., *AGCRN*, NeurIPS 2020.
- **Mechanism:** replace the GRU's internal linear transforms with **graph diffusion over the current frame's
  topology** (a "DCGRU/AGCRN cell"). The slow reliability state is then updated **through the constructed
  topology**, mirroring how consensus reliability actually spreads.
- **AGCRN-NAPL add-on:** node-adaptive parameters (a shared weight pool selected by a per-vehicle embedding) —
  vehicles are **not** exchangeable (relay vs edge roles), and NAPL captures that while still generalizing.
- **Landing:** swap `torch.nn.GRUCell` for a graph-conditioned cell that takes `(node_embedding, state,
  src_index, dst_index)`.

### B2 — (node-churn robustness) EvolveGCN / TGN memory
- **Papers:** Pareja et al., *EvolveGCN*, AAAI 2020; Rossi et al., *TGN*, 2020.
- **Problem it fixes:** the current docstring concedes "node identity must be stable across frames." Vehicles
  entering/leaving change N and **break per-node recurrence at the boundary**. EvolveGCN evolves the
  *edge-scorer weights* with a small RNN (no per-node state → churn-immune). The lighter fix keeps per-node
  memory but **keys it by a persistent vehicle id** (TGN memory module), zero-initialising new entrants.
- **Recommendation:** keep per-node memory but make it **id-keyed** (the minimal correctness fix), and add
  EvolveGCN-style weight evolution only if churn proves severe.

### B3 — (cheapest correct refactor) ROLAND
- **Paper:** You et al., *ROLAND*, KDD 2022.
- **Mechanism:** instead of one GRU on the *final* node features, put a lightweight GRU update on the hidden
  state at **every** GNN layer → the slow state is maintained hierarchically/multi-scale. Its incremental
  live-update training matches our streaming-frame regime.
- **Use:** the lowest-risk first step if we want temporal gains before committing to B1.

> **Recommended combination:** B1 (graph-coupled GRU) as the core + B2's id-keyed memory for churn. If a
> low-risk first cut is preferred, ship B3 first.

---

## 5. Axis C — Domain / "域自适应" (do NOT use adversarial DA)

This is where the literature verdict is sharpest. For a **surrogate-trained** constructor whose shift axes are
**measurable scalars** (density, SINR, load, scenario):

### C1 — FiLM conditioning + domain randomization — the correct realization of "域自适应"
- **Papers:** Perez et al., *FiLM*, AAAI 2018; Tobin et al., *Domain Randomization*, IROS 2017;
  Gulrajani & Lopez-Paz, *In Search of Lost Domain Generalization* (DomainBed), ICLR 2021.
- **Mechanism:** z = [density, SINR operating point, load, scenario one-hot] → small MLP → per-feature affine
  γ(z), β(z) modulating each GNN layer. **Rather than forcing invariance to the regime, give the regime to the
  model** so it *specializes* (sparse topology under heavy interference, denser under clean SINR). This also
  folds #4/#5's regime-specific behaviours into a single model.
- **Domain randomization:** widen the z envelope in training. DomainBed/GOOD's hard result is **coverage beats
  clever objectives** — a well-tuned ERM over a broad randomized distribution matches/beats DANN/IRM.
- **Landing:** add a regime channel to the harness; FiLM layers inside `HierarchicalGNNScorer`.

### C2 — The real domain gap: surrogate-vs-MC (objective mismatch) — first-class, not an afterthought
- **Framing:** Lambert et al., *Objective Mismatch in MBRL*, L4DC 2020; Janner et al., *MBPO*, NeurIPS 2019.
- **Risk:** the planner will **exploit exactly where the quenched mean-field is wrong** (it under-prices
  correlated/tail failures) — reward-hacking the surrogate.
- **Discipline (all compatible with no-MC-in-training):** (i) keep a periodic **Avalanche-MC oracle** in the
  validation/recalibration loop (surrogate proposes, MC judges); (ii) **uncertainty-penalize** the surrogate
  where the mean-field is least trustworthy (high-density / high-correlation); (iii) **hold-out evaluation
  always against MC, never the surrogate.**

---

## 6. What is overkill / deferred (honest scope)

- **Graph U-Net / DiffPool / SAGPool (pooling family):** *harmful* to an edge scorer — pooling destroys the
  per-edge resolution the scorer needs, and dense assignment matrices are O(N²) (bad for frame streams). The
  **one meaningful exception**: upgrade the hand-defined `region_context` into a **learned platoon hierarchy**
  — that is the correct landing for the "U-Net" idea in our setting, but it is a scope expansion → deferred.
- **Adversarial graph DA (DANN / UDA-GCN / AdaGCN):** training-unstable and DomainBed shows no average win over
  ERM. Fall back to **CORAL/MMD** (one second-order-alignment loss term) *only* if a pure unlabeled sim-to-real
  gap survives C1.
- **IRM / invariant-subgraph learning:** 2023–24 results **prove** graph invariant learning fails without
  environment labels — exactly our label-free setting. **Cite as motivation, do not implement.**

---

## 7. Staged implementation plan (ablation-friendly)

| Stage | Content | Files touched | Independently-ablatable metric |
|---|---|---|---|
| **S1 ✅ IMPLEMENTED** (decoupled from the temporal debate; lowest risk) | A1 GATv2 + A2 GCNII residual + A3 JK + A4 DropEdge + A5 ECA/SE — all opt-in, byte-identical off | `src/models/hierarchical_gnn.py`, `src/models/temporal_scorer.py`, `src/training/training_smoke.py`, `tests/models/test_structural_encoder_upgrades.py` | F_MC vs current baseline under the **same** quenched evaluator (training-ablation pending) |
| **S2 ⚠ B1 NULL** | B temporal coupling — **B1 graph-coupled GRU implemented** (`temporal_cell="graph_gru"`, opt-in, byte-identical-off) but the **capacity-matched ablation REFUTES it as a reliability mechanism** (`docs/S2_B1_ABLATION_RESULTS.md`): at identical 141k params, memory and graph-coupling are neutral-to-slightly-negative on held-out F at 10/20 dB, and temporal arms are less stable than the memoryless `static` scorer; the historical GRU-vs-static win was a capacity/init artifact. Only B1 effect = modest churn reduction. **Keep opt-in; do not default. Reconsider S2 priority vs S3.** | `src/models/temporal_scorer.py`, `tests/models/test_temporal_scorer.py`, `scripts/analysis/run_s2_temporal_ablation.py` | capacity-matched ablation (done) |
| **S3 ⚠ NULL** | C1 FiLM + domain randomization — **implemented** (`regime_dim>0`, opt-in, byte-identical-off, identity-at-init) but the ablation is a **null with negative headroom** (`docs/S3_FILM_ABLATION_RESULTS.md`): the per-regime specialist is the *worst* arm in both hidden and observable modes (one mixture-trained policy generalizes best), so FiLM/specialization can't help; the realistic regime is already self-adapted via channel features. **Keep opt-in; not default.** | `src/models/hierarchical_gnn.py`, `scripts/analysis/run_s3_film_ablation.py` | regime-averaged F, hidden vs observable (done) |
| **Throughout** | C2 surrogate-vs-MC discipline | evaluation side | surrogate optimism / MC hold-out |

**Decisions for each stage:**
- S1 is fully independent of "does temporal help" → do it first, get numbers immediately.
- S2 **must** include a capacity-matched ablation (same parameter count, same init, only recurrence differs)
  to cleanly separate "recurrence" from the "capacity/init" confound flagged in §0.
- Keep every new mechanism **opt-in and byte-identical when off** (the project's standing invariant), so each
  can be ablated in isolation and the baseline is always reproducible.

---

## 8. Bibliography (for the thesis)

- GATv2 — Brody, Alon, Yahav, ICLR 2022 — arXiv:2105.14491
- GCNII — Chen, Wei, Huang, Ding, Li, ICML 2020 — arXiv:2007.02133
- JKNet — Xu, Li, Tian, Sonobe, Kawarabayashi, Jegelka, ICML 2018 — arXiv:1806.03536
- DropEdge — Rong, Huang, Xia, Bian, Huang, ICLR 2020 — arXiv:1907.10903
- SE-Net — Hu, Shen, Sun, CVPR 2018 — arXiv:1709.01507; ECA-Net — Wang et al., CVPR 2020
- DCRNN — Li, Yu, Shahabi, Liu, ICLR 2018 — arXiv:1707.01926
- AGCRN — Bai, Yao, Li, Wang, Wang, NeurIPS 2020 — arXiv:2007.02842
- EvolveGCN — Pareja et al., AAAI 2020 — arXiv:1902.10191
- TGN — Rossi et al., 2020 — arXiv:2006.10637
- ROLAND — You, Du, Leskovec, KDD 2022 — arXiv:2208.07239
- FiLM — Perez, Strub, de Vries, Dumoulin, Courville, AAAI 2018 — arXiv:1709.07871
- Domain Randomization — Tobin et al., IROS 2017
- DomainBed — Gulrajani & Lopez-Paz, ICLR 2021
- Objective Mismatch in MBRL — Lambert et al., L4DC 2020; MBPO — Janner et al., NeurIPS 2019
- GNN for Scalable Radio Resource Management — Shen et al., IEEE JSAC 2021 — arXiv:2007.07632
- REGNN — Eisen & Ribeiro, IEEE TSP 2020 — arXiv:1909.01865
- Graphon transferability of GNNs — Ruiz, Chamon, Ribeiro, NeurIPS 2020 — arXiv:2112.04629
- Decentralized control with GNNs — Tolstaya et al., CoRL 2019 — arXiv:1903.10527

---

## 9. Consolidated conclusion — the model was not the bottleneck

All three axes were implemented (opt-in, tested, byte-identical-when-off) and ablated against the true
held-out failure under the corrected quenched evaluator. The honest, coherent result:

| axis | mechanism | verdict at the tested operating points |
|---|---|---|
| **S1** structural | GATv2 / GCNII / JK / ECA / DropEdge | **neutral at the deployed depth**; GCNII/JK are validated **depth enablers** (legacy over-smooths F 0.38→0.59 at depth 8; GCNII/JK hold ~0.37–0.39) — value only if we go deep |
| **S2** temporal | graph-coupled GRU (B1) | **null at 10/20 dB** under capacity-matching; the historical GRU-vs-static win was a capacity/init artifact. BUT at **30 dB** (`result/s2_ablation_hard30_more`, n=6) the ordering **reverses** — the capacity-matched 141k `no_memory` beats the 91.8k `static` — so this is **not** universal. Temporal stays OPEN (P2-1). |
| **S3** domain | FiLM regime conditioning | **null with negative headroom** at the tested points; a single mixture-trained policy generalizes across regimes better than per-regime specialists; observable regimes self-adapt via features |

(All F in this table are the **quenched** evaluator currency, train Q=11 / eval Q=21 — not Monte-Carlo.)

**Meta-finding (scoped to 10/20 dB).** At the 10/20 dB operating points tested, the function approximator
was not the reliability bottleneck: a well-tuned memoryless, shallow scorer is at/near the achievable
failure floor there. **This does NOT generalize:** at 30 dB strong coupling the shallow `static` scorer
is the *worst* arm and added capacity *does* move F (`result/s2_ablation_hard30_more`), so "shallow is at
the floor" is a 10/20 dB statement. The keystone reliability levers are in the **evaluator and objective**
— `#1` (quenched closed form), `#4` (effective query degree), `#5` (spatial initial confidence). The
architecture verdict was also reached on a fully-observable, deterministic-transition sim where memory
has no theoretical edge; **P1-1 adds the missing hidden state (AR(1) shadow, churn) and P2-1 re-asks the
architecture/temporal question on that realistic substrate with a sensitivity-gated protocol** before any
"stop adding model machinery" conclusion can be claimed beyond the narrow tested points.

**What the axes are still good for (kept opt-in):** S1 GCNII/JK if a future model genuinely needs depth;
S2 B1 if topology **churn / stability** is elevated to a first-class objective; S3 FiLM if a genuinely
**exogenous, hidden** shift with positive specialization headroom appears (true sim-to-real). The most
likely remaining high-value work is **not** model architecture but the **surrogate-vs-MC discipline**
(C2) and broadening the training **data distribution** (domain randomization), which the DomainBed/GOOD
evidence — and this S3 null — both point to.

---

*Design doc. S1/S2/S3 implemented and ablated. Each stage keeps the `TopologyConstructionLayer` and
quenched-evaluator interfaces unchanged and is opt-in / byte-identical-when-off.*
