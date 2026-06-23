# Protocol Semantics — True Binary Snowball over NR-V2X

**Status:** Phase 0 decision record (engineering plan §3, technical spec §3).
**Design basis:** `EFFECTIVE_SAMPLING_DYNAMICS_TECHNICAL_SPEC.md` §3, §4; `ESD_GNN_ENGINEERING_PLAN.md` §3, Phase 2.
**Supersedes:** the `src/mainline/snowball.py` streak automaton (a Snowflake-style
consecutive-quorum machine), which is retained only as a *named* Snowflake baseline.

---

## 0. Decision (spec §3.1, the binding choice)

The spec requires one of:

1. **(recommended)** implement a *true binary Snowball* confidence process, or
2. keep the streak automaton but rename the whole project to "Snowflake-like".

**This project adopts option 1: true binary Snowball.** The streak automaton is kept,
unchanged, as an explicitly-labelled `snowflake` baseline for the protocol ablation
(spec §10.1). No code, gate, result, or paper text may call the streak automaton
"Snowball" (spec §12 prohibition).

> **Defaults flagged for user override.** The reliability thresholds, horizon, and
> service profile below are adopted from the spec's *recommended* values
> (spec §3.3, §4.4). They are documented defaults, not user-confirmed requirements.
> A different service profile only changes constants in `OPTIMIZATION_CONTRACT.md`;
> it does not change any of the mathematics or code contracts here.

---

## 1. Colours, ground truth, eligible set

* Each consensus instance has a binary ground truth `Y* ∈ {+1, -1}` (spec §6.1).
* Protocol colours are binary `{+, -}`. We fix the convention **`+` = the colour
  aligned with `Y*`** ("correct"), **`-` = the opposite colour** ("wrong"). The
  protocol itself is colour-symmetric; correctness is a *labelling relative to `Y*`*
  used only by the evaluator and the safety/validity metrics, never by a node.
* Honest eligible set `H` = nodes that participate and can finalize. The threat model
  for the headline is **crash/asynchrony only** (no Byzantine equivocation): all nodes
  run the same honest protocol; "wrong" decisions arise from *correlated evidence*
  (spec §6.2), receiver congestion, and weak cuts — not adversarial votes. Byzantine
  extensions are out of scope for the headline (spec §4 defines `H` over honest nodes).
* **A query policy may never read `Y*` or any peer's current vote/preference**
  (spec §6.3, §10 prohibition). It sees only deployment-observable geometry, history,
  credibility, road region, and physical features.

---

## 2. Per-node state (finite horizon `R_max`)

Following spec §3.2, a node `i` holds

```
X_i(t) = ( d_i(t), pref_i(t), last_i(t), c_i(t), decision_i(t) )
```

| field | domain | meaning |
|-------|--------|---------|
| `d`        | `{-R_max … R_max}` | confidence difference `d⁺ − d⁻` |
| `pref`     | `{+, -}`           | preferred colour = `argmax` confidence (ties keep current) |
| `last`     | `{+, -, ⊥}`        | colour of the most recent successful quorum (`⊥` = none/broken) |
| `c`        | `{0 … β−1}`        | length of the current consecutive same-colour quorum streak |
| `decision` | `{U, +, -}`        | `U` undecided; `+`/`-` absorbing finalized colour |

`d⁺`, `d⁻` are the Snowball confidence counters (number of `+`/`-` quorums ever seen).
Only their difference `d` and the preference are needed for the update (proof in §3).
Over exactly `R_max` rounds, `|d| ≤ R_max`, so the finite-horizon state set is finite;
the reachable subset is enumerated by BFS in `src/protocol/binary_snowball.py`.

**Initial state** (round 0): `d=0`, `last=⊥`, `c=0`, `decision=U`, and
`pref = ` the node's own observation colour `O_i` (spec §6.2). The initial split between
`pref=+` and `pref=-` is set by the environment's per-node correct-observation
probability; it is *not* a free protocol parameter.

---

## 3. One-round update (the defining Snowball rule)

Each round the node samples a `k`-subset, receives responses, and forms a **ternary
quorum outcome** `o ∈ {+, -, 0}` with probabilities `(h⁺, h⁻, h⁰)` from the §5 quorum
DP (`+` = a correct-colour quorum of ≥ `α` votes, `-` = wrong-colour quorum, `0` = no
quorum). The strict-majority constraint `2α > k` makes `+` and `-` mutually exclusive.

```
o = 0  (no quorum):
        last ← ⊥ ;  c ← 0                      # streak broken
        d, pref unchanged                       # confidence PERSISTS  ← Snowball
o = +  (correct quorum):
        d ← d + 1
        if d > 0: pref ← +   elif d < 0: pref ← -   else: pref unchanged
        if last = +: c ← c + 1   else: last ← + ; c ← 1
        if c ≥ β: decision ← pref ; absorb
o = -  (wrong quorum):  symmetric (d ← d−1)
absorbing (decision ≠ U): stay put.
```

**Why `pref` and `last` are both kept.** `pref` follows accumulated confidence
(`sign(d)`); `last`/`c` track the *consecutive* streak. They can disagree: a single
opposite quorum starts a new streak (`last`, `c←1`) but does **not** flip `pref` unless
it drives `d` across zero. Finalization decides `pref` (= `decide(col)` in the
Avalanche pseudocode), which is the max-confidence colour, not necessarily `last`.

**This is exactly what distinguishes Snowball from the retained Snowflake baseline.**
In the Snowflake streak automaton a single opposite quorum flips the preference
immediately (`count ← 1` on the other side). In true Snowball the preference is
*sticky against accumulated confidence*. The discriminating sentinel
(`tests/protocol/test_binary_snowball.py::test_confidence_persistence_vs_snowflake`)
drives `+ + + 0 -` and asserts `pref` stays `+` (Snowball) where the Snowflake
automaton would read `-`.

**No-quorum semantics (documented choice).** A no-quorum round breaks the consecutive
streak (`c←0`) but preserves confidence `d` and `pref`. This is the standard
Snow-family "reset on non-reinforcement" applied to the *streak only*; persisting
confidence across an occasional missed round is the entire point of the Snowball
counters and is what makes it more robust than Snowflake under lossy V2X links.

---

## 4. Reliability and latency semantics (spec §4)

Let honest terminal states be `Y_i ∈ {+, -, U}`.

* **Agreement safety** `F_disagree = P(∃ i,j ∈ H : Y_i ≠ Y_j, Y_i,Y_j ≠ U)`.
* **Validity**       `F_wrong = P(∃ i ∈ H : Y_i = -)` (any honest node finalizes the
  wrong colour). All-wrong probability is also reported.
* **Deadline finality** `F_deadline(T_d) = P(T_all > T_d)`, where
  `T_all = inf{ t : ∀ i ∈ H, Y_i(t) = + }` is the wall-clock time at which every honest
  node has finalized the correct colour (spec §4.3). Round→wall-clock mapping is the
  per-round physical duration `τ_i(t)` from the round-coupled PHY (spec §7), **never** a
  fixed `tau_proxy`.

These three are the **hard constraints**. Reliability is never traded for latency or
energy (spec §4.4, non-negotiable constraint #1).

---

## 5. Exactness boundary (spec §12)

* The per-node finite-horizon chain (`build_transition` / `apply_round`) is the **exact**
  marginal evolution of the true-Snowball state *given* the per-round `(h⁺,h⁻,h⁰)`.
* The analytic *global* evaluator composes these per-node chains under a shared-latent
  correlation model; it is exact **only under that model's independence-given-`Z`
  assumption**, not for the unrestricted coupled process. The independent **dynamic MC**
  (spec §8, Phase 2) is the final judge of absolute safety/deadline and of ranking.
* No claim of "exact Avalanche global reliability" is permitted (spec §12 prohibition).

---

## 6. Acceptance (G1)

1. Hand-computed quorum-sequence trace matches the state path step-for-step.
2. Transition operator is row-stochastic; mass is conserved each round.
3. Confidence-persistence sentinel passes (Snowball ≠ Snowflake).
4. Single-node exact chain matches a brute-force enumeration of quorum-sequence paths.

Small-`N` *joint* exact reference (coupling nodes through the environment) is Phase 2
(`src/protocol/exact_small_n.py`), validated jointly with the dynamic MC under G6.
