"""G11 -- reliability-constrained superiority headline + the deferred CDQ-vs-ESP decision (D18->C).

Trains the ESD-GNN in BOTH modes (``use_cdq`` True/False) across several model seeds on a small
scene distribution, then scores -- on a disjoint set of larger held-out scenes, under paired CRN
via the canonical dynamic MC -- the trained policies against capability-matched baselines
(uniform, distance). Demonstrates train-small / deploy-large transfer (the GNN uses scale-agnostic
structural features, constraint #3/#5) and answers two questions with paired bootstrap + Bonferroni:

  1. Does the ESD-GNN beat the heuristic baselines on reliability (F_wrong)?
  2. Does CDQ beat ESP?  (the D18 question, deferred to scale by the user)

Ideal-link mode isolates the query-topology effect (constraint #2: headline optimizes topology,
power/blocklength fixed); the full-physics operating point is a follow-up. MC is the sole judge
at this scale (constraint #14). Run: ``python -m scripts.analysis.headline_comparison``.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from src.environment import ProtocolConfig, RoundPhysicsConfig, build_manhattan_scene, build_scenario
from src.models import ESDGNN, ESDGNNConfig, ESDGNNQueryPolicy
from src.optimization import (
    ReliabilityThresholds,
    compare_to_reference,
    evaluate_policies_paired,
    train_esd_gnn,
)
from src.sampling import DistanceQueryPolicy, UniformQueryPolicy

PHY = RoundPhysicsConfig(subchannels=12, slots_per_window=50)
PROTO = ProtocolConfig(k=3, alpha=2, beta=3, r_max=12)
# Defaults run a full-physics CONFIRMATION at representative scale (fits the 10-min tool cap).
# The publication-grade run is the SAME script with N_MODEL_SEEDS>=5, N_HELDOUT>=30, larger
# EVAL_GX/TRIALS -- a multi-hour offline job (no code change, just these constants).
N_MODEL_SEEDS = 2
N_HELDOUT = 6
TRIALS = 100
TRAIN_STEPS = 25
EVAL_GX = 4                      # held-out scene size (N ~ 96); train scenes are gx=3 (N~48)
EVAL_LINK = None                 # None = FULL physics chain (constraint #7); 1.0 = ideal-link


def _scene(gx, seed):
    sc = build_manhattan_scene(gx, gx, 4, block_m=110.0, comm_radius=95.0, int_radius=140.0,
                               generator=torch.Generator().manual_seed(seed))
    ev = build_scenario("one_biased_region", sc, base_node_err=0.05, region_bias=0.92)
    return sc, ev


def _train(use_cdq, model_seed, train_inst):
    torch.manual_seed(model_seed)
    model = ESDGNN(ESDGNNConfig(hidden_dim=24, r=4, n_enc=3, n_refine=2, k=3,
                                use_cdq=use_cdq)).double()
    train_esd_gnn(model, train_inst, PROTO, PHY, ReliabilityThresholds(),
                  steps=TRAIN_STEPS, lr=0.01, eta_mu=8.0, link_override=1.0)
    return model


def run() -> dict:
    train_inst = [_scene(3, s) for s in range(3)]                 # small training scenes (N~72)
    heldout = [(*_scene(EVAL_GX, 500 + s), 500 + s) for s in range(N_HELDOUT)]   # larger held-out
    N_eval = heldout[0][0].num_nodes
    print(f"training {N_MODEL_SEEDS} CDQ + {N_MODEL_SEEDS} ESP models on N={train_inst[0][0].num_nodes} "
          f"scenes; eval on {N_HELDOUT} held-out N={N_eval} scenes, {TRIALS} trials")
    cdq_models = [_train(True, s, train_inst) for s in range(N_MODEL_SEEDS)]
    esp_models = [_train(False, s, train_inst) for s in range(N_MODEL_SEEDS)]
    print("training done; scoring under paired CRN")

    def make_policies(scene):
        pols = {"uniform": UniformQueryPolicy(), "distance": DistanceQueryPolicy(beta_per_m=0.03)}
        for i, m in enumerate(cdq_models):
            pols[f"cdq_s{i}"] = ESDGNNQueryPolicy(m, scene)
        for i, m in enumerate(esp_models):
            pols[f"esp_s{i}"] = ESDGNNQueryPolicy(m, scene)
        return pols

    scores = evaluate_policies_paired(heldout, make_policies, PROTO, PHY, num_trials=TRIALS,
                                      link_override=EVAL_LINK, verbose=True)

    # aggregate model seeds into one "policy" per mode = per-scene mean over seeds
    def _agg(prefix):
        from src.optimization import PolicyScores
        members = [scores[n] for n in scores if n.startswith(prefix)]
        agg = PolicyScores(prefix.rstrip("_"))
        for key in ("F_wrong", "F_disagree", "latency", "latency_cvar", "energy", "finished_fraction"):
            cols = [m.metric(key) for m in members]
            getattr(agg, key).extend([sum(c[i] for c in cols) / len(cols) for i in range(N_HELDOUT)])
        return agg

    agg = {"uniform": scores["uniform"], "distance": scores["distance"],
           "esd_gnn_cdq": _agg("cdq_s"), "esd_gnn_esp": _agg("esp_s")}

    def _mean(s, k):
        v = s.metric(k); return sum(v) / len(v)

    summary = {n: {k: _mean(s, k) for k in
                   ("F_wrong", "F_disagree", "latency", "latency_cvar", "energy", "finished_fraction")}
               for n, s in agg.items()}
    vs_uniform = compare_to_reference(agg, "uniform", metric="F_wrong")
    cdq_vs_esp = compare_to_reference({"esd_gnn_esp": agg["esd_gnn_esp"], "esd_gnn_cdq": agg["esd_gnn_cdq"]},
                                      "esd_gnn_esp", metric="F_wrong")[0]
    out = {
        "config": {"model_seeds": N_MODEL_SEEDS, "heldout_scenes": N_HELDOUT, "trials": TRIALS,
                   "N_train": train_inst[0][0].num_nodes, "N_eval": N_eval,
                   "link": "ideal" if EVAL_LINK is not None else "full_physics"},
        "summary_mean": summary,
        "vs_uniform_Fwrong": [c.__dict__ for c in vs_uniform],
        "cdq_vs_esp_Fwrong": cdq_vs_esp.__dict__,
    }
    print("\n=== mean over held-out scenes ===")
    for n, s in summary.items():
        print(f"  {n:14s} F_wrong={s['F_wrong']:.4f}  F_disagree={s['F_disagree']:.4f}  "
              f"lat={s['latency']:.3f}  latCVaR={s['latency_cvar']:.3f}  "
              f"energy={s['energy']:.4f}  finished={s['finished_fraction']:.3f}")
    print("\n=== ESD-GNN vs uniform (paired, F_wrong; negative = GNN better) ===")
    for c in vs_uniform:
        print(f"  {c.name:14s} mean_diff={c.mean_diff:+.5f}  CI=({c.ci[0]:+.4f},{c.ci[1]:+.4f})  "
              f"sig={c.significant} better={c.better}")
    print("\n=== CDQ vs ESP (paired, F_wrong; negative = CDQ better) ===")
    print(f"  mean_diff={cdq_vs_esp.mean_diff:+.5f}  CI=({cdq_vs_esp.ci[0]:+.4f},{cdq_vs_esp.ci[1]:+.4f})  "
          f"significant={cdq_vs_esp.significant}  CDQ_better={cdq_vs_esp.better}")
    return out


if __name__ == "__main__":
    res = run()
    p = Path("result/headline"); p.mkdir(parents=True, exist_ok=True)
    (p / "headline.json").write_text(json.dumps(res, indent=2, default=str))
    print(f"\nwrote {p / 'headline.json'}")
