"""Campaign A, Lever 1 follow-up (b): DISTILLATION diagnosis -- is the GNN's parity a feature, capacity, or
generalization gap?

The MC oracle (free per-edge logits) beats distance by +0.10, but the trained GNN only matched distance. Why?
Use the per-scene oracle logits as a SUPERVISED target and distil the GNN to predict them:
  * in-sample fit LOW  -> FEATURE/CAPACITY gap (the GNN's geometric features can't even represent the oracle
    on scenes it trained on). Disambiguated by a capacity sweep (hidden_dim 16 vs 64): if the big GNN fits and
    the small one doesn't, it is capacity; if neither fits, it is FEATURES (the info the oracle uses is absent).
  * in-sample fit HIGH, held-out MC of the distilled GNN >> distance -> the headroom is GNN-REACHABLE and
    generalizes -> the original parity was a TRAINING/objective failure (REINFORCE didn't find what distillation
    can). in-sample HIGH but held-out corr/MC ~ distance -> GENERALIZATION gap (the per-scene oracle doesn't
    transfer).
Per-node centering removes the ESP law's per-node constant freedom; fit = per-node Pearson corr (scale/offset
invariant -- only the within-node ranking drives the k-subset).

Run: PYTHONPATH=. python docs/gate_evidence/esp_scale_v2/run_oracle_distill.py [--smoke]
"""
from __future__ import annotations

import json
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

from src.config.experiment_spec import build_experiment_spec
from src.config.service_profile import ConsensusServiceProfile
from src.environment import ProtocolConfig, RoundPhysicsConfig
from src.environment.candidate_graph import build_candidate_graph
from src.evaluation.esp_baselines import free_logit_policy, train_mc_edge_logit_oracle
from src.evaluation.esp_scale import _esp_config, build_scale_instance
from src.metrics import manifest as mf
from src.metrics import schema
from src.metrics.participation import uniform_participation
from src.models import ESDGNN
from src.models.esd_gnn import ESDGNNQueryPolicy, build_scene_features
from src.sampling.baseline_policies import DistanceQueryPolicy
from src.validation import run_dynamic_mc

SMOKE = "--smoke" in sys.argv
HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "oracle_distill_results.json")
PHY = RoundPhysicsConfig(subchannels=10, slots_per_window=40)
PROTO = ProtocolConfig(k=3, alpha=2, beta=3, r_max=6)
PROFILE = ConsensusServiceProfile.urban_default().replace(k=3, alpha=2, beta=3, max_poll_epochs=6)
GRID, SCEN, BE, CORR = (5, 5, 3), "matched_marginal_high", 0.35, 0.25
TRAIN_SCENES = [20, 21] if SMOKE else [20, 21, 22, 23]
HELDOUT_SCENES = [30] if SMOKE else [30, 31]
ORACLE_STEPS = 4 if SMOKE else 100
ORACLE_TRIALS = 20 if SMOKE else 120
DISTILL_STEPS = 30 if SMOKE else 500
HIDDEN_DIMS = [16] if SMOKE else [16, 64]
EVAL_TRIALS = 40 if SMOKE else 1500


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _per_node_center(x, src_index, N):
    sums = x.new_zeros(N).index_add(0, src_index, x)
    cnt = x.new_zeros(N).index_add(0, src_index, torch.ones_like(x))
    return x - (sums / cnt.clamp(min=1.0))[src_index]


def _per_node_corr(a, b, src_index, N):
    """Mean over source nodes of the Pearson corr between a[edges of i] and b[edges of i]."""
    ac, bc = _per_node_center(a, src_index, N), _per_node_center(b, src_index, N)
    num = torch.zeros(N, dtype=a.dtype).index_add(0, src_index, ac * bc)
    va = torch.zeros(N, dtype=a.dtype).index_add(0, src_index, ac * ac)
    vb = torch.zeros(N, dtype=a.dtype).index_add(0, src_index, bc * bc)
    deg = torch.zeros(N, dtype=a.dtype).index_add(0, src_index, torch.ones_like(a))
    corr = num / (va.sqrt() * vb.sqrt() + 1e-12)
    valid = deg >= 2                                       # corr only defined for nodes with >=2 candidates
    return float(corr[valid].mean())


def _build(seed):
    scene, ev = build_scale_instance(GRID, seed, scenario=SCEN, base_node_err=BE, corr_strength=CORR)
    gc = build_candidate_graph(scene.positions, scene.comm_radius)
    return scene, ev, gc


def main():
    t0 = time.perf_counter()
    git = mf.current_git_commit() or "uncommitted"
    omega_cache = {}
    # ---- 1. compute the MC oracle logits (the supervised target) for every scene ----
    data = {}
    for tag, seeds in [("train", TRAIN_SCENES), ("heldout", HELDOUT_SCENES)]:
        for s in seeds:
            scene, ev, gc = _build(s)
            omega_cache[s] = uniform_participation(scene.num_nodes)
            ts = time.perf_counter()
            tr = train_mc_edge_logit_oracle(scene, ev, PROFILE, PROTO, PHY, steps=ORACLE_STEPS,
                                            train_trials=ORACLE_TRIALS, init="distance", base_seed=7 * s)
            data[s] = {"tag": tag, "scene": scene, "ev": ev, "gc": gc, "oracle_logits": tr["logits"]}
            log(f"oracle logits scene {s} ({tag}): {gc.num_edges} edges ({time.perf_counter()-ts:.0f}s)")

    out = {"metric_namespace_version": "macrostate_v2", "experiment_family": "esp_performance_scale_v2",
           "experiment": "oracle_distillation_diagnosis", "query_family": "ESP", "smoke": SMOKE, "git_commit": git,
           "node_count": 120, "scenario": SCEN, "base_node_err": BE, "corr_strength": CORR,
           "train_scenes": TRAIN_SCENES, "heldout_scenes": HELDOUT_SCENES, "oracle_steps": ORACLE_STEPS,
           "hidden_dims": HIDDEN_DIMS, "note": "distil GNN to per-scene MC-oracle logits; diagnose feature/capacity/generalization.",
           "per_hidden_dim": {}}

    # ---- 2. distil a GNN (per capacity) to the oracle logits on TRAIN scenes ----
    for hd in HIDDEN_DIMS:
        torch.manual_seed(0)
        model = ESDGNN(_esp_config(hd, PROFILE.k)).double()
        feats = {s: build_scene_features(data[s]["scene"], model.cfg) for s in data}
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        for step in range(DISTILL_STEPS):
            opt.zero_grad()
            loss = 0.0
            for s in TRAIN_SCENES:
                gc = data[s]["gc"]; N = data[s]["scene"].num_nodes
                gnn_logits = torch.log(model(feats[s])[0])
                tgt = data[s]["oracle_logits"]
                loss = loss + ((_per_node_center(gnn_logits, gc.src_index, N)
                                - _per_node_center(tgt, gc.src_index, N)) ** 2).mean()
            loss = loss / len(TRAIN_SCENES)
            loss.backward()
            opt.step()
        with torch.no_grad():
            in_corr = sum(_per_node_corr(torch.log(model(feats[s])[0]), data[s]["oracle_logits"],
                                         data[s]["gc"].src_index, data[s]["scene"].num_nodes)
                          for s in TRAIN_SCENES) / len(TRAIN_SCENES)
            ho_corr = sum(_per_node_corr(torch.log(model(feats[s])[0]), data[s]["oracle_logits"],
                                         data[s]["gc"].src_index, data[s]["scene"].num_nodes)
                          for s in HELDOUT_SCENES) / len(HELDOUT_SCENES)
        out["per_hidden_dim"][str(hd)] = {"in_sample_corr": in_corr, "heldout_corr": ho_corr,
                                          "distill_final_loss": float(loss.detach()), "model": model, "feats": feats}
        log(f"hidden_dim={hd}: in-sample corr={in_corr:.3f} held-out corr={ho_corr:.3f}")

    # ---- 3. eval the largest distilled GNN under MC on held-out scenes vs distance vs oracle ----
    big = max(HIDDEN_DIMS)
    model = out["per_hidden_dim"][str(big)].pop("model")
    feats = out["per_hidden_dim"][str(big)].pop("feats")
    for hd in HIDDEN_DIMS:                                 # drop non-serialisable handles
        out["per_hidden_dim"][str(hd)].pop("model", None); out["per_hidden_dim"][str(hd)].pop("feats", None)
    ho_eval = {}
    for s in HELDOUT_SCENES:
        scene, ev, gc = data[s]["scene"], data[s]["ev"], data[s]["gc"]
        omega = omega_cache[s]; gen = 9000 + s

        def _mc(policy):
            r = run_dynamic_mc(scene, ev, policy, PROTO, PHY, num_trials=EVAL_TRIALS,
                               generator=torch.Generator().manual_seed(gen), service_profile=PROFILE, participation=omega)
            return r.basin_P_correct
        distilled = _mc(ESDGNNQueryPolicy(model, scene, features=feats[s]))
        dist = _mc(DistanceQueryPolicy(beta_per_m=0.04))
        oracle = _mc(free_logit_policy(data[s]["oracle_logits"]))
        ho_eval[str(s)] = {"distilled_gnn_Pc": distilled, "distance_Pc": dist, "oracle_Pc": oracle,
                           "distilled_minus_distance": distilled - dist, "oracle_minus_distance": oracle - dist}
        log(f"held-out scene {s}: distilled={distilled:.3f} distance={dist:.3f} oracle={oracle:.3f}")
    out["heldout_mc"] = ho_eval

    # ---- 4. verdict ----
    in_best = max(out["per_hidden_dim"][str(hd)]["in_sample_corr"] for hd in HIDDEN_DIMS)
    in_small = out["per_hidden_dim"][str(min(HIDDEN_DIMS))]["in_sample_corr"]
    ho_gnn = sum(v["distilled_minus_distance"] for v in ho_eval.values()) / len(ho_eval)
    ho_oracle = sum(v["oracle_minus_distance"] for v in ho_eval.values()) / len(ho_eval)
    out["headline"] = {"best_in_sample_corr": in_best, "small_in_sample_corr": in_small,
                       "mean_heldout_distilled_minus_distance": ho_gnn,
                       "mean_heldout_oracle_minus_distance": ho_oracle, "verdict": None}
    h = out["headline"]
    if in_best < 0.5:
        cap = "CAPACITY" if (in_best - in_small) > 0.15 else "FEATURES"
        h["verdict"] = (f"{cap}-LIMITED: the GNN cannot even fit the oracle in-sample (best corr {in_best:.2f}); "
                        f"the {'bigger model helps -> add capacity' if cap=='CAPACITY' else 'oracle uses info absent from the GNN geometric features -> add evidence/quorum features'}")
    elif ho_gnn > 0.03:
        h["verdict"] = (f"GNN-REACHABLE: distilled GNN fits in-sample (corr {in_best:.2f}) AND beats distance held-out "
                        f"({ho_gnn:+.3f}) -> the headroom is learnable; parity was a TRAINING/objective failure, fix the trainer")
    else:
        h["verdict"] = (f"GENERALIZATION gap: GNN fits oracle in-sample (corr {in_best:.2f}) but does NOT beat distance "
                        f"held-out ({ho_gnn:+.3f}) while the oracle still does ({ho_oracle:+.3f}) -> per-scene oracle "
                        f"doesn't transfer through the current features; richer/generalizing features needed")
    out["manifest"] = mf.build_manifest(
        build_experiment_spec(protocol_cfg=PROTO, service_profile=PROFILE, phy_cfg=PHY,
                              evidence_descriptor=f"{SCEN}:p={BE}:c={CORR}", scene_descriptor={"gx": 5, "gy": 5, "v": 3},
                              query_law="esp", full_physics=True),
        policy_hash="oracle_distill", checkpoint_hash="distill", model_seeds=TRAIN_SCENES, git_commit=git,
        manifest_id="Lever1b-distill")
    out["runtime_total_s"] = round(time.perf_counter() - t0, 1)
    json.dump(out, open(OUT, "w"), indent=2)
    forbidden = schema.forbidden_keys_in(out)
    assert not forbidden, f"namespace guard: forbidden keys {forbidden} (saved to {OUT})"
    log(f"DONE {out['runtime_total_s']}s; VERDICT: {h['verdict'][:90]}")


if __name__ == "__main__":
    main()
