"""G-TEMPORAL evidence (Phase 12): does a temporal MEMORY of the persistent correlated band help
the CDQ 2.0 query, judged by the macrostate basin first-hitting (independent MC)?

Three diversity drivers per epoch t (all matched-marginal-in-time, same q_i): no-memory (the current
noisy proxy x_t), memory (causal EMA m_t of the observable proxy), oracle (the held-out true active
band, eval-only upper bound). Two regimes: persistence=1 (the band persists) vs persistence=0
(iid-in-time control). Honest expectation (per Phase 10): the diversity benefit is a SCOPED P_correct
gain via faster quorum; the NEW temporal claim is that MEMORY recovers the oracle's targeting under
persistence and cannot under iid-in-time.

Run:  PYTHONPATH=. python docs/gate_evidence/macrostate/run_temporal_factorial.py
Writes docs/gate_evidence/macrostate/temporal_factorial_results.json.
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import statistics

import torch
import torch.nn.functional as F

from src.config.service_profile import ConsensusServiceProfile
from src.environment import ProtocolConfig, RoundPhysicsConfig, build_manhattan_scene
from src.environment.temporal_sequence import TemporalCorrelationSequence
from src.metrics.participation import uniform_participation
from src.metrics.temporal_memory import causal_ema
from src.sampling import DistanceQueryPolicy
from src.sampling.cdq2_wiring import CDQ2Policy
from src.validation import run_dynamic_mc

PHY = RoundPhysicsConfig(subchannels=10, slots_per_window=40)
PROTO = ProtocolConfig(k=3, alpha=2, beta=3, r_max=10)
PROFILE = ConsensusServiceProfile.urban_default().replace(k=3, alpha=2, beta=3, max_poll_epochs=10)
T, TRIALS, SEEDS, RHO, ETA = 8, 800, [0, 1, 2], 0.3, 8.0


def est_diversity(model, est):
    """Diversity Z_ij = [ 2*sqrt(est[band_j]) | one_hot(band_j) ] -- a SHARED 'correlated' component
    (large for peers in the estimated-correlated band, so after the kernel's unit-normalisation their
    DIRECTIONS align => the CDQ 2.0 kernel avoids co-selecting them) plus a per-band DIVERSE one-hot
    (low-est peers keep distinct directions => freely co-selectable). The estimate changes the
    DIRECTION of z (not just the magnitude, which unit-norm would wash out)."""
    n = est.numel()
    w = (2.0 * est.clamp_min(0.0).sqrt())
    def f(graph):
        b = model.sensor_of[graph.dst_index]
        shared = w[b].unsqueeze(-1)                              # [E, 1] est-driven shared component
        onehot = F.one_hot(b, n).to(torch.float64)              # [E, n] per-band diverse component
        return torch.cat([shared, onehot], dim=-1)              # [E, n+1]
    return f


def _mc(sc, model, policy, omega):
    rows = [run_dynamic_mc(sc, model, policy, PROTO, PHY, num_trials=TRIALS,
                           generator=torch.Generator().manual_seed(s), link_override=0.85,
                           service_profile=PROFILE, participation=omega) for s in SEEDS]
    return statistics.mean([r.basin_P_correct for r in rows])


def main():
    sc = build_manhattan_scene(3, 3, 3, block_m=120.0, comm_radius=95.0, int_radius=150.0,
                               generator=torch.Generator().manual_seed(0))
    omega = uniform_participation(sc.num_nodes)
    base = DistanceQueryPolicy(beta_per_m=0.05)
    out = {"config": {"T": T, "trials": TRIALS, "seeds": SEEDS, "rho": RHO, "eta": ETA}, "regimes": {}}
    for regime, persistence in (("persistent", 1.0), ("iid_in_time", 0.0)):
        seq = TemporalCorrelationSequence(sc, T=T, persistence=persistence, base_node_err=0.38,
                                          corr_strength=0.33, seed=7)
        proxy = seq.proxy_sequence()                      # [T, n_sensor] observable
        mem = causal_ema(proxy, RHO)                      # [T, n_sensor] memory estimate
        true_oh = seq.true_active_onehot()                # [T, n_sensor] eval-only
        n = seq.n_sensor
        agg = {"ESP": [], "no_memory": [], "memory": [], "oracle": []}
        for t in range(T):
            model = seq.model(t)
            agg["ESP"].append(_mc(sc, model, base, omega))
            for name, est in (("no_memory", proxy[t]), ("memory", mem[t]), ("oracle", true_oh[t])):
                pol = CDQ2Policy(base, r=n + 1, eta=ETA, diversity=est_diversity(model, est))
                agg[name].append(_mc(sc, model, pol, omega))
        cell = {k: round(statistics.mean(v), 4) for k, v in agg.items()}
        out["regimes"][regime] = cell
        print(f"[{regime}] mean basin P_correct over {T} epochs: ESP={cell['ESP']} "
              f"no_memory={cell['no_memory']} memory={cell['memory']} oracle={cell['oracle']}")
        print(f"           memory-vs-ESP={cell['memory']-cell['ESP']:+.4f}  "
              f"memory-vs-no_memory={cell['memory']-cell['no_memory']:+.4f}  "
              f"oracle-vs-ESP={cell['oracle']-cell['ESP']:+.4f}")

    path = os.path.join(os.path.dirname(__file__), "temporal_factorial_results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
