"""eta-risk-liveness curve harness (Guarded-CDQ2 round, G-ETA-RISK-LIVENESS).

Characterises how the CDQ2 diversity strength ``eta`` moves the system along the validity-liveness
trade-off (spec §3). The SAME quality ``s_ij`` (distance) and the SAME observable diversity embedding
``Z`` (deployment-observable sensor-group one-hot, constraint C2) are used at every ``eta``; only
``eta`` varies. ``eta=0`` is exactly ESP. The headline judge is the independent dynamic-MC macrostate
basin first-hitting (constraint #10) -- the gate is NOT judged by whether CDQ2 improves reliability,
only by whether it produces stable curves and identifies HOW probability mass moves (spec §3.7).

The wrong/split basins are HARD constraints (constraint #5): the curve reports whether eta buys a
deadline/liveness gain by moving mass deadline->correct (benign) or deadline->wrong / into split
(a validity cost that must be guarded -- the motivation for Guarded-CDQ2).
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

import torch

from src.evaluation.cdq2_factorial import observable_group_diversity, wilson_ci
from src.evaluation.esp_scale import build_scale_instance
from src.metrics import schema
from src.metrics.participation import uniform_participation
from src.sampling.baseline_policies import DistanceQueryPolicy
from src.sampling.cdq2_wiring import CDQ2Policy
from src.validation import run_dynamic_mc

__all__ = ["ETA_GRID", "cdq2_diversity_for", "eta_sweep", "classify_mass_shift", "MassShift"]

ETA_GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)     # spec §3.2


def cdq2_diversity_for(model, *, use_sensor: bool = True, use_map: bool = False):
    """Observable diversity closure ``graph -> [E, r]`` + its rank ``r`` (sensor/map group one-hots)."""
    div = observable_group_diversity(model, use_sensor=use_sensor, use_map=use_map, use_road=False)
    r = (int(model.sensor_of.max()) + 1 if use_sensor else 0) + \
        (int(model.map_of.max()) + 1 if use_map else 0)
    return div, r


def eta_sweep(grid, scene_seeds, *, scenario: str, base_node_err: float, corr_strength: float,
              profile, proto, phy, trials: int, link_override, eta_grid=ETA_GRID,
              base_beta: float = 0.04) -> dict:
    """Sweep ``eta`` at one (env family, physics) cell. Returns ``{eta: macro_block}`` (macrostate
    basin outcomes pooled over ``scene_seeds`` with Wilson CIs; the independent dynamic MC is the judge).

    The CDQ2 quality is the bare distance policy (so ``eta=0`` == ESP exactly); the diversity is the
    deployment-observable sensor one-hot of rank ``r = #sensor groups`` (a passed embedding sets the
    kernel rank directly -- the max-out-degree bound only applies to CDQ2Policy's default one-hot).
    """
    out = {}
    for eta in eta_grid:
        rows = []
        for s in scene_seeds:
            scene, ev = build_scale_instance(grid, s, scenario=scenario, base_node_err=base_node_err,
                                             corr_strength=corr_strength)
            omega = uniform_participation(scene.num_nodes)
            base = DistanceQueryPolicy(beta_per_m=base_beta)
            if eta == 0.0:
                pol = base                                   # ESP exactly (no diversity head at all)
            else:
                div, r = cdq2_diversity_for(ev, use_sensor=True, use_map=False)
                pol = CDQ2Policy(base, r=r, eta=eta, diversity=div)
            rows.append(run_dynamic_mc(scene, ev, pol, proto, phy, num_trials=trials,
                                       generator=torch.Generator().manual_seed(int(s)),
                                       link_override=link_override, service_profile=profile,
                                       participation=omega))
        n_pool = trials * len(list(scene_seeds))
        mean = lambda a: statistics.mean([getattr(r, a) for r in rows])
        P, Fw, Fs, Fd = (mean("basin_P_correct"), mean("basin_F_wrong"),
                         mean("basin_F_split"), mean("basin_F_deadline"))
        ci = {"macro_P_correct": wilson_ci(P, n_pool), "macro_F_wrong": wilson_ci(Fw, n_pool),
              "macro_F_split": wilson_ci(Fs, n_pool), "macro_F_deadline": wilson_ci(Fd, n_pool)}
        out[eta] = schema.macro_block(P, Fw, Fs, Fd, ci=ci)
    return out


@dataclass(frozen=True)
class MassShift:
    """How CDQ2(eta) moved probability mass relative to ESP (eta=0)."""

    label: str                  # deadline->correct | deadline->wrong | split->correct | mixed | none
    d_P_correct: float
    d_F_wrong: float
    d_F_split: float
    d_F_deadline: float
    wrong_increased: bool       # constraint #5/#12: a wrong-risk increase must NEVER be hidden


def classify_mass_shift(macro_esp: dict, macro_cdq2: dict, *, tol: float = 0.01) -> MassShift:
    """Classify the dominant probability-mass movement from ESP -> CDQ2(eta) (spec §3.7 acceptance).

    Categories: ``deadline->correct`` (Fd down, Pc up, no wrong increase), ``deadline->wrong``
    (Fd down but Fw up -- the validity cost), ``split->correct`` (Fs down, Pc up), ``mixed`` (a
    deadline gain AND a wrong/split increase), or ``none`` (no move beyond ``tol``). The
    ``wrong_increased`` flag is always reported (constraint #12 -- negative wrong-risk never hidden).
    """
    dP = macro_cdq2["macro_P_correct"] - macro_esp["macro_P_correct"]
    dFw = macro_cdq2["macro_F_wrong"] - macro_esp["macro_F_wrong"]
    dFs = macro_cdq2["macro_F_split"] - macro_esp["macro_F_split"]
    dFd = macro_cdq2["macro_F_deadline"] - macro_esp["macro_F_deadline"]
    wrong_up = dFw > tol
    split_up = dFs > tol
    deadline_down = dFd < -tol
    correct_up = dP > tol

    if deadline_down and (wrong_up or split_up):
        label = "mixed"                       # liveness gain bought partly with validity cost
    elif deadline_down and correct_up:
        label = "deadline->correct"           # benign liveness gain
    elif (dFs < -tol) and correct_up:
        label = "split->correct"
    elif wrong_up or split_up:
        label = "deadline->wrong" if wrong_up else "split-up"
    else:
        label = "none"
    return MassShift(label, dP, dFw, dFs, dFd, wrong_up)
