"""Generate operating-point re-calibration config variants from operating_point_v1.yaml.

Each variant keeps axis_visibility LOS (audit C-1 fix) and hardens the EVALUATOR interference
(physical block) to restore a genuinely hard, load-coupled point. Feature channel SINR mirrors the
base interference (channel block); feature-side density coupling stays 0 (historical). Tx/MCS stay
standard 23/8. The candidate cap is set to a LOOSE ceiling so the planner can over-flare and the
cost terms regularise it down to the load-optimal effective degree.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "configs" / "operating_point_v1.yaml"

# (name, interference_dbm, coupling_db, reference_load, max_out_degree, density)
VARIANTS = [
    ("A_cap8",  -82.0, 20.0, 1.0, 8, 200.0),   # op physics, loose ceiling 8 (enables over-flare/U)
    ("B_cap8",  -78.0, 25.0, 1.0, 8, 200.0),   # harder floor + stronger load coupling
    ("C_cap8",  -80.0, 30.0, 1.0, 8, 200.0),   # strong load coupling
    ("A_cap4",  -82.0, 20.0, 1.0, 4, 200.0),   # op physics, original cap 4 (fallback/comparison)
]


def main() -> None:
    base = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    out_dir = ROOT / "configs"
    written = []
    for name, itf, cpl, ref, cap, dens in VARIANTS:
        cfg = copy.deepcopy(base)
        cfg["operating_point_v1_name"] = f"operating_point_recalib_{name}"
        cfg["max_out_degree"] = int(cap)
        cfg["node_density_per_km2"] = float(dens)
        phys = dict(cfg.get("physical", {}))
        phys["interference_proxy_dbm"] = float(itf)
        phys["interference_density_coupling_db"] = float(cpl)
        phys["interference_reference_load"] = float(ref)
        cfg["physical"] = phys
        # feature SINR mirrors the base interference floor (no feature-side density coupling)
        cfg["channel"] = {
            "tx_power_dbm": float(phys.get("tx_power_dbm", 23.0)),
            "mcs_threshold_db": float(phys.get("mcs_threshold_db", 8.0)),
            "transition_width_db": float(phys.get("transition_width_db", 3.0)),
            "interference_proxy_dbm": float(itf),
        }
        cand = dict(cfg.get("candidate_graph", {}))
        cand["los_model"] = "axis_visibility"
        cfg["candidate_graph"] = cand
        path = out_dir / f"operating_point_recalib_{name}.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        written.append(path.name)
        print(f"wrote {path.name}: I={itf}dBm coupling={cpl}dB ref_load={ref} cap={cap} density={dens}")
    print("done:", written)


if __name__ == "__main__":
    main()
