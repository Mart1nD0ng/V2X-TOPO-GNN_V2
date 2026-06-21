"""D/E ablation: does the coupled C/D/E objective actually exercise delay (D) and
energy (E), or is it reliability-dominated / inert?

Trains the SAME model (same init) under four objective weightings and compares the
resulting C / F / D / E. Uses the SPEC reliability target (0.01) so the reliability
gate engages once F < target — that is the mechanism that is *supposed* to let D/E
drive the topology. If cranking w_D lowers D and cranking w_E lowers E, the coupling
is real and responsive; if the metrics don't move, D/E are inert.

  full         w_R=1 w_D=1 w_E=1   (production balance)
  rel_only     w_R=1 w_D=0 w_E=0   (control: ignores D/E)
  delay_heavy  w_R=1 w_D=10 w_E=0  (isolate D pressure)
  energy_heavy w_R=1 w_D=0 w_E=10  (isolate E pressure)

Outputs result/<run-name>/: de_ablation.json + figures/de_ablation.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.evaluation import evaluate_v2x_graph_consensus  # noqa: E402
from src.losses import compute_coupled_loss  # noqa: E402
from src.topology import TopologyConstructionLayer  # noqa: E402
from src.training.gradient_governance import (  # noqa: E402
    GradientGovernanceConfig,
    coupled_backward,
    make_balancer,
)
from src.training.training_smoke import (  # noqa: E402
    _avalanche_config,
    _evaluator_delay_config,
    _evaluator_energy_config,
    _evaluator_physical_config,
    _initial_preferences,
    _loss_config,
    _make_environment,
    _make_model,
    _normalized_config,
    load_training_smoke_config,
)

ARMS = [
    ("full", 1.0, 1.0),
    ("rel_only", 0.0, 0.0),
    ("delay_heavy", 10.0, 0.0),
    ("energy_heavy", 0.0, 10.0),
]


def _dbm_to_watt(value_dbm: float) -> float:
    return 10.0 ** ((float(value_dbm) - 30.0) / 10.0)


def _protocol_lower_bounds(cfg) -> dict[str, float]:
    avalanche = _avalanche_config(cfg)
    physical = _evaluator_physical_config(cfg)
    energy = _evaluator_energy_config(cfg)
    min_rounds = float(avalanche["beta"])
    packet_duration_s = float(energy.get("packet_duration_s", physical.get("single_hop_delay_s", 0.001)))
    tx_power_watt = float(energy.get("tx_power_watt", _dbm_to_watt(float(physical.get("tx_power_dbm", 23.0)))))
    per_query_energy = packet_duration_s * (
        tx_power_watt
        + float(energy.get("rx_power_watt", 0.3))
        + float(energy.get("processing_power_watt", 0.05))
    )
    min_energy = min_rounds * float(avalanche["k"]) * per_query_energy
    return {
        "min_delay_rounds": min_rounds,
        "min_energy_j": min_energy,
        "single_hop_delay_s": float(physical.get("single_hop_delay_s", packet_duration_s)),
        "per_query_energy_j": per_query_energy,
    }


def _forward(model_or_score, env, topology_layer, fixed_caps, *, eval_mode: bool = False):
    f = env["features"]
    if isinstance(model_or_score, torch.Tensor):
        score = model_or_score
    else:
        score = model_or_score(
            num_nodes=env["candidate"].num_nodes, src_index=f["src_index"], dst_index=f["dst_index"],
            node_features=f["node_features"], edge_features=f["edge_features"],
            region_id=f["region_id"], num_regions=f["num_regions"],
            edge_sector_id=f["edge_sector_id"], edge_is_cross_region=f["edge_is_cross_region"],
            use_structural_score_bias=False,
        )["edge_score"]
    topo = topology_layer(
        num_nodes=env["candidate"].num_nodes, src_index=f["src_index"], dst_index=f["dst_index"],
        edge_score=score, per_node_budget=fixed_caps,
    )
    sel = topo.selected_candidate_index
    ev = evaluate_v2x_graph_consensus(
        **topo.as_evaluation_kwargs(),
        distance_m=f["distance_m"].index_select(0, sel), los_flag=f["los_flag"].index_select(0, sel),
        node_initial_correct=env["ic"], node_initial_wrong=env["iw"],
        physical_config=_evaluator_physical_config(env["cfg"]),
        avalanche_config=_avalanche_config(env["cfg"], eval_mode=eval_mode),
        energy_config=_evaluator_energy_config(env["cfg"]),
        delay_config=_evaluator_delay_config(env["cfg"]),
    )
    return ev


def _metrics(ev) -> dict[str, float]:
    return {
        "C": float(ev["C_avalanche_node_mean"].mean()),
        "F": float(ev["F_avalanche_node_mean"].mean()),
        "D": float(ev["D_avalanche_rounds_mean"].mean()),
        # Phase 3.1 ms instrumentation: the physical latency the rounds represent (1 round = the
        # configured single-hop slot), reportable against a V2X budget (10-100 ms class).
        "D_ms": float(ev["D_avalanche_seconds_mean"].mean()) * 1000.0,
        "E": float(ev["E_consensus_node_mean"].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="D/E ablation")
    parser.add_argument("--config", default="configs/production_training_v1.yaml")
    parser.add_argument("--node-count", type=int, default=2000)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reliability-target", type=float, default=0.01)
    parser.add_argument("--governance", default="none",
                        choices=["none", "pcgrad", "gradnorm", "both"],
                        help="gradient governance (docs/GRADIENT_GOVERNANCE_DESIGN.md); "
                             "none = unchanged static-weight backward")
    parser.add_argument("--run-name", default="de_ablation")
    # P0-1 currency: Q=1 = mean-field surrogate (banned reliability conclusion); Q>=21 quenched.
    parser.add_argument("--quench", type=int, default=None, help="training quenched_quadrature override")
    parser.add_argument("--eval-quench", type=int, default=None, help="eval quadrature override (reported F/D/E)")
    # P1-2: per-round latency aggregation. 'max' makes D depend on the SLOWEST parallel query (a real
    # topology lever); 'mean' is the legacy weighted-mean (D pinned near the round floor).
    parser.add_argument("--delay-reduce", default=None, choices=["mean", "max"],
                        help="structural-delay per-round reduce mode (overrides config delay block)")
    args = parser.parse_args()
    governance = GradientGovernanceConfig.from_name(args.governance)

    config = dict(load_training_smoke_config(str(ROOT / args.config)))
    config["vehicle_count"] = int(args.node_count)
    config["seed"] = int(args.seed)
    if args.delay_reduce is not None:
        delay_block = dict(config.get("delay", {}))
        delay_block["structural_delay"] = True
        delay_block["structural_delay_reduce"] = args.delay_reduce
        config["delay"] = delay_block
    if args.quench is not None:
        config["quenched_quadrature"] = int(args.quench)
    if args.eval_quench is not None:
        config["eval_quenched_quadrature"] = int(args.eval_quench)
    cfg = _normalized_config(config)
    eval_q = int(cfg.get("eval_quenched_quadrature", cfg.get("quenched_quadrature", 1)))
    currency = "mean_field_surrogate" if eval_q <= 1 else f"quenched_Q{eval_q}"
    print(f"[currency] reported C/F/D/E at eval Q={eval_q} -> {currency}")
    candidate, features = _make_environment(cfg)
    ic, iw = _initial_preferences(cfg, candidate.num_nodes)
    env = {"cfg": cfg, "candidate": candidate, "features": features, "ic": ic, "iw": iw}

    budget = None if cfg["max_out_degree"] is None else int(cfg["max_out_degree"])
    topology_layer = TopologyConstructionLayer(
        max_out_degree=budget, support_mode=str(cfg["support_mode"]), temperature=1.0,
        topk_backend=str(cfg["topk_backend"]),
        gradient_mode=str(cfg.get("gradient_mode", "selected_row_softmax")),
        straight_through_temperature=cfg.get("straight_through_temperature", None),
    )
    fixed_caps = torch.full((candidate.num_nodes,), budget, dtype=torch.long) if budget else None

    base_loss = dict(_loss_config(cfg))
    base_loss["reliability_failure_target"] = float(args.reliability_target)
    base_loss["reliability_tail_failure_target"] = float(args.reliability_target)
    base_loss["use_reliability_gate"] = True

    delay_target = base_loss["delay_target_rounds"]
    energy_target = base_loss["energy_target_j"]
    lower_bounds = _protocol_lower_bounds(cfg)
    print(f"D/E ablation (N={candidate.num_nodes}, seed={args.seed}, steps={args.max_steps}, "
          f"rel target={args.reliability_target}; delay target={delay_target} rounds, energy target={energy_target} J)")

    # initial (untrained) metrics for reference
    torch.manual_seed(int(args.seed))
    init_model = _make_model(cfg)
    with torch.no_grad():
        init_m = _metrics(_forward(init_model, env, topology_layer, fixed_caps, eval_mode=True))
    # Defect fix (audit): under structural reduce modes the effective-rounds D sits orders of
    # magnitude above the protocol-era delay_target_rounds=5.0, so L_D=softplus((D-5)/tau) would
    # dominate the objective by 1-2 orders and distort every w_D>0 arm. When a reduce mode is
    # explicitly requested, recalibrate the delay target/tau from the UNTRAINED model's D so the
    # delay term starts in a comparable band to L_R (target = 0.5*D_init -> real but sane pressure).
    if args.delay_reduce is not None and init_m["D"] > base_loss["delay_target_rounds"] * 4.0:
        recal_target = init_m["D"] * 0.5
        base_loss["delay_target_rounds"] = recal_target
        base_loss["delay_p90_target_rounds"] = recal_target * 1.5
        base_loss["delay_tau"] = max(recal_target * 0.2, 1.0)
        delay_target = recal_target
        print(f"[delay recalibration] reduce={args.delay_reduce}: untrained D={init_m['D']:.1f} -> "
              f"delay_target_rounds={recal_target:.1f}, delay_tau={base_loss['delay_tau']:.1f} "
              f"(protocol-era target 5.0 would have made L_D dominate the loss)")
    print(f"{'arm':>13} | {'C':>7} | {'F':>7} | {'D(rounds)':>9} | {'E(J)':>10} | gate")
    print(f"{'(untrained)':>13} | {init_m['C']:>7.4f} | {init_m['F']:>7.4f} | {init_m['D']:>9.4f} | {init_m['E']:>10.3e} |  -")

    rows = []
    for name, wd, we in ARMS:
        loss_cfg = dict(base_loss)
        loss_cfg["weight_reliability"] = 1.0
        loss_cfg["weight_delay"] = wd
        loss_cfg["weight_energy"] = we
        torch.manual_seed(int(args.seed))  # same init for every arm
        model = _make_model(cfg)
        opt = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]))
        balancer = make_balancer(governance)  # fresh stateful balancer per arm (gradnorm)
        gate_engaged = False
        for _step in range(int(args.max_steps)):
            opt.zero_grad(set_to_none=True)
            ev = _forward(model, env, topology_layer, fixed_caps)
            lo = compute_coupled_loss(ev, loss_cfg)
            gate_engaged = gate_engaged or bool(lo["reliability_loss_should_weaken"])
            if governance.active:
                coupled_backward(lo, model.parameters(), governance, balancer)
            else:
                backward = lo.get("effective_backward_loss", lo["total_loss"])
                backward.backward()
            opt.step()
        with torch.no_grad():
            m = _metrics(_forward(model, env, topology_layer, fixed_caps, eval_mode=True))
        m.update({"arm": name, "w_delay": wd, "w_energy": we, "gate_engaged": gate_engaged})
        rows.append(m)
        print(f"{name:>13} | {m['C']:>7.4f} | {m['F']:>7.4f} | {m['D']:>9.4f} | {m['E']:>10.3e} | {gate_engaged}")

    by = {r["arm"]: r for r in rows}
    full, rel = by["full"], by["rel_only"]
    dh, eh = by["delay_heavy"], by["energy_heavy"]
    # Responsiveness must be judged by MAGNITUDE, not just sign: a 0.1% move under a
    # 10x weight is not meaningful coupling. Relative reduction vs the rel_only control.
    MEANINGFUL = 0.02  # >=2% relative reduction counts as a real response
    d_rel_reduction = (rel["D"] - dh["D"]) / rel["D"] if rel["D"] else 0.0
    e_rel_reduction = (rel["E"] - eh["E"]) / rel["E"] if rel["E"] else 0.0
    d_sign_ok = dh["D"] < rel["D"] - 1e-9
    e_sign_ok = eh["E"] < rel["E"] * (1 - 1e-9)
    d_meaningful = d_rel_reduction >= MEANINGFUL
    e_meaningful = e_rel_reduction >= MEANINGFUL
    # B-G3 (Track B, docs/COUPLING_AND_OPERATING_POINT_DESIGN.md): E must respond
    # INDEPENDENTLY of D. If E were merely D rescaled (E ~ D x const), the delay_heavy
    # arm would cut E as much as energy_heavy does, so E[delay_heavy]-E[energy_heavy] ~ 0.
    # A positive margin means energy pressure buys E reduction that delay pressure alone
    # does not -- proof of an independent energy lever (the retransmission-aware path).
    # Definition matches the contract: (E[delay_heavy] - E[energy_heavy]) / E[rel_only].
    e_independence_margin = (dh["E"] - eh["E"]) / rel["E"] if rel["E"] else 0.0
    e_independent = bool(e_meaningful and e_independence_margin >= 0.01)
    rel_ok = rel["F"] <= args.reliability_target
    summary = {
        "config": args.config, "node_count": candidate.num_nodes, "seed": args.seed,
        "governance": args.governance, "evaluator_currency": currency, "eval_quench": eval_q,
        "max_steps": args.max_steps, "reliability_target": args.reliability_target,
        "delay_target_rounds": delay_target, "energy_target_j": energy_target,
        "protocol_lower_bounds": lower_bounds,
        "untrained": init_m, "arms": rows,
        "delay_relative_reduction": d_rel_reduction,
        "energy_relative_reduction": e_rel_reduction,
        "delay_responsive_sign": bool(d_sign_ok), "energy_responsive_sign": bool(e_sign_ok),
        "delay_meaningful": bool(d_meaningful), "energy_meaningful": bool(e_meaningful),
        "energy_independence_margin": e_independence_margin,
        "energy_independent_of_delay": e_independent,
        "meaningful_threshold": MEANINGFUL,
        "D_vs_target_ratio": full["D"] / delay_target if delay_target else None,
        "E_vs_target_ratio": full["E"] / energy_target if energy_target else None,
        "D_vs_protocol_min_ratio": full["D"] / lower_bounds["min_delay_rounds"]
        if lower_bounds["min_delay_rounds"] else None,
        "E_vs_protocol_min_ratio": full["E"] / lower_bounds["min_energy_j"]
        if lower_bounds["min_energy_j"] else None,
        "rel_only_meets_target": bool(rel_ok),
    }
    if d_meaningful and e_meaningful:
        verdict = "D/E EXERCISED & RESPONSIVE (coupling is real)"
    elif d_meaningful or e_meaningful:
        verdict = "PARTIAL: only one of D/E is meaningfully responsive"
    elif (
        summary["D_vs_protocol_min_ratio"] is not None
        and summary["E_vs_protocol_min_ratio"] is not None
        and summary["D_vs_protocol_min_ratio"] <= 1.20
        and summary["E_vs_protocol_min_ratio"] <= 1.20
    ):
        verdict = (
            "D/E LOWER-BOUND SATURATED "
            "(delay and energy are within 20% of protocol minimum; topology has little remaining D/E leverage)"
        )
    elif d_sign_ok or e_sign_ok:
        verdict = ("D/E WEAKLY COUPLED / effectively inert "
                   f"(<{int(MEANINGFUL*100)}% move under 10x weight; topology has negligible leverage)")
    else:
        verdict = "D/E INERT (objective is effectively reliability-only at this operating point)"
    summary["verdict"] = verdict

    out_dir = ROOT / "result" / args.run_name
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "de_ablation.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _render(summary, out_dir / "figures" / "de_ablation.png")

    print(f"\ndelay reduction (delay_heavy vs rel_only): {100*d_rel_reduction:+.2f}%  "
          f"[{dh['D']:.4f} vs {rel['D']:.4f}]  meaningful={d_meaningful}")
    print(f"energy reduction (energy_heavy vs rel_only): {100*e_rel_reduction:+.2f}%  "
          f"[{eh['E']:.3e} vs {rel['E']:.3e}]  meaningful={e_meaningful}")
    print(f"energy independence (E[delay_heavy]-E[energy_heavy])/E[rel_only]: "
          f"{100*e_independence_margin:+.2f}%  independent={e_independent}  "
          f"[delay_heavy E={dh['E']:.3e} vs energy_heavy E={eh['E']:.3e}]")
    print(f"D is {summary['D_vs_target_ratio']:.1f}x its target; E is {summary['E_vs_target_ratio']:.0f}x its target")
    print(
        f"protocol lower bounds: D_min={lower_bounds['min_delay_rounds']:.1f} rounds, "
        f"E_min={lower_bounds['min_energy_j']:.3e} J; "
        f"full ratios D/D_min={summary['D_vs_protocol_min_ratio']:.2f}, "
        f"E/E_min={summary['E_vs_protocol_min_ratio']:.2f}"
    )
    print(f"VERDICT: {verdict}")


def _render(summary, out_path: Path) -> None:
    arms = [r["arm"] for r in summary["arms"]]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    fig.suptitle(f"D/E ablation (N={summary['node_count']}, seed={summary['seed']})    {summary['verdict']}",
                 fontsize=12, fontweight="bold")
    specs = [
        ("F", "failure F (lower=better)", summary["reliability_target"], "rel target"),
        ("D", "delay D (rounds, lower=better)", summary["delay_target_rounds"], "delay target"),
        ("E", "energy E (J, lower=better)", summary["energy_target_j"], "energy target"),
    ]
    colors = {"full": "#1f77b4", "rel_only": "#d62728", "delay_heavy": "#2ca02c", "energy_heavy": "#9467bd"}
    for ax, (key, label, tgt, tname) in zip(axes, specs):
        vals = [r[key] for r in summary["arms"]]
        ax.bar(arms, vals, color=[colors[a] for a in arms], alpha=0.85)
        if tgt is not None:
            ax.axhline(tgt, color="green", ls=":", lw=1.4, label=tname)
            ax.legend(fontsize=8)
        ax.set_title(label, fontsize=11)
        ax.tick_params(axis="x", labelrotation=20)
        ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
