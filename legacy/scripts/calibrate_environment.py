from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v2x_env.feasibility import combine_seed_reports, evaluate_environment_config, load_config


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return [value]


def _balanced_subset(values: list[Any], limit: int | None) -> list[Any]:
    if limit is None or len(values) <= limit:
        return values
    if limit <= 0:
        return []
    if limit == 1:
        return [values[0]]
    indices = [round(idx * (len(values) - 1) / (limit - 1)) for idx in range(limit)]
    selected: list[Any] = []
    seen: set[int] = set()
    for index in indices:
        if index not in seen:
            seen.add(index)
            selected.append(values[index])
    fill_index = 0
    while len(selected) < limit and fill_index < len(values):
        if fill_index not in seen:
            seen.add(fill_index)
            selected.append(values[fill_index])
        fill_index += 1
    return selected


def _search_dimensions(config: Mapping[str, Any]) -> dict[str, list[Any]]:
    search = config.get("search", {})
    if not isinstance(search, Mapping):
        return {}
    return {
        "vehicle_count": _as_list(search.get("vehicle_counts", [])),
        "tx_power_dbm": _as_list(search.get("tx_power_dbm", [])),
        "mcs_threshold_db": _as_list(search.get("mcs_threshold_db", [])),
        "candidate_radius_m": _as_list(search.get("candidate_radius_m", [])),
        "max_candidates_per_node": _as_list(search.get("max_candidates_per_node", [])),
        "bandwidth_mhz": _as_list(search.get("bandwidth_mhz", [])),
        "noise_dbm": _as_list(search.get("noise_dbm", [])),
        "interference_proxy_dbm": _as_list(search.get("interference_proxy_dbm", [])),
        "carrier_frequency_ghz": _as_list(search.get("carrier_frequency_ghz", [])),
        "nlos_penalty_db": _as_list(search.get("nlos_penalty_db", [])),
    }


def _coverage_by_dimension(config: Mapping[str, Any], variants: list[Mapping[str, Any]]) -> tuple[dict[str, list[Any]], dict[str, list[Any]]]:
    dimensions = _search_dimensions(config)
    observed: dict[str, set[Any]] = {name: set() for name in dimensions}
    for variant in variants:
        selected = variant["selected_config"]
        for name in observed:
            if name in selected:
                observed[name].add(selected[name])
    covered: dict[str, list[Any]] = {}
    uncovered: dict[str, list[Any]] = {}
    for name, values in dimensions.items():
        expected = [value for value in values if value is not None]
        covered[name] = [value for value in expected if value in observed[name]]
        uncovered[name] = [value for value in expected if value not in observed[name]]
    return covered, uncovered


def _variant_configs(config: Mapping[str, Any], smoke: bool) -> Iterable[dict[str, Any]]:
    search = config.get("search", {})
    if not isinstance(search, Mapping):
        raise ValueError("config search section must be a mapping")
    base_snapshot = dict(config.get("snapshot", {}))
    base_channel = {
        "carrier_frequency_ghz": 5.9,
        "bandwidth_mhz": 20.0,
        "tx_power_dbm": 23.0,
        "noise_dbm": -95.0,
        "interference_proxy_dbm": -78.0,
        "nlos_penalty_db": 12.0,
        "mcs_threshold_db": 8.0,
    }
    base_candidate = {
        "radius_m": 180.0,
        "max_candidates_per_node": 12,
        "cell_size_m": 180.0,
    }
    seeds = _as_list(search.get("seeds", [0]))
    vehicle_counts = _as_list(search.get("vehicle_counts", [base_snapshot.get("vehicle_count", 100)]))
    tx_power_values = _as_list(search.get("tx_power_dbm", [base_channel["tx_power_dbm"]]))
    threshold_values = _as_list(search.get("mcs_threshold_db", [base_channel["mcs_threshold_db"]]))
    radius_values = _as_list(search.get("candidate_radius_m", [base_candidate["radius_m"]]))
    cap_values = _as_list(search.get("max_candidates_per_node", [base_candidate["max_candidates_per_node"]]))
    bandwidth_values = _as_list(search.get("bandwidth_mhz", [base_channel["bandwidth_mhz"]]))
    noise_values = _as_list(search.get("noise_dbm", [base_channel["noise_dbm"]]))
    interference_values = _as_list(search.get("interference_proxy_dbm", [base_channel["interference_proxy_dbm"]]))
    frequency_values = _as_list(search.get("carrier_frequency_ghz", [base_channel["carrier_frequency_ghz"]]))
    nlos_penalties = _as_list(search.get("nlos_penalty_db", [base_channel["nlos_penalty_db"]]))

    non_vehicle_combos = list(product(
        tx_power_values,
        threshold_values,
        radius_values,
        cap_values,
        bandwidth_values,
        noise_values,
        interference_values,
        frequency_values,
        nlos_penalties,
    ))
    max_per_vehicle = int(search.get("smoke_variants_per_vehicle_count", 8)) if smoke else None
    for vehicle_count in vehicle_counts:
        combos_for_count = _balanced_subset(non_vehicle_combos, max_per_vehicle)
        for combo in combos_for_count:
            (
                tx_power,
                threshold,
                radius,
                cap,
                bandwidth,
                noise,
                interference,
                frequency,
                nlos_penalty,
            ) = combo
            yield {
                "snapshot": {
                    **base_snapshot,
                    "vehicle_count": int(vehicle_count),
                    "seed": None,
                },
                "seeds": [int(seed) for seed in seeds[: (2 if smoke else len(seeds))]],
                "channel": {
                    **base_channel,
                    "carrier_frequency_ghz": float(frequency),
                    "bandwidth_mhz": float(bandwidth),
                    "tx_power_dbm": float(tx_power),
                    "noise_dbm": float(noise),
                    "interference_proxy_dbm": float(interference),
                    "nlos_penalty_db": float(nlos_penalty),
                    "mcs_threshold_db": float(threshold),
                },
                "candidate_graph": {
                    **base_candidate,
                    "radius_m": float(radius),
                    "cell_size_m": float(radius),
                    "max_candidates_per_node": int(cap),
                },
                "baselines": dict(config.get("baselines", {})),
                "classification": dict(config.get("classification", {})),
            }


def _run_variant(config: Mapping[str, Any]) -> dict[str, Any]:
    seed_reports = []
    for seed in config["seeds"]:
        per_seed = dict(config)
        per_seed["snapshot"] = dict(config["snapshot"])
        per_seed["snapshot"]["seed"] = int(seed)
        seed_reports.append(evaluate_environment_config(per_seed))
    return combine_seed_reports(seed_reports, config)


def _choose_best(reports: list[dict[str, Any]]) -> dict[str, Any] | None:
    usable = [report for report in reports if report["classification"] == "usable"]
    if not usable:
        return None
    return max(
        usable,
        key=lambda report: (
            report["summary"]["mean_best_baseline_giant_component_ratio"],
            report["summary"]["mean_best_query_success_proxy"],
            -abs(report["summary"]["mean_average_candidate_degree"] - 12.0),
        ),
    )


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _write_markdown(path: Path, report: Mapping[str, Any]) -> None:
    best = report.get("best_usable_config")
    lines = [
        "# Environment Feasibility",
        "",
        "This M1 report uses a deterministic physical feasibility proxy. It does not report final Avalanche reliability.",
        "",
        f"- Search status: {report['status']}",
        f"- Variants evaluated: {len(report['variants'])}",
    ]
    if best:
        summary = best["summary"]
        selected = best["selected_config"]
        lines.extend(
            [
                f"- Best config hash: `{best['config_hash']}`",
                f"- Selected vehicle count: {selected['vehicle_count']}",
                f"- Tx power dBm: {selected['tx_power_dbm']}",
                f"- Candidate radius m: {selected['candidate_radius_m']}",
                f"- Candidate cap: {selected['max_candidates_per_node']}",
                f"- Mean candidate degree: {summary['mean_average_candidate_degree']:.2f}",
                f"- Mean candidate giant component ratio: {summary['mean_candidate_giant_component_ratio']:.3f}",
                f"- Mean best baseline giant component ratio: {summary['mean_best_baseline_giant_component_ratio']:.3f}",
                f"- Mean best query_success_proxy: {summary['mean_best_query_success_proxy']:.3f}",
                f"- Mean cap hit ratio: {summary['mean_cap_hit_ratio']:.3f}",
                f"- Mean candidate pair checks per node: {summary['mean_candidate_pair_checks_per_node']:.1f}",
                f"- Mean best baseline out degree: {summary['mean_best_baseline_out_degree']:.2f}",
                f"- Mean best baseline undirected average degree: {summary['mean_best_baseline_undirected_average_degree']:.2f}",
                f"- Strongest low-budget giant component ratio: {summary['strongest_low_budget_giant_component_ratio']:.3f}",
                f"- Strongest low-budget query_success_proxy: {summary['strongest_low_budget_query_success_proxy']:.3f}",
                f"- Best query_success_proxy exceeds too-easy threshold: {summary['best_query_success_proxy_exceeds_too_easy_threshold']}",
                "",
                "Classification: usable. The scenario is considered usable because medium-budget/backbone baselines connect the graph while low-budget KNN baselines do not meet connectivity. The physical query_success_proxy may still be near saturation and must be revisited before training.",
            ]
        )
        if summary.get("cap_hit_warning"):
            lines.append("")
            lines.append("Warning: mean_cap_hit_ratio >= 0.95, so the candidate cap is strongly active and must be swept before training.")
    else:
        lines.extend(
            [
                "",
                "No usable config was selected in this run. This is not a code failure for small smoke grids.",
            ]
        )
    lines.append("")
    lines.append("## Smoke Coverage")
    covered = report.get("smoke_covered_values_by_dimension", {})
    uncovered = report.get("smoke_uncovered_values_by_dimension", {})
    for name in sorted(covered):
        lines.append(f"- {name}: covered={covered[name]} uncovered={uncovered.get(name, [])}")
    lines.append("")
    lines.append("## Variant Summary")
    for item in report["variants"]:
        summary = item["summary"]
        selected = item["selected_config"]
        lines.append(
            f"- `{item['config_hash']}` {item['classification']}: tx={selected['tx_power_dbm']} dBm, "
            f"radius={selected['candidate_radius_m']} m, cap={selected['max_candidates_per_node']}, "
            f"candidate_degree={summary['mean_average_candidate_degree']:.2f}, "
            f"candidate_giant={summary['mean_candidate_giant_component_ratio']:.3f}, "
            f"best_query_success_proxy={summary['mean_best_query_success_proxy']:.3f}, "
            f"cap_hit_ratio={summary['mean_cap_hit_ratio']:.3f}, "
            f"pair_checks_per_node={summary['mean_candidate_pair_checks_per_node']:.1f}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_medium_config(path: Path, report: Mapping[str, Any], source_config: str, allow_empty: bool) -> bool:
    best = report.get("best_usable_config")
    if not best:
        if not allow_empty:
            return False
        payload = {
            "generated_from": source_config,
            "mode": report["mode"],
            "status": report["status"],
            "seeds": [],
            "config_hash": None,
        }
    else:
        payload = dict(best["full_config"])
        payload = {
            "generated_from": source_config,
            "config_hash": best["config_hash"],
            "mode": report["mode"],
            "seeds": best["summary"]["seeds"],
            **payload,
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml  # type: ignore

        text = yaml.safe_dump(payload, sort_keys=False)
    except Exception:
        text = json.dumps(payload, indent=2, sort_keys=False)
    path.write_text(text, encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate deterministic V2X environment feasibility.")
    parser.add_argument("--config", required=True, help="Path to physical search config YAML/JSON.")
    parser.add_argument("--smoke", action="store_true", help="Run a small CPU smoke grid.")
    parser.add_argument("--json-out", default="reports/environment_feasibility.json")
    parser.add_argument("--md-out", default="reports/environment_feasibility.md")
    parser.add_argument("--write-medium-config", default=None, help="Write selected usable config to this YAML path.")
    parser.add_argument("--allow-empty-medium", action="store_true", help="Allow overwriting medium config when no usable config exists.")
    args = parser.parse_args()

    config = load_config(args.config)
    variants = [_run_variant(variant) for variant in _variant_configs(config, smoke=args.smoke)]
    best = _choose_best(variants)
    covered, uncovered = _coverage_by_dimension(config, variants)
    report: dict[str, Any] = {
        "status": "usable_config_found" if best else "no_usable_config_found",
        "mode": "smoke" if args.smoke else "full",
        "smoke_covered_values_by_dimension": covered if args.smoke else {},
        "smoke_uncovered_values_by_dimension": uncovered if args.smoke else {},
        "best_usable_config": best,
        "variants": variants,
    }
    _write_json(ROOT / args.json_out, report)
    _write_markdown(ROOT / args.md_out, report)
    medium_written = False
    if args.write_medium_config:
        medium_written = _write_medium_config(ROOT / args.write_medium_config, report, args.config, args.allow_empty_medium)
    if best:
        selected = best["selected_config"]
        print(
            "Best usable config: "
            f"hash={best['config_hash']} tx={selected['tx_power_dbm']}dBm "
            f"radius={selected['candidate_radius_m']}m cap={selected['max_candidates_per_node']} "
            f"query_success_proxy={best['summary']['mean_best_query_success_proxy']:.3f}"
        )
        if args.write_medium_config:
            print(f"Medium config written: {args.write_medium_config}" if medium_written else "Medium config not written.")
    else:
        print("No usable config found in this search; reports were still written.")
        if args.write_medium_config and not medium_written:
            print(f"Medium config preserved: {args.write_medium_config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
