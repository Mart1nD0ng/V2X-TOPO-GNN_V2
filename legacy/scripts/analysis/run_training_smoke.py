from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.training_smoke import (  # noqa: E402
    load_training_smoke_config,
    run_curriculum_training_smoke,
    run_tiny_training_smoke,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the deterministic GNN training smoke.")
    parser.add_argument("--config", default="configs/training_smoke.yaml")
    parser.add_argument("--smoke", action="store_true", help="Compatibility flag for the smoke target.")
    parser.add_argument("--curriculum", action="store_true", help="Run the all-to-topk curriculum smoke.")
    parser.add_argument("--avalanche-profile", default=None)
    parser.add_argument("--json-out", default=".agent/tmp/training_smoke.json")
    parser.add_argument("--md-out", default=".agent/tmp/training_smoke.md")
    parser.add_argument("--no-write", action="store_true")
    return parser


def _contract_ok(report: Mapping[str, Any], *, curriculum: bool) -> bool:
    if curriculum:
        return bool(report.get("contract_ok")) and bool(report.get("finite_gradients_both_phases"))
    return (
        bool(report.get("contract_ok"))
        and bool(report.get("loss_finite_all_steps"))
        and bool(report.get("gradients_finite_all_steps"))
        and bool(report.get("parameters_changed"))
    )


def _summary_lines(report: Mapping[str, Any], *, curriculum: bool) -> list[str]:
    if curriculum:
        return [
            f"training_smoke_mode=curriculum",
            f"contract_ok={report.get('contract_ok')}",
            f"finite_gradients_both_phases={report.get('finite_gradients_both_phases')}",
            f"parameters_changed_both_phases={report.get('parameters_changed_both_phases')}",
            f"phase_0_final_loss={report.get('phase_0_final_loss')}",
            f"phase_1_final_loss={report.get('phase_1_final_loss')}",
            f"support_switch_active_edge_count_change={report.get('support_switch_active_edge_count_change')}",
        ]
    return [
        f"training_smoke_mode=tiny",
        f"contract_ok={report.get('contract_ok')}",
        f"optimizer_steps_completed={report.get('optimizer_steps_completed')}",
        f"initial_total_loss={report.get('initial_total_loss')}",
        f"final_total_loss={report.get('final_total_loss')}",
        f"loss_delta={report.get('loss_delta')}",
        f"parameters_changed={report.get('parameters_changed')}",
        f"gradients_finite_all_steps={report.get('gradients_finite_all_steps')}",
        f"support_mode={report.get('support_mode')}",
        f"topk_backend={report.get('topk_backend')}",
    ]


def _write_outputs(report: Mapping[str, Any], json_out: str | Path, md_out: str | Path, lines: list[str]) -> None:
    json_path = Path(json_out)
    if not json_path.is_absolute():
        json_path = ROOT / json_path
    md_path = Path(md_out)
    if not md_path.is_absolute():
        md_path = ROOT / md_path
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text("# Training Smoke\n\n" + "\n".join(f"- `{line}`" for line in lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = dict(load_training_smoke_config(ROOT / args.config))
    if args.avalanche_profile is not None:
        config["avalanche_profile"] = args.avalanche_profile
    report = run_curriculum_training_smoke(config) if args.curriculum else run_tiny_training_smoke(config)
    ok = _contract_ok(report, curriculum=args.curriculum)
    lines = _summary_lines(report, curriculum=args.curriculum)
    for line in lines:
        print(line)
    if not args.no_write:
        _write_outputs(report, args.json_out, args.md_out, lines)
        print(f"json_out={Path(args.json_out)}")
        print(f"md_out={Path(args.md_out)}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
