from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topology.constructor_profile import (  # noqa: E402
    DEFAULT_JSON_OUT,
    DEFAULT_MD_OUT,
    DEFAULT_READINESS_ARTIFACT,
    load_constructor_bottleneck_diagnostic_report,
    run_constructor_bottleneck_diagnostic,
    write_constructor_bottleneck_diagnostic_json,
    write_constructor_bottleneck_diagnostic_markdown,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile the sparse topology constructor.")
    parser.add_argument("--summary-only", action="store_true", help="Read an existing JSON report and print a summary.")
    parser.add_argument("--json-in", default=str(DEFAULT_JSON_OUT), help="Existing JSON report for --summary-only.")
    parser.add_argument("--json-out", default=str(DEFAULT_JSON_OUT), help="Path for the JSON report.")
    parser.add_argument("--md-out", default=str(DEFAULT_MD_OUT), help="Path for the Markdown report.")
    parser.add_argument("--readiness-artifact", default=str(DEFAULT_READINESS_ARTIFACT))
    parser.add_argument("--node-counts", default=None, help="Comma-separated node counts, for example 500,2000.")
    parser.add_argument("--candidate-degree", type=int, default=8)
    parser.add_argument("--max-out-degree", type=int, default=4)
    parser.add_argument("--include-all-mode", action="store_true")
    parser.add_argument("--topk-backend", choices=("legacy", "segmented_fast", "both"), default="legacy")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-write", action="store_true", help="Print only; do not create JSON or Markdown outputs.")
    return parser


def _print_summary(report: Mapping[str, Any]) -> None:
    print(f"constructor_bottleneck_status={report.get('constructor_bottleneck_status')}")
    print(f"optimization_recommendation={report.get('optimization_recommendation')}")
    print(f"recommended_next_stage={report.get('recommended_next_stage')}")
    print(f"case_count={report.get('case_count')}")
    for case in report.get("cases", []):
        if not isinstance(case, Mapping):
            continue
        print(
            "constructor_case "
            f"node_count={case.get('node_count')} "
            f"support_mode={case.get('support_mode')} "
            f"topk_backend={case.get('topk_backend')} "
            f"total_s={case.get('total_constructor_time_s')} "
            f"active_edges={case.get('active_edge_count')}"
        )
    for item in report.get("backend_comparisons", []):
        if not isinstance(item, Mapping):
            continue
        print(
            "constructor_backend_comparison "
            f"node_count={item.get('node_count')} "
            f"legacy_time_s={item.get('legacy_time_s')} "
            f"segmented_fast_time_s={item.get('segmented_fast_time_s')} "
            f"speedup_ratio={item.get('speedup_ratio')} "
            f"equivalence_ok={item.get('equivalence_ok')}"
        )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.summary_only:
        report = load_constructor_bottleneck_diagnostic_report(args.json_in)
    else:
        report = run_constructor_bottleneck_diagnostic(
            readiness_artifact=args.readiness_artifact,
            node_counts=args.node_counts,
            candidate_degree=args.candidate_degree,
            max_out_degree=args.max_out_degree,
            include_all_mode=args.include_all_mode,
            topk_backend=args.topk_backend,
            force=args.force,
        )

    _print_summary(report)
    if not args.no_write:
        write_constructor_bottleneck_diagnostic_json(report, args.json_out)
        write_constructor_bottleneck_diagnostic_markdown(report, args.md_out)
        print(f"json_out={Path(args.json_out)}")
        print(f"md_out={Path(args.md_out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
