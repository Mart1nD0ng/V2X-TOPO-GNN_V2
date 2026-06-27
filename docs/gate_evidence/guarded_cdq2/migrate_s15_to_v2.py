"""G-METRIC-NAMESPACE migration shim (Guarded-CDQ2 round, Phase 0 + Phase 1).

Reads the committed S15 factorial evidence (legacy bare keys P_correct/F_wrong/F_split/F_deadline) and
re-emits it as a namespaced ``macrostate_v2`` record set -- a PURE key-rename of already-computed
numbers (NO metric recomputation, constraint #13). The original S15 JSON is NOT overwritten (Phase 0:
"do not overwrite S15 evidence"); it is copied to an archive path and the clean version is written
under the Guarded-CDQ2 evidence dir.

Run:  python docs/gate_evidence/guarded_cdq2/migrate_s15_to_v2.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from src.metrics import namespaces as ns
from src.metrics import schema

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "docs/gate_evidence/macrostate/cdq2_factorial_results.json"
ARCHIVE = ROOT / "docs/gate_evidence/macrostate/archive/cdq2_factorial_results_S15.json"
OUT = ROOT / "docs/gate_evidence/guarded_cdq2/cdq2_factorial_namespaced.json"


def _family(arm_name: str) -> str:
    return "ESP" if arm_name == "ESP" else "CDQ2"


def migrate() -> dict:
    old = json.loads(SRC.read_text())
    # sanity: the legacy artifact genuinely carries forbidden bare keys (this migration is what cleans it)
    assert schema.forbidden_keys_in(old), "expected legacy bare keys in the S15 artifact"

    cells_out = {}
    for cell_id, cell in old["cells"].items():
        rec_cell = {}
        for arm in ("ESP", "CDQ2"):
            block = schema.migrate_legacy_factorial_cell(cell[arm])
            # attach the CIs that were stored under the legacy *_CI_<ARM> keys
            ci = {}
            if f"P_correct_CI_{arm}" in cell:
                ci["macro_P_correct_ci"] = tuple(cell[f"P_correct_CI_{arm}"])
            if f"F_wrong_CI_{arm}" in cell:
                ci["macro_F_wrong_ci"] = tuple(cell[f"F_wrong_CI_{arm}"])
            block.update(ci)
            rec = schema.build_result_record(policy=arm, query_family=_family(arm), macro=block)
            schema.validate_result(rec)            # headline-clean
            rec_cell[arm] = rec
        d = cell["delta"]
        rec_cell["delta"] = schema.macro_delta_block(d["P_correct"], d["F_wrong"],
                                                     d["F_split"], d["F_deadline"])
        cells_out[cell_id] = rec_cell

    out = {
        "metric_namespace_version": ns.METRIC_NAMESPACE_VERSION,
        "source": "migrated from docs/gate_evidence/macrostate/cdq2_factorial_results.json (S15)",
        "note": "pure key-rename of S15 numbers; no metric recomputation. Headline judge unchanged "
                "(independent dynamic-MC basin first-hitting).",
        "config": old["config"],
        "cells": cells_out,
    }
    # the migrated artifact must be free of every forbidden bare key
    assert not schema.forbidden_keys_in(out), schema.forbidden_keys_in(out)
    return out


def main() -> None:
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not ARCHIVE.exists():
        shutil.copy2(SRC, ARCHIVE)            # Phase 0: preserve S15 evidence (non-destructive)
    out = migrate()
    OUT.write_text(json.dumps(out, indent=2))
    print(f"archived  S15 -> {ARCHIVE.relative_to(ROOT)}")
    print(f"namespaced    -> {OUT.relative_to(ROOT)}  ({len(out['cells'])} cells, macrostate_v2)")


if __name__ == "__main__":
    main()
