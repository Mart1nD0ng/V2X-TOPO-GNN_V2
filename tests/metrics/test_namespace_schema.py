"""G-METRIC-NAMESPACE (Guarded-CDQ2 round, Phase 1): the canonical metric schema + namespaces.

Locks the contract from GUARDED_CDQ2_TECHNICAL_SPEC.md §7:
- every serialized headline metric lives in an explicit namespace (macro / strict_audit / diagnostic /
  sampling / cdq / runtime) and carries ``metric_namespace_version="macrostate_v2"``;
- ambiguous bare names (F, F_wrong, F_disagree, S_allcorrect, failure, reliability, D, delay,
  P_correct) may NEVER appear as a serialized key in a headline result;
- legacy node-union/product surrogates can only be emitted behind an explicit ``legacy=True`` flag and
  are rejected outright by the figure-guard;
- the four macrostate outcomes are complete and sum to 1.
"""

import copy
import json

import pytest

from src.metrics import namespaces as ns
from src.metrics import schema


# ---------------------------------------------------------------- namespaces vocabulary
def test_namespace_version_constant():
    assert ns.METRIC_NAMESPACE_VERSION == "macrostate_v2"


def test_namespace_of_by_prefix():
    assert ns.namespace_of("macro_F_wrong") == "macro"
    assert ns.namespace_of("strict_any_wrong") == "strict_audit"
    assert ns.namespace_of("diagnostic_node_wrong_mean") == "diagnostic"
    assert ns.namespace_of("sampling_keff") == "sampling"
    assert ns.namespace_of("cdq_eta") == "cdq"
    assert ns.namespace_of("surrogate_nodeunion_F_wrong") == "legacy"
    assert ns.namespace_of("totally_unknown_key") is None


def test_forbidden_bare_is_exact_match_only():
    # the bare tokens are forbidden ...
    for bad in ("F", "F_wrong", "F_disagree", "S_allcorrect", "failure", "reliability",
                "D", "delay", "P_correct"):
        assert ns.is_forbidden_key(bad), bad
    # ... but their fully-namespaced cousins are NOT (substring must not trip it)
    for ok in ("macro_F_wrong", "macro_P_correct", "macro_F_wrong_ci", "surrogate_nodeunion_F_wrong",
               "cdq_deadline_pressure", "macro_F_wrong_delta"):
        assert not ns.is_forbidden_key(ok), ok


def test_iter_keys_recurses():
    obj = {"a": 1, "b": {"c": 2, "d": [{"e": 3}, {"f": 4}]}}
    assert set(ns.iter_keys(obj)) == {"a", "b", "c", "d", "e", "f"}


# ---------------------------------------------------------------- macro block builder
def test_macro_block_is_fully_namespaced_and_sums_to_one():
    blk = schema.macro_block(0.62, 0.20, 0.03, 0.15)
    assert set(blk) >= {"macro_P_correct", "macro_F_wrong", "macro_F_split", "macro_F_deadline"}
    assert all(k.startswith("macro_") for k in blk)
    assert abs(sum(blk[k] for k in ns.MACRO_OUTCOME_KEYS) - 1.0) < 1e-9


def test_macro_block_rejects_non_summing_outcomes():
    with pytest.raises(schema.MetricSchemaError):
        schema.macro_block(0.62, 0.20, 0.03, 0.99)   # sums to 1.84


def test_macro_block_carries_optional_ci():
    blk = schema.macro_block(0.62, 0.20, 0.03, 0.15,
                             ci={"macro_F_wrong": (0.19, 0.21), "macro_P_correct": (0.61, 0.63)})
    assert blk["macro_F_wrong_ci"] == (0.19, 0.21)
    assert blk["macro_P_correct_ci"] == (0.61, 0.63)


# ---------------------------------------------------------------- full result record
def _good_record():
    return schema.build_result_record(
        policy="ESP", query_family="ESP",
        macro=schema.macro_block(0.62, 0.20, 0.03, 0.15),
        runtime={"runtime_seconds": 1.2})


def test_build_record_has_version_and_namespaced_blocks():
    rec = _good_record()
    assert rec["metric_namespace_version"] == "macrostate_v2"
    assert rec["policy"] == "ESP" and rec["query_family"] == "ESP"
    assert "macro" in rec
    schema.validate_result(rec)            # must not raise


def test_validate_rejects_wrong_version():
    rec = _good_record()
    rec["metric_namespace_version"] = "v1"
    with pytest.raises(schema.MetricSchemaError):
        schema.validate_result(rec)


def test_validate_rejects_ambiguous_bare_key_anywhere():
    rec = _good_record()
    rec["macro"]["F_wrong"] = 0.20         # smuggle a bare ambiguous key into the headline
    with pytest.raises(schema.MetricSchemaError):
        schema.validate_result(rec)


def test_validate_rejects_incomplete_macro_outcomes():
    rec = _good_record()
    del rec["macro"]["macro_F_split"]
    with pytest.raises(schema.MetricSchemaError):
        schema.validate_result(rec)


def test_validate_rejects_macro_outcomes_not_summing_to_one():
    rec = _good_record()
    rec["macro"]["macro_F_deadline"] = 0.95
    with pytest.raises(schema.MetricSchemaError):
        schema.validate_result(rec)


def test_validate_rejects_foreign_key_in_macro_block():
    rec = _good_record()
    rec["macro"]["sampling_keff"] = 3.0    # wrong namespace inside the macro block
    with pytest.raises(schema.MetricSchemaError):
        schema.validate_result(rec)


# ---------------------------------------------------------------- legacy / surrogate gating
def test_legacy_block_requires_explicit_flag():
    legacy = {"surrogate_nodeunion_F_wrong": 0.3, "surrogate_product_S_allcorrect": 0.5}
    with pytest.raises(schema.MetricSchemaError):
        schema.build_result_record(policy="ESP", query_family="ESP",
                                   macro=schema.macro_block(0.62, 0.20, 0.03, 0.15),
                                   legacy=legacy)                      # allow_legacy defaults False
    rec = schema.build_result_record(policy="ESP", query_family="ESP",
                                     macro=schema.macro_block(0.62, 0.20, 0.03, 0.15),
                                     legacy=legacy, allow_legacy=True)
    assert rec["legacy"]["surrogate_nodeunion_F_wrong"] == 0.3


def test_headline_validation_forbids_legacy_block():
    rec = schema.build_result_record(policy="ESP", query_family="ESP",
                                     macro=schema.macro_block(0.62, 0.20, 0.03, 0.15),
                                     legacy={"surrogate_nodeunion_F_wrong": 0.3}, allow_legacy=True)
    with pytest.raises(schema.MetricSchemaError):
        schema.validate_result(rec, headline=True)
    schema.validate_result(rec, headline=False)   # non-headline (diagnostics file) tolerates it


def test_figure_guard_rejects_any_legacy_or_bare_metric():
    rec = _good_record()
    schema.assert_no_legacy_metrics(rec)           # clean record passes
    rec2 = copy.deepcopy(rec)
    rec2["legacy"] = {"surrogate_nodeunion_F_wrong": 0.3}
    with pytest.raises(schema.MetricSchemaError):
        schema.assert_no_legacy_metrics(rec2)
    rec3 = copy.deepcopy(rec)
    rec3["macro"]["S_allcorrect"] = 0.5
    with pytest.raises(schema.MetricSchemaError):
        schema.assert_no_legacy_metrics(rec3)


# ---------------------------------------------------------------- migration shim (legacy reader)
def test_migrate_legacy_factorial_arm_produces_clean_macro_block():
    legacy_arm = {"P_correct": 0.62, "F_wrong": 0.20, "F_split": 0.03, "F_deadline": 0.15}
    blk = schema.migrate_legacy_factorial_cell(legacy_arm)
    schema.validate_macro_block(blk)               # passes
    # and the produced block has NO ambiguous bare key
    assert not any(ns.is_forbidden_key(k) for k in ns.iter_keys(blk))


def test_factorial_result_converter_is_clean():
    from src.evaluation.cdq2_factorial import FactorialResult
    r = FactorialResult(P_correct=0.62, F_wrong=0.20, F_split=0.03, F_deadline=0.15, n_pool=18000)
    blk = r.to_macro_block()
    schema.validate_macro_block(blk)
    rec = r.to_result_record(policy="CDQ2", query_family="CDQ2")
    schema.validate_result(rec)
    assert "macro_F_wrong_ci" in rec["macro"]


def test_dynamic_mc_result_macro_block_from_basins():
    from src.validation.dynamic_mc import DynamicMCResult
    r = DynamicMCResult(
        F_disagree=0.0, F_wrong=0.0, S_allcorrect=0.0,
        F_disagree_ci=(0.0, 0.0), F_wrong_ci=(0.0, 0.0), S_allcorrect_ci=(0.0, 0.0),
        decided_correct_freq=None, decided_wrong_freq=None, undecided_freq=None,
        mean_rounds_to_decide=0.0, mean_finalisation_time=0.0, finished_fraction=0.0, num_trials=100,
        basin_P_correct=0.62, basin_F_wrong=0.20, basin_F_split=0.03, basin_F_deadline=0.15,
        basin_F_wrong_ci=(0.19, 0.21), basin_tau_correct_mean=2.5)
    blk = r.macro_block()
    schema.validate_macro_block(blk)
    assert blk["macro_T_confirm"] == 2.5 and blk["macro_F_wrong_ci"] == (0.19, 0.21)


def test_dynamic_mc_result_macro_block_raises_when_unpopulated():
    from src.validation.dynamic_mc import DynamicMCResult
    r = DynamicMCResult(
        F_disagree=0.0, F_wrong=0.0, S_allcorrect=0.0,
        F_disagree_ci=(0.0, 0.0), F_wrong_ci=(0.0, 0.0), S_allcorrect_ci=(0.0, 0.0),
        decided_correct_freq=None, decided_wrong_freq=None, undecided_freq=None,
        mean_rounds_to_decide=0.0, mean_finalisation_time=0.0, finished_fraction=0.0, num_trials=100)
    with pytest.raises(ValueError):
        r.macro_block()       # basins are nan -> no service_profile was passed


# ---------------------------------------------------------------- adversarial-audit regressions
def test_headline_rejects_toplevel_legacy_smuggling():
    """Audit hole #1: a surrogate metric at top level / under a foreign key must NOT pass headline."""
    rec = _good_record()
    rec["surrogate_product_S_allcorrect"] = 0.5
    with pytest.raises(schema.MetricSchemaError):
        schema.validate_result(rec, headline=True)
    rec2 = _good_record()
    rec2["audit"] = {"surrogate_nodeunion_F_disagree": 0.1}    # foreign top-level key
    with pytest.raises(schema.MetricSchemaError):
        schema.validate_result(rec2, headline=True)


def test_unknown_toplevel_key_rejected():
    rec = _good_record()
    rec["sneaky"] = {"whatever": 1}
    with pytest.raises(schema.MetricSchemaError):
        schema.validate_result(rec)


def test_nan_macro_outcome_rejected():
    """Audit hole #2: NaN outcome must not slip through the sum-to-1 check (abs(nan-1)>tol is False)."""
    with pytest.raises(schema.MetricSchemaError):
        schema.macro_block(float("nan"), 0.0, 0.0, 0.0, sum_tol=1.0)
    blk = {"macro_P_correct": float("nan"), "macro_F_wrong": 0.0,
           "macro_F_split": 0.0, "macro_F_deadline": 0.0}
    with pytest.raises(schema.MetricSchemaError):
        schema.validate_macro_block(blk)


def test_concept_based_legacy_detection():
    """Audit hole #3: node-union/all-correct spellings WITHOUT the surrogate_ prefix are still legacy."""
    assert ns.is_legacy_key("nodeunion_F_wrong")
    assert ns.is_legacy_key("node_union_disagree")
    assert ns.is_legacy_key("global_product_allcorrect")
    assert not ns.is_legacy_key("macro_F_wrong")
    rec = _good_record()
    rec["runtime"]["runtime_nodeunion_proxy"] = 0.1
    with pytest.raises(schema.MetricSchemaError):
        schema.assert_no_legacy_metrics(rec)


def test_iter_keys_traverses_sets_and_mappings():
    from collections import OrderedDict
    assert "F_wrong" in set(ns.iter_keys(OrderedDict(a={"F_wrong": 1})))
    assert "z" in set(ns.iter_keys({"s": {"x", "y"}, "z": 1}))   # 'z' is a key; set values traversed


def test_builder_rejects_incomplete_macro_block():
    with pytest.raises(schema.MetricSchemaError):
        schema.build_result_record(policy="ESP", query_family="ESP",
                                   macro={"macro_P_correct": 0.6, "macro_F_wrong": 0.4})


def test_old_ambiguous_factorial_json_fails_validation():
    """The committed S15 factorial JSON uses bare F_wrong/P_correct keys -> must be rejected as a
    headline record (this is the cleanup biting)."""
    from pathlib import Path
    p = Path("docs/gate_evidence/macrostate/cdq2_factorial_results.json")
    if not p.exists():
        pytest.skip("S15 evidence not present")
    old = json.loads(p.read_text())
    bare = set(ns.iter_keys(old))
    assert any(ns.is_forbidden_key(k) for k in bare)   # it DOES contain forbidden bare keys
    with pytest.raises(schema.MetricSchemaError):
        schema.assert_no_forbidden_keys(old)
