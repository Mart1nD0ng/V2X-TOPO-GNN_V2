"""Canonical result schema + validators for the Guarded-CDQ2 round (GUARDED_CDQ2_TECHNICAL_SPEC.md §7.4).

Every serialized headline result is a *record* with an explicit ``metric_namespace_version`` and a small
set of namespace blocks (``macro`` / ``strict_audit`` / ``diagnostic`` / ``sampling`` / ``cdq`` /
``runtime``). This module builds those records and enforces the bans so a result writer or a figure
script cannot revive a legacy node-union/product surrogate as a headline metric.

Three guards, layered:
- ``assert_no_forbidden_keys``  -- no ambiguous bare token (F_wrong, S_allcorrect, ...) anywhere;
- ``validate_macro_block``      -- the macro block is complete (four outcomes) and sums to 1;
- ``validate_result``           -- the whole record: version, namespaced blocks, no legacy in headline.
``assert_no_legacy_metrics`` is the figure-script guard (constraint #13): a figure may only *read*
clean records and must refuse legacy/surrogate fields outright.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from src.metrics import namespaces as ns

# the only top-level keys a clean §7.4 result record may carry (everything else is smuggling)
_ALLOWED_TOPLEVEL = frozenset({
    "metric_namespace_version", "policy", "query_family", "hashes", "legacy",
    *ns.NAMESPACE_GROUPS,
})


class MetricSchemaError(ValueError):
    """Raised when a result record violates the macrostate_v2 metric-namespace contract."""


# ------------------------------------------------------------------ block builders
def macro_block(P_correct: float, F_wrong: float, F_split: float, F_deadline: float, *,
                T_confirm: float | None = None, D95: float | None = None, D99: float | None = None,
                CVaR99: float | None = None, ci: Mapping[str, tuple[float, float]] | None = None,
                sum_tol: float = 1e-6) -> dict[str, Any]:
    """Build a fully-namespaced macro block ``{macro_P_correct, macro_F_wrong, ...}``.

    The four outcomes are validated to sum to 1 (they are mutually-exclusive first-hitting basins).
    Optional latency/tail fields and per-key confidence intervals (``ci``) are added as ``*_ci`` keys.
    """
    blk: dict[str, Any] = {
        "macro_P_correct": float(P_correct),
        "macro_F_wrong": float(F_wrong),
        "macro_F_split": float(F_split),
        "macro_F_deadline": float(F_deadline),
    }
    for name, val in (("macro_T_confirm", T_confirm), ("macro_D95", D95),
                      ("macro_D99", D99), ("macro_CVaR99", CVaR99)):
        if val is not None:
            blk[name] = float(val)
    if ci:
        for key, lohi in ci.items():
            base = key if key.startswith("macro_") else f"macro_{key}"
            blk[f"{base}_ci"] = (float(lohi[0]), float(lohi[1]))
    validate_macro_block(blk, sum_tol=sum_tol)
    return blk


def macro_delta_block(dP_correct: float, dF_wrong: float, dF_split: float,
                      dF_deadline: float) -> dict[str, float]:
    """A namespaced *delta* block (policy_A - policy_B). Uses the ``_delta`` suffix so it is NOT
    mistaken for an outcome block (deltas sum to ~0, not 1)."""
    return {
        "macro_P_correct_delta": float(dP_correct),
        "macro_F_wrong_delta": float(dF_wrong),
        "macro_F_split_delta": float(dF_split),
        "macro_F_deadline_delta": float(dF_deadline),
    }


def build_result_record(*, policy: str, query_family: str,
                        macro: Mapping[str, Any],
                        strict_audit: Mapping[str, Any] | None = None,
                        diagnostic: Mapping[str, Any] | None = None,
                        sampling: Mapping[str, Any] | None = None,
                        cdq: Mapping[str, Any] | None = None,
                        runtime: Mapping[str, Any] | None = None,
                        hashes: Mapping[str, str] | None = None,
                        legacy: Mapping[str, Any] | None = None,
                        allow_legacy: bool = False) -> dict[str, Any]:
    """Assemble a §7.4 result record with ``metric_namespace_version="macrostate_v2"``.

    ``query_family`` must be ESP | CDQ2 | Guarded-CDQ2. ``hashes`` (physics/profile/evidence/scene/
    policy/checkpoint) are accepted here but their *enforcement* is G-RESULT-MANIFEST. A ``legacy``
    block (surrogate_*) is only permitted with ``allow_legacy=True`` and is rejected by the headline
    validator -- it exists solely for explicitly-flagged legacy/diagnostic files.
    """
    if query_family not in ("ESP", "CDQ2", "Guarded-CDQ2"):
        raise MetricSchemaError(f"query_family must be ESP|CDQ2|Guarded-CDQ2, got {query_family!r}")
    if legacy is not None and not allow_legacy:
        raise MetricSchemaError(
            "legacy/surrogate metrics may only be emitted with allow_legacy=True "
            "(node-union/product surrogates are never a headline metric)")

    rec: dict[str, Any] = {
        "metric_namespace_version": ns.METRIC_NAMESPACE_VERSION,
        "policy": str(policy),
        "query_family": query_family,
    }
    if hashes:
        rec["hashes"] = dict(hashes)

    blocks = {
        ns.MACRO: macro, ns.STRICT_AUDIT: strict_audit, ns.DIAGNOSTIC: diagnostic,
        ns.SAMPLING: sampling, ns.CDQ: cdq, ns.RUNTIME: runtime,
    }
    for group, block in blocks.items():
        if block is None:
            continue
        if group == ns.MACRO:
            validate_macro_block(block)        # the macro block is required AND must be complete
        else:
            _check_block_namespace(group, block)
        rec[group] = dict(block)
    if legacy is not None:
        rec["legacy"] = dict(legacy)
    return rec


# ------------------------------------------------------------------ validators
def assert_no_forbidden_keys(obj: Any) -> None:
    """Raise if any ambiguous bare token (F_wrong, S_allcorrect, P_correct, ...) appears as a key
    anywhere in ``obj`` (recursively). The exact-match ban-list is in ``namespaces.FORBIDDEN_BARE``."""
    offenders = sorted({k for k in ns.iter_keys(obj) if ns.is_forbidden_key(k)})
    if offenders:
        raise MetricSchemaError(
            f"ambiguous bare metric keys are forbidden in a serialized result: {offenders} "
            f"(use the macro_/surrogate_/... namespace)")


def validate_macro_block(block: Mapping[str, Any], *, sum_tol: float = 1e-6) -> None:
    """The macro block must contain exactly the four outcome keys (plus optional macro_* extras), use
    only canonical macro_* vocabulary, and the four outcomes must sum to 1."""
    assert_no_forbidden_keys(block)
    for key in block:
        if ns.namespace_of(key) != ns.MACRO:
            raise MetricSchemaError(f"non-macro key {key!r} in a macro block")
        if ns.canonical_base(key) not in ns.MACRO_KEYS:
            raise MetricSchemaError(f"unknown macro key {key!r} (not in the macrostate_v2 vocabulary)")
    missing = [k for k in ns.MACRO_OUTCOME_KEYS if k not in block]
    if missing:
        raise MetricSchemaError(f"incomplete macro outcomes -- missing {missing}")
    vals = [float(block[k]) for k in ns.MACRO_OUTCOME_KEYS]
    if not all(math.isfinite(v) for v in vals):
        raise MetricSchemaError("macro outcomes must be finite (no NaN/inf basin probabilities)")
    total = sum(vals)
    if abs(total - 1.0) > sum_tol:
        raise MetricSchemaError(
            f"macro outcomes must sum to 1 (P_correct+F_wrong+F_split+F_deadline), got {total:.6f}")


def _check_block_namespace(group: str, block: Mapping[str, Any]) -> None:
    """Every key in a namespace block must belong to that namespace (no cross-namespace smuggling)."""
    if group == ns.MACRO:
        validate_macro_block(block) if _has_all_outcomes(block) else _check_macro_keys_only(block)
        return
    assert_no_forbidden_keys(block)
    for key in block:
        g = ns.namespace_of(key)
        if g is None or g != group:
            raise MetricSchemaError(f"key {key!r} does not belong to the {group!r} namespace block")


def _has_all_outcomes(block: Mapping[str, Any]) -> bool:
    return all(k in block for k in ns.MACRO_OUTCOME_KEYS)


def _check_macro_keys_only(block: Mapping[str, Any]) -> None:
    assert_no_forbidden_keys(block)
    for key in block:
        if ns.namespace_of(key) != ns.MACRO or ns.canonical_base(key) not in ns.MACRO_KEYS:
            raise MetricSchemaError(f"unknown/foreign macro key {key!r}")


def validate_result(record: Mapping[str, Any], *, headline: bool = True, sum_tol: float = 1e-6) -> None:
    """Validate a full result record against the macrostate_v2 contract.

    headline=True (default) additionally forbids any legacy/surrogate block -- a headline result must
    be free of node-union/product surrogates. headline=False tolerates an explicitly-flagged legacy
    block (for diagnostic/comparison files) but still bans bare ambiguous keys.
    """
    if record.get("metric_namespace_version") != ns.METRIC_NAMESPACE_VERSION:
        raise MetricSchemaError(
            f"metric_namespace_version must be {ns.METRIC_NAMESPACE_VERSION!r}, "
            f"got {record.get('metric_namespace_version')!r}")
    assert_no_forbidden_keys(record)

    # top-level whitelist: a §7.4 record may carry ONLY the known keys. This closes legacy/foreign-metric
    # smuggling under an unrecognized top-level key or as a bare top-level scalar (adversarial hole #1).
    unknown = [k for k in record if k not in _ALLOWED_TOPLEVEL]
    if unknown:
        raise MetricSchemaError(f"unknown top-level keys in result record: {sorted(unknown)}")

    if ns.MACRO not in record:
        raise MetricSchemaError("a result record must carry a macro block (the headline outcomes)")
    validate_macro_block(record[ns.MACRO], sum_tol=sum_tol)

    for group in ns.NAMESPACE_GROUPS:
        if group in record and group != ns.MACRO:
            _check_block_namespace(group, record[group])

    # whole-record legacy scan: a headline record must be free of ANY legacy/surrogate metric, wherever
    # it sits (not just a literal "legacy" block) -- the figure guard and the headline validator agree.
    legacy_keys = legacy_keys_in(record)
    if headline and (legacy_keys or "legacy" in record):
        raise MetricSchemaError(
            "a headline result must not contain any legacy/surrogate metric "
            f"(found {legacy_keys or ['legacy block']}); use headline=False for diagnostic files")
    if "legacy" in record:
        for key in ns.iter_keys(record["legacy"]):
            if ns.namespace_of(key) in ns.NAMESPACE_GROUPS:
                raise MetricSchemaError(f"clean-namespace key {key!r} must not live in the legacy block")


def assert_no_legacy_metrics(obj: Any) -> None:
    """Figure-script guard (constraint #13): refuse any legacy/surrogate field or bare ambiguous key.
    A figure may only consume clean records; it must never recompute or revive a legacy metric."""
    assert_no_forbidden_keys(obj)
    legacy = sorted({k for k in ns.iter_keys(obj) if ns.is_legacy_key(k)})
    if legacy or (isinstance(obj, Mapping) and "legacy" in obj):
        raise MetricSchemaError(
            f"figure inputs must be free of legacy/surrogate metrics, found {legacy or ['legacy block']}")


# ------------------------------------------------------------------ migration shim (legacy readers only)
_LEGACY_FACTORIAL_MAP = {
    "P_correct": "macro_P_correct", "F_wrong": "macro_F_wrong",
    "F_split": "macro_F_split", "F_deadline": "macro_F_deadline",
}


def migrate_legacy_factorial_cell(arm: Mapping[str, Any], *, sum_tol: float = 1e-6) -> dict[str, Any]:
    """Compatibility shim: rename a legacy factorial arm ``{P_correct, F_wrong, F_split, F_deadline}``
    into a clean macro block. Pure key-renaming of already-computed numbers -- NO metric recomputation."""
    try:
        return macro_block(arm["P_correct"], arm["F_wrong"], arm["F_split"], arm["F_deadline"],
                           sum_tol=sum_tol)
    except KeyError as e:  # pragma: no cover - defensive
        raise MetricSchemaError(f"legacy factorial arm missing outcome {e}") from e


def forbidden_keys_in(obj: Any) -> list[str]:
    """Return the sorted list of forbidden bare keys present in ``obj`` (for reporting, not raising)."""
    return sorted({k for k in ns.iter_keys(obj) if ns.is_forbidden_key(k)})


def legacy_keys_in(obj: Any) -> list[str]:
    return sorted({k for k in ns.iter_keys(obj) if ns.is_legacy_key(k)})


__all__ = [
    "MetricSchemaError", "macro_block", "macro_delta_block", "build_result_record",
    "assert_no_forbidden_keys", "validate_macro_block", "validate_result",
    "assert_no_legacy_metrics", "migrate_legacy_factorial_cell",
    "forbidden_keys_in", "legacy_keys_in",
]
