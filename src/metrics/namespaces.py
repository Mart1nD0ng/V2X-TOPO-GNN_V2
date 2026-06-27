"""Canonical metric namespaces for the Guarded-CDQ2 round (GUARDED_CDQ2_TECHNICAL_SPEC.md §7).

The codebase mixes four kinds of quantity that must never be confused in a serialized headline:
macrostate basin outcomes, strict node-pair audits, legacy node-union/product *surrogates*, and
per-node diagnostics. This module is the single source of truth for the explicit namespaces and the
ban-list, so neither a result writer nor a figure script can silently revive a legacy metric.

A *fully-qualified* key carries its namespace as a prefix (``macro_F_wrong`` not ``F_wrong``); the
prefix makes the key self-describing even when a block is flattened into a paper table. The
ambiguous *bare* tokens (``F_wrong``, ``S_allcorrect`` ...) are forbidden as serialized keys.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Iterator

METRIC_NAMESPACE_VERSION = "macrostate_v2"

# ------------------------------------------------------------------ namespace group names
MACRO = "macro"
STRICT_AUDIT = "strict_audit"
DIAGNOSTIC = "diagnostic"
SAMPLING = "sampling"
CDQ = "cdq"
RUNTIME = "runtime"
LEGACY = "legacy"           # surrogate_* (node-union / global product) -- NEVER a headline metric

# the namespace blocks a clean result record may carry (legacy is opt-in + non-headline only)
NAMESPACE_GROUPS = (MACRO, STRICT_AUDIT, DIAGNOSTIC, SAMPLING, CDQ, RUNTIME)

# prefix -> group (order matters only for readability; prefixes are disjoint)
_PREFIX_GROUP = (
    ("macro_", MACRO),
    ("strict_", STRICT_AUDIT),
    ("diagnostic_", DIAGNOSTIC),
    ("sampling_", SAMPLING),
    ("cdq_", CDQ),
    ("runtime_", RUNTIME),
    ("surrogate_", LEGACY),
)

# ------------------------------------------------------------------ canonical vocabularies (spec §7.2)
# the four basin outcomes that are mutually exclusive and sum to 1 (headline)
MACRO_OUTCOME_KEYS = ("macro_P_correct", "macro_F_wrong", "macro_F_split", "macro_F_deadline")
MACRO_KEYS = frozenset(MACRO_OUTCOME_KEYS + (
    "macro_T_confirm", "macro_D95", "macro_D99", "macro_CVaR99",
))

STRICT_KEYS = frozenset({"strict_any_disagreement", "strict_any_wrong", "strict_any_unfinalized"})

# legacy / surrogate metrics -- explicitly namespaced so they are *traceably* legacy, never bare
SURROGATE_KEYS = frozenset({
    "surrogate_product_S_allcorrect",
    "surrogate_nodeunion_F_wrong",
    "surrogate_nodeunion_F_disagree",
})

DIAGNOSTIC_KEYS = frozenset({
    "diagnostic_node_failure_mean", "diagnostic_node_wrong_mean",
    "diagnostic_node_deadline_mean", "diagnostic_spatial_F_i",
})

SAMPLING_KEYS = frozenset({
    "sampling_progress_g", "sampling_drift_delta", "sampling_keff",
    "sampling_selected_corr", "sampling_minority_exposure", "sampling_receiver_load",
})

CDQ_KEYS = frozenset({
    "cdq_eta", "cdq_guard_active", "cdq_risk_slack_wrong",
    "cdq_risk_slack_split", "cdq_deadline_pressure",
})

# ------------------------------------------------------------------ forbidden bare tokens (spec §7.3)
# Exact serialized keys that are banned everywhere in a result. The §7.3 core list plus the bare
# macrostate outcomes (which must always be macro_*). Substrings of namespaced keys do NOT match --
# the check is exact-key equality, so ``macro_F_wrong`` is fine and ``F_wrong`` is not.
FORBIDDEN_BARE = frozenset({
    "F", "F_wrong", "F_split", "F_deadline", "F_disagree",
    "S_allcorrect", "failure", "reliability", "D", "delay",
    "P_correct",
})

# recognized non-outcome suffixes on a canonical key (interval / upper-bound / delta variants)
_CANONICAL_SUFFIXES = ("_ci", "_ucb", "_lcb", "_delta", "_mean", "_std")


def namespace_of(key: str) -> str | None:
    """Return the namespace group for a fully-qualified key, or ``None`` if it has no known prefix."""
    for prefix, group in _PREFIX_GROUP:
        if key.startswith(prefix):
            return group
    return None


# concept-based legacy markers -- caught even when not spelled with the surrogate_ prefix (a figure
# script may be handed hand-written / legacy JSON; constraint #13 is concept-, not prefix-, scoped).
_LEGACY_SUBSTRINGS = ("nodeunion", "node_union", "allcorrect", "global_product", "globalproduct")


def is_legacy_key(key: str) -> bool:
    """True for surrogate/node-union/product legacy metrics (the only metrics gated behind legacy=True).

    Matches the explicit ``surrogate_`` namespace, the canonical surrogate vocabulary, AND concept-based
    spellings (node-union / all-correct / global-product) so a renamed legacy metric cannot slip past
    the figure guard."""
    low = key.lower()
    return (key.startswith("surrogate_") or key in SURROGATE_KEYS
            or any(s in low for s in _LEGACY_SUBSTRINGS))


def is_forbidden_key(key: str) -> bool:
    """True iff ``key`` is an ambiguous bare token banned from any serialized result (exact match)."""
    return key in FORBIDDEN_BARE


def canonical_base(key: str) -> str:
    """Strip a single recognized variant suffix (``_ci`` / ``_ucb`` / ``_delta`` ...) for vocabulary
    checks, so ``macro_F_wrong_ci`` validates against the canonical ``macro_F_wrong``."""
    for suf in _CANONICAL_SUFFIXES:
        if key.endswith(suf):
            return key[: -len(suf)]
    return key


def vocabulary_for(group: str) -> frozenset[str]:
    return {
        MACRO: MACRO_KEYS, STRICT_AUDIT: STRICT_KEYS, DIAGNOSTIC: DIAGNOSTIC_KEYS,
        SAMPLING: SAMPLING_KEYS, CDQ: CDQ_KEYS, LEGACY: SURROGATE_KEYS,
    }.get(group, frozenset())


def iter_keys(obj) -> Iterator[str]:
    """Recursively yield every mapping key in a nested structure (for ban-list scans).

    Traverses any ``Mapping`` (not just ``dict``) and any non-string ``Iterable`` (list/tuple/set), so a
    forbidden token cannot hide inside a Mapping subclass or a set value."""
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            yield k
            yield from iter_keys(v)
    elif isinstance(obj, Iterable) and not isinstance(obj, (str, bytes)):
        for v in obj:
            yield from iter_keys(v)
