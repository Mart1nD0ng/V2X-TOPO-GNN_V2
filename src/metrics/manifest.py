"""Result manifest & hash enforcement for the Guarded-CDQ2 round (GUARDED_CDQ2_ENGINEERING_PLAN.md §3).

A namespace-clean result is necessary but not sufficient: a HEADLINE result must also be reproducible
and hash-bound. This module builds the manifest (the ``hashes`` block of a §7.4 record) from an
``ExperimentSpec`` + policy/checkpoint hashes + tracked model seeds + a provenance id, and enforces:

- every required hash is present and non-empty (physics / service-profile / evidence / scene / protocol /
  policy / checkpoint / experiment-config);
- a provenance id exists (git commit OR manifest id);
- model seeds are tracked, unique, and (for a headline) meet a minimum count -- no untracked seed,
  no single-seed headline (forbidden shortcut: "Use single model seed for headline");
- train/eval physics mismatch fails fast unless the differing axis is a registered OOD axis, and the
  ideal/full-link distinction always fails (constraint #9) -- delegated to ``check_train_eval_compatible``;
- the recorded hashes actually match the eval spec (a result cannot claim a spec it did not run under).
"""

from __future__ import annotations

import subprocess
from typing import Any, Iterable, Mapping

from src.config.experiment_spec import ExperimentSpec, check_train_eval_compatible
from src.metrics.schema import MetricSchemaError, validate_result

# the hash fields a reproducible headline result must carry (plan §3 / spec §7.4)
REQUIRED_HASH_KEYS = (
    "physics_hash", "service_profile_hash", "evidence_hash", "scene_distribution_hash",
    "protocol_hash", "policy_hash", "checkpoint_hash", "experiment_config_hash",
)
# the eval-spec fields whose recorded hash must match the spec the result claims to have run under
_SPEC_BOUND_FIELDS = ("physics_hash", "service_profile_hash", "evidence_hash",
                      "scene_distribution_hash", "protocol_hash")


def current_git_commit() -> str | None:
    """Best-effort short git commit of the working tree (None if unavailable / not a repo)."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:  # pragma: no cover - environment dependent
        return None


def build_manifest(spec: ExperimentSpec, *, policy_hash: str, checkpoint_hash: str,
                   model_seeds: Iterable[int], git_commit: str | None = None,
                   manifest_id: str | None = None) -> dict[str, Any]:
    """Assemble the manifest (``hashes`` block) for a result record.

    ``checkpoint_hash`` is the trained-model fingerprint (use a sentinel like ``"uniform-esp"`` for a
    non-learned policy). ``model_seeds`` are the training seeds that produced the headline ensemble.
    """
    return {
        "physics_hash": spec.physics_hash,
        "service_profile_hash": spec.service_profile_hash,
        "evidence_hash": spec.evidence_hash,
        "scene_distribution_hash": spec.scene_distribution_hash,
        "protocol_hash": spec.protocol_hash,
        "policy_hash": str(policy_hash),
        "checkpoint_hash": str(checkpoint_hash),
        "experiment_config_hash": spec.config_hash(),
        "query_law": spec.query_law,
        "full_physics": bool(spec.full_physics),
        "git_commit": git_commit,
        "manifest_id": manifest_id,
        "model_seeds": list(model_seeds),
    }


def _hashes(record: Mapping[str, Any]) -> Mapping[str, Any]:
    h = record.get("hashes")
    if not isinstance(h, Mapping):
        raise MetricSchemaError("result record has no 'hashes' manifest block")
    return h


def validate_manifest(record: Mapping[str, Any], *, require_seeds: bool = True,
                      min_seeds: int = 1, headline: bool = True) -> None:
    """Validate a result record's manifest (and, via ``validate_result``, its namespaces).

    Raises :class:`MetricSchemaError` on any missing/empty required hash, a missing provenance id, an
    untracked/duplicate model seed, fewer than ``min_seeds`` seeds, or an invalid macro block.
    """
    validate_result(record, headline=headline)          # version + namespaces + macro outcomes
    h = _hashes(record)

    missing = [k for k in REQUIRED_HASH_KEYS if not str(h.get(k, "")).strip()]
    if missing:
        raise MetricSchemaError(f"result manifest missing/empty required hash(es): {missing}")

    if not (str(h.get("git_commit") or "").strip() or str(h.get("manifest_id") or "").strip()):
        raise MetricSchemaError("result manifest needs a provenance id (git_commit or manifest_id)")

    if not record.get("query_family"):
        raise MetricSchemaError("result record missing query_family")

    if require_seeds:
        seeds = h.get("model_seeds")
        if not isinstance(seeds, (list, tuple)) or len(seeds) == 0:
            raise MetricSchemaError("result manifest has no tracked model_seeds (untracked seed)")
        if len(set(seeds)) != len(seeds):
            raise MetricSchemaError(f"duplicate model seeds in manifest: {seeds}")
        if len(seeds) < min_seeds:
            raise MetricSchemaError(
                f"headline needs >= {min_seeds} model seeds (no single-seed headline), got {len(seeds)}")


def assert_train_eval_consistent(record: Mapping[str, Any], train: ExperimentSpec,
                                 evaluation: ExperimentSpec) -> None:
    """Fail-fast train/eval consistency (plan §3 task 2).

    1. ``check_train_eval_compatible`` -- physics/protocol/profile/evidence/scene/query must match
       unless a registered OOD axis permits it; ideal/full-link always blocks (constraint #9).
    2. the result's recorded hashes must match the ``evaluation`` spec it claims to have run under
       (so a result cannot be relabelled with a spec it did not actually use).
    """
    check_train_eval_compatible(train, evaluation)
    h = _hashes(record)
    for field in _SPEC_BOUND_FIELDS:
        recorded, expected = h.get(field), getattr(evaluation, field)
        if recorded != expected:
            raise MetricSchemaError(
                f"recorded {field}={recorded!r} does not match the eval spec ({expected!r}); "
                f"the result was not produced under the spec it claims")


__all__ = [
    "REQUIRED_HASH_KEYS", "current_git_commit", "build_manifest",
    "validate_manifest", "assert_train_eval_consistent",
]
