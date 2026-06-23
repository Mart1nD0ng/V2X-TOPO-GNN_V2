"""G10 (spec H5): single mathematical mainline -- structural invariants + discriminativeness.

Checks:
  1. ``src/mainline`` is the sole live derivation package under ``src/`` (legacy quarantined,
     not importable as ``src.<pkg>``, and ``legacy/`` blocked at import time).
  2. The live path imports zero legacy modules (static AST + dynamic-import string args), and
     the live derivation (recursive, incl. ``tests/`` and subdirs) has zero forbidden closures.
  3. BEHAVIOURAL: a dense radius graph is not degree-capped (H2), regardless of how a cap
     would be written.
  4. DISCRIMINATIVENESS (anti-degradation §7): the detector flags PARAPHRASED reintroductions
     (expit / 1-over-1-plus-exp logistic, sorted-then-sliced cap, a renamed beta-tail), catches
     a forbidden closure planted in a NESTED subdir, and does not false-positive on benign code.
"""

from __future__ import annotations

import importlib
import importlib.util
import re
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "gates"))
from gate_g10 import (  # noqa: E402
    EXPECTED_LEGACY, FORBIDDEN, behavioural_no_cap, dynamic_legacy_imports,
    legacy_imports, scan_forbidden,
)


def test_src_has_only_mainline():
    subdirs = sorted(p.name for p in (ROOT / "src").iterdir() if p.is_dir() and p.name != "__pycache__")
    assert subdirs == ["mainline"], subdirs


def test_legacy_quarantined_blocked_and_not_importable():
    assert (ROOT / "legacy" / "ARCHIVED.md").exists()
    legacy_pkgs = set(p.name for p in (ROOT / "legacy" / "src").iterdir() if p.is_dir())
    assert EXPECTED_LEGACY <= legacy_pkgs, (EXPECTED_LEGACY - legacy_pkgs)
    for pkg in ("consensus", "evaluation", "topology"):
        assert not (ROOT / "src" / pkg).exists()
        assert importlib.util.find_spec(f"src.{pkg}") is None
    # the legacy/ package is import-blocked at runtime (dynamic resurrection fails loudly)
    try:
        importlib.import_module("legacy")  # G10-allow (verifies the block; not a resurrection)
        assert False, "legacy/ must raise ImportError"
    except ImportError:
        pass


def test_live_path_has_no_legacy_imports():
    assert legacy_imports() == [], legacy_imports()
    assert dynamic_legacy_imports() == [], dynamic_legacy_imports()


def test_live_derivation_has_no_forbidden_closures():
    forb = scan_forbidden()
    assert sum(len(v) for v in forb.values()) == 0, forb


def test_behavioural_no_degree_cap():  # G10-allow (name mentions the token; asserts no cap)
    # H2: a dense neighbourhood is fully connected; a token-free cap would truncate it
    max_out, expected = behavioural_no_cap()
    assert max_out == expected, (max_out, expected)
    # independent confirmation at a second size
    gen = torch.Generator().manual_seed(3)
    from src.mainline.topology import build_candidate_graph
    pos = torch.rand(18, 2, generator=gen, dtype=torch.float64) * 4.0
    g = build_candidate_graph(pos, 50.0)
    assert int(torch.bincount(g.src_index, minlength=18).max()) == 17


def test_detector_catches_paraphrases():
    # the regexes must flag NON-canonical reintroductions (not just the literal tokens)
    samples = {
        "beta-tail quorum": ["scipy.special.betainc(a, b, x)", "from m import betabinomial_upper_tail"],  # G10-allow
        "mean-field node-marginal F": ["F = compute_topology_query_support(g)", "p_correct_query(g)"],  # G10-allow
        "logistic-BLER ell proxy": ["torch.sigmoid((sinr - t) / w)", "scipy.special.expit((snr-t)/w)",  # G10-allow
                                     "1.0 / (1.0 + torch.exp(-(sinr - t)))"],  # G10-allow
        "hard degree cap / top-k": ["scores.topk(k).indices", "sorted(cand)[:MAX_NEIGHBOURS]",  # G10-allow
                                     "keep = cand[:fanout]", "def prune_to_fixed_fanout(x): ..."],  # G10-allow
    }
    for name, lines in samples.items():
        for line in lines:
            assert re.search(FORBIDDEN[name], line, re.I), (name, line)
    # benign code must NOT trip the detectors
    assert not re.search(FORBIDDEN["hard degree cap / top-k"], "max_deg = int(deg.max())", re.I)  # G10-allow
    assert not re.search(FORBIDDEN["logistic-BLER ell proxy"], "P = P_min + dP * torch.sigmoid(r)", re.I)  # G10-allow


def test_scan_is_discriminative_end_to_end(tmp_path=None):
    # plant a paraphrased forbidden closure in a NESTED subdir of a live root and confirm the
    # recursive scan flags it (then clean up) -- proves the gate is not blind to subdirs/tests.
    probe = ROOT / "scripts" / "analysis" / "_g10_disc_probe"
    shutil.rmtree(probe, ignore_errors=True)
    probe.mkdir(parents=True)
    try:
        lines = [
            "import torch",
            "def ell(sinr, t, w):",
            "    return torch.special.expit((sinr - t) / w)",  # G10-allow
            "def pick(scores, cap):",
            "    return sorted(scores)[:cap]",  # G10-allow
        ]
        (probe / "m.py").write_text("\n".join(lines) + "\n", encoding="utf-8")
        forb = scan_forbidden()
        flat = [s for v in forb.values() for s in v]
        assert any("_g10_disc_probe" in s for s in flat), flat  # the nested closure IS detected
        assert len(forb["logistic-BLER ell proxy"]) >= 1
    finally:
        shutil.rmtree(probe, ignore_errors=True)
    # tree restored -> clean again
    assert sum(len(v) for v in scan_forbidden().values()) == 0


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
