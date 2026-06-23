"""G10 (spec H5 / §11.11): single mathematical mainline -- DISCRIMINATIVE.

Acceptance: the repository has ONE live mathematical derivation -- ``src/mainline/`` -- and no
parallel / conflicting legacy derivation coexists on the live path.  The forbidden legacy
closures (mean-field per-node-marginal ``F``, iid beta-tail quorum, logistic-BLER ``ell``
proxy, fixed degree caps / top-k truncation) are QUARANTINED into ``legacy/`` (decision D11),
frozen historical-reproduction material imported by nothing live.

Per anti-degradation §7 the gate must be DISCRIMINATIVE: a reintroduced forbidden closure or
legacy dependency -- in ANY form -- must FAIL.  Hardened (after the G10 adversarial review) so
the following evasions are caught:
  * PARAPHRASED closures: the regexes match SEMANTIC shapes, not just canonical tokens -- a
    logistic reliability via ``expit`` / ``1/(1+exp(.))`` / ``sigmoid(.sinr.)``, a degree cap
    via ``sorted(.)[:MAX_NEIGHBOURS]`` / ``prune_to_fixed_fanout`` / ``[:fanout]``, a renamed
    beta-tail (``betabinomial`` / regularized incomplete beta).
  * a BEHAVIOURAL no-degree-cap check (build a dense radius graph; assert out-degree == the
    true neighbour count) catches a token-free cap regardless of how it is written (H2).
  * RECURSIVE scan over the whole live derivation surface incl. ``tests/`` and subdirectories
    (a closure in a nested helper or a test file no longer hides); legitimate self-test /
    detector lines are exempted by a ``# G10-allow`` sentinel, not by excluding a directory.
  * DYNAMIC legacy imports: ``importlib.import_module("legacy...")`` / ``__import__`` with a
    legacy string arg are flagged (AST), and ``legacy/__init__.py`` raises ImportError so any
    runtime resurrection (even an obfuscated symbol) fails loudly.
  * STRONG structural conditions: ``legacy/src`` must contain the 7 expected packages by name,
    the FBL ell producer must be importable & callable, and ``src.<legacy>`` must be unresolvable.

NOTE (paper, §0/§8): ``paper/main.tex`` still narrates the *superseded* derivation (betainc
beta-tail quorum + node-mean ``F``).  The spec FORBIDS editing paper headline/claim text until
ALL gates are green; reconciling the paper math to ``src/mainline`` is therefore a tracked
post-gate §8 proposal item (see REFACTOR_PROGRESS conflicts), NOT gated here.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import re
import sys

from _common import ROOT, GateResult, main_single, run_pytest  # type: ignore

EXPECTED_LEGACY = {"consensus", "evaluation", "losses", "models", "topology", "training", "v2x_env"}
# derivation surface scanned for forbidden CLOSURES (recursive, incl tests/); gates excluded --
# they are the detector layer that legitimately DEFINES these patterns.  Self-test / test-name
# occurrences are exempted line-by-line with the ``# G10-allow`` sentinel, not by directory.
DERIV_ROOTS = ("src/mainline", "scripts/analysis", "tests")
DERIV_EXTRA = ("conftest.py",)
# whole live path for the legacy-import scans
LIVE_ROOTS = ("src/mainline", "scripts/gates", "scripts/analysis", "tests")
LIVE_EXTRA = ("conftest.py",)
SENTINEL = "G10-allow"

FORBIDDEN = {
    "beta-tail quorum": r"betainc\s*\(|betabinomial|_RegularizedBetaInc|regularized_incomplete_beta",
    "mean-field node-marginal F": r"compute_topology_query_support|p_correct_query|topology_query_support",
    "logistic-BLER ell proxy": (
        r"sigmoid\s*\([^)]*(sinr|snr)|\bexpit\s*\(|special\.expit|1\.?0?\s*/\s*\(\s*1\.?0?\s*\+\s*\w*\.?exp\s*\("),
    "hard degree cap / top-k": (
        r"\.topk\s*\(|\btop_k\b|\btopk\b|degree_cap|max_degree\b|max_?neighbou?rs?\b|fixed_fanout|"
        r"prune_to|\[\s*:\s*(MAX_\w+|cap\w*|fanout|n_?neigh\w*)\s*\]"),
}
DYN_IMPORT_FUNCS = {"import_module", "__import__", "find_spec", "load_module"}


def _py_files(roots, extra):
    out = []
    for r in roots:
        out += [p for p in (ROOT / r).rglob("*.py") if "__pycache__" not in p.parts]
    for e in extra:
        if (ROOT / e).exists():
            out.append(ROOT / e)
    return sorted(set(out))


def scan_forbidden() -> dict:
    """``{closure_name: [path:line]}`` of forbidden closures in the live derivation surface.

    Recursive; skips lines bearing the ``# G10-allow`` sentinel (the self-test samples and a
    test name that legitimately mention the patterns).
    """
    files = _py_files(DERIV_ROOTS, DERIV_EXTRA)
    out: dict = {}
    for name, pat in FORBIDDEN.items():
        rx = re.compile(pat, re.I)
        hits = []
        for p in files:
            for i, line in enumerate(p.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                if SENTINEL in line:
                    continue
                if rx.search(line):
                    hits.append(f"{p.relative_to(ROOT).as_posix()}:{i}")
        out[name] = hits
    return out


def _call_name(node) -> str:
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ""


def legacy_imports() -> list:
    """Static (AST) legacy imports across the whole live path -- ``(path, module)``."""
    hits = []
    for p in _py_files(LIVE_ROOTS, LIVE_EXTRA):
        try:
            tree = ast.parse(p.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:
            continue
        for n in ast.walk(tree):
            mods = ([a.name for a in n.names] if isinstance(n, ast.Import)
                    else ([n.module] if isinstance(n, ast.ImportFrom) and n.module else []))
            for m in mods:
                parts = m.split(".")
                if (parts and parts[0] == "legacy") or (len(parts) >= 2 and parts[0] == "src" and parts[1] in EXPECTED_LEGACY):
                    hits.append((p.relative_to(ROOT).as_posix(), m))
    return hits


def dynamic_legacy_imports() -> list:
    """AST calls to import_module/__import__/find_spec with a legacy string arg.

    Skips lines bearing the ``# G10-allow`` sentinel (the gate/test calls that import ``legacy``
    only to VERIFY it is blocked).
    """
    hits = []
    for p in _py_files(LIVE_ROOTS, LIVE_EXTRA):
        text = p.read_text(encoding="utf-8", errors="ignore")
        src_lines = text.splitlines()
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for n in ast.walk(tree):
            if isinstance(n, ast.Call) and _call_name(n.func) in DYN_IMPORT_FUNCS and n.args:
                a0 = n.args[0]
                if isinstance(a0, ast.Constant) and isinstance(a0.value, str):
                    ln = getattr(n, "lineno", 0)
                    if 1 <= ln <= len(src_lines) and SENTINEL in src_lines[ln - 1]:
                        continue
                    parts = a0.value.split(".")
                    if (parts and parts[0] == "legacy") or (len(parts) >= 2 and parts[0] == "src" and parts[1] in EXPECTED_LEGACY):
                        hits.append((p.relative_to(ROOT).as_posix(), a0.value))
    return hits


def behavioural_no_cap() -> tuple:
    """Build a DENSE radius graph (all nodes mutual neighbours) and return the realised max
    out-degree vs the true neighbour count.  A token-free degree cap truncates this; no cap
    keeps the full N-1 (H2), independent of how a cap would have been written."""
    import torch
    from src.mainline.topology import build_candidate_graph
    N = 24
    gen = torch.Generator().manual_seed(0)
    pos = torch.rand(N, 2, generator=gen, dtype=torch.float64) * 5.0  # all within radius
    g = build_candidate_graph(pos, 50.0)
    max_out = int(torch.bincount(g.src_index, minlength=N).max())
    return max_out, N - 1


def _spec_unresolvable(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is None
    except (ImportError, ModuleNotFoundError, ValueError):
        return True


def _legacy_pkg_blocked() -> bool:
    try:
        importlib.import_module("legacy")  # G10-allow (verifies the block; not a resurrection)
        return False
    except ImportError:
        return True
    except Exception:
        return False


def run() -> GateResult:
    evidence: dict = {}

    forb = scan_forbidden()
    forb_total = sum(len(v) for v in forb.values())
    leg = legacy_imports()
    dyn = dynamic_legacy_imports()

    src_subdirs = sorted(p.name for p in (ROOT / "src").iterdir() if p.is_dir() and p.name != "__pycache__")
    legacy_src = set(p.name for p in (ROOT / "legacy" / "src").iterdir() if p.is_dir()) if (ROOT / "legacy" / "src").exists() else set()
    legacy_names_ok = EXPECTED_LEGACY <= legacy_src
    archived_md = (ROOT / "legacy" / "ARCHIVED.md").exists()
    legacy_unimportable = all(_spec_unresolvable(f"src.{p}") for p in EXPECTED_LEGACY)
    legacy_blocked = _legacy_pkg_blocked()

    fbl_mod = importlib.import_module("src.mainline.finite_blocklength")
    fbl_ok = callable(getattr(fbl_mod, "channel_dispersion", None)) and callable(getattr(fbl_mod, "fbl_error", None))

    cap_max, cap_expected = behavioural_no_cap()
    cap_ok = cap_max == cap_expected

    evidence["forbidden closures in live derivation (recursive, incl tests/)"] = (
        f"{forb_total}  (" + ", ".join(f"{k.split()[0]}:{len(v)}" for k, v in forb.items()) + ")")
    if forb_total:
        evidence["  forbidden sites"] = "; ".join(s for v in forb.values() for s in v[:4])
    evidence["legacy imports on live path (static / dynamic)"] = f"{len(leg)} / {len(dyn)}"
    if leg or dyn:
        evidence["  legacy import sites"] = "; ".join(f"{p}->{m}" for p, m in (leg + dyn)[:5])
    evidence["src/ live packages"] = f"{src_subdirs}"
    evidence["legacy/src has 7 expected packages"] = f"{legacy_names_ok} ({sorted(legacy_src)})"
    evidence["legacy unimportable (src.*) / legacy/ pkg blocked"] = f"{legacy_unimportable} / {legacy_blocked}"
    evidence["ARCHIVED.md present"] = f"{archived_md}"
    evidence["single FBL ell producer (import+callable)"] = f"{fbl_ok}"
    evidence["behavioural no-degree-cap (dense out-deg == N-1)"] = f"{cap_max} == {cap_expected} -> {cap_ok}"

    single_mainline = (
        forb_total == 0 and len(leg) == 0 and len(dyn) == 0 and src_subdirs == ["mainline"]
        and legacy_names_ok and archived_md and legacy_unimportable and legacy_blocked
        and fbl_ok and cap_ok
    )

    tests_ok, tail = run_pytest("tests/test_g10_single_mainline.py")
    evidence["pytest"] = "passed" if tests_ok else f"FAILED\n{tail}"

    passed = bool(single_mainline and tests_ok)
    return GateResult(
        gate="G10",
        title="single mathematical mainline (H5): one live derivation, legacy quarantined",
        passed=passed,
        evidence=evidence,
        notes="src/mainline is the sole live derivation; 7 legacy packages + 37 scripts quarantined under "
              "legacy/ (import-blocked, scanned recursively incl tests/). Discriminative against paraphrased "
              "closures (semantic regexes + behavioural no-cap check), dynamic legacy imports (AST + runtime "
              "ImportError guard). Paper-math reconciliation is a tracked post-gate §8 item (paper edits "
              "forbidden until all gates green).",
    )


if __name__ == "__main__":
    sys.exit(main_single(run))
