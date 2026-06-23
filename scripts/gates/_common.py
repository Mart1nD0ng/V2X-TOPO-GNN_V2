"""Shared scaffolding for the acceptance-gate checks (G1-G11).

Each gate module exposes ``run() -> GateResult``.  ``run_all_gates.py`` collects
them into a single table and a JSON evidence file.  Gates must produce *numeric
evidence* from reproducible code, never hard-coded verdicts (anti-cheat rule §7).
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Windows consoles default to GBK and choke on the status emoji; force UTF-8.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass


@dataclass
class GateResult:
    gate: str
    title: str
    passed: bool
    evidence: dict = field(default_factory=dict)
    notes: str = ""

    @property
    def status_icon(self) -> str:
        return "🟢" if self.passed else "🔴"


def run_pytest(target: str) -> tuple[bool, str]:
    """Run a pytest target as a subprocess; return (passed, tail-of-output)."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", target, "-q"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    tail = "\n".join(out.strip().splitlines()[-12:])
    return proc.returncode == 0, tail


def grep_repo(pattern: str, *, globs: tuple[str, ...] = ("*.py",), exclude_dirs: tuple[str, ...] = ()) -> list[str]:
    """Return ``path:line: text`` hits for a regex across the repo source.

    Uses Python's ``re`` so it works without ripgrep.  ``exclude_dirs`` matches any
    path component (e.g. ``archive``, ``.venv``).
    """
    import re

    rx = re.compile(pattern)
    default_excludes = {".git", ".venv", "__pycache__", "node_modules", ".prism"}
    excludes = default_excludes | set(exclude_dirs)
    hits: list[str] = []
    for g in globs:
        for path in ROOT.rglob(g):
            if any(part in excludes for part in path.parts):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for i, line in enumerate(text.splitlines(), 1):
                if rx.search(line):
                    rel = path.relative_to(ROOT).as_posix()
                    hits.append(f"{rel}:{i}: {line.strip()}")
    return hits


def print_result(result: GateResult) -> None:
    print(f"\n{'=' * 70}")
    print(f"{result.status_icon} {result.gate}: {result.title}")
    print(f"{'=' * 70}")
    for key, value in result.evidence.items():
        print(f"  {key}: {value}")
    if result.notes:
        print(f"  notes: {result.notes}")
    print(f"  => {'PASS' if result.passed else 'FAIL'}")


def main_single(run: Callable[[], GateResult]) -> int:
    result = run()
    print_result(result)
    return 0 if result.passed else 1
