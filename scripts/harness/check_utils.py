from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable, Sequence

ROOT = Path(os.environ.get("V2X_HARNESS_ROOT", Path(__file__).resolve().parents[2])).resolve()

CODE_SUFFIXES = {".py", ".yaml", ".yml", ".json", ".toml"}
IMPLEMENTATION_ROOTS = ("src", "core", "physics", "models", "training", "eval", "configs", "scripts")
IMPLEMENTATION_DIRS = IMPLEMENTATION_ROOTS
EXCLUDED_DIR_NAMES = {"docs", ".agents", "reports", ".git", ".venv", "node_modules", "__pycache__"}
EXCLUDED_RELATIVE_PREFIXES = {
    ("tests", "fixtures"),
    ("scripts", "harness"),
}


def project_path(path: str | Path) -> Path:
    return ROOT / path


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def existing_paths(paths: Iterable[str | Path]) -> list[Path]:
    result: list[Path] = []
    for raw in paths:
        path = project_path(raw)
        if path.exists():
            result.append(path)
    return result


def relative_parts(path: Path) -> tuple[str, ...]:
    try:
        return path.resolve().relative_to(ROOT).parts
    except ValueError:
        return path.parts


def is_excluded(path: Path) -> bool:
    parts = relative_parts(path)
    if any(part in EXCLUDED_DIR_NAMES for part in parts):
        return True
    for prefix in EXCLUDED_RELATIVE_PREFIXES:
        if parts[: len(prefix)] == prefix:
            return True
    return False


def iter_files(paths: Iterable[str | Path], suffixes: set[str] | None = None) -> Iterable[Path]:
    allowed = suffixes or CODE_SUFFIXES
    for path in existing_paths(paths):
        if is_excluded(path):
            continue
        if path.is_file() and path.suffix in allowed:
            yield path
        elif path.is_dir():
            for child in path.rglob("*"):
                if child.is_file() and child.suffix in allowed and not is_excluded(child):
                    yield child


def iter_implementation_files(suffixes: set[str] | None = None) -> Iterable[Path]:
    yield from iter_files(IMPLEMENTATION_ROOTS, suffixes=suffixes)


def _iter_matching_implementation_files(path_terms: set[str]) -> Iterable[Path]:
    for path in iter_implementation_files():
        parts = tuple(part.lower() for part in relative_parts(path))
        joined = "/".join(parts)
        if any(term in joined for term in path_terms):
            yield path


def iter_consensus_files() -> Iterable[Path]:
    yield from _iter_matching_implementation_files({"consensus", "avalanche", "snowball", "evaluator"})


def iter_loss_files() -> Iterable[Path]:
    yield from _iter_matching_implementation_files({"loss", "objective", "training"})


def iter_consensus_or_loss_files() -> Iterable[Path]:
    seen: set[Path] = set()
    for path in list(iter_consensus_files()) + list(iter_loss_files()):
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        yield path


def iter_consensus_evaluator_loss_files() -> Iterable[Path]:
    yield from iter_consensus_or_loss_files()


def iter_sparse_complexity_files() -> Iterable[Path]:
    yield from _iter_matching_implementation_files({"candidate", "graph", "topology", "env", "scale", "baseline", "feasibility"})


def find_regex_violations(
    files: Iterable[Path],
    patterns: Sequence[tuple[str, str]],
    flags: int = re.IGNORECASE,
) -> list[tuple[Path, int, str, str]]:
    violations: list[tuple[Path, int, str, str]] = []
    for path in files:
        text = read_text(path)
        for pattern, message in patterns:
            for match in re.finditer(pattern, text, flags):
                line = text.count("\n", 0, match.start()) + 1
                violations.append((path, line, message, match.group(0)))
    return violations


def fail_with_violations(title: str, violations: Sequence[tuple[Path, int, str, str]]) -> None:
    print(f"{title}: FAILED")
    for path, line, message, snippet in violations:
        rel = path.relative_to(ROOT)
        print(f"- {rel}:{line}: {message} [{snippet!r}]")
    raise SystemExit(1)


def pass_check(title: str, details: str | None = None) -> None:
    if details:
        print(f"{title}: OK - {details}")
    else:
        print(f"{title}: OK")


def todo_check(title: str, details: str) -> None:
    print(f"{title}: TODO - {details}")
