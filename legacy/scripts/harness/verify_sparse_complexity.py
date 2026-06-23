from __future__ import annotations

import re

from check_utils import fail_with_violations, iter_sparse_complexity_files, pass_check, todo_check

FORBIDDEN = (
    (r"torch\.cdist\s*\(", "Do not build dense all-pairs distance matrices for candidate generation"),
    (r"(itertools\.)?combinations\s*\(", "Do not enumerate all node pairs for candidate generation"),
    (r"pos\s*\[\s*:\s*,\s*None\s*\]\s*-\s*pos\s*\[\s*None\s*,\s*:\s*\]", "Do not build dense all-pairs broadcast differences"),
    (r"torch\.(zeros|ones|empty|full)\s*\(\s*(\(\s*)?n\s*,\s*n\b", "Do not allocate dense NxN tensors"),
    (r"np\.(zeros|ones|empty|full)\s*\(\s*\(\s*n\s*,\s*n\s*\)", "Do not allocate dense NxN arrays"),
    (r"for\s+\w+\s+in\s+range\(\s*n\s*\)[\s\S]{0,400}for\s+\w+\s+in\s+range\(\s*n\s*\)", "Nested all-node loops indicate O(N^2) candidate generation"),
)


def main() -> None:
    files = list(iter_sparse_complexity_files())
    if not files:
        todo_check("sparse complexity", "candidate graph implementation is not present yet")
        return

    violations = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        for pattern, message in FORBIDDEN:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                line = text.count("\n", 0, match.start()) + 1
                violations.append((path, line, message, match.group(0)))

    if violations:
        fail_with_violations("sparse complexity", violations)
    pass_check("sparse complexity", f"scanned {len(files)} candidate/eval files")


if __name__ == "__main__":
    main()
