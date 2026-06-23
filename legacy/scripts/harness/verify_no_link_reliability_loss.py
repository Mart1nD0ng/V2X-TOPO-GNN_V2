from __future__ import annotations

from check_utils import fail_with_violations, find_regex_violations, iter_implementation_files, pass_check, todo_check

FORBIDDEN = (
    (r"\b(avg_)?link_reliability_loss\b", "Link reliability must not be a direct loss variable/function"),
    (r"\bavg_reliability_loss\b", "Average reliability must not be a direct loss variable/function"),
    (r"\b(sinr|bler|harq|coverage)_loss\b", "Physics diagnostics must not be direct loss variables/functions"),
    (r"\b(LinkReliability|AvgLinkReliability|AvgReliability|SINR|Sinr|BLER|Bler|HARQ|Harq|Coverage)Loss\b", "Physics diagnostics must not be direct loss classes"),
    (r"['\"](avg_)?link_reliability_loss['\"]", "Link reliability must not be a direct loss key"),
    (r"['\"](sinr|bler|harq|coverage)_loss['\"]", "Physics diagnostics must not be direct loss keys"),
)


def main() -> None:
    files = list(iter_implementation_files())
    if not files:
        todo_check("no link reliability loss", "loss implementation is not present yet")
        return
    violations = find_regex_violations(files, FORBIDDEN)
    if violations:
        fail_with_violations("no link reliability loss", violations)
    pass_check("no link reliability loss", f"scanned {len(files)} training files")


if __name__ == "__main__":
    main()
