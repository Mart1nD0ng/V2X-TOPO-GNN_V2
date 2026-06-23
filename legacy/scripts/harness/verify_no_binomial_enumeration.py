from __future__ import annotations

from check_utils import fail_with_violations, find_regex_violations, iter_consensus_or_loss_files, pass_check, todo_check

FORBIDDEN = (
    (r"for\s+\w+\s+in\s+range\(\s*alpha\s*,\s*k\s*(\+\s*1)?\s*\)", "Use incomplete beta, not binomial tail enumeration"),
    (r"sum\s*\([\s\S]{0,160}range\(\s*alpha\s*,\s*k\s*(\+\s*1)?\s*\)", "Use incomplete beta, not binomial tail enumeration"),
    (r"\bmath\.comb\b", "Do not enumerate binomial coefficients in evaluator"),
    (r"from\s+math\s+import\s+[^#\n]*\bcomb\b", "Do not import direct comb for binomial enumeration"),
    (r"(?<!\.)\bcomb\s*\(", "Do not enumerate binomial coefficients in evaluator"),
    (r"\b(scipy\.stats\.)?binom\.(pmf|cdf)\b", "Do not use binomial PMF/CDF enumeration in training evaluator"),
    (r"\b(math\.)?factorial\s*\(", "Do not implement binomial probabilities through factorial formulas"),
)


def main() -> None:
    files = list(iter_consensus_or_loss_files())
    if not files:
        todo_check("no binomial enumeration", "Avalanche evaluator is not implemented yet")
        return
    violations = find_regex_violations(files, FORBIDDEN)
    if violations:
        fail_with_violations("no binomial enumeration", violations)
    pass_check("no binomial enumeration", f"scanned {len(files)} evaluator/loss files")


if __name__ == "__main__":
    main()
