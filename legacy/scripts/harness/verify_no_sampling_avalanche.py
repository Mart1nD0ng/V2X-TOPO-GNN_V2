from __future__ import annotations

from check_utils import fail_with_violations, find_regex_violations, iter_consensus_or_loss_files, pass_check, todo_check

FORBIDDEN = (
    (r"\brandom\.sample\b", "Avalanche evaluator must not use random sampling"),
    (r"\b(np|numpy)\.random\b", "Avalanche evaluator must be deterministic"),
    (r"\btorch\.multinomial\b", "Avalanche evaluator must not sample validators"),
    (r"\btorch\.bernoulli\b", "Avalanche evaluator must not sample successes"),
    (r"\btorch\.distributions\.Binomial\([^)]*\)\.sample\b", "Do not sample binomial outcomes"),
    (r"\bmonte[\s_-]*carlo\w*\b", "Do not use Monte Carlo reliability in evaluator or loss"),
)


def main() -> None:
    files = list(iter_consensus_or_loss_files())
    if not files:
        todo_check("no sampling Avalanche", "Avalanche evaluator is not implemented yet")
        return
    violations = find_regex_violations(files, FORBIDDEN)
    if violations:
        fail_with_violations("no sampling Avalanche", violations)
    pass_check("no sampling Avalanche", f"scanned {len(files)} evaluator/loss files")


if __name__ == "__main__":
    main()
