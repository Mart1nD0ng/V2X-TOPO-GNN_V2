"""Evaluation harnesses (Phase 10+): fair, MC-judged policy comparisons."""

from .cdq2_factorial import (
    FactorialResult,
    esp_vs_cdq2_cell,
    observable_group_diversity,
    run_factorial_cell,
    wilson_ci,
)

__all__ = [
    "observable_group_diversity",
    "wilson_ci",
    "FactorialResult",
    "run_factorial_cell",
    "esp_vs_cdq2_cell",
]
