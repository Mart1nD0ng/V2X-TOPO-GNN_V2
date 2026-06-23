"""Mainline training helpers.

The ~100 diagnostic/milestone modules were moved to ``archive/diagnostics/src/training/``
(see ``docs/MAINLINE.md``). Mainline code imports the submodules directly, e.g.
``from src.training.training_smoke import ...`` / ``from src.training.gradient_governance
import ...``.
"""

from .training_smoke import (
    load_training_smoke_config,
    run_curriculum_training_smoke,
    run_tiny_training_smoke,
)

__all__ = [
    "load_training_smoke_config",
    "run_curriculum_training_smoke",
    "run_tiny_training_smoke",
]
