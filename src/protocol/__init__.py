"""Consensus protocol layer (engineering plan §2).

Modules:

* ``binary_snowball`` -- the **true binary Snowball** finite-horizon confidence
  process (spec §3.2, ``docs/PROTOCOL_SEMANTICS.md``). Preference follows
  accumulated confidence ``d = d⁺ − d⁻`` and is sticky against single opposite
  quorums; this is the protocol-semantics correction over the legacy
  ``src/mainline/snowball.py`` Snowflake streak automaton (kept only as a named
  baseline).

The exact small-``N`` joint reference and the independent dynamic MC are added in
Phase 2 (``exact_small_n``, validated under G6).
"""

from .binary_snowball import (
    SnowballLayout,
    apply_round,
    initial_distribution,
    readout_preference,
    reachable_states,
    simulate_trajectory,
    snowball_layout,
    terminal_outcomes,
    transition_matrix,
)

__all__ = [
    "SnowballLayout",
    "apply_round",
    "initial_distribution",
    "readout_preference",
    "reachable_states",
    "simulate_trajectory",
    "snowball_layout",
    "terminal_outcomes",
    "transition_matrix",
]
