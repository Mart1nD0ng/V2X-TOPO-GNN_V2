"""G-P0-PHYSICS P0-A: confirmation-time delay = τ_C · Δ_poll (survival-sum, no off-by-one).

Spec §5.3 / §6. The legacy analytic delay summed ``1 - S[1:]`` (an off-by-one that reported
a one-round finish as 0). The macrostate confirmation time is ``T_confirm = τ_C · Δ_poll``
built from the validated first-hitting epoch, so a first-epoch finish returns ONE poll
window, and ``E[τ] = Σ_r P(τ>r)`` holds exactly.
"""

import torch

from src.config.service_profile import ConsensusServiceProfile
from src.metrics.first_hitting import (
    basin_first_hitting_batched,
    confirm_time_stats,
    first_hitting_outcome,
)

PROFILE = ConsensusServiceProfile.urban_default().replace(poll_window_ms=10.0)  # Δ_poll = 0.01 s


def test_first_epoch_finish_is_one_window_not_zero():
    # β=1-style: correct mass crosses ρ_f at epoch 1 ⇒ τ=1 ⇒ T_confirm = 1·Δ_poll, never 0.
    C = torch.tensor([0.0, 0.7, 0.8], dtype=torch.float64)
    W = torch.tensor([0.0, 0.1, 0.1], dtype=torch.float64)
    fh = first_hitting_outcome(C, W, PROFILE)
    assert fh.outcome == "correct" and fh.tau == 1
    code = torch.tensor([0]); tau = torch.tensor([1])
    stats = confirm_time_stats(code, tau, PROFILE)
    assert abs(stats["mean_confirm_s"] - PROFILE.poll_window_s) < 1e-12
    assert stats["mean_confirm_s"] > 0.0


def test_survival_sum_identity_holds():
    # a batch of correct runs with assorted first-hit epochs
    code = torch.zeros(5, dtype=torch.int64)            # all correct
    tau = torch.tensor([1, 2, 2, 4, 5], dtype=torch.int64)
    stats = confirm_time_stats(code, tau, PROFILE)
    # E[τ] == Σ_r P(τ > r)  (the correct survival sum; the legacy off-by-one breaks this)
    assert abs(stats["mean_tau"] - stats["survival_sum_tau"]) < 1e-12
    assert abs(stats["mean_confirm_s"] - stats["mean_tau"] * PROFILE.poll_window_s) < 1e-12


def test_cvar_is_upper_tail_of_confirm_time():
    code = torch.zeros(100, dtype=torch.int64)
    tau = torch.arange(1, 101, dtype=torch.int64)       # 1..100
    prof = PROFILE.replace(latency_quantile=0.95, max_poll_epochs=100)
    stats = confirm_time_stats(code, tau, prof)
    # CVaR_0.95 (mean of worst 5%) exceeds the mean
    assert stats["cvar_confirm_s"] > stats["mean_confirm_s"]


def test_no_correct_runs_returns_nan_not_zero():
    code = torch.tensor([1, 3, 1], dtype=torch.int64)   # wrong/deadline only
    tau = torch.tensor([2, 21, 3], dtype=torch.int64)
    stats = confirm_time_stats(code, tau, PROFILE)
    assert stats["num_correct"] == 0
    assert stats["mean_confirm_s"] != stats["mean_confirm_s"]  # nan (no correct runs to time)
