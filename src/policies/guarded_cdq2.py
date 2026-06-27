"""Guarded-CDQ2: CDQ2 diversity gated by a reliability/deadline guard (spec §4).

ESP (eta=0) is the reliability-first DEFAULT (constraint #3). CDQ2 diversity (eta>0) is a
liveness/deadline extension that, in a too-tight-deadline or low-reliability-slack regime, can RAISE
the wrong/split basins (the G-ETA-RISK-LIVENESS finding: diversity selects more distant, weaker-link
peers). So eta>0 is enabled ONLY when there is reliability slack AND deadline pressure (constraint #4):

  hard guard:  eta_guarded = eta_raw   if  m_w >= delta_w  and  m_s >= delta_s ;  else 0
  soft guard:  eta_guarded = eta_raw * sigmoid((m_w-delta_w)/T_w) * sigmoid((m_s-delta_s)/T_s)
                                      * sigmoid((p_d-delta_d)/T_d)

with slack ``m_w = eps_w - Fw_UCB``, ``m_s = eps_s - Fs_UCB`` (eps the HARD wrong/split budgets,
constraint #5) and deadline pressure ``p_d`` (e.g. F_deadline_UCB). The guard inputs are calibrated
UCB estimates (an ESP pre-pass MC, or analytic); the headline judge is the independent dynamic MC.
The hard guard is the deployable rule; the soft guard is its differentiable relaxation for training.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "GuardConfig", "reliability_slack", "hard_guard_eta", "guard_factor",
    "soft_guard_eta", "GuardedCDQ2Policy",
]


@dataclass(frozen=True)
class GuardConfig:
    """Reliability/deadline guard parameters (spec §4.2-§4.4).

    ``eps_w``/``eps_s`` are the HARD wrong/split budgets (from the service profile). ``delta_*`` are
    the safety margins (enable diversity only with margin to spare). ``T_*`` are the soft-guard
    temperatures. ``delta_d`` is the deadline-pressure threshold (enable diversity only when liveness
    is actually under pressure -- no point risking validity for a deadline that is comfortably met).
    """

    eps_w: float = 1e-3
    eps_s: float = 1e-3
    delta_w: float = 2e-4
    delta_s: float = 2e-4
    delta_d: float = 0.05          # deadline-pressure threshold (enable only if p_d exceeds this)
    T_w: float = 1e-4
    T_s: float = 1e-4
    T_d: float = 0.05

    @classmethod
    def from_profile(cls, profile, *, delta_frac: float = 0.2, **over) -> "GuardConfig":
        """Build from a :class:`ConsensusServiceProfile`: ``eps`` = the profile budgets, ``delta`` =
        ``delta_frac`` of each budget (default 20% margin). Overridable per field."""
        kw = dict(eps_w=profile.max_wrong_basin_probability, eps_s=profile.max_split_basin_probability,
                  delta_w=delta_frac * profile.max_wrong_basin_probability,
                  delta_s=delta_frac * profile.max_split_basin_probability,
                  T_w=max(1e-9, delta_frac * profile.max_wrong_basin_probability),
                  T_s=max(1e-9, delta_frac * profile.max_split_basin_probability))
        kw.update(over)
        return cls(**kw)


def reliability_slack(Fw_ucb: float, Fs_ucb: float, eps_w: float, eps_s: float) -> tuple[float, float]:
    """Reliability slack ``(m_w, m_s) = (eps_w - Fw_UCB, eps_s - Fs_UCB)`` (spec §4.2). Positive slack
    means the wrong/split UCB sits below the hard budget with room to spare."""
    return eps_w - Fw_ucb, eps_s - Fs_ucb


def hard_guard_eta(eta_raw: float, *, Fw_ucb: float, Fs_ucb: float, cfg: GuardConfig) -> float:
    """Hard guard (spec §4.3): keep ``eta_raw`` only if BOTH wrong and split slacks clear their
    margins; otherwise fall back to ESP (``eta=0``). Reliability is never traded (constraint #4/#5)."""
    m_w, m_s = reliability_slack(Fw_ucb, Fs_ucb, cfg.eps_w, cfg.eps_s)
    return float(eta_raw) if (m_w >= cfg.delta_w and m_s >= cfg.delta_s) else 0.0


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def guard_factor(*, Fw_ucb: float, Fs_ucb: float, p_d: float, cfg: GuardConfig) -> float:
    """Soft guard factor ``G = sigma((m_w-delta_w)/T_w) sigma((m_s-delta_s)/T_s) sigma((p_d-delta_d)/T_d)``
    in [0, 1] (spec §4.4). G -> 1 when reliability slack AND deadline pressure are high; G -> 0 when
    either reliability slack is gone (validity at risk) or there is no deadline pressure to relieve."""
    m_w, m_s = reliability_slack(Fw_ucb, Fs_ucb, cfg.eps_w, cfg.eps_s)
    return (_sigmoid((m_w - cfg.delta_w) / cfg.T_w)
            * _sigmoid((m_s - cfg.delta_s) / cfg.T_s)
            * _sigmoid((p_d - cfg.delta_d) / cfg.T_d))


def soft_guard_eta(eta_raw: float, *, Fw_ucb: float, Fs_ucb: float, p_d: float,
                   cfg: GuardConfig) -> float:
    """Soft guard (spec §4.4): ``eta_guarded = eta_raw * G``."""
    return float(eta_raw) * guard_factor(Fw_ucb=Fw_ucb, Fs_ucb=Fs_ucb, p_d=p_d, cfg=cfg)


class GuardedCDQ2Policy:
    """A deployable policy that activates CDQ2 diversity only when the guard permits (spec §4).

    Construction decides ``eta_eff`` from the calibrated slack estimate ``(Fw_ucb, Fs_ucb, p_d)`` and
    the guard ``mode`` ("hard"/"soft"). If ``eta_eff`` is ~0 the policy IS ESP (delegates to the bare
    ``base`` quality policy, ``query_law="esp"``); otherwise it is a ``CDQ2Policy`` at ``eta_eff``
    (``query_law="cdq2"``). All policy-interface attributes proxy to the inner policy, so the canonical
    episode / dynamic MC consume it transparently. ``guard_active`` / ``eta_eff`` expose the decision.
    """

    name = "guarded_cdq2"

    def __init__(self, base, r: int, eta_raw: float, diversity, cfg: GuardConfig, *, mode: str,
                 Fw_ucb: float, Fs_ucb: float, p_d: float, eta_floor: float = 1e-9):
        if mode == "hard":
            self.eta_eff = hard_guard_eta(eta_raw, Fw_ucb=Fw_ucb, Fs_ucb=Fs_ucb, cfg=cfg)
        elif mode == "soft":
            self.eta_eff = soft_guard_eta(eta_raw, Fw_ucb=Fw_ucb, Fs_ucb=Fs_ucb, p_d=p_d, cfg=cfg)
        else:
            raise ValueError(f"mode must be 'hard' or 'soft', got {mode!r}")
        self.mode = mode
        self.eta_raw = float(eta_raw)
        self.guard_active = self.eta_eff > eta_floor
        self.slack_w, self.slack_s = reliability_slack(Fw_ucb, Fs_ucb, cfg.eps_w, cfg.eps_s)
        self.deadline_pressure = float(p_d)
        if self.guard_active:
            from src.sampling.cdq2_wiring import CDQ2Policy
            self._inner = CDQ2Policy(base, r=r, eta=self.eta_eff, diversity=diversity)
        else:
            self._inner = base          # literal fall-back to ESP (the default, constraint #3)

    @property
    def query_law(self) -> str:
        return getattr(self._inner, "query_law", "esp")

    def __getattr__(self, item):
        # proxy any policy-interface attribute (kernel/log_weights/eta/r/diversity) to the inner policy
        return getattr(self.__dict__["_inner"], item)
