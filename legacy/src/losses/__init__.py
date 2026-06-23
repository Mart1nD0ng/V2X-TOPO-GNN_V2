"""Coupled C/D/E objective utilities."""

from .coupled_objective import CoupledLossConfig, compute_coupled_loss
from .gradnorm import GradNormBalancer, compute_task_gradient_norms, compute_task_gradient_vectors
from .pcgrad import merge_pcgrad, pcgrad_project

__all__ = [
    "CoupledLossConfig",
    "GradNormBalancer",
    "compute_coupled_loss",
    "compute_task_gradient_norms",
    "compute_task_gradient_vectors",
    "merge_pcgrad",
    "pcgrad_project",
]
