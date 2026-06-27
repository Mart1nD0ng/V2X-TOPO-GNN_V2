"""Pytest bootstrap: put the repo root on sys.path so ``import src.*`` resolves.

Also pins single-threaded BLAS/OpenMP BEFORE numpy/torch initialise their thread pools.
On this Windows box torch's bundled OpenMP and numpy's MKL LAPACK abort (``Fatal Python
error: Aborted``) when both spin up multi-threaded — it strikes ``numpy.linalg.eigvalsh``
inside the Gauss-Legendre quadrature used by the finite-blocklength link model. Forcing one
thread is a stability config only; it changes no numerical result or gate threshold.
"""

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
