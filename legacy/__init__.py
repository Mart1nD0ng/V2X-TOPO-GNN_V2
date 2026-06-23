"""Import guard for the quarantined legacy tree (G10 / decision D11).

Importing ``legacy`` (or any ``legacy.*`` submodule) on the live mainline path is a
hard error: the legacy derivations (mean-field F, beta-tail quorum, logistic-BLER ell,
degree caps) are *archived historical-reproduction material* and must never be a live
dependency of ``src/mainline`` (H5).  This guard makes a dynamic resurrection
(``importlib.import_module("legacy.src.consensus")``) fail loudly instead of silently
re-activating a forbidden closure.

To reproduce a legacy result you must opt in explicitly and NOT through this package:
add ``legacy/`` to ``sys.path`` and import the bare ``src.<pkg>`` name (see
``legacy/ARCHIVED.md``); that path does not import the ``legacy`` package and so is not
blocked.
"""

raise ImportError(
    "legacy/ is archived historical-reproduction material (G10/D11) and is NOT importable "
    "on the live mainline path. The single live derivation is src/mainline/. "
    "See legacy/ARCHIVED.md to opt into a legacy reproduction explicitly."
)
