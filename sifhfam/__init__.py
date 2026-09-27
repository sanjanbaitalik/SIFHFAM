"""Neurocomputing major-revision evidence package.

Modes
-----
legacy_repro         : call old behavior without modifying it.
revision_canonical   : scientifically corrected/transparent implementation.

This package never modifies files under ``Outputs/``, ``fs_experiments/`` or
any other frozen legacy artifact.  All new evidence is written under
``SIFHFAM_EVIDENCE/``.
"""

__version__ = "1.0.0"

MODES = ("legacy_repro", "revision_canonical")
