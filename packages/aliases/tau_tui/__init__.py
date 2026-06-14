"""tau_tui — Tau-branded import alias for `pi_tui`.

The canonical package is `pi_tui` (PI lineage). This thin alias lets you write
`import tau_tui` (and any submodule). It is the SAME module object — no copy,
no drift. See the project README "Credits & lineage".
"""
import importlib as _importlib
import sys as _sys

_sys.modules[__name__] = _importlib.import_module("pi_tui")
