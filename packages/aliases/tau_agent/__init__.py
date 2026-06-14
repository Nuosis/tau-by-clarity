"""tau_agent — Tau-branded import alias for `pi_agent`.

The canonical package is `pi_agent` (PI lineage). This thin alias lets you write
`import tau_agent` (and any submodule). It is the SAME module object — no copy,
no drift. See the project README "Credits & lineage".
"""
import importlib as _importlib
import sys as _sys

_sys.modules[__name__] = _importlib.import_module("pi_agent")
