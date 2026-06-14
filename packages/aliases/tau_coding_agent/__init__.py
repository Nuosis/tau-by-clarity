"""tau_coding_agent — Tau-branded import alias for `pi_coding_agent`.

The canonical package is `pi_coding_agent` (PI lineage). This thin alias lets you write
`import tau_coding_agent` (and any submodule). It is the SAME module object — no copy,
no drift. See the project README "Credits & lineage".
"""
import importlib as _importlib
import sys as _sys

_sys.modules[__name__] = _importlib.import_module("pi_coding_agent")
