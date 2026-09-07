"""Single source of truth for the package version.

This module exists so that both the runtime library
(:data:`driftflow.__version__`) and the build system (which reads
``version = {attr = "driftflow.__version__"}`` in ``pyproject.toml``)
derive the version from exactly one place.  Bump the version here and
nowhere else.
"""

__version__ = "1.0.0"
