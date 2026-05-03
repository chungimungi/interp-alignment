"""Compatibility alias for the single-layer crosscoder package.

The maintained implementations live in sibling variant packages:
`interp_utils.crosscoder-singlelayer` and `interp_utils.crosscoder-multilayer`.
This package keeps legacy `interp_utils.crosscoder.*` module paths resolving to
single-layer modules.
"""
from pathlib import Path

_SINGLELAYER_DIR = Path(__file__).resolve().parent.parent / "crosscoder-singlelayer"
__path__ = [str(_SINGLELAYER_DIR)]
