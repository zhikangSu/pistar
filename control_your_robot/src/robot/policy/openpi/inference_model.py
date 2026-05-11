"""
Compatibility bridge for imports like `robot.policy.openpi.inference_model`.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.append(_SRC)

_CLIENT_SRC = os.path.join(_ROOT, "packages", "openpi-client", "src")
if _CLIENT_SRC not in sys.path:
    sys.path.append(_CLIENT_SRC)

from openpi.inference_model import PI0_DUAL, PI0_LIBERO, PI0_SINGLE, _normalize_checkpoint_dir

__all__ = ["PI0_DUAL", "PI0_LIBERO", "PI0_SINGLE", "_normalize_checkpoint_dir"]
