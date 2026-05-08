"""pytest config — adds repo root to sys.path so `from brain import ...`
and `from tasks.X import ...` work from any test file."""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
