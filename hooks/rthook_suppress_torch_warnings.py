"""
Runtime hook to suppress PyTorch deprecation warnings and fix inspect issues.
This runs when the packaged executable starts.
"""

import os
import sys
import warnings

# Suppress all PyTorch deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="torch")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# Suppress the multiprocessing redirects warning on Windows
os.environ["PYINSTALLER_SUPPRESS_WARNINGS"] = "1"

# Disable torch compilation features that require source inspection
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

# Fix inspect.getsource() issues in frozen executables
# PyTorch tries to inspect source code which doesn't exist in frozen apps
if getattr(sys, "frozen", False):
    import inspect

    # Store original functions
    _orig_getsource = inspect.getsource
    _orig_getsourcelines = inspect.getsourcelines
    _orig_findsource = inspect.findsource

    def _getsource_wrapper(obj):
        """Return empty string instead of raising OSError"""
        try:
            return _orig_getsource(obj)
        except (OSError, TypeError):
            return ""

    def _getsourcelines_wrapper(obj):
        """Return empty list instead of raising OSError"""
        try:
            return _orig_getsourcelines(obj)
        except (OSError, TypeError):
            return ([], 0)

    def _findsource_wrapper(obj):
        """Return empty result instead of raising OSError"""
        try:
            return _orig_findsource(obj)
        except (OSError, TypeError):
            return ([], 0)

    # Monkey-patch inspect module
    inspect.getsource = _getsource_wrapper
    inspect.getsourcelines = _getsourcelines_wrapper
    inspect.findsource = _findsource_wrapper
