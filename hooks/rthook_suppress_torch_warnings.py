"""
Runtime hook to suppress PyTorch deprecation warnings.
This runs when the packaged executable starts.
"""
import warnings
import os

# Suppress all PyTorch deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, module='torch')

# Suppress the multiprocessing redirects warning on Windows
os.environ['PYINSTALLER_SUPPRESS_WARNINGS'] = '1'

