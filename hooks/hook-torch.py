"""
Custom PyInstaller hook for torch to suppress MKL DLL collection errors
and exclude unnecessary distributed training modules.
This overrides the default hook-torch from pyinstaller-hooks-contrib.
"""
from PyInstaller.utils.hooks import collect_submodules, collect_dynamic_libs
import warnings

# Suppress deprecation warnings during hook execution
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Collect torch submodules, excluding distributed training modules we don't need
# These cause deprecation warnings and aren't used for inference
excludes = [
    'torch.distributed._sharding_spec',
    'torch.distributed._sharded_tensor',
    'torch.distributed._shard.checkpoint',
    'torch.distributed.elastic.multiprocessing.redirects',
]

hiddenimports = []
for module in collect_submodules('torch'):
    # Skip excluded modules
    if not any(module.startswith(excl) for excl in excludes):
        hiddenimports.append(module)

# Collect torch DLLs (this works fine without conda_support)
binaries = collect_dynamic_libs('torch')

# Skip MKL DLL collection to avoid conda_support import error
# The necessary DLLs will be picked up by collect_dynamic_libs anyway

