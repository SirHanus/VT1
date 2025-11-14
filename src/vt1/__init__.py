"""
VT1 - Ice Hockey Video Analysis Pipeline
"""

import os
import sys
from pathlib import Path

# Configure cache directories to be under vt1 folder when running as frozen executable
frozen = getattr(sys, "frozen", False)
if frozen:
    # Running as exe: use AppData/Local/vt1 as base for all cache/data
    # app_data = Path(os.environ.get("LOCALAPPDATA", os.path.expanduser("~/.local")))
    vt1_base = Path(".") / "vt1"

    # Set Hugging Face cache directories
    if "HF_HOME" not in os.environ:
        os.environ["HF_HOME"] = str(vt1_base / "cache" / "huggingface")

    if "TRANSFORMERS_CACHE" not in os.environ:
        os.environ["TRANSFORMERS_CACHE"] = str(
            vt1_base / "cache" / "huggingface" / "transformers"
        )

    if "HF_HUB_CACHE" not in os.environ:
        os.environ["HF_HUB_CACHE"] = str(vt1_base / "cache" / "huggingface" / "hub")

    # Set torch hub cache directory
    if "TORCH_HOME" not in os.environ:
        os.environ["TORCH_HOME"] = str(vt1_base / "cache" / "torch")
