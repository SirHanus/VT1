# Shared Qt binding selector for the GUI tabs (PyQt6-only)
# Note: This module does NOT run the app. Run gui_offline.py to start the GUI.
import sys

try:
    from PyQt6 import QtCore, QtGui, QtWidgets  # type: ignore
except Exception as e:  # pragma: no cover
    print("PyQt6 is required. Install with: pip install PyQt6", file=sys.stderr)
    raise SystemExit(1)
