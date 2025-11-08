from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from PyQt6 import QtWidgets


class HelpTab(QtWidgets.QWidget):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        lay = QtWidgets.QVBoxLayout(self)
        self.viewer = QtWidgets.QTextBrowser()
        self.viewer.setOpenExternalLinks(True)
        self.viewer.setReadOnly(True)
        self.viewer.setMinimumHeight(300)
        lay.addWidget(self.viewer, 1)
        self._load_readme_into_viewer()

    def _find_docs(self) -> tuple[Optional[Path], Optional[Path]]:
        # Try to find docs in multiple locations:
        # 1. Bundled with PyInstaller (_MEIPASS)
        # 2. Development repo root (pyproject.toml marker)
        # 3. Relative to executable

        # Check if running as PyInstaller bundle
        if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
            # Running in PyInstaller bundle
            bundle_dir = Path(sys._MEIPASS)
            gui_md = bundle_dir / "GUI.md"
            readme = bundle_dir / "README.md"
            return (
                gui_md if gui_md.exists() else None,
                readme if readme.exists() else None,
            )

        # Development mode: find repo root by looking for pyproject.toml
        current = Path(__file__).resolve()
        root = None
        for parent in [current] + list(current.parents):
            if (parent / "pyproject.toml").exists():
                root = parent
                break

        if root is None:
            # Fallback: assume parents[2] (works in some cases)
            root = Path(__file__).resolve().parents[2]

        gui_md = root / "GUI.md"
        readme = root / "README.md"
        return (
            gui_md if gui_md.exists() else None,
            readme if readme.exists() else None,
        )

    def _load_readme_into_viewer(self):
        gui_md, readme = self._find_docs()
        target = gui_md or readme
        if not target:
            self.viewer.setPlainText(
                "GUI.md/README.md not found. Keep docs at repo root."
            )
            return

        # Set search paths so relative links work
        if target:
            self.viewer.setSearchPaths([str(target.parent)])

        try:
            text = target.read_text(encoding="utf-8")
        except Exception:
            try:
                text = target.read_text(errors="ignore")
            except Exception:
                self.viewer.setPlainText(f"Failed to read {target}")
                return
        # If showing GUI.md and README also exists, append backlink.
        if gui_md and target == gui_md and readme:
            text += "\n\n---\nBack to project overview: see README.md"
        try:
            self.viewer.setMarkdown(text)
            return
        except Exception:
            pass
        html = None
        try:
            import markdown  # type: ignore

            html = markdown.markdown(text, extensions=["fenced_code", "tables"])
        except Exception:
            html = None
        if html:
            # Basic style for readability
            styled = f"""
            <html><head><style>
            body {{ font-family: Segoe UI, Arial, sans-serif; font-size: 11pt; }}
            pre, code {{ font-family: Consolas, monospace; }}
            pre {{ background:#f7f7f7; padding:8px; border-radius:4px; overflow-x:auto; }}
            code {{ background:#f2f2f2; padding:1px 3px; border-radius:3px; }}
            h1, h2, h3 {{ margin-top: 0.8em; }}
            a {{ color:#2a64ad; text-decoration:none; }} a:hover {{ text-decoration:underline; }}
            </style></head><body>{html}</body></html>
            """
            self.viewer.setHtml(styled)
        else:
            self.viewer.setPlainText(text)
