from __future__ import annotations
from typing import Optional
from pathlib import Path
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

    def _find_readme(self) -> Optional[Path]:
        # Walk up from this file to repo root, look for README.md
        p = Path(__file__).resolve()
        for parent in [p.parent, *p.parents]:
            cand = parent.parent.parent / 'README.md' if parent.name == 'gui' else parent / 'README.md'
            # The above ensures we also check repo root if we're at src/vt1/gui
            if cand.exists():
                return cand
        return None

    def _load_readme_into_viewer(self):
        readme = self._find_readme()
        if not readme:
            self.viewer.setPlainText("README.md not found. Please keep README.md at the repository root.")
            return
        try:
            text = readme.read_text(encoding='utf-8')
        except Exception:
            try:
                text = readme.read_text(errors='ignore')
            except Exception:
                self.viewer.setPlainText(f"Failed to read {readme}")
                return
        # Preferred: render with Qt's built-in Markdown support
        try:
            self.viewer.setMarkdown(text)
            return
        except Exception:
            pass
        # Fallback: render via python-markdown if available
        html: Optional[str] = None
        try:
            import markdown  # type: ignore
            html = markdown.markdown(text, extensions=['fenced_code', 'tables'])
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
            </style></head><body>{html}</body></html>
            """
            self.viewer.setHtml(styled)
        else:
            self.viewer.setPlainText(text)
