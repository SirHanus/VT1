from __future__ import annotations
import sys
from pathlib import Path
# Ensure package path when run as a script (no package context)
if __package__ is None or __package__ == "":
    src_dir = Path(__file__).resolve().parents[2]  # .../src
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

from PyQt6 import QtWidgets
from vt1.gui.pipeline_tab import PipelineTab
from vt1.gui.clustering_tab import ClusteringTab
from vt1.gui.help_tab import HelpTab  # Help tab import

class App(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SAM2 + YOLO Pose - Offline Pipeline GUI")
        self.resize(1150, 780)
        tabs = QtWidgets.QTabWidget(self)
        lay = QtWidgets.QVBoxLayout(self); lay.addWidget(tabs)
        tabs.addTab(PipelineTab(self), "Pipeline")
        tabs.addTab(ClusteringTab(self), "Team Clustering")
        tabs.addTab(HelpTab(self), "Help")  # new Help tab

    def closeEvent(self, event):  # ensure any running QProcess in tabs can terminate
        # Could extend to emit a signal to tabs to stop processes if needed.
        super().closeEvent(event)

def main():
    app = QtWidgets.QApplication(sys.argv)
    w = App(); w.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
