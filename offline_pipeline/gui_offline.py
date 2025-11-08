# python
# Simple PyQt GUI to run offline_pipeline/sam_offline.py with selectable options and show progress
from __future__ import annotations
import os
import sys
import re
from pathlib import Path
from typing import Optional

from gui_qt import QtWidgets
from gui_pipeline_tab import PipelineTab
from gui_clustering_tab import ClusteringTab


class SamOfflineGUI(QtWidgets.QWidget):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("SAM2 + YOLO Pose - Offline Pipeline GUI")
        self.resize(1150, 780)
        tabs = QtWidgets.QTabWidget(self)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(tabs)
        # Pipeline tab
        self.pipeline_tab = PipelineTab(self)
        tabs.addTab(self.pipeline_tab, "Pipeline")
        # Clustering tab
        self.cluster_tab = ClusteringTab(self)
        tabs.addTab(self.cluster_tab, "Team Clustering")


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = SamOfflineGUI()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
