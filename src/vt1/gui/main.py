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
from vt1.gui.help_tab import HelpTab
from vt1.gui.startup_dialog import show_startup_dialog, run_training_workflow
from vt1.config import settings


class App(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VT1 - Ice Hockey Pipeline GUI")
        self.resize(1150, 780)
        tabs = QtWidgets.QTabWidget(self)
        lay = QtWidgets.QVBoxLayout(self);
        lay.addWidget(tabs)
        tabs.addTab(PipelineTab(self), "Pipeline")
        tabs.addTab(ClusteringTab(self), "Team Clustering")
        tabs.addTab(HelpTab(self), "Help")

    def closeEvent(self, event):
        # Could extend to emit a signal to tabs to stop processes if needed.
        super().closeEvent(event)


def check_first_run() -> bool:
    """Check if this is the first run (no team models exist)."""
    cfg = settings()
    umap_pkl = cfg.team_models_dir / "umap.pkl"
    kmeans_pkl = cfg.team_models_dir / "kmeans.pkl"
    return not (umap_pkl.exists() and kmeans_pkl.exists())


def main():
    app = QtWidgets.QApplication(sys.argv)

    # Show startup dialog if first run
    if check_first_run():
        choice = show_startup_dialog()
        if choice == "exit":
            return 0
        elif choice == "run":
            # Run training workflow
            success = run_training_workflow()
            if success:
                QtWidgets.QMessageBox.information(
                    None,
                    "Training Complete",
                    "Team clustering models have been created successfully!\n\n"
                    "You can now use the Pipeline tab with team coloring enabled."
                )
            else:
                reply = QtWidgets.QMessageBox.warning(
                    None,
                    "Training Failed",
                    "The training workflow encountered errors.\n\n"
                    "You can try again later from the Team Clustering tab.\n\n"
                    "Continue to main GUI?",
                    QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No
                )
                if reply == QtWidgets.QMessageBox.StandardButton.No:
                    return 1
        # else: choice == "skip", continue to main GUI

    w = App()
    w.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
