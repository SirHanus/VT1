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
from vt1.gui.finetuning_tab import FinetuningTab
from vt1.gui.help_tab import HelpTab
from vt1.gui.startup_dialog import show_startup_dialog, run_training_workflow
from vt1.config import settings


class App(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VT1 - Ice Hockey Pipeline GUI")
        self.resize(1150, 780)
        tabs = QtWidgets.QTabWidget(self)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(tabs)
        tabs.addTab(PipelineTab(self), "Pipeline")
        tabs.addTab(ClusteringTab(self), "Team Clustering")
        tabs.addTab(FinetuningTab(self), "Fine-tuning")
        tabs.addTab(HelpTab(self), "Help")

    def closeEvent(self, event):
        # Could extend to emit a signal to tabs to stop processes if needed.
        super().closeEvent(event)


def check_first_run() -> bool:
    """Check if this is the first run (no team models exist AND no skip marker)."""
    cfg = settings()
    umap_pkl = cfg.team_models_dir / "umap.pkl"
    kmeans_pkl = cfg.team_models_dir / "kmeans.pkl"

    # Check for skip marker file (user chose to skip setup)
    skip_marker = cfg.repo_root / ".vt1_setup_skipped"

    # If models exist OR user already skipped, don't show dialog
    if (umap_pkl.exists() and kmeans_pkl.exists()) or skip_marker.exists():
        return False

    return True


def mark_setup_skipped():
    """Create a marker file indicating user skipped the setup dialog."""
    cfg = settings()
    skip_marker = cfg.repo_root / ".vt1_setup_skipped"
    try:
        skip_marker.touch()
    except Exception:
        pass


def main():
    app = QtWidgets.QApplication(sys.argv)

    # Show startup dialog if first run
    if check_first_run():
        choice = show_startup_dialog()
        if choice == "exit":
            # User closed the dialog - exit without marking as skipped
            return 0
        elif choice == "run":
            # Run training workflow
            mark_setup_skipped()
            success = run_training_workflow()
            if success:
                QtWidgets.QMessageBox.information(
                    None,
                    "Training Complete",
                    "Team clustering models have been created successfully!\n\n"
                    "You can now use the Pipeline tab with team coloring enabled.",
                )
            else:
                reply = QtWidgets.QMessageBox.warning(
                    None,
                    "Training Failed",
                    "The training workflow encountered errors.\n\n"
                    "You can try again later from the Team Clustering tab.\n\n"
                    "Continue to main GUI?",
                    QtWidgets.QMessageBox.StandardButton.Yes
                    | QtWidgets.QMessageBox.StandardButton.No,
                )
                if reply == QtWidgets.QMessageBox.StandardButton.No:
                    return 1
        elif choice == "skip":
            # User chose to skip - mark it so we don't ask again
            mark_setup_skipped()
        # Continue to main GUI (for 'run' or 'skip')

    w = App()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()

    # Headless dispatcher: allow running modules via the frozen GUI exe without opening a window
    # Usage: vt1-gui.exe --module-run <module.name> [args...]
    if len(sys.argv) > 1 and sys.argv[1] == "--module-run":
        import runpy

        if len(sys.argv) < 3:
            print("Usage: --module-run <module> [args...]", file=sys.stderr)
            sys.exit(2)
        module_name = sys.argv[2]
        module_args = sys.argv[3:]
        # Simulate `python -m module_name [args...]`
        sys.argv = [module_name] + module_args
        try:
            runpy.run_module(module_name, run_name="__main__", alter_sys=True)
            sys.exit(0)
        except SystemExit as e:
            # Propagate exit codes from the target module
            raise
        except Exception as e:
            print(f"[GUI] Headless module-run failed: {e}", file=sys.stderr)
            sys.exit(1)

    # Otherwise, launch the GUI normally
    main()
