#!/usr/bin/env python3

import sys
import os
import traceback
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QStatusBar,
    QMenuBar, QAction, QMessageBox, QSplitter,
)
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QFont

def _global_exception_hook(exctype, value, tb):
    try:
        err = "".join(traceback.format_exception(exctype, value, tb))
        print(f"[FATAL] Unhandled exception:\n{err}")
        try:
            QMessageBox.critical(None, "Unhandled Error", err[:2000])
        except Exception:
            pass
    finally:
        return

sys.excepthook = _global_exception_hook
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gui_trainer.ui.main_window import MainTrainingWindow
from gui_trainer.ui.visualization_window import VisualizationWindow
from gui_trainer.ui.log_window import LogMonitorWindow
from gui_trainer.core.training_backend import TrainingBackend
from gui_trainer.utils.config_utils import ConfigManager
from gui_trainer.utils.visualization_utils import VisualizationManager
from gui_trainer.utils.logger import get_logger, setup_gui_logging, set_log_level, LogLevel


class CIMFTrainerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CIMF Estrus Stage Classifier v1.0")
        self.setGeometry(100, 100, 1400, 900)
        self.logger = get_logger()
        set_log_level(LogLevel.INFO)
        self.training_backend = TrainingBackend()
        self.config_manager = ConfigManager()
        self.viz_manager = VisualizationManager("./visualizations")
        self.main_training_window = None
        self.visualization_window = None
        self.log_window = None
        self.current_config = None
        self.training_metrics = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "val_auc": [],
            "val_sensitivity": [],
            "val_specificity": [],
            "loss_components": {},
        }
        self.init_ui()
        self.setup_connections()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        self.main_training_window = MainTrainingWindow()
        splitter.addWidget(self.main_training_window)
        self.create_right_panel(splitter)
        splitter.setSizes([600, 800])
        self.create_menu_bar()
        self.create_status_bar()

    def create_right_panel(self, parent):
        tab_widget = QTabWidget()
        parent.addWidget(tab_widget)
        self.visualization_window = VisualizationWindow()
        tab_widget.addTab(self.visualization_window, "Visualization")
        self.log_window = LogMonitorWindow()
        tab_widget.addTab(self.log_window, "Training Log")
        setup_gui_logging(self.on_log_message)

    def create_menu_bar(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        new_action = QAction("New Config", self)
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self.new_config)
        file_menu.addAction(new_action)
        open_action = QAction("Open Config", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.main_training_window.load_config)
        file_menu.addAction(open_action)
        save_action = QAction("Save Config", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.main_training_window.save_config)
        file_menu.addAction(save_action)
        file_menu.addSeparator()
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        train_menu = menubar.addMenu("Training")
        toggle_action = QAction("Start/Stop Training", self)
        toggle_action.setShortcut("F5")
        toggle_action.triggered.connect(self.main_training_window.toggle_training)
        train_menu.addAction(toggle_action)
        help_menu = menubar.addMenu("Help")
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def create_status_bar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_label = QLabel("Ready")
        self.status_bar.addWidget(self.status_label)
        self.gpu_label = QLabel()
        self.update_gpu_info()
        self.status_bar.addPermanentWidget(self.gpu_label)

    def setup_connections(self):
        self.main_training_window.training_started.connect(self.on_training_started)
        self.main_training_window.training_stopped.connect(self.on_training_stopped)

    def update_gpu_info(self):
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                self.gpu_label.setText(f"GPU: {gpu_name} ({gpu_memory:.1f}GB)")
            else:
                self.gpu_label.setText("GPU: Not detected")
        except Exception as e:
            self.gpu_label.setText(f"GPU: Error ({str(e)})")

    @pyqtSlot(dict)
    def on_training_started(self, config):
        self.current_config = config
        self.training_metrics = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "val_auc": [],
            "val_sensitivity": [],
            "val_specificity": [],
            "loss_components": {},
        }
        callbacks = {
            "epoch_finished": self.on_epoch_finished,
            "training_finished": self.on_training_finished,
            "error_occurred": self.on_training_error,
            "progress_updated": self.on_progress_updated,
            "log_message": self.on_log_message,
        }
        try:
            self.training_backend.start_training(config, callbacks)
            self.status_label.setText("Training...")
            self.visualization_window.start_training_progress(config["epochs"])
            self.log_window.add_log("Training started", "INFO")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start training: {str(e)}")
            self.main_training_window.on_training_finished()

    @pyqtSlot()
    def on_training_stopped(self):
        try:
            if hasattr(self.training_backend, "stop_training"):
                self.training_backend.stop_training()
            self.status_label.setText("Training stopped")
            self.visualization_window.on_training_finished()
            self.log_window.add_log("Training stopped by user", "WARNING")
        except Exception as e:
            self.log_window.add_log(f"Error stopping training: {str(e)}", "ERROR")

    @pyqtSlot(int, dict)
    def on_epoch_finished(self, epoch, metrics):
        self.training_metrics["train_loss"].append(metrics["train_loss"])
        self.training_metrics["val_loss"].append(metrics["val_loss"])
        self.training_metrics["train_acc"].append(metrics["train_acc"])
        self.training_metrics["val_acc"].append(metrics["val_acc"])
        for key in ("val_auc", "val_sensitivity", "val_specificity"):
            if key not in self.training_metrics:
                self.training_metrics[key] = []
            self.training_metrics[key].append(metrics.get(key))
        for key, value in metrics.get("loss_components", {}).items():
            if key not in self.training_metrics["loss_components"]:
                self.training_metrics["loss_components"][key] = []
            self.training_metrics["loss_components"][key].append(value)
        try:
            self.visualization_window.update_training_curves(self.training_metrics)
        except Exception as e:
            self.log_window.add_log(f"Update curves failed: {e}", "ERROR")
        try:
            progress_percent = int((epoch / self.current_config["epochs"]) * 100)
            self.visualization_window.update_training_progress(
                progress_percent, epoch, self.current_config["epochs"],
                metrics.get("epoch_time", 0.0),
            )
        except Exception as e:
            self.log_window.add_log(f"Update progress failed: {e}", "ERROR")
        log_msg = f"Epoch {epoch} completed - Val accuracy: {metrics['val_acc']:.2f}%"
        if metrics.get("is_best"):
            log_msg += " (Best)"
        self.log_window.add_log(log_msg, "INFO")

    @pyqtSlot(dict)
    def on_training_finished(self, final_metrics):
        self.status_label.setText("Training completed")
        self.main_training_window.on_training_finished()
        self.visualization_window.on_training_finished()
        try:
            summary_path = self.viz_manager.create_training_summary(
                self.training_metrics, self.current_config,
            )
            self.visualization_window.show_training_summary(summary_path)
        except Exception as e:
            self.log_window.add_log(f"Failed to generate visualization: {str(e)}", "ERROR")
        msg = (
            f"Training completed!\nBest validation accuracy: "
            f"{final_metrics['best_val_acc']:.2f}% (Epoch {final_metrics['best_epoch']})"
        )
        QMessageBox.information(self, "Training Completed", msg)
        self.log_window.add_log(
            f"Training completed - Best accuracy: {final_metrics['best_val_acc']:.2f}%", "SUCCESS",
        )

    @pyqtSlot(str)
    def on_training_error(self, error_msg):
        self.status_label.setText("Training error")
        self.main_training_window.on_training_finished()
        self.visualization_window.on_training_finished()
        QMessageBox.critical(self, "Training Error", f"Error occurred during training:\n{error_msg}")
        self.log_window.add_log(f"Training error: {error_msg}", "ERROR")

    @pyqtSlot(int, int, str)
    def on_progress_updated(self, current, total, message):
        pass

    def on_log_message(self, message, level="INFO"):
        try:
            if self.log_window is not None:
                self.log_window.add_log(message, level)
            else:
                print(f"[{level}] {message}")
        except Exception as e:
            print(f"GUI log error: {e}")
            print(f"[{level}] {message}")

    def new_config(self):
        self.main_training_window.reset_config()

    def show_about(self):
        about_text = (
            "<h3>CIMF Estrus Stage Classifier v1.0</h3>"
            "<p>PyQt5-based CIMF model for dairy cow estrus stage classification</p>"
            "<p><b>Classes (3):</b> pre_estrus, estrus, post_estrus</p>"
            "<p><b>Modalities:</b> EEG spectrogram and audio spectrogram</p>"
            "<p><b>Modules:</b> MCAF (cross-modal alignment) and CIRE (causal invariant embedding)</p>"
        )
        QMessageBox.about(self, "About", about_text)

    def closeEvent(self, event):
        if self.training_backend.is_training_active():
            reply = QMessageBox.question(
                self, "Confirm Exit", "Training is in progress, exit anyway?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                self.training_backend.stop_training()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("CIMF GUI Trainer")
    app.setApplicationVersion("1.0")
    app.setFont(QFont("Segoe UI", 9))
    try:
        main_window = CIMFTrainerApp()
        main_window.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Application startup failed: {str(e)}")
        print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
