from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar, QGroupBox, QTextEdit,
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class VisualizationWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.total_epochs = 0
        self.metrics_history = {
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

    def init_ui(self):
        layout = QVBoxLayout()
        progress_group = QGroupBox("Training Progress")
        progress_layout = QVBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_label = QLabel("Ready")
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.progress_label)
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        metrics_group = QGroupBox("Latest Metrics")
        metrics_layout = QHBoxLayout()
        self.auc_label = QLabel("Val AUC: --")
        self.sensitivity_label = QLabel("Val Sensitivity: --")
        self.specificity_label = QLabel("Val Specificity: --")
        self.acc_label = QLabel("Val Accuracy: --")
        for lbl in (self.auc_label, self.sensitivity_label, self.specificity_label, self.acc_label):
            lbl.setAlignment(Qt.AlignCenter)
            metrics_layout.addWidget(lbl)
        metrics_group.setLayout(metrics_layout)
        layout.addWidget(metrics_group)
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setMaximumHeight(80)
        layout.addWidget(self.summary_text)
        self.setLayout(layout)

    def start_training_progress(self, total_epochs: int):
        self.total_epochs = total_epochs
        self.metrics_history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "val_auc": [],
            "val_sensitivity": [],
            "val_specificity": [],
            "loss_components": {},
        }
        self.progress_bar.setValue(0)
        self.progress_label.setText(f"Training started (0/{total_epochs} epochs)")

    def update_training_progress(self, percent: int, current_epoch: int, total_epochs: int, epoch_time: float):
        self.progress_bar.setValue(min(100, max(0, percent)))
        self.progress_label.setText(
            f"Epoch {current_epoch}/{total_epochs} | {epoch_time:.1f}s/epoch"
        )

    def update_training_curves(self, metrics: dict):
        for key in ("train_loss", "val_loss", "train_acc", "val_acc", "val_auc", "val_sensitivity", "val_specificity"):
            if key in metrics and metrics[key]:
                self.metrics_history[key] = list(metrics[key])
        if "loss_components" in metrics:
            self.metrics_history["loss_components"] = metrics["loss_components"]
        self._redraw_plots()
        if self.metrics_history["val_auc"]:
            self.auc_label.setText(f"Val AUC: {self.metrics_history['val_auc'][-1]:.4f}")
        if self.metrics_history["val_sensitivity"]:
            self.sensitivity_label.setText(
                f"Val Sensitivity: {self.metrics_history['val_sensitivity'][-1]:.2f}%"
            )
        if self.metrics_history["val_specificity"]:
            self.specificity_label.setText(
                f"Val Specificity: {self.metrics_history['val_specificity'][-1]:.2f}%"
            )
        if self.metrics_history["val_acc"]:
            self.acc_label.setText(f"Val Accuracy: {self.metrics_history['val_acc'][-1]:.2f}%")

    def update_training_plots(self, train_loss, val_loss, train_acc, val_acc):
        self.metrics_history["train_loss"] = list(train_loss)
        self.metrics_history["val_loss"] = list(val_loss)
        self.metrics_history["train_acc"] = list(train_acc)
        self.metrics_history["val_acc"] = list(val_acc)
        self._redraw_plots()

    def _redraw_plots(self):
        self.figure.clear()
        epochs = range(1, len(self.metrics_history.get("train_loss", [])) + 1)
        ax1 = self.figure.add_subplot(2, 2, 1)
        if self.metrics_history["train_loss"]:
            ax1.plot(epochs, self.metrics_history["train_loss"], label="Train")
        if self.metrics_history["val_loss"]:
            ax1.plot(epochs, self.metrics_history["val_loss"], label="Val")
        ax1.set_title("Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax2 = self.figure.add_subplot(2, 2, 2)
        if self.metrics_history["train_acc"]:
            ax2.plot(epochs, self.metrics_history["train_acc"], label="Train")
        if self.metrics_history["val_acc"]:
            ax2.plot(epochs, self.metrics_history["val_acc"], label="Val")
        ax2.set_title("Accuracy (%)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax3 = self.figure.add_subplot(2, 2, 3)
        if self.metrics_history["val_auc"]:
            ax3.plot(epochs, self.metrics_history["val_auc"], label="Val AUC", color="#E91E63")
        ax3.set_title("Validation AUC")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax4 = self.figure.add_subplot(2, 2, 4)
        if self.metrics_history["val_sensitivity"]:
            ax4.plot(epochs, self.metrics_history["val_sensitivity"], label="Sensitivity", color="#00BCD4")
        if self.metrics_history["val_specificity"]:
            ax4.plot(epochs, self.metrics_history["val_specificity"], label="Specificity", color="#795548")
        ax4.set_title("Sensitivity / Specificity (%)")
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        self.figure.suptitle("CIMF Estrus Stage Classification", fontsize=12)
        self.figure.tight_layout()
        self.canvas.draw()

    def on_training_finished(self):
        self.progress_label.setText("Training finished")
        self.progress_bar.setValue(100)

    def show_training_summary(self, summary_path: str):
        self.summary_text.setText(f"Training summary saved: {summary_path}")
