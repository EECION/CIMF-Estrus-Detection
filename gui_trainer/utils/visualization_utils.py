import os
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .estrus_stages import EstrusStageRegistry


class ClassificationVisualizer:
    STAGE_COLORS = ["#FF6B6B", "#6BAEFF", "#6BFFB5"]

    def __init__(self, output_dir: str = "./visualizations"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_training_curves(self, metrics: Dict[str, List], save_name: str = "training_curves.png") -> str:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("CIMF Estrus Stage Classification Training", fontsize=14)
        epochs = range(1, len(metrics.get("train_loss", [])) + 1)
        if metrics.get("train_loss"):
            axes[0, 0].plot(epochs, metrics["train_loss"], label="Train Loss", color="#2196F3")
        if metrics.get("val_loss"):
            axes[0, 0].plot(epochs, metrics["val_loss"], label="Val Loss", color="#FF9800")
        axes[0, 0].set_title("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        if metrics.get("train_acc"):
            axes[0, 1].plot(epochs, metrics["train_acc"], label="Train Acc", color="#4CAF50")
        if metrics.get("val_acc"):
            axes[0, 1].plot(epochs, metrics["val_acc"], label="Val Acc", color="#9C27B0")
        axes[0, 1].set_title("Accuracy (%)")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        if metrics.get("val_auc"):
            axes[1, 0].plot(epochs, metrics["val_auc"], label="Val AUC", color="#E91E63")
        axes[1, 0].set_title("Validation AUC")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        if metrics.get("val_sensitivity"):
            axes[1, 1].plot(epochs, metrics["val_sensitivity"], label="Sensitivity", color="#00BCD4")
        if metrics.get("val_specificity"):
            axes[1, 1].plot(epochs, metrics["val_specificity"], label="Specificity", color="#795548")
        axes[1, 1].set_title("Sensitivity / Specificity (%)")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=120)
        plt.close()
        return save_path

    def plot_confusion_matrix(self, confusion: np.ndarray, save_name: str = "confusion_matrix.png") -> str:
        stage_names = EstrusStageRegistry.get_class_names()
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(confusion, cmap="Blues")
        ax.set_xticks(range(len(stage_names)))
        ax.set_yticks(range(len(stage_names)))
        ax.set_xticklabels(stage_names, rotation=45, ha="right")
        ax.set_yticklabels(stage_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("CIMF Confusion Matrix")
        for i in range(confusion.shape[0]):
            for j in range(confusion.shape[1]):
                ax.text(j, i, int(confusion[i, j]), ha="center", va="center", color="black")
        fig.colorbar(im, ax=ax)
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=120)
        plt.close()
        return save_path


class VisualizationManager:
    def __init__(self, output_dir: str = "./visualizations"):
        self.output_dir = output_dir
        self.visualizer = ClassificationVisualizer(output_dir)
        os.makedirs(output_dir, exist_ok=True)

    def create_training_summary(
        self,
        metrics: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
    ) -> str:
        save_path = self.visualizer.plot_training_curves(metrics, "cimf_training_summary.png")
        return save_path

    def save_confusion_matrix(self, confusion: np.ndarray, name: str = "confusion.png") -> str:
        return self.visualizer.plot_confusion_matrix(confusion, name)
