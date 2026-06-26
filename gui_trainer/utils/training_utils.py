import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


class MetricsCalculator:
    def __init__(self, num_classes: int = 3):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.all_logits = []
        self.all_targets = []

    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        eeg_features: torch.Tensor = None,
        audio_features: torch.Tensor = None,
    ):
        if predictions is not None:
            self.all_logits.append(predictions.detach().cpu())
        if targets is not None:
            self.all_targets.append(targets.detach().cpu())

    def get_all_metrics(self) -> dict:
        if not self.all_logits or not self.all_targets:
            return {
                "accuracy": 0.0,
                "auc": 0.0,
                "sensitivity": 0.0,
                "specificity": 0.0,
            }
        logits = torch.cat(self.all_logits, dim=0)
        targets = torch.cat(self.all_targets, dim=0)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        targets_np = targets.numpy()
        preds_np = preds.numpy()
        accuracy = float((preds_np == targets_np).mean() * 100.0)
        auc = self._compute_auc(probs.numpy(), targets_np)
        sensitivity = self._compute_sensitivity(targets_np, preds_np)
        specificity = self._compute_specificity(targets_np, preds_np)
        return {
            "accuracy": accuracy,
            "auc": auc,
            "sensitivity": sensitivity,
            "specificity": specificity,
        }

    def _compute_auc(self, probs: np.ndarray, targets: np.ndarray) -> float:
        try:
            if len(np.unique(targets)) < 2:
                return 0.0
            return float(roc_auc_score(targets, probs, multi_class="ovr", average="macro"))
        except Exception:
            return 0.0

    def _compute_sensitivity(self, targets: np.ndarray, preds: np.ndarray) -> float:
        recalls = []
        for cls in range(self.num_classes):
            tp = np.sum((targets == cls) & (preds == cls))
            fn = np.sum((targets == cls) & (preds != cls))
            if tp + fn > 0:
                recalls.append(tp / (tp + fn))
        if not recalls:
            return 0.0
        return float(np.mean(recalls) * 100.0)

    def _compute_specificity(self, targets: np.ndarray, preds: np.ndarray) -> float:
        specs = []
        for cls in range(self.num_classes):
            tn = np.sum((targets != cls) & (preds != cls))
            fp = np.sum((targets != cls) & (preds == cls))
            if tn + fp > 0:
                specs.append(tn / (tn + fp))
        if not specs:
            return 0.0
        return float(np.mean(specs) * 100.0)
