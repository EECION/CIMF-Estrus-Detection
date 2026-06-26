import os
from typing import Any, Dict, Optional

import torch


class CheckpointManager:
    def save_checkpoint(
        self,
        model,
        optimizer,
        scheduler,
        epoch: int,
        loss: float,
        metrics: Dict[str, Any],
        save_path: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> bool:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)) or ".", exist_ok=True)
            model_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            payload = {
                "epoch": epoch,
                "model_state_dict": model_state,
                "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "loss": loss,
                "metrics": metrics,
                "config": config or {},
            }
            torch.save(payload, save_path)
            return True
        except Exception:
            return False

    def load_checkpoint(
        self,
        checkpoint_path: str,
        model,
        optimizer=None,
        scheduler=None,
        map_location: str = "cpu",
    ) -> Optional[Dict[str, Any]]:
        if not os.path.isfile(checkpoint_path):
            return None
        try:
            checkpoint = torch.load(checkpoint_path, map_location=map_location)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            if hasattr(model, "module"):
                model.module.load_state_dict(state_dict, strict=False)
            else:
                model.load_state_dict(state_dict, strict=False)
            if optimizer and checkpoint.get("optimizer_state_dict"):
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if scheduler and checkpoint.get("scheduler_state_dict"):
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            return {
                "epoch": checkpoint.get("epoch", 0),
                "metrics": checkpoint.get("metrics", {}),
                "config": checkpoint.get("config", {}),
            }
        except Exception:
            return None

    def get_checkpoint_info(self, checkpoint_path: str) -> str:
        if not os.path.isfile(checkpoint_path):
            return "Checkpoint file not found"
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            lines = [
                f"File: {os.path.basename(checkpoint_path)}",
                f"Epoch: {checkpoint.get('epoch', 'N/A')}",
                f"Loss: {checkpoint.get('loss', 'N/A')}",
            ]
            metrics = checkpoint.get("metrics", {})
            if metrics:
                lines.append(f"Val Accuracy: {metrics.get('val_acc', 'N/A')}")
                lines.append(f"Best Epoch: {metrics.get('best_epoch', 'N/A')}")
            config = checkpoint.get("config", {})
            if config:
                lines.append(f"Backbone: {config.get('backbone', 'N/A')}")
                lines.append(f"Test Fold: {config.get('test_fold', 'N/A')}")
            return "\n".join(lines)
        except Exception as e:
            return f"Failed to read checkpoint: {e}"

    def load_checkpoint_config(self, checkpoint_path: str) -> Dict[str, Any]:
        if not os.path.isfile(checkpoint_path):
            return {}
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            return checkpoint.get("config", {})
        except Exception:
            return {}

    def list_checkpoints(self, directory: str) -> list:
        if not os.path.isdir(directory):
            return []
        files = [f for f in os.listdir(directory) if f.endswith((".pth", ".ckpt"))]
        files.sort(key=lambda x: os.path.getmtime(os.path.join(directory, x)), reverse=True)
        return files
