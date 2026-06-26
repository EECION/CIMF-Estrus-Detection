import os
from typing import Optional

import torch
import torchvision.models as models


class ModelUtils:
    BACKBONE_FILES = {
        "ResNet50-512": "resnet50_512.pth",
        "ResNet50-64": "resnet50_64.pth",
        "ViT-B-768": "vit_b_768.pth",
        "Swin-T-768": "swin_t_768.pth",
    }

    def load_pretrained_weights(
        self,
        model,
        backbone: str,
        weights_path: Optional[str] = None,
        weights_dir: Optional[str] = None,
    ) -> bool:
        path = self._resolve_weights_path(backbone, weights_path, weights_dir)
        if not path or not os.path.exists(path):
            return self._load_imagenet_backbone(model, backbone)
        try:
            state = torch.load(path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            elif isinstance(state, dict) and "model" in state:
                state = state["model"]
            adapted = self._adapt_state_dict(state)
            targets = []
            if hasattr(model, "module"):
                targets.append(model.module)
            else:
                targets.append(model)
            loaded_any = False
            for target in targets:
                for attr in ("backbone_eeg", "backbone_shared"):
                    if hasattr(target, attr):
                        sub = getattr(target, attr)
                        if hasattr(sub, "backbone"):
                            missing, unexpected = sub.backbone.load_state_dict(adapted, strict=False)
                            if len(missing) < len(sub.backbone.state_dict()):
                                loaded_any = True
            return loaded_any
        except Exception:
            return self._load_imagenet_backbone(model, backbone)

    def _resolve_weights_path(
        self,
        backbone: str,
        weights_path: Optional[str],
        weights_dir: Optional[str],
    ) -> Optional[str]:
        if weights_path and os.path.isfile(weights_path):
            return weights_path
        if weights_dir and os.path.isdir(weights_dir):
            candidate = os.path.join(weights_dir, self.BACKBONE_FILES.get(backbone, ""))
            if os.path.isfile(candidate):
                return candidate
        return None

    def _adapt_state_dict(self, state: dict) -> dict:
        adapted = {}
        for key, value in state.items():
            new_key = key
            if new_key.startswith("module."):
                new_key = new_key[7:]
            if new_key.startswith("backbone."):
                new_key = new_key[len("backbone."):]
            adapted[new_key] = value
        return adapted

    def _load_imagenet_backbone(self, model, backbone: str) -> bool:
        try:
            if "ResNet50" not in backbone:
                return False
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            resnet_state = resnet.state_dict()
            adapted = {}
            for key, value in resnet_state.items():
                if key.startswith("fc."):
                    continue
                adapted[key] = value
            targets = model.module if hasattr(model, "module") else model
            loaded = False
            for attr in ("backbone_eeg", "backbone_shared"):
                if hasattr(targets, attr):
                    sub = getattr(targets, attr)
                    if hasattr(sub, "backbone"):
                        sub.backbone.load_state_dict(adapted, strict=False)
                        loaded = True
            return loaded
        except Exception:
            return False
