import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional


@dataclass
class CIMFDataConfig:
    data_root: str = "./data"
    test_fold: int = 0
    total_folds: int = 5
    num_classes: int = 3
    image_size: int = 224


@dataclass
class CIMFModelConfig:
    backbone: str = "ResNet50-512"
    num_classes: int = 3
    gamma: float = 1.0
    tau: float = 0.4
    lambda_in: float = 1.0
    lambda_cross: float = 1.0
    lambda_hybrid: float = 1.0
    lambda_single: float = 1.0
    lambda_causal: float = 1.0
    lambda_cls: float = 1.0
    use_pretrained: bool = True
    weights_path: str = ""


@dataclass
class CIMFTrainingConfig:
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 5e-3
    epochs: int = 50
    optimizer: str = "AdamW"
    scheduler: str = "Cosine"
    grad_clip: float = 1.0
    patience: int = 20
    save_interval: int = 5
    auto_save: bool = True
    quick_test: bool = False
    val_timeout_sec: int = 60
    val_max_stall: int = 2
    lambda_in: float = 1.0
    lambda_cross: float = 1.0
    lambda_hybrid: float = 1.0
    lambda_single: float = 1.0
    lambda_causal: float = 1.0
    lambda_cls: float = 1.0


@dataclass
class CIMFConfig:
    data: CIMFDataConfig = field(default_factory=CIMFDataConfig)
    model: CIMFModelConfig = field(default_factory=CIMFModelConfig)
    training: CIMFTrainingConfig = field(default_factory=CIMFTrainingConfig)


class ConfigManager:
    DEFAULT_FLAT = {
        "data_path": "./data",
        "quick_test": False,
        "backbone": "ResNet50-512",
        "test_fold": 0,
        "batch_size": 64,
        "lr": 1e-4,
        "weight_decay": 5e-3,
        "epochs": 50,
        "gamma": 1.0,
        "tau": 0.4,
        "optimizer": "AdamW",
        "scheduler": "Cosine",
        "grad_clip": 1.0,
        "patience": 20,
        "auto_save": True,
        "save_interval": 5,
        "checkpoint_path": "./checkpoints",
        "checkpoint_file": "",
        "use_pretrained": True,
        "weights_path": "",
        "lambda_in": 1.0,
        "lambda_cross": 1.0,
        "lambda_hybrid": 1.0,
        "lambda_single": 1.0,
        "lambda_causal": 1.0,
        "lambda_cls": 1.0,
        "val_timeout_sec": 60,
        "val_max_stall": 2,
    }

    def get_default_config(self) -> Dict[str, Any]:
        return dict(self.DEFAULT_FLAT)

    def to_cimf_config(self, flat: Dict[str, Any]) -> CIMFConfig:
        return CIMFConfig(
            data=CIMFDataConfig(
                data_root=flat.get("data_path", "./data"),
                test_fold=int(flat.get("test_fold", 0)),
                total_folds=5,
                num_classes=3,
            ),
            model=CIMFModelConfig(
                backbone=flat.get("backbone", "ResNet50-512"),
                num_classes=3,
                gamma=float(flat.get("gamma", 1.0)),
                tau=float(flat.get("tau", 0.4)),
                lambda_in=float(flat.get("lambda_in", 1.0)),
                lambda_cross=float(flat.get("lambda_cross", 1.0)),
                lambda_hybrid=float(flat.get("lambda_hybrid", 1.0)),
                lambda_single=float(flat.get("lambda_single", 1.0)),
                lambda_causal=float(flat.get("lambda_causal", 1.0)),
                lambda_cls=float(flat.get("lambda_cls", 1.0)),
                use_pretrained=bool(flat.get("use_pretrained", True)),
                weights_path=flat.get("weights_path", ""),
            ),
            training=CIMFTrainingConfig(
                batch_size=int(flat.get("batch_size", 64)),
                learning_rate=float(flat.get("lr", 1e-4)),
                weight_decay=float(flat.get("weight_decay", 5e-3)),
                epochs=int(flat.get("epochs", 50)),
                optimizer=flat.get("optimizer", "AdamW"),
                scheduler=flat.get("scheduler", "Cosine"),
                grad_clip=float(flat.get("grad_clip", 1.0)),
                patience=int(flat.get("patience", 20)),
                save_interval=int(flat.get("save_interval", 5)),
                auto_save=bool(flat.get("auto_save", True)),
                quick_test=bool(flat.get("quick_test", False)),
                val_timeout_sec=int(flat.get("val_timeout_sec", 60)),
                val_max_stall=int(flat.get("val_max_stall", 2)),
                lambda_in=float(flat.get("lambda_in", 1.0)),
                lambda_cross=float(flat.get("lambda_cross", 1.0)),
                lambda_hybrid=float(flat.get("lambda_hybrid", 1.0)),
                lambda_single=float(flat.get("lambda_single", 1.0)),
                lambda_causal=float(flat.get("lambda_causal", 1.0)),
                lambda_cls=float(flat.get("lambda_cls", 1.0)),
            ),
        )

    def save_config(self, config: Dict[str, Any], filepath: str):
        payload = {"config": config}
        os.makedirs(os.path.dirname(os.path.abspath(filepath)) or ".", exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def load_config(self, filepath: str) -> Dict[str, Any]:
        with open(filepath, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict) and "config" in payload:
            merged = dict(self.DEFAULT_FLAT)
            merged.update(payload["config"])
            return merged
        if isinstance(payload, dict):
            merged = dict(self.DEFAULT_FLAT)
            merged.update(payload)
            return merged
        return dict(self.DEFAULT_FLAT)

    def save_cimf_config(self, config: CIMFConfig, filepath: str):
        flat = self.flatten_cimf_config(config)
        self.save_config(flat, filepath)

    def load_cimf_config(self, filepath: str) -> CIMFConfig:
        flat = self.load_config(filepath)
        return self.to_cimf_config(flat)

    def flatten_cimf_config(self, config: CIMFConfig) -> Dict[str, Any]:
        return {
            "data_path": config.data.data_root,
            "test_fold": config.data.test_fold,
            "backbone": config.model.backbone,
            "batch_size": config.training.batch_size,
            "lr": config.training.learning_rate,
            "weight_decay": config.training.weight_decay,
            "epochs": config.training.epochs,
            "gamma": config.model.gamma,
            "tau": config.model.tau,
            "optimizer": config.training.optimizer,
            "scheduler": config.training.scheduler,
            "grad_clip": config.training.grad_clip,
            "patience": config.training.patience,
            "save_interval": config.training.save_interval,
            "auto_save": config.training.auto_save,
            "quick_test": config.training.quick_test,
            "use_pretrained": config.model.use_pretrained,
            "weights_path": config.model.weights_path,
            "lambda_in": config.model.lambda_in,
            "lambda_cross": config.model.lambda_cross,
            "lambda_hybrid": config.model.lambda_hybrid,
            "lambda_single": config.model.lambda_single,
            "lambda_causal": config.model.lambda_causal,
            "lambda_cls": config.model.lambda_cls,
            "val_timeout_sec": config.training.val_timeout_sec,
            "val_max_stall": config.training.val_max_stall,
        }

    def config_to_dict(self, config: CIMFConfig) -> Dict[str, Any]:
        return {
            "data": asdict(config.data),
            "model": asdict(config.model),
            "training": asdict(config.training),
        }
