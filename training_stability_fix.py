#!/usr/bin/env python3

import os
import sys
import torch
import traceback


def diagnose_system():
    print("=== System Diagnosis ===")
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")


def check_imports():
    print("\n=== Module Import Check ===")
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    modules = [
        "torch", "torchvision", "numpy", "PyQt5",
        "gui_trainer.utils.config_utils",
        "gui_trainer.utils.logger",
        "gui_trainer.utils.dual_modal_dataset",
        "gui_trainer.models.cimf_model",
        "gui_trainer.core.training_backend",
    ]
    for module in modules:
        try:
            __import__(module)
            print(f"OK: {module}")
        except ImportError as e:
            print(f"FAIL: {module}: {e}")


def check_config():
    print("\n=== Configuration Check ===")
    try:
        import json
        with open("stable_config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        print("OK: stable_config.json loaded")
        train_config = config.get("config", {})
        for param in ["data_path", "backbone", "batch_size", "lr", "epochs", "test_fold"]:
            print(f"  {param}: {train_config.get(param, 'missing')}")
        data_path = train_config.get("data_path", "./data")
        if os.path.exists(data_path):
            print(f"OK: data path exists: {data_path}")
            for subdir in ["eeg", "audio"]:
                subpath = os.path.join(data_path, subdir)
                if os.path.isdir(subpath):
                    print(f"  OK: {subdir}/ exists")
                else:
                    print(f"  Warning: {subdir}/ missing")
    except Exception as e:
        print(f"Config check failed: {e}")


def check_memory_usage():
    print("\n=== Memory Check ===")
    try:
        import psutil
        mem = psutil.virtual_memory()
        print(f"Memory usage: {mem.percent:.1f}%")
    except ImportError:
        print("psutil not installed")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"GPU {i}: {allocated:.2f}GB / {total:.2f}GB")


def main():
    print("CIMF GUI Trainer Stability Diagnostic")
    print("=" * 50)
    try:
        diagnose_system()
        check_imports()
        check_config()
        check_memory_usage()
        print("\n=== Diagnosis Complete ===")
    except Exception as e:
        print(f"Diagnosis error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
