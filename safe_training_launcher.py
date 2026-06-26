#!/usr/bin/env python3

import os
import sys
import signal
import traceback


def setup_environment():
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"


def check_dependencies():
    required_modules = ["torch", "torchvision", "numpy", "PyQt5"]
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
            print(f"OK: {module}")
        except ImportError:
            missing_modules.append(module)
            print(f"Missing: {module}")
    if missing_modules:
        print(f"Install with: pip install {' '.join(missing_modules)}")
        return False
    return True


def setup_signal_handlers():
    def signal_handler(signum, frame):
        print(f"\nReceived signal {signum}, exiting...")
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def run_diagnosis():
    print("Running quick diagnosis...")
    data_path = "./data"
    if os.path.exists(data_path):
        print(f"Data directory found: {data_path}")
        for subdir in ["eeg", "audio"]:
            subpath = os.path.join(data_path, subdir)
            if os.path.isdir(subpath):
                print(f"OK: {subdir}/ exists")
            else:
                print(f"Warning: {subdir}/ not found")
    else:
        print(f"Warning: data directory not found: {data_path}")
    try:
        import gui_trainer
        print("OK: gui_trainer module importable")
    except ImportError as e:
        print(f"Error: gui_trainer import failed: {e}")
        return False
    return True


def launch_gui():
    from main import main
    main()


def main():
    print("CIMF GUI Trainer - Safe Launcher")
    print("=" * 50)
    setup_signal_handlers()
    try:
        setup_environment()
        if not check_dependencies():
            sys.exit(1)
        if not run_diagnosis():
            sys.exit(1)
        print("\nAll checks passed, launching GUI...")
        launch_gui()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
