#!/usr/bin/env python3

import sys
import os
import subprocess


def check_and_install_deps():
    print("Checking core dependencies...")
    core_deps = ["torch", "torchvision", "PyQt5", "numpy", "matplotlib"]
    missing_deps = []
    for dep in core_deps:
        try:
            __import__(dep)
            print(f"OK: {dep}")
        except ImportError:
            print(f"Missing: {dep}")
            missing_deps.append(dep)
    if missing_deps:
        print(f"Installing missing dependencies: {missing_deps}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing_deps])
            return True
        except subprocess.CalledProcessError:
            print("Automatic install failed. Run: pip install " + " ".join(missing_deps))
            return False
    return True


def main():
    print("CIMF GUI Trainer Quick Start")
    print("=" * 40)
    if not os.path.exists("main.py"):
        print("Error: run this script from the project root directory")
        sys.exit(1)
    if not check_and_install_deps():
        sys.exit(1)
    print("\nLaunching GUI trainer...")
    from main import main as launch_main
    launch_main()


if __name__ == "__main__":
    main()
