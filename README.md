# CIMF: Cross-modal Invariant Multi-modal Fusion for Estrus Stage Classification

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

## Overview

**CIMF-Estrus-Detection** is a PyQt5-based training framework for dairy cow estrus stage classification using paired EEG and audio spectrogram modalities. The model combines MCAF (Multi-modal Cross-modal Alignment Fusion) and CIRE (Causal Invariant Relation Embedding) modules with a 3-class estrus stage taxonomy.

### Estrus Stage Classes (3 Categories)

| Stage | Description |
|-------|-------------|
| **pre_estrus** | Pre-estrus behavioral and physiological patterns |
| **estrus** | Active estrus stage with peak mating receptivity |
| **post_estrus** | Post-estrus recovery phase patterns |

## Architecture

- **MCAF Module**: Cross-modal proxy learning with in-modal, cross-modal, hybrid, and single-modal losses
- **CIRE Module**: Causal invariant embedding via Fourier amplitude augmentation
- **Fourier Augmentation**: Cross-modal amplitude mixing with configurable tau
- **Classification Head**: Cross-entropy loss on fused EEG/audio features

## Quick Start

### Installation

```bash
pip install -r requirements-core.txt
pip install -r requirements.txt
```

### Dataset Preparation

```bash
data/
├── eeg/
│   ├── pre_estrus/
│   ├── estrus/
│   └── post_estrus/
└── audio/
    ├── pre_estrus/
    ├── estrus/
    └── post_estrus/
```

Paired samples are matched by filename stem across EEG and audio directories. Cross-subject 5-fold evaluation uses test folds F0 through F4.

### Training

```bash
python main.py
python quick_start.py
python safe_training_launcher.py
```

### Configuration

Edit `stable_config.json` for default training parameters including backbone, test_fold, gamma (default 1.0), and CIRE lambda weights.

## Project Structure

```
CIMF-Estrus-Detection/
├── main.py
├── quick_start.py
├── safe_training_launcher.py
├── training_stability_fix.py
├── stable_config.json
├── requirements.txt
├── requirements-core.txt
└── gui_trainer/
    ├── models/cimf_model.py
    ├── core/training_backend.py
    ├── core/checkpoint_manager.py
    ├── core/log_parser.py
    ├── core/training_thread.py
    ├── ui/main_window.py
    ├── ui/visualization_window.py
    ├── ui/log_window.py
    └── utils/
```

## Evaluation Metrics

- Accuracy
- AUC (one-vs-rest macro)
- Sensitivity (macro recall)
- Specificity (macro)

Training artifacts are saved under `checkpoints/Task_3CLS/` including checkpoints, CSV logs, t-SNE plots, and confusion matrices.

## License

Licensed under Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.

Contact: dongyan@ljjtyy.com
