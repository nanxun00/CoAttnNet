# GLANet

Code repository for the paper:

**GLANet: Global--Local Attention Network for Robust Seismic Phase Picking**

## Overview

This project provides training, evaluation, and visualization scripts for robust seismic phase picking experiments.

Main components include:

- `phase_core.py`: core training and evaluation pipeline.
- `phase_run.py`: ablation and experiment entry scripts.
- `plot_*.py` and `phase_draw.py`: figure generation scripts.

## Privacy and Path Configuration

Machine-specific absolute paths have been removed from the codebase.

Use environment variables (or defaults) to configure data/checkpoint locations:

- `GLANET_DATA_ROOT`: root directory for shared project outputs.
- `GLANET_CKPT_ROOT`: checkpoint directory (default: `checkpoints` under data root).
- `GLANET_H5_ROOT`: H5 dataset root directory.
- `GLANET_CACHE_DIR`: HuggingFace/datasets cache directory.
- `GLANET_CEED_LOCAL_DIR`: optional local CEED `.h5` data directory.
- `PHASENET_OUTPUT_DIR`: optional output root directory for training artifacts.

If no variables are set, scripts use repository-relative defaults where possible.

## `.env.example` Usage

An environment template is provided at `.env.example`.

1. Copy template:
   - Linux/macOS: `cp .env.example .env`
   - Windows PowerShell: `Copy-Item .env.example .env`
2. Edit `.env` with your local dataset/checkpoint paths.
3. Export the variables to your shell (or use your preferred `.env` loader) before running scripts.

## Quick Start

1. Create environment from `environment.yml`.
2. Prepare dataset files (NPZ or H5) and checkpoints.
3. Configure environment variables (recommended via `.env.example`).
4. Run experiments, for example:
   - `python phase_run.py`

## Citation

If you use this repository in your work, please cite the GLANet paper above.
