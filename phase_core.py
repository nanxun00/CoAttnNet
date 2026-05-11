"""
Phase picking core process (simplified version):
- Training loss only uses cross-entropy (soft CE)
- Threshold strategy retains only fixed thresholds with P/S grid search
- Evaluation retains only deterministic one-shot forward
"""

from __future__ import annotations

import os
import json
from datetime import datetime

# Must be set before importing datasets; empty means using default cache locations.
CEED_CACHE_DIR = os.environ.get("GLANET_CACHE_DIR", "")
# If No space left on device is still reported, please set the environment variables PHASENET_OUTPUT_DIR and CEED_CACHE_DIR to the same disk at the same time (see OUTPUT_BASE_DIR below)
if CEED_CACHE_DIR:
    os.environ["HF_DATASETS_CACHE"] = CEED_CACHE_DIR
    os.environ["HF_HUB_CACHE"] = os.path.join(CEED_CACHE_DIR, "hub")
    os.environ["HF_HOME"] = CEED_CACHE_DIR

import csv
import math
import random
from typing import Dict, Any
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from utils.repro import seed_everything, seed_worker, torch_generator

from phase_model import PhaseNetUNet
from data import WaveformDataset
from ceed_data import CEEDDataset
from three_channel_h5_dataset import ThreeChannelH5Dataset

# Drawing: relies on the general drawing functions in single_ablation_visualization
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore
except Exception:
    plt = None

from single_ablation_visualization import (
    plot_losses,
    estimate_snr,
    plot_one_sample_visual,
    plot_time_residual_distribution,
    plot_representative_waveforms_grid,
    plot_pca_visualization,
    plot_pr_curve,
    plot_snr_stratified,
    plot_max_prob_histogram,
)

# ===== Data and training parameters =====
DATA_SOURCE = "ceed"  # "npz" | "ceed" | "h5_three_channel"

# NPZ
DATA_ROOT = "dataset"
TRAIN_DIR = os.path.join(DATA_ROOT, "waveform_train")
TRAIN_CSV = os.path.join(DATA_ROOT, "waveform_train_split.csv")
VALID_CSV = os.path.join(DATA_ROOT, "waveform_valid_split.csv")

# CEED
# If you have downloaded CEED's .h5 locally, fill in the directory path;
# If empty, the default behavior will be used (online download)
CEED_LOCAL_DIR = os.environ.get("GLANET_CEED_LOCAL_DIR", "")  # Optional local CEED .h5 directory.
CEED_DATASET_NAME = "CEED.py"
CEED_TRAIN_SPLIT = "train"
CEED_VALID_SPLIT = "train"   # The validation set is divided from the train split
CEED_TEST_SPLIT = "test"     # Independent test set, directly use HF’s test split
CEED_LIMIT_TRAIN = 9000      # The number loaded from train split is divided into training set and validation set in proportion.
CEED_VAL_RATIO = 0.2        # 20% of the train part is the verification set (about 8:2), increase the proportion of the verification set
CEED_LIMIT_TEST = 1000       # Number of test sets loaded from test split (reduces test set size)
# Division: train split → 9:1 → training/validation; test uses test split directly
CEED_WAVEFORM_KEY = None  # e.g. "waveform" / "seismogram"
CEED_P_KEY = None         # e.g. "p_arrival_sample"
CEED_S_KEY = None         # e.g. "s_arrival_sample"



# Three-channel H5 dataset (/waveforms channel_ud|ns|ew, /arrival_times pg|sg seconds)
# Partitioning and CEED roughly "reverse" the ratio of validation set to test set: the test set shrinks and the validation set grows
H5_THREE_CHANNEL_ROOT = os.environ.get("GLANET_H5_ROOT", "dataset")  # Root directory of H5 files.
H5_TEST_RATIO = 0.08     # Test set proportion (about 8%), first draw out
H5_TRAIN_VAL_RATIO = 0.78  # After deducting test, about 78% training / 22% verification (overall about 72%/20%/8%)
H5_LIMIT = 10000        # The total number of samples participating in the division
H5_LIMIT_TRAIN = None   # The maximum number of samples used in the training set, None means no limit
H5_LIMIT_VAL = None     # Maximum number of samples used in the validation set
H5_LIMIT_TEST = None    # The maximum number of samples used in the test set
H5_ARRIVAL_RELATIVE_TO_SEGMENT = True  # True: pg/sg is the second relative to the beginning of the segment; False: absolute time (the segment is P-10s～P+110s)
H5_FILTER_NATURAL_ONLY = False  # Whether to keep only natural_earthquake==1
H5_ALLOW_TYPES = None  # For example ("eq",) only retains natural earthquake types

# Strict data checking: After it is turned on, inconsistencies in sampling rate/timeout/channel length, etc. will directly throw an error (it is recommended to turn it on first when accessing new data)
# Manually switch True/False directly here (environment variables are no longer read)
H5_STRICT_CHECK = False

# CEED_CACHE_DIR has been moved to the top of the file (before import) to ensure HuggingFace checks for the correct partition

# Output root directory: PhaseNet/ablation_unet, metrics, graphs, etc. are written here. Consistent with the data set cache path to avoid filling up the system disk
OUTPUT_BASE_DIR = os.environ.get("PHASENET_OUTPUT_DIR", CEED_CACHE_DIR if CEED_CACHE_DIR else ".")
OUT_ROOT = os.path.join(OUTPUT_BASE_DIR, "PhaseNet", "ablation_unet")
os.makedirs(OUT_ROOT, exist_ok=True)
METRICS_CSV = os.path.join(OUT_ROOT, "metrics.csv")
METRICS_HEADER = [
    "name", "best_val", "train_last", "valid_last",
    "use_cbam", "kernels",
    "thr_p", "thr_s",
    "time_acc", "mcc",
    "p_prec", "p_rec", "p_f1", "s_prec", "s_rec", "s_f1",
    # Time residual statistics (unit: seconds)
    "p_res_mean_sec", "p_res_std_sec", "p_res_mae_sec",
    "s_res_mean_sec", "s_res_std_sec", "s_res_mae_sec",
]

MODEL_CLASS_MAP: Dict[str, type[nn.Module]] = {
    "PhaseNetUNet": PhaseNetUNet,
}


# Training / Random Seed (default strategy targeting "effect first")
# Empirically: For deep models containing multi-scale + softgate + phasewise, AdamW + moderate weight decay is more stable;
# The learning rate is slightly lower than 1e-2 and the training epochs are slightly longer to ensure convergence.
EPOCHS = 120
BATCH_SIZE = 32
LR = 5e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0
CROP_LEN = 3000
LABEL_WIDTH = 51
LABEL_SIGMA_SEC = 0.1
SAMPLE_RATE = 100.0
N_VIS = 4
SEED = 2025  # Changed from 42 to 2024, consistent with train.py

# Cache the dataset partition index and underlying dataset objects to avoid reloading each time
_dataset_split_cache = {}
_base_dataset_cache = {}  # Cache underlying dataset objects (without enhancements)
_ceed_test_ds_cache = None  # CEED test split is only loaded once to avoid repeatedly printing "Generating test split" when there are multiple configs.

def _get_split_cache_key(use_waveform_aug: bool, aug_params: tuple) -> str:
    """Generate the key to partition the cache of the data set"""
    if DATA_SOURCE == "h5_three_channel":
        return f"{DATA_SOURCE}_{H5_THREE_CHANNEL_ROOT}_{H5_TEST_RATIO}_{H5_TRAIN_VAL_RATIO}_{H5_LIMIT}_{SEED}"
    return f"{DATA_SOURCE}_{CEED_LIMIT_TRAIN}_{CEED_VAL_RATIO}_{CEED_LIMIT_TEST}_{SEED}_{use_waveform_aug}_{aug_params}"

# Whether to do an additional independent evaluation of the test split in the CEED process (forward + indicators only)
RUN_TEST_EVAL = True

# Dynamic Thresholding (Grid Search F1)
DYN_THRESH_GRID = [x / 100.0 for x in range(10, 96, 5)]
DYN_TOL_SAMPLES = 10

# Optimal model selection: by validation set F1 (recommended) instead of val loss, aligned with reviewer/actual metrics
BEST_MODEL_BY_F1 = True
BEST_METRIC_WEIGHTS = (0.5, 0.5)  # (w_p, w_s), can be changed to (0.0, 1.0) only with S-F1

# Early stopping is only enabled for three-channel H5: training will stop if the optimal index of the validation set no longer improves for several consecutive epochs.
H5_EARLY_STOP_PATIENCE = 20

# Learning rate scheduling (consistent with train.py: ReduceLROnPlateau)
SCHEDULER = "plateau"  # "plateau" | "none"
PLATEAU_FACTOR = 0.5
PLATEAU_PATIENCE = 8   # Avoid dropping lr too low too early (like 1e-6)
PLATEAU_MIN_LR = 1e-6
PLATEAU_THRESHOLD = 1e-4

def _write_comparison_format(case_dir: str) -> None:
    """Write the comparison data format description in JSON to facilitate users to draw comparison charts after adding baselines such as AR."""
    fmt = {
        "description": "将对比方法的数据按下列格式写入同目录，运行 plot_paper_figures.py --compare 可绘制多方法对比图。",
        "pr_snr_data": {
            "filename_pattern": "pr_snr_data_<method>.json（如 pr_snr_data_ar.json）",
            "keys": ["has_p", "has_s", "max_prob_p", "max_prob_s", "p_ok", "s_ok", "snr"],
            "key_descriptions": {
                "has_p": "bool list, 每样本是否有 P 波真值",
                "has_s": "bool list, 每样本是否有 S 波真值",
                "max_prob_p": "float list, 每样本 P 通道最大概率",
                "max_prob_s": "float list, 每样本 S 通道最大概率",
                "p_ok": "bool list, 每样本 P 拾取是否在容差内正确",
                "s_ok": "bool list, 每样本 S 拾取是否在容差内正确",
                "snr": "float list, 每样本 SNR (dB)",
            },
        },
        "time_residuals": {
            "filename_pattern": "time_residuals_<method>.json（如 time_residuals_ar.json）",
            "keys": ["p_residuals_signed", "s_residuals_signed"],
            "key_descriptions": {
                "p_residuals_signed": "float list, P 波预测索引 − 真值索引（样本数）",
                "s_residuals_signed": "float list, S 波预测索引 − 真值索引（样本数）",
            },
        },
        "pca": {
            "filename_pattern": "pca_<method>.npz（如 pca_ar.npz）",
            "arrays": ["features", "labels"],
            "descriptions": {
                "features": "shape (N, D), 每样本特征向量（12 维：max_p, max_s, mean_p, mean_s, std_p, std_s, entropy_p, entropy_s, peak_width_p, peak_width_s, margin_p, margin_s）",
                "labels": "shape (N,) int, 0=无震相 1=P_only 2=S_only 3=P+S",
            },
        },
    }
    with open(os.path.join(case_dir, "comparison_data_format.json"), "w", encoding="utf-8") as f:
        json.dump(fmt, f, ensure_ascii=False, indent=2)


def _select_phase_peaks(prob: np.ndarray, threshold: float, min_interval: int, cap: int) -> list[int]:
    """Select candidate peaks from the probabilistic sequence based on threshold/NMS/interval, up to cap (in descending order of probability)."""
    thr = float(threshold)
    indices = np.where(prob >= thr)[0]
    if indices.size == 0:
        return []
    sorted_indices = sorted(indices, key=lambda idx: -prob[idx])
    peaks: list[int] = []
    for idx in sorted_indices:
        if all(abs(idx - prev) >= min_interval for prev in peaks):
            peaks.append(idx)
            if len(peaks) >= cap:
                break
    return sorted(peaks)


def apply_global_cap(peaks: list[int], prob: np.ndarray, cap: int) -> tuple[list[int], int]:
    if cap <= 0 or not peaks:
        return peaks, 0
    sorted_by_prob = sorted(peaks, key=lambda idx: -prob[idx])
    kept = sorted(sorted_by_prob[:cap])
    removed = len(peaks) - len(kept)
    return kept, removed


def enforce_p_before_s(p_peaks: list[int], s_peaks: list[int], min_gap: int, p_prob: np.ndarray, s_prob: np.ndarray) -> tuple[list[int], int]:
    if min_gap <= 0 or not p_peaks or not s_peaks:
        return s_peaks, 0
    threshold = max(p_peaks[:2]) + min_gap
    filtered = [s for s in s_peaks if s > threshold]
    if filtered:
        return filtered, len(s_peaks) - len(filtered)
    # Try using the second strongest P
    if len(p_peaks) > 1:
        threshold = p_peaks[1] + min_gap
        filtered = [s for s in s_peaks if s > threshold]
        if filtered:
            return filtered, len(s_peaks) - len(filtered)
    # If U is still not satisfied, the original S is retained, but the first two
    return s_peaks[: min(2, len(s_peaks))], len(s_peaks) - min(2, len(s_peaks))


def _apply_ps_spacing(peaks: list[int], other_peaks: list[int], min_gap: int) -> list[int]:
    if min_gap <= 0 or not other_peaks:
        return peaks
    filtered = []
    for idx in peaks:
        if all(abs(idx - prev) >= min_gap for prev in other_peaks):
            filtered.append(idx)
    return filtered

# ===== General Tools =====
def _center_crop_time(y: torch.Tensor, target_T: int) -> torch.Tensor:
    """Center-aligned crop/padding in time dimension to target length."""
    T = y.shape[-1]
    if T == target_T:
        return y
    if T < target_T:
        pad = target_T - T
        left = pad // 2
        right = pad - left
        return F.pad(y, (left, right))
    start = (T - target_T) // 2
    return y[..., start : start + target_T]


def soft_ce(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (-(y * torch.log_softmax(logits, dim=1))).mean()

def combined_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """
    Cross entropy loss (soft label version).

    y is one-hot/soft label, use soft cross entropy: -(y * log_softmax).mean()
    """
    return soft_ce(logits, y)


def build_datasets(case: Dict[str, Any] = None):
    """Build training/validation data sets. Can be used to pass CEED augmentation parameters (aug_noise_* etc.) when case is not None.
    Waveform augmentation is turned off, use_waveform_augmentation is no longer read from case.
    """
    aug = case or {}
    use_waveform_aug = False  # Waveform enhancement is no longer used in ablation
    aug_noise_min = float(aug.get("aug_noise_snr_db_min", 3.0))
    aug_noise_max = float(aug.get("aug_noise_snr_db_max", 20.0))
    aug_amp_min = float(aug.get("aug_amplitude_min", 0.5))
    aug_amp_max = float(aug.get("aug_amplitude_max", 2.0))
    aug_params = (aug_noise_min, aug_noise_max, aug_amp_min, aug_amp_max)

    if DATA_SOURCE == "npz":
        train_ds = WaveformDataset(
            TRAIN_DIR,
            TRAIN_CSV,
            crop_len=CROP_LEN,
            label_width=LABEL_WIDTH,
            training=True,
            sampling_rate=SAMPLE_RATE,
            label_sigma_sec=LABEL_SIGMA_SEC,
        )
        valid_ds = WaveformDataset(
            TRAIN_DIR,
            VALID_CSV,
            crop_len=CROP_LEN,
            label_width=LABEL_WIDTH,
            training=False,
            sampling_rate=SAMPLE_RATE,
            label_sigma_sec=LABEL_SIGMA_SEC,
        )
        return train_ds, valid_ds, None
    elif DATA_SOURCE == "ceed":
        cache_key = _get_split_cache_key(use_waveform_aug, aug_params)
        if cache_key in _dataset_split_cache:
            train_idx, val_idx = _dataset_split_cache[cache_key]
            print(f"[build_datasets] ✓ 使用缓存的 CEED 划分（train={len(train_idx)}, val={len(val_idx)}），test 使用独立 test split")
            full_train_ds_aug = CEEDDataset(
                dataset_name=CEED_DATASET_NAME,
                split=CEED_TRAIN_SPLIT,
                limit=CEED_LIMIT_TRAIN,
                waveform_key=CEED_WAVEFORM_KEY,
                p_key=CEED_P_KEY,
                s_key=CEED_S_KEY,
                sampling_rate=SAMPLE_RATE,
                crop_len=CROP_LEN,
                label_sigma_sec=LABEL_SIGMA_SEC,
                label_width=LABEL_WIDTH,
                training=True,
                local_dir=CEED_LOCAL_DIR,
                use_waveform_augmentation=use_waveform_aug,
                aug_noise_snr_db_min=aug_noise_min,
                aug_noise_snr_db_max=aug_noise_max,
                aug_amplitude_min=aug_amp_min,
                aug_amplitude_max=aug_amp_max,
            )
        else:
            print(f"[build_datasets] CEED：从 train split 加载（limit={CEED_LIMIT_TRAIN}），9:1 划分训练/验证；test 使用 test split")
            full_train_ds_aug = CEEDDataset(
                dataset_name=CEED_DATASET_NAME,
                split=CEED_TRAIN_SPLIT,
                limit=CEED_LIMIT_TRAIN,
                waveform_key=CEED_WAVEFORM_KEY,
                p_key=CEED_P_KEY,
                s_key=CEED_S_KEY,
                sampling_rate=SAMPLE_RATE,
                crop_len=CROP_LEN,
                label_sigma_sec=LABEL_SIGMA_SEC,
                label_width=LABEL_WIDTH,
                training=True,
                local_dir=CEED_LOCAL_DIR,
                use_waveform_augmentation=use_waveform_aug,
                aug_noise_snr_db_min=aug_noise_min,
                aug_noise_snr_db_max=aug_noise_max,
                aug_amplitude_min=aug_amp_min,
                aug_amplitude_max=aug_amp_max,
            )
            n_full = len(full_train_ds_aug)
            if n_full < 2:
                raise ValueError(f"CEED train 样本不足（{n_full}），至少需要 2 条")
            val_size = max(1, int(n_full * CEED_VAL_RATIO))
            train_size = n_full - val_size
            g = torch.Generator().manual_seed(SEED)
            perm = torch.randperm(n_full, generator=g)
            train_idx = perm[:train_size].tolist()
            val_idx = perm[train_size:].tolist()
            _dataset_split_cache[cache_key] = (train_idx, val_idx)
            print(f"[build_datasets] ✓ CEED train 划分完成并已缓存（train={train_size}, val={val_size}）")
        train_ds = Subset(full_train_ds_aug, train_idx)
        full_train_ds_clean = CEEDDataset(
            dataset_name=CEED_DATASET_NAME,
            split=CEED_TRAIN_SPLIT,
            limit=CEED_LIMIT_TRAIN,
            waveform_key=CEED_WAVEFORM_KEY,
            p_key=CEED_P_KEY,
            s_key=CEED_S_KEY,
            sampling_rate=SAMPLE_RATE,
            crop_len=CROP_LEN,
            label_sigma_sec=LABEL_SIGMA_SEC,
            label_width=LABEL_WIDTH,
            training=False,
            local_dir=CEED_LOCAL_DIR,
        )
        valid_ds = Subset(full_train_ds_clean, val_idx)
        global _ceed_test_ds_cache
        if _ceed_test_ds_cache is None:
            _ceed_test_ds_cache = CEEDDataset(
                dataset_name=CEED_DATASET_NAME,
                split=CEED_TEST_SPLIT,
                limit=CEED_LIMIT_TEST,
                waveform_key=CEED_WAVEFORM_KEY,
                p_key=CEED_P_KEY,
                s_key=CEED_S_KEY,
                sampling_rate=SAMPLE_RATE,
                crop_len=CROP_LEN,
                label_sigma_sec=LABEL_SIGMA_SEC,
                label_width=LABEL_WIDTH,
                training=False,
                local_dir=CEED_LOCAL_DIR,
            )
        test_ds = _ceed_test_ds_cache
        return train_ds, valid_ds, test_ds
    elif DATA_SOURCE == "h5_three_channel":
        cache_key = _get_split_cache_key(False, ())
        if cache_key in _dataset_split_cache:
            train_idx, val_idx, test_idx = _dataset_split_cache[cache_key]
            print(
                f"[build_datasets] ✓ 使用缓存的三通道 H5 划分（train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}）"
            )
        else:
            full_ds = ThreeChannelH5Dataset(
                root_dir=H5_THREE_CHANNEL_ROOT,
                crop_len=CROP_LEN,
                sampling_rate=SAMPLE_RATE,
                label_sigma_sec=LABEL_SIGMA_SEC,
                label_width=LABEL_WIDTH,
                training=True,
                arrival_relative_to_segment=H5_ARRIVAL_RELATIVE_TO_SEGMENT,
                filter_natural_only=H5_FILTER_NATURAL_ONLY,
                allow_earthquake_types=H5_ALLOW_TYPES,
                strict_check=H5_STRICT_CHECK,
                limit=H5_LIMIT,
            )
            n_full = len(full_ds)
            if n_full < 2:
                raise ValueError(f"三通道 H5 样本数不足（{n_full}），无法划分")
            test_size = int(n_full * H5_TEST_RATIO) if H5_TEST_RATIO and H5_TEST_RATIO > 0 else 0
            remainder = n_full - test_size
            val_size = max(1, int(remainder * (1 - H5_TRAIN_VAL_RATIO)))
            train_size = remainder - val_size
            if train_size < 1:
                raise ValueError(
                    f"三通道 H5 划分后训练集为空（n_full={n_full}, test={test_size}, val={val_size}）"
                )
            g = torch.Generator().manual_seed(SEED)
            perm = torch.randperm(n_full, generator=g)
            test_idx = perm[n_full - test_size :].tolist() if test_size > 0 else []
            train_idx = perm[:train_size].tolist()
            val_idx = perm[train_size : train_size + val_size].tolist()
            _dataset_split_cache[cache_key] = (train_idx, val_idx, test_idx)
            print(
                f"[build_datasets] ✓ 三通道 H5 划分完成（先划出 test={len(test_idx)}，剩余 9:1 → train={train_size}, val={val_size}）"
            )

        full_train_ds = ThreeChannelH5Dataset(
            root_dir=H5_THREE_CHANNEL_ROOT,
            crop_len=CROP_LEN,
            sampling_rate=SAMPLE_RATE,
            label_sigma_sec=LABEL_SIGMA_SEC,
            label_width=LABEL_WIDTH,
            training=True,
            arrival_relative_to_segment=H5_ARRIVAL_RELATIVE_TO_SEGMENT,
            filter_natural_only=H5_FILTER_NATURAL_ONLY,
            allow_earthquake_types=H5_ALLOW_TYPES,
            strict_check=H5_STRICT_CHECK,
            limit=H5_LIMIT,
        )
        full_valid_ds = ThreeChannelH5Dataset(
            root_dir=H5_THREE_CHANNEL_ROOT,
            crop_len=CROP_LEN,
            sampling_rate=SAMPLE_RATE,
            label_sigma_sec=LABEL_SIGMA_SEC,
            label_width=LABEL_WIDTH,
            training=False,
            arrival_relative_to_segment=H5_ARRIVAL_RELATIVE_TO_SEGMENT,
            filter_natural_only=H5_FILTER_NATURAL_ONLY,
            allow_earthquake_types=H5_ALLOW_TYPES,
            strict_check=H5_STRICT_CHECK,
            limit=H5_LIMIT,
        )
        # Optional: Set an upper limit on the quantity of each collection (corresponding to CEED_LIMIT_TRAIN / CEED_LIMIT_TEST)
        train_idx_use = train_idx[:H5_LIMIT_TRAIN] if H5_LIMIT_TRAIN is not None else train_idx
        val_idx_use = val_idx[:H5_LIMIT_VAL] if H5_LIMIT_VAL is not None else val_idx
        test_idx_use = test_idx[:H5_LIMIT_TEST] if (H5_LIMIT_TEST is not None and test_idx) else test_idx
        train_ds = Subset(full_train_ds, train_idx_use)
        valid_ds = Subset(full_valid_ds, val_idx_use)
        if test_idx_use:
            full_test_ds = ThreeChannelH5Dataset(
                root_dir=H5_THREE_CHANNEL_ROOT,
                crop_len=CROP_LEN,
                sampling_rate=SAMPLE_RATE,
                label_sigma_sec=LABEL_SIGMA_SEC,
                label_width=LABEL_WIDTH,
                training=False,
                arrival_relative_to_segment=H5_ARRIVAL_RELATIVE_TO_SEGMENT,
                filter_natural_only=H5_FILTER_NATURAL_ONLY,
                allow_earthquake_types=H5_ALLOW_TYPES,
                strict_check=H5_STRICT_CHECK,
                limit=H5_LIMIT,
            )
            test_ds = Subset(full_test_ds, test_idx_use)
        else:
            test_ds = None
    else:
        raise ValueError("DATA_SOURCE must be 'npz', 'ceed', or 'h5_three_channel'")
    return train_ds, valid_ds, test_ds


@torch.inference_mode()
def eval_loss(model, loader, device, threshold_balance_weight: float = 0.1,
              class_weights: torch.Tensor = None, epoch: int = 0, total_epochs: int = 0,
) -> float:
    model.eval(); total = 0.0; n = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [valid]", leave=False) if epoch > 0 else loader
    for x, y, _ in pbar:
        x = x.to(device); y = y.to(device)
        logits = model(x)
        if y.shape[-1] != logits.shape[-1]:
            y = _center_crop_time(y, logits.shape[-1])
        loss = combined_loss(
            logits,
            y,
        )
        bs = x.size(0); total += float(loss.item()) * bs; n += bs
        if epoch > 0:
            pbar.set_postfix({"loss": f"{total/max(1,n):.4f}"})
    return total / max(1, n)


@torch.inference_mode()
def _collect_conf_err(model, loader, device, tol_samples: int, max_batches: int = 200):
    """Collect the (conf, err, has_gt) lists (for P and S respectively) used for threshold calibration.

    - conf: maximum probability of this phase channel for each sample
    - err: If there is a true value, it is |pred_idx - gt_idx| (number of sampling points); if there is no true value, it is None
    - has_gt: Whether the phase true value exists in this sample
    """
    _ = tol_samples  # err itself is measured in samples, tol is used by external best_threshold
    model.eval()
    p_conf, p_err, p_has = [], [], []
    s_conf, s_err, s_has = [], [], []
    n_batches = 0
    for x, y, _meta in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        if y.shape[-1] != logits.shape[-1]:
            y = _center_crop_time(y, logits.shape[-1])
        probs_np = torch.softmax(logits, dim=1).cpu().numpy()
        y_np = y.cpu().numpy()
        B = probs_np.shape[0]
        for b in range(B):
            # Class P (channel 1)
            has_p = bool(y_np[b, 1].max() > 1e-6)
            p_idx = int(np.argmax(probs_np[b, 1]))
            p_c = float(probs_np[b, 1, p_idx])
            p_e = abs(p_idx - int(np.argmax(y_np[b, 1]))) if has_p else None
            p_conf.append(p_c)
            p_err.append(p_e)
            p_has.append(has_p)

            # Class S (Channel 2)
            has_s = bool(y_np[b, 2].max() > 1e-6)
            s_idx = int(np.argmax(probs_np[b, 2]))
            s_c = float(probs_np[b, 2, s_idx])
            s_e = abs(s_idx - int(np.argmax(y_np[b, 2]))) if has_s else None
            s_conf.append(s_c)
            s_err.append(s_e)
            s_has.append(has_s)

        n_batches += 1
        if n_batches >= int(max_batches):
            break

    return (p_conf, p_err, p_has), (s_conf, s_err, s_has)


# Compatible with old calls: collect_conf_err(model, loader, device) → no batch limit
@torch.inference_mode()
def collect_conf_err(model, loader, device):
    return _collect_conf_err(model, loader, device, tol_samples=DYN_TOL_SAMPLES, max_batches=10**9)


def _best_threshold(confs, errs, has_gts, tol: int, grid: list[float]):
    """Lightweight wrapper for best_threshold, keeping naming consistent with the paper/script snippet."""
    return best_threshold(confs, errs, has_gts, tol=tol, grid=grid, debug=False)


def best_threshold(confs, errs, has_gts, tol: int, grid: list[float], debug: bool = False):
    """
    Grid search for optimal threshold
    
    Args:
        confs: Confidence list
        errs: error list (None means there is no real label)
        has_gts: whether there is a real tag list
        tol: tolerance error (number of sampling points)
        grid: threshold grid
        debug: whether to output debugging information
    
    Returns:
        (best_thr, best_f1): best threshold and corresponding F1 score
    """
    best_thr, best_f1 = 0.5, -1.0
    best_stats = None
    
    # Statistics (for debugging)
    total_samples = len(confs)
    samples_with_gt = sum(has_gts)
    samples_without_gt = total_samples - samples_with_gt
    
    if debug and total_samples > 0:
        print(f"  总样本数: {total_samples}, 有标签: {samples_with_gt}, 无标签: {samples_without_gt}")
        if samples_with_gt > 0:
            valid_errs = [e for e, h in zip(errs, has_gts) if h and e is not None]
            if valid_errs:
                print(f"  有效误差统计: mean={np.mean(valid_errs):.1f}, "
                      f"min={np.min(valid_errs)}, max={np.max(valid_errs)}, "
                      f"<=tol({tol})的比例={np.mean(np.array(valid_errs) <= tol):.1%}")
            valid_confs = [c for c, h in zip(confs, has_gts) if h]
            if valid_confs:
                print(f"  有标签样本的置信度统计: mean={np.mean(valid_confs):.3f}, "
                      f"min={np.min(valid_confs):.3f}, max={np.max(valid_confs):.3f}")
    
    for thr in grid:
        tp = fp = fn = 0
        for c, e, h in zip(confs, errs, has_gts):
            if c >= thr:
                if h and e is not None and e <= tol: tp += 1
                else: fp += 1
            else:
                if h: fn += 1
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
        if f1 > best_f1: 
            best_f1 = f1
            best_thr = thr
            best_stats = (tp, fp, fn, prec, rec, f1)
    
    # If all thresholds result in F1=0, return 0.0 instead of -1.0
    if best_f1 < 0:
        best_f1 = 0.0
    
    if debug and best_stats:
        tp, fp, fn, prec, rec, f1 = best_stats
        print(f"  最佳阈值={best_thr:.3f}: TP={tp}, FP={fp}, FN={fn}, "
              f"Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")
    
    return best_thr, best_f1


@torch.inference_mode()
def eval_detailed(model, loader, device, thr_p: float = None, thr_s: float = None, 
                 uncertainty_threshold_options: dict = None,
                 tol: int = 10,
                 current_epoch: int | None = None,
                 quiet: bool = False) -> Dict[str, float]:
    """
    Detailed evaluation function (fixed threshold). uncertainty_threshold_options are only used for post-processing options such as structural.
    """
    model.eval()
    time_correct = 0; n_time = 0
    p_tp = p_fp = p_fn = p_tn = 0
    s_tp = s_fp = s_fn = s_tn = 0
    # For temporal residual distribution plots: pred_idx - gt_idx
    all_p_residual_signed: list[int] = []
    all_s_residual_signed: list[int] = []
    s_conf_list, s_thr_list = [], []
    opts = uncertainty_threshold_options or {}
    structural_opts = {
        "enabled": bool(opts.get("use_structural_postproc", True)),
        "min_interval_same": int(opts.get("postproc_min_interval_same", 20)),
        "min_interval_ps": int(opts.get("postproc_min_interval_ps", 30)),
        "candidate_limit": int(opts.get("postproc_candidate_limit", 10)),
        "cap_p": int(opts.get("postproc_cap_p", 1)),
        "cap_s": int(opts.get("postproc_cap_s", 1)),
        "enforce_p_before_s": bool(opts.get("postproc_enforce_p_before_s", False)),
    }
    # Count the number of candidates removed by structural post-processing
    structural_stats = {
        "cap_p": 0,
        "cap_s": 0,
        "order": 0,
    }
    candidate_lengths: dict[str, list[int]] = {"p": [], "s": []}
    zero_candidate_counts = {"p": 0, "s": 0}
    samples_processed = 0
    threshold_pass_counts = {"p": 0, "s": 0}
    uncertainty_stats = {
        "p": {"tp": [], "fp": [], "fn": []},
        "s": {"tp": [], "fp": [], "fn": []},
    }
    for x, y, _ in loader:
        x = x.to(device); y = y.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        use_per_sample_threshold = False
        suggested_thr = None
        if y.shape[-1] != logits.shape[-1]:
            y = _center_crop_time(y, logits.shape[-1])
        pred_cls = probs.argmax(dim=1); true_cls = y.argmax(dim=1)
        time_correct += (pred_cls == true_cls).sum().item(); n_time += pred_cls.numel()
        
        probs_np = probs.cpu().numpy(); y_np = y.cpu().numpy()
        B = probs_np.shape[0]
        uncertainty_np = np.zeros((B, 2))  # It only occupies space under the fixed threshold path and is used by the inner layer cur_unc_p/s
        if use_per_sample_threshold:
            suggested_thr_np = suggested_thr.cpu().numpy()  # [B, 2]
        else:
            suggested_thr_np = None
        for b in range(B):
            sample_thr_p = float(suggested_thr_np[b, 0]) if use_per_sample_threshold else (thr_p if thr_p is not None else 0.5)
            sample_thr_s = float(suggested_thr_np[b, 1]) if use_per_sample_threshold else (thr_s if thr_s is not None else 0.5)
            p_prob = probs_np[b, 1]
            s_prob = probs_np[b, 2]
            if use_per_sample_threshold:
                threshold_pass_counts["p"] += int(float(p_prob.max()) >= sample_thr_p)
                threshold_pass_counts["s"] += int(float(s_prob.max()) >= sample_thr_s)
                samples_processed += 1
            cur_unc_p = float(uncertainty_np[b, 0]) if use_per_sample_threshold else None
            cur_unc_s = float(uncertainty_np[b, 1]) if use_per_sample_threshold else None
            if structural_opts["enabled"]:
                p_candidates = _select_phase_peaks(
                    p_prob,
                    sample_thr_p,
                    structural_opts["min_interval_same"],
                    structural_opts["candidate_limit"],
                )
                p_candidates, removed_p_cap = apply_global_cap(p_candidates, p_prob, structural_opts["cap_p"])
                structural_stats["cap_p"] += removed_p_cap
            else:
                p_candidates = [int(np.argmax(p_prob))]
            candidate_lengths["p"].append(len(p_candidates))
            if len(p_candidates) == 0:
                zero_candidate_counts["p"] += 1
            p_pred = False; p_idx = int(np.argmax(p_prob)); p_conf = float(p_prob[p_idx])
            for idx in p_candidates:
                p_val = float(p_prob[idx])
                if p_val >= sample_thr_p:
                    p_pred = True
                    p_idx = int(idx)
                    p_conf = p_val
                    break
            if structural_opts["enabled"] and not p_pred:
                p_idx = int(np.argmax(p_prob))
                p_conf = float(p_prob[p_idx])
            if structural_opts["enabled"]:
                s_candidates = _select_phase_peaks(
                    s_prob,
                    sample_thr_s,
                    structural_opts["min_interval_same"],
                    structural_opts["candidate_limit"],
                )
                s_candidates = _apply_ps_spacing(s_candidates, [p_idx] if p_pred else [], structural_opts["min_interval_ps"])
                if structural_opts["enforce_p_before_s"]:
                    s_candidates, removed_order = enforce_p_before_s(
                        p_candidates, s_candidates, structural_opts["min_interval_ps"], p_prob, s_prob
                    )
                    structural_stats["order"] += removed_order
                s_candidates, removed_s_cap = apply_global_cap(s_candidates, s_prob, structural_opts["cap_s"])
                structural_stats["cap_s"] += removed_s_cap
            else:
                s_candidates = [int(np.argmax(s_prob))]
            candidate_lengths["s"].append(len(s_candidates))
            if len(s_candidates) == 0:
                zero_candidate_counts["s"] += 1
            s_pred = False; s_idx = int(np.argmax(s_prob)); s_conf = float(s_prob[s_idx])
            for idx in s_candidates:
                s_val = float(s_prob[idx])
                if s_val >= sample_thr_s:
                    s_pred = True
                    s_idx = int(idx)
                    s_conf = s_val
                    break
            if structural_opts["enabled"] and not s_pred:
                s_idx = int(np.argmax(s_prob))
                s_conf = float(s_prob[s_idx])
            has_p = bool(y_np[b, 1].max() > 1e-6)
            if has_p:
                gt = int(np.argmax(y_np[b, 1])); err = abs(p_idx - gt)
                all_p_residual_signed.append(p_idx - gt)
                if p_pred and p_conf >= sample_thr_p:
                    if err <= tol: p_tp += 1
                    else: p_fp += 1
                else:
                    p_fn += 1
            else:
                if p_pred and p_conf >= sample_thr_p:
                    p_fp += 1
                else:
                    p_tn += 1
            if use_per_sample_threshold and cur_unc_p is not None:
                if has_p:
                    if p_pred and p_conf >= sample_thr_p:
                        if err <= tol:
                            uncertainty_stats["p"]["tp"].append(cur_unc_p)
                        else:
                            uncertainty_stats["p"]["fp"].append(cur_unc_p)
                    else:
                        uncertainty_stats["p"]["fn"].append(cur_unc_p)
                else:
                    if p_pred and p_conf >= sample_thr_p:
                        uncertainty_stats["p"]["fp"].append(cur_unc_p)
            has_s = bool(y_np[b, 2].max() > 1e-6)
            if has_s:
                if use_per_sample_threshold and s_pred:
                    s_conf_list.append(s_conf)
                    s_thr_list.append(sample_thr_s)
                gt = int(np.argmax(y_np[b, 2])); err = abs(s_idx - gt)
                all_s_residual_signed.append(s_idx - gt)
                if s_pred and s_conf >= sample_thr_s:
                    if err <= tol: s_tp += 1
                    else: s_fp += 1
                else:
                    s_fn += 1
            else:
                if s_pred and s_conf >= sample_thr_s:
                    s_fp += 1
                else:
                    s_tn += 1
            if use_per_sample_threshold and cur_unc_s is not None:
                if has_s:
                    if s_pred and s_conf >= sample_thr_s:
                        if err <= tol:
                            uncertainty_stats["s"]["tp"].append(cur_unc_s)
                        else:
                            uncertainty_stats["s"]["fp"].append(cur_unc_s)
                    else:
                        uncertainty_stats["s"]["fn"].append(cur_unc_s)
                else:
                    if s_pred and s_conf >= sample_thr_s:
                        uncertainty_stats["s"]["fp"].append(cur_unc_s)
    acc = time_correct / max(1, n_time)
    if not quiet and candidate_lengths["p"]:
        def _print_len_stats(vals: list[int], tag: str):
            arr = np.array(vals, dtype=np.float32)
            q = np.quantile(arr, [0.1, 0.5, 0.9])
            print(
                f"[eval_detailed] {tag} candidate len: mean={arr.mean():.2f}, std={arr.std():.2f}, "
                f"q10={q[0]:.1f}, q50={q[1]:.1f}, q90={q[2]:.1f}",
                flush=True,
            )
        _print_len_stats(candidate_lengths["p"], "P")
        _print_len_stats(candidate_lengths["s"], "S")
        samples_total = len(candidate_lengths["p"])
        sample_denom = samples_total if samples_total else 1
        print(
            f"[eval_detailed] zero candidate rate: P={zero_candidate_counts['p']/sample_denom:.1%}, "
            f"S={zero_candidate_counts['s']/sample_denom:.1%}",
            flush=True,
        )
        pass_total = samples_processed if samples_processed else 1
        print(
            f"[eval_detailed] threshold pass counts (per sample tracked): P={threshold_pass_counts['p']}/{pass_total}, "
            f"S={threshold_pass_counts['s']}/{pass_total}",
            flush=True,
        )
        print(
            f"[eval_detailed] structural pruning: cap_p={structural_stats['cap_p']}, "
            f"cap_s={structural_stats['cap_s']}, order={structural_stats['order']}",
            flush=True,
        )
        for phase in ("p", "s"):
            stats = uncertainty_stats[phase]
            if stats["tp"] or stats["fp"] or stats["fn"]:
                for tag in ("tp", "fp", "fn"):
                    arr = np.array(stats[tag]) if stats[tag] else np.array([0.0])
                    print(
                        f"[eval_detailed] {phase.upper()} {tag} uncertainty: mean={arr.mean():.4f}, std={arr.std():.4f}",
            flush=True,
        )
    # Print diagnostics when using context and S-F1=0 to facilitate troubleshooting when thr_s is too high or s_c is too low
    if not quiet and use_per_sample_threshold and s_conf_list and (s_tp + s_fp) == 0:
        s_conf_arr = np.array(s_conf_list)
        s_thr_arr = np.array(s_thr_list)
        above = np.sum(s_conf_arr >= s_thr_arr)
        print(f"[eval_detailed] S-F1=0 诊断（有S样本）: S预测概率 mean={s_conf_arr.mean():.4f}, min={s_conf_arr.min():.4f}, max={s_conf_arr.max():.4f} | "
              f"预测S阈值 mean={s_thr_arr.mean():.4f}, min={s_thr_arr.min():.4f}, max={s_thr_arr.max():.4f} | "
              f"s_c>=thr_s 的样本数={above}/{len(s_conf_list)}", flush=True)
    def _prf(tp, fp, fn):
        p = tp / max(1, tp + fp); r = tp / max(1, tp + fn)
        f1 = 0.0 if (p + r) == 0 else 2 * p * r / (p + r)
        return p, r, f1
    p_prec, p_rec, p_f1 = _prf(p_tp, p_fp, p_fn); s_prec, s_rec, s_f1 = _prf(s_tp, s_fp, s_fn)
    # MCC (sample-level binary classification: whether any phase is correctly predicted on the sample), consistent with the definition in baselines_run
    def _mcc(tp: int, fp: int, fn: int, tn: int) -> float:
        denom = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        if denom <= 0:
            return 0.0
        return float((tp * tn - fp * fn) / math.sqrt(denom))
    # Summarize TP/FP/FN/TN of P/S as "whether P or S was correctly predicted on the sample"
    tp_all = p_tp + s_tp
    fp_all = p_fp + s_fp
    fn_all = p_fn + s_fn
    tn_all = p_tn + s_tn
    mcc = _mcc(tp_all, fp_all, fn_all, tn_all)
    # Calculation time residual statistics (unit: seconds)
    if all_p_residual_signed:
        p_res_arr = np.asarray(all_p_residual_signed, dtype=np.float32) / float(SAMPLE_RATE)
        p_res_mean = float(p_res_arr.mean())
        p_res_std = float(p_res_arr.std())
        p_res_mae = float(np.abs(p_res_arr).mean())
    else:
        p_res_mean = p_res_std = p_res_mae = None
    if all_s_residual_signed:
        s_res_arr = np.asarray(all_s_residual_signed, dtype=np.float32) / float(SAMPLE_RATE)
        s_res_mean = float(s_res_arr.mean())
        s_res_std = float(s_res_arr.std())
        s_res_mae = float(np.abs(s_res_arr).mean())
    else:
        s_res_mean = s_res_std = s_res_mae = None
    return dict(
        time_acc=float(acc),
        mcc=float(mcc),
        p_prec=float(p_prec),
        p_rec=float(p_rec),
        p_f1=float(p_f1),
        s_prec=float(s_prec),
        s_rec=float(s_rec),
        s_f1=float(s_f1),
        # Time residual statistics (seconds)
        p_res_mean_sec=p_res_mean,
        p_res_std_sec=p_res_std,
        p_res_mae_sec=p_res_mae,
        s_res_mean_sec=s_res_mean,
        s_res_std_sec=s_res_std,
        s_res_mae_sec=s_res_mae,
        p_residuals_signed=all_p_residual_signed if all_p_residual_signed else None,
        s_residuals_signed=all_s_residual_signed if all_s_residual_signed else None,
    )


# MC Dropout / Selective Prediction related branches have been removed (currently only the deterministic evaluation process remains).


def collect_pca_features_and_labels(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Collect window-level features and labels for PCA.

    - Tags: 0=BG-window, 1=P-window, 2=S-window (based on GT P/S or its neighborhood)
    - Sampling: up to N windows (default 1000) are equally sampled for each category, and the BG window is far away from any P/S
    - Feature: do mean/std pooling on softmax probability within the window to obtain a concise representation of 3×2 dimensions
    """
    model.eval()
    feats: list[list[float]] = []
    labs: list[int] = []
    max_per_class = 1000
    counts = {0: 0, 1: 0, 2: 0}
    win = 50  # Half window length (in time steps)

    def _add_feat(feat: np.ndarray, label: int) -> None:
        if counts[label] >= max_per_class:
            return
        feats.append(feat.astype(np.float32).tolist())
        labs.append(label)
        counts[label] += 1

    with torch.no_grad():
        for x, y, _ in loader:
            # If all three categories have reached the upper limit, it can be ended early.
            if all(c >= max_per_class for c in counts.values()):
                break
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            if y.shape[-1] != logits.shape[-1]:
                y = _center_crop_time(y, logits.shape[-1])
            probs = torch.softmax(logits, dim=1).cpu().numpy()  # [B, 3, T]
            y_np = y.cpu().numpy()  # [B, 3, T]
            B, C, T = probs.shape
            for b in range(B):
                if all(c >= max_per_class for c in counts.values()):
                    break
                prob_ch = probs[b]  # [3, T]
                y_b = y_np[b]
                has_p = bool(y_b[1].max() > 1e-6)
                has_s = bool(y_b[2].max() > 1e-6)

                def _window_feat(center: int) -> np.ndarray:
                    lo = max(0, center - win)
                    hi = min(T, center + win + 1)
                    w = prob_ch[:, lo:hi]  # [3, L]
                    if w.shape[1] == 0:
                        return np.zeros(6, dtype=np.float32)
                    mean = w.mean(axis=1)
                    std = w.std(axis=1) + 1e-8
                    return np.concatenate([mean, std], axis=0)

                # P-window
                if has_p and counts[1] < max_per_class:
                    p_idx = int(np.argmax(y_b[1]))
                    feat_p = _window_feat(p_idx)
                    _add_feat(feat_p, 1)

                # S-window
                if has_s and counts[2] < max_per_class:
                    s_idx = int(np.argmax(y_b[2]))
                    feat_s = _window_feat(s_idx)
                    _add_feat(feat_s, 2)

                # BG-window: away from any P/S truth position
                if counts[0] < max_per_class:
                    bg_mask = (y_b[1] < 1e-6) & (y_b[2] < 1e-6)
                    cand_idx = np.where(bg_mask)[0]
                    if cand_idx.size > 0:
                        if has_p:
                            p_idx = int(np.argmax(y_b[1]))
                            cand_idx = cand_idx[np.abs(cand_idx - p_idx) >= win]
                        if has_s and cand_idx.size > 0:
                            s_idx = int(np.argmax(y_b[2]))
                            cand_idx = cand_idx[np.abs(cand_idx - s_idx) >= win]
                    if cand_idx.size > 0:
                        center = int(np.random.default_rng(2024).choice(cand_idx))
                        feat_bg = _window_feat(center)
                        _add_feat(feat_bg, 0)

    if not feats:
        return np.empty((0, 6), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return np.asarray(feats, dtype=np.float32), np.asarray(labs, dtype=np.int64)


def collect_pr_snr_data(
    model,
    loader,
    device,
    tol_samples: int = DYN_TOL_SAMPLES,
) -> dict:
    """Collect data required for PR curves and performance stratified by SNR: has_p, has_s, max_prob_p, max_prob_s, p_ok, s_ok, snr."""
    model.eval()
    has_p, has_s = [], []
    max_prob_p, max_prob_s = [], []
    p_ok, s_ok = [], []
    snr_list = []
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            if y.shape[-1] != logits.shape[-1]:
                y = _center_crop_time(y, logits.shape[-1])
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            y_np = y.cpu().numpy()
            x_np = x.cpu().numpy()
            B = probs.shape[0]
            for b in range(B):
                prob_p = probs[b, 1]
                prob_s = probs[b, 2]
                mp_p = float(np.max(prob_p))
                mp_s = float(np.max(prob_s))
                p_idx = int(np.argmax(prob_p))
                s_idx = int(np.argmax(prob_s))
                hp = bool(y_np[b, 1].max() > 1e-6)
                hs = bool(y_np[b, 2].max() > 1e-6)
                p_gt = int(np.argmax(y_np[b, 1])) if hp else None
                s_gt = int(np.argmax(y_np[b, 2])) if hs else None
                po = hp and p_gt is not None and abs(p_idx - p_gt) <= tol_samples
                so = hs and s_gt is not None and abs(s_idx - s_gt) <= tol_samples
                snr_val = estimate_snr(x_np[b], p_idx=p_gt)
                has_p.append(hp)
                has_s.append(hs)
                max_prob_p.append(mp_p)
                max_prob_s.append(mp_s)
                p_ok.append(po)
                s_ok.append(so)
                snr_list.append(snr_val)
    return {
        "has_p": has_p,
        "has_s": has_s,
        "max_prob_p": max_prob_p,
        "max_prob_s": max_prob_s,
        "p_ok": p_ok,
        "s_ok": s_ok,
        "snr": snr_list,
    }


@torch.inference_mode()
def save_visuals(model, dataset, device, out_dir: str, n: int = 4):
    os.makedirs(out_dir, exist_ok=True)
    idxs = list(range(len(dataset)))
    random.seed(SEED)
    random.shuffle(idxs)
    idxs = idxs[:n]
    for idx in idxs:
        x_t, y_t, name = dataset[idx]
        x = x_t.unsqueeze(0).to(device)
        y = y_t.unsqueeze(0).to(device)
        logits = model(x)
        if y.shape[-1] != logits.shape[-1]:
            y = _center_crop_time(y, logits.shape[-1])
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        y_np = y.squeeze(0).cpu().numpy()
        x_np = x_t.cpu().numpy()
        p_idx = int(np.argmax(y_np[1])) if y_np[1].max() > 1e-6 else None
        snr = estimate_snr(x_np, p_idx=p_idx)
        plot_one_sample_visual(x_np, y_np, probs, name, snr, os.path.join(out_dir, f"vis_{idx}.png"))


@torch.inference_mode()
def save_representative_waveforms_2x2(
    model: nn.Module,
    valid_loader: DataLoader,
    device: torch.device,
    out_path: str,
    tol: int,
    thr_p: float = 0.5,
    thr_s: float = 0.5,
    max_scan_samples: int = 2000,
    snr_high: float = 15.0,
    snr_low: float = 5.0,
    ch_missing_std_eps: float = 1e-6,
    ch_missing_max_eps: float = 1e-6,
    extra_loaders: list[DataLoader] | None = None,
    rep_npz_path: str | None = None,
) -> None:
    """
    Automatically filter 4 types of representative samples from the validation set and output a 2×2 grid plot:
    (a) Normal SNR: SNR≥snr_high, P/S are picked up correctly
    (b) Low SNR: snr_low≤SNR<snr_high, P/S are picked up correctly
    (c) Very low SNR: SNR<snr_low, P/S are picked up correctly
    (d) Channel missing: It is detected that at least one channel is approximately all 0 (std/max is very small); if it is not found, use Normal samples to synthesize a missing channel version.

    Note: To avoid scanning the full validation set, check max_scan_samples at most.
    """
    if plt is None:
        return

    model.eval()

    def _has_phase(y_np_b: np.ndarray, cls: int) -> bool:
        return bool(y_np_b[cls].max() > 1e-6)

    def _phase_gt_idx(y_np_b: np.ndarray, cls: int) -> int:
        return int(np.argmax(y_np_b[cls]))

    def _phase_pred_idx_conf(prob_b: np.ndarray, cls: int) -> tuple[int, float]:
        idx = int(np.argmax(prob_b[cls]))
        return idx, float(prob_b[cls, idx])

    def _channel_missing_mask(x_np_b: np.ndarray) -> np.ndarray:
        # x_np_b: [C, T]
        stds = np.std(x_np_b, axis=1)
        maxabs = np.max(np.abs(x_np_b), axis=1)
        return (stds <= ch_missing_std_eps) | (maxabs <= ch_missing_max_eps)

    # ---------- Batch 1: for 2×2 plots ----------
    chosen: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, str, float]] = {}
    chosen_name: dict[str, str] = {}
    seen_names: set[str] = set()
    scanned = 0

    # Representative samples can be scanned from multiple DataLoaders in sequence: valid first, then train, then test
    loaders: list[DataLoader] = [valid_loader]
    if extra_loaders:
        loaders.extend(extra_loaders)

    for loader in loaders:
        for x, y, names in loader:
            if scanned >= max_scan_samples:
                break
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            if y.shape[-1] != logits.shape[-1]:
                y = _center_crop_time(y, logits.shape[-1])
            probs = torch.softmax(logits, dim=1)
            x_np = x.detach().cpu().numpy()
            y_np = y.detach().cpu().numpy()
            probs_np = probs.detach().cpu().numpy()
            B = x_np.shape[0]

            for b in range(B):
                if scanned >= max_scan_samples:
                    break
                scanned += 1
                name_b = str(names[b]) if hasattr(names, "__getitem__") else str(scanned)
                if name_b in seen_names:
                    continue

                # There must be both true values ​​of P and S (the example diagram is more intuitive)
                has_p = _has_phase(y_np[b], 1)
                has_s = _has_phase(y_np[b], 2)
                if not (has_p and has_s):
                    continue

                # Estimate SNR (using P ground truth positions)
                p_gt = _phase_gt_idx(y_np[b], 1)
                snr_b = float(estimate_snr(x_np[b], p_idx=p_gt))

                # Whether the prediction "passes the threshold" and is correct within a tolerance (with argmax + fixed thr)
                p_pred_idx, p_conf = _phase_pred_idx_conf(probs_np[b], 1)
                s_pred_idx, s_conf = _phase_pred_idx_conf(probs_np[b], 2)
                p_pass = p_conf >= float(thr_p)
                s_pass = s_conf >= float(thr_s)
                p_ok = p_pass and (abs(p_pred_idx - p_gt) <= int(tol))
                s_gt = _phase_gt_idx(y_np[b], 2)
                s_ok = s_pass and (abs(s_pred_idx - s_gt) <= int(tol))

                # Channel missing detection (at least one channel is approximately all 0s)
                miss_mask = _channel_missing_mask(x_np[b])
                has_missing = bool(np.any(miss_mask))

                # Choose the missing channel first (choose the right one first)
                if "missing" not in chosen and has_missing and p_ok and s_ok:
                    chosen["missing"] = (x_np[b], y_np[b], probs_np[b], "Channel missing", snr_b)
                    chosen_name["missing"] = name_b
                    seen_names.add(name_b)
                    continue

                # Normal/Low/Very low: Give priority to samples with P/S OK
                if p_ok and s_ok:
                    if snr_b >= snr_high and "normal" not in chosen:
                        chosen["normal"] = (x_np[b], y_np[b], probs_np[b], "Normal SNR", snr_b)
                        chosen_name["normal"] = name_b
                        seen_names.add(name_b)
                        continue
                    if snr_low <= snr_b < snr_high and "low" not in chosen:
                        chosen["low"] = (x_np[b], y_np[b], probs_np[b], "Low SNR", snr_b)
                        chosen_name["low"] = name_b
                        seen_names.add(name_b)
                        continue
                    if snr_b < snr_low and "very_low" not in chosen:
                        chosen["very_low"] = (x_np[b], y_np[b], probs_np[b], "Very low SNR", snr_b)
                        chosen_name["very_low"] = name_b
                        seen_names.add(name_b)
                        continue

            if len(chosen) >= 4 and all(k in chosen for k in ("normal", "low", "very_low", "missing")):
                break
        if len(chosen) >= 4 and all(k in chosen for k in ("normal", "low", "very_low", "missing")):
            break

    # If there are no real missing channel samples, use normal samples to synthesize the missing channel version to get the bottom of things.
    if "missing" not in chosen and "normal" in chosen:
        x0, y0, _p0, _t0, _snr0 = chosen["normal"]
        x_syn = np.array(x0, copy=True)
        # Keep CH0 and set the rest to 0 (the horizontal component of the simulated station is invalid)
        if x_syn.shape[0] >= 3:
            x_syn[1, :] = 0.0
            x_syn[2, :] = 0.0
        elif x_syn.shape[0] >= 2:
            x_syn[1, :] = 0.0
        x_t = torch.from_numpy(x_syn).unsqueeze(0).to(device)
        logits = model(x_t)
        probs_syn = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
        p_gt0 = int(np.argmax(y0[1])) if y0[1].max() > 1e-6 else None
        snr_syn = float(estimate_snr(x_syn, p_idx=p_gt0))
        chosen["missing"] = (x_syn, y0, probs_syn, "Channel missing", snr_syn)
        # The synthesized missing channel sample does not have a clear original name, and a placeholder can be used
        chosen_name.setdefault("missing", "synthetic_missing_from_" + chosen_name.get("normal", "unknown"))

    # Assemble 4 pictures (if a certain category is not found, it will be degenerated and filled with existing samples to ensure that it does not collapse)
    def _fallback_pick(keys: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, float] | None:
        for k in keys:
            if k in chosen:
                return chosen[k]
        return None

    s_normal = _fallback_pick(["normal", "low", "very_low", "missing"])
    s_low = _fallback_pick(["low", "normal", "very_low", "missing"])
    s_very = _fallback_pick(["very_low", "low", "normal", "missing"])
    s_miss = _fallback_pick(["missing", "normal", "low", "very_low"])

    if not (s_normal and s_low and s_very and s_miss):
        return

    samples = [s_normal, s_low, s_very, s_miss]
    plot_representative_waveforms_grid(samples, out_path=out_path)

    # ---------- Batch 2: used for representative_compare.npz to avoid duplication of name with batch 1 ----------
    if rep_npz_path:
        try:
            exclude_names = set(chosen_name.values())

            chosen2: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, str, float]] = {}
            seen_names2: set[str] = set(exclude_names)
            scanned2 = 0

            for x, y, names in valid_loader:
                if scanned2 >= max_scan_samples:
                    break
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                if y.shape[-1] != logits.shape[-1]:
                    y = _center_crop_time(y, logits.shape[-1])
                probs = torch.softmax(logits, dim=1)
                x_np = x.detach().cpu().numpy()
                y_np = y.detach().cpu().numpy()
                probs_np = probs.detach().cpu().numpy()
                B = x_np.shape[0]

                for b in range(B):
                    if scanned2 >= max_scan_samples:
                        break
                    scanned2 += 1
                    name_b = str(names[b]) if hasattr(names, "__getitem__") else f"sample_{scanned2}"
                    if name_b in seen_names2:
                        continue

                    # There must be truth values ​​for both P and S
                    has_p = _has_phase(y_np[b], 1)
                    has_s = _has_phase(y_np[b], 2)
                    if not (has_p and has_s):
                        continue

                    p_gt = _phase_gt_idx(y_np[b], 1)
                    snr_b = float(estimate_snr(x_np[b], p_idx=p_gt))

                    p_pred_idx, p_conf = _phase_pred_idx_conf(probs_np[b], 1)
                    s_pred_idx, s_conf = _phase_pred_idx_conf(probs_np[b], 2)
                    p_pass = p_conf >= float(thr_p)
                    s_pass = s_conf >= float(thr_s)
                    p_ok = p_pass and (abs(p_pred_idx - p_gt) <= int(tol))
                    s_gt = _phase_gt_idx(y_np[b], 2)
                    s_ok = s_pass and (abs(s_pred_idx - s_gt) <= int(tol))

                    miss_mask = _channel_missing_mask(x_np[b])
                    has_missing = bool(np.any(miss_mask))

                    if "missing" not in chosen2 and has_missing and p_ok and s_ok:
                        chosen2["missing"] = (x_np[b], y_np[b], probs_np[b], "Channel missing", snr_b)
                        seen_names2.add(name_b)
                        continue

                    if p_ok and s_ok:
                        if snr_b >= snr_high and "normal" not in chosen2:
                            chosen2["normal"] = (x_np[b], y_np[b], probs_np[b], "Normal SNR", snr_b)
                            seen_names2.add(name_b)
                            continue
                        if snr_low <= snr_b < snr_high and "low" not in chosen2:
                            chosen2["low"] = (x_np[b], y_np[b], probs_np[b], "Low SNR", snr_b)
                            seen_names2.add(name_b)
                            continue
                        if snr_b < snr_low and "very_low" not in chosen2:
                            chosen2["very_low"] = (x_np[b], y_np[b], probs_np[b], "Very low SNR", snr_b)
                            seen_names2.add(name_b)
                            continue

                if len(chosen2) >= 4 and all(k in chosen2 for k in ("normal", "low", "very_low", "missing")):
                    break

            # Do the same missing channel synthesis at the bottom of the pocket (if the real missing is still not found)
            if "missing" not in chosen2 and "normal" in chosen2:
                x0, y0, _p0, _t0, _snr0 = chosen2["normal"]
                x_syn = np.array(x0, copy=True)
                if x_syn.shape[0] >= 3:
                    x_syn[1, :] = 0.0
                    x_syn[2, :] = 0.0
                elif x_syn.shape[0] >= 2:
                    x_syn[1, :] = 0.0
                x_t = torch.from_numpy(x_syn).unsqueeze(0).to(device)
                logits = model(x_t)
                probs_syn = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
                p_gt0 = int(np.argmax(y0[1])) if y0[1].max() > 1e-6 else None
                snr_syn = float(estimate_snr(x_syn, p_idx=p_gt0))
                chosen2["missing"] = (x_syn, y0, probs_syn, "Channel missing", snr_syn)

            def _fallback_pick2(keys: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, float] | None:
                for k in keys:
                    if k in chosen2:
                        return chosen2[k]
                return None

            s_normal2 = _fallback_pick2(["normal", "low", "very_low", "missing"])
            s_low2 = _fallback_pick2(["low", "normal", "very_low", "missing"])
            s_very2 = _fallback_pick2(["very_low", "low", "normal", "missing"])
            s_miss2 = _fallback_pick2(["missing", "normal", "low", "very_low"])

            if not (s_normal2 and s_low2 and s_very2 and s_miss2):
                print("[save_representative_waveforms_2x2] representative_compare.npz: 未能找到完整的第二批样本，跳过导出", flush=True)
                return

            samples2 = [s_normal2, s_low2, s_very2, s_miss2]
            x_list = np.stack([s[0] for s in samples2], axis=0)  # [4, C, T]
            y_list = np.stack([s[1] for s in samples2], axis=0)  # [4, 3, T]
            snr_list = np.array([float(s[4]) for s in samples2], dtype=float)
            name_list = np.array(["normal", "low", "very_low", "missing"], dtype=object)
            np.savez(
                rep_npz_path,
                x_list=x_list,
                y_list=y_list,
                snr_list=snr_list,
                names=name_list,
            )
            print(f"[save_representative_waveforms_2x2] 代表性样本（第二批）已保存: {rep_npz_path}", flush=True)
        except Exception as e:
            print(f"[save_representative_waveforms_2x2] 保存 representative_compare.npz 失败: {e}", flush=True)


def plot_dynamic_threshold_pipeline(
    model,
    loader,
    device,
    out_dir: str,
    top_k_exceptions: int = 5,
    uncertainty_mode: str = "entropy",
    uncertainty_threshold_options: dict | None = None,
    tol_samples: int = DYN_TOL_SAMPLES,
):
    # Uncertainty threshold/dynamic threshold visualization has been removed (current version only retains fixed threshold and P/S grid search calibration).
    _ = (
        model,
        loader,
        device,
        out_dir,
        top_k_exceptions,
        uncertainty_mode,
        uncertainty_threshold_options,
        tol_samples,
    )
    return

    thr_opts = _get_uncertainty_threshold_kwargs(uncertainty_threshold_options)
    time_win = thr_opts.get("time_window", 0)
    aggregate = thr_opts.get("aggregate", "mean")
    threshold_kwargs = {k: v for k, v in thr_opts.items() if k not in ("time_window", "aggregate")}
    
    thresholds_p, thresholds_s = [], []
    snrs, uncertainties_p, uncertainties_s = [], [], []
    errors_p, errors_s = [], []  # "TP"/"FP"/"FN"/"TN"
    deviations_p, deviations_s = [], []
    sample_indices = []
    # Only save abnormal sample waveform data (FP/FN) to save memory
    waveforms_list, probs_p_list, probs_s_list, labels_p_list, labels_s_list = [], [], [], [], []
    thresholds_p_list, thresholds_s_list = [], []
    exception_indices = []  # Record the location of the abnormal sample in sample_indices
    
    with torch.no_grad():
        opts = uncertainty_threshold_options or {}
        stable_dynamic = bool(opts.get("stable_dynamic", False))
        stable_state: dict | None = None
        batch_offset = 0
        for batch_idx, (x, y, _) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)
            if y.shape[-1] != x.shape[-1]:
                y = _center_crop_time(y, x.shape[-1])
            
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            
            # Calculation Uncertainty
            fusion_alpha = float((uncertainty_threshold_options or {}).get("fusion_alpha", 0.5))
            use_phase_channels = bool((uncertainty_threshold_options or {}).get("use_phase_channels", True))
            uncertainty = compute_uncertainty_from_probs(
                probs, mode=uncertainty_mode, time_window=time_win, aggregate=aggregate,
                fusion_alpha=fusion_alpha, use_phase_channels=use_phase_channels,
            )
            if uncertainty.dim() == 3:
                uncertainty = uncertainty.mean(dim=-1)
            
            # Dynamic threshold calculation
            if stable_dynamic:
                suggested_thr, stable_state = _stable_dynamic_threshold(
                    uncertainty, stable_state, opts
                )
            else:
                suggested_thr = uncertainty_to_threshold(uncertainty, **threshold_kwargs)
            thr_p_batch = suggested_thr[:, 0].cpu().numpy()
            thr_s_batch = suggested_thr[:, 1].cpu().numpy()
            unc_batch = uncertainty.cpu().numpy()  # [B, 2]
            
            # Process each sample
            B = x.shape[0]
            x_np = x.cpu().numpy()
            y_np = y.cpu().numpy()
            probs_np = probs.cpu().numpy()
            
            for b in range(B):
                prob_p = float(probs_np[b, 1, :].max())
                prob_s = float(probs_np[b, 2, :].max())
                p_idx = int(np.argmax(probs_np[b, 1, :]))
                s_idx = int(np.argmax(probs_np[b, 2, :]))
                has_p = bool(y_np[b, 1].max() > 1e-6)
                has_s = bool(y_np[b, 2].max() > 1e-6)
                p_gt = int(np.argmax(y_np[b, 1])) if has_p else None
                s_gt = int(np.argmax(y_np[b, 2])) if has_s else None
                p_ok = has_p and p_gt is not None and abs(p_idx - p_gt) <= tol_samples
                s_ok = has_s and s_gt is not None and abs(s_idx - s_gt) <= tol_samples
                
                pass_p = prob_p >= thr_p_batch[b]
                pass_s = prob_s >= thr_s_batch[b]
                p_err = _compute_error_label(pass_p, p_ok, has_p)
                s_err = _compute_error_label(pass_s, s_ok, has_s)
                
                snr_val = estimate_snr(x_np[b], p_idx=p_gt)
                
                thresholds_p.append(thr_p_batch[b])
                thresholds_s.append(thr_s_batch[b])
                snrs.append(snr_val)
                uncertainties_p.append(float(unc_batch[b, 0]))
                uncertainties_s.append(float(unc_batch[b, 1]))
                errors_p.append(p_err)
                errors_s.append(s_err)
                sample_idx = batch_offset + b
                sample_indices.append(sample_idx)
                
                # Only save abnormal sample data (FP/FN) to save memory
                if p_err in ("FP", "FN") or s_err in ("FP", "FN"):
                    waveforms_list.append(x_np[b].copy())
                    probs_p_list.append(probs_np[b, 1, :].copy())
                    probs_s_list.append(probs_np[b, 2, :].copy())
                    labels_p_list.append(y_np[b, 1, :].copy())
                    labels_s_list.append(y_np[b, 2, :].copy())
                    thresholds_p_list.append(thr_p_batch[b])
                    thresholds_s_list.append(thr_s_batch[b])
                    # Record the position of the abnormal sample in the original sample_indices (used for subsequent index mapping)
                    exception_indices.append(len(sample_indices) - 1)
            
            batch_offset += B
    
    if len(thresholds_p) == 0:
        print("[plot_dynamic_threshold_pipeline] 没有收集到数据，跳过", flush=True)
        return
    
    # Convert to numpy array
    thresholds_p = np.array(thresholds_p)
    thresholds_s = np.array(thresholds_s)
    snrs = np.array(snrs)
    uncertainties_p = np.array(uncertainties_p)
    uncertainties_s = np.array(uncertainties_s)
    errors_p = np.array(errors_p)
    errors_s = np.array(errors_s)
    sample_indices = np.array(sample_indices)
    
    # Calculate the deviation from the threshold (relative to the median)
    median_thr_p = np.median(thresholds_p)
    median_thr_s = np.median(thresholds_s)
    deviations_p = np.abs(thresholds_p - median_thr_p)
    deviations_s = np.abs(thresholds_s - median_thr_s)
    
    # Drawing
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Threshold vs Uncertainty
    ax0 = axes[0]
    # P wave
    tp_mask_p = errors_p == "TP"
    fp_mask_p = errors_p == "FP"
    fn_mask_p = errors_p == "FN"
    if tp_mask_p.any():
        ax0.scatter(uncertainties_p[tp_mask_p], thresholds_p[tp_mask_p], s=deviations_p[tp_mask_p] * 100 + 10,
                   c="gray", alpha=0.3, label="P TP")
    if fp_mask_p.any():
        ax0.scatter(uncertainties_p[fp_mask_p], thresholds_p[fp_mask_p], s=deviations_p[fp_mask_p] * 100 + 10,
                   c="red", alpha=0.8, label="P FP")
    if fn_mask_p.any():
        ax0.scatter(uncertainties_p[fn_mask_p], thresholds_p[fn_mask_p], s=deviations_p[fn_mask_p] * 100 + 10,
                   c="orange", alpha=0.8, label="P FN")
    # S wave
    tp_mask_s = errors_s == "TP"
    fp_mask_s = errors_s == "FP"
    fn_mask_s = errors_s == "FN"
    if tp_mask_s.any():
        ax0.scatter(uncertainties_s[tp_mask_s], thresholds_s[tp_mask_s], s=deviations_s[tp_mask_s] * 100 + 10,
                   c="gray", alpha=0.3, marker="x", label="S TP")
    if fp_mask_s.any():
        ax0.scatter(uncertainties_s[fp_mask_s], thresholds_s[fp_mask_s], s=deviations_s[fp_mask_s] * 100 + 10,
                   c="red", alpha=0.8, marker="x", label="S FP")
    if fn_mask_s.any():
        ax0.scatter(uncertainties_s[fn_mask_s], thresholds_s[fn_mask_s], s=deviations_s[fn_mask_s] * 100 + 10,
                   c="orange", alpha=0.8, marker="x", label="S FN")
    
    ax0.set_xlabel("Uncertainty")
    ax0.set_ylabel("Dynamic Threshold")
    ax0.set_title("Threshold vs Uncertainty")
    ax0.legend(loc="upper right", fontsize=9)
    ax0.grid(True, alpha=0.3)
    
    # Label abnormal samples (FP/FN maximum deviation)
    for wave_type, thr, err, dev, unc_arr, idx_arr in zip(
        ["P", "S"], [thresholds_p, thresholds_s], [errors_p, errors_s],
        [deviations_p, deviations_s], [uncertainties_p, uncertainties_s], [sample_indices, sample_indices]
    ):
        for etype in ["FP", "FN"]:
            mask = err == etype
            if mask.sum() > 0:
                top_k = min(top_k_exceptions, mask.sum())
                top_idx = np.argsort(dev[mask])[-top_k:]
                for i in np.where(mask)[0][top_idx]:
                    ax0.annotate(f"{wave_type}{idx_arr[i]}", 
                               (unc_arr[i], thr[i]), 
                               color="red" if etype == "FP" else "orange",
                               fontsize=7, alpha=0.8)
    
    # Right: Threshold vs SNR
    ax1 = axes[1]
    if tp_mask_p.any():
        ax1.scatter(snrs[tp_mask_p], thresholds_p[tp_mask_p], s=deviations_p[tp_mask_p] * 100 + 10,
                   c="gray", alpha=0.3, label="P TP")
    if fp_mask_p.any():
        ax1.scatter(snrs[fp_mask_p], thresholds_p[fp_mask_p], s=deviations_p[fp_mask_p] * 100 + 10,
                   c="red", alpha=0.8, label="P FP")
    if fn_mask_p.any():
        ax1.scatter(snrs[fn_mask_p], thresholds_p[fn_mask_p], s=deviations_p[fn_mask_p] * 100 + 10,
                   c="orange", alpha=0.8, label="P FN")
    if tp_mask_s.any():
        ax1.scatter(snrs[tp_mask_s], thresholds_s[tp_mask_s], s=deviations_s[tp_mask_s] * 100 + 10,
                   c="gray", alpha=0.3, marker="x", label="S TP")
    if fp_mask_s.any():
        ax1.scatter(snrs[fp_mask_s], thresholds_s[fp_mask_s], s=deviations_s[fp_mask_s] * 100 + 10,
                   c="red", alpha=0.8, marker="x", label="S FP")
    if fn_mask_s.any():
        ax1.scatter(snrs[fn_mask_s], thresholds_s[fn_mask_s], s=deviations_s[fn_mask_s] * 100 + 10,
                   c="orange", alpha=0.8, marker="x", label="S FN")
    
    ax1.set_xlabel("SNR")
    ax1.set_ylabel("Dynamic Threshold")
    ax1.set_title("Threshold vs SNR (highlight FP/FN)")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Label abnormal samples (right picture)
    for wave_type, thr, err, dev, idx_arr in zip(
        ["P", "S"], [thresholds_p, thresholds_s], [errors_p, errors_s],
        [deviations_p, deviations_s], [sample_indices, sample_indices]
    ):
        for etype in ["FP", "FN"]:
            mask = err == etype
            if mask.sum() > 0:
                top_k = min(top_k_exceptions, mask.sum())
                top_idx = np.argsort(dev[mask])[-top_k:]
                for i in np.where(mask)[0][top_idx]:
                    ax1.annotate(f"{wave_type}{idx_arr[i]}", 
                               (snrs[i], thr[i]), 
                               color="red" if etype == "FP" else "orange",
                               fontsize=7, alpha=0.8)
    
    plt.tight_layout()
    out_path = os.path.join(out_dir, "dynamic_threshold_pipeline.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[INFO] 论文级动态阈值图已生成：{out_path}", flush=True)
    
    # -----------------------------
    # 4️⃣ Automatically extract and draw FP/FN maximum abnormal sample waveform + threshold curve
    # -----------------------------
    if len(waveforms_list) == 0:
        print(f"[INFO] 没有异常样本（FP/FN），跳过异常样本波形图", flush=True)
    else:
        waveforms_arr = np.array(waveforms_list)
        probs_p_arr = np.array(probs_p_list)
        probs_s_arr = np.array(probs_s_list)
        labels_p_arr = np.array(labels_p_list)
        labels_s_arr = np.array(labels_s_list)
        thresholds_p_arr = np.array(thresholds_p_list)
        thresholds_s_arr = np.array(thresholds_s_list)
        exception_indices_arr = np.array(exception_indices)  # The position of the abnormal sample in sample_indices
        
        # Find top-k FP/FN sample indices (sorted by deviation)
        # Note: errors_p/errors_s and deviations_p/deviations_s are for all samples
        # exception_indices_arr records which samples are abnormal (position in sample_indices)
        for wave_type, err_arr_all, dev_arr_all, probs_arr, labels_arr, thresholds_arr in zip(
            ["P", "S"], [errors_p, errors_s], [deviations_p, deviations_s],
            [probs_p_arr, probs_s_arr], [labels_p_arr, labels_s_arr],
            [thresholds_p_arr, thresholds_s_arr]
        ):
            for etype in ["FP", "FN"]:
                # Find abnormal samples of this type among all samples
                mask_all = err_arr_all == etype
                if mask_all.sum() == 0:
                    continue
                
                # Find the location of these exception samples in exception_indices
                exception_mask = np.isin(exception_indices_arr, np.where(mask_all)[0])
                if exception_mask.sum() == 0:
                    continue
                
                # Get the deviation amount of abnormal samples (taken from dev_arr_all of all samples)
                exception_orig_indices = exception_indices_arr[exception_mask]
                dev_exceptions = dev_arr_all[exception_orig_indices]
                
                top_k = min(top_k_exceptions, len(dev_exceptions))
                top_local_indices = np.argsort(dev_exceptions)[-top_k:]
                selected_exception_indices = np.where(exception_mask)[0][top_local_indices]
                
                for rank, exc_idx in enumerate(selected_exception_indices):
                    orig_idx = sample_indices[exception_indices_arr[exc_idx]]
                    waveform = waveforms_arr[exc_idx]
                    prob_curve = probs_arr[exc_idx]
                    label_curve = labels_arr[exc_idx]
                    threshold_val = thresholds_arr[exc_idx]
                    deviation_val = dev_arr_all[exception_indices_arr[exc_idx]]
                    
                    # Plot waveform + probability curve + threshold line (using dual y-axis)
                    fig, ax1 = plt.subplots(1, 1, figsize=(12, 4))
                    T_prob = len(prob_curve)
                    # Waveform (use first channel, or average all channels)
                    if waveform.ndim == 2:
                        waveform_plot = waveform[0, :] if waveform.shape[0] > 0 else waveform.mean(axis=0)
                    else:
                        waveform_plot = np.asarray(waveform).flatten()
                    # Alignment length: The model output may be different from the input length due to downsampling. Take the minimum length and crop it.
                    T = min(len(waveform_plot), T_prob)
                    t = np.arange(T)
                    waveform_plot = waveform_plot[:T]
                    prob_curve = prob_curve[:T]
                    label_curve = label_curve[:T]
                    
                    # Left y-axis: Waveform
                    color_wave = "black"
                    ax1.set_xlabel("Time (samples)")
                    ax1.set_ylabel("Waveform Amplitude", color=color_wave)
                    ax1.plot(t, waveform_plot, color=color_wave, alpha=0.6, linewidth=0.8, label="Waveform")
                    ax1.tick_params(axis="y", labelcolor=color_wave)
                    ax1.grid(True, alpha=0.3)
                    
                    # Right y-axis: probability curve and threshold
                    ax2 = ax1.twinx()
                    color_prob = "blue"
                    color_thr = "red"
                    ax2.set_ylabel("Probability / Threshold", color=color_prob)
                    ax2.plot(t, prob_curve, color=color_prob, alpha=0.8, linewidth=1.5, label=f"{wave_type} Prediction Prob")
                    ax2.axhline(y=threshold_val, color=color_thr, linestyle="--", linewidth=2, label=f"Dynamic Threshold ({threshold_val:.3f})")
                    ax2.tick_params(axis="y", labelcolor=color_prob)
                    ax2.set_ylim(0, 1.0)
                    
                    # True label position (if any)
                    if label_curve.max() > 1e-6:
                        gt_idx = int(np.argmax(label_curve))
                        ax1.axvline(x=gt_idx, color="green", linestyle=":", alpha=0.7, linewidth=1.5, label=f"{wave_type} Ground Truth")
                    
                    # Merge legend
                    lines1, labels1 = ax1.get_legend_handles_labels()
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)
                    
                    ax1.set_title(f"{wave_type}-wave {etype} Sample #{orig_idx} (Deviation={deviation_val:.4f}, Rank={rank+1}/{top_k})")
                    
                    plt.tight_layout()
                    sample_out_path = os.path.join(out_dir, f"{wave_type}_{etype}_sample_{orig_idx}_rank{rank+1}.png")
                    fig.savefig(sample_out_path, dpi=300, bbox_inches="tight")
                    plt.close(fig)
        
        print(f"[INFO] 异常样本波形图已生成（每类波 top-{top_k_exceptions} FP/FN）", flush=True)


def run_case(case: Dict[str, Any]):
    case_seed = case.get("seed", SEED)
    seed_everything(case_seed, deterministic=True)
    eval_only = bool(case.get("eval_only", False))
    print(f"[{case['name']}] 初始化设备...", flush=True)
    print(f"[{case['name']}] 使用随机种子: {case_seed}", flush=True)
    cuda_ok = torch.cuda.is_available()
    if cuda_ok:
        device = torch.device("cuda")
        try:
            print(f"[{case['name']}] 使用设备: {device} ({torch.cuda.get_device_name(0)})", flush=True)
        except Exception:
            print(f"[{case['name']}] 使用设备: {device}", flush=True)
    else:
        device = torch.device("cpu")
        print(f"[{case['name']}] 使用设备: {device}（未检测到可用 CUDA，若需 GPU 请检查 PyTorch 是否为 GPU 版及驱动）", flush=True)

    # print(f"[{case['name']}] Initialize device...", flush=True)
    # device = torch.device("cpu")
    # print(f"[{case['name']}] uses device: {device} (forced to run on CPU)", flush=True)
    
    print(f"[{case['name']}] 加载数据集...", flush=True)
    train_ds, valid_ds, test_ds = build_datasets(case)
    print(f"[{case['name']}] 数据集加载完成: 训练集={len(train_ds)}, 验证集={len(valid_ds)}" +
          (f", 测试集={len(test_ds)}" if test_ds is not None else ""), flush=True)
    g_train = torch_generator(case_seed)
    g_valid = torch_generator(case_seed + 1)
    batch_size = case.get("batch_size", BATCH_SIZE)  # Deep Ensemble and other configurable small batches can avoid OOM
    if batch_size != BATCH_SIZE:
        print(f"[{case['name']}] 使用 batch_size={batch_size}（默认 {BATCH_SIZE}）", flush=True)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False,  # Forces the CPU to run without pinned memory
        worker_init_fn=seed_worker,
        generator=g_train,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,  # Forces the CPU to run without pinned memory
        worker_init_fn=seed_worker,
        generator=g_valid,
    )
    run_name = f"{case['name']}_seed{case_seed}"
    log_split_info(run_name, len(train_ds), len(valid_ds), seed_value=case_seed)
    case_dir = os.path.join(OUT_ROOT, run_name)
    os.makedirs(case_dir, exist_ok=True)

    # Build the model (retaining only the parameters required by the current main process)
    _drop_rate = case.get("drop_rate", case.get("dropout", 0.0))
    _pool_size = case.get("pool_size", 4)
    model_cfg = dict(
        in_ch=3,
        n_class=3,
        depths=case.get("depths", 5),
        filters_root=case.get("filters_root", 8),
        kernels=case.get("kernels", (7,)),
        pool_size=_pool_size,
        drop_rate=_drop_rate,
        use_cbam=case.get("use_cbam", False),
        use_separable=case.get("use_separable", False),
        use_gated_attention=case.get("use_gated_attention", False),
    )
    print(f"[{case['name']}] 构建模型...", flush=True)
    model_cls = MODEL_CLASS_MAP.get(case.get("model_class", "PhaseNetUNet"), PhaseNetUNet)
    model = model_cls(**model_cfg).to(device)
    has_dropout = any("Dropout" in m.__class__.__name__ for m in model.modules())
    n_dropout = sum(1 for m in model.modules() if "Dropout" in m.__class__.__name__)
    params = sum(p.numel() for p in model.parameters())
    size_mb = params * 4.0 / (1024.0 ** 2)  # float32 parameters occupy approximately MB of memory
    print(
        f"[{case['name']}] 模型构建完成，参数量: {params/1e6:.2f}M (~{params/1e3:.1f}k)，"
        f"size≈{size_mb:.2f}MB",
        flush=True,
    )
    print(f"[{case['name']}] has_dropout_modules: {has_dropout}, dropout_count: {n_dropout}", flush=True)

    total_epochs = EPOCHS
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    if SCHEDULER == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=PLATEAU_FACTOR, patience=PLATEAU_PATIENCE,
            threshold=PLATEAU_THRESHOLD, min_lr=PLATEAU_MIN_LR,
        )
    else:
        scheduler = None

    # This copy does not use class balancing losses
    class_weights_tensor = None

    # Training records
    tr_hist, va_hist = [], []
    best_by_f1 = bool(case.get("best_model_by_f1", BEST_MODEL_BY_F1))
    best = -1.0 if best_by_f1 else float("inf")
    threshold_balance_weight = case.get("threshold_balance_weight", 0.03)

    # Record the optimal model of the validation set (minimum by valid_loss)
    best_state_dict = None
    best_epoch = 1

    if eval_only:
        # Evaluation mode only: skip training and load existing best_model.pt directly from disk
        best_path = os.path.join(case_dir, "best_model.pt")
        if os.path.exists(best_path):
            print(f"[{case['name']}] eval_only=True，加载已有最优模型: {best_path}", flush=True)
            state = torch.load(best_path, map_location=device)
            model.load_state_dict(state)
            # Placeholder training/validation loss to avoid subsequent index tr_hist[-1]/va_hist[-1] errors.
            tr_hist = [0.0]
            va_hist = [0.0]
            best = 0.0
        else:
            print(f"[{case['name']}] eval_only=True 但未找到 {best_path}，将执行正常训练流程。", flush=True)
            eval_only = False

    # Normal training process
    if not eval_only:
        enable_early_stop = (DATA_SOURCE == "h5_three_channel")
        no_improve_epochs = 0
        for ep in range(1, total_epochs + 1):
            current_threshold_weight = threshold_balance_weight
            model.train()
            total_tr = 0.0
            n_tr = 0

            pbar = tqdm(train_loader, desc=f"Epoch {ep}/{total_epochs} [train]", leave=False)
            for x, y, _ in pbar:
                x = x.to(device)
                y = y.to(device)
                opt.zero_grad(set_to_none=True)

                logits = model(x)
                if y.shape[-1] != logits.shape[-1]:
                    y = _center_crop_time(y, logits.shape[-1])

                loss = combined_loss(logits, y)

                loss.backward()
                opt.step()
                bs = x.size(0)
                total_tr += float(loss.item()) * bs
                n_tr += bs
                pbar.set_postfix(
                    {
                        "loss": f"{total_tr / max(1, n_tr):.4f}",
                        "lr": f"{opt.param_groups[0]['lr']:.2e}",
                    }
                )

            tr_loss = total_tr / max(1, n_tr)
            va_loss = eval_loss(
                model,
                valid_loader,
                device,
                threshold_balance_weight=current_threshold_weight,
                class_weights=class_weights_tensor,
                epoch=ep,
                total_epochs=total_epochs,
            )
            if scheduler is not None:
                scheduler.step(va_loss)
            cur_lr = opt.param_groups[0].get("lr", LR)
            tr_hist.append(tr_loss)
            va_hist.append(va_loss)

            improved = False
            if best_by_f1:
                _fixed_p = float(case.get("fixed_thr_p", 0.5))
                _fixed_s = float(case.get("fixed_thr_s", 0.5))
                eval_metrics = eval_detailed(
                    model,
                    valid_loader,
                    device,
                    thr_p=_fixed_p,
                    thr_s=_fixed_s,
                    uncertainty_threshold_options=case,
                    current_epoch=ep,
                    tol=DYN_TOL_SAMPLES,
                )
                weights = case.get("best_metric_weights", BEST_METRIC_WEIGHTS)
                w_p, w_s = float(weights[0]), float(weights[1])
                selection = w_p * eval_metrics["p_f1"] + w_s * eval_metrics["s_f1"]
                if selection > best:
                    best = selection
                    best_epoch = ep
                    best_state_dict = copy.deepcopy(model.state_dict())
                    improved = True
            else:
                if va_loss < best:
                    best = va_loss
                    best_epoch = ep
                    best_state_dict = copy.deepcopy(model.state_dict())
                    improved = True

            if enable_early_stop:
                if improved:
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1

            best_label = "best_f1" if best_by_f1 else "best_val"
            print(
                f"[{case['name']}] epoch {ep:02d}/{total_epochs} train={tr_loss:.4f} valid={va_loss:.4f} "
                f"{best_label}={best:.4f} lr={cur_lr:.2e}"
                + (f" no_improve={no_improve_epochs}/{H5_EARLY_STOP_PATIENCE}" if enable_early_stop else ""),
                flush=True,
            )

            if enable_early_stop and no_improve_epochs >= H5_EARLY_STOP_PATIENCE:
                print(
                    f"[{case['name']}] EarlyStopping(H5) 触发：连续 {H5_EARLY_STOP_PATIENCE} 个 epoch 未提升，提前结束训练。",
                    flush=True,
                )
                break

    # After training is completed, roll back to the optimal model weights of the validation set and save them to disk (the existing best_model.pt will no longer be overwritten in eval_only mode)
    if best_state_dict is not None and not eval_only:
        model.load_state_dict(best_state_dict)
        criterion = "best_f1" if best_by_f1 else "best_val_loss"
        print(f"[{case['name']}] 使用验证集最优模型进行评估（best_epoch={best_epoch}, {criterion}={best:.4f}）", flush=True)
        best_path = os.path.join(case_dir, "best_model.pt")
        torch.save(best_state_dict, best_path)
        print(f"[{case['name']}] 最优模型已保存: {best_path}", flush=True)

    # Threshold strategy selection
    has_fixed_p = "fixed_thr_p" in case
    has_fixed_s = "fixed_thr_s" in case
    fixed_thr_p = float(case.get("fixed_thr_p", 0.5))
    fixed_thr_s = float(case.get("fixed_thr_s", 0.5))

    # rule:
    # - If fixed_thr_p/fixed_thr_s are provided explicitly, fixed thresholds are used by default (unless grid search is explicitly enabled).
    # - If a fixed threshold is not explicitly provided, grid search calibration is performed on P/S by default.
    enable_grid = bool(case.get("grid_search_ps_thresholds", (not has_fixed_p and not has_fixed_s)))
    grid = list(case.get("ps_threshold_grid", DYN_THRESH_GRID))
    thr_max_batches = int(case.get("ps_threshold_max_batches", 200))

    if enable_grid:
        print(
            f"[{case['name']}] P/S 阈值网格搜索标定 "
            f"(grid={len(grid)} points, tol={DYN_TOL_SAMPLES} samples, max_batches={thr_max_batches})...",
            flush=True,
        )
        (p_conf, p_err, p_has), (s_conf, s_err, s_has) = _collect_conf_err(
            model, valid_loader, device, tol_samples=DYN_TOL_SAMPLES, max_batches=thr_max_batches
        )
        thr_p, f1_p = _best_threshold(p_conf, p_err, p_has, tol=DYN_TOL_SAMPLES, grid=grid)
        thr_s, f1_s = _best_threshold(s_conf, s_err, s_has, tol=DYN_TOL_SAMPLES, grid=grid)
        print(
            f"[{case['name']}] 网格最优阈值: P={thr_p:.3f} (F1={f1_p:.4f}), "
            f"S={thr_s:.3f} (F1={f1_s:.4f})",
            flush=True,
        )
    else:
        # Use fixed threshold (can be overridden by configuration)
        print(f"[{case['name']}] 使用固定阈值 P={fixed_thr_p:.3f} S={fixed_thr_s:.3f} ...", flush=True)
        thr_p = fixed_thr_p
        thr_s = fixed_thr_s
        
        # Optional: Check the predicted probability distribution (for diagnostics)
        print(f"[{case['name']}] 检查预测概率分布...", flush=True)
        model.eval()
        all_p_probs, all_s_probs = [], []
        with torch.inference_mode():
            for x, y, _ in valid_loader:
                x = x.to(device); y = y.to(device)
                logits = model(x)
                if y.shape[-1] != logits.shape[-1]:
                    y = _center_crop_time(y, logits.shape[-1])
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                y_np = y.cpu().numpy()
                B = probs.shape[0]
                for b in range(B):
                    # Only count the maximum probability of samples with true labels
                    if y_np[b, 1].max() > 1e-6:  # There is P wave
                        p_max_prob = float(np.max(probs[b, 1]))
                        all_p_probs.append(p_max_prob)
                    if y_np[b, 2].max() > 1e-6:  # Have S wave
                        s_max_prob = float(np.max(probs[b, 2]))
                        all_s_probs.append(s_max_prob)
        
        if all_p_probs:
            print(f"[{case['name']}] P波预测概率统计: mean={np.mean(all_p_probs):.3f}, std={np.std(all_p_probs):.3f}, "
                  f"min={np.min(all_p_probs):.3f}, max={np.max(all_p_probs):.3f}, "
                  f"超过0.5的比例={np.mean(np.array(all_p_probs) >= 0.5):.1%}", flush=True)
        if all_s_probs:
            print(f"[{case['name']}] S波预测概率统计: mean={np.mean(all_s_probs):.3f}, std={np.std(all_s_probs):.3f}, "
                  f"min={np.min(all_s_probs):.3f}, max={np.max(all_s_probs):.3f}, "
                  f"超过0.5的比例={np.mean(np.array(all_s_probs) >= 0.5):.1%}", flush=True)
        
    if not enable_grid:
        # Optional: Calculate F1 corresponding to a fixed threshold (for recording)
        (p_conf, p_err, p_has), (s_conf, s_err, s_has) = _collect_conf_err(
            model, valid_loader, device, tol_samples=DYN_TOL_SAMPLES, max_batches=thr_max_batches
        )
        _, f1_p = _best_threshold(p_conf, p_err, p_has, tol=DYN_TOL_SAMPLES, grid=[thr_p])
        _, f1_s = _best_threshold(s_conf, s_err, s_has, tol=DYN_TOL_SAMPLES, grid=[thr_s])
        print(
            f"[{case['name']}] 固定阈值: P={thr_p:.3f} (F1={f1_p:.4f}), "
            f"S={thr_s:.3f} (F1={f1_s:.4f})",
            flush=True,
        )

    # Detailed metrics: Uniform use of a single forward deterministic path.
    metrics_full = eval_detailed(
        model,
        valid_loader,
        device,
        thr_p=thr_p,
        thr_s=thr_s,
        uncertainty_threshold_options=case,
        current_epoch=total_epochs,
        tol=DYN_TOL_SAMPLES,
    )

    test_loader_for_vis = None

    # ===== Extra: Evaluate on an independent test set (the tests of CEED/H5 are all divided from train, consistent with val) =====
    if RUN_TEST_EVAL and test_ds is not None:
        try:
            print(f"[{case['name']}] 在独立测试集上评估（不再调参，仅前向计算指标）...", flush=True)
            g_test = torch_generator(SEED + 2)
            test_loader = DataLoader(
                test_ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=NUM_WORKERS,
                pin_memory=False,  # Turn off pinned memory while CPU is running
                worker_init_fn=seed_worker,
                generator=g_test,
            )
            # The TEST set uniformly uses deterministic eval_detailed as the full indicator to facilitate alignment with the baseline.
            test_metrics = eval_detailed(
                model,
                test_loader,
                device,
                thr_p=thr_p,
                thr_s=thr_s,
                uncertainty_threshold_options=case,
                current_epoch=total_epochs,
                tol=DYN_TOL_SAMPLES,
            )
            def _fmt_res_t(key: str) -> str:
                v = test_metrics.get(key)
                return f"{float(v):.4f}" if v is not None else "N/A"
            print(
                f"[{case['name']}] TEST 结果: "
                f"time_acc={test_metrics.get('time_acc', float('nan')):.4f}, "
                f"P-Prec={test_metrics.get('p_prec', float('nan')):.4f}, "
                f"P-Rec={test_metrics.get('p_rec', float('nan')):.4f}, "
                f"P-F1={test_metrics.get('p_f1', float('nan')):.4f}, "
                f"S-Prec={test_metrics.get('s_prec', float('nan')):.4f}, "
                f"S-Rec={test_metrics.get('s_rec', float('nan')):.4f}, "
                f"S-F1={test_metrics.get('s_f1', float('nan')):.4f} | "
                f"p_res(mean/std/mae)s=({_fmt_res_t('p_res_mean_sec')}/{_fmt_res_t('p_res_std_sec')}/{_fmt_res_t('p_res_mae_sec')}), "
                f"s_res=({_fmt_res_t('s_res_mean_sec')}/{_fmt_res_t('s_res_std_sec')}/{_fmt_res_t('s_res_mae_sec')})",
                flush=True,
            )
            # The test set DataLoader is also available for candidate samples of representative waveform samples.
            test_loader_for_vis = test_loader
        except Exception as e:
            print(f"[{case['name']}] 在测试集上评估失败（{e}），仅保留验证集指标。", flush=True)

    # Output: Uniformly use the run_name directory with seed to avoid multiple seed coverage/confusion
    case_dir = os.path.join(OUT_ROOT, run_name)
    os.makedirs(case_dir, exist_ok=True)

    # Residual output time residual distribution based on deterministic eval_detailed
    p_res_full = metrics_full.get("p_residuals_signed")
    s_res_full = metrics_full.get("s_residuals_signed")
    if p_res_full is not None and s_res_full is not None:
        time_res_full = {
            "p_residuals_signed": p_res_full,
            "s_residuals_signed": s_res_full,
        }
        with open(os.path.join(case_dir, "time_residuals.json"), "w", encoding="utf-8") as f:
            json.dump(time_res_full, f)
        with open(os.path.join(case_dir, "time_residuals_ours.json"), "w", encoding="utf-8") as f:
            json.dump(time_res_full, f)
        if plt is not None:
            plot_time_residual_distribution(
                p_res_full,
                s_res_full,
                os.path.join(case_dir, "time_residual_distribution.png"),
                sample_rate=SAMPLE_RATE,
            )

    # PCA visualization: use the validation set forward to obtain P/S probability and statistical features, and reduce the dimension to 2D to display P/S/noise separation
    if plt is not None and case.get("generate_visualizations", True):
        try:
            features, labels = collect_pca_features_and_labels(model, valid_loader, device)
            if len(features) >= 2:
                np.savez(
                    os.path.join(case_dir, "pca_features_labels.npz"),
                    features=features,
                    labels=labels,
                    allow_pickle=True,
                )
                np.savez(
                    os.path.join(case_dir, "pca_ours.npz"),
                    features=features,
                    labels=labels,
                    allow_pickle=True,
                )
                plot_pca_visualization(
                    features,
                    labels,
                    os.path.join(case_dir, "pca_visualization.png"),
                    label_names=["BG window", "P window", "S window"],
                )
        except Exception as e:
            print(f"[{case['name']}] PCA 可视化跳过（{e}）", flush=True)

    # PR curve and performance plot stratified by SNR (class imbalance/low SNR advantage demonstrated)
    if plt is not None and case.get("generate_visualizations", True):
        try:
            pr_snr = collect_pr_snr_data(model, valid_loader, device, tol_samples=DYN_TOL_SAMPLES)
            if pr_snr["has_p"]:
                with open(os.path.join(case_dir, "pr_snr_data.json"), "w", encoding="utf-8") as f:
                    json.dump(pr_snr, f)
                with open(os.path.join(case_dir, "pr_snr_data_ours.json"), "w", encoding="utf-8") as f:
                    json.dump(pr_snr, f)
                # The comparison is explained in a data format, which is convenient for writing baseline data such as AR and then drawing a comparison chart.
                _write_comparison_format(case_dir)
                plot_pr_curve(pr_snr, os.path.join(case_dir, "pr_curve.png"))
                plot_snr_stratified(pr_snr, os.path.join(case_dir, "snr_stratified.png"), fixed_thr=0.5)
                plot_max_prob_histogram(pr_snr, os.path.join(case_dir, "max_prob_histogram.png"), bins=15)
        except Exception as e:
            print(f"[{case['name']}] PR/SNR 图跳过（{e}）", flush=True)

    generate_visualizations = case.get("generate_visualizations", True)  # Visual charts are generated by default
    # Loss curve default output (not affected by generate_visualizations=False)
    if plt is not None:
        plot_losses(tr_hist, va_hist, os.path.join(case_dir, "loss_curve.png"))

    if plt is not None and generate_visualizations:
        print(f"[{case['name']}] 生成样本可视化...", flush=True)
        save_visuals(model, valid_ds, device, os.path.join(case_dir, "figs"), n=N_VIS)
        # Representative 2×2 waveform example plot (Normal / Low / Very low / Channel missing)
        try:
            rep_out = os.path.join(case_dir, "representative_waveforms_2x2.png")
            rep_npz = None
            name_str = str(case.get("name", "")).lower()
            if name_str == "full" or "full" in name_str:
                rep_npz = os.path.join(case_dir, "representative_compare.npz")
            extra_rep_loaders = [train_loader]
            if test_loader_for_vis is not None:
                extra_rep_loaders.append(test_loader_for_vis)

            save_representative_waveforms_2x2(
                model,
                valid_loader,
                device,
                out_path=rep_out,
                tol=DYN_TOL_SAMPLES,
                # Consistent with eval_detailed: use the final threshold adopted in the current round (either a fixed value or a grid-scaled value).
                thr_p=float(thr_p),
                thr_s=float(thr_s),
                max_scan_samples=int(case.get("rep_waveform_max_scan", 2000)),
                extra_loaders=extra_rep_loaders,
                rep_npz_path=rep_npz,
            )
        except Exception:
            pass
        print(f"[{case['name']}] 可视化图表生成完成（loss_curve.png + figs/vis_*.png）", flush=True)
    elif plt is None:
        print(f"[{case['name']}] 跳过可视化图表生成（matplotlib 不可用）", flush=True)
    elif not generate_visualizations:
        print(f"[{case['name']}] 已输出 loss_curve.png，跳过样本可视化（generate_visualizations=False）", flush=True)

    # Unified F1 source: f1_p/f1_s and p_f1/s_f1 are both from the same round eval_detailed (metrics_full),
    # Avoid inconsistency between the two columns due to different logic between best_threshold(argmax) and eval_detailed(including structural post-processing)
    metrics: Dict[str, Any] = dict(metrics_full)

    # Result row name: <case_name>_ceed under CEED data source, other data sources keep the original name
    row_name = case["name"]
    if DATA_SOURCE == "ceed" and not row_name.endswith("_ceed"):
        row_name = f"{row_name}_ceed"

    row = dict(
        name=row_name, best_val=best, train_last=tr_hist[-1], valid_last=va_hist[-1],
        use_cbam=case["use_cbam"], kernels=list(case["kernels"]),
        thr_p=thr_p, thr_s=thr_s,
        f1_p=float(metrics_full["p_f1"]), f1_s=float(metrics_full["s_f1"]),
        **metrics,
    )
    print(f"[{case['name']}] 训练和评估完成，最终结果: P-F1={metrics_full['p_f1']:.4f}, S-F1={metrics_full['s_f1']:.4f}", flush=True)
    return row


def log_split_info(case_name: str, train_size: int, valid_size: int, seed_value: int = SEED):
    info_dir = os.path.join(OUT_ROOT, case_name)
    os.makedirs(info_dir, exist_ok=True)
    info = {
        "case": case_name,
        "seed": seed_value,
        "timestamp": datetime.now().isoformat(),
        "data_source": DATA_SOURCE,
        "ceed_dataset": CEED_DATASET_NAME,
        "ceed_train_split": CEED_TRAIN_SPLIT,
        "ceed_valid_split": CEED_VALID_SPLIT,
        "train_samples": train_size,
        "valid_samples": valid_size,
    }
    with open(os.path.join(info_dir, "split_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)


def append_csv(path: str, header: list[str], rows: list[Dict[str, Any]]):
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        for r in rows:
            thr_p_str = f"{r['thr_p']:.2f}" if r.get("thr_p") is not None else "N/A"
            thr_s_str = f"{r['thr_s']:.2f}" if r.get("thr_s") is not None else "N/A"
            # Time residual statistics (may be None, use N/A placeholder)
            def _fmt_res(key: str) -> str:
                v = r.get(key, None)
                return f"{float(v):.4f}" if v is not None else "N/A"
            w.writerow([
                r["name"], f"{r['best_val']:.6f}", f"{r['train_last']:.6f}", f"{r['valid_last']:.6f}",
                r["use_cbam"], r["kernels"],
                thr_p_str, thr_s_str,
                f"{r['time_acc']:.4f}", f"{r.get('mcc', 0.0):.4f}",
                f"{r['p_prec']:.4f}", f"{r['p_rec']:.4f}", f"{r['p_f1']:.4f}",
                f"{r['s_prec']:.4f}", f"{r['s_rec']:.4f}", f"{r['s_f1']:.4f}",
                _fmt_res("p_res_mean_sec"), _fmt_res("p_res_std_sec"), _fmt_res("p_res_mae_sec"),
                _fmt_res("s_res_mean_sec"), _fmt_res("s_res_std_sec"), _fmt_res("s_res_mae_sec"),
            ])


