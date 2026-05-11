"""
Compare baseline models (non-U‑Net architecture): TCN, ResASPP, Inception1D

Output (generated for each baseline model)
- Training/validation loss curve: PhaseNet/baselines/<model>/loss_curve.png
- Random visualization diagram: PhaseNet/baselines/<model>/figs/vis_*.png
- Metric summary: PhaseNet/baselines/metrics.csv (including Precision/Recall/F1/Acc and optimal threshold)

illustrate
- These models are structurally different from the existing U‑Net backbone (U‑Net structure without up/down sampling) and have similar parameter quantities to the main model to highlight the advantages of the main model.
- The evaluation strategy is aligned with single_ablation_unet_core: take one point for each sample P/S (this script uses argmax), tol=10 sampling points to determine TP,
  No true value but predicted P/S is counted as FP, and Precision/Recall/F1 are defined consistently to facilitate direct comparison with ablation results.
"""

from __future__ import annotations

import argparse
import json
import os
import csv
import random
from typing import Dict, Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch.utils.data import DataLoader, Subset
from utils.repro import seed_everything, seed_worker, torch_generator

from data import WaveformDataset
from model import UNet1D
from three_channel_h5_dataset import ThreeChannelH5Dataset  # Three-channel H5 dataset
from single_ablation_visualization import plot_time_residual_distribution  # type: ignore
from single_ablation_unet_core import (  # type: ignore
    DATA_SOURCE as CORE_DATA_SOURCE,
    CEED_LOCAL_DIR,
    CEED_WAVEFORM_KEY,
    CEED_P_KEY,
    CEED_S_KEY,
    H5_THREE_CHANNEL_ROOT,
    H5_TEST_RATIO,
    H5_TRAIN_VAL_RATIO,
    H5_LIMIT,
    H5_LIMIT_TRAIN,
    H5_LIMIT_VAL,
    H5_LIMIT_TEST,
    H5_ARRIVAL_RELATIVE_TO_SEGMENT,
    H5_FILTER_NATURAL_ONLY,
    H5_ALLOW_TYPES,
)

# Important: You must import CEEDDataset after importing single_ablation_unet_core.
# Otherwise, the environment variable settings for the HF cache directory (HF_HOME / HF_DATASETS_CACHE / HF_HUB_CACHE) in single_ablation_unet_core may not have time to take effect.
# As a result, the datasets cache may still fall into the default user cache directory and trigger insufficient disk space.
from ceed_data import CEEDDataset

# Drawing (use Agg in non-display environment)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore
except Exception:
    plt = None


# ===== Data and training parameters (shares DATA_SOURCE selection with single_ablation_unet_core) =====
DATA_SOURCE = CORE_DATA_SOURCE  # "npz" | "ceed" | "h5_three_channel"

# NPZ (not currently used, reserved for compatibility)
DATA_ROOT = "dataset"
TRAIN_DIR = os.path.join(DATA_ROOT, "waveform_train")
TRAIN_CSV = os.path.join(DATA_ROOT, "waveform_train_split.csv")
VALID_CSV = os.path.join(DATA_ROOT, "waveform_valid_split.csv")

# CEED configuration (aligned with single_ablation_unet_core)
CEED_DATASET_NAME = "CEED.py"
CEED_TRAIN_SPLIT = "train"
CEED_TEST_SPLIT = "test"
CEED_VAL_RATIO = 0.2
CEED_LIMIT_TRAIN = 9000
CEED_LIMIT_TEST = 1000

OUTPUT_ROOT = os.path.join("PhaseNet", "baselines")
os.makedirs(OUTPUT_ROOT, exist_ok=True)
METRICS_CSV = os.path.join(OUTPUT_ROOT, "metrics.csv")

# Training parameters/random seed (same value as single_ablation_unet_core)
EPOCHS = 50
BATCH_SIZE = 256
LR = 5e-4
WEIGHT_DECAY = 0.0

# PhaseNet original configuration (phasenet/train.py + model.py), only used by PhaseNet baseline
PHASENET_TRAIN_CONFIG = {
    "batch_size": 20,
    "lr": 0.01,
    "epochs": 100,
    "decay_rate": 0.9,
    "decay_step": None,  # Set to len(train_ds)//batch_size during runtime
}
NUM_WORKERS = 0
CROP_LEN = 3000
LABEL_WIDTH = 51
LABEL_SIGMA_SEC = 0.1
SAMPLE_RATE = 100.0
N_VIS = 4
SEED = 2025  # Consistent with single_ablation_unet_core.SEED

# Dynamic threshold (tol is consistent with the main training script)
DYN_TOL_SAMPLES = 10

# ===== Optimal model preservation strategy (consistent with single_ablation_unet_core) =====
BEST_MODEL_BY_F1 = True
BEST_METRIC_WEIGHTS = (0.5, 0.5)  # (w_p, w_s)
SAVE_BEST_MODEL = True


# ===== Auxiliary: Reference model parameter quantities (for alignment of magnitudes) =====
def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def approx_target_params() -> int:
    # Use a similar configuration to train.py to estimate the main model parameters.
    ref = UNet1D(in_ch=3, n_class=3, base_ch=16, depth=4, kernels=(3,7,15),
                 dropout=0.0, factor=4, use_cbam=True, use_separable=False,
                 cbam_reduction=4, cbam_spatial_kernel=5)
    return count_params(ref)


TARGET_PARAMS = approx_target_params()  # For printing reference only


# ===== Model Definition (Non-U‑Net) =====
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=7, d=1):
        super().__init__()
        pad = (k // 2) * d
        self.conv = nn.Conv1d(in_ch, out_ch, k, padding=pad, dilation=d, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class TCNBlock(nn.Module):
    def __init__(self, ch, k=3, dil=1, drop=0.1):
        super().__init__()
        # Use 'same-length' padding so residual add keeps shape.
        # For odd kernels, pad = (k//2) * dilation preserves T.
        pad = (k // 2) * dil
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, k, padding=pad, dilation=dil, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Conv1d(ch, ch, k, padding=pad, dilation=dil, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return x + self.net(x)


class BaselineTCN(nn.Module):
    """Deep TCN (dilated convolution residual), the number of parameters is similar to the main model."""
    def __init__(self, in_ch=3, n_class=3, width=128, n_blocks=10):
        super().__init__()
        self.stem = nn.Conv1d(in_ch, width, kernel_size=1)
        blocks = []
        for i in range(n_blocks):
            blocks.append(TCNBlock(width, k=3, dil=2**(i % 5), drop=0.1))
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Conv1d(width, n_class, kernel_size=1)

    def forward(self, x):
        x = self.stem(x)
        x = self.tcn(x)
        return self.head(x)


class ASPP1D(nn.Module):
    def __init__(self, ch, out_ch, dilations=(1,2,4,8,16)):
        super().__init__()
        self.branches = nn.ModuleList([ConvBNReLU(ch, out_ch, k=3, d=d) for d in dilations])
        self.fuse = nn.Sequential(
            nn.Conv1d(out_ch * len(dilations), out_ch, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_ch), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        feats = [b(x) for b in self.branches]
        x = torch.cat(feats, dim=1)
        return self.fuse(x)


class BaselineResASPP(nn.Module):
    """ResNet1D + ASPP: No upsampling and downsampling, capturing multiple scales through atrous convolution."""
    def __init__(self, in_ch=3, n_class=3, width=160, n_blocks=6):
        super().__init__()
        self.stem = ConvBNReLU(in_ch, width, k=7, d=1)
        blocks = []
        for _ in range(n_blocks):
            blocks.append(nn.Sequential(ConvBNReLU(width, width, k=3), ConvBNReLU(width, width, k=3)))
        self.backbone = nn.Sequential(*blocks)
        self.aspp = ASPP1D(width, width, dilations=(1,2,4,8,16))
        self.head = nn.Conv1d(width, n_class, kernel_size=1)

    def forward(self, x):
        x = self.stem(x)
        x = self.backbone(x) + x
        x = self.aspp(x)
        return self.head(x)


class InceptionBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, ks=(3,7,15)):
        super().__init__()
        mid = out_ch // len(ks)
        self.pre = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
        self.branches = nn.ModuleList([ConvBNReLU(out_ch, mid, k=k) for k in ks])
        self.fuse = nn.Sequential(nn.Conv1d(mid * len(ks), out_ch, kernel_size=1, bias=False), nn.BatchNorm1d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.pre(x)
        feats = [b(x) for b in self.branches]
        x = torch.cat(feats, dim=1)
        return self.fuse(x)


class BaselineInception(nn.Module):
    """Inception style 1D CNN, maintaining resolution and adjustable parameter amount."""
    def __init__(self, in_ch=3, n_class=3, width=128, n_blocks=8):
        super().__init__()
        blocks = [InceptionBlock1D(in_ch if i==0 else width, width, ks=(3,7,15)) for i in range(n_blocks)]
        self.net = nn.Sequential(*blocks)
        self.head = nn.Conv1d(width, n_class, kernel_size=1)

    def forward(self, x):
        x = self.net(x)
        return self.head(x)


class PhaseNetUNet(nn.Module):
    """PhaseNet style 1D U‑Net (ported according to the official source code UNet structure).

    - depths = 5
    - filters_root = 8
    - kernel_size = 7
    - pool_size = 4
    - Conv -> BN -> ReLU -> Dropout order is consistent with the original implementation
    """

    def __init__(
        self,
        in_ch: int = 3,
        n_class: int = 3,
        depths: int = 5,
        filters_root: int = 8,
        kernel_size: int = 7,
        pool_size: int = 4,
        drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        assert depths >= 1
        self.depths = depths
        self.pool_size = pool_size
        self.drop_rate = drop_rate
        self.kernel_size = kernel_size

        def conv_block(in_c: int, out_c: int) -> nn.Sequential:
            layers: list[nn.Module] = [
                nn.Conv1d(in_c, out_c, kernel_size, padding=kernel_size // 2, bias=False),
                nn.BatchNorm1d(out_c),
                nn.ReLU(inplace=True),
            ]
            if drop_rate and drop_rate > 0.0:
                layers.append(nn.Dropout(p=drop_rate))
            return nn.Sequential(*layers)

        # Input convolution: 3 channels -> filters_root
        in_layers: list[nn.Module] = [
            nn.Conv1d(in_ch, filters_root, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(filters_root),
            nn.ReLU(inplace=True),
        ]
        if drop_rate and drop_rate > 0.0:
            in_layers.append(nn.Dropout(p=drop_rate))
        self.input_conv = nn.Sequential(*in_layers)

        # downsampling path
        downs: list[nn.Module] = []
        pools: list[nn.Module] = []
        ch_in = filters_root
        for d in range(depths):
            filters = int((2 ** d) * filters_root)
            downs.append(conv_block(ch_in, filters))
            ch_in = filters
            if d < depths - 1:
                pool_layers: list[nn.Module] = [
                    nn.Conv1d(
                        ch_in,
                        ch_in,
                        kernel_size,
                        stride=pool_size,
                        padding=kernel_size // 2,
                        bias=False,
                    ),
                    nn.BatchNorm1d(ch_in),
                    nn.ReLU(inplace=True),
                ]
                if drop_rate and drop_rate > 0.0:
                    pool_layers.append(nn.Dropout(p=drop_rate))
                pools.append(nn.Sequential(*pool_layers))
        self.downs = nn.ModuleList(downs)
        self.pools = nn.ModuleList(pools)

        # upsampling path
        ups: list[nn.Module] = []
        up_convs: list[nn.Module] = []
        ch = ch_in
        for d in range(depths - 2, -1, -1):
            filters = int((2 ** d) * filters_root)
            up_layers: list[nn.Module] = [
                nn.ConvTranspose1d(ch, filters, kernel_size=pool_size, stride=pool_size),
                nn.BatchNorm1d(filters),
                nn.ReLU(inplace=True),
            ]
            if drop_rate and drop_rate > 0.0:
                up_layers.append(nn.Dropout(p=drop_rate))
            ups.append(nn.Sequential(*up_layers))
            # Channels doubled after cascading skips
            up_convs.append(conv_block(filters * 2, filters))
            ch = filters
        self.ups = nn.ModuleList(ups)
        self.up_convs = nn.ModuleList(up_convs)

        # Output header: 1×1 convolution to 3 classes
        self.head = nn.Conv1d(ch, n_class, kernel_size=1)

    @staticmethod
    def _center_crop(x: torch.Tensor, target_len: int) -> torch.Tensor:
        """Center or fill to target_len in the time dimension (corresponding to the TF version of crop_and_concat behavior)."""
        T = x.shape[-1]
        if T == target_len:
            return x
        if T < target_len:
            pad = target_len - T
            left = pad // 2
            right = pad - left
            return F.pad(x, (left, right))
        # T > target_len
        start = (T - target_len) // 2
        return x[..., start : start + target_len]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input convolution
        x = self.input_conv(x)

        # downsample + save skip
        skips: list[torch.Tensor] = []
        h = x
        for d in range(self.depths):
            h = self.downs[d](h)
            skips.append(h)
            if d < self.depths - 1:
                h = self.pools[d](h)

        # upsampling
        for i, up in enumerate(self.ups):
            d = self.depths - 2 - i
            h = up(h)
            skip = skips[d]
            if h.shape[-1] != skip.shape[-1]:
                h = self._center_crop(h, skip.shape[-1])
            h = torch.cat([skip, h], dim=1)
            h = self.up_convs[i](h)

        logits = self.head(h)
        return logits


# ===== Dataset construction (consistent with CEED/H5 partitioning logic of single_ablation_unet_core) =====
_dataset_split_cache: dict[str, tuple[list[int], list[int]]] = {}


def build_datasets():
    """Build training/validation data sets.

    - For CEED: read CEED_LIMIT_TRAIN samples from train split, divide train/val by CEED_VAL_RATIO;
      This is consistent with the approach in single_ablation_unet_core (except that no additional test set is built here).
    """
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

    if DATA_SOURCE == "ceed":
        cache_key = f"{CEED_DATASET_NAME}_{CEED_TRAIN_SPLIT}_{CEED_LIMIT_TRAIN}_{CEED_VAL_RATIO}_{SEED}"
        if cache_key in _dataset_split_cache:
            train_idx, val_idx = _dataset_split_cache[cache_key]
        else:
            full_train_ds = CEEDDataset(
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
            )
            n_full = len(full_train_ds)
            if n_full < 2:
                raise ValueError(f"CEED train 样本不足（{n_full}），至少需要 2 条")
            val_size = max(1, int(n_full * CEED_VAL_RATIO))
            train_size = n_full - val_size
            g = torch.Generator().manual_seed(SEED)
            perm = torch.randperm(n_full, generator=g)
            train_idx = perm[:train_size].tolist()
            val_idx = perm[train_size:].tolist()
            _dataset_split_cache[cache_key] = (train_idx, val_idx)
            print(f"[build_datasets] CEED train 划分完成并已缓存（train={train_size}, val={val_size}）")

        # Reconstruct the data set object to ensure that the behavior of the training set with enhancement and the validation set without enhancement is consistent with the main script
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
        )
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
        train_ds = Subset(full_train_ds_aug, train_idx)
        valid_ds = Subset(full_train_ds_clean, val_idx)

        test_ds = CEEDDataset(
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
        return train_ds, valid_ds, test_ds

    if DATA_SOURCE == "h5_three_channel":
        # Three-channel H5: Refer to the H5 division ideas in single_ablation_unet_core
        base_ds_train = ThreeChannelH5Dataset(
            root_dir=H5_THREE_CHANNEL_ROOT,
            crop_len=CROP_LEN,
            sampling_rate=SAMPLE_RATE,
            label_sigma_sec=LABEL_SIGMA_SEC,
            label_width=LABEL_WIDTH,
            training=True,
            arrival_relative_to_segment=H5_ARRIVAL_RELATIVE_TO_SEGMENT,
            filter_natural_only=H5_FILTER_NATURAL_ONLY,
            allow_earthquake_types=tuple(H5_ALLOW_TYPES) if H5_ALLOW_TYPES else None,
            strict_check=False,
            limit=H5_LIMIT,
        )
        n_full = len(base_ds_train)
        if n_full < 3:
            raise ValueError(f"H5 数据集样本不足（{n_full}），至少需要 3 条用于 train/val/test 划分")

        # First draw out test, and then divide the remaining samples into train/val according to H5_TRAIN_VAL_RATIO
        n_test = max(1, int(n_full * H5_TEST_RATIO))
        n_trainval = n_full - n_test
        g = torch.Generator().manual_seed(SEED)
        perm = torch.randperm(n_full, generator=g).tolist()
        test_idx = perm[:n_test]
        trainval_idx = perm[n_test:]

        n_train = max(1, int(n_trainval * H5_TRAIN_VAL_RATIO))
        n_val = max(1, n_trainval - n_train)
        train_idx = trainval_idx[:n_train]
        val_idx = trainval_idx[n_train:n_train + n_val]

        # Optional: apply H5_LIMIT_TRAIN/VAL/TEST (if not None)
        if H5_LIMIT_TRAIN is not None:
            train_idx = train_idx[: H5_LIMIT_TRAIN]
        if H5_LIMIT_VAL is not None:
            val_idx = val_idx[: H5_LIMIT_VAL]
        if H5_LIMIT_TEST is not None:
            test_idx = test_idx[: H5_LIMIT_TEST]

        train_ds = Subset(base_ds_train, train_idx)

        base_ds_eval = ThreeChannelH5Dataset(
            root_dir=H5_THREE_CHANNEL_ROOT,
            crop_len=CROP_LEN,
            sampling_rate=SAMPLE_RATE,
            label_sigma_sec=LABEL_SIGMA_SEC,
            label_width=LABEL_WIDTH,
            training=False,
            arrival_relative_to_segment=H5_ARRIVAL_RELATIVE_TO_SEGMENT,
            filter_natural_only=H5_FILTER_NATURAL_ONLY,
            allow_earthquake_types=tuple(H5_ALLOW_TYPES) if H5_ALLOW_TYPES else None,
            strict_check=False,
            limit=H5_LIMIT,
        )
        valid_ds = Subset(base_ds_eval, val_idx)
        test_ds = Subset(base_ds_eval, test_idx)
        print(
            f"[build_datasets] H5 three-channel 划分完成（train={len(train_ds)}, val={len(valid_ds)}, test={len(test_ds)}）",
            flush=True,
        )
        return train_ds, valid_ds, test_ds

    raise ValueError("DATA_SOURCE must be 'npz' or 'ceed' or 'h5_three_channel'")


# ===== Training/Assessment and Tools =====
def _center_crop_time(y: torch.Tensor, target_T: int) -> torch.Tensor:
    T = y.shape[-1]
    if T == target_T:
        return y
    if T < target_T:
        pad = target_T - T
        left = pad // 2
        right = pad - left
        return F.pad(y, (left, right))
    start = (T - target_T) // 2
    return y[..., start:start+target_T]


def soft_ce(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (-(y * torch.log_softmax(logits, dim=1))).mean()


@torch.inference_mode()
def eval_loss(model, loader, device) -> float:
    model.eval(); total = 0.0; n = 0
    for x, y, _ in loader:
        x = x.to(device); y = y.to(device)
        logits = model(x)
        if y.shape[-1] != logits.shape[-1]:
            y = _center_crop_time(y, logits.shape[-1])
        loss = soft_ce(logits, y)
        bs = x.size(0); total += float(loss.item()) * bs; n += bs
    return total / max(1, n)


@torch.inference_mode()
def collect_conf_err(model, loader, device):
    p_conf, p_err, p_has = [], [], []
    s_conf, s_err, s_has = [], [], []
    for x, y, _ in loader:
        x = x.to(device); y = y.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        y_np = y.cpu().numpy(); B = probs.shape[0]
        for b in range(B):
            has_p = bool(y_np[b, 1].max() > 1e-6)
            p_idx = int(np.argmax(probs[b, 1])); p_c = float(probs[b, 1, p_idx])
            p_e = abs(p_idx - int(np.argmax(y_np[b, 1]))) if has_p else None
            p_conf.append(p_c); p_err.append(p_e); p_has.append(has_p)
            has_s = bool(y_np[b, 2].max() > 1e-6)
            s_idx = int(np.argmax(probs[b, 2])); s_c = float(probs[b, 2, s_idx])
            s_e = abs(s_idx - int(np.argmax(y_np[b, 2]))) if has_s else None
            s_conf.append(s_c); s_err.append(s_e); s_has.append(has_s)
    return (p_conf, p_err, p_has), (s_conf, s_err, s_has)


def best_threshold(confs, errs, has_gts, tol: int, grid: list[float]):
    best_thr, best_f1 = 0.5, -1.0
    for thr in grid:
        tp = fp = fn = 0
        for c, e, h in zip(confs, errs, has_gts):
            if c >= thr:
                if h and e is not None and e <= tol:
                    tp += 1
                else:
                    fp += 1
            else:
                if h:
                    fn += 1
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
        if f1 > best_f1:
            best_f1 = f1; best_thr = thr
    return best_thr, best_f1


@torch.inference_mode()
def eval_detailed(model, loader, device, thr_p: float, thr_s: float, tol: int = 10) -> Dict[str, float]:
    """Same as the counting rule of single_ablation_unet_core.eval_detailed: tol is TP, and there is no true value but it is predicted to be FP. The point taken is the argmax of this script (no structural post-processing)."""
    model.eval()
    time_correct = 0; n_time = 0
    p_tp = p_fp = p_fn = p_tn = 0
    s_tp = s_fp = s_fn = s_tn = 0
    # Time residual (for mean/std/mae, the unit is finally converted to seconds)
    all_p_residual_signed: list[int] = []
    all_s_residual_signed: list[int] = []
    for x, y, _ in loader:
        x = x.to(device); y = y.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred_cls = probs.argmax(dim=1); true_cls = y.argmax(dim=1)
        time_correct += (pred_cls == true_cls).sum().item(); n_time += pred_cls.numel()
        probs_np = probs.cpu().numpy(); y_np = y.cpu().numpy(); B = probs_np.shape[0]
        for b in range(B):
            # P: Consistent with single_ablation_unet_core - if there is a true value, it is TP/FP/FN; if there is no true value but it is predicted, it is FP
            has_p = bool(y_np[b, 1].max() > 1e-6)
            p_idx = int(np.argmax(probs_np[b, 1])); p_c = float(probs_np[b, 1, p_idx])
            if has_p:
                gt = int(np.argmax(y_np[b, 1])); err = abs(p_idx - gt)
                all_p_residual_signed.append(p_idx - gt)
                if p_c >= thr_p:
                    if err <= tol: p_tp += 1
                    else: p_fp += 1
                else:
                    p_fn += 1
            else:
                if p_c >= thr_p:
                    p_fp += 1
                else:
                    p_tn += 1
            # S: Same as above
            has_s = bool(y_np[b, 2].max() > 1e-6)
            s_idx = int(np.argmax(probs_np[b, 2])); s_c = float(probs_np[b, 2, s_idx])
            if has_s:
                gt = int(np.argmax(y_np[b, 2])); err = abs(s_idx - gt)
                all_s_residual_signed.append(s_idx - gt)
                if s_c >= thr_s:
                    if err <= tol: s_tp += 1
                    else: s_fp += 1
                else:
                    s_fn += 1
            else:
                if s_c >= thr_s:
                    s_fp += 1
                else:
                    s_tn += 1
    acc = time_correct / max(1, n_time)
    def _prf(tp, fp, fn):
        p = tp / max(1, tp + fp); r = tp / max(1, tp + fn)
        f1 = 0.0 if (p + r) == 0 else 2 * p * r / (p + r)
        return p, r, f1
    p_prec, p_rec, p_f1 = _prf(p_tp, p_fp, p_fn); s_prec, s_rec, s_f1 = _prf(s_tp, s_fp, s_fn)
    # MCC (sample-level binary classification: whether any phase is correctly predicted)
    def _mcc(tp, fp, fn, tn):
        denom = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        if denom <= 0:
            return 0.0
        return float((tp * tn - fp * fn) / np.sqrt(denom))
    # Summarize TP/FP/FN/TN of P/S as "whether P or S was correctly predicted on the sample"
    tp_all = p_tp + s_tp
    fp_all = p_fp + s_fp
    fn_all = p_fn + s_fn
    tn_all = p_tn + s_tn
    mcc = _mcc(tp_all, fp_all, fn_all, tn_all)
    # Residual statistics (seconds)
    if all_p_residual_signed:
        p_arr = np.asarray(all_p_residual_signed, dtype=np.float32) / float(SAMPLE_RATE)
        p_res_mean = float(p_arr.mean())
        p_res_std = float(p_arr.std())
        p_res_mae = float(np.abs(p_arr).mean())
    else:
        p_res_mean = p_res_std = p_res_mae = None
    if all_s_residual_signed:
        s_arr = np.asarray(all_s_residual_signed, dtype=np.float32) / float(SAMPLE_RATE)
        s_res_mean = float(s_arr.mean())
        s_res_std = float(s_arr.std())
        s_res_mae = float(np.abs(s_arr).mean())
    else:
        s_res_mean = s_res_std = s_res_mae = None
    return dict(
        time_acc=float(acc),
        p_prec=float(p_prec),
        p_rec=float(p_rec),
        p_f1=float(p_f1),
        s_prec=float(s_prec),
        s_rec=float(s_rec),
        s_f1=float(s_f1),
        mcc=mcc,
        p_res_mean_sec=p_res_mean,
        p_res_std_sec=p_res_std,
        p_res_mae_sec=p_res_mae,
        s_res_mean_sec=s_res_mean,
        s_res_std_sec=s_res_std,
        s_res_mae_sec=s_res_mae,
        p_residuals_signed=all_p_residual_signed,
        s_residuals_signed=all_s_residual_signed,
    )


def plot_losses(tr, va, out_path: str):
    if plt is None: return
    plt.figure(figsize=(7,4)); xs = list(range(1, len(tr)+1))
    plt.plot(xs, tr, label="train"); plt.plot(xs, va, label="valid")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss Curves"); plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()


@torch.inference_mode()
def save_visuals(model, dataset, device, out_dir: str, n: int = 4):
    if plt is None: return
    os.makedirs(out_dir, exist_ok=True)
    idxs = list(range(len(dataset))); random.shuffle(idxs); idxs = idxs[:n]
    for idx in idxs:
        x_t, y_t, name = dataset[idx]
        x = x_t.unsqueeze(0).to(device); y = y_t.unsqueeze(0).to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        y_np = y.squeeze(0).cpu().numpy(); x_np = x_t.cpu().numpy()
        T = probs.shape[-1]; t = np.arange(T)
        fig, axes = plt.subplots(2,1, figsize=(12,6), sharex=True)
        ax0 = axes[0]; shift = 3.0; colors = ["tab:blue","tab:orange","tab:green"]
        for c in range(min(3, x_np.shape[0])):
            ax0.plot(t, x_np[c, :T] + c*shift, color=colors[c%3])
        if y_np[1].max()>1e-6: ax0.axvline(int(np.argmax(y_np[1])), color='g', ls='--', alpha=.7)
        if y_np[2].max()>1e-6: ax0.axvline(int(np.argmax(y_np[2])), color='b', ls='--', alpha=.7)
        ax0.axvline(int(np.argmax(probs[1])), color='g', ls=':', alpha=.9)
        ax0.axvline(int(np.argmax(probs[2])), color='b', ls=':', alpha=.9)
        ax0.set_title(str(name))
        ax1 = axes[1]
        ax1.plot(t, probs[0], label='BG'); ax1.plot(t, probs[1], label='P'); ax1.plot(t, probs[2], label='S')
        ax1.set_ylim(0,1); ax1.legend(loc='upper right'); ax1.set_xlabel('Time (samples)'); ax1.set_ylabel('Probability')
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, f"vis_{idx}.png"), dpi=150); plt.close(fig)


def train_and_eval(tag: str, build_model: Callable[[], nn.Module], eval_only: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds, valid_ds, test_ds = build_datasets()
    g = torch_generator(SEED)

    # PhaseNet uses original configuration, and other models use global configuration.
    if tag == "PhaseNet":
        cfg = PHASENET_TRAIN_CONFIG.copy()
        cfg["decay_step"] = max(1, len(train_ds) // cfg["batch_size"])
        batch_size = cfg["batch_size"]
        lr = cfg["lr"]
        epochs = cfg["epochs"]
        decay_rate = cfg["decay_rate"]
        decay_step = cfg["decay_step"]
        use_phasenet_scheduler = True
    else:
        batch_size = BATCH_SIZE
        lr = LR
        epochs = EPOCHS
        use_phasenet_scheduler = False

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    test_loader = None
    if test_ds is not None:
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g,
        )

    # The run ID also contains the model name, data source type and random seed to facilitate distinguishing different DATA_SOURCE/seeds
    run_tag = f"{tag}_{DATA_SOURCE}_seed{SEED}"
    model = build_model().to(device)
    params = count_params(model)
    size_mb = params * 4.0 / (1024.0 ** 2)  # The float32 parameter takes up approximately MB of memory.
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    # PhaseNet original text: exponential_decay(staircase=True), decay every decay_step batch
    if use_phasenet_scheduler:
        def lr_lambda(step):
            return decay_rate ** (step // decay_step)
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    else:
        scheduler = None

    print(f"Model {run_tag} params: {count_params(model)/1e6:.2f}M | Target≈{TARGET_PARAMS/1e6:.2f}M")

    # ----------------------------
    # Eval-only: skip training loop
    # ----------------------------
    if eval_only:
        out_dir_eval = os.path.join(OUTPUT_ROOT, run_tag)
        best_path = os.path.join(out_dir_eval, "best_model.pt")
        if not os.path.exists(best_path):
            raise FileNotFoundError(
                f"[baselines_run] eval-only=True but checkpoint not found: {best_path}\n"
                f"Expected path pattern: {os.path.join(OUTPUT_ROOT, f'{tag}_{DATA_SOURCE}_seed{SEED}', 'best_model.pt')}"
            )

        state = torch.load(best_path, map_location=device)
        model.load_state_dict(state)
        model.eval()

        thr_p = 0.5
        thr_s = 0.5
        metrics = eval_detailed(model, valid_loader, device, thr_p, thr_s, tol=DYN_TOL_SAMPLES)

        test_metrics = {}
        if test_loader is not None:
            print(f"[{run_tag}] 在独立测试集上评估...", flush=True)
            test_metrics = eval_detailed(model, test_loader, device, thr_p, thr_s, tol=DYN_TOL_SAMPLES)
            for k in [
                "time_acc",
                "p_prec",
                "p_rec",
                "p_f1",
                "s_prec",
                "s_rec",
                "s_f1",
                "mcc",
                "p_res_mean_sec",
                "p_res_std_sec",
                "p_res_mae_sec",
                "s_res_mean_sec",
                "s_res_std_sec",
                "s_res_mae_sec",
            ]:
                if k in test_metrics:
                    metrics[f"test_{k}"] = test_metrics[k]

        selection = float(BEST_METRIC_WEIGHTS[0]) * float(metrics["p_f1"]) + float(BEST_METRIC_WEIGHTS[1]) * float(metrics["s_f1"])

        # Output directories and files
        os.makedirs(out_dir_eval, exist_ok=True)
        save_visuals(model, valid_ds, device, os.path.join(out_dir_eval, "figs"), n=N_VIS)

        # Visualization of temporal residual distribution (validation set/test set)
        if metrics.get("p_residuals_signed") or metrics.get("s_residuals_signed"):
            p_res = metrics.get("p_residuals_signed", [])
            s_res = metrics.get("s_residuals_signed", [])
            out_path_res = os.path.join(out_dir_eval, "time_residual_distribution_valid.png")
            try:
                plot_time_residual_distribution(
                    residuals_p=p_res,
                    residuals_s=s_res,
                    out_path=out_path_res,
                    sample_rate=SAMPLE_RATE,
                )
                print(f"[{run_tag}] 验证集时间残差分布已保存: {out_path_res}")
            except Exception as e:
                print(f"[{run_tag}] 绘制验证集时间残差分布失败: {e}")

        if test_loader is not None and test_metrics.get("p_residuals_signed") or test_metrics.get("s_residuals_signed"):
            p_res_t = test_metrics.get("p_residuals_signed", [])
            s_res_t = test_metrics.get("s_residuals_signed", [])
            out_path_res_t = os.path.join(out_dir_eval, "time_residual_distribution_test.png")
            try:
                plot_time_residual_distribution(
                    residuals_p=p_res_t,
                    residuals_s=s_res_t,
                    out_path=out_path_res_t,
                    sample_rate=SAMPLE_RATE,
                )
                print(f"[{run_tag}] 测试集时间残差分布已保存: {out_path_res_t}")
            except Exception as e:
                print(f"[{run_tag}] 绘制测试集时间残差分布失败: {e}")

        time_res_path = os.path.join(out_dir_eval, f"time_residuals_{run_tag}.json")
        with open(time_res_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "p_residuals_signed": metrics.get("p_residuals_signed", []),
                    "s_residuals_signed": metrics.get("s_residuals_signed", []),
                },
                f,
                ensure_ascii=False,
            )
        print(f"[{run_tag}] 时间残差已保存: {time_res_path}")

        row = dict(
            name=run_tag,
            best_val=selection,
            train_last=selection,
            valid_last=selection,
            params=params,
            size_mb=size_mb,
            thr_p=thr_p,
            thr_s=thr_s,
            **metrics,
        )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return row

    tr_hist, va_hist = [], []
    best_by_f1 = bool(BEST_MODEL_BY_F1)
    best = -1.0 if best_by_f1 else float("inf")
    best_state_dict = None
    best_epoch = 1
    global_step = 0
    for ep in range(1, epochs + 1):
        model.train(); total_tr = 0.0; n_tr = 0
        for x, y, _ in train_loader:
            x = x.to(device); y = y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            if y.shape[-1] != logits.shape[-1]:
                y = _center_crop_time(y, logits.shape[-1])
            loss = soft_ce(logits, y)
            loss.backward(); opt.step()
            if scheduler is not None:
                scheduler.step()
                global_step += 1
            bs = x.size(0); total_tr += float(loss.item()) * bs; n_tr += bs
        tr_loss = total_tr / max(1, n_tr)
        va_loss = eval_loss(model, valid_loader, device)
        tr_hist.append(tr_loss); va_hist.append(va_loss)

        if best_by_f1:
            thr_p = 0.5
            thr_s = 0.5
            eval_metrics = eval_detailed(model, valid_loader, device, thr_p, thr_s, tol=DYN_TOL_SAMPLES)
            w_p, w_s = float(BEST_METRIC_WEIGHTS[0]), float(BEST_METRIC_WEIGHTS[1])
            selection = w_p * float(eval_metrics["p_f1"]) + w_s * float(eval_metrics["s_f1"])
            if selection > best:
                best = selection
                best_epoch = ep
                best_state_dict = copy.deepcopy(model.state_dict())
            best_label = "best_f1"
            best_disp = best
        else:
            if va_loss < best:
                best = va_loss
                best_epoch = ep
                best_state_dict = copy.deepcopy(model.state_dict())
            best_label = "best_val"
            best_disp = best

        print(
            f"[{run_tag}] epoch {ep:02d}/{epochs} train={tr_loss:.4f} valid={va_loss:.4f} {best_label}={best_disp:.4f}",
            flush=True,
        )

    if SAVE_BEST_MODEL and best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        out_dir = os.path.join(OUTPUT_ROOT, run_tag)
        os.makedirs(out_dir, exist_ok=True)
        best_path = os.path.join(out_dir, "best_model.pt")
        torch.save(best_state_dict, best_path)
        criterion = "best_f1" if best_by_f1 else "best_val_loss"
        print(f"[{run_tag}] 已回滚到验证集最优模型（best_epoch={best_epoch}, {criterion}={best:.4f}），并保存: {best_path}", flush=True)

    # Threshold (optional)
    thr_p = 0.5; thr_s = 0.5

    # Detailed indicators of the verification set
    metrics = eval_detailed(model, valid_loader, device, thr_p, thr_s, tol=DYN_TOL_SAMPLES)

    # Test set evaluation (if independent test_ds exists)
    test_metrics = {}
    if test_loader is not None:
        print(f"[{run_tag}] 在独立测试集上评估...", flush=True)
        test_metrics = eval_detailed(model, test_loader, device, thr_p, thr_s, tol=DYN_TOL_SAMPLES)
        # Only keep key indicators with the same name as the validation set, with the prefix test_
        for k in [
            "time_acc",
            "p_prec",
            "p_rec",
            "p_f1",
            "s_prec",
            "s_rec",
            "s_f1",
            "mcc",
            "p_res_mean_sec",
            "p_res_std_sec",
            "p_res_mae_sec",
            "s_res_mean_sec",
            "s_res_std_sec",
            "s_res_mae_sec",
        ]:
            if k in test_metrics:
                metrics[f"test_{k}"] = test_metrics[k]

    # Output directories and files
    out_dir = os.path.join(OUTPUT_ROOT, run_tag)
    os.makedirs(out_dir, exist_ok=True)
    plot_losses(tr_hist, va_hist, os.path.join(out_dir, "loss_curve.png"))
    save_visuals(model, valid_ds, device, os.path.join(out_dir, "figs"), n=N_VIS)

    # Visualization of temporal residual distribution (validation set/test set)
    # Use sample-level signed residual (unit: sampling point), consistent with single_ablation_unet_core
    if metrics.get("p_residuals_signed") or metrics.get("s_residuals_signed"):
        p_res = metrics.get("p_residuals_signed", [])
        s_res = metrics.get("s_residuals_signed", [])
        out_path_res = os.path.join(out_dir, "time_residual_distribution_valid.png")
        try:
            plot_time_residual_distribution(
                residuals_p=p_res,
                residuals_s=s_res,
                out_path=out_path_res,
                sample_rate=SAMPLE_RATE,
            )
            print(f"[{run_tag}] 验证集时间残差分布已保存: {out_path_res}")
        except Exception as e:
            print(f"[{run_tag}] 绘制验证集时间残差分布失败: {e}")

    if test_loader is not None and test_metrics.get("p_residuals_signed") or test_metrics.get("s_residuals_signed"):
        p_res_t = test_metrics.get("p_residuals_signed", [])
        s_res_t = test_metrics.get("s_residuals_signed", [])
        out_path_res_t = os.path.join(out_dir, "time_residual_distribution_test.png")
        try:
            plot_time_residual_distribution(
                residuals_p=p_res_t,
                residuals_s=s_res_t,
                out_path=out_path_res_t,
                sample_rate=SAMPLE_RATE,
            )
            print(f"[{run_tag}] 测试集时间残差分布已保存: {out_path_res_t}")
        except Exception as e:
            print(f"[{run_tag}] 绘制测试集时间残差分布失败: {e}")

    # Save the time residual list in the same format as single_ablation_unet_core, which is convenient for comparison with plot_time_residual_grid and other methods.
    time_res_path = os.path.join(out_dir, f"time_residuals_{run_tag}.json")
    with open(time_res_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "p_residuals_signed": metrics.get("p_residuals_signed", []),
                "s_residuals_signed": metrics.get("s_residuals_signed", []),
            },
            f,
            ensure_ascii=False,
        )
    print(f"[{run_tag}] 时间残差已保存: {time_res_path}")

    row = dict(
        name=run_tag,
        best_val=best,
        train_last=tr_hist[-1],
        valid_last=va_hist[-1],
        params=params,
        size_mb=size_mb,
        thr_p=thr_p,
        thr_s=thr_s,
        **metrics,
    )

    # Explicitly release the GPU memory of the current model to avoid OOM of the next baseline in the same process
    del model, opt, best_state_dict, train_loader, valid_loader
    if test_loader is not None:
        del test_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return row


def append_csv(path: str, header: list[str], rows: list[Dict[str, Any]]):
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        for r in rows:
            def _fmt_res(key: str) -> str:
                v = r.get(key, None)
                return f"{float(v):.4f}" if v is not None else "N/A"
            # The number of parameters is converted to k (thousands), size_mb is still MB (M), and only the numerical value is scaled and displayed.
            params_raw = int(r.get("params", 0))
            params_k = params_raw / 1_000.0
            size_mb_raw = float(r.get("size_mb", 0.0))
            w.writerow([
                r["name"],
                f"{r['best_val']:.6f}",
                f"{r['train_last']:.6f}",
                f"{r['valid_last']:.6f}",
                f"{params_k:.1f}",          # The params column is in units of k
                f"{size_mb_raw:.1f}",       # The size_mb column is in MB, denoted as M
                f"{r['p_prec']:.4f}", f"{r['p_rec']:.4f}", f"{r['p_f1']:.4f}",
                f"{r['s_prec']:.4f}", f"{r['s_rec']:.4f}", f"{r['s_f1']:.4f}",
                f"{r.get('time_acc', 0.0):.4f}",
                f"{r.get('mcc', 0.0):.4f}",
                _fmt_res("p_res_mean_sec"), _fmt_res("p_res_std_sec"), _fmt_res("p_res_mae_sec"),
                _fmt_res("s_res_mean_sec"), _fmt_res("s_res_std_sec"), _fmt_res("s_res_mae_sec"),
                # Test set metric (if there is no test set, the value is N/A)
                _fmt_res("test_p_prec"), _fmt_res("test_p_rec"), _fmt_res("test_p_f1"),
                _fmt_res("test_s_prec"), _fmt_res("test_s_rec"), _fmt_res("test_s_f1"),
                _fmt_res("test_time_acc"), _fmt_res("test_mcc"),
                _fmt_res("test_p_res_mean_sec"), _fmt_res("test_p_res_std_sec"), _fmt_res("test_p_res_mae_sec"),
                _fmt_res("test_s_res_mean_sec"), _fmt_res("test_s_res_std_sec"), _fmt_res("test_s_res_mae_sec"),
            ])


def main():
    global SEED
    parser = argparse.ArgumentParser(description="Train baseline models (TCN / ResASPP / Inception / PhaseNet) on CEED/NPZ.")
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="随机种子（与 single_test_all.py 保持一致，默认 2025）",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="GPU ID，单个如 0，多个如 0,1（不指定则按 torch.cuda.is_available() 自动选择）",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="仅加载已训练好的 best_model.pt 并重新评估（跳过训练）。checkpoint 路径为 OUTPUT_ROOT/run_tag/best_model.pt。"
    )
    args = parser.parse_args()

    SEED = int(args.seed)
    print(f"[baselines_run] 使用随机种子: {SEED}", flush=True)

    # Specify the visible GPU (if not specified, the PyTorch default visible device is used)
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"[baselines_run] 指定可见 GPU: {args.gpu}", flush=True)

    # Fixed randomness (including Python/NumPy/PyTorch; turn off cuDNN benchmark)
    seed_everything(SEED, deterministic=True)
    eval_only = bool(getattr(args, "eval_only", False))
    models: list[tuple[str, Callable[[], nn.Module]]] = [
        ("BaselineTCN", lambda: BaselineTCN(in_ch=3, n_class=3, width=128, n_blocks=10)),
        ("BaselineResASPP", lambda: BaselineResASPP(in_ch=3, n_class=3, width=160, n_blocks=6)),
        ("BaselineInception", lambda: BaselineInception(in_ch=3, n_class=3, width=128, n_blocks=8)),
        # PhaseNet 1D version of the original UNet structure (as a baseline for comparison)
        ("PhaseNet", lambda: PhaseNetUNet(in_ch=3, n_class=3, depths=5, filters_root=8, kernel_size=7, pool_size=4, drop_rate=0.0)),
    ]

    # CSV column: basic indicators + parameter amount/model size + time residual statistics for easy alignment and comparison with PhaseRiskNet
    header = [
        "name", "best_val", "train_last", "valid_last",
        "params", "size_mb",
        # Validation set metrics
        "p_prec", "p_rec", "p_f1",
        "s_prec", "s_rec", "s_f1",
        "time_acc", "mcc",
        "p_res_mean_sec", "p_res_std_sec", "p_res_mae_sec",
        "s_res_mean_sec", "s_res_std_sec", "s_res_mae_sec",
        # Test set metrics (written if test_ds exists; N/A otherwise)
        "test_p_prec", "test_p_rec", "test_p_f1",
        "test_s_prec", "test_s_rec", "test_s_f1",
        "test_time_acc", "test_mcc",
        "test_p_res_mean_sec", "test_p_res_std_sec", "test_p_res_mae_sec",
        "test_s_res_mean_sec", "test_s_res_std_sec", "test_s_res_mae_sec",
    ]
    # First read the existing metrics.csv. If a <model>_seed<SEED> has been written, skip it.
    existing_names: set[str] = set()
    if os.path.exists(METRICS_CSV):
        try:
            with open(METRICS_CSV, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    n = (row.get("name") or "").strip()
                    if n:
                        existing_names.add(n)
            print(f"[baselines_run] 已存在 {len(existing_names)} 条历史记录，将自动跳过重复项。", flush=True)
        except Exception as e:
            print(f"[baselines_run] 读取已有 {METRICS_CSV} 失败，将不做去重跳过：{e}", flush=True)

    print(f"[baselines_run] 指标将逐模型追加写入: {METRICS_CSV}", flush=True)
    ds_tag = DATA_SOURCE
    for name, builder in models:
        # Add the data set type to name to avoid confusion of results under different DATA_SOURCE
        expected_name = f"{name}_{ds_tag}_seed{SEED}"
        if expected_name in existing_names:
            print(f"\n=== Skip baseline (already exists): {expected_name} ===", flush=True)
            continue
        print(f"\n=== Running baseline: {name} ===", flush=True)
        try:
            r = train_and_eval(name, builder, eval_only=eval_only)
        except Exception as e:
            print(f"[baselines_run] 运行 {name} 失败，已跳过：{e}", flush=True)
            continue

        mapped = {
            "name": r["name"],
            "best_val": r["best_val"],
            "train_last": r["train_last"],
            "valid_last": r["valid_last"],
            "params": r.get("params", 0),
            "size_mb": r.get("size_mb", 0.0),
            "p_prec": r["p_prec"],
            "p_rec": r["p_rec"],
            "p_f1": r["p_f1"],
            "s_prec": r["s_prec"],
            "s_rec": r["s_rec"],
            "s_f1": r["s_f1"],
            "time_acc": r.get("time_acc", 0.0),
            "mcc": r.get("mcc", 0.0),
            "p_res_mean_sec": r.get("p_res_mean_sec", None),
            "p_res_std_sec": r.get("p_res_std_sec", None),
            "p_res_mae_sec": r.get("p_res_mae_sec", None),
            "s_res_mean_sec": r.get("s_res_mean_sec", None),
            "s_res_std_sec": r.get("s_res_std_sec", None),
            "s_res_mae_sec": r.get("s_res_mae_sec", None),
            "test_p_prec": r.get("test_p_prec", None),
            "test_p_rec": r.get("test_p_rec", None),
            "test_p_f1": r.get("test_p_f1", None),
            "test_s_prec": r.get("test_s_prec", None),
            "test_s_rec": r.get("test_s_rec", None),
            "test_s_f1": r.get("test_s_f1", None),
            "test_time_acc": r.get("test_time_acc", None),
            "test_mcc": r.get("test_mcc", None),
            "test_p_res_mean_sec": r.get("test_p_res_mean_sec", None),
            "test_p_res_std_sec": r.get("test_p_res_std_sec", None),
            "test_p_res_mae_sec": r.get("test_p_res_mae_sec", None),
            "test_s_res_mean_sec": r.get("test_s_res_mean_sec", None),
            "test_s_res_std_sec": r.get("test_s_res_std_sec", None),
            "test_s_res_mae_sec": r.get("test_s_res_mae_sec", None),
        }
        try:
            append_csv(METRICS_CSV, header, [mapped])
            print(f"[baselines_run] 已写入一行指标: {mapped['name']}", flush=True)
            existing_names.add(str(mapped["name"]))
        except Exception as e:
            print(f"[baselines_run] 写入 {METRICS_CSV} 失败（{mapped.get('name','?')}）：{e}", flush=True)


if __name__ == "__main__":
    main()
