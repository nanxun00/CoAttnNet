"""
Figure 5: CEED baseline comparison; Figure 6: Diting baseline comparison.
Each graph is (a) detection performance s_rec / s_f1 / p_f1 grouped histogram + (b) p_res_mae_sec / s_res_mae_sec grouped histogram.
Color scheme: Scheme 2, fresh and technological style (Google color system, modern and refreshing).

usage:
  python plot_baseline_ceed_diting.py
  python plot_baseline_ceed_diting.py -o fig5.png --dataset ceed
  python plot_baseline_ceed_diting.py -o fig6.png --dataset diting
  python plot_baseline_ceed_diting.py --dataset all
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Comment translated from Chinese.
MODEL_ORDER: List[str] = [
    "GLANet",
    "GLANet-G",
    "PhaseNet",
    "EQTransformer",
    "TCN",
    "UNet",
    "Inception",
    "ResASPP",
    "EPick",
]

# Comment translated from Chinese.
MODEL_LABELS: Dict[str, str] = {
    "GLANet": "GLANet",
    "GLANet-G": "GLANet-G",
    "PhaseNet": "PhaseNet",
    "EQTransformer": "EQTransformer",
    "TCN": "TCN",
    "UNet": "UNet",
    "Inception": "Inception",
    "ResASPP": "ResASPP",
    "EPick": "EPick",
}

# CSV ± model
# s_rec, s_f1, p_f1, p_res_mae_sec, s_res_mae_sec
CEED_METRICS: Dict[str, Tuple[float, float, float, float, float]] = {
    "PhaseNet": (0.9382, 0.8887, 0.9745, 0.1301, 0.1407),
    "EQTransformer": (0.9366, 0.8689, 0.9601, 0.1043, 0.1514),
    "TCN": (0.9197, 0.8758, 0.9534, 0.2839, 0.2950),
    "UNet": (0.9007, 0.8563, 0.9531, 0.1608, 0.1757),
    "Inception": (0.9488, 0.8635, 0.9455, 0.4753, 0.4024),
    "ResASPP": (0.8914, 0.8369, 0.9316, 0.5152, 0.4859),
    "EPick": (0.8893, 0.8516, 0.9589, 0.1155, 0.1829),
    "GLANet": (0.9557, 0.9006, 0.9797, 0.0761, 0.1003),
    "GLANet-G": (0.9972, 0.9084, 0.9808, 0.0761, 0.1003),
}

DITING_METRICS: Dict[str, Tuple[float, float, float, float, float]] = {
    "PhaseNet": (0.4191, 0.3732, 1.0000, 0.0006, 1.2296),
    "EQTransformer": (0.4091, 0.2811, 1.0000, 0.0000, 1.1883),
    "TCN": (0.1620, 0.2311, 0.5801, 0.7542, 2.1492),
    "UNet": (0.0000, 0.0000, 1.0000, 0.0139, 11.2669),
    "Inception": (0.1200, 0.1893, 0.5105, 1.4805, 3.6823),
    "ResASPP": (0.1126, 0.1621, 0.5429, 2.0789, 3.9825),
    "EPick": (0.2291, 0.2895, 1.0000, 0.0000, 0.8920),
    "GLANet": (0.7526, 0.4724, 1.0000, 0.0050, 0.7386),
    "GLANet-G": (0.7526, 0.4724, 1.0000, 0.0050, 0.7386),
}

# plot_ablation_detection_mae.py / / /
C_S_REC = "#AFCFA8"
C_S_F1 = "#EAC0A2"
C_P_F1 = "#AFC4E3"
C_S_MAE = "#AFCFA8"
C_P_MAE = "#90A7CC"
EDGE_S_BLUE = "#8FB88A"
EDGE_S_F1 = "#C8926E"
EDGE_P_F1 = "#7D95BE"
EDGE_P_MAE = "#6D84AD"


def _configure_font() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Calibri", "DejaVu Sans", "Helvetica", "sans-serif"],
            "axes.unicode_minus": False,
        }
    )


def _stack_rows(order: List[str], table: Dict[str, Tuple[float, float, float, float, float]]):
    s_rec, s_f1, p_f1, p_mae, s_mae = [], [], [], [], []
    labels: List[str] = []
    for m in order:
        if m not in table:
            raise KeyError(f"Missing model: {m}")
        sr, sf, pf, pm, sm = table[m]
        s_rec.append(sr)
        s_f1.append(sf)
        p_f1.append(pf)
        p_mae.append(pm)
        s_mae.append(sm)
        labels.append(MODEL_LABELS[m])
    return (
        np.array(s_rec, dtype=np.float64),
        np.array(s_f1, dtype=np.float64),
        np.array(p_f1, dtype=np.float64),
        np.array(p_mae, dtype=np.float64),
        np.array(s_mae, dtype=np.float64),
        labels,
    )


def plot_one_figure(
    dataset_title: str,
    table: Dict[str, Tuple[float, float, float, float, float]],
    out_path: str,
) -> None:
    _configure_font()
    s_rec, s_f1, p_f1, p_mae, s_mae, xlabels = _stack_rows(MODEL_ORDER, table)
    n = len(xlabels)
    x = np.arange(n, dtype=float)
    width = 0.22
    w_sr = 0.26

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(15.0, 5.2), constrained_layout=True)

    # (a)
    bars_sr = ax_a.bar(
        x - width,
        s_rec,
        w_sr,
        label="S-Recall",
        color=C_S_REC,
        edgecolor=EDGE_S_BLUE,
        linewidth=0.85,
        zorder=3,
    )
    bars_sf = ax_a.bar(
        x,
        s_f1,
        width,
        label="S-F1",
        color=C_S_F1,
        edgecolor=EDGE_S_F1,
        linewidth=0.5,
        zorder=2,
    )
    bars_pf = ax_a.bar(
        x + width,
        p_f1,
        width,
        label="P-F1",
        color=C_P_F1,
        edgecolor=EDGE_P_F1,
        linewidth=0.5,
        alpha=0.92,
        zorder=1,
    )

    # PhaseRiskNet
    for rects in (bars_sr, bars_sf, bars_pf):
        rects[0].set_linewidth(1.2)
        rects[0].set_edgecolor("#111827")

    ymin_det = float(np.min([s_rec.min(), s_f1.min(), p_f1.min()]))
    ymax_det = float(np.max([s_rec.max(), s_f1.max(), p_f1.max()]))
    pad = max(0.02, (ymax_det - ymin_det) * 0.08)
    ax_a.set_ylim(max(0.0, ymin_det - pad), min(1.0, ymax_det + pad))

    ax_a.set_xticks(x)
    ax_a.set_xticklabels(xlabels, rotation=28, ha="right", fontsize=8)
    ax_a.set_ylabel("Score", fontsize=11)
    ax_a.grid(axis="y", linestyle="--", alpha=0.35)
    ax_a.legend(loc="lower right", fontsize=8, framealpha=0.95)

    # Diting 1.0 GLANet / GLANet-G 1.0000
    is_diting = dataset_title.strip().lower() == "diting"
    prn_only = frozenset({"GLANet", "GLANet-G"})

    def autolabel3(rects, fmt="{:.4f}", prn_idx: int = 0):
        for i, rect in enumerate(rects):
            h = rect.get_height()
            if is_diting and np.isclose(h, 1.0, rtol=0, atol=1e-5):
                if MODEL_ORDER[i] not in prn_only:
                    continue
            fs = 6.2 if i == prn_idx else 5.0
            fw = "bold" if i == prn_idx else "normal"
            ax_a.annotate(
                fmt.format(h),
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=fs,
                fontweight=fw,
                rotation=90,
            )

    autolabel3(bars_sr)
    autolabel3(bars_sf)
    autolabel3(bars_pf)

    ax_a.text(
        0.02,
        1.02,
        "(a)",
        transform=ax_a.transAxes,
        fontsize=12,
        fontweight="bold",
        va="bottom",
        clip_on=False,
    )
    ax_a.set_title(f"{dataset_title} — Detection performance", fontsize=11, fontweight="bold")

    # (b) MAE
    w2 = 0.32
    bars_s = ax_b.bar(
        x - w2 / 2,
        s_mae,
        w2,
        label="S-Res MAE",
        color=C_S_MAE,
        edgecolor=EDGE_S_BLUE,
        linewidth=0.7,
        zorder=2,
    )
    bars_p = ax_b.bar(
        x + w2 / 2,
        p_mae,
        w2,
        label="P-Res MAE",
        color=C_P_MAE,
        edgecolor=EDGE_P_MAE,
        linewidth=0.7,
        zorder=1,
    )

    for r in (bars_s[0], bars_p[0]):
        r.set_linewidth(1.2)
        r.set_edgecolor("#111827")

    ymin_m = float(np.min([s_mae.min(), p_mae.min()]))
    ymax_m = float(np.max([s_mae.max(), p_mae.max()]))
    pad_m = max(0.02, (ymax_m - ymin_m) * 0.06)
    ax_b.set_ylim(max(0.0, ymin_m - pad_m), ymax_m + pad_m)

    ax_b.set_xticks(x)
    ax_b.set_xticklabels(xlabels, rotation=28, ha="right", fontsize=8)
    ax_b.set_ylabel("Mean Absolute Error (s)", fontsize=11)
    ax_b.grid(axis="y", linestyle="--", alpha=0.35)
    ax_b.legend(loc="upper right", fontsize=8, framealpha=0.95)

    for i, rect in enumerate(bars_s):
        h = rect.get_height()
        fs = 6.5 if i == 0 else 5.5
        fw = "bold" if i == 0 else "normal"
        ytxt = 3 if h > 1e-6 else 8
        va = "bottom" if h > 1e-6 else "bottom"
        ax_b.annotate(
            f"{h:.4f}",
            xy=(rect.get_x() + rect.get_width() / 2, max(h, 1e-6)),
            xytext=(0, ytxt),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=fs,
            fontweight=fw,
            color=C_S_MAE,
        )
    for i, rect in enumerate(bars_p):
        h = rect.get_height()
        fs = 6.5 if i == 0 else 5.5
        fw = "bold" if i == 0 else "normal"
        # 14pt 9pt 0.0000
        y_off_red = 9 if h > 1e-6 else 6
        ax_b.annotate(
            f"{h:.4f}",
            xy=(rect.get_x() + rect.get_width() / 2, max(h, 1e-6)),
            xytext=(0, y_off_red),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fs,
            fontweight=fw,
            color=C_P_MAE,
        )

    ax_b.text(
        0.02,
        1.02,
        "(b)",
        transform=ax_b.transAxes,
        fontsize=12,
        fontweight="bold",
        va="bottom",
        clip_on=False,
    )
    ax_b.set_title(f"{dataset_title} — Arrival-time MAE", fontsize=11, fontweight="bold")

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {os.path.abspath(out_path)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--out",
        default=None,
        help="输出路径；仅在与 --dataset ceed|diting 联用时生效",
    )
    parser.add_argument(
        "--dataset",
        choices=["ceed", "diting", "all"],
        default="all",
        help="ceed / diting / 一次生成两张",
    )
    args = parser.parse_args()
    base = os.path.dirname(os.path.abspath(__file__))

    if args.dataset in ("ceed", "all"):
        out = (
            args.out
            if args.dataset == "ceed" and args.out
            else os.path.join(base, "figure5_baseline_ceed.png")
        )
        plot_one_figure("CEED", CEED_METRICS, out)

    if args.dataset in ("diting", "all"):
        out = (
            args.out
            if args.dataset == "diting" and args.out
            else os.path.join(base, "figure6_baseline_diting.png")
        )
        plot_one_figure("Diting", DITING_METRICS, out)


if __name__ == "__main__":
    main()
