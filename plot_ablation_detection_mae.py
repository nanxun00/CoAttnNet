"""
Ablation experiment visualization:
- Figure (a) Plot only P-F1 / S-F1
- Figure (b) plots only p_res_mae_sec / s_res_mae_sec

usage:
  python plot_ablation_detection_mae.py
  python plot_ablation_detection_mae.py -o my_ablation_fig.png
"""

from __future__ import annotations

import argparse
import csv
import io
import os
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# (CSV name, )
ABLATION_ORDER: List[Tuple[str, str]] = [
    ("Base", "Base"),
    ("Base + Multi", "Base+Multi"),
    ("Base + Multi + Sg", "Base+Multi+Sg"),
    ("Base + Multi + CBAM", "Base+Multi+CBAM"),
    ("Base + Multi + CBAM + Sg", "Base+Multi+CBAM+Sg"),
]

# / / /
C_P_F1 = "#AFC4E3"   # Comment translated from Chinese.
C_S_F1 = "#EAC0A2"   # CBAM
C_S_MAE = "#AFCFA8"  # Conv
C_P_MAE = "#90A7CC"  # S-MAE
EDGE_S_F1 = "#C8926E"
EDGE_P_F1 = "#7D95BE"

CSV_TEXT = """name,p_prec,p_rec,p_f1,s_prec,s_rec,s_f1,p_res_mae_sec,s_res_mae_sec
Base,0.9341±0.0022,0.9636±0.0098,0.9486±0.0058,0.8176±0.0095,0.9340±0.0158,0.8719±0.0119,0.1429±0.0258,0.1522±0.0096
Base + Multi,0.9627±0.0023,0.9849±0.0006,0.9737±0.0012,0.8638±0.0048,0.9343±0.0094,0.8977±0.0044,0.0640±0.0170,0.1058±0.0125
Base + Multi + Sg,0.9666±0.0012,0.9808±0.0055,0.9737±0.0028,0.8624±0.0102,0.9469±0.0063,0.9027±0.0030,0.0603±0.0094,0.1153±0.0082
Base + Multi + CBAM,0.9633±0.0021,0.9847±0.0030,0.9739±0.0021,0.8527±0.0084,0.9674±0.0036,0.9064±0.0038,0.0547±0.0045,0.1098±0.0161
Base + Multi + CBAM + Sg,0.9635±0.0026,0.9870±0.0052,0.9751±0.0023,0.8655±0.0061,0.9530±0.0101,0.9072±0.0063,0.0686±0.0284,0.1057±0.0052
"""


def parse_mean(cell: str) -> float:
    s = cell.strip().split("±")[0].strip()
    return float(s)


def load_rows(csv_str: str) -> Dict[str, Dict[str, float]]:
    reader = csv.DictReader(io.StringIO(csv_str.strip()))
    out: Dict[str, Dict[str, float]] = {}
    for row in reader:
        name = row["name"].strip()
        out[name] = {k: parse_mean(v) for k, v in row.items() if k != "name"}
        out[name]["name"] = name  # type: ignore
    return out


def build_arrays(
    table: Dict[str, Dict[str, float]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    s_f1, p_f1, p_mae, s_mae = [], [], [], []
    labels: List[str] = []
    for key, short in ABLATION_ORDER:
        if key not in table:
            raise KeyError(f"Missing row in CSV: {key}")
        r = table[key]
        s_f1.append(r["s_f1"])
        p_f1.append(r["p_f1"])
        p_mae.append(r["p_res_mae_sec"])
        s_mae.append(r["s_res_mae_sec"])
        labels.append(short)
    return (
        np.array(s_f1, dtype=np.float64),
        np.array(p_f1, dtype=np.float64),
        np.array(p_mae, dtype=np.float64),
        np.array(s_mae, dtype=np.float64),
        labels,
    )


def plot_figure(out_path: str) -> None:
    table = load_rows(CSV_TEXT)
    s_f1, p_f1, p_mae, s_mae, xlabels = build_arrays(table)

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(14.5, 5.2), constrained_layout=True
    )

    # (a) P-F1 / S-F1
    n = len(xlabels)
    x = np.arange(n, dtype=float)
    width = 0.32
    bars_sf = ax_a.bar(
        x - width / 2,
        s_f1,
        width,
        label="S-F1",
        color=C_S_F1,
        edgecolor=EDGE_S_F1,
        linewidth=0.5,
        zorder=2,
    )
    bars_pf = ax_a.bar(
        x + width / 2,
        p_f1,
        width,
        label="P-F1",
        color=C_P_F1,
        edgecolor=EDGE_P_F1,
        linewidth=0.5,
        alpha=0.92,
        zorder=1,
    )

    ax_a.set_xticks(x)
    ax_a.set_xticklabels(xlabels, rotation=18, ha="right", fontsize=9)
    ax_a.set_ylabel("Score", fontsize=11)
    ax_a.set_ylim(0.86, 0.99)
    ax_a.axhline(1.0, color="#e5e7eb", linewidth=0.8, zorder=0)
    ax_a.grid(axis="y", linestyle="--", alpha=0.35)
    ax_a.legend(loc="lower right", fontsize=9, framealpha=0.95, ncol=2)

    # Comment translated from Chinese.
    def autolabel_bars(rects, fmt="{:.4f}"):
        for rect in rects:
            h = rect.get_height()
            x_center = rect.get_x() + rect.get_width() / 2
            ax_a.annotate(
                fmt.format(h),
                xy=(x_center, h),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )

    autolabel_bars(bars_sf)
    autolabel_bars(bars_pf)

    ax_a.text(
        0.02,
        1.05,
        "(a)",
        transform=ax_a.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        clip_on=False,
    )
    ax_a.set_title("P/S F1 comparison", fontsize=11, fontweight="bold")

    # (b) MAE
    xi = np.arange(n)
    ax_b.plot(
        xi,
        s_mae,
        marker="o",
        linewidth=2.0,
        markersize=7,
        color=C_S_MAE,
        label="S-Res MAE",
        zorder=3,
    )
    ax_b.plot(
        xi,
        p_mae,
        marker="s",
        linewidth=2.0,
        markersize=6,
        color=C_P_MAE,
        label="P-Res MAE",
        zorder=2,
    )

    ax_b.set_xticks(xi)
    ax_b.set_xticklabels(xlabels, rotation=18, ha="right", fontsize=9)
    ax_b.set_ylabel("Mean Absolute Error (s)", fontsize=11)
    ax_b.set_ylim(0.0, 0.16)
    ax_b.grid(axis="y", linestyle="--", alpha=0.35)
    ax_b.legend(loc="upper right", fontsize=9, framealpha=0.95)

    for i in range(n):
        ax_b.annotate(
            f"{s_mae[i]:.4f}",
            (xi[i], s_mae[i]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=7,
            color=C_S_MAE,
        )
        ax_b.annotate(
            f"{p_mae[i]:.4f}",
            (xi[i], p_mae[i]),
            textcoords="offset points",
            xytext=(0, -12),
            ha="center",
            fontsize=7,
            color=C_P_MAE,
        )

    ax_b.text(
        0.02,
        1.05,
        "(b)",
        transform=ax_b.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        clip_on=False,
    )
    ax_b.set_title("Arrival-time MAE", fontsize=11, fontweight="bold")

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {os.path.abspath(out_path)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--out",
        default="ablation_detection_mae_figure.png",
        help="输出图片路径",
    )
    args = parser.parse_args()
    plot_figure(args.out)


if __name__ == "__main__":
    main()
