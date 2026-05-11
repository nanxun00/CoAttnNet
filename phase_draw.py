import argparse
import logging
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.lines import Line2D


MODEL_ORDER = [
    "PhaseRiskNet",
    "PhaseRiskNet-S",
    "TCN",
    # "UNet", #
    "Inception",
    "ResASPP",
    "PhaseNet",
    "EQTransformer",
    "EPick",
]

PREFERRED_MODEL_ORDER = [
    "PhaseRiskNet",
    "PhaseRiskNet-S",
    "PhaseNet",
    "EQTransformer",
    "UNet",
]

COLOR_Z = "#1f77b4"
COLOR_N = "#17becf"
COLOR_E = "#2ca02c"
COLOR_P = "#ff7f0e"
COLOR_S = "#9467bd"
COLOR_VP = "#d62728"
COLOR_VS = "#8c564b"


def _pick_index(curve: np.ndarray) -> int:
    return int(np.argmax(np.asarray(curve, dtype=float)))


def _sec_to_idx(arrival_sec: float, n_points: int, duration_sec: float) -> int:
    x = float(arrival_sec) / max(duration_sec, 1e-8) * float(n_points)
    return int(np.clip(np.round(x), 0, max(0, n_points - 1)))


def _to_ps_channels(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if y.ndim == 1:
        p = y.astype(float)
        s = np.zeros_like(p)
        return p, s
    if y.ndim != 2:
        raise ValueError(f"模型输出维度应为1或2，得到形状 {y.shape}")
    arr = y
    if arr.shape[0] > arr.shape[1] and arr.shape[1] in (2, 3):
        arr = arr.T
    if arr.shape[0] == 2:
        return arr[0].astype(float), arr[1].astype(float)
    if arr.shape[0] == 3:
        return arr[1].astype(float), arr[2].astype(float)
    p = arr[0].astype(float)
    s = arr[1].astype(float) if arr.shape[0] > 1 else np.zeros_like(p)
    return p, s


def _idx_to_sec(idx: int, n_points: int, duration_sec: float) -> float:
    if n_points <= 0:
        return 0.0
    return float(idx) / float(n_points) * float(duration_sec)


def _abs_ps_index_errors(
    model_out: np.ndarray,
    p_sec: float,
    s_sec: float,
    duration_sec: float,
) -> Tuple[float, float]:
    """Returns the absolute error of P/S at the sampling point; returns inf when the label is invalid."""
    if (not np.isfinite(p_sec)) or (not np.isfinite(s_sec)):
        return float("inf"), float("inf")
    p_curve, s_curve = _to_ps_channels(model_out)
    t_len = min(len(p_curve), len(s_curve))
    if t_len <= 0:
        return float("inf"), float("inf")
    p_pred = _pick_index(p_curve[:t_len])
    s_pred = _pick_index(s_curve[:t_len])
    p_gt = _sec_to_idx(float(p_sec), t_len, duration_sec)
    s_gt = _sec_to_idx(float(s_sec), t_len, duration_sec)
    return abs(float(p_pred - p_gt)), abs(float(s_pred - s_gt))


def _combined_primary_index_error(
    ds: Dict[str, np.ndarray],
    idx: int,
    p_sec: float,
    s_sec: float,
    duration_sec: float,
) -> float:
    """Sum of P/S absolute errors of PhaseRiskNet and PhaseRiskNet-S (4 items in total)."""
    primary = ["PhaseRiskNet", "PhaseRiskNet-S"]
    total = 0.0
    for m in primary:
        e_p, e_s = _abs_ps_index_errors(ds[m][idx], p_sec, s_sec, duration_sec)
        total += e_p + e_s
    return total


def _others_wrong_count(
    ds: Dict[str, np.ndarray],
    idx: int,
    p_sec: float,
    s_sec: float,
    duration_sec: float,
    tol_samples: int,
) -> int:
    # phase_pick strict-contrast EQTransformer/EPick
    others = [
        m
        for m in MODEL_ORDER
        if m not in ("PhaseRiskNet", "PhaseRiskNet-S", "EQTransformer", "EPick")
    ]
    n_wrong = 0
    for m in others:
        if m not in ds:
            continue
        if not _is_model_correct_one_sample(ds[m][idx], p_sec, s_sec, duration_sec, tol_samples):
            n_wrong += 1
    return n_wrong


def _fallback_indices_min_primary_max_others_wrong(
    ds: Dict[str, np.ndarray],
    p_arr: np.ndarray,
    s_arr: np.ndarray,
    duration_sec: float,
    tol_samples: int,
    need_count: int,
    exclude: set,
) -> List[int]:
    """
    The final answer: take the first need_count items according to (main model comprehensive index error in ascending order, other model error numbers in descending order).
    """
    primary = ["PhaseRiskNet", "PhaseRiskNet-S"]
    n = int(len(p_arr))
    rows: List[Tuple[float, int, int]] = []
    for idx in range(n):
        if any(m not in ds for m in primary):
            continue
        p_sec = float(p_arr[idx])
        s_sec = float(s_arr[idx])
        ce = _combined_primary_index_error(ds, idx, p_sec, s_sec, duration_sec)
        if not np.isfinite(ce):
            continue
        wc = _others_wrong_count(ds, idx, p_sec, s_sec, duration_sec, tol_samples)
        # (ce , -wc wc )
        rows.append((ce, -wc, idx))
    rows.sort()
    out: List[int] = []
    for _ce, _neg_wc, idx in rows:
        if idx in exclude:
            continue
        out.append(idx)
        if len(out) >= need_count:
            break
    return out


def _is_model_correct_one_sample(
    model_out: np.ndarray,
    p_sec: float,
    s_sec: float,
    duration_sec: float,
    tol_samples: int,
) -> bool:
    if (not np.isfinite(p_sec)) or (not np.isfinite(s_sec)):
        return False
    p_curve, s_curve = _to_ps_channels(model_out)
    t_len = min(len(p_curve), len(s_curve))
    if t_len <= 0:
        return False
    p_pred = _pick_index(p_curve[:t_len])
    s_pred = _pick_index(s_curve[:t_len])
    p_gt = _sec_to_idx(float(p_sec), t_len, duration_sec)
    s_gt = _sec_to_idx(float(s_sec), t_len, duration_sec)
    return abs(p_pred - p_gt) <= tol_samples and abs(s_pred - s_gt) <= tol_samples


def _auto_select_indices(
    ds: Dict[str, np.ndarray],
    p_arr: np.ndarray,
    s_arr: np.ndarray,
    duration_sec: float,
    tol_samples: int,
    need_count: int = 4,
) -> Tuple[list[int], int, int, bool]:
    primary = ["PhaseRiskNet", "PhaseRiskNet-S"]
    tf_external = frozenset({"EQTransformer", "EPick"})
    # EQT/EPick
    others_compare = [m for m in MODEL_ORDER if m not in primary and m not in tf_external]
    strict_chosen: list[int] = []   # PyTorch EQT/EPick
    loose_chosen: list[int] = []    # EQTransformer/EPick/PhaseNet
    ranked_primary_ok: list[Tuple[int, int]] = []  # (idx, wrong_count)
    strict_total = 0
    loose_total = 0
    key_others = ["EQTransformer", "EPick", "PhaseNet"]
    n = int(len(p_arr))
    for idx in range(n):
        if any(m not in ds for m in primary):
            continue
        ok_primary = all(
            _is_model_correct_one_sample(ds[m][idx], float(p_arr[idx]), float(s_arr[idx]), duration_sec, tol_samples)
            for m in primary
        )
        if not ok_primary:
            continue
        other_flags = [
            not _is_model_correct_one_sample(ds[m][idx], float(p_arr[idx]), float(s_arr[idx]), duration_sec, tol_samples)
            for m in others_compare
            if m in ds
        ]
        if len(other_flags) == 0:
            continue
        wrong_count = int(sum(1 for f in other_flags if f))
        ranked_primary_ok.append((idx, wrong_count))
        all_others_wrong = all(other_flags)
        key_in_ds = [m for m in key_others if m in ds]
        key_others_wrong = bool(key_in_ds) and all(
            not _is_model_correct_one_sample(ds[m][idx], float(p_arr[idx]), float(s_arr[idx]), duration_sec, tol_samples)
            for m in key_in_ds
        )
        if all_others_wrong:
            strict_total += 1
            if len(strict_chosen) < need_count:
                strict_chosen.append(idx)
        elif key_others_wrong:
            loose_total += 1
            if len(loose_chosen) < need_count:
                loose_chosen.append(idx)

    if len(strict_chosen) >= need_count:
        return strict_chosen[:need_count], strict_total, strict_total, False

    # EQT/EPick/PhaseNet
    combined = strict_chosen + loose_chosen
    if len(combined) >= need_count:
        return combined[:need_count], strict_total, strict_total + loose_total, False

    # primary “ ” 0
    ranked_primary_ok.sort(key=lambda x: x[1], reverse=True)
    chosen_set = set(combined)
    for idx, _wrong_count in ranked_primary_ok:
        if idx in chosen_set:
            continue
        combined.append(idx)
        chosen_set.add(idx)
        if len(combined) >= need_count:
            break

    used_final_fallback = False
    if len(combined) < need_count:
        used_final_fallback = True
        fb = _fallback_indices_min_primary_max_others_wrong(
            ds, p_arr, s_arr, duration_sec, tol_samples, need_count, chosen_set
        )
        for idx in fb:
            if idx not in chosen_set:
                combined.append(idx)
                chosen_set.add(idx)
            if len(combined) >= need_count:
                break

    return combined[:need_count], strict_total, strict_total + loose_total, used_final_fallback


def _to_2d_waveform(x: np.ndarray) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError(f"原始波形维度应为2，得到形状 {x.shape}")
    if x.shape[0] == 3:
        return x
    if x.shape[1] == 3:
        return x.T
    raise ValueError(f"原始波形应包含3个分量，得到形状 {x.shape}")


def _safe_time_sec(x: object) -> Optional[float]:
    """Convert the input to float as much as possible and do a finiteness check; if it cannot be converted to NaN/is, None will be returned."""
    try:
        v = float(x)
    except Exception:
        return None
    return v if np.isfinite(v) else None


def _normalize_to_unit(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    m = np.max(np.abs(x))
    if m < eps:
        return x.copy()
    return x / m


def build_time_axis(num_samples: int, duration_sec: float) -> np.ndarray:
    return np.linspace(0.0, duration_sec, num_samples, endpoint=False)


def _panel_tag(k: int) -> str:
    """Generate subgraph tags: (a), (b), ... and degenerate to (n) after more than 26."""
    if 0 <= k < 26:
        return f"({chr(ord('a') + k)})"
    return f"({k + 1})"


def _build_plot_model_order(ds: Dict[str, np.ndarray]) -> List[str]:
    """Fixed model order: PRN, PRN-S, PhaseNet, EQT, UNet, remaining models."""
    ds_keys = set(ds.keys())
    ordered = [m for m in PREFERRED_MODEL_ORDER if m in ds_keys]
    others = [m for m in MODEL_ORDER if m not in PREFERRED_MODEL_ORDER and m in ds_keys]
    return ordered + others


def _display_model_name(model_name: str) -> str:
    if model_name == "UNet":
        return "U-net"
    if model_name == "PhaseRiskNet":
        return "GLANet"
    if model_name == "PhaseRiskNet-S":
        return "GLANet-G"
    return model_name


def _configure_font_times_new_roman() -> None:
    """Use Times New Roman only."""
    # Windows/Linux
    font_paths = [
        "C:/Windows/Fonts/times.ttf",
        "C:/Windows/Fonts/timesbd.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/times.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for fp in font_paths:
        try:
            font_manager.fontManager.addfont(fp)
        except Exception:
            pass
    # Times New Roman
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    plt.rcParams["font.family"] = ["Times New Roman"]
    plt.rcParams["axes.unicode_minus"] = False


def plot_panel(
    fig: plt.Figure,
    outer_spec,
    panel_tag: str,
    sample_name: str,
    waveform_3xt: np.ndarray,
    model_outputs: Dict[str, np.ndarray],
    p_time_sec: float,
    s_time_sec: float,
    duration_sec: float = 14.0,
    model_order_plot: Optional[List[str]] = None,
    show_model_labels: bool = True,
    show_amp_label: bool = True,
    show_sample_name: bool = True,
) -> plt.Axes:
    order = model_order_plot if model_order_plot is not None else list(MODEL_ORDER)
    n_models = len(order)
    inner = outer_spec.subgridspec(1 + n_models, 1, hspace=0.28, height_ratios=[1.25] + [1.0] * n_models)

    t = build_time_axis(waveform_3xt.shape[1], duration_sec)
    xlim = (0.0, duration_sec)
    ylim = (-1.2, 1.2)

    ax0 = fig.add_subplot(inner[0, 0])
    z = _normalize_to_unit(waveform_3xt[0])
    n = _normalize_to_unit(waveform_3xt[1])
    e = _normalize_to_unit(waveform_3xt[2])
    ax0.plot(t, z, lw=1.0, color=COLOR_Z, label="Z")
    ax0.plot(t, n, lw=1.0, color=COLOR_N, label="N")
    ax0.plot(t, e, lw=1.0, color=COLOR_E, label="E")
    if p_time_sec is not None:
        ax0.axvline(p_time_sec, color=COLOR_VP, ls="--", lw=1.0, alpha=0.9)
    if s_time_sec is not None:
        ax0.axvline(s_time_sec, color=COLOR_VS, ls="--", lw=1.0, alpha=0.9)
    ax0.set_xlim(*xlim)
    ax0.set_ylim(*ylim)
    ax0.set_ylabel("Amp." if show_amp_label else "", fontsize=8)
    if show_sample_name:
        ax0.set_title(f"{sample_name}", loc="left", fontsize=10, fontweight="bold")
    else:
        ax0.set_title("")
    ax0.grid(alpha=0.15, lw=0.4)
    ax0.tick_params(axis="x", labelbottom=False)

    for i, model_name in enumerate(order, start=1):
        ax = fig.add_subplot(inner[i, 0], sharex=ax0)
        p, s = _to_ps_channels(model_outputs[model_name])
        p = _normalize_to_unit(p)
        s = _normalize_to_unit(s)
        tt = build_time_axis(len(p), duration_sec)
        ax.plot(tt, p, lw=1.0, color=COLOR_P, label="P")
        ax.plot(tt, s, lw=1.0, color=COLOR_S, label="S")
        # Comment translated from Chinese.
        p_pred_idx = _pick_index(p)
        s_pred_idx = _pick_index(s)
        ax.axvline(_idx_to_sec(p_pred_idx, len(p), duration_sec), color=COLOR_P, ls="-", lw=0.8, alpha=0.7)
        ax.axvline(_idx_to_sec(s_pred_idx, len(s), duration_sec), color=COLOR_S, ls="-", lw=0.8, alpha=0.7)
        if p_time_sec is not None:
            ax.axvline(p_time_sec, color=COLOR_VP, ls="--", lw=0.9, alpha=0.85)
        if s_time_sec is not None:
            ax.axvline(s_time_sec, color=COLOR_VS, ls="--", lw=0.9, alpha=0.85)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_yticks([-1.0, 0.0, 1.0])
        if show_model_labels:
            ax.set_ylabel(_display_model_name(model_name), fontsize=7, rotation=0, labelpad=28, va="center")
        else:
            ax.set_ylabel("")
        ax.grid(alpha=0.12, lw=0.35)
        if i < n_models:
            ax.tick_params(axis="x", labelbottom=False)
        else:
            ax.set_xlabel("Time (s)", fontsize=9)
        ax.tick_params(axis="both", labelsize=7)
    return ax0


def main():
    parser = argparse.ArgumentParser(description="读取 phase_pick.py 生成的结果文件并绘图")
    parser.add_argument("--pred-npz", type=str, required=True, help="phase_pick 生成的结果 npz")
    parser.add_argument("--sample-indices", type=str, default="0,1,2,3", help="样本索引，逗号分隔（支持任意个）")
    parser.add_argument("--n-cols", type=int, default=2, help="网格列数（默认2）")
    parser.add_argument("--max-panels", type=int, default=0, help="最多绘制多少个样本；0表示不截断")
    parser.add_argument(
        "--debug-log",
        action="store_true",
        help="输出调试日志：打印 p/s 真值是否 finite，以及主模型 S 的 argmax 索引误差（采样点）。",
    )
    parser.add_argument(
        "--auto-select-prn-correct-others-wrong",
        action="store_true",
        help="自动筛选4条：优先双主模型全对且 PyTorch 对比模型全错（EQT/EPick 不参与）；不足时回退到 EQT/EPick/PhaseNet 全错",
    )
    parser.add_argument("--tol-samples", type=int, default=10, help="判定答对的容差（采样点）")
    parser.add_argument("--output", type=str, default="phase_pick_compare.png", help="输出图片路径")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--duration-sec", type=float, default=14.0, help="时间轴范围上限（秒）")
    parser.add_argument("--hide-sample-name", action="store_true", help="隐藏子图标题中的样本名/索引")
    args = parser.parse_args()

    ds = np.load(args.pred_npz, allow_pickle=True)
    raw = ds["raw"]
    p_arr = ds["p_arrival_sec"]
    s_arr = ds["s_arrival_sec"]
    names = ds["sample_names"] if "sample_names" in ds else np.array([f"sample_{i}" for i in range(raw.shape[0])], dtype=object)

    model_order_plot = _build_plot_model_order(ds)
    missing = [m for m in MODEL_ORDER if m not in ds]
    if missing:
        print(f"[phase_draw] 提示: npz 中缺少模型键 {missing}，将不绘制这些行", flush=True)
    if not model_order_plot:
        raise KeyError(f"预测文件无任何 MODEL_ORDER 中的模型键，当前键: {list(ds.keys())}")

    if args.auto_select_prn_correct_others_wrong:
        indices, strict_total, total_matched, used_final_fallback = _auto_select_indices(
            ds=ds,
            p_arr=p_arr,
            s_arr=s_arr,
            duration_sec=float(args.duration_sec),
            tol_samples=int(args.tol_samples),
            need_count=4,
        )
        if len(indices) < 4:
            raise ValueError(
                f"自动筛选仅找到 {len(indices)} 条（总样本 {raw.shape[0]}），不足4条。"
                f" 可放宽 --tol-samples（当前 {args.tol_samples}）或手动指定 --sample-indices。"
            )
        extra = ""
        if used_final_fallback:
            extra = "；已启用最终兜底（主模型综合索引误差最小，其次其它模型错误数最多）"
        print(
            f"[phase_draw] 自动筛选完成: 严格匹配(PyTorch对比全错,不含EQT/EPick) {strict_total} 条，"
            f"回退后总匹配 {total_matched} 条；若仍不足则按其它模型错误数降序补齐{extra}，使用索引 {indices}",
            flush=True,
        )
    else:
        indices = [int(x.strip()) for x in args.sample_indices.split(",") if x.strip() != ""]
        if len(indices) < 1:
            raise ValueError("请至少提供1个样本索引，例如 --sample-indices 0 或 0,1,2,3")

    if args.max_panels and int(args.max_panels) > 0:
        indices = indices[: int(args.max_panels)]

    if args.debug_log:
        tol = int(args.tol_samples)
        duration_sec = float(args.duration_sec)
        primary = ["PhaseRiskNet", "PhaseRiskNet-S"]
        for idx in indices:
            s_raw = s_arr[idx]
            p_raw = p_arr[idx]
            p_time = _safe_time_sec(p_raw)
            s_time = _safe_time_sec(s_raw)
            print(
                f"[phase_draw][debug] idx={idx} name={names[idx]} raw_p={p_raw!r} raw_s={s_raw!r} -> "
                f"p_time_sec={p_time} s_time_sec={s_time} (finite={_safe_time_sec(s_raw) is not None})",
                flush=True,
            )
            for m in primary:
                if m not in ds:
                    continue
                p_sec_val = float(p_time) if p_time is not None else float("nan")
                s_sec_val = float(s_time) if s_time is not None else float("nan")
                p_curve, s_curve = _to_ps_channels(ds[m][idx])
                t_len = min(len(p_curve), len(s_curve))
                if t_len <= 0:
                    continue
                s_pred_idx = int(np.argmax(np.asarray(s_curve[:t_len], dtype=float)))
                s_gt_idx = _sec_to_idx(s_sec_val, t_len, duration_sec)
                s_err = abs(s_pred_idx - s_gt_idx) if np.isfinite(s_sec_val) else float("inf")
                ok = s_err <= tol
                print(
                    f"[phase_draw][debug]   {m}: s_pred_idx={s_pred_idx}, s_gt_idx={s_gt_idx}, "
                    f"s_err={s_err} <= tol({tol}) ? {ok}",
                    flush=True,
                )

    _configure_font_times_new_roman()
    n_panels = len(indices)
    n_cols = max(1, int(args.n_cols))
    n_rows = int(np.ceil(n_panels / float(n_cols)))
    fig_w = max(8.0, 7.2 * n_cols)
    fig_h = max(6.0, 5.0 * n_rows)
    fig = plt.figure(figsize=(fig_w, fig_h))
    # Comment translated from Chinese.
    outer = fig.add_gridspec(n_rows, n_cols, wspace=0.08, hspace=0.30)
    # Comment translated from Chinese.
    fig.subplots_adjust(top=0.965, bottom=0.10, left=0.06, right=0.99)

    panel_axes: List[plt.Axes] = []
    for k, idx in enumerate(indices):
        r, c = divmod(k, n_cols)
        cell = outer[r, c]
        wf = _to_2d_waveform(raw[idx])
        model_outputs = {m: ds[m][idx] for m in model_order_plot}
        ax0 = plot_panel(
            fig=fig,
            outer_spec=cell,
            panel_tag=_panel_tag(k),
            sample_name=str(names[idx]),
            waveform_3xt=wf,
            model_outputs=model_outputs,
            p_time_sec=_safe_time_sec(p_arr[idx]),
            s_time_sec=_safe_time_sec(s_arr[idx]),
            duration_sec=args.duration_sec,
            model_order_plot=model_order_plot,
            show_model_labels=(c == 0),
            show_amp_label=(c == 0),
            show_sample_name=(not args.hide_sample_name),
        )
        panel_axes.append(ax0)

    # panel
    for k, ax0 in enumerate(panel_axes):
        pos = ax0.get_position()
        x_center = 0.5 * (pos.x0 + pos.x1)
        y = min(0.992, pos.y1 + 0.018)
        fig.text(x_center, y, _panel_tag(k), ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Comment translated from Chinese.
    shared_handles = [
        Line2D([0], [0], color=COLOR_Z, lw=1.2, label="Z waveform"),
        Line2D([0], [0], color=COLOR_N, lw=1.2, label="N waveform"),
        Line2D([0], [0], color=COLOR_E, lw=1.2, label="E waveform"),
        Line2D([0], [0], color=COLOR_P, lw=1.2, label="P (manual)"),
        Line2D([0], [0], color=COLOR_S, lw=1.2, label="S (manual)"),
        Line2D([0], [0], color=COLOR_VP, lw=1.2, ls="--", label="GT P arrival"),
        Line2D([0], [0], color=COLOR_VS, lw=1.2, ls="--", label="GT S arrival"),
        Line2D([0], [0], color=COLOR_P, lw=1.2, ls="-", label="Pred P arrival"),
        Line2D([0], [0], color=COLOR_S, lw=1.2, ls="-", label="Pred S arrival"),
    ]
    fig.legend(
        handles=shared_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=len(shared_handles),
        frameon=False,
        fontsize=8.5,
        handlelength=2.0,
        columnspacing=0.9,
        handletextpad=0.5,
        borderaxespad=0.2,
    )

    fig.savefig(args.output, dpi=args.dpi)
    print(f"[phase_draw] figure saved: {args.output}")


if __name__ == "__main__":
    main()

