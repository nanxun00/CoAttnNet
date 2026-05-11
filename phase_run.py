"""
adaptive/phase_run.py: Comparison and expansion experiment entrance with PhaseNetUNet as the backbone.
"""

import copy
import csv
import os
import argparse
from typing import Dict, List

from phase_core import run_case, append_csv, METRICS_CSV, METRICS_HEADER, OUT_ROOT, SEED
import phase_core as core
from utils.repro import seed_everything


def print_separator(title: str = ""):
    print("\n" + "=" * 80)
    if title:
        print(f"  {title}")
        print("=" * 80)
    print()


ABLATION_CONFIGS: Dict[str, Dict] = {
    "phasenet_baseline": {
        "name": "phasenet_full_big",
        "model_class": "PhaseNetUNet",
        "depths": 5,
        "filters_root": 8,
        "kernels": (3, 7, 15),
        "pool_size": 4,
        "drop_rate": 0.1,
        "use_cbam": True,
        "use_separable": False,
        "use_gated_attention": True,
        "grid_search_ps_thresholds": False,
        "fixed_thr_p": 0.5,
        "fixed_thr_s": 0.5,
        "eval_only": False,
        "generate_visualizations": True,
    }
}


def run_test(config: Dict, quick_mode: bool = False) -> Dict:
    print_separator(f"测试: {config['name']}")
    print(f"[{config['name']}] 准备数据与模型...", flush=True)
    if quick_mode:
        print("⚠️  快速模式：小数据集 + 少量 epoch", flush=True)
        _ep = core.EPOCHS
        core.EPOCHS = 5
        try:
            result = run_case(config)
        finally:
            core.EPOCHS = _ep
    else:
        result = run_case(config)
    print(f"[{config['name']}] 完成。", flush=True)
    return result


def _parse_grid_floats(spec: str) -> List[float]:
    vals: List[float] = []
    for part in spec.split(","):
        p = part.strip()
        if p:
            vals.append(float(p))
    if not vals:
        raise ValueError("网格列表为空，例如: 0,0.25,0.35,0.5,0.75,1.0")
    return vals


def _cbam_sg_suffix(s: float) -> str:
    t = ("%.4f" % float(s)).rstrip("0").rstrip(".")
    return t.replace(".", "p").replace("-", "m")


def _grid_cbam_softgate_score(row: Dict, metric: str) -> float:
    if metric == "best_val":
        return float(row["best_val"])
    if metric == "macro_f1":
        return (float(row["p_f1"]) + float(row["s_f1"])) / 2.0
    raise ValueError(f"未知 metric={metric!r}，可选: best_val, macro_f1")


def run_cbam_softgate_strength_grid(
    base_key: str,
    strengths: List[float],
    metric: str,
    quick_mode: bool,
    append_main_metrics_csv: bool,
) -> Dict:
    """Do a grid search for cbam_softgate_strength; the indicators come from the result dictionary of each run_case."""
    if base_key not in ABLATION_CONFIGS:
        raise KeyError(f"未知配置 key={base_key!r}，可选: {list(ABLATION_CONFIGS.keys())[:8]}...")
    base = ABLATION_CONFIGS[base_key]
    best_val_higher_is_better = bool(base.get("best_model_by_f1", core.BEST_MODEL_BY_F1))
    if not base.get("use_cbam", False):
        print("警告: 基线 use_cbam=False，cbam_softgate_strength 不改变网络行为。", flush=True)
    if base.get("cbam_modulate_softgate", True) is False:
        print(
            "警告: 基线 cbam_modulate_softgate=False 时门控侧不使用时序调制，"
            "强度网格无效；请设为 True。",
            flush=True,
        )

    print_separator(f"cbam_softgate_strength 网格 (base={base_key}, 选优指标={metric})")
    print(f"强度列表 ({len(strengths)} 点): {strengths}", flush=True)

    grid_csv = os.path.join(OUT_ROOT, "grid_cbam_softgate_strength.csv")
    os.makedirs(OUT_ROOT, exist_ok=True)
    grid_header = [
        "base_key",
        "cbam_softgate_strength",
        "run_name",
        "best_val",
        "p_f1",
        "s_f1",
        "macro_f1",
        "metric_key",
    ]

    all_results: List[Dict] = []
    rows_for_file: List[List] = []

    for s in strengths:
        cfg = copy.deepcopy(base)
        cfg["cbam_softgate_strength"] = float(s)
        if "cbam_modulate_softgate" not in cfg:
            cfg["cbam_modulate_softgate"] = True
        suf = _cbam_sg_suffix(s)
        cfg["name"] = f"{base['name']}_cbamSG{suf}"
        print(f"\n>>> 网格点 cbam_softgate_strength={float(s):g}  name={cfg['name']}", flush=True)
        res = run_test(cfg, quick_mode=quick_mode)
        all_results.append(res)
        if append_main_metrics_csv:
            append_csv(METRICS_CSV, METRICS_HEADER, [res])
        p_f1 = float(res["p_f1"])
        s_f1 = float(res["s_f1"])
        macro = (p_f1 + s_f1) / 2.0
        rows_for_file.append([base_key, float(s), res["name"], res["best_val"], p_f1, s_f1, macro, metric])

    n = len(all_results)
    if metric == "best_val" and not best_val_higher_is_better:
        best_i = min(range(n), key=lambda i: float(all_results[i]["best_val"]))
    else:
        best_i = max(range(n), key=lambda i: _grid_cbam_softgate_score(all_results[i], metric))
    best_s = float(strengths[best_i])
    best_row = all_results[best_i]

    write_header = not os.path.exists(grid_csv)
    with open(grid_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(grid_header)
        w.writerows(rows_for_file)

    print("\n" + "=" * 80, flush=True)
    _cmp = (
        "best_val 按验证损失选小（best_model_by_f1=False）"
        if metric == "best_val" and not best_val_higher_is_better
        else "score 列越大越好"
    )
    print(f"网格结果摘要（{_cmp}）", flush=True)
    print(f"{'strength':>10}  {'best_val':>10}  {'P-F1':>8}  {'S-F1':>8}  {'macro_f1':>9}  {'score':>10}", flush=True)
    for s, r in zip(strengths, all_results):
        mac = (float(r["p_f1"]) + float(r["s_f1"])) / 2.0
        sc = _grid_cbam_softgate_score(r, metric)
        print(
            f"{float(s):>10.4f}  {float(r['best_val']):>10.6f}  "
            f"{float(r['p_f1']):>8.4f}  {float(r['s_f1']):>8.4f}  {mac:>9.4f}  {sc:>10.6f}",
            flush=True,
        )
    print("=" * 80, flush=True)
    print(
        f"按 {metric} 最优: cbam_softgate_strength={best_s:g}  "
        f"run={best_row['name']}  best_val={float(best_row['best_val']):.6f}  "
        f"P-F1={float(best_row['p_f1']):.4f} S-F1={float(best_row['s_f1']):.4f}",
        flush=True,
    )
    print(f"汇总已追加写入: {grid_csv}", flush=True)

    return {
        "best_cbam_softgate_strength": best_s,
        "metric": metric,
        "base_key": base_key,
        "best_row": best_row,
        "all_results": all_results,
        "grid_csv": grid_csv,
    }


def main():
    print("=" * 80, flush=True)
    print("adaptive/phase_run.py 启动（PhaseNetUNet 主干）", flush=True)
    print(f"工作目录: {os.getcwd()}", flush=True)
    print("=" * 80, flush=True)

    parser = argparse.ArgumentParser(description="PhaseNetUNet 运行入口（基座配置）")
    parser.add_argument("--quick", action="store_true", help="快速测试（小数据、少 epoch）")
    parser.add_argument("--gpu", type=str, default=None, help="GPU ID，如 0 或 0,1,2,3（设置 CUDA_VISIBLE_DEVICES）")
    parser.add_argument("--seed", type=int, default=SEED, help="全局随机种子")
    parser.add_argument(
        "--include-ablation",
        action="store_true",
        help="顺序运行 ABLATION_CONFIGS 中的全部配置（与字典定义顺序一致）",
    )
    parser.add_argument(
        "--baseline-key",
        type=str,
        default="phasenet_baseline",
        help="未加 --include-ablation 时：只跑这一条配置（默认 phasenet_baseline）",
    )
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"[phase] 指定可见 GPU: {args.gpu}", flush=True)

    core.SEED = args.seed
    print(f"[phase] 使用随机种子: {args.seed}", flush=True)
    seed_everything(args.seed, deterministic=True)

    results: Dict[str, Dict] = {}

    if args.include_ablation:
        if not ABLATION_CONFIGS:
            raise ValueError("ABLATION_CONFIGS 为空，无法运行。")
        print_separator("运行全部配置（ABLATION_CONFIGS）")
        for key, cfg in ABLATION_CONFIGS.items():
            print(f"[phase] 配置键: {key}", flush=True)
            results[key] = run_test(cfg, quick_mode=args.quick)
            append_csv(METRICS_CSV, METRICS_HEADER, [results[key]])
    else:
        baseline_key = args.baseline_key.strip() or "phasenet_baseline"
        if baseline_key not in ABLATION_CONFIGS:
            available = ", ".join(sorted(ABLATION_CONFIGS.keys()))
            raise KeyError(
                f"未找到配置键 {baseline_key!r}。当前可用键: {available}。"
                f"请使用 --baseline-key <键名>，或改用 --include-ablation 跑全部。"
            )
        print_separator("单配置运行")
        results["baseline"] = run_test(ABLATION_CONFIGS[baseline_key], quick_mode=args.quick)
        append_csv(METRICS_CSV, METRICS_HEADER, [results["baseline"]])

    print_separator("测试完成")


if __name__ == "__main__":
    main()

