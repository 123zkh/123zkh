"""Refit the speed-acceleration CO2 model after global 3-sigma filtering."""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-co2-3sigma")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/co2-3sigma-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

import analyze_co2_intensity as co2
import analyze_pems_driving_conditions as base
from fit_co2_speed_acceleration import design_matrix, equation


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "pems_co2_speed_acceleration_fit_3sigma"
GROUP_ORDER = ["总体", *base.ROAD_ORDER]


def clean_3sigma(frame):
    variables = {
        "speed_mps": "速度 v（m/s）",
        "acceleration_mps2": "加速度 a（m/s²）",
        "co2_gps": "CO2排放强度（g/s）",
    }
    usable = frame.dropna(subset=list(variables)).copy()
    bounds = []
    keep = pd.Series(True, index=usable.index)
    for column, label in variables.items():
        mean = usable[column].mean()
        std = usable[column].std(ddof=1)
        lower, upper = mean - 3 * std, mean + 3 * std
        current = usable[column].between(lower, upper)
        keep &= current
        bounds.append({"变量": label, "均值": mean, "标准差": std,
                       "下限（均值−3σ）": lower, "上限（均值+3σ）": upper,
                       "超限样本数": int((~current).sum())})
    return usable.loc[keep].copy(), pd.DataFrame(bounds), int((~keep).sum())


def fit_group(frame, group):
    subset = frame if group == "总体" else frame[frame["road_type"] == group]
    subset = subset.dropna(subset=["speed_mps", "acceleration_mps2", "co2_gps"]).copy()
    X = design_matrix(subset)
    y = subset["co2_gps"].to_numpy(float)
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    predicted = X @ beta
    residual = y - predicted
    sse = np.sum(residual ** 2)
    sst = np.sum((y - y.mean()) ** 2)
    n, p = len(y), X.shape[1]
    r2 = 1 - sse / sst if sst else np.nan
    adjusted = 1 - (1 - r2) * (n - 1) / (n - p) if n > p else np.nan
    beta_values = np.asarray(beta)
    return {
        "分组": group, "样本数": n,
        "截距": beta[0], "速度 v": beta[1], "速度平方 v²": beta[2],
        "加速度 a": beta[3], "加速度平方 a²": beta[4], "交互项 v×a": beta[5],
        "R²": r2, "调整R²": adjusted,
        "RMSE（g/s）": np.sqrt(np.mean(residual ** 2)),
        "MAE（g/s）": np.mean(np.abs(residual)),
        "拟合方程": equation(beta_values),
    }, subset, y, predicted


def draw(fits):
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    font_manager.fontManager.addfont(font_path)
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=font_path).get_name(),
                         "axes.unicode_minus": False, "font.size": 10})
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.2), constrained_layout=True)
    axes = axes.ravel()
    cmap = plt.get_cmap("viridis")
    all_observed = np.concatenate([fits[group][2] for group in GROUP_ORDER])
    norm = plt.Normalize(float(all_observed.min()), float(all_observed.max()))
    for index, group in enumerate(GROUP_ORDER):
        axis = axes[index]
        result, _, observed, predicted = fits[group]
        step = max(1, len(observed) // 3500)
        axis.scatter(observed[::step], predicted[::step], c=observed[::step],
                     cmap=cmap, norm=norm, s=10, alpha=0.42, linewidths=0)
        lower = min(0.0, float(observed.min()), float(predicted.min()))
        upper = max(float(observed.max()), float(predicted.max())) * 1.04
        axis.plot([lower, upper], [lower, upper], "--", color="#CC5A6A", linewidth=1.2)
        axis.set_xlim(lower, upper)
        axis.set_ylim(lower, upper)
        axis.set_title(group, fontsize=13, weight="bold", pad=8)
        axis.set_xlabel("实测 CO2 排放强度（g/s）")
        axis.set_ylabel("拟合 CO2 排放强度（g/s）")
        axis.grid(color="#DCEAF6", linewidth=0.7)
        axis.text(0.04, 0.96,
                  f"R² = {result['R²']:.3f}\nRMSE = {result['RMSE（g/s）']:.3f} g/s\nN = {result['样本数']:,}",
                  transform=axis.transAxes, va="top", ha="left",
                  bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none", "pad": 3})
    axes[-1].axis("off")
    scalar = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    scalar.set_array([])
    colorbar = fig.colorbar(scalar, ax=axes[:-1], shrink=0.78, pad=0.025, aspect=28)
    colorbar.set_label("实测 CO2 排放强度（g/s）")
    fig.suptitle("3σ异常值处理后速度—加速度二次多项式拟合效果", fontsize=18, weight="bold")
    fig.savefig(OUT.with_suffix(".png"), dpi=250, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")


def main():
    frame, raw_zero = co2.load_data()
    frame["speed_mps"] = frame["speed_kmh"] / 3.6
    clean, bounds, removed = clean_3sigma(frame)
    fits = {}
    results = []
    for group in GROUP_ORDER:
        result, subset, observed, predicted = fit_group(clean, group)
        fits[group] = (result, subset, observed, predicted)
        results.append(result)
    summary = pd.DataFrame(results)
    bounds.to_csv(OUT.with_name(OUT.name + "_3sigma_bounds.csv"), index=False, encoding="utf-8-sig")
    summary.to_csv(OUT.with_name(OUT.name + "_summary.csv"), index=False, encoding="utf-8-sig")
    draw(fits)
    report = [
        "# 3σ异常值处理后的速度—加速度二次多项式拟合",
        "",
        "处理规则：对速度、加速度和瞬时CO2排放强度分别计算均值和标准差，任一变量超出均值±3倍标准差的样本剔除。CO2原始0值先按时间序列插值。模型形式为 CO2 = β0 + β1v + β2v² + β3a + β4a² + β5va。",
        "",
        f"原始零值插值数量：{raw_zero}；3σ处理前有效样本数：{len(frame.dropna(subset=['speed_mps', 'acceleration_mps2', 'co2_gps'])):,}；剔除异常样本数：{removed:,}；处理后样本数：{len(clean):,}。",
        "",
        summary.to_string(index=False),
    ]
    OUT.with_name(OUT.name + "_report.md").write_text("\n".join(report), encoding="utf-8")
    print(bounds.round(4).to_string(index=False))
    print(summary.round(4).to_string(index=False))
    print(OUT.with_suffix(".png"))
    print(OUT.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
