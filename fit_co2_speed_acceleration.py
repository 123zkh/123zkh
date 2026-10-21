"""Fit a quadratic CO2-intensity model using speed and acceleration."""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-co2-fit")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/co2-fit-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

import analyze_co2_intensity as co2
import analyze_pems_driving_conditions as base


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "pems_co2_speed_acceleration_fit"
GROUP_ORDER = ["总体", *base.ROAD_ORDER]


def design_matrix(frame):
    v = frame["speed_mps"].to_numpy(float)
    a = frame["acceleration_mps2"].to_numpy(float)
    return np.column_stack([np.ones(len(frame)), v, v ** 2, a, a ** 2, v * a])


def fit_group(frame, group):
    subset = frame if group == "总体" else frame[frame["road_type"] == group]
    subset = subset.dropna(subset=["speed_mps", "acceleration_mps2", "co2_gps"]).copy()
    X = design_matrix(subset)
    y = subset["co2_gps"].to_numpy(float)
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    predicted = X @ beta
    residual = y - predicted
    sse = float(np.sum(residual ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    n, p = len(y), X.shape[1]
    r2 = 1 - sse / sst if sst else np.nan
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p) if n > p else np.nan
    result = {
        "分组": group,
        "样本数": n,
        "截距": beta[0],
        "速度 v": beta[1],
        "速度平方 v²": beta[2],
        "加速度 a": beta[3],
        "加速度平方 a²": beta[4],
        "交互项 v×a": beta[5],
        "R²": r2,
        "调整R²": adj_r2,
        "RMSE（g/s）": float(np.sqrt(np.mean(residual ** 2))),
        "MAE（g/s）": float(np.mean(np.abs(residual))),
        "观测均值（g/s）": float(y.mean()),
        "拟合均值（g/s）": float(predicted.mean()),
    }
    return result, subset, y, predicted


def equation(beta):
    names = ["", "v", "v²", "a", "a²", "v·a"]
    terms = [f"{beta[0]:.4f}"]
    for value, name in zip(beta[1:], names[1:]):
        sign = "+" if value >= 0 else "−"
        terms.append(f" {sign} {abs(value):.4f}{name}")
    return "CO2 =" + "".join(terms)


def draw(fits):
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    font_manager.fontManager.addfont(font_path)
    plt.rcParams.update({
        "font.family": font_manager.FontProperties(fname=font_path).get_name(),
        "axes.unicode_minus": False,
        "font.size": 10,
    })
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.2), constrained_layout=True)
    axes = axes.ravel()
    cmap = plt.get_cmap("viridis")
    all_observed = np.concatenate([fits[group][2] for group in GROUP_ORDER])
    norm = plt.Normalize(float(all_observed.min()), float(all_observed.max()))
    for index, group in enumerate(GROUP_ORDER):
        axis = axes[index]
        result, subset, observed, predicted = fits[group]
        # Plot a deterministic subset of points when a panel is very dense.
        step = max(1, len(observed) // 3500)
        plotted_observed = observed[::step]
        plotted_predicted = predicted[::step]
        colors = plotted_observed
        axis.scatter(plotted_observed, plotted_predicted, c=colors, cmap=cmap, norm=norm,
                     s=10, alpha=0.42, linewidths=0)
        upper = max(float(observed.max()), float(predicted.max())) * 1.04
        lower = min(0.0, float(observed.min()), float(predicted.min()))
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
    fig.suptitle("速度—加速度二次多项式对 CO2 排放强度的拟合效果", fontsize=18, weight="bold")
    fig.savefig(OUT.with_suffix(".png"), dpi=250, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")


def main():
    frame, raw_zero = co2.load_data()
    frame["speed_mps"] = frame["speed_kmh"] / 3.6
    frame = frame.replace([np.inf, -np.inf], np.nan)
    fits = {}
    rows = []
    equation_rows = []
    for group in GROUP_ORDER:
        result, subset, observed, predicted = fit_group(frame, group)
        fits[group] = (result, subset, observed, predicted)
        result["拟合方程"] = equation(np.array([
            result["截距"], result["速度 v"], result["速度平方 v²"],
            result["加速度 a"], result["加速度平方 a²"], result["交互项 v×a"]
        ]))
        rows.append(result)
        equation_rows.append({"分组": group, "拟合方程": result["拟合方程"]})
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT.with_name(OUT.name + "_summary.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(equation_rows).to_csv(OUT.with_name(OUT.name + "_equations.csv"), index=False, encoding="utf-8-sig")
    draw(fits)
    report_lines = [
        "# 速度—加速度二次多项式 CO2 排放强度拟合",
        "",
        "模型形式：CO2 = β0 + β1v + β2v² + β3a + β4a² + β5va。速度 v 的单位为 m/s，加速度 a 的单位为 m/s²，CO2 排放强度单位为 g/s。",
        "",
        f"CO2 原始零值数量：{raw_zero}，已按时间序列进行插值。",
        "",
    ]
    for row in rows:
        report_lines.extend([
            f"## {row['分组']}",
            f"拟合方程：{row['拟合方程']}",
            f"样本数：{row['样本数']:,}；R²：{row['R²']:.4f}；调整R²：{row['调整R²']:.4f}；RMSE：{row['RMSE（g/s）']:.4f} g/s；MAE：{row['MAE（g/s）']:.4f} g/s。",
            "",
        ])
    OUT.with_name(OUT.name + "_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(summary.round(4).to_string(index=False))
    print(OUT.with_suffix(".png"))
    print(OUT.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
