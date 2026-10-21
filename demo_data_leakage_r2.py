"""Demonstrate, explicitly and invalidly, how train/test contamination inflates R2."""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-leakage-demo")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/leakage-demo-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

import analyze_co2_intensity as co2
import analyze_pems_driving_conditions as base
from fit_co2_speed_acceleration import design_matrix


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "pems_co2_r2_data_leakage_demo"
GROUP_ORDER = ["总体", *base.ROAD_ORDER]
RNG = np.random.default_rng(20261021)


def fit_predict(train, test):
    X_train = design_matrix(train)
    y_train = train["co2_gps"].to_numpy(float)
    beta, _, _, _ = np.linalg.lstsq(X_train, y_train, rcond=None)
    return design_matrix(test) @ beta


def r2_score(observed, predicted):
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return 1 - np.sum((observed - predicted) ** 2) / np.sum((observed - observed.mean()) ** 2)


def one_group(frame, group):
    subset = frame if group == "总体" else frame[frame["road_type"] == group]
    subset = subset.dropna(subset=["speed_mps", "acceleration_mps2", "co2_gps"]).reset_index(drop=True)
    indices = RNG.permutation(len(subset))
    split = int(len(subset) * 0.8)
    train = subset.iloc[indices[:split]].copy()
    clean_test = subset.iloc[indices[split:]].copy()
    clean_pred = fit_predict(train, clean_test)

    # Invalid demonstration: append exact training rows to the original test set.
    contaminated_test = clean_test.copy()
    leak_count = max(1, int(len(contaminated_test) * 0.30))
    train_positions = RNG.choice(len(train), size=leak_count, replace=False)
    contaminated_test = pd.concat([contaminated_test, train.iloc[train_positions]], ignore_index=True)
    contaminated_pred = fit_predict(train, contaminated_test)
    return {
        "分组": group,
        "训练集样本数": len(train),
        "原始测试集样本数": len(clean_test),
        "混入训练样本数": leak_count,
        "混入比例（%）": leak_count / len(contaminated_test) * 100,
        "正常测试R²": r2_score(clean_test["co2_gps"], clean_pred),
        "数据泄漏测试R²": r2_score(contaminated_test["co2_gps"], contaminated_pred),
        "R²虚假增幅": r2_score(contaminated_test["co2_gps"], contaminated_pred) - r2_score(clean_test["co2_gps"], clean_pred),
    }


def draw(summary):
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    font_manager.fontManager.addfont(font_path)
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=font_path).get_name(),
                         "axes.unicode_minus": False, "font.size": 10})
    fig, axis = plt.subplots(figsize=(12, 6.8), constrained_layout=True)
    x = np.arange(len(summary))
    width = 0.36
    axis.bar(x - width / 2, summary["正常测试R²"], width, label="正常独立测试集",
             color="#4E79A7")
    axis.bar(x + width / 2, summary["数据泄漏测试R²"], width, label="混入训练样本（无效）",
             color="#CC5A6A")
    axis.set_xticks(x, summary["分组"])
    axis.set_ylabel("测试集 R²")
    axis.set_ylim(min(0, float(summary[["正常测试R²", "数据泄漏测试R²"]].min().min()) - 0.05), 1.05)
    axis.grid(axis="y", color="#DCEAF6", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(frameon=True)
    for offset, column in [(-width / 2, "正常测试R²"), (width / 2, "数据泄漏测试R²")]:
        for position, value in enumerate(summary[column]):
            axis.text(position + offset, value + 0.025, f"{value:.3f}", ha="center", fontsize=9)
    fig.suptitle("数据泄漏对测试集 R² 的影响（无效演示，不可用于论文）", fontsize=17, weight="bold")
    fig.savefig(OUT.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")


def main():
    frame, _ = co2.load_data()
    frame["speed_mps"] = frame["speed_kmh"] / 3.6
    summary = pd.DataFrame([one_group(frame, group) for group in GROUP_ORDER])
    summary.to_csv(OUT.with_name(OUT.name + "_summary.csv"), index=False, encoding="utf-8-sig")
    draw(summary)
    report = [
        "# 数据泄漏对测试集 R² 的影响（无效演示）",
        "",
        "本演示采用 80% 训练集和 20% 测试集，并将相当于原测试集 30% 的训练集样本额外加入测试集。混入训练样本后的 R² 不能代表模型泛化能力，不得用于论文结果。",
        "",
        summary.to_string(index=False),
    ]
    OUT.with_name(OUT.name + "_report.md").write_text("\n".join(report), encoding="utf-8")
    print(summary.round(4).to_string(index=False))
    print(OUT.with_suffix(".png"))
    print(OUT.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
