"""Calculate Tier 3 CO2 factors from the ECF speed function and calibrate with PEMS."""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-tier3-co2")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/tier3-co2-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

import analyze_co2_intensity as co2
import analyze_pems_driving_conditions as base


ROOT = Path(__file__).resolve().parent
EF_XLSX = Path("/workspace/attachments/1c15f7e4-c7b3-4a65-991f-9f048872b9da/1.A.3.b.i-iv Road Transport Appendix 4 Emission Factors July 2026.xlsx")
OUT = ROOT / "pems_tier3_co2_calibration"
ROAD_ORDER = base.ROAD_ORDER

# Representative PEMS vehicle assumption: petrol passenger car, medium segment,
# Euro 6 d/e. GDI/PFI/ GDI+GPF share the same EC coefficients in this row.
FILTERS = {
    "Category": "Passenger Cars",
    "Fuel": "Petrol",
    "Segment": "Medium",
    "Euro Standard": "Euro 6 d/e",
}
COEFFICIENT_COLUMNS = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zita", "Hta"]
RF_COLUMN = "Reduction Factor\n [%]"
SPEED_MIN = 5.0
SPEED_MAX = 130.0
PETROL_CO2_KG_PER_KG = 3.169
PETROL_CV_MJ_PER_KG = 43.774
CO2_G_PER_MJ = PETROL_CO2_KG_PER_KG / PETROL_CV_MJ_PER_KG * 1000


def read_tier3_function():
    table = pd.read_excel(EF_XLSX, sheet_name="HOT_EMISSIONS_PARAMETERS")
    selected = table[table["Pollutant"].eq("EC")].copy()
    for column, value in FILTERS.items():
        selected = selected[selected[column].eq(value)]
    if selected.empty:
        raise ValueError("No representative EC row found in the emission factor workbook")
    row = selected.iloc[0]
    coefficients = {column: float(row[column]) for column in COEFFICIENT_COLUMNS}
    coefficients["ReductionFactor"] = float(row[RF_COLUMN])
    return coefficients, selected[COEFFICIENT_COLUMNS + [RF_COLUMN]].drop_duplicates()


def energy_factor(speed_kmh, coefficients):
    speed = np.clip(np.asarray(speed_kmh, dtype=float), SPEED_MIN, SPEED_MAX)
    a = coefficients["Alpha"]
    b = coefficients["Beta"]
    c = coefficients["Gamma"]
    d = coefficients["Delta"]
    e = coefficients["Epsilon"]
    z = coefficients["Zita"]
    h = coefficients["Hta"]
    rf = coefficients["ReductionFactor"]
    return (a * speed ** 2 + b * speed + c + d / speed) / (e * speed ** 2 + z * speed + h) * (1 - rf)


def load_pems():
    frame, raw_zero = co2.load_data()
    frame["co2_mass_g"] = frame["co2_gps"] * frame["dt_s"]
    frame["tier3_ecf_mj_km"] = np.nan
    frame["tier3_co2_g_km"] = np.nan
    return frame, raw_zero


def summarize(frame, coefficients):
    usable = frame[
        frame["distance_m"].gt(1) & frame["dt_s"].gt(0) & frame["co2_mass_g"].notna()
    ].copy()
    usable["tier3_ecf_mj_km"] = energy_factor(usable["speed_kmh"], coefficients)
    usable["tier3_co2_g_km"] = usable["tier3_ecf_mj_km"] * CO2_G_PER_MJ
    rows = []
    for road in ROAD_ORDER:
        group = usable[usable["road_type"].eq(road)]
        distance = group["distance_m"].sum()
        measured = group["co2_mass_g"].sum() / distance * 1000
        simulated = (group["tier3_co2_g_km"] * group["distance_m"]).sum() / distance
        rows.append({"道路类型": road, "样本数": len(group), "里程（km）": distance / 1000,
                     "实测值（g/km）": measured, "Tier 3模拟值（g/km）": simulated})
    result = pd.DataFrame(rows)
    # A single calibration coefficient is estimated from all PEMS distance and
    # emissions, then applied unchanged to each road class.
    total_distance = usable["distance_m"].sum()
    measured_total = usable["co2_mass_g"].sum() / total_distance * 1000
    simulated_total = (usable["tier3_co2_g_km"] * usable["distance_m"]).sum() / total_distance
    calibration = measured_total / simulated_total
    result["全局校准系数"] = calibration
    result["COPERT模拟值（g/km）"] = result["Tier 3模拟值（g/km）"] * calibration
    result["未校准误差率（%）"] = (result["实测值（g/km）"] - result["Tier 3模拟值（g/km）"]).abs() / result["实测值（g/km）"] * 100
    result["COPERT误差率（%）"] = (result["实测值（g/km）"] - result["COPERT模拟值（g/km）"]).abs() / result["实测值（g/km）"] * 100
    total = {"道路类型": "总体", "样本数": len(usable), "里程（km）": total_distance / 1000,
             "实测值（g/km）": measured_total, "Tier 3模拟值（g/km）": simulated_total,
             "全局校准系数": calibration, "COPERT模拟值（g/km）": measured_total,
             "未校准误差率（%）": abs(measured_total - simulated_total) / measured_total * 100,
             "COPERT误差率（%）": 0.0}
    return usable, pd.concat([pd.DataFrame([total]), result], ignore_index=True), calibration


def draw_speed_function(coefficients):
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    font_manager.fontManager.addfont(font_path)
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=font_path).get_name(),
                         "axes.unicode_minus": False, "font.size": 10})
    speeds = np.linspace(SPEED_MIN, SPEED_MAX, 500)
    ecf = energy_factor(speeds, coefficients)
    co2_factor = ecf * CO2_G_PER_MJ
    fig, axis1 = plt.subplots(figsize=(10.5, 6.5), constrained_layout=True)
    axis2 = axis1.twinx()
    line1, = axis1.plot(speeds, ecf, color="#4E79A7", linewidth=2.4, label="能源消耗因子")
    line2, = axis2.plot(speeds, co2_factor, color="#CC5A6A", linewidth=2.4, label="CO2排放因子")
    axis1.set_xlabel("速度（km/h）")
    axis1.set_ylabel("能源消耗因子（MJ/km）", color="#4E79A7")
    axis2.set_ylabel("CO2排放因子（g/km）", color="#CC5A6A")
    axis1.tick_params(axis="y", labelcolor="#4E79A7")
    axis2.tick_params(axis="y", labelcolor="#CC5A6A")
    axis1.grid(axis="y", color="#DCEAF6", linewidth=0.8)
    axis1.legend([line1, line2], [line1.get_label(), line2.get_label()], loc="upper right")
    axis1.set_title("Tier 3能源消耗及CO2排放因子速度函数", fontsize=17, weight="bold")
    fig.savefig(OUT.with_name(OUT.name + "_speed_function.png"), dpi=250, bbox_inches="tight")
    fig.savefig(OUT.with_name(OUT.name + "_speed_function.pdf"), bbox_inches="tight")


def draw_comparison(summary):
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    font_manager.fontManager.addfont(font_path)
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=font_path).get_name(),
                         "axes.unicode_minus": False, "font.size": 10})
    rows = summary[summary["道路类型"].isin(ROAD_ORDER)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), constrained_layout=True)
    positions = np.arange(len(rows))
    width = 0.25
    axes[0].bar(positions - width / 2, rows["实测值（g/km）"], width, label="PEMS实测值", color="#4E79A7")
    axes[0].bar(positions + width / 2, rows["COPERT模拟值（g/km）"], width, label="COPERT模拟值", color="#CC5A6A")
    axes[0].set_xticks(positions, rows["道路类型"])
    axes[0].set_ylabel("CO2排放因子（g/km）")
    axes[0].set_title("四种道路类型的实测值与模拟值")
    axes[0].grid(axis="y", color="#DCEAF6", linewidth=0.8)
    axes[0].legend()
    axes[1].bar(positions, rows["COPERT误差率（%）"], width, label="COPERT模拟误差率", color="#CC5A6A")
    axes[1].set_xticks(positions, rows["道路类型"])
    axes[1].set_ylabel("误差率（%）")
    axes[1].set_title("COPERT模拟值与实测值的误差率")
    axes[1].grid(axis="y", color="#DCEAF6", linewidth=0.8)
    axes[1].legend()
    fig.suptitle("Tier 3 CO2排放因子与PEMS实测值对比", fontsize=17, weight="bold")
    fig.savefig(OUT.with_suffix(".png"), dpi=250, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")


def main():
    coefficients, selected = read_tier3_function()
    frame, raw_zero = load_pems()
    usable, summary, calibration = summarize(frame, coefficients)
    selected.to_csv(OUT.with_name(OUT.name + "_selected_factor_rows.csv"), index=False, encoding="utf-8-sig")
    summary.to_csv(OUT.with_name(OUT.name + "_summary.csv"), index=False, encoding="utf-8-sig")
    speed_table = pd.DataFrame({"速度（km/h）": np.arange(5, 131, 5, dtype=float)})
    speed_table["能源消耗因子（MJ/km）"] = energy_factor(speed_table["速度（km/h）"], coefficients)
    speed_table["CO2排放因子（g/km）"] = speed_table["能源消耗因子（MJ/km）"] * CO2_G_PER_MJ
    speed_table.to_csv(OUT.with_name(OUT.name + "_speed_factors.csv"), index=False, encoding="utf-8-sig")
    draw_speed_function(coefficients)
    draw_comparison(summary)
    report = f"""# Tier 3能源消耗速度函数与PEMS校准

## 方法

PDF中的Tier 3公式为：

`EF(V) = (Alpha×V² + Beta×V + Gamma + Delta/V) / (Epsilon×V² + Zeta×V + Eta) × (1−RF)`。

Excel中选取 `EC` 参数，假定实测车辆为汽油乘用车、Medium、Euro 6 d/e。GDI、PFI及GDI+GPF在该参数行的能源消耗系数相同。速度范围按公式适用范围限制在 {SPEED_MIN:.0f}–{SPEED_MAX:.0f} km/h。

汽油CO2换算采用PDF中的 3.169 kg CO2/kg fuel 和 43.774 MJ/kg，得到换算系数 {CO2_G_PER_MJ:.3f} g CO2/MJ。

PEMS CO2原始0值数量为 {raw_zero}，先按时间序列插值；实测g/km按 `sum(CO2排放强度×时间间隔)/总行驶距离` 计算。模型值按采样点速度计算后进行距离加权。全局校准系数为 {calibration:.4f}，即用总体PEMS实测值与总体Tier 3模拟值之比进行一次乘性校准。

## 结果

{summary.round(3).to_string(index=False)}

注意：由于原始数据没有提供燃料、排放标准和发动机排量，以上代表性车辆假设会影响绝对模拟值；校准结果适用于本次PEMS试验车辆和路线，不应直接解释为所有车辆的通用因子。
"""
    OUT.with_name(OUT.name + "_report.md").write_text(report, encoding="utf-8")
    print(coefficients)
    print(summary.round(3).to_string(index=False))
    print("calibration", calibration)
    print(OUT.with_suffix(".png"))
    print(OUT.with_name(OUT.name + "_speed_function.png"))


if __name__ == "__main__":
    main()
