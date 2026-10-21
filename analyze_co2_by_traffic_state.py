"""Compare CO2 emissions under traffic states inferred from matched road speeds."""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-traffic-state")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/traffic-state-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import analyze_pems_driving_conditions as base


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "pems_co2_by_traffic_state"
STATE_ORDER = ["畅通", "缓行", "拥堵"]
PANEL_ORDER = ["总体", *base.ROAD_ORDER]
STATE_COLORS = {"畅通": "#4E79A7", "缓行": "#F28E2B", "拥堵": "#CC5A6A"}


def load_data():
    frames = []
    for path, name in base.FILES:
        points, _ = base.read_trial(path, name)
        frame = base.enrich_trial(points)
        co2 = frame["co2_gps"].where(frame["co2_gps"] > 0)
        frame["co2_gps_raw"] = frame["co2_gps"]
        frame["co2_gps"] = co2.interpolate(limit_direction="both")
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    bounds = (
        combined["lon"].min() - 0.01, combined["lat"].min() - 0.01,
        combined["lon"].max() + 0.01, combined["lat"].max() + 0.01,
    )
    reference_lat = combined["lat"].mean()
    road_points, road_classes = base.load_road_samples(bounds, reference_lat)
    tree = cKDTree(np.array([
        base.xy_m(lon, lat, reference_lat) for lon, lat in road_points
    ]))
    frames[0] = base.match_road_class(frames[0], tree, road_classes, force_branch=True)
    frames[1] = base.match_road_class(frames[1], tree, road_classes)
    return pd.concat(frames, ignore_index=True)


def classify_states(frame):
    # Use a road-class-specific 85th percentile so a low-speed branch road is
    # not treated as congested merely because it has a lower design speed.
    valid_speed = frame["speed_kmh"].where(frame["speed_kmh"] > 1)
    free_speed = frame.assign(_valid_speed=valid_speed).groupby("road_type")[
        "_valid_speed"
    ].quantile(0.85)
    free_speed = free_speed.clip(lower=20)
    frame["free_flow_speed_kmh"] = frame["road_type"].map(free_speed)
    frame["speed_ratio"] = frame["speed_kmh"] / frame["free_flow_speed_kmh"]

    frame["traffic_state"] = np.select(
        [frame["speed_ratio"] >= 0.70, frame["speed_ratio"] >= 0.40],
        ["畅通", "缓行"], default="拥堵"
    )
    frame.loc[frame["speed_kmh"].isna(), "traffic_state"] = pd.NA

    # Each row represents approximately one second. Use the positive distance
    # increment to convert instantaneous mass rate (g/s) to distance-based CO2.
    frame["co2_mass_g"] = frame["co2_gps"] * frame["dt_s"]
    frame["co2_g_per_km"] = np.where(
        (frame["distance_m"] > 1) & (frame["dt_s"] > 0),
        frame["co2_mass_g"] / frame["distance_m"] * 1000,
        np.nan,
    )
    return frame, free_speed


def summary_table(frame):
    rows = []
    panel_frames = {"总体": frame}
    panel_frames.update({name: frame[frame["road_type"] == name] for name in base.ROAD_ORDER})
    for panel in PANEL_ORDER:
        subset_panel = panel_frames[panel]
        for state in STATE_ORDER:
            subset = subset_panel[subset_panel["traffic_state"] == state]
            intensity = subset["co2_gps"].dropna()
            distance_emission = subset["co2_g_per_km"].dropna()
            rows.append({
                "分组": panel,
                "交通状态": state,
                "样本数": len(subset),
                "样本占比（%）": len(subset) / len(subset_panel) * 100 if len(subset_panel) else 0,
                "车速均值（km/h）": subset["speed_kmh"].mean(),
                "速度比均值": subset["speed_ratio"].mean(),
                "CO2排放强度均值（g/s）": intensity.mean(),
                "CO2排放强度中位数（g/s）": intensity.median(),
                "单位距离排放均值（g/km）": distance_emission.mean(),
                "单位距离排放中位数（g/km）": distance_emission.median(),
            })
    return pd.DataFrame(rows)


def draw(frame):
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    font_manager.fontManager.addfont(font_path)
    plt.rcParams.update({
        "font.family": font_manager.FontProperties(fname=font_path).get_name(),
        "axes.unicode_minus": False,
        "font.size": 10,
    })
    panel_frames = {"总体": frame}
    panel_frames.update({name: frame[frame["road_type"] == name] for name in base.ROAD_ORDER})

    fig, axes = plt.subplots(2, 5, figsize=(21, 8.8), sharex=False, constrained_layout=True)
    fig.patch.set_facecolor("white")
    metrics = [
        ("co2_gps", "CO2排放强度（g/s）", "瞬时 CO2 排放强度"),
        ("co2_g_per_km", "单位距离排放量（g/km）", "单位距离 CO2 排放量"),
    ]
    for row, (column, ylabel, row_title) in enumerate(metrics):
        for col, panel in enumerate(PANEL_ORDER):
            axis = axes[row, col]
            values = []
            labels = []
            for state in STATE_ORDER:
                data = panel_frames[panel].loc[
                    panel_frames[panel]["traffic_state"] == state, column
                ].dropna().to_numpy()
                values.append(data if len(data) else np.array([np.nan]))
                labels.append(state)
            box = axis.boxplot(
                values, tick_labels=labels, patch_artist=True, widths=0.58,
                showfliers=False, medianprops={"color": "#263746", "linewidth": 1.5},
                whiskerprops={"color": "#6B7C8A"}, capprops={"color": "#6B7C8A"},
                boxprops={"edgecolor": "#6B7C8A", "linewidth": 1.0},
            )
            for patch, state in zip(box["boxes"], STATE_ORDER):
                patch.set_facecolor(STATE_COLORS[state])
                patch.set_alpha(0.72)
            axis.set_title(panel, fontsize=12, weight="bold", pad=7)
            axis.set_ylabel(ylabel if col == 0 else "")
            axis.grid(axis="y", color="#DCEAF6", linewidth=0.8)
            axis.set_axisbelow(True)
            axis.tick_params(axis="x", labelrotation=0)
            if row == 0:
                axis.set_ylim(bottom=0)
            else:
                axis.set_ylim(bottom=0)
        axes[row, 0].text(
            -0.33, 1.08, row_title, transform=axes[row, 0].transAxes,
            fontsize=13, weight="bold", color="#284B63", va="bottom"
        )

    fig.suptitle("不同交通状态下的车辆 CO2 排放水平", fontsize=18, weight="bold", y=1.02)
    fig.savefig(OUT.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")


def main():
    frame = load_data()
    frame, free_speed = classify_states(frame)
    summary = summary_table(frame)
    summary.to_csv(OUT.with_name(OUT.name + "_summary.csv"), index=False, encoding="utf-8-sig")
    free_speed.rename("自由流参考速度（km/h）").reindex(base.ROAD_ORDER).to_csv(
        OUT.with_name(OUT.name + "_free_flow_speed.csv"), encoding="utf-8-sig"
    )
    frame[["trial", "time", "road_type", "speed_kmh", "free_flow_speed_kmh",
           "speed_ratio", "traffic_state", "co2_gps_raw", "co2_gps", "distance_m",
           "co2_g_per_km"]].to_csv(
               OUT.with_name(OUT.name + "_samples.csv"), index=False, encoding="utf-8-sig"
           )
    draw(frame)
    print("Free-flow reference speeds (km/h):")
    print(free_speed.reindex(base.ROAD_ORDER).round(2).to_string())
    print("\nSummary:")
    print(summary.round(3).to_string(index=False))
    print(OUT.with_suffix(".png"))
    print(OUT.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
