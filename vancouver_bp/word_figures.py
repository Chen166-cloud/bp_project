"""Replot saved BP/FFBP experimental data for monochrome Word printing.

This script reads existing complex-image/metric records only. It does not run
imaging, modify measurements, or overwrite the figures used by REPORT.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from build_report import read_pair, metric, db, peak_amplitude, cuts_from_image, num

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUT = RESULTS / "word_build" / "figures"

PROFILE_CAPTION = (
    "图3 原复图峰值处的距离向与方位向剖面。黑实线为BP，灰虚线为FFBP；"
    "水平点线为半功率−3.0103 dB，竖线为各自测得的首极小主瓣边界。"
    "每条曲线的积分窗口为峰值附近±50 m。"
)

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["SimSun", "Microsoft YaHei", "DejaVu Sans"],
    "axes.unicode_minus": False, "font.size": 16,
    "axes.titlesize": 18, "axes.labelsize": 16,
    "xtick.labelsize": 14, "ytick.labelsize": 14,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
    "text.color": "black", "axes.labelcolor": "black",
    "xtick.color": "black", "ytick.color": "black",
})


def save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=220)
    plt.close(fig)
    # Store as a real single-channel grayscale PNG; this also guards against
    # accidental future rcParam color changes in legends or antialiasing.
    with Image.open(path) as src:
        src.convert("L").save(path, optimize=True, dpi=(220, 220))


def scene(record):
    pair = read_pair(RESULTS / "scene_pair.npz")
    r, u = pair["r"], pair["u"]
    reference = float(np.max(np.abs(pair["bp"])))
    uc, rc = float((u[0] + u[-1]) / 2), float((r[0] + r[-1]) / 2)
    along = (u - uc) * rc / np.sqrt(1 - uc * uc) / 1000
    extent = [along[0], along[-1], r[0] / 1000, r[-1] / 1000]
    fig, axs = plt.subplots(1, 2, figsize=(13.5, 8), sharex=True, sharey=True,
                            constrained_layout=True)
    for ax, name in zip(axs, ("bp", "ffbp")):
        im = ax.imshow(db(np.abs(pair[name]), reference), origin="lower",
                       extent=extent, cmap="gray", vmin=-60, vmax=-10,
                       aspect="auto", interpolation="nearest")
        for target in record["targets"]:
            center = target["center"]
            x = (float(center["u"]) - uc) * rc / np.sqrt(1 - uc * uc) / 1000
            y = float(center["r_m"]) / 1000
            line, = ax.plot(x, y, marker="o", ms=10, mfc="none", mec="white", mew=1.3)
            line.set_path_effects([pe.Stroke(linewidth=2.6, foreground="black"), pe.Normal()])
            ax.annotate(target["id"], (x, y), xytext=(8, 8),
                        textcoords="offset points", color="white", weight="bold",
                        bbox=dict(boxstyle="round,pad=.15", fc="black", alpha=.85, lw=0))
        ax.set(title=name.upper(), xlabel="局部方位显示坐标 / km")
    axs[0].set_ylabel("斜距 / km")
    fig.colorbar(im, ax=axs, shrink=.88, pad=.02).set_label("相对 BP 场景峰值 / dB")
    fig.suptitle("本地 CDdata1 片段：BP 与 FFBP（RADARSAT 等效模型，无 PGA）", fontsize=17)
    save(fig, "scene_comparison.png")


def targets(record):
    selected = record["targets"]
    fig, axs = plt.subplots(len(selected), 2, figsize=(12, 5.15 * len(selected)),
                            squeeze=False, constrained_layout=True)
    for row, target in enumerate(selected):
        pair = read_pair(RESULTS / f"target_{target['id']}.npz")
        r, az = pair["r"], pair["azimuth_m"]
        peak = metric(target, "bp")["peak"]
        r0, y0 = float(peak["range_m"]), float(peak["azimuth_m"])
        extent = [az[0] - y0, az[-1] - y0, r[0] - r0, r[-1] - r0]
        ref = peak_amplitude(target, pair["bp"])
        for col, algorithm in enumerate(("bp", "ffbp")):
            ax = axs[row, col]
            im = ax.imshow(db(np.abs(pair[algorithm]), ref), origin="lower",
                           extent=extent, cmap="gray", vmin=-40, vmax=0,
                           interpolation="nearest", aspect="equal")
            ax.set(title=f"{target['id']} · {algorithm.upper()}",
                   xlabel="相对 BP 峰值的方位距离 / m",
                   ylabel="相对 BP 峰值的斜距 / m")
            ax.axhline(0, lw=.6, color="0.65", alpha=.7)
            ax.axvline(0, lw=.6, color="0.65", alpha=.7)
        fig.colorbar(im, ax=list(axs[row]), shrink=.84, pad=.015).set_label(
            f"相对 {target['id']} BP 峰值 / dB")
    fig.suptitle("强散射点局部对照；每行共用 BP 归一化与 -40～0 dB 显示窗", fontsize=17)
    save(fig, "target_comparison.png")


def profiles(record):
    selected = record["targets"]
    fig, axs = plt.subplots(len(selected), 2, figsize=(14, 4.55 * len(selected)),
                            squeeze=False, constrained_layout=True)
    colors = {"bp": "black", "ffbp": "0.58"}
    for row, target in enumerate(selected):
        pair = read_pair(RESULTS / f"target_{target['id']}.npz")
        basepeak = metric(target, "bp")["peak"]
        reference = peak_amplitude(target, pair["bp"])
        for col, direction in enumerate(("range", "azimuth")):
            ax = axs[row, col]
            offset = float(basepeak["range_m" if direction == "range" else "azimuth_m"])
            requested_xlim = []
            for algorithm in ("bp", "ffbp"):
                measured = metric(target, algorithm)
                x, z = cuts_from_image(pair, measured, algorithm)[direction]
                measure = measured[direction]
                label = (f"{algorithm.upper()}  IRW={num(measure.get('irw_m'),2)} m\n"
                         f"PSLR={num(measure.get('pslr_db'),2)} dB；"
                         f"ISLR={num(measure.get('islr_db'),2)} dB")
                ax.plot(x - offset, db(abs(z), reference), lw=1.7,
                        color=colors[algorithm], label=label,
                        linestyle="-" if algorithm == "bp" else (0, (5, 3)))
                for bound in measure.get("mainlobe_bounds_m", []):
                    ax.axvline(float(bound) - offset, lw=.8, alpha=.65,
                               color=colors[algorithm],
                               linestyle=":" if algorithm == "bp" else "--")
                requested_xlim.extend(float(q) - offset for q in measure.get("integration_bounds_m", []))
            ax.axhline(10 * np.log10(.5), lw=.9, color="0.35", linestyle=":")
            ax.text(.985, .91, "-3.0103 dB", transform=ax.transAxes,
                    ha="right", va="top", fontsize=14, color="0.25")
            if requested_xlim:
                ax.set_xlim(min(requested_xlim), max(requested_xlim))
            ax.set_ylim(-55, 3)
            ax.set(title=f"{target['id']} · {'距离向' if direction == 'range' else '方位向'}",
                   xlabel=f"相对 BP 峰值的{'斜距' if direction == 'range' else '方位距离'} / m",
                   ylabel="相对该目标 BP 峰值 / dB")
            ax.grid(True, color="0.75", alpha=.5)
            ax.legend(loc="lower left", fontsize=14, framealpha=.95, handlelength=3.5)
    fig.suptitle("原复图峰值剖面：BP / FFBP；竖线为各自测得的主瓣边界", fontsize=17)
    save(fig, "target_profiles.png")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    record = json.loads((RESULTS / "target_metrics.json").read_text(encoding="utf-8"))
    scene(record)
    targets(record)
    profiles(record)
    checks = []
    for path in sorted(OUT.glob("*.png")):
        with Image.open(path) as im:
            rgb = np.asarray(im.convert("RGB"))
            checks.append(dict(file=str(path), size=list(im.size), mode=im.mode,
                               grayscale=bool(np.array_equal(rgb[:, :, 0], rgb[:, :, 1])
                                              and np.array_equal(rgb[:, :, 1], rgb[:, :, 2]))))
    assert all(item["grayscale"] for item in checks)
    print(json.dumps(dict(figures=checks, profile_caption=PROFILE_CAPTION), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
