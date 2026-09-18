"""Build the compact direct-BP versus FFBP summary figure used by the report."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "sar_quality" / "results"
points = json.loads((RESULTS / "point_targets.json").read_text(encoding="utf-8"))
rois = json.loads((RESULTS / "real_rois.json").read_text(encoding="utf-8"))
benchmark = json.loads((ROOT / "fast_bp" / "results" / "benchmark.json").read_text(encoding="utf-8"))

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

ranges = np.array([row["range_m"] for row in points])
for algorithm, label, style in [
    ("direct_bp", "Direct BP", "-o"),
    ("ffbp", "FFBP", "--s"),
]:
    axes[0].plot(
        ranges,
        [row["metrics"][algorithm]["ground_range"]["irw_m"] for row in points],
        style,
        label=f"{label}, range",
    )
    axes[0].plot(
        ranges,
        [row["metrics"][algorithm]["along_track"]["irw_m"] for row in points],
        style,
        alpha=0.72,
        label=f"{label}, azimuth",
    )
axes[0].set(
    xlabel="Point-target slant range (m)",
    ylabel="Half-power IRW (m)",
    title="Ideal point-target resolution",
)
axes[0].legend(fontsize=8)

x = np.arange(1, len(rois) + 1)
width = 0.36
nav_error = [100 * row["fidelity"]["nav"]["relative_complex_error"] for row in rois]
af_error = [100 * row["fidelity"]["autofocus"]["relative_complex_error"] for row in rois]
axes[1].bar(x - width / 2, nav_error, width, label="Navigation")
axes[1].bar(x + width / 2, af_error, width, label="Autofocus")
axes[1].set(
    xlabel="Fixed real-data ROI",
    ylabel="FFBP relative complex error (%)",
    title="Real-data fidelity to direct BP",
    xticks=x,
    ylim=(0, 4.5),
)
axes[1].legend(fontsize=8)

times = [benchmark["direct_bp_seconds"], benchmark["ffbp_seconds"]]
bars = axes[2].bar(["Direct BP", "FFBP"], times, color=["#4c78a8", "#f58518"])
axes[2].set(ylabel="Core runtime (s)", title="Same-grid CPU benchmark")
for bar, value in zip(bars, times):
    axes[2].text(bar.get_x() + bar.get_width() / 2, value + 0.18, f"{value:.3f} s",
                 ha="center", va="bottom", fontsize=9)
axes[2].text(0.5, max(times) * 0.72, f"Speed-up: {benchmark['speedup']:.2f}x",
             ha="center", fontsize=11)

for ax in axes:
    ax.grid(alpha=0.22, axis="y")

target = RESULTS / "bp_vs_ffbp_summary.png"
fig.savefig(target, dpi=180)
plt.close(fig)
print(target)
