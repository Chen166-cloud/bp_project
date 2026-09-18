"""Render the four CPU/GPU BP/FFBP results used in the group report."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "gpu_baseline"


def fidelity(reference, estimate):
    reference = np.asarray(reference, dtype=np.complex128)
    estimate = np.asarray(estimate, dtype=np.complex128)
    error = np.linalg.norm((estimate - reference).ravel()) / np.linalg.norm(reference.ravel())
    coherence = abs(np.vdot(reference.ravel(), estimate.ravel())) / (
        np.linalg.norm(reference.ravel()) * np.linalg.norm(estimate.ravel())
    )
    return {
        "relative_complex_error": float(error),
        "ser_db": float(-20.0 * np.log10(max(float(error), 1e-300))),
        "complex_coherence": float(coherence),
    }


def main():
    metrics = json.loads((OUT / "metrics.json").read_text(encoding="utf-8"))
    axes = np.load(OUT / "grid_axes.npz")
    r = axes["r"]
    u = axes["u"]
    images = {
        "cpu_bp": np.load(OUT / "cpu_direct_reference.npy"),
        "gpu_bp": np.load(OUT / "direct_mixed.npy"),
        "cpu_ffbp": np.load(OUT / "cpu_ffbp_reference.npy"),
        "gpu_ffbp": np.load(OUT / "ffbp_mixed.npy"),
    }
    times_ms = {
        "cpu_bp": 1000.0 * metrics["cpu_reference"]["direct_seconds"],
        "gpu_bp": metrics["gpu_runs"]["mixed"]["direct"]["event_timing"]["median_ms"],
        "cpu_ffbp": 1000.0 * metrics["cpu_reference"]["ffbp_seconds"],
        "gpu_ffbp": metrics["gpu_runs"]["mixed"]["ffbp"]["event_timing"]["median_ms"],
    }
    labels = {
        "cpu_bp": "(a) CPU direct BP",
        "gpu_bp": "(b) GPU direct BP / fixed mixed",
        "cpu_ffbp": "(c) CPU FFBP",
        "gpu_ffbp": "(d) GPU FFBP / fixed mixed",
    }
    order = (("cpu_bp", "gpu_bp"), ("cpu_ffbp", "gpu_ffbp"))
    # Use the CPU direct-BP peak as the fixed amplitude reference for all panels.
    peak = float(np.abs(images["cpu_bp"]).max())
    plot_extent = (float(u[0]), float(u[-1]), float(r[0]), float(r[-1]))
    fig, panels = plt.subplots(
        2, 2, figsize=(10.8, 8.0), constrained_layout=True, sharex=True, sharey=True
    )
    shown = None
    for row in range(2):
        for col in range(2):
            key = order[row][col]
            db = 20.0 * np.log10(np.abs(images[key]) / peak + 1e-12)
            shown = panels[row, col].imshow(
                db,
                origin="lower",
                aspect="auto",
                extent=plot_extent,
                vmin=-50,
                vmax=0,
                cmap="magma",
                interpolation="nearest",
            )
            time_text = (
                f"{times_ms[key] / 1000.0:.3f} s"
                if times_ms[key] >= 1000.0
                else f"{times_ms[key]:.2f} ms"
            )
            panels[row, col].set_title(f"{labels[key]}  |  {time_text}")
            if row == 1:
                panels[row, col].set_xlabel("u = sin(horizontal angle)")
            if col == 0:
                panels[row, col].set_ylabel("Slant range (m)")
    fig.colorbar(shown, ax=panels, label="Magnitude (dB, common reference)", shrink=0.92)
    fig.savefig(OUT / "bp_ffbp_cpu_gpu_fourway.png", dpi=180)
    plt.close(fig)

    names = ["CPU direct BP", "GPU direct BP", "CPU FFBP", "GPU FFBP"]
    keys = ["cpu_bp", "gpu_bp", "cpu_ffbp", "gpu_ffbp"]
    values = [times_ms[key] for key in keys]
    colors = ["#4c78a8", "#f58518", "#4c78a8", "#f58518"]
    fig, ax = plt.subplots(figsize=(8.6, 3.6), constrained_layout=True)
    bars = ax.bar(names, values, color=colors)
    ax.set_yscale("log")
    ax.set_ylabel("Resident compute time (ms)")
    ax.grid(axis="y", which="both", alpha=0.25)
    for bar, value in zip(bars, values):
        label = f"{value / 1000.0:.3f} s" if value >= 1000 else f"{value:.2f} ms"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value * 1.12,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylim(30, max(values) * 1.8)
    fig.savefig(OUT / "bp_ffbp_cpu_gpu_timing.png", dpi=180)
    plt.close(fig)

    summary = {
        "selection": {
            "cpu": "FP64 arithmetic with complex64 input/output reference",
            "gpu": "fixed mixed: FP64 geometry and phase; FP32 interpolation and accumulation; complex64 storage",
        },
        "time_ms": times_ms,
        "speedup": {
            "gpu_bp_vs_cpu_bp": float(times_ms["cpu_bp"] / times_ms["gpu_bp"]),
            "gpu_ffbp_vs_cpu_ffbp": float(times_ms["cpu_ffbp"] / times_ms["gpu_ffbp"]),
            "cpu_ffbp_vs_cpu_bp": float(times_ms["cpu_bp"] / times_ms["cpu_ffbp"]),
            "gpu_ffbp_vs_gpu_bp": float(times_ms["gpu_bp"] / times_ms["gpu_ffbp"]),
            "gpu_ffbp_vs_cpu_bp_combined": float(times_ms["cpu_bp"] / times_ms["gpu_ffbp"]),
        },
        "fidelity": {
            "gpu_bp_vs_cpu_bp": fidelity(images["cpu_bp"], images["gpu_bp"]),
            "cpu_ffbp_vs_cpu_bp": fidelity(images["cpu_bp"], images["cpu_ffbp"]),
            "gpu_ffbp_vs_cpu_ffbp": fidelity(images["cpu_ffbp"], images["gpu_ffbp"]),
            "gpu_ffbp_vs_cpu_bp_total": fidelity(images["cpu_bp"], images["gpu_ffbp"]),
        },
        "grid": {
            "shape": list(images["cpu_bp"].shape),
            "r_m": [float(r[0]), float(r[-1])],
            "u": [float(u[0]), float(u[-1])],
        },
    }
    (OUT / "bp_ffbp_cpu_gpu_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
