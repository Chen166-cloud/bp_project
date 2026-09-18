"""CPU-only comparison of saved complex BP images on their polar grid.

Run after the requested guard images exist:

    python research_gpu_bp/plot_resident_images.py

All panels use the peak magnitude of the same direct-BP reference. SER and
error images use raw complex differences without phase or amplitude fitting.
This script imports no CUDA packages and runs no GPU computation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = ROOT / "fast_bp" / "results" / "resident_20260917"
DEFAULT_OUTPUT = ROOT / "research_gpu_bp" / "figures"
CANDIDATES = (
    ("batch_small_fp32_batched_graph.npy", "Batched FFBP, FP32"),
    ("guard_l1024_os12_g8_fp32.npy", "Refined FFBP, FP32"),
    ("guard_l1024_os12_g8_mixed.npy", "Refined FFBP, mixed"),
)
EDGE_PIXELS = 8
AMPLITUDE_LIMITS = (-45.0, 0.0)
ERROR_LIMITS = (-65.0, -15.0)


def _db(magnitude, peak):
    """20 log10 relative to one common reference peak (display clipping later)."""
    tiny = np.finfo(np.float64).tiny
    return 20.0 * np.log10(np.maximum(magnitude / peak, tiny))


def _ser(reference_energy, error_energy):
    return (float("inf") if error_energy == 0.0 else
            float(10.0 * np.log10(reference_energy / error_energy)))


def _json_number(value):
    return float(value) if np.isfinite(value) else None


def error_statistics(reference, candidate, boundary_width=EDGE_PIXELS):
    """Describe raw error energy spatially; this is not a causal attribution."""
    reference = np.asarray(reference, dtype=np.complex128)
    candidate = np.asarray(candidate, dtype=np.complex128)
    if candidate.shape != reference.shape or reference.ndim != 2:
        raise ValueError("candidate and reference must be matching 2-D images")
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(candidate)):
        raise ValueError("nonfinite complex images cannot be compared")
    if min(reference.shape) <= 2 * boundary_width:
        raise ValueError("image is too small for the fixed boundary width")
    error_power = np.abs(candidate - reference) ** 2
    reference_power = np.abs(reference) ** 2
    total_error = float(error_power.sum(dtype=np.float64))
    total_reference = float(reference_power.sum(dtype=np.float64))
    if total_reference <= 0:
        raise ValueError("reference image must have nonzero energy")
    boundary = np.ones(reference.shape, dtype=bool)
    boundary[boundary_width:-boundary_width, boundary_width:-boundary_width] = False
    output = {
        "relative_complex_l2_error": float(np.sqrt(total_error / total_reference)),
        "ser_db": _json_number(_ser(total_reference, total_error)),
        "ser_infinite": total_error == 0.0,
        "reference_energy": total_reference,
        "complex_error_energy": total_error,
        "boundary_width_pixels": int(boundary_width),
        "boundary_definition": "union of first/last 8 rows and first/last 8 columns on the output polar grid",
        "interpretation": "descriptive energy partition only; no attribution of the error mechanism",
    }
    for name, mask in (("boundary", boundary), ("interior", ~boundary)):
        pixels = int(np.count_nonzero(mask))
        region_error = float(error_power[mask].sum(dtype=np.float64))
        region_reference = float(reference_power[mask].sum(dtype=np.float64))
        output[name] = {
            "pixels": pixels,
            "pixel_fraction": pixels / reference.size,
            "error_energy": region_error,
            "fraction_of_total_error_energy": region_error / total_error if total_error else 0.0,
            "error_energy_per_pixel": region_error / pixels,
            "reference_energy": region_reference,
            "relative_complex_l2_error": (float(np.sqrt(region_error / region_reference))
                                           if region_reference > 0 else None),
        }
    return output


def _extent(r, u):
    for name, axis in (("r", r), ("u", u)):
        if axis.ndim != 1 or len(axis) < 2 or not np.all(np.isfinite(axis)):
            raise ValueError(f"{name} must be a finite 1-D axis with at least two samples")
        spacing = np.diff(axis)
        if not np.all(spacing > 0) or not np.allclose(spacing, spacing[0], rtol=1e-8, atol=1e-12):
            raise ValueError(f"{name} must be uniformly sampled and strictly increasing")
    return (float(u[0] - (u[1] - u[0]) / 2),
            float(u[-1] + (u[-1] - u[-2]) / 2),
            float(r[0] - (r[1] - r[0]) / 2),
            float(r[-1] + (r[-1] - r[-2]) / 2))


def generate(results_dir=DEFAULT_RESULTS, output_dir=DEFAULT_OUTPUT):
    results_dir, output_dir = Path(results_dir), Path(output_dir)
    reference_path = results_dir / "small_direct_reference.npy"
    axes_path = results_dir / "small_axes.npz"
    required = [reference_path, axes_path, *(results_dir / name for name, _ in CANDIDATES)]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Waiting for saved inputs:\n" + "\n".join(missing))

    reference = np.load(reference_path, allow_pickle=False).astype(np.complex128)
    with np.load(axes_path, allow_pickle=False) as saved_axes:
        r, u = np.asarray(saved_axes["r"], dtype=float), np.asarray(saved_axes["u"], dtype=float)
    if reference.shape != (len(r), len(u)):
        raise ValueError(f"reference shape {reference.shape} does not match the polar axes")
    extent = _extent(r, u)
    peak = float(np.max(np.abs(reference)))
    if not np.isfinite(peak) or peak <= 0:
        raise ValueError("reference peak must be finite and positive")

    candidates = []
    records = []
    for filename, title in CANDIDATES:
        image = np.load(results_dir / filename, allow_pickle=False).astype(np.complex128)
        statistics = error_statistics(reference, image)
        candidates.append((image, title, statistics))
        records.append({"file": filename, "label": title, **statistics})

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10.0,
        "axes.titlesize": 11.0, "axes.labelsize": 10.5,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    figure, axes = plt.subplots(2, len(candidates), figsize=(13.0, 8.0),
                                sharex=True, sharey=True, layout="constrained")
    for column, (image, title, statistics) in enumerate(candidates):
        ser_label = "infinite" if statistics["ser_infinite"] else f"{statistics['ser_db']:.2f} dB"
        axes[0, column].set_title(f"{title}\nSER = {ser_label}", pad=9)
        amplitude_map = axes[0, column].imshow(
            _db(np.abs(image), peak), origin="lower", extent=extent,
            aspect="auto", interpolation="nearest", cmap="viridis",
            vmin=AMPLITUDE_LIMITS[0], vmax=AMPLITUDE_LIMITS[1], rasterized=True,
        )
        error_map = axes[1, column].imshow(
            _db(np.abs(image - reference), peak), origin="lower", extent=extent,
            aspect="auto", interpolation="nearest", cmap="magma",
            vmin=ERROR_LIMITS[0], vmax=ERROR_LIMITS[1], rasterized=True,
        )
        axes[1, column].set_xlabel(r"$u = \sin(\theta)$")
        for row in range(2):
            ax = axes[row, column]
            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.yaxis.set_major_locator(MaxNLocator(5))
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            ax.tick_params(direction="out", length=3, width=0.7)
    axes[0, 0].set_ylabel("Image magnitude\n$r$ (m)")
    axes[1, 0].set_ylabel("Complex difference magnitude\n$r$ (m)")
    amplitude_bar = figure.colorbar(amplitude_map, ax=axes[0, :], shrink=0.96, pad=0.025)
    amplitude_bar.set_label(r"$20\log_{10}(|I|/\max |I_{\mathrm{ref}}|)$ (dB)")
    amplitude_bar.set_ticks([-45, -30, -15, 0])
    error_bar = figure.colorbar(error_map, ax=axes[1, :], shrink=0.96, pad=0.025)
    error_bar.set_label(r"$20\log_{10}(|I-I_{\mathrm{ref}}|/\max |I_{\mathrm{ref}}|)$ (dB)")
    error_bar.set_ticks([-65, -55, -45, -35, -25, -15])
    figure.suptitle(
        "GPU-resident SAR BP: common-reference comparison\n"
        "Polar coordinates; refined: leaf = 1024, oversampling = 12, guard = 8 pixels",
        fontsize=12.5,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "resident_small_image_comparison"
    figure.savefig(stem.with_suffix(".png"), dpi=220, facecolor="white")
    figure.savefig(stem.with_suffix(".pdf"), dpi=220, facecolor="white")
    plt.close(figure)

    result = {
        "reference": str(reference_path.resolve()),
        "reference_peak_magnitude": peak,
        "reference_shape": list(reference.shape),
        "coordinates": "polar: rows r (m), columns u = sin(theta); not Cartesian",
        "r_center_endpoints_m": [float(r[0]), float(r[-1])],
        "u_center_endpoints": [float(u[0]), float(u[-1])],
        "normalization": "all amplitude and complex-difference panels divided by the same reference peak",
        "alignment": "none: no global phase, amplitude, shift, or per-candidate normalization fit",
        "ser_definition": "10*log10(sum(abs(reference)**2) / sum(abs(candidate-reference)**2))",
        "metrics_arithmetic": "CPU NumPy complex128 differences and float64 energy reductions",
        "amplitude_display_limits_db": list(AMPLITUDE_LIMITS),
        "difference_display_limits_db": list(ERROR_LIMITS),
        "metrics_use_unclipped_images": True,
        "boundary_width_pixels": EDGE_PIXELS,
        "source_sha256": {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in required},
        "candidates": records,
        "figures": [str(stem.with_suffix(ext).resolve()) for ext in (".png", ".pdf")],
    }
    stem.with_suffix(".json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8",
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = generate(args.results_dir, args.output_dir)
    for record in result["candidates"]:
        print(f"{record['file']}: SER={record['ser_db']:.4f} dB, "
              f"relative complex L2={record['relative_complex_l2_error']:.8g}, "
              f"boundary error fraction={record['boundary']['fraction_of_total_error_energy']:.4f}")
    print("Saved " + ", ".join(result["figures"]))


if __name__ == "__main__":
    main()
