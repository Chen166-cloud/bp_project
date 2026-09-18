r"""Run the CUDA direct-BP/FFBP precision baseline on the fixed real-data ROI.

The experiment uses one FP64 host-side FFBP plan for all arithmetic policies.
Run from the project root with::

    D:\develop\Anaconda\python.exe fast_bp\gpu_baseline.py --repeats 3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal.windows import hann

from ffbp import bp_polar, form_ffbp
from gpu_backend import CudaBaseline, MODES, build_ffbp_plan, cp, cuda_environment
from sar_io import C, FC, FS, Capture, write_json


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = Path(__file__).resolve().parent / "results" / "gpu_baseline"
EXTENT = (1380.0, 1470.0, -0.03, 0.03)
LEAF_SIZE = 64
RANGE_SPACING = 0.25
ANGULAR_OVERSAMPLE = 3.0
SIDE = 1


def file_sha256(path, chunk_bytes=16 << 20):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest().upper()


def fidelity(reference, estimate):
    reference = np.asarray(reference, dtype=np.complex128)
    estimate = np.asarray(estimate, dtype=np.complex128)
    finite = np.isfinite(estimate)
    ref_norm = np.linalg.norm(reference.ravel())
    est_norm = np.linalg.norm(estimate.ravel())
    error = np.linalg.norm((estimate - reference).ravel()) / ref_norm
    mask = np.abs(reference) >= np.abs(reference).max() * 10.0 ** (-30.0 / 20.0)
    mask &= finite
    phase = np.angle(estimate[mask] * np.conj(reference[mask]))
    if phase.size:
        bias = np.angle(np.mean(np.exp(1j * phase)))
        residual = np.angle(np.exp(1j * (phase - bias)))
        phase_std = np.rad2deg(residual.std())
        phase_median_abs = np.rad2deg(np.median(np.abs(residual)))
        phase_p95_abs = np.rad2deg(np.percentile(np.abs(residual), 95))
    else:
        bias = phase_std = phase_median_abs = phase_p95_abs = math.nan
    ref_peak = np.unravel_index(np.argmax(np.abs(reference)), reference.shape)
    est_peak = np.unravel_index(np.argmax(np.abs(estimate)), estimate.shape)
    peak_shift = [int(est_peak[0] - ref_peak[0]), int(est_peak[1] - ref_peak[1])]
    peak_db = 20.0 * np.log10(
        max(float(np.abs(estimate).max()), 1e-300)
        / max(float(np.abs(reference).max()), 1e-300)
    )
    magnitude_error = np.linalg.norm((np.abs(estimate) - np.abs(reference)).ravel()) / np.linalg.norm(
        np.abs(reference).ravel()
    )
    return {
        "relative_complex_error": float(error),
        "ser_db": float(-20.0 * np.log10(max(error, 1e-300))),
        "complex_coherence": float(
            abs(np.vdot(reference.ravel(), estimate.ravel())) / (ref_norm * est_norm)
        ),
        "relative_magnitude_error": float(magnitude_error),
        "finite_fraction": float(finite.mean()),
        "phase_bias_deg": float(np.rad2deg(bias)),
        "phase_std_deg_above_minus30db": float(phase_std),
        "phase_median_abs_deg_above_minus30db": float(phase_median_abs),
        "phase_p95_abs_deg_above_minus30db": float(phase_p95_abs),
        "phase_mask_samples": int(mask.sum()),
        "peak_amplitude_difference_db": float(peak_db),
        "reference_peak_pixel": [int(x) for x in ref_peak],
        "estimate_peak_pixel": [int(x) for x in est_peak],
        "peak_shift_pixels": peak_shift,
    }


def timing_summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "samples_ms": [float(x) for x in values],
        "median_ms": float(np.median(values)),
        "q1_ms": float(np.percentile(values, 25)),
        "q3_ms": float(np.percentile(values, 75)),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
    }


def timing_summary_seconds(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "samples_s": [float(x) for x in values],
        "median_s": float(np.median(values)),
        "q1_s": float(np.percentile(values, 25)),
        "q3_s": float(np.percentile(values, 75)),
        "min_s": float(values.min()),
        "max_s": float(values.max()),
    }


def gpu_operator_validation(backend):
    """Exercise odd pulse counts, nonzero phase, both k signs and both sides."""
    from ffbp import bp_polar as cpu_bp_polar

    rng = np.random.default_rng(20260916)
    records = []
    for sign, side in ((-1, 1), (1, -1)):
        pulses, samples = 31, 512
        slow = np.arange(pulses, dtype=np.float64)
        position = np.column_stack(
            [
                0.05 * np.sin(0.37 * slow),
                np.linspace(-15.0, 15.0, pulses) + 0.03 * np.sin(slow),
                18.0 + 0.1 * np.cos(0.23 * slow),
            ]
        )
        data = (
            rng.standard_normal((pulses, samples))
            + 1j * rng.standard_normal((pulses, samples))
        ).astype(np.complex64)
        weight = hann(pulses)
        phase = np.linspace(-0.2, 0.3, pulses)
        range_start, data_dr = 900.0, 0.5
        extent = (1080.0, 1120.0, -0.05, 0.05)
        signed_fc = sign * FC
        k = 4.0 * np.pi * signed_fc / C
        plan = build_ffbp_plan(
            position,
            signed_fc,
            extent,
            side=side,
            leaf_size=8,
            range_spacing=0.5,
            angular_oversample=3.0,
        )
        cpu_ffbp, _ = form_ffbp(
            data,
            position,
            range_start,
            data_dr,
            signed_fc,
            extent,
            side=side,
            leaf_size=8,
            range_spacing=0.5,
            angular_oversample=3.0,
            weight=weight,
            phase=phase,
            verbose=False,
        )
        cpu_direct = cpu_bp_polar(
            data,
            position,
            plan.origin,
            plan.r0,
            plan.dr,
            plan.nr,
            plan.u0,
            plan.du,
            plan.nu,
            range_start,
            data_dr,
            k,
            weight,
            phase,
            side,
        )
        if cpu_ffbp.image.shape != plan.shape:
            raise AssertionError("CPU and frozen GPU plans produced different synthetic grids")
        device = backend.upload(data, position, weight, phase, plan)
        for mode in MODES:
            gpu_ffbp, _, _ = backend.run_ffbp(plan, device, mode, range_start, data_dr, k)
            ffbp_host, _ = backend.download(gpu_ffbp)
            gpu_direct, _, _ = backend.run_direct(
                plan, device, mode, range_start, data_dr, k, chunk_pulses=7
            )
            direct_host, _ = backend.download(gpu_direct)
            record = {
                "k_sign": sign,
                "side": side,
                "mode": mode,
                "ffbp_vs_cpu": fidelity(cpu_ffbp.image, ffbp_host),
                "direct_vs_cpu": fidelity(cpu_direct, direct_host),
            }
            records.append(record)
            if mode == "fp64":
                assert record["ffbp_vs_cpu"]["relative_complex_error"] <= 1e-5
                assert record["direct_vs_cpu"]["relative_complex_error"] <= 1e-5
            elif mode == "mixed":
                assert record["ffbp_vs_cpu"]["relative_complex_error"] <= 1e-3
                assert record["direct_vs_cpu"]["relative_complex_error"] <= 1e-3
            assert record["ffbp_vs_cpu"]["finite_fraction"] == 1.0
            assert record["direct_vs_cpu"]["finite_fraction"] == 1.0
            del gpu_ffbp, gpu_direct
        del device
        cp.get_default_memory_pool().free_all_blocks()
    return {
        "passed": True,
        "cases": records,
        "coverage": [
            "positive and negative signed carrier wavenumber",
            "left and right looking geometry",
            "odd pulse count and recursive tail split",
            "nonzero per-pulse phase",
            "fractional four-point Lagrange interpolation",
        ],
    }


def cpu_reference(data, position, weight, phase, plan, range_start, data_dr, k, repeats):
    ffbp_times, direct_times = [], []
    ffbp = direct = counts = None
    axis_error = 0.0
    for repeat in range(repeats):
        start = time.perf_counter()
        ffbp, counts = form_ffbp(
            data,
            position,
            range_start,
            data_dr,
            -FC,
            EXTENT,
            side=SIDE,
            leaf_size=LEAF_SIZE,
            range_spacing=RANGE_SPACING,
            angular_oversample=ANGULAR_OVERSAMPLE,
            weight=weight,
            phase=phase,
            verbose=False,
        )
        ffbp_times.append(time.perf_counter() - start)
        if ffbp.image.shape != plan.shape:
            raise AssertionError((ffbp.image.shape, plan.shape))
        axis_error = max(
            float(np.max(np.abs(ffbp.origin - plan.origin))),
            abs(float(ffbp.r0 - plan.r0)),
            abs(float(ffbp.u0 - plan.u0)),
            abs(float(ffbp.du - plan.du)),
        )
        if axis_error > 1e-12:
            raise AssertionError(f"CPU and frozen plan axes differ by {axis_error}")
        start = time.perf_counter()
        direct = bp_polar(
            data,
            position,
            plan.origin,
            plan.r0,
            plan.dr,
            plan.nr,
            plan.u0,
            plan.du,
            plan.nu,
            range_start,
            data_dr,
            k,
            weight,
            phase,
            SIDE,
        )
        direct_times.append(time.perf_counter() - start)
        print(
            f"CPU repeat {repeat + 1}/{repeats}: direct={direct_times[-1]:.3f} s, "
            f"FFBP={ffbp_times[-1]:.3f} s",
            flush=True,
        )
    direct_seconds = float(np.median(direct_times))
    ffbp_seconds = float(np.median(ffbp_times))
    return (
        np.ascontiguousarray(direct),
        np.ascontiguousarray(ffbp.image),
        {
            "direct_seconds": float(direct_seconds),
            "ffbp_seconds": float(ffbp_seconds),
            "direct_timing": timing_summary_seconds(direct_times),
            "ffbp_timing": timing_summary_seconds(ffbp_times),
            "ffbp_speedup_over_direct": float(direct_seconds / ffbp_seconds),
            "axis_max_abs_error": axis_error,
            "ffbp_counts": counts,
            "ffbp_vs_direct": fidelity(direct, ffbp.image),
            "threads": 16,
            "processor": platform.processor() or "Intel Core i9-14900HX",
            "timing_scope": "warmed Numba kernels; same pulses and final grid; excludes range compression and disk I/O",
        },
    )


def save_plots(out, images, cpu_direct, cpu_ffbp, plan, result):
    mode_labels = {"fp64": "FP64", "fp32": "FP32", "mixed": "Fixed mixed"}
    algorithms = (("direct", "Direct BP"), ("ffbp", "FFBP"))
    r = plan.r0 + plan.dr * np.arange(plan.nr)
    u = plan.u0 + plan.du * np.arange(plan.nu)
    extent = (float(u[0]), float(u[-1]), float(r[0]), float(r[-1]))
    global_peak = max(float(np.abs(value).max()) for value in images.values())

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.3), constrained_layout=True, sharex=True, sharey=True)
    shown = None
    for row, (algorithm, title) in enumerate(algorithms):
        for col, mode in enumerate(MODES):
            db = 20.0 * np.log10(np.abs(images[(algorithm, mode)]) / global_peak + 1e-12)
            shown = axes[row, col].imshow(
                db,
                origin="lower",
                aspect="auto",
                extent=extent,
                vmin=-50,
                vmax=0,
                cmap="magma",
                interpolation="nearest",
            )
            axes[row, col].set_title(f"{title} / {mode_labels[mode]}")
            if row == 1:
                axes[row, col].set_xlabel("u = sin(horizontal angle)")
            if col == 0:
                axes[row, col].set_ylabel("Slant range (m)")
    fig.colorbar(shown, ax=axes, label="Magnitude (dB, common reference)", shrink=0.92)
    fig.savefig(out / "gpu_baseline_images.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.3), constrained_layout=True, sharex=True, sharey=True)
    shown = None
    references = {"direct": cpu_direct, "ffbp": cpu_ffbp}
    for row, (algorithm, title) in enumerate(algorithms):
        reference = references[algorithm]
        scale = float(np.abs(reference).max())
        for col, mode in enumerate(MODES):
            error_db = 20.0 * np.log10(
                np.abs(images[(algorithm, mode)] - reference) / scale + 1e-12
            )
            shown = axes[row, col].imshow(
                error_db,
                origin="lower",
                aspect="auto",
                extent=extent,
                vmin=-100,
                vmax=-20,
                cmap="viridis",
                interpolation="nearest",
            )
            axes[row, col].set_title(f"{title} / {mode_labels[mode]}")
            if row == 1:
                axes[row, col].set_xlabel("u = sin(horizontal angle)")
            if col == 0:
                axes[row, col].set_ylabel("Slant range (m)")
    fig.colorbar(shown, ax=axes, label="Complex error (dB relative to CPU peak)", shrink=0.92)
    fig.savefig(out / "gpu_baseline_error_maps.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True)
    x = np.arange(len(MODES))
    width = 0.35
    direct_ms = [result["gpu_runs"][mode]["direct"]["event_timing"]["median_ms"] for mode in MODES]
    ffbp_ms = [result["gpu_runs"][mode]["ffbp"]["event_timing"]["median_ms"] for mode in MODES]
    axes[0].bar(x - width / 2, direct_ms, width, label="Direct BP")
    axes[0].bar(x + width / 2, ffbp_ms, width, label="FFBP")
    axes[0].set_yscale("log")
    axes[0].set_xticks(x, [mode_labels[m] for m in MODES])
    axes[0].set_ylabel("Resident GPU compute time (ms)")
    axes[0].grid(axis="y", which="both", alpha=0.25)
    axes[0].legend()
    markers = {"direct": "o", "ffbp": "s"}
    colors = {"fp64": "#1f77b4", "fp32": "#ff7f0e", "mixed": "#2ca02c"}
    for algorithm, title in algorithms:
        for mode in MODES:
            run = result["gpu_runs"][mode][algorithm]
            axes[1].scatter(
                run["event_timing"]["median_ms"],
                run["vs_cpu_same_algorithm"]["ser_db"],
                marker=markers[algorithm],
                color=colors[mode],
                s=55,
            )
            axes[1].annotate(
                f"{title}, {mode_labels[mode]}",
                (run["event_timing"]["median_ms"], run["vs_cpu_same_algorithm"]["ser_db"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
            )
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Resident GPU compute time (ms)")
    axes[1].set_ylabel("SER versus same-algorithm CPU reference (dB)")
    axes[1].grid(which="both", alpha=0.25)
    fig.savefig(out / "gpu_baseline_performance_quality.png", dpi=180)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--cpu-repeats", type=int, default=3)
    parser.add_argument("--chunk-pulses", type=int, default=64)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--skip-operator-validation", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.repeats < 1 or args.cpu_repeats < 1 or args.warmups < 0:
        raise ValueError("repeats must be positive and warmups must be nonnegative")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    dat_path = ROOT / "2018-08-14_18-54-08.dat"
    rc_path = ROOT / "fast_bp" / "results" / "range_compressed.npy"
    position_path = ROOT / "fast_bp" / "results" / "autofocus_position.npy"
    capture = Capture(dat_path)
    data = np.load(rc_path, mmap_mode="r")
    position = np.load(position_path)
    weight = hann(capture.n)
    phase = np.zeros(capture.n, dtype=np.float64)
    range_start = C * (capture.delay - capture.tp / 2.0) / 2.0
    data_dr = C / (2.0 * FS)
    k = -4.0 * np.pi * FC / C
    if data.shape != (capture.n, capture.nr) or data.dtype != np.complex64:
        raise AssertionError((data.shape, data.dtype, capture.n, capture.nr))
    plan = build_ffbp_plan(
        position,
        -FC,
        EXTENT,
        side=SIDE,
        leaf_size=LEAF_SIZE,
        range_spacing=RANGE_SPACING,
        angular_oversample=ANGULAR_OVERSAMPLE,
    )
    print(f"Frozen plan: {plan.shape}, {plan.counts}", flush=True)

    backend = CudaBaseline()
    print(f"CUDA kernels compiled in {backend.compile_seconds:.3f} s", flush=True)
    operator = None
    if not args.skip_operator_validation:
        operator = gpu_operator_validation(backend)
        write_json(out / "operator_validation.json", operator)
        print("GPU operator validation passed.", flush=True)

    direct_cpu, ffbp_cpu, cpu_record = cpu_reference(
        data, position, weight, phase, plan, range_start, data_dr, k, args.cpu_repeats
    )
    np.save(out / "cpu_direct_reference.npy", direct_cpu)
    np.save(out / "cpu_ffbp_reference.npy", ffbp_cpu)
    np.savez(
        out / "grid_axes.npz",
        r=plan.r0 + plan.dr * np.arange(plan.nr),
        u=plan.u0 + plan.du * np.arange(plan.nu),
        origin=plan.origin,
    )
    print(
        f"CPU references: direct={cpu_record['direct_seconds']:.3f} s, "
        f"FFBP={cpu_record['ffbp_seconds']:.3f} s",
        flush=True,
    )

    staging_start = time.perf_counter()
    staged_data = np.array(data, dtype=np.complex64, order="C", copy=True)
    host_staging_ms = 1000.0 * (time.perf_counter() - staging_start)
    device = backend.upload(staged_data, position, weight, phase, plan)
    del staged_data
    print(
        f"Host staging: {host_staging_ms:.1f} ms; H2D: {device.h2d_event_ms:.1f} ms event, "
        f"{device.h2d_wall_ms:.1f} ms wall, "
        f"{device.bytes_uploaded / 2**30:.3f} GiB",
        flush=True,
    )
    images = {}
    gpu_runs = {}
    for mode in args.modes:
        gpu_runs[mode] = {}
        for algorithm in ("ffbp", "direct"):
            event_times, wall_times, d2h_times = [], [], []
            repeat_images = []
            warmup_times = []
            for warmup in range(args.warmups):
                if algorithm == "ffbp":
                    warm_device, warm_event, _ = backend.run_ffbp(
                        plan, device, mode, range_start, data_dr, k
                    )
                else:
                    warm_device, warm_event, _ = backend.run_direct(
                        plan,
                        device,
                        mode,
                        range_start,
                        data_dr,
                        k,
                        chunk_pulses=args.chunk_pulses,
                    )
                del warm_device
                warmup_times.append(float(warm_event))
                print(
                    f"{algorithm.upper():6s} {mode:5s} warmup {warmup + 1}/{args.warmups}: "
                    f"event={warm_event:.3f} ms",
                    flush=True,
                )
            for repeat in range(args.repeats):
                if algorithm == "ffbp":
                    image_device, event_ms, wall_ms = backend.run_ffbp(
                        plan, device, mode, range_start, data_dr, k
                    )
                else:
                    image_device, event_ms, wall_ms = backend.run_direct(
                        plan,
                        device,
                        mode,
                        range_start,
                        data_dr,
                        k,
                        chunk_pulses=args.chunk_pulses,
                    )
                image, d2h_ms = backend.download(image_device)
                del image_device
                event_times.append(event_ms)
                wall_times.append(wall_ms)
                d2h_times.append(d2h_ms)
                repeat_images.append(image)
                print(
                    f"{algorithm.upper():6s} {mode:5s} repeat {repeat + 1}/{args.repeats}: "
                    f"event={event_ms:.3f} ms wall={wall_ms:.3f} ms D2H={d2h_ms:.3f} ms",
                    flush=True,
                )
            reference = ffbp_cpu if algorithm == "ffbp" else direct_cpu
            determinism = [
                float(np.linalg.norm((item - repeat_images[0]).ravel()) / np.linalg.norm(repeat_images[0].ravel()))
                for item in repeat_images[1:]
            ]
            selected = repeat_images[-1]
            images[(algorithm, mode)] = selected
            np.save(out / f"{algorithm}_{mode}.npy", selected)
            record = {
                "event_timing": timing_summary(event_times),
                "wall_timing": timing_summary(wall_times),
                "d2h_timing": timing_summary(d2h_times),
                "warmup_event_ms": warmup_times,
                "resident_input_to_result_ms": float(np.median(wall_times) + np.median(d2h_times)),
                "host_ram_to_result_ms": float(
                    device.h2d_wall_ms + np.median(wall_times) + np.median(d2h_times)
                ),
                "mmap_stage_to_result_ms": float(
                    host_staging_ms
                    + device.h2d_wall_ms
                    + np.median(wall_times)
                    + np.median(d2h_times)
                ),
                "determinism_relative_errors": determinism,
                "vs_cpu_same_algorithm": fidelity(reference, selected),
            }
            gpu_runs[mode][algorithm] = record
            if mode == "fp64" and record["vs_cpu_same_algorithm"]["relative_complex_error"] > 1e-4:
                raise AssertionError(f"FP64 {algorithm} parity failed: {record}")
            if mode == "mixed" and record["vs_cpu_same_algorithm"]["relative_complex_error"] > 1e-3:
                raise AssertionError(f"Mixed {algorithm} parity failed: {record}")
            if record["vs_cpu_same_algorithm"]["finite_fraction"] != 1.0:
                raise AssertionError(f"Non-finite result in {algorithm}/{mode}")
        gpu_runs[mode]["ffbp_vs_direct_same_precision"] = fidelity(
            images[("direct", mode)], images[("ffbp", mode)]
        )
        write_json(out / "progress.json", {"completed_modes": list(gpu_runs), "gpu_runs": gpu_runs})

    if set(args.modes) != set(MODES):
        print("Partial mode run complete; plots and final metrics require all three modes.", flush=True)
        return

    for algorithm in ("direct", "ffbp"):
        for mode in MODES:
            gpu_runs[mode][algorithm]["vs_gpu_fp64_same_algorithm"] = fidelity(
                images[(algorithm, "fp64")], images[(algorithm, mode)]
            )

    environment = cuda_environment()
    fp32_target = {}
    for algorithm in ("direct", "ffbp"):
        quality = gpu_runs["fp32"][algorithm]["vs_cpu_same_algorithm"]
        fp32_target[algorithm] = {
            "passed": bool(
                quality["relative_complex_error"] <= 0.01
                and quality["complex_coherence"] >= 0.9999
                and quality["phase_std_deg_above_minus30db"] <= 1.0
                and abs(quality["peak_amplitude_difference_db"]) <= 0.5
                and max(abs(x) for x in quality["peak_shift_pixels"]) <= 1
            ),
            "relative_complex_error_max": 0.01,
            "ser_db_min": 40.0,
            "complex_coherence_min": 0.9999,
            "phase_std_deg_max": 1.0,
        }
    result = {
        "status": "completed",
        "experiment": {
            "name": "GPU direct BP and FFBP precision baseline",
            "date": time.strftime("%Y-%m-%d %H:%M:%S %z"),
            "repeats": int(args.repeats),
            "warmups": int(args.warmups),
            "cpu_repeats": int(args.cpu_repeats),
            "direct_chunk_pulses": int(args.chunk_pulses),
            "block_size": int(backend.block_size),
            "nvrtc_compile_seconds": float(backend.compile_seconds),
            "timing_policy": "CUDA events with synchronization for resident compute; host clock for H2D, D2H and wall time",
            "fast_math": False,
        },
        "data": {
            "raw_file": str(dat_path),
            "raw_bytes": int(dat_path.stat().st_size),
            "raw_sha256": file_sha256(dat_path),
            "range_compressed_file": str(rc_path),
            "range_compressed_shape": list(data.shape),
            "range_compressed_dtype": str(data.dtype),
            "range_compressed_bytes": int(rc_path.stat().st_size),
            "range_compressed_sha256": file_sha256(rc_path),
            "position_file": str(position_path),
            "position_sha256": file_sha256(position_path),
            "valid_pulses": int(capture.n),
            "range_samples": int(capture.nr),
            "carrier_hz": FC,
            "sample_rate_hz": FS,
            "bandwidth_hz": float(capture.bandwidth),
            "pulse_width_s": float(capture.tp),
            "prf_hz": float(capture.prf),
            "range_start_m": float(range_start),
            "range_sample_spacing_m": float(data_dr),
            "signed_wavenumber_rad_m": float(k),
            "aperture_window": "scipy.signal.windows.hann(N, sym=True)",
            "per_pulse_phase": "zeros",
            "trajectory": "autofocus_position.npy",
        },
        "plan": {
            "extent": list(EXTENT),
            "side": SIDE,
            "leaf_size": LEAF_SIZE,
            "range_spacing_m": RANGE_SPACING,
            "angular_oversample": ANGULAR_OVERSAMPLE,
            "origin_m": plan.origin.tolist(),
            "r0_m": plan.r0,
            "dr_m": plan.dr,
            "u0": plan.u0,
            "du": plan.du,
            "shape": list(plan.shape),
            "crop_r": list(plan.crop_r),
            "crop_u": list(plan.crop_u),
            "counts": plan.counts,
            "planning_precision": "FP64 shared by all arithmetic modes",
        },
        "precision_policies": {
            "fp64": {
                "geometry_distance_sampling_phase": "FP64",
                "interpolation_accumulation": "FP64",
                "input_and_node_storage": "complex64",
            },
            "fp32": {
                "geometry_distance_sampling_phase": "FP32",
                "interpolation_accumulation": "FP32",
                "input_and_node_storage": "complex64",
            },
            "mixed": {
                "geometry_distance_sampling_phase": "FP64",
                "interpolation_accumulation": "FP32",
                "input_and_node_storage": "complex64",
            },
        },
        "environment": environment,
        "transfer": {
            "mmap_to_host_staging_ms": float(host_staging_ms),
            "h2d_event_ms": device.h2d_event_ms,
            "h2d_wall_ms": device.h2d_wall_ms,
            "bytes_uploaded": device.bytes_uploaded,
            "peak_cupy_pool_used_bytes": int(backend.peak_pool_used_bytes),
        },
        "operator_validation": operator,
        "cpu_reference": cpu_record,
        "gpu_runs": gpu_runs,
        "acceptance": {
            "fp64_relative_error_max": 1e-4,
            "mixed_relative_error_max": 1e-3,
            "fp32_engineering_target": fp32_target,
            "all_outputs_finite": True,
            "operator_validation_passed": None if operator is None else operator["passed"],
        },
        "files": {
            "kernel_source": str(backend.source_path),
            "script": str(Path(__file__).resolve()),
            "cpu_direct": str(out / "cpu_direct_reference.npy"),
            "cpu_ffbp": str(out / "cpu_ffbp_reference.npy"),
            "grid_axes": str(out / "grid_axes.npz"),
        },
    }
    write_json(out / "metrics.json", result)
    save_plots(out, images, direct_cpu, ffbp_cpu, plan, result)
    (out / "progress.json").unlink(missing_ok=True)
    print(f"Completed. Metrics: {out / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
