"""Independent raw-LFM point-target check for the RADARSAT BP adapter.

This generates continuous-delay raw echoes analytically, without using either
of the interpolation kernels being tested.  The geometry is the equivalent
straight-track slant plane used by the experiment, not an ECEF orbit model.

Run from the repository root: python vancouver_bp/validate.py
"""
from pathlib import Path
import hashlib
import json
import sys
import time

import numpy as np
from scipy import fft, signal

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "fast_bp"))
from ffbp import backproject_points, evaluate_image, form_ffbp

C = 299792458.0
FC = 5.3e9
FS = 32.317e6
PRF = 1256.98
V = 7062.0
TP = 41.74e-6
KR = -0.72135e12
T0 = 0.0065956


def analytic_raw_compressed(positions, target, samples=3072, oversample=4):
    """Return compressed echoes and independent analytic range predictions.

    A sample of the raw echo is
      rect((tau - 2R/c)/Tp) exp(j*pi*Kr*(tau-2R/c)^2) exp(-j*4*pi*fc*R/c).
    The reference chirp is centred at sample floor(L/2).  For linear
    correlation, extracting at offset L-1-floor(L/2) keeps output index q
    associated with the raw time tau=T0+q/Fs (received pulse centre).
    """
    distance = np.linalg.norm(positions - target, axis=1)
    tau = T0 + np.arange(samples, dtype=np.float64) / FS
    length = round(TP * FS)
    ref_time = (np.arange(length) - length // 2) / FS
    reference = (np.exp(1j * np.pi * KR * ref_time**2)
                 * signal.windows.hamming(length, sym=False))
    nfft = fft.next_fast_len(samples + length - 1)
    matched_spectrum = fft.fft(np.conj(reference[::-1]), nfft)
    offset = length - 1 - length // 2
    output = np.empty((len(positions), samples * oversample), np.complex64)
    for first in range(0, len(positions), 64):
        last = min(first + 64, len(positions))
        delta = tau[None, :] - 2 * distance[first:last, None] / C
        raw = (np.exp(1j * np.pi * KR * delta**2
                      - 1j * 4 * np.pi * FC * distance[first:last, None] / C)
               * (np.abs(delta) <= TP / 2))
        spectrum = fft.fft(raw, nfft, axis=1) * matched_spectrum[None, :]
        correlation = fft.ifft(spectrum, axis=1)
        upsampled = signal.resample(correlation, nfft * oversample, axis=1)
        output[first:last] = upsampled[:, offset * oversample:
                                     (offset + samples) * oversample]
    predicted_original_index = (2 * distance / C - T0) * FS
    return np.ascontiguousarray(output), distance, predicted_original_index, {
        "raw_samples": samples,
        "reference_samples": length,
        "linear_correlation_fft_length": nfft,
        "correlation_output_start_index_before_upsampling": offset,
        "range_fft_oversample": oversample,
        "compressed_sample_spacing_m": C / (2 * FS * oversample),
        "compressed_first_sample_range_m": C * T0 / 2,
        "minimum_raw_pulse_margin_samples": float(min(
            predicted_original_index.min() - length / 2,
            samples - 1 - predicted_original_index.max() - length / 2)),
    }


def run_case(pulses, doppler_hz=-7040.0):
    start = time.perf_counter()
    slow_time = (np.arange(pulses) - (pulses - 1) / 2) / PRF
    positions = np.column_stack((np.zeros(pulses), V * slow_time,
                                 np.zeros(pulses))).astype(np.float64)
    range_start = C * T0 / 2
    dr = C / (2 * FS)
    target_range = range_start + (1536 + 0.37) * dr
    u = doppler_hz * C / (2 * FC * V)
    target = np.array((target_range * np.sqrt(1 - u*u),
                       target_range * u, 0.0))
    data, distances, predicted, compression = analytic_raw_compressed(
        positions, target)
    peak_index = np.argmax(np.abs(data), axis=1) / 4
    compression_peak_error = float(np.max(np.abs(peak_index - predicted)))
    k = 4 * np.pi * FC / C
    weights = signal.windows.hann(pulses).astype(np.float64)
    phases = np.zeros(pulses, np.float64)
    # A small but finely sampled cut detects phase-sign and focus failures.
    # It is not used as an external resolution calibration target.
    offsets = np.linspace(-80, 80, 201)
    xyz = np.column_stack((np.full(len(offsets), target[0]),
                           target[1] + offsets, np.zeros(len(offsets))))
    direct = backproject_points(data, positions, xyz, range_start, dr/4,
                               k, weights, phases)
    wrong_sign = backproject_points(data, positions, xyz, range_start, dr/4,
                                   -k, weights, phases)
    peak_offset = float(offsets[np.argmax(np.abs(direct))])
    wrong_ratio = float(np.max(np.abs(wrong_sign)) / np.max(np.abs(direct)))
    extent = (target_range - 40, target_range + 40,
              u - 100 / target_range, u + 100 / target_range)
    fast, stats = form_ffbp(data, positions, range_start, dr/4, FC, extent,
                           leaf_size=32, range_spacing=0.5,
                           angular_oversample=6, weight=weights, verbose=False)
    sampled = evaluate_image(fast.image, fast.origin, fast.r0, fast.dr,
                             fast.u0, fast.du, xyz, k)
    d128 = direct.astype(np.complex128)
    s128 = sampled.astype(np.complex128)
    relative_error = float(np.linalg.norm(s128-d128) / np.linalg.norm(d128))
    coherence = float(abs(np.vdot(d128, s128)) /
                      (np.linalg.norm(d128)*np.linalg.norm(s128)))
    passed = (compression_peak_error < 0.14 and abs(peak_offset) <= 0.8
              and relative_error < 0.02 and wrong_ratio < 0.35
              and compression["minimum_raw_pulse_margin_samples"] > 0)
    return {
        "pulses": pulses,
        "doppler_at_aperture_centre_hz": doppler_hz,
        "target_xyz_slant_plane_m": target.tolist(),
        "target_range_at_aperture_centre_m": float(target_range),
        "target_original_sample_fraction": 0.37,
        "aperture_length_m": float(np.ptp(positions[:, 1])),
        "range_migration_peak_to_peak_m": float(np.ptp(distances)),
        "compression": compression,
        "max_range_compression_peak_error_original_samples": compression_peak_error,
        "expected_argmax_quantization_half_bin_original_samples": 0.125,
        "direct_bp_positive_k_along_track_peak_offset_m": peak_offset,
        "negative_k_to_positive_k_peak_amplitude_ratio": wrong_ratio,
        "ffbp_relative_complex_error_on_common_cut": relative_error,
        "ffbp_ser_db_on_common_cut": float(-20*np.log10(relative_error)),
        "ffbp_complex_coherence_on_common_cut": coherence,
        "ffbp_polar_shape": list(fast.image.shape),
        "ffbp_processing_seconds_including_first_jit_if_needed": stats["elapsed_s"],
        "case_seconds_including_compression_and_jit_if_needed": time.perf_counter()-start,
        "passed": bool(passed),
    }


def main():
    cases = [run_case(256), run_case(1728)]
    report = {
        "purpose": "Independent analytic-raw, off-grid target validation; no PGA",
        "model": "Equivalent straight-track slant plane; analytic negative chirp and negative carrier phase",
        "parameter_source": "RADARSAT.m nominal parameters, exact SI speed of light",
        "parameters": dict(c_m_s=C, carrier_hz=FC, sample_rate_hz=FS,
                           prf_hz=PRF, velocity_m_s=V, pulse_width_s=TP,
                           chirp_rate_hz_s=KR, first_fast_time_s=T0),
        "bp_compensation": "+j*4*pi*fc*R/c for the generated negative carrier phase",
        "range_timing_convention": "Output index q corresponds to received pulse centre at T0+q/Fs",
        "scope": "Tests implementation conventions and BP/FFBP consistency. Does not prove the real MAT data timing, orbit or chirp convention.",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "ffbp_source_sha256": hashlib.sha256((ROOT / "fast_bp/ffbp.py").read_bytes()).hexdigest(),
        "cases": cases,
        "passed": all(case["passed"] for case in cases),
    }
    output = ROOT / "vancouver_bp/results/synthetic_validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    if not report["passed"]:
        raise SystemExit("Synthetic validation failed")


if __name__ == "__main__":
    main()
