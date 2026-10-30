"""Point-response measurements from linear complex RADARSAT BP images.

The measured IRW is a half-power (-3.0103 dB) width.  On uncalibrated real
scatterers it is an apparent response width, not an independently calibrated
system resolution.  Coordinates are slant range and along-track metres unless
the caller supplies a different, explicitly documented coordinate transform.

No background subtraction, image sharpening, or amplitude-domain interpolation
is performed.  A light smoothing is used only to locate mainlobe minima; all
width, peak, and energy measurements use the original power samples.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


def _trapz(y, x, axis=-1):
    return np.trapezoid(y, x, axis=axis) if hasattr(np, "trapezoid") else np.trapz(y, x, axis=axis)


def _db(value):
    return float(10.0 * np.log10(max(float(value), np.finfo(float).tiny)))


def _axis_and_signal(x, z):
    x = np.asarray(x, dtype=np.float64)
    z = np.asarray(z)
    if x.ndim != 1 or z.ndim != 1 or len(x) != len(z) or len(x) < 5:
        raise ValueError("x and z must be matching one-dimensional arrays with at least five samples")
    if not np.all(np.isfinite(x)) or np.any(np.diff(x) <= 0):
        raise ValueError("x must be finite and strictly increasing")
    return x, z


def _crossing(x0, x1, p0, p1, level=0.5):
    if p1 == p0:
        return float((x0 + x1) / 2)
    return float(x0 + (level - p0) * (x1 - x0) / (p1 - p0))


def _minimum_indices(x, p, peak, il, ir, irw):
    # About 4% of the measured IRW, capped well below a mainlobe width.  This
    # suppresses oversampled noise ripples without interpreting every raw local
    # fluctuation as the first physical null.  This does not denoise the metric.
    dx = float(np.median(np.diff(x)))
    sigma = min(4.0, max(0.5, 0.04 * irw / dx))
    smooth = gaussian_filter1d(p, sigma, mode="nearest")
    minima = find_peaks(-smooth)[0]
    left = minima[(minima < il) & (minima > 0)]
    right = minima[(minima > ir) & (minima < len(p) - 1)]

    def choose(indices, reverse):
        indices = indices[::-1] if reverse else indices
        # The first sufficiently deep minimum is preferable to an arbitrarily
        # small derivative wiggle near the -3 dB crossing.  If no -10 dB minimum
        # exists, retain a shallow candidate as an explicitly unreliable bound.
        for limit in (0.1, 0.25):
            candidates = [int(i) for i in indices if p[i] <= limit]
            if candidates:
                return candidates[0]
        return None

    return choose(left, True), choose(right, False), sigma


def profile_metrics(x, z, *, expected_irw_m=None, integration_half_width_m=None, peak_index=None):
    """Measure a complex 1-D response with explicit finite-window conventions.

    PSLR = 10 log10(max(side power) / peak power).
    ISLR = 10 log10(integral(side power) / integral(main power)).
    Mainlobe boundaries are the first qualified minima outside both half-power
    crossings.  A minimum shallower than -15 dB, a truncated requested window,
    or a second half-power component is flagged; raw values remain available.
    ``expected_irw_m`` is a diagnostic, never a forced measurement boundary.
    """
    x, z = _axis_and_signal(x, z)
    power = np.abs(z.astype(np.complex128)) ** 2
    result = dict(
        valid=False, sidelobes_valid=False,
        width_definition="power half-maximum; -3.0102999566 dB",
        sidelobe_definition="1-D first-qualified-minima mainlobe; finite integration window",
        background_subtracted=False,
        expected_irw_m=None if expected_irw_m is None else float(expected_irw_m),
        warnings=[],
    )
    if not np.all(np.isfinite(power)) or not np.any(power > 0):
        return dict(result, reason="nonfinite or zero power")
    peak_all = int(np.argmax(power)) if peak_index is None else int(peak_index)
    if not 0 <= peak_all < len(power) or power[peak_all] <= 0:
        raise ValueError("peak_index must identify a positive-power sample inside the supplied cut")
    centre = float(x[peak_all])
    requested_truncated = False
    if integration_half_width_m is not None:
        hw = float(integration_half_width_m)
        if hw <= 0:
            raise ValueError("integration_half_width_m must be positive")
        requested_truncated = centre - hw < x[0] or centre + hw > x[-1]
        mask = (x >= centre - hw) & (x <= centre + hw)
        original_indices = np.flatnonzero(mask)
        x, power = x[mask], power[mask]
    else:
        original_indices = np.arange(len(x))
    if len(x) < 5:
        return dict(result, reason="integration window contains fewer than five samples")
    peak = int(peak_all - original_indices[0])
    peak_power = float(power[peak])
    p = power / peak_power
    dx = np.diff(x)
    result.update(
        peak_coordinate_m=float(x[peak]), peak_sample_index=int(original_indices[peak]),
        peak_power=peak_power, peak_amplitude=float(np.sqrt(peak_power)),
        step_m=float(np.median(dx)), uniform_axis=bool(np.allclose(dx, np.median(dx), rtol=1e-5, atol=1e-9)),
        integration_bounds_m=[float(x[0]), float(x[-1])],
        requested_integration_half_width_m=integration_half_width_m,
        integration_window_truncated=bool(requested_truncated),
    )
    below_left = np.flatnonzero(p[:peak] <= 0.5)
    below_right = np.flatnonzero(p[peak + 1:] <= 0.5) + peak + 1
    if not len(below_left) or not len(below_right):
        return dict(result, reason="half-power crossings absent on one or both sides")
    il, ir = int(below_left[-1]), int(below_right[0])
    xl = _crossing(x[il], x[il + 1], p[il], p[il + 1])
    xr = _crossing(x[ir - 1], x[ir], p[ir - 1], p[ir])
    irw = float(xr - xl)
    result.update(valid=True, irw_m=irw, apparent_resolution_m=irw,
                  half_power_bounds_m=[xl, xr], samples_per_irw=float(irw / np.median(dx)))
    if irw / np.median(dx) < 8:
        result["warnings"].append("fewer than eight samples per IRW; use finer physical sampling for point-response analysis")
    if expected_irw_m is not None:
        result["irw_to_expected_ratio"] = irw / float(expected_irw_m)
    ml, mr, sigma = _minimum_indices(x, p, peak, il, ir, irw)
    result["minima_detection_smoothing_sigma_samples"] = float(sigma)
    if ml is None or mr is None:
        return dict(result, reason="qualified first minima absent; sidelobe ratios not identified")
    if ml == 0 or mr == len(p) - 1:
        return dict(result, reason="mainlobe touches integration boundary")
    main = float(_trapz(p[ml:mr + 1], x[ml:mr + 1]))
    side = float(_trapz(p[:ml + 1], x[:ml + 1]) + _trapz(p[mr:], x[mr:]))
    side_values = np.r_[p[:ml], p[mr + 1:]]
    if main <= 0 or not len(side_values):
        return dict(result, reason="main/side integration regions invalid")
    depths = [_db(p[ml]), _db(p[mr])]
    # Estimate a background level only as a diagnostic.  Subtracting clutter
    # could make a real scatterer look more ideal and is intentionally avoided.
    edge_count = max(2, len(p) // 10)
    outer_median = float(np.median(np.r_[p[:edge_count], p[-edge_count:]]))
    half_components = np.count_nonzero(np.diff(np.r_[False, p >= .5, False].astype(int)) == 1)
    nulls_deep = bool(max(depths) <= -15.0)
    reliable = nulls_deep and not requested_truncated and half_components == 1
    result.update(
        sidelobes_valid=bool(reliable),
        pslr_db=_db(float(np.max(side_values))), islr_db=_db(side / main),
        mainlobe_bounds_m=[float(x[ml]), float(x[mr])],
        first_minima_width_m=float(x[mr] - x[ml]),
        mainlobe_minima_db=depths,
        mainlobe_sample_indices=[int(original_indices[ml]), int(original_indices[mr])],
        normalized_main_energy_m=main, normalized_side_energy_m=side,
        outer_background_median_db=_db(outer_median),
        estimated_peak_to_outer_background_db=-_db(outer_median),
        half_power_components=int(half_components),
        sidelobe_confidence="qualified" if reliable else "limited",
    )
    if not nulls_deep:
        result["warnings"].append("mainlobe minima shallower than -15 dB: sidelobe ratios include clutter or overlapping responses")
    if requested_truncated:
        result["warnings"].append("requested integration window exceeds supplied cut")
    if half_components != 1:
        result["warnings"].append("multiple half-power components: another response lies in the measurement window")
    return result


def analyze_target(r, y, image, *, expected_irw=(None, None), peak_hint=None,
                   search_radius_m=None, integration_half_widths_m=None,
                   return_profiles=False):
    """Measure one target in image[r_index, along_track_index].

    ``peak_hint=(r_m,y_m)`` plus ``search_radius_m=(r_radius,y_radius)`` limits
    localization to the intended scatterer; this does not crop integration.
    The r/y axes should be physically oversampled before calling.  For plots,
    ``return_profiles=True`` returns ``(json_metrics, array_profiles)``.
    """
    r, y = np.asarray(r, float), np.asarray(y, float)
    image = np.asarray(image)
    if image.shape != (len(r), len(y)):
        raise ValueError("image shape must be (len(r), len(y))")
    if np.any(np.diff(r) <= 0) or np.any(np.diff(y) <= 0):
        raise ValueError("physical axes must be strictly increasing")
    power = np.abs(image.astype(np.complex128)) ** 2
    if not np.all(np.isfinite(power)) or not np.any(power > 0):
        raise ValueError("target image has nonfinite or zero power")
    selection = np.ones(image.shape, bool)
    if peak_hint is not None and search_radius_m is not None:
        dr, dy = search_radius_m if np.ndim(search_radius_m) else (search_radius_m,) * 2
        selection = (np.abs(r[:, None] - peak_hint[0]) <= dr) & (np.abs(y[None, :] - peak_hint[1]) <= dy)
        if not np.any(selection):
            raise ValueError("peak search window contains no pixels")
    ir, iy = np.unravel_index(np.argmax(np.where(selection, power, -np.inf)), image.shape)
    half = (None, None) if integration_half_widths_m is None else integration_half_widths_m
    range_metrics = profile_metrics(r, image[:, iy], expected_irw_m=expected_irw[0], integration_half_width_m=half[0], peak_index=ir)
    az_metrics = profile_metrics(y, image[ir, :], expected_irw_m=expected_irw[1], integration_half_width_m=half[1], peak_index=iy)
    result = dict(
        peak=dict(range_m=float(r[ir]), azimuth_m=float(y[iy]), index=[int(ir), int(iy)],
                  amplitude=float(np.sqrt(power[ir, iy])), power=float(power[ir, iy])),
        coordinates=dict(range="slant-range metres supplied by caller", azimuth="along-track metres supplied by caller"),
        range=range_metrics, azimuth=az_metrics,
        apparent_resolution_m=dict(slant_range=range_metrics.get("irw_m"), azimuth=az_metrics.get("irw_m")),
        interpretation="Uncalibrated strong-scatterer apparent response; no independent point-target or ground-range resolution certification.",
        background_subtracted=False,
    )
    # Optional rectangle metric: both 1-D cuts define the rectangle.  This is
    # labelled explicitly and is not interchangeable with CEOS/ESA 2xIRW rules.
    two = dict(valid=False, definition="2-D rectangle bounded by the two 1-D qualified first minima; same finite cut window")
    if "mainlobe_sample_indices" in range_metrics and "mainlobe_sample_indices" in az_metrics:
        a, b = range_metrics["mainlobe_sample_indices"]
        c, d = az_metrics["mainlobe_sample_indices"]
        ri = (r >= range_metrics["integration_bounds_m"][0]) & (r <= range_metrics["integration_bounds_m"][1])
        yi = (y >= az_metrics["integration_bounds_m"][0]) & (y <= az_metrics["integration_bounds_m"][1])
        pr = power[np.ix_(ri, yi)] / power[ir, iy]
        total = float(_trapz(_trapz(pr, y[yi], axis=1), r[ri]))
        main = float(_trapz(_trapz(power[a:b + 1, c:d + 1] / power[ir, iy], y[c:d + 1], axis=1), r[a:b + 1]))
        rm, ym = np.meshgrid(r[ri], y[yi], indexing="ij")
        smask = (rm < r[a]) | (rm > r[b]) | (ym < y[c]) | (ym > y[d])
        if total > main > 0 and np.any(smask):
            two.update(valid=bool(range_metrics["sidelobes_valid"] and az_metrics["sidelobes_valid"]),
                       islr_db=_db((total - main) / main), pslr_db=_db(pr[smask].max()),
                       mainlobe_rectangle_m=[float(r[a]), float(r[b]), float(y[c]), float(y[d])],
                       integration_rectangle_m=[float(r[ri][0]), float(r[ri][-1]), float(y[yi][0]), float(y[yi][-1])])
    result["two_dimensional_first_minima_rectangle"] = two
    if return_profiles:
        return result, dict(range_x_m=r.copy(), range_complex=image[:, iy].copy(),
                            azimuth_x_m=y.copy(), azimuth_complex=image[ir, :].copy())
    return result


def selfcheck():
    """Independent analytic sinc checks; meaningful convention validation."""
    from scipy.integrate import quad
    from scipy.optimize import brentq, minimize_scalar

    scale = 5.0
    x = np.arange(-100.0, 100.0 + .005, .01)
    z = np.sinc(x / scale) * np.exp(1j * (0.13 * x + .7))
    got = profile_metrics(x, z)
    half = brentq(lambda u: np.sinc(u) ** 2 - .5, 0, 1)
    expected_irw = 2 * half * scale
    first_sidelobe = minimize_scalar(lambda u: -np.sinc(u) ** 2, bounds=(1, 2), method="bounded")
    expected_pslr = _db(-first_sidelobe.fun)
    main = quad(lambda u: np.sinc(u) ** 2, -1, 1)[0]
    side = 2 * quad(lambda u: np.sinc(u) ** 2, 1, 20, limit=300)[0]
    expected_islr = _db(side / main)
    assert got["sidelobes_valid"], got
    assert abs(got["irw_m"] - expected_irw) < 1e-4, got
    assert abs(got["pslr_db"] - expected_pslr) < .001, got
    assert abs(got["islr_db"] - expected_islr) < .002, got
    gaussian = profile_metrics(x, np.exp(-.5 * (x / 3) ** 2))
    assert gaussian["valid"] and not gaussian["sidelobes_valid"]
    flat = profile_metrics(x, np.ones_like(x))
    assert not flat["valid"]
    shifted = profile_metrics(x, np.sinc((x - .123) / scale))
    assert abs(shifted["irw_m"] - expected_irw) < .001
    r = np.arange(-30., 30.01, .1)
    y = np.arange(-40., 40.01, .1)
    image = np.sinc(r[:, None] / 4) * np.sinc(y[None, :] / 6) * np.exp(1j * (.2 * r[:, None] + .3 * y[None, :]))
    target = analyze_target(r, y, image, integration_half_widths_m=(25, 35))
    assert abs(target["range"]["irw_m"] - 2 * half * 4) < .003
    assert abs(target["azimuth"]["irw_m"] - 2 * half * 6) < .003
    assert target["two_dimensional_first_minima_rectangle"]["valid"]
    return dict(passed=True, sinc=got,
                independent_expectations=dict(irw_m=expected_irw, pslr_db=expected_pslr, finite_window_islr_db=expected_islr),
                gaussian_minima_absent_correctly_flagged=True, flat_half_power_absent_correctly_flagged=True,
                off_grid_peak_irw_m=shifted["irw_m"], two_dimensional_interface_passed=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--selfcheck", type=Path)
    args = parser.parse_args()
    result = selfcheck()
    if args.selfcheck:
        args.selfcheck.parent.mkdir(parents=True, exist_ok=True)
        args.selfcheck.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
