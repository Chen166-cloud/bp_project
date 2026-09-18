"""Explicit SAR metric conventions; operates on linear complex samples."""
import numpy as np
from scipy.signal import find_peaks


def focus_metrics(z):
    z = np.asarray(z)
    valid = np.isfinite(z)
    out = dict(samples=int(z.size), finite_fraction=float(valid.mean()),
               nonzero_fraction=float(np.mean(abs(z) > 0)))
    if not valid.all() or not np.any(abs(z) > 0):
        return dict(out, valid=False, reason='Non-finite or zero-energy image')
    a = abs(z).astype(np.float64).ravel()
    a /= a.max()
    intensity = a*a
    p = intensity/intensity.sum()
    entropy = float(-np.sum(p[p > 0]*np.log(p[p > 0])))
    return dict(out, valid=True, entropy_nats=entropy,
                normalized_entropy=entropy/np.log(z.size),
                intensity_contrast=float(intensity.std()/intensity.mean()),
                amplitude_contrast=float(a.std()/a.mean()),
                brightest_1pct_energy_fraction=float(np.sort(p)[-max(1, z.size//100):].sum()))


def fidelity(reference, estimate):
    a = np.asarray(reference, np.complex128).ravel()
    b = np.asarray(estimate, np.complex128).ravel()
    norm = np.linalg.norm(a)
    err = np.linalg.norm(b-a)/norm
    mask = abs(a) >= abs(a).max()*10**(-30/20)
    ph = np.angle(b[mask]*a[mask].conj())
    offset = np.angle(np.mean(np.exp(1j*ph)))
    residual = np.angle(np.exp(1j*(ph-offset)))
    return dict(relative_complex_error=float(err),
                ser_db=float(-20*np.log10(max(err, 1e-300))),
                complex_coherence=float(abs(np.vdot(a,b))/(norm*np.linalg.norm(b))),
                phase_bias_deg=float(np.rad2deg(offset)),
                phase_std_deg_above_minus30db=float(np.rad2deg(residual.std())),
                phase_mask_samples=int(mask.sum()))


def impulse_metrics(x, z):
    """1-D half-power IRW; first-minima mainlobe; fixed finite cut sidelobes.

    PSLR = 10log10(max(side power)/peak power).
    ISLR = 10log10(integral(side power)/integral(main power)).
    These are 1-D first-null metrics, NOT the Sentinel-1 2-D rectangle metric.
    """
    x = np.asarray(x, float); p = abs(np.asarray(z))**2
    if not np.isfinite(p).all() or p.max() <= 0:
        return dict(valid=False, reason='Invalid power')
    peak = int(np.argmax(p)); p = p/p[peak]
    left = np.flatnonzero(p[:peak] <= .5)
    right = np.flatnonzero(p[peak+1:] <= .5)+peak+1
    if not len(left) or not len(right):
        return dict(valid=False, reason='Half-power crossing absent')
    il, ir = int(left[-1]), int(right[0])
    xl = np.interp(.5, p[il:il+2], x[il:il+2])
    xr = np.interp(.5, p[ir-1:ir+1][::-1], x[ir-1:ir+1][::-1])
    mins = find_peaks(-p)[0]
    ml, mr = mins[mins < il], mins[mins > ir]
    if not len(ml) or not len(mr):
        return dict(valid=True, irw_m=float(xr-xl), peak_offset_m=float(x[peak]),
                    sidelobes_valid=False, reason='First minima absent')
    ml, mr = int(ml[-1]), int(mr[0])
    em = np.trapz(p[ml:mr+1], x[ml:mr+1])
    es = np.trapz(p[:ml+1], x[:ml+1])+np.trapz(p[mr:], x[mr:])
    return dict(valid=True, sidelobes_valid=True, irw_m=float(xr-xl),
                peak_offset_m=float(x[peak]),
                pslr_db=float(10*np.log10(max(p[:ml].max(), p[mr+1:].max()))),
                islr_db=float(10*np.log10(es/em)),
                mainlobe_bounds_m=[float(x[ml]), float(x[mr])],
                mainlobe_minima_db=[float(10*np.log10(p[ml]+1e-300)), float(10*np.log10(p[mr]+1e-300))],
                integration_bounds_m=[float(x[0]), float(x[-1])], step_m=float(x[1]-x[0]))
