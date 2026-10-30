"""RADARSAT-1 Vancouver raw-echo adapter; no data-driven phase correction.

Slant-plane coordinates use x=closest-approach slant distance, y=along track.
This is the textbook equivalent straight-track model, not an ECEF orbit fit.
"""
from __future__ import annotations
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
from scipy import fft, signal
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "fast_bp"))
from gpu_backend import CudaBaseline, build_ffbp_plan, cp, cuda_environment

OUT = ROOT / "vancouver_bp" / "results"
C = 299792458.0
PARAMETERS = dict(carrier_hz=5.3e9, sample_rate_hz=32.317e6,
                  chirp_rate_hz_s=-0.72135e12, pulse_duration_s=41.74e-6,
                  prf_hz=1256.98, first_sample_time_s=6.5956e-3,
                  equivalent_speed_m_s=7062.0, doppler_ambiguity_number=-6)

def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

def prepare(oversample=4, window="hamming"):
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / f"range_compressed_os{oversample}_{window}.npy"
    meta_path = cache.with_suffix(".json")
    if cache.exists() and meta_path.exists():
        return np.load(cache, mmap_mode="r"), json.loads(meta_path.read_text(encoding="utf-8"))
    start = time.perf_counter()
    raw = loadmat(ROOT / "CDdata1.mat")["data"].astype(np.complex64)
    na, nr = raw.shape
    p = PARAMETERS
    fs, tp = p["sample_rate_hz"], p["pulse_duration_s"]
    length = round(tp * fs)
    t = (np.arange(length) - length//2) / fs
    ref = np.exp(1j*np.pi*p["chirp_rate_hz_s"]*t*t).astype(np.complex64)
    w = signal.windows.hamming(length, sym=False) if window == "hamming" else np.ones(length)
    ref *= w.astype(np.float32)
    nf = fft.next_fast_len(nr + length - 1)
    spectrum = fft.fft(raw, nf, axis=1, workers=8)
    spectrum *= fft.fft(np.conj(ref[::-1]), nf)[None, :]
    correlation = fft.ifft(spectrum, axis=1, workers=8)
    offset = length-1-length//2
    if oversample > 1:
        correlation = signal.resample(correlation, nf*oversample, axis=1)
    rc = np.ascontiguousarray(correlation[:, offset*oversample:(offset+nr)*oversample],
                              dtype=np.complex64)
    rstart = C*p["first_sample_time_s"]/2
    dr = C/(2*fs*oversample)
    pair = np.sum(raw[1:].astype(np.complex128)*raw[:-1].conj())
    fd_base = float(np.angle(pair)*p["prf_hz"]/(2*np.pi))
    fd = fd_base + p["doppler_ambiguity_number"]*p["prf_hz"]
    meta = dict(parameters=p, raw_shape=[na,nr], raw_sha256=hashlib.sha256(
        (ROOT/"CDdata1.mat").read_bytes()).hexdigest(), range_start_m=rstart,
        compressed_spacing_m=dr, oversample=oversample, range_window=window,
        reference_samples=length, correlation_crop_offset_samples=offset,
        range_coordinate_convention="received chirp centre at raw sample k",
        valid_full_reference_raw_sample_interval=[length//2, nr-(length-length//2)],
        valid_full_reference_slant_range_m=[rstart+length//2*C/(2*fs),
            rstart+(nr-(length-length//2))*C/(2*fs)],
        baseband_doppler_hz=fd_base, total_doppler_centroid_hz=fd,
        sin_squint=fd*C/(2*p["carrier_hz"]*p["equivalent_speed_m_s"]),
        preprocessing_seconds=time.perf_counter()-start,
        autofocus=False, agc_compensation="not repeated; extraction history unavailable",
        source="existing local CDdata1.mat; acquisition provenance and Vancouver attribution unverified; model parameters from local RADARSAT.m",
        source_provenance_verified=False)
    np.save(cache, rc)
    write_json(meta_path, meta)
    print(json.dumps(meta,ensure_ascii=False), flush=True)
    return rc, meta

def geometry(na):
    t = (np.arange(na)-(na-1)/2)/PARAMETERS["prf_hz"]
    pos = np.zeros((na,3), dtype=np.float64)
    pos[:,1] = t*PARAMETERS["equivalent_speed_m_s"]
    return pos

class Imager:
    def __init__(self, oversample=4, window="hamming"):
        self.data, self.meta = prepare(oversample,window)
        self.position = geometry(len(self.data))
        self.weight = signal.windows.hamming(len(self.data),sym=False)
        self.phase = np.zeros(len(self.data))
        self.backend = CudaBaseline()
        self.k = 4*np.pi*PARAMETERS["carrier_hz"]/C
        self.device = None
        self.meta["gpu"] = cuda_environment()
        self.meta["azimuth_window"] = "Hamming across all 1728 recorded pulses"

    def plan(self, extent, spacing, angular_oversample=4, leaf_size=64):
        return build_ffbp_plan(self.position, PARAMETERS["carrier_hz"], extent,
                leaf_size=leaf_size,range_spacing=spacing,angular_oversample=angular_oversample)

    def upload(self, plan):
        if self.device is None:
            self.device = self.backend.upload(self.data,self.position,self.weight,self.phase,plan)
        else:
            self.device.origins64=cp.asarray(plan.origins)
            self.device.origins32=cp.asarray(plan.origins.astype(np.float32))

    def image(self, plan, algorithm="bp"):
        self.assert_no_autofocus()
        self.upload(plan)
        fn = self.backend.run_direct if algorithm=="bp" else self.backend.run_ffbp
        z,event,wall = fn(plan,self.device,"fp64",self.meta["range_start_m"],
                         self.meta["compressed_spacing_m"],self.k)
        image,download = self.backend.download(z)
        del z
        return image, dict(algorithm=algorithm,kernel_ms=event,wall_ms=wall,
                          download_ms=download,shape=list(plan.shape),plan_counts=plan.counts)

    def assert_no_autofocus(self):
        """Fail before imaging if an extra phase or corrected trajectory appears."""
        if self.phase.shape != (len(self.data),) or not np.all(self.phase == 0):
            raise ValueError('No-PGA experiment requires identically zero additional pulse phase')
        if not np.array_equal(self.position, geometry(len(self.data))):
            raise ValueError('No-PGA experiment requires the fixed equivalent straight trajectory')

def axes(plan):
    return plan.r0+np.arange(plan.nr)*plan.dr, plan.u0+np.arange(plan.nu)*plan.du

def fidelity(reference, estimate):
    a=reference.astype(np.complex128).ravel(); b=estimate.astype(np.complex128).ravel()
    er=float(np.linalg.norm(a-b)/np.linalg.norm(a))
    return dict(relative_complex_error=er,ser_db=float(-20*np.log10(max(er,1e-300))),
                complex_coherence=float(abs(np.vdot(a,b))/(np.linalg.norm(a)*np.linalg.norm(b))),
                relative_amplitude_error=float(np.linalg.norm(abs(a)-abs(b))/np.linalg.norm(abs(a))))

def save_image(path, image, plan, **extra):
    r,u=axes(plan)
    np.savez_compressed(path,image=image,r=r,u=u,origin=plan.origin,**extra)
