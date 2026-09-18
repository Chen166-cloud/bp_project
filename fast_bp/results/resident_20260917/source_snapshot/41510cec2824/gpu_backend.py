"""CUDA C++ backend for reproducible direct BP and FFBP baselines.

The kernels are compiled from ``gpu/kernels.cu`` with NVIDIA NVRTC through
CuPy.  The implementation deliberately mirrors ``ffbp.py`` instead of applying
kernel fusion or fast-math optimizations.  This keeps the first GPU result a
numerical baseline against which later optimizations can be measured.
"""
from __future__ import annotations

import os
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


_DLL_HANDLES = []


def _configure_cuda_runtime():
    """Locate the local CUDA runtime before CuPy is imported.

    This workstation has CUDA 12.2 bundled with MATLAB.  ``FAST_BP_CUDA_ROOT``
    and ``FAST_BP_CUDA_DLL_DIR`` allow the same code to use a standalone CUDA
    Toolkit on another machine without editing source files.
    """
    default_root = Path(r"D:\Program Files\MATLAB\R2024a\sys\cuda\win64\cuda")
    default_dll = Path(r"D:\Program Files\MATLAB\R2024a\bin\win64")
    configured = os.environ.get("FAST_BP_CUDA_ROOT") or os.environ.get("CUDA_PATH")
    cuda_root = Path(configured) if configured else default_root
    if not cuda_root.exists():
        raise RuntimeError(
            "CUDA runtime not found. Set FAST_BP_CUDA_ROOT to a CUDA Toolkit root."
        )
    configured_dll = os.environ.get("FAST_BP_CUDA_DLL_DIR")
    candidates = [Path(configured_dll)] if configured_dll else []
    candidates.extend([cuda_root / "bin", default_dll])
    dll_dirs = []
    for candidate in candidates:
        if candidate.exists() and candidate not in dll_dirs:
            dll_dirs.append(candidate)
    os.environ["CUDA_PATH"] = str(cuda_root)
    os.environ["PATH"] = os.pathsep.join([*(str(p) for p in dll_dirs), os.environ["PATH"]])
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        for directory in dll_dirs:
            _DLL_HANDLES.append(os.add_dll_directory(str(directory)))
    return cuda_root, dll_dirs


CUDA_ROOT, CUDA_DLL_DIRS = _configure_cuda_runtime()
import cupy as cp  # noqa: E402  (CUDA DLL search path must be configured first)


MODES = ("fp64", "fp32", "mixed")


@dataclass
class NodePlan:
    index: int
    a: int
    b: int
    origin: np.ndarray
    r0: float
    dr: float
    nr: int
    u0: float
    du: float
    nu: int
    side: int
    left: Optional["NodePlan"] = None
    right: Optional["NodePlan"] = None

    @property
    def is_leaf(self):
        return self.left is None


@dataclass
class FFBPPlan:
    root: NodePlan
    origins: np.ndarray
    crop_r: tuple[int, int]
    crop_u: tuple[int, int]
    origin: np.ndarray
    r0: float
    dr: float
    nr: int
    u0: float
    du: float
    nu: int
    side: int
    counts: dict
    extent: tuple[float, float, float, float]
    leaf_size: int
    angular_oversample: float

    @property
    def shape(self):
        return self.nr, self.nu


@dataclass
class DeviceInputs:
    data: cp.ndarray
    pos64: cp.ndarray
    pos32: cp.ndarray
    weight64: cp.ndarray
    weight32: cp.ndarray
    phase64: cp.ndarray
    phase32: cp.ndarray
    origins64: cp.ndarray
    origins32: cp.ndarray
    h2d_event_ms: float
    h2d_wall_ms: float
    bytes_uploaded: int


def build_ffbp_plan(
    position,
    fc,
    extent,
    side=1,
    leaf_size=128,
    range_spacing=0.25,
    angular_oversample=2.5,
):
    """Create the exact FP64 node tree and crop used by ``form_ffbp``.

    The returned plan is shared by every GPU precision policy.  In particular,
    FP32 never controls ``ceil``, guard cells, node bounds, or root cropping.
    """
    position = np.ascontiguousarray(position, dtype=np.float64)
    wavelength = 299792458.0 / abs(float(fc))
    root_origin = position.mean(axis=0)
    rmin, rmax, umin, umax = (float(x) for x in extent)
    us = np.linspace(umin, umax, 129)
    rs = np.linspace(rmin, rmax, 129)
    rb = np.r_[np.full_like(us, rmin), np.full_like(us, rmax), rs, rs]
    ub = np.r_[us, us, np.full_like(rs, umin), np.full_like(rs, umax)]
    ground = np.sqrt(rb**2 - root_origin[2] ** 2)
    bx = root_origin[0] + side * ground * np.sqrt(1.0 - ub**2)
    by = root_origin[1] + ground * ub
    counts = {
        "leaves": 0,
        "merges": 0,
        "leaf_pixel_pulse_products": 0,
        "merged_pixels": 0,
    }
    nodes = []

    def node(a, b):
        origin = position[a:b].mean(axis=0)
        dx = bx - origin[0]
        dy = by - origin[1]
        gr = np.sqrt(dx * dx + dy * dy)
        rr = np.sqrt(gr * gr + origin[2] ** 2)
        uu = dy / gr
        aperture = np.linalg.norm(np.ptp(position[a:b, :2], axis=0))
        du = min(0.025, wavelength / (2.0 * max(aperture, 0.1) * angular_oversample))
        u0 = max(-0.995, float(uu.min()) - 8.0 * du)
        u1 = min(0.995, float(uu.max()) + 8.0 * du)
        r0 = float(rr.min()) - 8.0 * range_spacing
        nr = int(np.ceil((rr.max() - r0) / range_spacing)) + 9
        nu = int(np.ceil((u1 - u0) / du)) + 1
        current = NodePlan(
            index=len(nodes),
            a=a,
            b=b,
            origin=np.ascontiguousarray(origin, dtype=np.float64),
            r0=r0,
            dr=float(range_spacing),
            nr=nr,
            u0=u0,
            du=float(du),
            nu=nu,
            side=int(side),
        )
        nodes.append(current)
        if b - a <= leaf_size:
            counts["leaves"] += 1
            counts["leaf_pixel_pulse_products"] += nr * nu * (b - a)
        else:
            mid = (a + b) // 2
            current.left = node(a, mid)
            current.right = node(mid, b)
            counts["merges"] += 1
            counts["merged_pixels"] += 2 * nr * nu
        return current

    root = node(0, len(position))
    root_r = root.r0 + root.dr * np.arange(root.nr)
    root_u = root.u0 + root.du * np.arange(root.nu)
    ri = np.flatnonzero((root_r >= rmin) & (root_r <= rmax))
    ui = np.flatnonzero((root_u >= umin) & (root_u <= umax))
    if len(ri) == 0 or len(ui) == 0:
        raise ValueError("Requested extent does not intersect the planned root grid")
    origins = np.ascontiguousarray(np.stack([item.origin for item in nodes]), dtype=np.float64)
    counts = dict(counts)
    counts.update(
        nodes=len(nodes),
        shape=[int(len(ri)), int(len(ui))],
        direct_pixel_pulse_products=int(len(ri) * len(ui) * len(position)),
    )
    return FFBPPlan(
        root=root,
        origins=origins,
        crop_r=(int(ri[0]), int(ri[-1]) + 1),
        crop_u=(int(ui[0]), int(ui[-1]) + 1),
        origin=root.origin.copy(),
        r0=float(root.r0 + ri[0] * root.dr),
        dr=float(root.dr),
        nr=int(len(ri)),
        u0=float(root.u0 + ui[0] * root.du),
        du=float(root.du),
        nu=int(len(ui)),
        side=int(side),
        counts=counts,
        extent=tuple(float(x) for x in extent),
        leaf_size=int(leaf_size),
        angular_oversample=float(angular_oversample),
    )


class CudaBaseline:
    """Compile and execute the parity-first CUDA kernels."""

    kernel_names = (
        "bp_polar_fp64",
        "bp_polar_fp32",
        "bp_polar_mixed",
        "direct_chunk_fp64",
        "direct_chunk_fp32",
        "direct_chunk_mixed",
        "regrid_fp64",
        "regrid_fp32",
        "regrid_mixed",
        "add_float2_inplace",
        "cast_double2_to_float2",
        "crop_float2",
    )

    def __init__(self, block_size=128):
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.block_size = int(block_size)
        source_path = Path(__file__).resolve().parent / "gpu" / "kernels.cu"
        source = source_path.read_text(encoding="utf-8")
        start = time.perf_counter()
        self.module = cp.RawModule(
            code=source,
            options=("-std=c++11",),
            name_expressions=self.kernel_names,
        )
        self.kernels = {name: self.module.get_function(name) for name in self.kernel_names}
        cp.cuda.runtime.deviceSynchronize()
        self.compile_seconds = time.perf_counter() - start
        self.source_path = source_path
        self.peak_pool_used_bytes = cp.get_default_memory_pool().used_bytes()

    def _launch(self, name, count, args):
        blocks = (int(count) + self.block_size - 1) // self.block_size
        self.kernels[name]((blocks,), (self.block_size,), args)

    def _track_memory(self):
        used = cp.get_default_memory_pool().used_bytes()
        self.peak_pool_used_bytes = max(self.peak_pool_used_bytes, int(used))

    def upload(self, data, position, weight, phase, plan):
        data = np.ascontiguousarray(data) if not data.flags.c_contiguous else data
        if data.dtype != np.complex64 or data.ndim != 2:
            raise TypeError("range-compressed data must be C-contiguous complex64 [pulse, range]")
        position64 = np.ascontiguousarray(position, dtype=np.float64)
        weight64 = np.ascontiguousarray(weight, dtype=np.float64)
        phase64 = np.ascontiguousarray(phase, dtype=np.float64)
        if position64.shape != (data.shape[0], 3):
            raise ValueError("position shape does not match the pulse count")
        if weight64.shape != (data.shape[0],) or phase64.shape != (data.shape[0],):
            raise ValueError("weight and phase must have one value per pulse")
        host_start = time.perf_counter()
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record()
        device = DeviceInputs(
            data=cp.asarray(data),
            pos64=cp.asarray(position64),
            pos32=cp.asarray(np.ascontiguousarray(position64, dtype=np.float32)),
            weight64=cp.asarray(weight64),
            weight32=cp.asarray(np.ascontiguousarray(weight64, dtype=np.float32)),
            phase64=cp.asarray(phase64),
            phase32=cp.asarray(np.ascontiguousarray(phase64, dtype=np.float32)),
            origins64=cp.asarray(plan.origins),
            origins32=cp.asarray(np.ascontiguousarray(plan.origins, dtype=np.float32)),
            h2d_event_ms=0.0,
            h2d_wall_ms=0.0,
            bytes_uploaded=0,
        )
        end.record()
        end.synchronize()
        device.h2d_event_ms = float(cp.cuda.get_elapsed_time(start, end))
        device.h2d_wall_ms = 1000.0 * (time.perf_counter() - host_start)
        arrays = (
            device.data,
            device.pos64,
            device.pos32,
            device.weight64,
            device.weight32,
            device.phase64,
            device.phase32,
            device.origins64,
            device.origins32,
        )
        device.bytes_uploaded = int(sum(item.nbytes for item in arrays))
        self._track_memory()
        return device

    @staticmethod
    def _origin(device, node, mode):
        return (device.origins32 if mode == "fp32" else device.origins64)[node.index]

    def _leaf(self, node, device, mode, range_start, data_dr, k):
        count = node.nr * node.nu
        output = cp.empty(count, dtype=cp.complex64)
        n = node.b - node.a
        data = device.data[node.a : node.b]
        if mode == "fp32":
            args = (
                data,
                np.int32(device.data.shape[1]),
                device.pos32[node.a : node.b],
                np.int32(n),
                self._origin(device, node, mode),
                np.float32(node.r0),
                np.float32(node.dr),
                np.int32(node.nr),
                np.float32(node.u0),
                np.float32(node.du),
                np.int32(node.nu),
                np.float32(range_start),
                np.float32(data_dr),
                np.float32(k),
                device.weight32[node.a : node.b],
                device.phase32[node.a : node.b],
                np.int32(node.side),
                output,
            )
        elif mode == "mixed":
            args = (
                data,
                np.int32(device.data.shape[1]),
                device.pos64[node.a : node.b],
                np.int32(n),
                self._origin(device, node, mode),
                np.float64(node.r0),
                np.float64(node.dr),
                np.int32(node.nr),
                np.float64(node.u0),
                np.float64(node.du),
                np.int32(node.nu),
                np.float64(range_start),
                np.float64(data_dr),
                np.float64(k),
                device.weight32[node.a : node.b],
                device.phase64[node.a : node.b],
                np.int32(node.side),
                output,
            )
        else:
            args = (
                data,
                np.int32(device.data.shape[1]),
                device.pos64[node.a : node.b],
                np.int32(n),
                self._origin(device, node, mode),
                np.float64(node.r0),
                np.float64(node.dr),
                np.int32(node.nr),
                np.float64(node.u0),
                np.float64(node.du),
                np.int32(node.nu),
                np.float64(range_start),
                np.float64(data_dr),
                np.float64(k),
                device.weight64[node.a : node.b],
                device.phase64[node.a : node.b],
                np.int32(node.side),
                output,
            )
        self._launch(f"bp_polar_{mode}", count, args)
        self._track_memory()
        return output

    def _regrid(self, image, source, destination, device, mode, k):
        count = destination.nr * destination.nu
        output = cp.empty(count, dtype=cp.complex64)
        if mode == "fp32":
            scalar = np.float32
        else:
            scalar = np.float64
        args = (
            image,
            np.int32(source.nr),
            np.int32(source.nu),
            self._origin(device, source, mode),
            scalar(source.r0),
            scalar(source.dr),
            scalar(source.u0),
            scalar(source.du),
            self._origin(device, destination, mode),
            scalar(destination.r0),
            scalar(destination.dr),
            np.int32(destination.nr),
            scalar(destination.u0),
            scalar(destination.du),
            np.int32(destination.nu),
            scalar(k),
            np.int32(destination.side),
            output,
        )
        self._launch(f"regrid_{mode}", count, args)
        self._track_memory()
        return output

    def _execute_node(self, node, device, mode, range_start, data_dr, k):
        if node.is_leaf:
            return self._leaf(node, device, mode, range_start, data_dr, k)
        left_child = self._execute_node(node.left, device, mode, range_start, data_dr, k)
        left = self._regrid(left_child, node.left, node, device, mode, k)
        del left_child
        right_child = self._execute_node(node.right, device, mode, range_start, data_dr, k)
        right = self._regrid(right_child, node.right, node, device, mode, k)
        del right_child
        count = node.nr * node.nu
        self._launch("add_float2_inplace", count, (left, right, np.int32(count)))
        del right
        self._track_memory()
        return left

    def run_ffbp(self, plan, device, mode, range_start, data_dr, k):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        wall_start = time.perf_counter()
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record()
        root = self._execute_node(plan.root, device, mode, range_start, data_dr, k)
        output = cp.empty(plan.nr * plan.nu, dtype=cp.complex64)
        self._launch(
            "crop_float2",
            plan.nr * plan.nu,
            (
                root,
                np.int32(plan.root.nu),
                np.int32(plan.crop_r[0]),
                np.int32(plan.crop_u[0]),
                np.int32(plan.nr),
                np.int32(plan.nu),
                output,
            ),
        )
        del root
        end.record()
        end.synchronize()
        event_ms = float(cp.cuda.get_elapsed_time(start, end))
        wall_ms = 1000.0 * (time.perf_counter() - wall_start)
        self._track_memory()
        return output.reshape(plan.shape), event_ms, wall_ms

    def run_direct(
        self,
        plan,
        device,
        mode,
        range_start,
        data_dr,
        k,
        chunk_pulses=64,
    ):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if chunk_pulses <= 0:
            raise ValueError("chunk_pulses must be positive")
        count = plan.nr * plan.nu
        if mode == "fp64":
            accumulator = cp.zeros(count, dtype=cp.complex128)
            output = cp.empty(count, dtype=cp.complex64)
        else:
            accumulator = cp.zeros(count, dtype=cp.complex64)
            output = accumulator
        node = plan.root
        pulses = int(device.data.shape[0])
        if mode == "fp32":
            position = device.pos32
            origin = self._origin(device, node, mode)
            scalar = np.float32
            weight = device.weight32
            phase = device.phase32
        elif mode == "mixed":
            position = device.pos64
            origin = self._origin(device, node, mode)
            scalar = np.float64
            weight = device.weight32
            phase = device.phase64
        else:
            position = device.pos64
            origin = self._origin(device, node, mode)
            scalar = np.float64
            weight = device.weight64
            phase = device.phase64
        fixed_args = (
            device.data,
            np.int32(device.data.shape[1]),
            position,
            np.int32(pulses),
            origin,
            scalar(plan.r0),
            scalar(plan.dr),
            np.int32(plan.nr),
            scalar(plan.u0),
            scalar(plan.du),
            np.int32(plan.nu),
            scalar(range_start),
            scalar(data_dr),
            scalar(k),
            weight,
            phase,
            np.int32(plan.side),
        )
        wall_start = time.perf_counter()
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record()
        for begin in range(0, pulses, int(chunk_pulses)):
            finish = min(pulses, begin + int(chunk_pulses))
            self._launch(
                f"direct_chunk_{mode}",
                count,
                (*fixed_args, np.int32(begin), np.int32(finish), accumulator),
            )
        if mode == "fp64":
            self._launch(
                "cast_double2_to_float2",
                count,
                (accumulator, output, np.int32(count)),
            )
        end.record()
        end.synchronize()
        event_ms = float(cp.cuda.get_elapsed_time(start, end))
        wall_ms = 1000.0 * (time.perf_counter() - wall_start)
        if mode == "fp64":
            del accumulator
        self._track_memory()
        return output.reshape(plan.shape), event_ms, wall_ms

    @staticmethod
    def download(array):
        start = time.perf_counter()
        result = cp.asnumpy(array)
        return result, 1000.0 * (time.perf_counter() - start)


def cuda_environment():
    props = cp.cuda.runtime.getDeviceProperties(0)
    name = props["name"]
    if isinstance(name, bytes):
        name = name.decode(errors="replace")
    free, total = cp.cuda.runtime.memGetInfo()
    try:
        nvrtc_version = list(cp.cuda.nvrtc.getVersion())
    except Exception:
        nvrtc_version = None
    return {
        "gpu": name,
        "compute_capability": f"{props['major']}.{props['minor']}",
        "gpu_memory_bytes": int(total),
        "gpu_free_bytes_at_query": int(free),
        "driver_version": int(cp.cuda.runtime.driverGetVersion()),
        "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
        "nvrtc_version": nvrtc_version,
        "cupy_version": cp.__version__,
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cuda_root": str(CUDA_ROOT),
        "cuda_dll_dirs": [str(x) for x in CUDA_DLL_DIRS],
        "kernel_backend": "CuPy RawModule / NVIDIA NVRTC",
        "fast_math": False,
    }
