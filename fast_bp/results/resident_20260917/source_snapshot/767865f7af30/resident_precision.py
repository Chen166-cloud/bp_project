"""Build the MatX-inspired phase-LUT direct-BP resident comparison.

This module does not import CuPy or initialize CUDA until ``make_lut_program``
is called. The constructor prepares the LUT and returns an uncaptured
ResidentProgram. Its execute/replay timings include zeroing and every BP
chunk, including each chunk's per-pixel FP64 reference-phase evaluation.

Example (for either sign of k and either plan.side):

    program = make_lut_program(backend, plan, device, range_start, data_dr, k)
    program.execute()
    # Synchronize and compare program.output with the same-grid direct BP.
    # Only after this comparison, optionally capture then time graph replay:
    program.capture_graph()
    program.replay()
    # Synchronize before program.close(), as for every ResidentProgram.

The LUT is an optimization inspired by MatX's integer-bin/fractional-bin
phase decomposition, not a reproduction or benchmark of the full MatX BP.
The existing four-point Lagrange interpolation remains unchanged.
"""
from __future__ import annotations

import copy
from pathlib import Path
import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .resident_backend import ResidentProgram


_LUT_KERNEL = "direct_chunk_mixed_phase_lut"
_LUT_FILL = "fill_range_phase_lut"
_MATX_SOURCE = (
    "https://github.com/NVIDIA/MatX/blob/"
    "a108745201415478e17fa72e9e328dc0514ae7fb/"
    "include/matx/kernels/sar_bp.cuh"
)


def _attach_lut_to_schedule(program, lut):
    """Patch both execution and inspection schedules before any graph capture.

    Pure Python: also usable with a mock program to inspect argument routing
    without compiling or executing CUDA. ResidentProgram.execute uses
    _schedule; operations is the matching public inspection representation.
    """
    if program._graph is not None:
        raise RuntimeError("append LUT arguments before graph capture")
    if len(program._schedule) != len(program.operations):
        raise RuntimeError("resident schedule and operation list differ")
    for entry, operation in zip(program._schedule, program.operations):
        name, kernel, grid, block, args = entry
        if name != "direct_chunk_mixed" or operation["name"] != name:
            raise RuntimeError("expected an exclusively mixed direct-BP schedule")
        if operation["kernel"] is not kernel or operation["args"] is not args:
            raise RuntimeError("resident schedule and operation arguments differ")
    for index, (entry, operation) in enumerate(zip(program._schedule, program.operations)):
        _, kernel, grid, block, args = entry
        lut_args = (*args, lut)
        program._schedule[index] = (_LUT_KERNEL, kernel, grid, block, lut_args)
        operation.update(name=_LUT_KERNEL, args=lut_args)


def make_lut_program(
    backend, plan, device, range_start, data_dr, k, chunk_pulses=256,
) -> ResidentProgram:
    """Prepare a reusable FP64-geometry, FP32-phase-LUT direct BP program.

    Inputs and quality reference must match ordinary direct BP. ``k`` retains
    its sign; no conjugation or side conversion is applied here. The returned
    output is complex64, with FP32 interpolation and accumulation.

    This constructor compiles kernels and synchronously prepares the LUT,
    outside steady-state resident execution. ``metadata['prepare_ms']`` is
    LUT allocation, generation, and completion wall time; compilation and
    CUDA-event generation-only duration are recorded separately. The LUT is
    valid only while (samples, range_start, data_dr, k) is unchanged. Report
    preparation plus execution when those values change per image.
    """
    if int(chunk_pulses) != chunk_pulses or chunk_pulses <= 0:
        raise ValueError("chunk_pulses must be a positive integer")
    if not all(np.isfinite(float(value)) for value in (range_start, data_dr, k)):
        raise ValueError("range_start, data_dr and k must be finite")
    if data_dr <= 0:
        raise ValueError("data_dr must be positive")
    samples = int(device.data.shape[1])
    if samples <= 0:
        raise ValueError("range data must have at least one sample")

    try:
        from .gpu_backend import cp
        from .resident_backend import ResidentProgram
    except ImportError:
        from gpu_backend import cp
        from resident_backend import ResidentProgram

    probe_path = Path(__file__).resolve().parent / "gpu" / "precision_probe.cu"
    source = (backend.source_path.read_text(encoding="utf-8") + "\n" +
              probe_path.read_text(encoding="utf-8"))
    compile_start = time.perf_counter()
    module = cp.RawModule(
        code=source, options=("-std=c++11",),
        name_expressions=(_LUT_KERNEL, _LUT_FILL),
    )
    direct_kernel = module.get_function(_LUT_KERNEL)
    fill_kernel = module.get_function(_LUT_FILL)
    compile_ms = (time.perf_counter() - compile_start) * 1000.0

    # A private backend mapping leaves the original baseline backend intact.
    proxy = copy.copy(backend)
    proxy.kernels = dict(backend.kernels)
    proxy.kernels["direct_chunk_mixed"] = direct_kernel
    proxy.module = module  # Keep the compiled module alive for graph replay.

    # Exclude preceding uploads/work from one-time LUT preparation measurement.
    cp.cuda.get_current_stream().synchronize()
    prepare_start = time.perf_counter()
    lut = cp.empty(samples, dtype=cp.complex64)
    start, stop = cp.cuda.Event(), cp.cuda.Event()
    start.record()
    block_size = int(backend.block_size)
    fill_kernel(((samples + block_size - 1) // block_size,), (block_size,), (
        np.int32(samples), np.float64(range_start), np.float64(data_dr),
        np.float64(k), lut,
    ))
    stop.record()
    stop.synchronize()
    prepare_ms = (time.perf_counter() - prepare_start) * 1000.0
    generation_ms = float(cp.cuda.get_elapsed_time(start, stop))

    program = ResidentProgram(
        proxy, plan, device, "mixed", range_start, data_dr, k,
        algorithm="direct", chunk_pulses=int(chunk_pulses),
    )
    _attach_lut_to_schedule(program, lut)
    # ResidentProgram.close() clears _buffers and both argument schedules.
    # This keeps the LUT alive throughout execution and graph lifetime while
    # allowing close() to release its ownership in the usual way.
    program._buffers.append(lut)
    lut_bytes = int(lut.nbytes)
    program.metadata.update({
        "variant": "matx_inspired_phase_lut",
        "kernel_counts": {_LUT_KERNEL: program.launch_count},
        "arithmetic": (
            "FP64 geometry and range coordinates; four-point FP32 Lagrange; "
            "FP64-generated complex64 LUT; ordinary sincosf fractional phase; "
            "FP32 accumulation; no fast_math"
        ),
        "inspiration": "MatX integer-bin/fractional-bin phase LUT; not full MatX reproduction",
        "inspiration_source": _MATX_SOURCE,
        "lut_bytes": lut_bytes,
        "lut_entries": samples,
        "lut_dtype": "complex64",
        "lut_generation_precision": "FP64 sincos, then complex64 storage",
        "lut_dependency_key": {
            "samples": samples, "range_start": float(range_start),
            "data_dr": float(data_dr), "k": float(k),
        },
        "prepare_ms": float(prepare_ms),
        "prepare_ms_scope": "LUT allocation, generation and synchronization; excludes compilation and program workspace",
        "lut_generation_device_ms": generation_ms,
        "module_compile_host_ms": float(compile_ms),
        "lut_preparation_in_execute": False,
        "per_pixel_reference_phase_in_execute": "FP64 sincos(-k*r), once per pixel per chunk",
        "large_pulse_phase_limit": "large phase[n] can still incur FP32 increment-angle quantization",
        "workspace_bytes": int(program.metadata["workspace_bytes"]) + lut_bytes,
        "workspace_buffer_count": len(program._buffers),
        "workspace_bytes_includes_lut": True,
        "total_owned_device_bytes": int(program.metadata["total_owned_device_bytes"]) + lut_bytes,
    })
    return program
