"""Small Nsight target for a single, already-validated resident BP program.

Construction, H2D upload, graph capture, and warmup happen before
cudaProfilerStart. No benchmark results are written: profiler instrumentation
changes execution and must not replace the independent CUDA-event benchmark.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.signal.windows import hann

from resident_experiment import (
    ROOT, EXTENTS, RSTART, DATA_DR, K, CudaBaseline, ResidentProgram,
    cp, make_plan,
)
from batched_resident_backend import BatchedResidentProgram


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("direct", "ffbp"), default="ffbp")
    parser.add_argument("--variant", choices=("original", "fused", "batched"), default="batched")
    parser.add_argument("--mode", choices=("fp32", "mixed", "fp64"), default="fp32")
    parser.add_argument("--scale", choices=tuple(EXTENTS), default="small")
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--leaf", type=int, default=64)
    parser.add_argument("--internal-os", type=float, default=3.0)
    parser.add_argument("--chunk-pulses", type=int, default=256)
    args = parser.parse_args()
    if args.iterations < 1 or args.warmups < 0 or args.leaf < 1 or args.chunk_pulses < 1:
        parser.error("iterations, leaf and chunk-pulses must be positive; warmups must be nonnegative")
    if args.internal_os <= 0:
        parser.error("internal-os must be positive")
    if args.algorithm == "direct" and args.variant != "original":
        parser.error("direct BP requires --variant original")
    return args


def main():
    args = parse_args()
    backend = CudaBaseline()
    data = np.load(ROOT / "fast_bp/results/range_compressed.npy", mmap_mode="r")
    position = np.load(ROOT / "fast_bp/results/autofocus_position.npy")
    extent = EXTENTS[args.scale]
    base = make_plan(position, extent)
    # Match the formal sweep: changing internal sampling must not change the
    # requested output pixels or give a candidate a different image workload.
    plan = make_plan(position, extent, args.leaf, args.internal_os, base=base)
    device = backend.upload(data, position, hann(len(position)), np.zeros(len(position)), plan)
    if args.variant == "batched":
        program = BatchedResidentProgram(backend, plan, device, args.mode, RSTART, DATA_DR, K)
    else:
        program = ResidentProgram(
            backend, plan, device, args.mode, RSTART, DATA_DR, K,
            algorithm=args.algorithm, variant=args.variant,
            chunk_pulses=args.chunk_pulses,
        )
    stream = cp.cuda.get_current_stream()
    try:
        if args.graph:
            if not program.capture_graph():
                raise RuntimeError(f"Graph capture failed: {program.metadata['graph_error']}")
            execute = program.replay
        else:
            execute = program.execute
        for _ in range(args.warmups):
            execute()
            stream.synchronize()
        stream.synchronize()
        print(json.dumps({
            "purpose": "profiling only; use separate uninstrumented event results for speedups",
            "arguments": vars(args), "program": program.metadata,
            "capture_range": "cudaProfilerStart/Stop; NVTX resident_profile if available",
        }, ensure_ascii=False), flush=True)

        # CuPy 13.6's cuda.profiler.start/stop alias these runtime functions.
        # An active CUDA context already exists from construction and warmup.
        cp.cuda.runtime.profilerStart()
        nvtx_pushed = False
        try:
            try:
                cp.cuda.nvtx.RangePush("resident_profile")
                nvtx_pushed = True
            except (AttributeError, RuntimeError) as exc:
                print(f"NVTX unavailable; CUDA profiler API capture remains active: {exc}", flush=True)
            for _ in range(args.iterations):
                execute()
                # Preserve one-image completion boundaries used by the event
                # experiment; the wait is visible in a Systems trace.
                stream.synchronize()
        finally:
            if nvtx_pushed:
                cp.cuda.nvtx.RangePop()
            cp.cuda.runtime.profilerStop()
        print("Resident profile range completed.", flush=True)
    finally:
        stream.synchronize()
        program.close()


if __name__ == "__main__":
    main()
