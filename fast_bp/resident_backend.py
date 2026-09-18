"""Preallocated, asynchronous BP execution for resident-compute experiments.

All input arrays stay owned by the caller. Construction uploads only the small
node-origin table and allocates image workspace. ``execute`` and ``replay`` do
not allocate images, synchronize, record events, or transfer data to the host.
The caller must time those methods with CUDA events on the current stream.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

try:
    from .gpu_backend import MODES, cp
except ImportError:
    from gpu_backend import MODES, cp


_FUSED_MODULES = {}


class ResidentProgram:
    """A reusable direct-BP or FFBP schedule with fixed inputs and grid.

    ``launch_count`` counts CUDA kernels. A direct program additionally performs
    one async memset per execution, included in ``device_operation_count``.
    Outputs are overwritten by the next execution; copy them before reuse if
    an independent result is needed. Programs are not safe for concurrent
    execution because their workspace is shared between invocations.
    """

    def __init__(
        self, backend, plan, device, mode, range_start, data_dr, k,
        algorithm="ffbp", variant="original", chunk_pulses=64,
    ):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if algorithm not in ("direct", "ffbp"):
            raise ValueError("algorithm must be direct or ffbp")
        if variant not in ("original", "fused"):
            raise ValueError("variant must be original or fused")
        if algorithm == "direct" and variant != "original":
            raise ValueError("fused variant is only defined for FFBP")
        if chunk_pulses <= 0:
            raise ValueError("chunk_pulses must be positive")
        self.backend = backend
        self.plan = plan
        self.device = device
        self.mode = mode
        self.algorithm = algorithm
        self.variant = variant
        self.range_start = range_start
        self.data_dr = data_dr
        self.k = k
        self.chunk_pulses = int(chunk_pulses)
        self.scalar = np.float32 if mode == "fp32" else np.float64
        self._schedule = []
        self.operations = []
        self._buffers = []
        self._graph = None
        self._graph_stream = None
        self._closed = False
        self._accumulator = None
        # A plan sweep can share the large device.data allocation. A per-plan
        # origin table prevents accidentally reusing origins of an older plan.
        self._origins = cp.asarray(np.ascontiguousarray(plan.origins, dtype=self.scalar))
        self._origin_views = {i: self._origins[i] for i in range(len(plan.origins))}
        if algorithm == "direct":
            self._build_direct()
        else:
            if variant == "fused":
                self._prepare_fused_module()
            root_image = self._build_node(plan.root)
            flat = self._new_buffer(plan.nr * plan.nu)
            self._enqueue("crop_float2", plan.nr * plan.nu, (
                root_image, np.int32(plan.root.nu), np.int32(plan.crop_r[0]),
                np.int32(plan.crop_u[0]), np.int32(plan.nr), np.int32(plan.nu), flat,
            ))
            self.output = flat.reshape(plan.shape)
        self.launch_count = len(self._schedule)
        counts = Counter(entry[0] for entry in self._schedule)
        workspace_bytes = sum(int(a.nbytes) for a in self._buffers)
        self.metadata = {
            "algorithm": algorithm,
            "variant": variant,
            "mode": mode,
            "shape": [int(x) for x in plan.shape],
            "block_size": int(backend.block_size),
            "chunk_pulses": self.chunk_pulses if algorithm == "direct" else None,
            "kernel_launch_count": self.launch_count,
            "kernel_counts": dict(counts),
            "memset_count": int(algorithm == "direct"),
            "device_operation_count": self.launch_count + int(algorithm == "direct"),
            "workspace_bytes": workspace_bytes,
            "workspace_buffer_count": len(self._buffers),
            "origin_table_bytes": int(self._origins.nbytes),
            "total_owned_device_bytes": workspace_bytes + int(self._origins.nbytes),
            "workspace_policy": "all schedule buffers preallocated; no liveness reuse",
            "input_data_shared": True,
            "output_dtype": "complex64",
            "zeroing_in_execute": algorithm == "direct",
            "graph_supported": None,
            "graph_error": None,
            "arithmetic": "baseline kernels; no fast_math",
            "fusion": "two original child regrids rounded to float2 then FP32 add" if variant == "fused" else None,
        }

    def _new_buffer(self, count, dtype=None):
        array = cp.empty(int(count), dtype=cp.complex64 if dtype is None else dtype)
        self._buffers.append(array)
        return array

    def _origin(self, node):
        return self._origin_views[node.index]

    def _enqueue(self, name, count, args, kernel=None):
        count = int(count)
        if count <= 0:
            raise ValueError("empty image grids are unsupported")
        blocks = (count + self.backend.block_size - 1) // self.backend.block_size
        if kernel is None:
            kernel = self.backend.kernels[name]
        self._schedule.append((name, kernel, (blocks,), (self.backend.block_size,), args))
        self.operations.append({
            "name": name, "count": count, "kernel": kernel,
            "grid": (blocks,), "block": (self.backend.block_size,), "args": args,
        })

    def _precision_arrays(self):
        d = self.device
        if self.mode == "fp32":
            return d.pos32, d.weight32, d.phase32
        if self.mode == "mixed":
            return d.pos64, d.weight32, d.phase64
        return d.pos64, d.weight64, d.phase64

    def _build_direct(self):
        p, d, s = self.plan, self.device, self.scalar
        count = p.nr * p.nu
        self._accumulator = self._new_buffer(
            count, cp.complex128 if self.mode == "fp64" else cp.complex64,
        )
        flat = self._new_buffer(count) if self.mode == "fp64" else self._accumulator
        position, weight, phase = self._precision_arrays()
        pulses = int(d.data.shape[0])
        fixed = (
            d.data, np.int32(d.data.shape[1]), position, np.int32(pulses),
            self._origin(p.root), s(p.r0), s(p.dr), np.int32(p.nr),
            s(p.u0), s(p.du), np.int32(p.nu), s(self.range_start),
            s(self.data_dr), s(self.k), weight, phase, np.int32(p.side),
        )
        for begin in range(0, pulses, self.chunk_pulses):
            end = min(pulses, begin + self.chunk_pulses)
            self._enqueue(f"direct_chunk_{self.mode}", count, (
                *fixed, np.int32(begin), np.int32(end), self._accumulator,
            ))
        if self.mode == "fp64":
            self._enqueue("cast_double2_to_float2", count, (
                self._accumulator, flat, np.int32(count),
            ))
        self.output = flat.reshape(p.shape)

    def _leaf(self, node):
        d, s = self.device, self.scalar
        position, weight, phase = self._precision_arrays()
        sl = slice(node.a, node.b)
        output = self._new_buffer(node.nr * node.nu)
        self._enqueue(f"bp_polar_{self.mode}", node.nr * node.nu, (
            d.data[sl], np.int32(d.data.shape[1]), position[sl],
            np.int32(node.b - node.a), self._origin(node), s(node.r0),
            s(node.dr), np.int32(node.nr), s(node.u0), s(node.du),
            np.int32(node.nu), s(self.range_start), s(self.data_dr),
            s(self.k), weight[sl], phase[sl], np.int32(node.side), output,
        ))
        return output

    def _source_args(self, image, source):
        s = self.scalar
        return (image, np.int32(source.nr), np.int32(source.nu),
                self._origin(source), s(source.r0), s(source.dr),
                s(source.u0), s(source.du))

    def _destination_args(self, destination):
        s = self.scalar
        return (self._origin(destination), s(destination.r0), s(destination.dr),
                np.int32(destination.nr), s(destination.u0), s(destination.du),
                np.int32(destination.nu), s(self.k), np.int32(destination.side))

    def _regrid(self, image, source, destination):
        output = self._new_buffer(destination.nr * destination.nu)
        self._enqueue(f"regrid_{self.mode}", destination.nr * destination.nu, (
            *self._source_args(image, source),
            *self._destination_args(destination), output,
        ))
        return output

    def _build_node(self, node):
        if node.is_leaf:
            return self._leaf(node)
        left_child = self._build_node(node.left)
        if self.variant == "original":
            # Keep the original depth-first launch order exactly.
            left = self._regrid(left_child, node.left, node)
            right_child = self._build_node(node.right)
            right = self._regrid(right_child, node.right, node)
            self._enqueue("add_float2_inplace", node.nr * node.nu, (
                left, right, np.int32(node.nr * node.nu),
            ))
            return left
        right_child = self._build_node(node.right)
        output = self._new_buffer(node.nr * node.nu)
        name = f"merge_regrid_{self.mode}"
        self._enqueue(name, node.nr * node.nu, (
            *self._source_args(left_child, node.left),
            *self._source_args(right_child, node.right),
            *self._destination_args(node), output,
        ), kernel=self._fused_kernels[name])
        return output

    def _prepare_fused_module(self):
        source_path = Path(__file__).resolve().parent / "gpu" / "resident_kernels.cu"
        baseline_source = self.backend.source_path.read_text(encoding="utf-8")
        extension_source = source_path.read_text(encoding="utf-8")
        cache_key = (int(cp.cuda.runtime.getDevice()), baseline_source, extension_source)
        if cache_key not in _FUSED_MODULES:
            names = tuple(f"merge_regrid_{mode}" for mode in MODES)
            module = cp.RawModule(
                code=baseline_source + "\n" + extension_source,
                options=("-std=c++11",), name_expressions=names,
            )
            kernels = {name: module.get_function(name) for name in names}
            _FUSED_MODULES[cache_key] = (module, kernels)
        self._fused_module, self._fused_kernels = _FUSED_MODULES[cache_key]

    def execute(self):
        """Enqueue one complete image computation on the current CUDA stream."""
        if self._closed:
            raise RuntimeError("program is closed")
        if self._accumulator is not None:
            cp.cuda.runtime.memsetAsync(
                self._accumulator.data.ptr, 0, self._accumulator.nbytes,
                cp.cuda.get_current_stream().ptr,
            )
        for _, kernel, grid, block, args in self._schedule:
            kernel(grid, block, args)
        return self.output

    def capture_graph(self, warmup=True):
        """Prepare a graph outside timed regions; record unsupported cases.

        Warmup on an independent nonblocking stream resolves lazy CUDA state.
        This setup method may synchronize; ``execute`` and ``replay`` never do.
        """
        if self._closed:
            raise RuntimeError("program is closed")
        if self._graph is not None:
            return True
        stream = cp.cuda.Stream(non_blocking=True)
        begun = False
        try:
            if not hasattr(stream, "begin_capture") or not hasattr(stream, "end_capture"):
                raise RuntimeError("this CuPy build has no CUDA stream capture API")
            # Ensure caller uploads and any previous execution have completed
            # before the independent capture stream touches their allocations.
            cp.cuda.get_current_stream().synchronize()
            with stream:
                if warmup:
                    self.execute()
                    stream.synchronize()
                stream.begin_capture()
                begun = True
                self.execute()
                graph = stream.end_capture()
                begun = False
            self._graph = graph
            self._graph_stream = stream
            self.metadata["graph_supported"] = True
            self.metadata["graph_error"] = None
            return True
        except Exception as exc:
            if begun:
                try:
                    with stream:
                        stream.end_capture()
                except Exception:
                    pass
            self._graph = None
            self.metadata["graph_supported"] = False
            self.metadata["graph_error"] = f"{type(exc).__name__}: {exc}"
            return False

    def replay(self):
        """Enqueue the captured graph on the current stream, without a fence."""
        if self._closed:
            raise RuntimeError("program is closed")
        if self._graph is None:
            raise RuntimeError("capture_graph() must succeed before replay()")
        self._graph.launch(stream=cp.cuda.get_current_stream())
        return self.output

    def close(self):
        """Drop owned references after the caller has synchronized its work.

        CuPy can retain released allocations in its memory pool. This method
        does not clear the global pool or synchronize unrelated GPU work.
        """
        self._graph = None
        self._graph_stream = None
        self._schedule.clear()
        self.operations.clear()
        self._buffers.clear()
        self._origin_views.clear()
        self._origins = None
        self._accumulator = None
        self.output = None
        self.device = None
        self._closed = True
