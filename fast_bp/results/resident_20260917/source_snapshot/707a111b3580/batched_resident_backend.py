"""Level-batched, preallocated FFBP with the baseline floating-point policy.

Every CUDA block belongs to exactly one tree node. All leaves run in one grid;
one grid per subtree height merges ready children, followed by the usual crop.
Node images are independent allocations, addressed through a device pointer
table. Geometry, sample interpolation, and child float2 rounding are unchanged.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    from .resident_backend import ResidentProgram, MODES, cp
except ImportError:
    from resident_backend import ResidentProgram, MODES, cp


_BATCHED_MODULES = {}


class BatchedResidentProgram(ResidentProgram):
    """ResidentProgram-compatible, fused FFBP batched across each tree level."""

    def __init__(
        self, backend, plan, device, mode, range_start, data_dr, k,
        chunk_pulses=64,
    ):
        self._table_buffers = []
        self._batch_stages = []
        super().__init__(
            backend, plan, device, mode, range_start, data_dr, k,
            algorithm="ffbp", variant="fused", chunk_pulses=chunk_pulses,
        )
        self.variant = "batched_fused"
        table_bytes = sum(int(a.nbytes) for a in self._table_buffers)
        self.metadata.update({
            "variant": self.variant,
            "batch_stages": self._batch_stages,
            "extra_table_bytes": table_bytes,
            "extra_table_count": len(self._table_buffers),
            "total_owned_device_bytes": self.metadata["total_owned_device_bytes"] + table_bytes,
            "batch_mapping": "one block per node pixel tile; nodes grouped by subtree height",
            "fusion": "original leaf arithmetic; each parent child regrid rounded to float2 then FP32 add",
        })

    def _table(self, host):
        array = cp.asarray(np.ascontiguousarray(host))
        self._table_buffers.append(array)
        return array

    def _prepare_fused_module(self):
        directory = Path(__file__).resolve().parent / "gpu"
        baseline = self.backend.source_path.read_text(encoding="utf-8")
        fused = (directory / "resident_kernels.cu").read_text(encoding="utf-8")
        batched = (directory / "batched_resident_kernels.cu").read_text(encoding="utf-8")
        key = (int(cp.cuda.runtime.getDevice()), baseline, fused, batched)
        if key not in _BATCHED_MODULES:
            names = tuple(
                f"batched_{family}_{mode}"
                for family in ("bp_polar", "merge_regrid") for mode in MODES
            )
            module = cp.RawModule(
                code=baseline + "\n" + fused + "\n" + batched,
                options=("-std=c++11",), name_expressions=names,
            )
            kernels = {name: module.get_function(name) for name in names}
            _BATCHED_MODULES[key] = (module, kernels)
        self._batched_module, self._batched_kernels = _BATCHED_MODULES[key]

    def _build_node(self, root):
        nodes = {}
        by_height = {}

        def collect(node):
            nodes[node.index] = node
            if node.is_leaf:
                height = 0
            else:
                height = max(collect(node.left), collect(node.right)) + 1
            by_height.setdefault(height, []).append(node)
            return height

        collect(root)
        if sorted(nodes) != list(range(len(nodes))):
            raise ValueError("batched FFBP requires dense node indices starting from zero")
        indices = np.zeros((len(nodes), 7), dtype=np.int32)
        geometry = np.zeros((len(nodes), 4), dtype=self.scalar)
        pointers = np.zeros(len(nodes), dtype=np.uint64)
        images = {}
        for index, node in sorted(nodes.items()):
            image = self._new_buffer(node.nr * node.nu)
            images[index] = image
            pointers[index] = image.data.ptr
            indices[index] = (
                node.a, node.b, node.nr, node.nu, node.side,
                -1 if node.is_leaf else node.left.index,
                -1 if node.is_leaf else node.right.index,
            )
            geometry[index] = (node.r0, node.dr, node.u0, node.du)
        self._node_table = self._table(indices)
        self._geometry_table = self._table(geometry)
        self._pointer_table = self._table(pointers)
        position, weight, phase = self._precision_arrays()
        d, s = self.device, self.scalar
        block_size = self.backend.block_size

        for height, stage_nodes in sorted(by_height.items()):
            stage_nodes = sorted(stage_nodes, key=lambda item: item.index)
            mappings = [
                (node.index, start)
                for node in stage_nodes
                for start in range(0, node.nr * node.nu, block_size)
            ]
            block_table = self._table(np.asarray(mappings, dtype=np.int32))
            block_count = len(mappings)
            if height == 0:
                name = f"batched_bp_polar_{self.mode}"
                args = (
                    d.data, np.int32(d.data.shape[1]), position, weight, phase,
                    self._origins, self._node_table, self._geometry_table,
                    self._pointer_table, block_table,
                    s(self.range_start), s(self.data_dr), s(self.k),
                )
            else:
                name = f"batched_merge_regrid_{self.mode}"
                args = (
                    self._origins, self._node_table, self._geometry_table,
                    self._pointer_table, block_table, s(self.k),
                )
            # _enqueue's grid formula is exact because count is a multiple of
            # block_size. The final block of EACH node masks unused threads.
            self._enqueue(
                name, block_count * block_size, args,
                kernel=self._batched_kernels[name],
            )
            pixels = sum(int(node.nr * node.nu) for node in stage_nodes)
            self.operations[-1].update({
                "height": height, "node_count": len(stage_nodes),
                "valid_pixels": pixels,
            })
            self._batch_stages.append({
                "height": height, "kernel": name,
                "node_count": len(stage_nodes), "block_count": block_count,
                "valid_pixels": pixels,
                "padded_threads": block_count * block_size,
                "block_table_bytes": int(block_table.nbytes),
            })
        return images[root.index]

    def close(self):
        super().close()
        self._table_buffers.clear()
        self._node_table = None
        self._geometry_table = None
        self._pointer_table = None
