"""CPU-only structural audit of four-point FFBP regrid support.

The function accepts a gpu_backend.FFBPPlan but does not import CUDA/CuPy.
It propagates every potentially required 4x4 interpolation sample from the
final root crop to the leaves.  It is deliberately conservative when an
interpolation weight is exactly zero.  It audits FP64 geometry; floating
point rounding near integer boundaries is not a GPU numerical proof.
"""
from __future__ import annotations

import copy
import numpy as np


def extend_plan_radial_guard(plan, extra_cells):
    """Return a deep copy with extra radial cells on both sides of every node.

    All original node sample coordinates and the final output grid are kept.
    The root crop moves by ``extra_cells``.  No device arrays are created.
    ``result.guard_extension`` records this operation and cumulative padding.
    """
    if isinstance(extra_cells, bool) or int(extra_cells) != extra_cells or extra_cells < 0:
        raise ValueError("extra_cells must be a nonnegative integer")
    extra_cells = int(extra_cells)
    result = copy.deepcopy(plan)
    counts = {"leaves": 0, "merges": 0, "leaf_pixel_pulse_products": 0,
              "merged_pixels": 0, "nodes": 0}

    def extend(node):
        node.r0 -= extra_cells * node.dr
        node.nr += 2 * extra_cells
        counts["nodes"] += 1
        if node.is_leaf:
            counts["leaves"] += 1
            counts["leaf_pixel_pulse_products"] += node.nr * node.nu * (node.b - node.a)
        else:
            counts["merges"] += 1
            counts["merged_pixels"] += 2 * node.nr * node.nu
            extend(node.left)
            extend(node.right)

    extend(result.root)
    result.crop_r = tuple(int(v + extra_cells) for v in plan.crop_r)
    counts["shape"] = [int(result.nr), int(result.nu)]
    counts["direct_pixel_pulse_products"] = int(result.nr * result.nu * (result.root.b - result.root.a))
    result.counts = dict(plan.counts) | counts
    previous = getattr(plan, "guard_extension", {}).get("cumulative_extra_cells_per_side", 0)
    result.guard_extension = {
        "extra_cells_per_side": extra_cells,
        "cumulative_extra_cells_per_side": int(previous + extra_cells),
        "changed_axes": "r only; every node padded on both sides",
        "output_grid_unchanged": True,
        "root_crop_r_before": list(plan.crop_r),
        "root_crop_r_after": list(result.crop_r),
        "root_shape_before": [int(plan.root.nr), int(plan.root.nu)],
        "root_shape_after": [int(result.root.nr), int(result.root.nu)],
    }
    for name in ("r0", "dr", "nr", "u0", "du", "nu", "side", "crop_u"):
        assert getattr(result, name) == getattr(plan, name), name
    assert np.array_equal(result.origin, plan.origin)
    return result


def audit_support(plan, batch_size=65536):
    records = []
    root_required = np.zeros((plan.root.nr, plan.root.nu), dtype=bool)
    root_required[slice(*plan.crop_r), slice(*plan.crop_u)] = True

    def propagate(destination, source, required):
        indexes = np.flatnonzero(required.ravel())
        child_required = np.zeros(source.nr * source.nu, dtype=bool)
        invalid_count = 0
        minimum_margin = np.inf
        invalid_example = None
        for begin in range(0, len(indexes), batch_size):
            p = indexes[begin:begin + batch_size]
            ir, iu = np.divmod(p, destination.nu)
            r = destination.r0 + ir * destination.dr
            u = destination.u0 + iu * destination.du
            gr = np.sqrt(np.maximum(0., r * r - destination.origin[2] ** 2))
            x = destination.origin[0] + destination.side * gr * np.sqrt(np.maximum(0., 1. - u * u))
            y = destination.origin[1] + gr * u
            dx = x - source.origin[0]
            dy = y - source.origin[1]
            g = np.sqrt(dx * dx + dy * dy)
            rr = np.sqrt(g * g + source.origin[2] ** 2)
            with np.errstate(divide="ignore", invalid="ignore"):
                qr = (rr - source.r0) / source.dr
                qu = (dy / g - source.u0) / source.du
            finite = np.isfinite(qr) & np.isfinite(qu) & (g != 0.)
            sr = np.floor(np.where(finite, qr, -1.)).astype(np.int64)
            su = np.floor(np.where(finite, qu, -1.)).astype(np.int64)
            margin = np.minimum.reduce((sr - 1, su - 1, source.nr - 3 - sr, source.nu - 3 - su))
            valid = finite & (margin >= 0)
            invalid_count += int(np.count_nonzero(~valid))
            if len(margin):
                minimum_margin = min(minimum_margin, int(margin.min()))
            if invalid_example is None and np.any(~valid):
                first = int(np.flatnonzero(~valid)[0])
                invalid_example = {"destination_flat_index": int(p[first]), "qr": float(qr[first]), "qu": float(qu[first])}
            sr, su = sr[valid], su[valid]
            for a in (-1, 0, 1, 2):
                for b in (-1, 0, 1, 2):
                    child_required[(sr + a) * source.nu + su + b] = True
        records.append({
            "destination_index": int(destination.index),
            "source_index": int(source.index),
            "required_destination_sites": int(len(indexes)),
            "invalid_required_stencils": invalid_count,
            "minimum_integer_support_margin_cells": None if not np.isfinite(minimum_margin) else int(minimum_margin),
            "required_source_sites": int(np.count_nonzero(child_required)),
            "invalid_example": invalid_example,
        })
        return child_required.reshape(source.nr, source.nu)

    def walk(node, required):
        if node.is_leaf:
            return
        for child in (node.left, node.right):
            child_required = propagate(node, child, required)
            walk(child, child_required)

    walk(plan.root, root_required)
    failures = [row for row in records if row["invalid_required_stencils"]]
    return {
        "method": "CPU FP64 geometry; conservative 4x4 support propagation from final crop",
        "passed": not failures,
        "checked_edges": len(records),
        "invalid_required_stencils": int(sum(row["invalid_required_stencils"] for row in records)),
        "edges_with_invalid_required_stencils": len(failures),
        "records": records,
        "limitations": [
            "Audits regrid support only; not leaf raw range support or interpolation accuracy",
            "No exact equivalence assertion to CUDA FMA or FP32 index rounding",
            "Conservative inclusion of zero-weight stencil samples can overstate downstream requirements",
            "Invalid stencil counts are per tree edge, not unique final output pixels",
        ],
    }
