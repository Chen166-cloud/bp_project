# CUDA BP baseline

`kernels.cu` contains the CUDA C++ kernels used by the direct BP and FFBP
precision baseline. `gpu_backend.py` compiles the source with NVIDIA NVRTC via
CuPy; no Python expression is used inside the image-formation kernels.

The three fixed arithmetic policies are:

| Mode | Geometry, sample coordinate, phase | Interpolation, accumulation | Input and node storage |
|---|---|---|---|
| `fp64` | FP64 | FP64 | complex64 |
| `fp32` | FP32 | FP32 | complex64 |
| `mixed` | FP64 | FP32 | complex64 |

All modes use the same FP64 host-generated node tree and final grid. Fast math
is disabled. Direct BP is split into 64-pulse launches for the Windows display
driver watchdog, while preserving one accumulator and the original pulse order.

On this workstation, MATLAB R2024a supplies CUDA 12.2/NVRTC. The backend finds
that installation automatically. For another machine, set
`FAST_BP_CUDA_ROOT` and, on Windows when needed, `FAST_BP_CUDA_DLL_DIR`.

```powershell
D:\develop\Anaconda\python.exe -m pip install -r fast_bp\gpu\requirements-gpu.txt
D:\develop\Anaconda\python.exe fast_bp\gpu_baseline.py --repeats 5 --warmups 1 --cpu-repeats 3
```

Outputs are written to `fast_bp/results/gpu_baseline/`. The kernel source is
portable CUDA C++, but a standalone `nvcc` executable build on Windows still
requires MSVC Build Tools and a Windows SDK; neither was installed for this
experiment.
