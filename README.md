# SAR BP / FFBP 成像研究项目

本仓库汇集了合成孔径雷达（SAR）反投影（BP）与快速分解反投影（FFBP）成像的 MATLAB、Python 和 CUDA 实现，以及实测数据的质量评价、GPU 基线实验和组会材料。

## 主要内容

- 根目录：传统 BP、距离-多普勒（RD）、Chirp Scaling（CS）等 MATLAB 脚本与相关成像结果。
- `fast_bp/`：基于实测数据的 FFBP 主流程，包括距离压缩、导航诊断、自聚焦、CPU 并行实现和 CUDA GPU 直接 BP/FFBP 基线。
- `research_gpu_bp/`：GPU 驻留计算实验、性能/精度分析脚本、图表和研究记录。
- `sar_quality/`：成像质量与分辨率评价脚本、结果和文献材料。

## 环境

Python 主流程依赖 Python 3、NumPy、SciPy、Numba、Matplotlib 和 Pillow：

```powershell
pip install -r fast_bp/requirements.txt
```

GPU 基线还需要兼容 CUDA 12 的 CuPy：

```powershell
pip install -r fast_bp/gpu/requirements-gpu.txt
```

MATLAB 脚本可在 MATLAB 环境中直接运行；`run_fast_bp.m` 可调用 Python FFBP 主流程。

## 快速开始

在项目根目录执行：

```powershell
python fast_bp/diagnose.py
python fast_bp/validate.py
python fast_bp/run.py --name ffbp_navigation
python fast_bp/autofocus.py
python fast_bp/run.py --position-file fast_bp/results/autofocus_position.npy --name ffbp_final
python fast_bp/render_results.py
```

若具备 CUDA 环境，可运行：

```powershell
python fast_bp/gpu_baseline.py --repeats 5 --warmups 1 --cpu-repeats 3 --chunk-pulses 64
```

详细的数据约定、处理范围、结果解释和验证边界见 [fast_bp/README.md](fast_bp/README.md)。

## 数据说明

原始 SAR 数据及中间数值缓存体积很大（单个文件可超过 1 GB），未纳入 Git 仓库。它们通过 `.gitignore` 排除，包括 `.dat`、`.mat`、`.npy` 与 `.npz` 文件。复现实测处理时，请将所需原始数据放回项目根目录；例如主流程默认使用 `2018-08-14_18-54-08.dat`。

仓库保留源码、实验记录、轻量级图表和质量评价结果，便于代码审阅与复现实验流程。

## 注意事项

成像坐标为局部米制坐标，当前结果未进行 DEM、地面控制点或绝对辐射标定。实验中的自聚焦补偿主要用于改善内部相干性，不应直接视为独立测得的平台真实运动误差。
