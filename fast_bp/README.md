# 2018-08-14 实测数据快速 BP 成像

本目录提供从原始 `2018-08-14_18-54-08.dat` 到复数 SAR 图像的可复现处理程序。采用真正的快速分解反投影（FFBP）：短子孔径局部极坐标成像、逐级提高角分辨率、插值并相干合并。完整场景处理使用 NumPy/SciPy 和 Numba 的 16 线程 CPU 内核；固定 ROI 另提供 CUDA C++ 的直接 BP/FFBP 多精度基线。没有直接调用原目录中简化的 `raw_bp.m` 或 `dot.m`。

## 先看结果

- `results/ffbp_final_preview.png`：推荐首先查看，显示窗为相对峰值 −65 至 −10 dB，便于看见弱散射地物；该显示调整不改变复数成像数据。
- `results/ffbp_final_ground_overview.png`：自聚焦后的二维地面图，带米制坐标。
- `results/ffbp_final_ground.mat`：复数地面图 `image_complex`、距离轴、相位中心及主要参数，可直接在 MATLAB 中加载。
- `results/ffbp_final_polar.png`：完整极坐标灰度图。
- `results/ffbp_final_complex_demodulated.npy`、`ffbp_final_axes.npz`：完整复数极坐标结果和坐标；复数值保留了子孔径合并所需的径向去载频约定。
- `results/ffbp_final_metrics.json`：最终实际输入、参数、核心计算时间和实测直接 BP 对照。
- `results/benchmark.json`：相同实测 ROI、相同输出网格的 FFBP/直接 BP 计时。
- `GPU直接BP与FFBP基线实验.md`：RTX 4070 Laptop 上 FP64、FP32 和固定混合精度的同网格结果、误差与计时。
- `results/gpu_baseline/`：GPU/CPU 复数图像、原始指标 JSON 和对比图。
- `SOURCES.md`：论文、权威机构和开源仓库检索记录。

## 数据与参数依据

| 参数 | 使用值或处理 | 项目依据 |
|---|---|---|
| 载频 | 1.4 GHz | `read_rawdata_1.m`、实测 RD/BP 脚本 |
| 距离采样率 | 400 MHz | 原始解码器与采样窗定义 |
| 带宽、脉宽、PRF | 200 MHz、1 μs、1000 Hz | 原始帧头逐帧核对 |
| 回波窗 | 50 μs，20,000 复数采样/脉冲 | 原始帧头 |
| 有效脉冲 | 12,757，帧号 26781–39537，无跳号 | 原始 `.dat` 帧头，不使用 MAT 图像 |
| 有效时间 | 12.756 s | 帧号/PRF |
| 高度 | 初始 18 m，保留垂向速度积分变化 | `raw_cs.m:35` 的 `H=18`，以及其运动补偿写法 |
| 轨迹 | 逐帧 ENU 速度积分，平均约 17.09 m/s，孔径约 218 m | 原始导航；原脚本的 `[-7;16.23;-0.01]` 与首帧相符 |
| 调频率符号 | 负号 | `Kr=-Br/Tp`；正负匹配滤波实测对照支持负号 |
| 距离时刻约定 | `R[k]=c*(t0-Tp/2+k/fs)/2`；`t0=3 μs` | 实测 RD、CS 的 `Rref=(t0-Tp/2+...)*c/2`；匹配滤波参考以脉冲中心为零 |
| BP 相位符号 | `exp(-j*4*pi*fc*R/c)` | 正反相位符号实测对照；只改变 BP 载频补偿符号，不把负调频率与它混为一谈 |
| 场景侧向 | 默认运动方向左侧为正跨航迹 | `raw_cs.m` 的目标放在北向正侧提供线索，左右侧诊断也支持此选择；这仍是未经过天线外参标定的场景约定 |
| 波束 | 原脚本有 30°，但未硬裁切 | 观测数据覆盖比该参数更宽；未把未校准的波束值用来删掉有效回波 |

原始文件大小为 1 GiB。按长度可划分出 13,388 个完整帧槽位，但尾部 631 个槽位没有有效帧头，不能当作真实脉冲。项目旧 MAT 中同一尾部也含零值导航。详情见 `capture_audit.json`。

首帧航向记录约 292°，导航速度推算的运动方向约 113°，两者近乎相反，说明不能直接把该航向当成天线波束方向。文件注释中有车辆试验线索，但没有可信的完整天线安装角、杆臂和平台型号记录。本次没有用 RADARSAT 示例的 7062 m/s 或 CS 试验分支的 60 m/s 替代实测轨迹。

## 处理范围与图像含义

全部有效脉冲均参与最终成像，不在最终 FFBP 中抽稀脉冲。匹配滤波覆盖每帧全部 20,000 个距离采样。根据实测功率随距离分布，最终高分辨率场景选为孔径中心斜距 500–2800 m、`sin(angle)` 为 −0.65 至 +0.55。采集窗的更远距离主要为低信噪比数据，保留在距离压缩缓存，可用命令行扩展处理。

完整极坐标图的距离网格间距 0.25 m；角度间距根据实际孔径长度和 3 倍过采样确定。地面图网格为 1 m。**网格间距不等于实测分辨率**。带宽对应的未加窗名义距离分辨率约 0.75 m，实际还受窗函数、轨迹误差、地形和采集系统影响。

地面显示采用平地模型，没有 DEM、地面控制点或绝对辐射定标。坐标是局部米制坐标，不是已地理配准的经纬度地图。图像亮度是相对幅度 dB，不能直接解释为校准的后向散射系数。

预览图的扇形黑色边界是本次处理范围的裁剪，不代表地物。硬件触发时刻和系统延迟未独立标定，所以距离轴遵循项目约定，不能作为经过外场标定的绝对测距成果。

自聚焦基于多个区域的强散射候选点提取复数相位梯度，拟合二维有效相位中心修正，固定高度。该修正兼含导航、系统及场景模型的误差，**不是独立测得的真实运动误差**。程序比较实际生成的迭代图，保留固定诊断场景上强度对比度最高的已评估轨迹，见 `autofocus_metrics.json`。未经自聚焦的初始探索图为 `ffbp_full_*`，其距离参考尚未扣除半脉宽，不宜直接做逐像素前后差分。

## 环境与复现

本机已验证 Python：`D:\develop\Anaconda\python.exe`；NumPy 1.26.4、SciPy 1.17.0、Numba 0.59.1。完整场景结果仍由 CPU 生成；GPU 基线使用 CuPy 13.6.0 和 MATLAB R2024a 所带的 NVIDIA NVRTC 12.2 编译 `gpu/kernels.cu`。首次运行会产生 Numba 与 NVRTC 编译缓存。

在项目根目录运行：

```powershell
python fast_bp/diagnose.py
python fast_bp/validate.py
python fast_bp/validate_autofocus.py
python fast_bp/run.py --name ffbp_navigation
python fast_bp/autofocus.py
python fast_bp/run.py --position-file fast_bp/results/autofocus_position.npy --name ffbp_final
python fast_bp/benchmark.py
python fast_bp/render_results.py
python fast_bp/gpu_baseline.py --repeats 5 --warmups 1 --cpu-repeats 3 --chunk-pulses 64
```

`run.py` 首次生成约 2.04 GB 的 `range_compressed.npy`，后续直接复用匹配的缓存。临时压缩文件完成后才重命名；源文件大小、修改时间或压缩参数改变会重建缓存。提供的所有结果仅在新增的 `fast_bp/` 下，旧脚本和旧成像图未覆盖。

在 MATLAB 中可直接查看最终复数图：

```matlab
s = load('fast_bp/results/ffbp_final_ground.mat');
im = abs(s.image_complex);
db = 20*log10(im/max(im(:)) + eps);
imagesc(s.cross_track_m, s.along_track_m, db, [-60 -10]);
axis xy image; colormap gray; colorbar;
xlabel('Cross-track (m)'); ylabel('Along-track (m)');
```

也可运行根目录的 `run_fast_bp.m`，使用已生成的自聚焦轨迹重新成像并显示结果。

## 验证和限制

1. 独立模拟多个不在网格上的点散射体，包含不均匀采样与横向/垂向运动，对 4、8、16 叶子分解分别与直接 BP 比较；同时检查目标位置和复数误差。
2. 独立模拟分数采样延迟的原始 LFM，检查匹配滤波峰位置及群延迟处理。
3. 独立加入已知二维相位中心误差，检查自聚焦估计方向和误差减小。
4. 最终实测图在 4,000 个相同坐标上与直接 BP 对照，其中包含随机内部像素和强散射候选像素。比较双方使用完全相同的输入、修正轨迹和窗函数。
5. 同一实测 ROI 上计时，排除 JIT、距离压缩、磁盘导出和自聚焦。局部加速比不能直接宣称为全场景加速比。

这些检查验证数值实现和内部一致性；直接 BP 与 FFBP 共用的错误几何假设无法由这种比较发现。绝对位置、真实地物类型和实测分辨率仍需天线外参、地形或角反射器等地面参考确认。
