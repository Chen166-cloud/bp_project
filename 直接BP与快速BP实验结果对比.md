# 直接 BP 与快速 BP（FFBP）实验结果对比

整理日期：2026-09-16

## 1. 数据与实验条件

| 项目 | 设置 |
| --- | --- |
| 实测数据 | [2018-08-14_18-54-08.dat](<D:/Users/Chen/Desktop/research/试验数据/2018-08-14_18-54-08.dat>)，1 GiB |
| 有效数据 | 12,757 个脉冲，每脉冲 20,000 个距离采样点 |
| 雷达参数 | 载频 1.4 GHz，带宽 200 MHz，脉宽 1 μs，采样率 400 MHz，PRF 1 kHz |
| 轨迹 | 导航速度积分；沿轨孔径 218.00 m；初始高度 18 m，平地模型 |
| 预处理 | LFM 距离压缩；距离 Hamming 窗，方位 Hann 窗 |
| 计算平台 | Intel Core i9-14900HX，CPU 16 线程，未使用 GPU |

直接 BP 对每个像素使用全部脉冲进行距离插值、相位补偿和相干累加。FFBP 将孔径划分为短子孔径，在局部极坐标中成像后逐级相干合并。本实验采用最多 64 脉冲的叶节点；完整孔径对应 256 个叶节点和 255 次合并。两种方法使用相同输入、轨迹、窗函数及输出坐标。

![BP 与 FFBP 结果概览](<D:/Users/Chen/Desktop/research/试验数据/sar_quality/results/bp_vs_ffbp_summary.png>)

**图 1.** 左：理想点目标半功率宽度；中：六个实测区域内 FFBP 相对直接 BP 的复数误差；右：同网格 CPU 核心计时。

## 2. 理想点目标实验

在 800、1400 和 2200 m 斜距分别生成孤立点目标原始 LFM 回波。仿真采用实际雷达参数和导航轨迹，不加入噪声、运动误差、天线方向图及传播衰减。IRW 为一维半功率全宽；PSLR、ISLR 采用负 dB 的“旁瓣/主瓣”定义，主瓣边界取峰值两侧首个局部极小值 [2,4]。

### 2.1 分辨率与复数保真度

| 斜距 / m | BP 距离 IRW / m | FFBP 距离 IRW / m | 距离变化 | BP 方位 IRW / m | FFBP 方位 IRW / m | 方位变化 | 复数误差 | 相干度 γ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 800 | 0.9817 | 0.9940 | +1.25% | 0.5692 | 0.5695 | +0.04% | 3.306% | 0.9999199 |
| 1400 | 0.9816 | 0.9932 | +1.18% | 0.9943 | 0.9904 | −0.40% | 3.107% | 0.9999199 |
| 2200 | 0.9813 | 0.9910 | +0.99% | 1.5619 | 1.5629 | +0.06% | 2.693% | 0.9999238 |

FFBP 的距离向 IRW 比直接 BP 增加 0.99%～1.25%；方位向 IRW 变化不超过 0.40%。两种算法的主瓣宽度基本一致。

### 2.2 旁瓣

| 斜距 / m | 方向 | BP PSLR / dB | FFBP PSLR / dB | BP ISLR / dB | FFBP ISLR / dB |
| --- | --- | --- | --- | --- | --- |
| 800 | 距离 | −38.22 | −37.06 | −32.27 | −32.32 |
| 800 | 方位 | −31.04 | −29.98 | −32.16 | −30.35 |
| 1400 | 距离 | −38.10 | −36.91 | −30.73 | −30.85 |
| 1400 | 方位 | −31.49 | −30.50 | −32.63 | −30.82 |
| 2200 | 距离 | −38.06 | −37.12 | −30.43 | −30.53 |
| 2200 | 方位 | −31.67 | −30.59 | −32.84 | −30.83 |

FFBP 的距离向 PSLR 上升约 0.94～1.19 dB，距离向 ISLR 变化较小；方位向 PSLR 上升约 0.99～1.08 dB，方位向 ISLR 上升约 1.81～2.02 dB。快速分解主要影响低幅旁瓣，主瓣变化较小。

![点目标响应剖面](<D:/Users/Chen/Desktop/research/试验数据/sar_quality/results/point_target_profiles.png>)

**图 2.** 三个斜距处的距离向和方位向点目标剖面。实线为直接 BP，虚线为 FFBP；主瓣接近重合，旁瓣区可见差异。

## 3. 实测数据对比

在实测场景中选取六个固定的 60 m × 60 m 区域，采用 0.25 m 网格。导航轨迹和自聚焦轨迹各形成六组直接 BP/FFBP 对照，共 12 组。

| 轨迹 | 复数误差范围 | 平均复数误差 | SER 范围 / dB | 平均 SER / dB | 相干度范围 | 平均相干度 |
| --- | --- | --- | --- | --- | --- | --- |
| 导航轨迹 | 2.851%～3.917% | 3.393% | 28.14～30.90 | 29.44 | 0.999549～0.999882 | 0.999749 |
| 自聚焦轨迹 | 3.008%～3.902% | 3.416% | 28.17～30.43 | 29.37 | 0.999554～0.999856 | 0.999746 |
| 全部 12 组 | 2.851%～3.917% | 3.404% | 28.14～30.90 | 29.41 | 0.999549～0.999882 | 0.999747 |

两种算法的归一化强度熵平均绝对差为 1.57×10⁻⁴，强度对比度平均相对差为 0.189%。FFBP 在六个区域内保持了直接 BP 的主要结构、能量分布和复数相位关系。

![实测区域对比](<D:/Users/Chen/Desktop/research/试验数据/sar_quality/results/real_roi_comparison.png>)

**图 3.** 六个固定区域的成像结果。每行对应一个区域；四列依次为直接 BP/导航轨迹、FFBP/导航轨迹、直接 BP/自聚焦轨迹和 FFBP/自聚焦轨迹。每一行采用共同的幅度参考和相同显示范围。

## 4. 计算效率

| 项目 | 直接 BP | FFBP |
| --- | --- | --- |
| 输入脉冲 | 12,757 | 12,757 |
| 输出网格 | 360 × 365 | 360 × 365 |
| CPU 核心耗时 | 10.363 s | 0.491 s |
| 相对直接 BP 加速比 | 1.00 | 21.10 |
| FFBP 对 BP 的复数误差 | — | 3.286% |
| FFBP 与 BP 的相干度 | — | 0.9998335 |

计时采用已预热的相同输出网格，不包括原始数据解码、距离压缩、JIT 编译、自聚焦和文件导出。完整 FFBP 极坐标图为 9200 × 7312，核心成像耗时 56.48 s；未运行同尺寸完整直接 BP，因而不对整幅图耗时作比例外推。

## 5. 结论

在本实验设置下，FFBP 以约 1% 的距离主瓣展宽和约 1～2 dB 的方位旁瓣上升为代价，保持了直接 BP 的主要复数成像结果，并在指定同网格区域获得 21.10 倍 CPU 核心加速。

点目标结果描述理想信号模型下的算法响应。实测场景尚无角反射器或已知地面点，本文不将仿真 IRW 作为外场绝对分辨率。高相干度表示 FFBP 接近同条件直接 BP，不包含轨迹、定时和平地模型的共同误差。

## 数据与复现文件

- 点目标数据：[point_targets.json](<D:/Users/Chen/Desktop/research/试验数据/sar_quality/results/point_targets.json>)
- 实测区域数据：[real_rois.json](<D:/Users/Chen/Desktop/research/试验数据/sar_quality/results/real_rois.json>)
- 严格计时记录：[benchmark.json](<D:/Users/Chen/Desktop/research/试验数据/fast_bp/results/benchmark.json>)
- FFBP 实现：[ffbp.py](<D:/Users/Chen/Desktop/research/试验数据/fast_bp/ffbp.py>)
- 图 1 生成程序：[build_bp_comparison_figure.py](<D:/Users/Chen/Desktop/research/试验数据/sar_quality/build_bp_comparison_figure.py>)

## 参考文献

1. Ulander L M H, Hellsten H, Stenström G. [Synthetic-aperture radar processing using fast factorized back-projection](https://doi.org/10.1109/TAES.2003.1238734). *IEEE Transactions on Aerospace and Electronic Systems*, 2003, 39(3): 760–776.
2. Doerry A W. [Anatomy of a SAR Impulse Response](https://doi.org/10.2172/920461). Sandia National Laboratories, SAND2007-5042, 2007.
3. Góes J A, Castro V, Bins L S, et al. [Spiral SAR Imaging with Fast Factorized Back-Projection: A Phase Error Analysis](https://doi.org/10.3390/s21155099). *Sensors*, 2021, 21(15): 5099.
4. Huber S. [Spaceborne SAR Systems with Digital Beamforming and Reflector Antenna](https://elib.dlr.de/76420/1/Dissertation_Huber.pdf). Karlsruhe Institute of Technology, 2014, Appendix G.
