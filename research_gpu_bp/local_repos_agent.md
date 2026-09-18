# 本地 GPU BP 审计与开源基线核查

检索与只读审计日期：2026-09-17。未修改原始数据、原始代码，未运行新的 GPU 性能实验。数值来自现有 JSON；源码判断与优化收益假设明确分开。

**按用户最新限定，科研主问题是数据已驻留 GPU 时的 BP 计算提速。** 主计时从每次成像必需的初始化/计算开始，到最终复图在 GPU 上有效可用结束，包含本次必需的清零、设备重排、核执行、中间图处理以及执行区间内的供给空隙。输入驻留前的上传、输出下载和读盘不作为本课题的性能目标。通用预分配、建树和编译对全部基线统一处理；可复用准备成本另报，逐帧必须重建的计划/查表/重排不能移出计时。下述 H2D 信息仅是既有记录的适用边界与附带部署资料。

## 1. 已有实验能证明什么

来源：[metrics.json](</D:/Users/Chen/Desktop/research/试验数据/fast_bp/results/gpu_baseline/metrics.json>)、[operator_validation.json](</D:/Users/Chen/Desktop/research/试验数据/fast_bp/results/gpu_baseline/operator_validation.json>)、[gpu_backend.py](</D:/Users/Chen/Desktop/research/试验数据/fast_bp/gpu_backend.py>)、[kernels.cu](</D:/Users/Chen/Desktop/research/试验数据/fast_bp/gpu/kernels.cu>)、[gpu_baseline.py](</D:/Users/Chen/Desktop/research/试验数据/fast_bp/gpu_baseline.py>)。

- GPU：RTX 4070 Laptop，CC 8.9，约 8 GB；Windows WDDM。记录环境为 CuPy 13.6.0、Python 3.12.3、NVRTC 12.2、CUDA runtime 12.9；驱动 API 版本 13.3 不是所用编译器版本。当前只读 nvidia-smi 确认型号与驱动 610.88；没有据此追认实验时温度或功耗。
- 数据：1.4 GHz、200 MHz 带宽、400 MHz 采样、1 kHz PRF，12,757 个连续有效脉冲，距离压缩数据为 complex64 [12757,20000]，约 2.041 GB。已有轨迹经过 autofocus，平地 z=0，Hann 方位窗。
- 本轮 GPU 基线仅为 r=1380–1470 m、u=-0.03–0.03 的 360×365 极坐标 ROI。全幅 9200×7312 图已有 CPU FFBP 56.48 s 记录，不能用 ROI GPU 时间外推全幅。
- FFBP 固定计划：leaf_size 上限64，平衡二叉树最终256叶、255合并、511节点、8层合并。叶节点实际约49/50脉冲，不能把上限64误写为每个叶子严格64。四点 Lagrange；每个二维重采样需4×4邻域。
- 计划边界、保护单元、ceil/crop 均由 FP64 主机计划冻结，避免三种精度偷偷改变网格。

| 算法 | 算术 | 原记录GPU驻留中位/ms | 相对同算法CPU复误差 | 同算法SER/dB |
|---|---|---:|---:|---:|
| 直接BP | FP64 | 1158.554 | 2.13e-10 | 193.42 |
| 直接BP | FP32 | 38.050 | 3.685e-3 | 48.67 |
| 直接BP | mixed | 611.359 | 1.879e-6 | 114.52 |
| FFBP | FP64 | 93.853 | 7.31e-9 | 162.72 |
| FFBP | FP32 | 45.877 | 1.150e-2 | 38.78 |
| FFBP | mixed | 50.085 | 2.040e-7 | 133.81 |

mixed 在本代码中仍为 FP64 几何、sqrt、采样坐标和 sincos；只有插值与累加为 FP32。FP64 模式的输入和所有存储节点仍是 complex64，不能称完整双精度数据链路。

FP64/mixed FFBP 相对直接 BP 的复误差约3.286%、SER 29.67 dB、强像素去均值相位标准差0.9105°，峰值下降0.31365 dB。133.81 dB 是对同算法 FFBP 的移植一致性，不能用于声称对直接 BP 的成像误差为133.81 dB。根任务另做的统一 cpu_direct_reference 重分析得到 FP32 FFBP 总误差3.49707%；旧 JSON 的3.46473%是对 FP32 direct，不同参考不能混用。

## 2. 基线公平性和容易过度解读之处

1. 268×是 GPU FP32 对 CPU FP64算术 Numba直接BP，混入精度变化；它可作工程对比，不能作为同精度硬件加速结论。CPU是16线程，不能写单核基线。
2. CPU FFBP 计时调用 form_ffbp，包含主机建树；GPU将计划冻结在计时之前。需要分别报告 plan、compute、transfer、full-pipeline。
3. run_direct 在计时开始前 cp.zeros 初始化累加器并分配输出；FFBP 在计时内逐节点分配。公平驻留比较应统一通用预分配策略，并将每次计算必需的初始化、设备重排和算子执行全部计入，从输入已驻留到输出在GPU上可用。
4. CUDA event 包含所在流的运行区间与发射造成的空隙，不等于纯核执行时间之和。Python递归与小核发射瓶颈需要 Nsight Systems 确认。
5. FP32 FFBP 5次时间52.88→49.97→45.88→40.50→40.11 ms，顺序漂移显著，warmup只有1次。需要交错/随机次序试验、温度/时钟/功耗记录和更多独立重复，不能精确解释当前1.21×排序幅度。
6. 本课题排除 H2D/D2H 合理且与用户范围一致；约50 ms驻留计算的科研价值不应被约1秒上传抵消。附带部署记录：H2D一次约1017 ms，普通RAM staging约1112 ms；主机RAM至结果约1068 ms是分项相加估计，并非完整流水实测。mmap staging在CPU参考读取后进行，不等价冷盘I/O；这些数值不进入本课题的主比较。
7. CPU/GPU近乎一致与重复零差异，只验证这份算子定义、这组输入、当前调度的复现性。参考输出本身存储complex64，极高SER不代表物理真值精度。
8. 相位统计仅覆盖参考峰值以上-30 dB并去除圆均值相位偏置；应同时保留原始偏置和弱散射区局部误差。图像复相干系数接近1不等价干涉产品所有像素相位正确。
9. operator_validation包含奇数脉冲、两组符号/视向、非零phase、分数插值，有价值；这不是所有波段/距离/轨迹/强动态范围的广泛验证。两组测试将符号和视向配对，也不是二者笛卡尔积四组合。
10. 现有实测无标定角反射器和地面真值；不能靠自然亮点把IRW/PSLR/ISLR当真实系统分辨率。原始13,388槽中631无效槽，但保留的12,757脉冲编号连续，不能误写“丢失631个连续孔径脉冲”。
11. “FP32 FFBP启动与中间图访存占主导”是有源码依据的待测假设，现有材料未见 profiler 证据。

## 3. 最具体的工程机会与可发展的问题

### A. FFBP树按层batch并融合重采样

当前 _execute_node 深度优先在同一流逐节点发射：256 leaf + 510 regrid + 255 add + 1 crop = 1022次核启动。左右子图分别写全幅父网格，再额外加法；可融合为一个父节点核，减少中间读写和启动。同层节点使用packed buffers/offsets做batch，稳定形状用CUDA Graph复用；预分配可消除运行期 allocator噪声。

融合会改变complex64存储/加法舍入点。必须并存“严格匹配旧舍入顺序”的基线与“重新评估误差后允许融合”的候选，而非只看幅度图。科学问题应是层数、局部角带宽、插值阶数、算术精度、误差传播和资源占用如何共同决定分区，单纯kernel fusion不足顶刊贡献。

### B. 精度不要只做FP32/FP64枚举

本地 fixed-mixed direct仍611 ms，FP32仅38 ms，说明去除全部FP64的潜在回报很大；这不证明FP64吞吐为唯一瓶颈。k≈58.684 rad/m时1°相位误差对应径向误差约0.297 mm；FP32对约1.4 km数值的减法易损失低位，但全FP32的实测误差仍需按场景判断。

可研究局部参考+残差坐标、按节点/子块误差界选数值策略，以及插值误差/近似距离误差/累加误差的统一预算。但MatX已有FloatFloat/TaylorFast/PhaseLUT，gotcha-back已有增量相位/牛顿迭代；这些原语不是空白。新意须落在可证明的自适应策略、跨几何可泛化的选择、可复用实现和真实下游质量。

### C. 不急于Tensor Core化

当前热点包含不规则4/16点gather、sqrt/sincos和复数求和，不能直接映射成大GEMM。先测距离/相位路径、缓存命中、内存交易、寄存器溢出、SM/SFU占用。若使用低秩/局部多项式把一组像素—脉冲相位映射为密集块，必须把构造、gather重排、额外padding与误差控制全部计入，证明总成本低于优化CUDA直接核。单凭理论TFLOPS不构成可行性证据。

### D. 强基线和可复现实验矩阵

至少比较 optimized direct BP、optimized FFBP、MatX、torchbp，并确保同网格/同插值或给出同质量预算的Pareto。覆盖本地近地L波段真实轨迹、GOTCHA公开数据、可控点目标/曲线轨迹/更长距离与更高载频。4070做快速迭代；最终最好加入另一代消费GPU以及FP64更强的服务器GPU或嵌入式GPU，避免结论仅是某一产品FP64削弱的副作用。具体型号应根据合作资源选，不建议现阶段先采购。

### E. 驻留数据流与附带部署信息

驻留优化应研究像素块×脉冲块的几何复用、range-profile gather局部性、共享内存暂存和寄存器累计，用 profiler 测量实际显存/L2/L1流量、缓存命中、同步和寄存器压力。2 GB输入数组不代表BP会从DRAM读取所有2 GB；上传体积缩小不自动等于驻留执行变快。若为了局部性做设备侧packing，逐次必需的packing应计入驻留总时间；静态可复用packing另报准备成本与复用条件。

附带部署估算：最终direct ROI每脉冲249–264 bins、独立紧凑窗口约26.123 MB，公共窗口约27.453 MB，相比原输入2.041 GB。这些是根任务对保存轨迹/网格的几何字节估计，尚无GPU加速实测，不作为本次主课题收益或优先实验。该窗口不能直接用于FFBP，全部叶节点、guard和中间节点仍需重新证明覆盖。若以后研究主机上传流水，需另计CPU packing、pinned memory复制和边界验证；应作为独立部署实验。

## 4. 官方或作者开源基线核查

本轮使用GitHub公开仓库页内embedded metadata取得default-branch HEAD，再访问对应commit页提取committedDate。匿名API 403限流，未捏造API响应；下列为本轮观察到的HEAD，不表示之后不会更新。所有仓库只读检查，未在本机编译或跑通。

| 仓库 | 证据与用途 | 许可 | 观察到的HEAD / 日期 | 建议等级 |
|---|---|---|---|---|
| [NVIDIA/MatX](https://github.com/NVIDIA/MatX) | 官方CUDA direct BP；Double/Mixed/FloatFloat/TaylorFast/Float及PhaseLUT | BSD-3-Clause | [a108745](https://github.com/NVIDIA/MatX/commit/a108745201415478e17fa72e9e328dc0514ae7fb)，2026-09-14（原页面-04时区） | 核心工程基线；sar_bp为experimental，Added in version head；不可据文档页抬头断言stable1.1已包含 |
| [Ttl/torchbp](https://github.com/Ttl/torchbp) | 作者C++/CUDA PyTorch扩展；direct、FFBP/AFBP/CFBP、autofocus与可微算子 | MIT | [b371884](https://github.com/Ttl/torchbp/commit/b371884a8ef49be39de350f529a4f83ada5b5e65)，2026-09-13 | 核心算法/可微基线；有tests与benchmark，当前README测试CUDA12.9，需绑定确切PyTorch ABI |
| [tbensonatl/gotcha-back](https://github.com/tbensonatl/gotcha-back) | 七种列出的CUDA路径，FP64/mixed/增量相位/Newton-Raphson/shared参考距离/texture/FP32，并带SER指标 | BSD-3-Clause | [06da38a](https://github.com/tbensonatl/gotcha-back/commit/06da38ad67d1812955dfa689f53fcc0ac1f45c0e)，2024-01-28 | 核心direct BP基线，作者明确为hobbyist；勿写成AFRL官方GPU实现 |
| [dm6718/RITSAR](https://github.com/dm6718/RITSAR) | BP、数字聚束、FFBP及多进程FFBP，GOTCHA/Sandia例子 | [MIT](https://github.com/dm6718/RITSAR/blob/master/LICENSE) | [0e36d2e](https://github.com/dm6718/RITSAR/commit/0e36d2ed50a92304c9b2c92ad1e1e46e042011e5)，2016-10-04 | 算法核对/CPU参考；不能把multiprocessing称CUDA |
| [IMS-AS-LUH/sar-sim](https://github.com/IMS-AS-LUH/sar-sim) | Hannover IMS机构FMCW仿真器；sarsim/simjob.py含Numba CUDA BP，支持载入信号 | 本轮根目录与源文件未发现明确许可，LICENSE返回404 | [98781db](https://github.com/IMS-AS-LUH/sar-sim/commit/98781db587cc50c64ab92ab68c4bfbb44f8cdfc1)，2024-03-19 | 仿真/正确性/数据约定参考，非极限性能SOTA；引用复用前确认许可 |

gotcha-back README列出的性能表为作者机器/数据/SER口径，本地不借用作已复现速度。其FP32在GOTCHA约15 dB SER，与本地48.67 dB差别提示几何、频率、相位定义的重要性，而非本地实现已经更准确。

torchbp README的370 Gbackprojections/s（3090Ti）也是作者报告，不与本地44 G左右跨机器直接排名；其benchmark用complex64、FP32位置、不同网格与脉冲数。当前[官方文档](https://ttl.github.io/torchbp/)明确已有FFBP/AFBP/CFBP及DEM、autofocus等能力。

sar-sim相关论文：[Fahnemann等，RADAR 2022](https://ieeexplore.ieee.org/document/10106365)，DOI [10.1049/ICP.2023.1281](https://doi.org/10.1049/ICP.2023.1281)。机构作者实现有可信来源，但FMCW信号相位约定需改造，不可把本地脉冲LFM数据直接喂入而不核验。

其他搜索到的[atalNZM课程项目](https://github.com/atalNZM/Synthetic-Aperture-Radar-Backprojection-Imager)为MIT许可教学代码、仅理想模拟场景，可读texture/shared方案的正反例，但不能替代SOTA。对于跨域方法，[CERN/TIGRE](https://github.com/CERN/TIGRE)是可信CT GPU前/反投影代码，可参考大数据分块和多GPU编排；CT的实值投影不具有SAR载频相干相位敏感性，误差容限不能照搬。

## 5. MatX源码与输入语义：最需要提前避免的坑

已检查并保存三份源码快照；[来源与校验清单](</D:/Users/Chen/Desktop/research/试验数据/research_gpu_bp/references/matx_sources.json>)记录固定commit和SHA256，本地文件为[operator](</D:/Users/Chen/Desktop/research/试验数据/research_gpu_bp/references/MatX_include_matx_operators_sar_bp.h>)、[transform](</D:/Users/Chen/Desktop/research/试验数据/research_gpu_bp/references/MatX_include_matx_transforms_sar_bp.h>)、[kernels](</D:/Users/Chen/Desktop/research/试验数据/research_gpu_bp/references/MatX_include_matx_kernels_sar_bp.cuh>)。对应官方固定版本链接：

- [operator API](https://github.com/NVIDIA/MatX/blob/a108745201415478e17fa72e9e328dc0514ae7fb/include/matx/operators/sar_bp.h)
- [transform dispatch](https://github.com/NVIDIA/MatX/blob/a108745201415478e17fa72e9e328dc0514ae7fb/include/matx/transforms/sar_bp.h)
- [CUDA kernels](https://github.com/NVIDIA/MatX/blob/a108745201415478e17fa72e9e328dc0514ae7fb/include/matx/kernels/sar_bp.cuh)
- [官方sar_bp文档](https://nvidia.github.io/MatX/api/signalimage/radar/sar_bp.html)

源码要点（不是未实测的速度主张）：

- transform用16×16线程块；每像素线程在寄存器累加，最后读initial_image、写output各一次。
- pulse block为256。FloatFloat、TaylorFast以及带PhaseLUT的Float/Mixed缓存每脉冲几何量到shared。未缓存整条range profile，仍从全局range_profiles按当前像素range bin取样。
- range_profiles通常为[pulse,range]，取相邻两个样本做线性插值。相邻像素可能共用缓存行但不保证理想合并访问。
- PhaseLUT查整数bin相位，再以分数bin小相位调用__sincosf；不是对大绝对距离直接用低精度三角函数。
- TaylorFast以CTA有效区域中点为参考，FP64计算参考量，FP32局部二阶近似，可选三阶；固定/零高度是调用者断言，无运行时验证。
- contiguous tensor-view启用unit-stride指针访问快路径，computed operators/zip等可能改变路径；需看最终参数类型，不能仅语义等价就认定性能等价。
- transform每次调用在PhaseLUT路径仍发射LUT填充核，实测需包含它或明确报告复用策略。该算子是CUDA transform，不能把MatX总体JIT fusion优势默认套在sar_bp之上。

MatX默认采样坐标：

    b = (R_pixel - range_to_mcp[p]) / del_r + floor(num_range_bins/2)

本地代码则为：

    q = (R_pixel - range_start) / data_dr
    phase = k * (R_pixel - R_reference_pixel) + phase[p]

因此适配要同时处理距离轴、FFT-shift的零差距位置、MCP相位参考、负k符号、Hann权重、per-pulse相位，以及输出每像素去载频项。仅把数组FFTshift是不够的。应从这两组公式推导等价变换并用单脉冲、点目标、随机复数profiles、正负k逐步验证；若构造常量MCP以映射现有绝对距离轴，应将对应常量和每像素输出相位补偿一起推导验证。

MatX是双点线性插值，本地是四点Lagrange；GOTCHA预处理和torchbp的插值也各异。比较时必须选同插值实现，或把精度与吞吐画成共同参考下的Pareto边界并计入必要上采样与预处理开销。这比先追“多少倍”更关键。

## 6. 额外先行工作提醒

[IEICE官方记录：A study of fast factorized backprojection on multiple GPUs for large-sized SAR imaging，2023](https://ken.ieice.org/ken/paper/20230303YCTv/eng/)明确面向信号和图像无法装入GPU的多GPU FFBP。out-of-core多GPU FFBP本身不是未探索问题，应交由总文献表进一步核实其发表层级、正文和后续工作。
