# GPU 跨领域方法迁移到 SAR BP：证据、边界与研究假设

检索日期：2026-09-17。按用户最新要求修订：**研究目标为数据已在 GPU 显存中的驻留计算提速**。本文区分文献已证实的事实、基于本地结果的解释，以及尚未验证的研究假设。未运行 GPU 实验，未修改现有算法代码。本地参数与计时由主任务对已有实验文件的审计提供，本文没有独立重测。

## 1. 最值得注意的结论

最有希望的主线是：**相干误差约束下的 BP/FFBP 执行树分割与融合**，联合选择 direct 叶计算、合并深度、融合层数、精度、插值和 GPU 内部数据布局，优化驻留时间 `T_resident`。“联合精度与性能自动调参”本身已有直接先例；潜在独特贡献是把 FFBP 树上的误差传播、相位计算及 DRAM/L2/shared/register 成本联系起来，决定在哪里保留中间子图、在哪里融合或重算。

目前 RTX 4070 Laptop 的小 ROI 驻留实验中，direct FP32 为 38.05 ms，FFBP FP32 为 45.88 ms，直接 BP 反而更快；高精度时 direct 为 1158.55 ms、FFBP 为 93.85 ms。这是研究算法选择边界的好起点，但只证明这个 ROI、实现及精度下的结果，尚不能断定 FFBP 的瓶颈就是 launch 或带宽。

2.04 GB 是输入分配体积，**不代表每次驻留 BP 都读取 2.04 GB**。已有 GPU gather 可能只访问当前 ROI 的距离样本，L1/L2 也可能已有很强复用；26 MB 几何窗口和 H2D 计时均不能证明驻留加速。距离窗在本研究中的作用是构造每个 tile 的依赖足迹、估计真实可复用的数据以及设计 DRAM→shared 的预取。磁盘、RAM 和 H2D 优化保留作背景资料，不列为科研主线或首要实验。

计时起点是输入、轨迹及必要元数据已经在 GPU 上，终点是完整复图可由后续 GPU 算子使用。`T_resident` 必须包含每次运行所需的 GPU 重排、临时图初始化、核执行、GPU 归约、必要同步以及依赖图中的 launch 间隙。编译、计划构造、图捕获和可复用预计算单独报告；同时给出首次使用成本和复用次数/摊销临界点，禁止把每帧必须重做的工作放入一次性准备。GPU 事件计时之外补充同步的 host 调用墙钟时间，区分设备执行和 host 提交开销；H2D/D2H 不混进驻留主指标。

## 2. 已核验的权威来源与可迁移部分

| 来源 | 已有贡献/事实 | 对 BP 的借鉴与限制 |
|---|---|---|
| [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) | 官方提供数据驻留、合并访问、共享内存、pinned memory、异步传输及计时方法 | 都应作为实现基线，不能单独宣称算法创新。先证明重复读取或传输确实存在 |
| [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/) | 层次 Roofline、执行管线、warp stall、缓存与访存分析 | 常规 FLOP Roofline 不能独立解释 BP 的 sqrt/sincos、FP64、延迟依赖和启动开销；需联合看指令及时间线 |
| [CUDA Advanced Kernel Programming](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html) | LDGSTS 异步拷贝支持 cc8.0+；TMA 为 cc9.0+ | 本地 Ada cc8.9 可以评估 cp.async；不能把 Hopper TMA 路径当作 4070 可用特性 |
| [FlashAttention, NeurIPS 2022](https://arxiv.org/abs/2205.14135)，[作者代码](https://github.com/dao-ailab/flash-attention) | 用 tiling 减少 HBM/SRAM 之间的数据移动，避免物化大中间矩阵 | BP 中值得迁移的是 IO 分析、融合和分块。成熟的像素线程 BP 已在寄存器累加，未必有 Npix×Np 中间矩阵可消除，不能期待照搬收益 |
| [FlashAttention-3](https://arxiv.org/abs/2407.08608) | 利用 Hopper 异步 Tensor Core/TMA、warp specialization 和低精度 | “重叠不同执行单元”可启发 BP 设计，但 BP 几何、插值、相位的依赖链不同；先实现适合 Ada 的方案 |
| [Real-time High-resolution X-Ray Computed Tomography, ICS 2024，ORNL 官方记录](https://impact.ornl.gov/en/publications/real-time-high-resolution-x-ray-computed-tomography/)，[DOI](https://doi.org/10.1145/3650200.3656634) | Tensor Core 优化 CT 反投影并恢复精度；相对高度优化生产库，A100/H100 报告 1.91×/2.15× | 很强的跨领域先例。CT 几何投影可矩阵化；SAR 还包含非线性斜距、相干相位和复数插值，收益不能直接移植 |
| [Petascale XCT, SC20 作者稿](https://arxiv.org/abs/2009.07226)，[作者 PDF](https://merthidayetoglu.github.io/samples/Hidayetoglu_PetascaleXCT.pdf) | 内存中心 (反)投影、混合精度、分层通信以及 3D 数据分解 | 借鉴按数据足迹组织子问题、先本地归约再跨节点通信；其超大规模 CT 的速度与规模不能当作 SAR 预测 |
| [CERN/TIGRE](https://github.com/CERN/TIGRE)，[官方调优说明](https://github.com/CERN/TIGRE/blob/master/Frontispiece/Tune_TIGRE.md) | 多 GPU CT projector/backprojector，批次和每线程体素数可调 | 参考成熟 CUDA 数据流、out-of-core 和多 GPU 框架；CT 插值精度容忍度不等于 SAR 相位容忍度 |
| [Recovering single precision accuracy from Tensor Cores, IJHPCA 2022](https://arxiv.org/abs/2203.03341) | 对低精度输入误差及 Tensor Core 内部舍入进行补偿，恢复 FP32 GEMM 精度 | 仅当 BP 能变成足够大的 GEMM 才有意义；“使用 FP32 累加”不能恢复已经量化掉的输入相位 |
| [CUTLASS 官方文档](https://docs.nvidia.com/cutlass/latest/overview.html)，[功能矩阵](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/functionality.html) | 提供矩阵分块、异步数据移动和多种混合精度 GEMM | 可作为矩阵化局部算子的后端；需选真实硬件支持的数据格式与 API |
| [RIKEN GEMMul8](https://github.com/RIKEN-RCCS/GEMMul8) | 基于 Ozaki Scheme II，用低精度矩阵引擎模拟较高精度 GEMM | 用来估算补偿代价与收益，不适合把逐像素 sqrt/sincos 直接替代成“高精度 Tensor Core” |
| [cuFINUFFT 论文](https://arxiv.org/abs/2102.08463) | cache-aware 重排、负载均衡、共享内存 spreading | BP 不必整体换 NUFFT，也可只借鉴 gather 的空间分桶和工作量平衡 |
| [当前 FINUFFT 代码](https://github.com/flatironinstitute/finufft)，[v2.5 GPU API](https://finufft.readthedocs.io/en/v2.5.0/python_gpu.html) | 当前 GPU API 已有 1D/2D/3D type-3；旧独立 cuFINUFFT 仓库是 legacy | 不要误报“GPU 只支持 type-1/2”。但 type-3 仍是双线性 Fourier 相位，不能直接表达任意近场球面波距离 |
| [A Butterfly Algorithm for Synthetic Aperture Radar Imaging，作者 PDF](https://math.mit.edu/icg/papers/butterfly-radar.pdf)，[SIAM DOI](https://doi.org/10.1137/100811593) | SAR 振荡积分的互补低秩/蝶形快速算法已有成熟数学先例 | 可研究 GPU 化与局部低秩算子，但不能把“低秩加速 SAR”当作全新发现；适用假设必须匹配 |
| [OptiX 官方页](https://developer.nvidia.com/rtx/ray-tracing/optix)，[Hardware-Accelerated SAR Simulation with NVIDIA-RTX Technology](https://arxiv.org/abs/2005.09736) | RT Core/OptiX 用于射线遍历，论文展示 SAR 正向仿真加速 | 是仿真而不是标准相干 BP 的证据。稠密 BP 通常没有可剪枝的 ray–triangle 可见性查询 |
| [NCCL 官方文档](https://docs.nvidia.com/deeplearning/nccl/) | 拓扑感知 collective 通信 | 用于分孔径后的复图归约，或二维分块后的 reduce-scatter；不是单卡当前首要任务 |
| [LLNL LEAP 功能说明](https://github.com/LLNL/LEAP/blob/main/LEAP_features.md) | 强调匹配的 forward/backprojector 对及其迭代算法意义 | 若扩展到迭代 SAR/自动聚焦/可微 BP，必须验证 adjoint consistency，不能只测幅度图 |

## 3. 新近相关工作已经占据了哪些创新空间

1. [Mixed-Precision Backprojection Using FP32 and FP64 for Boosting Satellite SAR Imaging on Embedded GPUs，MCSoC 2025](https://waseda.elsevierpure.com/en/publications/mixed-precision-backprojection-using-fp32-and-fp64-for-boosting-s/)，DOI [10.1109/MCSoC67473.2025.00101](https://doi.org/10.1109/MCSoC67473.2025.00101)：代表点基准距离 FP64，差量 FP32，AGX Orin 上 FFBP 比 FP64 基线快 10.3×。因此“高精度参考距离+低精度残差”不是新的主贡献。
2. [Synthetic Aperture Radar Processing Using Flexible and Seamless Factorized Back-Projection，Remote Sensing 2025](https://www.mdpi.com/2072-4292/17/6/1046)：根据计算成本决定继续层次合并或提前全局合并。仅做 FFBP 停止深度选择已不够。
3. [Accuracy-Aware Mixed-Precision GPU Auto-Tuning，TPDS 2026，作者机构页](https://ac.liacs.nl/blog/accuracy-tuning-tpds)，DOI [10.1109/TPDS.2026.3659324](https://doi.org/10.1109/TPDS.2026.3659324)：已将精度与 block、tiling、vector 等参数联合自动调节，研究精度—性能权衡。应列为方法学基线，而不是只比较手写单一精度代码。
4. [Kernel Float，TOMS 52(1), 2026](https://doi.org/10.1145/3779120)，[作者代码](https://github.com/KernelTuner/kernel_float)：提供可配置精度、数学函数近似及向量抽象。正式卷期是 2026，作者代码中引用也可能按 online-first 写 2025，应注明版本。
5. [Fast and Energy Efficient Image Formation of SAR Satellite Data using Embedded GPUs，IRS 2025](https://doi.org/10.23919/IRS64527.2025.11046049) 已出现在 [Kernel Tuner 研究软件目录](https://research-software-directory.org/software/kernel-tuner) 的相关成果中。仅宣称 SAR 自动调参也有先例；本文未读到其完整实现细节，不能声称复现。

## 4. 直接可用、需要改造、暂不值得三类判断

| 方法 | 判断 | 实验前应问的问题 |
|---|---|---|
| ROI 距离窗用于 GPU 依赖足迹和局部布局 | 驻留收益待证明 | 未裁剪 gather 原本访问多少 DRAM/L2 sectors？重新打包的每次 GPU 成本是否超过缓存收益？ |
| 像素×脉冲双重分块、shared memory、cp.async | 直接可用但性能有条件 | L1/L2 原本是否已完成复用？距离窗长度和每 block 共享像素数是否带来净节省？ |
| FFBP 层级批处理、CUDA Graph、减少 Python/host 启动 | 直接可用但先测 | 1022 次 launch 中究竟多少墙钟时间在启动、同步、内核？图捕获/重用次数是否足够摊销？ |
| FFBP 多层融合、寄存器/shared 中间子图 | 需结构改造 | 能消除多少真实 DRAM 中间图读写？跨层插值 halo 和重算是否导致 shared/register 溢出及 occupancy 下降？ |
| 误差预算驱动的精度与分解联合选择 | 需要算法改造 | 超越普通 Kernel Tuner 和 FFBP 停止条件的独特理论与结构是什么？ |
| Tensor Core 的几何/局部低秩计算 | 需要较大算法改造 | 是否形成足够宽的 GEMM？矩阵构造、补偿、sqrt/sincos/gather 仍占多少时间？ |
| NUFFT/Butterfly/局部 Fourier 展开 | 需要改变算子实现 | 近场曲率残差能否受控？预处理能否摊销？是否比同质量 FFBP 更好？ |
| RT Core/BVH 直接加速标准 BP | 当前大概率不值得 | 稠密 coherent sum 无可见性剔除；BVH 查询不能替代距离、复插值与相位累加 |
| FP16/FP8 全流程 BP | 当前不建议 | 绝对坐标量化可造成多周相位误差；弱目标/旁瓣动态范围是否首先失真？ |
| 多 GPU | 扩展阶段 | 当前单卡驻留数据约 2 GB；驻留计算能否覆盖新增卡间通信，需独立研究 |
| 磁盘/RAM 打包、异步 H2D | 范围外背景 | 可以改变系统加载成本，但不能作为本次驻留计算提速的证据 |

纹理硬件插值需要特别小心。NVIDIA [CUDA Programming Guide 的线性过滤说明](https://docs.nvidia.com/cuda/archive/12.8.1/pdf/CUDA_C_Programming_Guide.pdf)指出权重使用带 8 位小数的定点格式；它不等价于任意 FP32 软件线性插值。要单独比较插值误差、旁瓣和复相位，不应只按幅度图判断。

## 5. 和当前数据对应的推导

### 5.1 严格的 ROI 距离窗：作为 GPU 内部依赖足迹

本地输入为已距离压缩 `range_compressed.npy`，12757×20000 complex64，采样距离间隔 0.3747405725 m，起点 374.7405725 m。令 ROI 的三维包围球中心为 c、半径为 ρ，单站第 p 个天线位置为 a_p，则对任意像素 x：

    max(0, ||a_p-c||-ρ) ≤ R_p(x) ≤ ||a_p-c||+ρ.

这是三角不等式给出的精确界。把区间映射到**原始全局采样索引**，向外取整并加插值支撑/边界保护，即可为每个脉冲保留一个连续窗口。不必先计算全部 Np×Npix 距离。双站场景应分别约束 Tx/Rx 两段距离再相加，并匹配实际时延定义。

主任务用本地轨迹与最终 direct ROI 计算出：逐脉冲窗口约 26.123 MB、公共窗口约 27.453 MB，相对分配体积 2.04112 GB，前者约小 78.1 倍。**这仅是几何足迹与分配体积的比值，并非驻留 BP 的访存削减或加速预测。**未打包的 kernel 已可能只读取这部分样本；最终 direct ROI 的足迹也不能直接用于 FFBP 全部中间网格。公共窗只比逐脉冲窗多约 5.1%，ragged packing 未必有价值。

若比较原始布局、公共窗布局、逐脉冲窗口布局，均从同样的已驻留原始输入开始：每次需要的 GPU 打包必须纳入 `T_resident`，可复用布局另报准备成本和复用条件。正确性上保留原全局索引并在整数层减窗口 offset，避免重设浮点 range_start 带来新的舍入差异；边界外零值/有效性规则必须保持。DRAM bytes、L2 sectors、缓存命中、长记分牌停顿、寄存器占用和 kernel 时间须共同报告。

该界更直接的研究用途是 GPU shared-memory tile：每脉冲预取当前像素块所需的局部距离窗，重用数据而保持输出累加在寄存器内。粗略有效复用比为 `每块像素数×插值支撑 / 需要加载的距离样本数`，但必须与现有 L1/L2 命中比较，不能把理论重复访问直接等同 DRAM 流量。若已是 FP64 或特殊函数吞吐受限，应先优化相位计算，shared memory 本身不会解决该瓶颈。

### 5.2 相位尺度与精度预算

单站载频约 1.4 GHz，λ≈0.21414 m，相位误差满足：

    δφ = 4π δR / λ.

例如，0.01 rad 的**设计相位预算**对应 δR≈0.170 mm；这不是论文通用合格标准。1.4 km 附近单个 FP32 距离值的 ULP≈0.122 mm，几何运算、相减、相位参数约化会叠加误差，因此“输入精度为 FP32”不足以证明可接受。FP16 在这一量级的间距约 1 m，直接量化绝对距离明显不合适。

如果将 R 写成 R0+ΔR，要让高精度基准、局部差量和最后相位约化都进入误差分析。可以研究避免大数相消的恒等式（令 v=a-c，d=x-c）：

    R(x)-R(c) = (||d||² - 2 v·d) / (||v-d|| + ||v||).

这不是新算法声明；用途是建立可追踪的数值误差与计算成本。把 `||a||²+||x||²-2a·x` 直接塞进低精度 Tensor Core，可能因大数相消而更差。

若单项为 s_p exp(iφ_p)，用近似 ŝ_p、φ̂_p，可用三角不等式给出：

    |Î-I| ≤ Σ_p |ŝ_p-s_p| + Σ_p |s_p| |φ̂_p-φ_p| + E_acc,

其中利用了 |exp(iα)-exp(iβ)|≤|α-β|。插值误差可再用局部幅度/复信号变化界估计。该界是绝对误差，转成相对图像误差时会受相干抵消放大；不能把弱像素的相位误差直接和强像素混算。保守界若过松，将无法指导高速实现，这也是明确的失败条件。

建议同时报告复数 NRMSE/SER、亮点掩膜内相位误差、PSLR/ISLR、位置/分辨率、强弱目标共存结果。若有位移/干涉应用，追加相干系数与相位偏差。应分别报告未对齐误差和去除单一全局相位后的误差，不能只挑更漂亮的一项。

## 6. 五个可证伪的研究假设

### H1：相干误差约束的 FFBP 执行树分割与融合（首选主线）

**假设**：在给定复数误差/旁瓣/相位阈值下，对 FFBP 执行树进行有误差约束的分割，联合决定哪些层保留中间复图、哪些层融合、哪些子树改为 direct，能在未见轨迹和 ROI 上获得比固定 direct/FFBP、按运算次数停止及通用调参更短的 `T_resident`。

执行树的候选动作包括：direct 叶孔径、停止层、相邻层融合、按同层形状/足迹分桶批处理、插值阶数、各阶段精度及 tile。误差由叶到根传播，成本同时计入几何/相位指令、插值访问、中间子图 DRAM 读写、共享内存 halo、寄存器压力、重算和 launch。简单 Graph replay 或批处理通常不改变数学算子；多层融合也必须保留原网格及完整依赖 halo，不能因跳过原插值悄然改变算子后仍称完全等价。

一个可研究的组合形式是：若节点 `y_v=Σ_c B_vc y_c`，近似算子与状态为 `B̂_vc, ŷ_c`，可界定

    e_v ≤ Σ_c (||B_vc|| e_c + ||B̂_vc-B_vc|| ||ŷ_c||) + ρ_v.

这里 `ρ_v` 是本节点有限精度累加误差，`B` 包括插值与相位校正。单位模相位旋转不放大精确向量范数，而插值算子的范数与权重、支撑有关；不能无依据把所有层误差按相同系数相加。研究难点是让可计算的界足够紧、足够便宜，并区分已证明上界与经验预测。不能先用 direct 参考图遍历误差，再把该 oracle 当作运行时能力。

**对照**：当前全部实现、Flexible & Seamless 策略、MatX/GOTCHA 同质量强内核；无融合/同层 batch/Graph replay/两层融合分别消融。把 MatX 视作可选强叶内核而非刻意削弱的竞品。Kernel Tuner 应能搜索相同合法动作和精度空间，在相同评测预算下比较黑箱搜索与物理模型引导的选择；若只给通用 tuner 更少变量，对照不公平。评测驻留时间、质量违约率、相对穷举可行最优的 regret、准备成本和复用次数。留出轨迹/场景/载频做测试。

**失败条件**：离开训练场景就违约；误差界过松导致总选 FP64；融合的 halo/重算/寄存器压力吃掉收益；只需一个固定融合配置就已同样好；准备成本不能摊销；不能超过同质量强 direct；只胜全 FP64。现有 1022 次 launch 是计数事实，尚不足以证明融合一定有效。

### H2：几何足迹驱动的 GPU 内部数据复用（驻留计算支线）

**假设**：按像素块与脉冲块的几何距离足迹设计预取、排列和共享内存缓存，能在不改变合法采样及相位计算的条件下，减少 BP 的实际 DRAM/L2 流量或访存等待，并降低 `T_resident`。这个假设必须由计数器和驻留计时证实，输入裁剪比例不构成证据。

**对照**：未重排的优化 gather、相同布局仅重排像素次序、公共/逐脉冲窗口布局、显式 shared 缓存、cp.async 双缓冲。分别改变像素 tile、脉冲块、插值支撑和坐标非规则程度。所有方案使用同一已驻留输入，逐次 GPU 重排和初始化计入总时间。另报首次 GPU 缓存状态与稳态重复状态，避免只测特别有利的 L2 热缓存。

**论文价值**：普通 shared memory、缓存或 block 调优偏工程；有价值的增量是将几何足迹与误差允许的 tile 尺寸同时纳入决策，并解释什么时候强缓存 gather 胜过显式预取、什么时候融合 FFBP 减少中间图流量。可以成为 H1 的驻留成本模型和高效内核部分。

**失败条件**：L1/L2 已吸收大部分复用；主要瓶颈是 FP64/SFU/依赖链；距离窗过宽；重排/同步开销大；共享内存和寄存器用量降低 occupancy；DRAM 字节虽下降但驻留总时间不降。若失败，应保留强 gather 路径，不强行把 IO 优化作为论文贡献。

### H3：带残差控制的局部低秩算子 + Tensor Core（高风险、高上限）

**假设**：对子孔径×局部像素块消除参考相位后，残余传播/插值算子在指定误差下具有足够低的秩，能用批量 GEMM 替代主要 BP 工作，同时构造与补偿成本可摊销。

单纯几何 dot 的内积长度只有 3，往往太瘦；sqrt/sincos/gather 仍留在 CUDA core 上。更合理的是把局部展开、低秩基或多个通道/频率/帧放到 GEMM。对一帧、一通道、当前小 ROI 不应预设获益。

**对照**：相同误差的 direct、FFBP、局部低秩 FP32 GEMM、低精度无补偿/有补偿 GEMM、Butterfly 风格方法。分别报告可复用建模时间、完整驻留应用时间、重复 k 帧摊销、秩随孔径/近场曲率/带宽变化以及总存储；每帧需要的 GPU 格式转换、布局重排与残差修正必须计入应用时间。

**失败条件**：秩逼近块大小；一次使用无法摊销；精度恢复吞掉 Tensor Core 收益；数据搬运和中间矩阵流量抵消算力；高动态范围场景恶化。

### H4：曲率残差受控的 NUFFT/直接 BP 混合算子（偏算法路线）

**假设**：把局部近似可 Fourier 化的相位交给 NUFFT，把大曲率块交给 direct/FFBP，能在不规则轨迹下以可量化相位误差实现更优时间—质量曲线。

type-3 可以处理非均匀源和非均匀频率，但不能自动解决球面距离的非双线性相位。必须明确局部线性化/高阶残差、振幅权重、波数坐标及适用几何。全局远场场景应与已有 PFA/RMA/NUFFT 比，不能只挑 BP。

**对照**：纯 NUFFT/已有频域成像、全 direct、FFBP、固定块大小混合。从相同驻留数据出发，把每次必要的 spreading、FFT、残差补偿和重排计入 `T_resident`，plan 等可复用成本另列。若用于迭代重建，做复数内积伴随测试 `〈Ax,y〉≈〈x,A* y〉`，并报告重建收敛而非只看单次输出。

**失败条件**：大量块落入 direct；残差补偿高阶开销太大；频率/空间插值重复；几何近似误差超出预算；条件更简单的频域算法已经更快更准。

### H5：通信与相干误差联合约束的多 GPU/超显存 BP（后续扩展）

**假设**：按像素和脉冲的二维分解，根据输入复制成本、复图归约成本与互连带宽选择分区，能在给定相干误差下优于固定像素或固定孔径切分。

像素切分需要复制相关回波，但输出通常无需求和；脉冲切分省输入复制，却需要复数图像归约。局部距离窗使复制量随几何变化；分层归约/通信低精度又引入误差。将它们统一建模才有研究含量。

**对照**：1/2/4/8 GPU 强弱扩展，PCIe 与 NVLink 分开；以数据驻留后的完整分布式计算为主指标，计入每次必要的跨卡交换、GPU 重排和归约，初始化/输入分发另报；无压缩/低精度通信分别测。严格控制更改归约顺序引起的误差。

**失败条件**：小 ROI 被通信启动支配；只有高带宽机器上成立；多卡速度来自增加总显存而不是算法；没有足够硬件验证。当前 4070 单卡应先做 H1/H2，不必立即铺开此方向。

## 7. 建议的先导顺序与停止规则

1. 统一 direct FP64 参考，重新计算所有现有复图误差；FFBP FP32 对 FFBP FP64 的误差不能代替其对 direct 参考的算法总误差。
2. 固定数据驻留的计时边界，用 Nsight Systems 看 launch 间隙、依赖和同步，用 Nsight Compute 抽样典型 direct/FFBP 内核：FP64/SFU 指令、DRAM bytes、L2 sectors、warp stall、寄存器/shared 和 occupancy。测量温度/功耗状态及重复分布。先确定是哪种资源制约驻留时间，再决定优化方向。
3. 在完全相同数学流程下做 FFBP 同层 batch、Graph replay 和小范围层间融合三个独立消融；区分减少 host 提交、增加并行度、减少中间图 DRAM 读写各自贡献。若要融合，先计算多层插值依赖 halo，保留原采样/插值，单测精度不变性和额外重算量。
4. 对强 direct 路径评估几何足迹指导的 shared/cp.async，但先测原有 L1/L2 实际复用。任何每次 GPU 重排成本均计入；不将 26 MB 窗口与 2 GB 分配体积比值当作收益。
5. 小规模扫 `ROI×脉冲数×叶大小×融合层数×精度×插值`，画同误差下的驻留交叉点与 Pareto 曲线。若 direct FP32 一直占优，研究其成立范围；不预设 FFBP 必须获胜。再决定发展 H1 的执行树计划器或 H3/H4 的算子重构。

## 8. 相对 MatX 与 Kernel Tuner 的最强贡献判断

**仅把 MatX 的多个精度 kernel 和 Kernel Tuner 拼接成参数搜索，不足以形成强新意。** MatX 类实现应提供同质量的强 leaf kernel；通用 tuner 应能访问相同合法配置。潜在最强贡献是一个受物理误差约束的执行树变换方法：

- 给 FFBP 每个节点定义算子误差、复图误差和可计算的数据依赖足迹，并证明/验证从叶到根如何组合；分别处理几何/相位误差、插值误差、浮点累加，而非只用最终幅度 SSIM。
- 在误差约束下联合决定算法树如何截断、分割与融合，使中间复图尽量停留于寄存器/shared，同时控制 halo、重算及 occupancy；解释 FLOP 更少为何仍可能更慢，并预测这种交叉点。
- 给出无需在目标场景先算完整 direct 参考的快速决策机制，在未见轨迹和硬件上验证误差违约率、驻留时间及选择 regret；与相同搜索空间/预算的黑箱搜索、固定最优配置和强 direct 比较。

这仍是**待验证研究命题**，不是已经证明的新算法。第一阶段最关键的证据不是加速倍数，而是：统一误差约束后确实存在多种最优树结构；融合在部分区域有效且成本模型能预测失败边界；误差估计足够紧且无需昂贵参考。如果最优策略在所有实测场景都退化成同一个 direct FP32 kernel，或普通 tuner 在相同预算下效果一样，则应降低该主线的创新预期，转向相位算子/低秩结构的 H3/H4。

目标期刊/会议的选择应由贡献决定：相干成像误差理论、非规则轨迹与实测质量偏 TGRS；可推广的 GPU 系统模型、跨架构实现和强可复现基准偏 TPDS/ICS/SC。以上是研究定位建议，不能保证录用；首先应完成最小先导实验来排除无效方向。
