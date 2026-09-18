# SAR BP 的 GPU 加速文献检索记录

检索日期：2026-09-17。这里的 BP 指 radar backprojection，不是神经网络 backpropagation。检索以 IEEE/ACM 出版记录、期刊出版页、作者大学/机构存档及 NVIDIA 官方文档为依据。数字只保留查到的原文结果；“全文可得”不等于本次逐页精读。若本次只访问摘要则明确注明。不同数据、精度、任务规模与硬件之间的 speedup 不能直接排名。

## 可核验的核心 GPU 论文（17 项）

| 年份 / 论文 | 来源与访问层次 | 已核验的方法 / 数字 | 对选题的意义 |
|---|---|---|---|
| 2010，Ahmed Fasih, Timothy Hartley，**GPU-accelerated synthetic aperture radar backprojection in CUDA**，IEEE Radar Conference | [DOI](https://doi.org/10.1109/RADAR.2010.5494395)；[OSUBP 项目关联论文](https://github.com/bambang/osubp)。IEEE 页面直接访问失败；摘要和作者代码项目可见，没有核验完整计时表 | 单 GPU CUDA BP；输入超显存时沿 range/aperture 分块；公开数据含变化的起始频率。没有在这里给未核验的 speedup | 普通像素并行、纹理插值、分块不是新问题；有老代码基线可追溯 |
| 2012，A. Capozzoli 等，**NUFFT-based SAR backprojection on multiple GPUs**，TyWRRS | [IEEE/DOI](https://doi.org/10.1109/TYWRRS.2012.6381104)，读到官方摘要/引言，其余需登录 | CUDA + NUFFT + 多 GPU；Jazz 双 GPU 节点比单 GPU 时间约减半；AFRL circular 数据整体处理约 19 s。摘要未完整给出规模，不能与现代卡直接对比 | “NUFFT 与 GPU 合用”“多 GPU BP”都已有明确先例 |
| 2013，A. Breglia, A. Capozzoli, C. Curcio, A. Liseno，**GPU-accelerated fast and fast factorized SAR backprojections**，ACES 2013，pp.827–832 | [那不勒斯大学存档](https://www.iris.unina.it/handle/11588/564392)，本次只读取摘要 | CUDA FBP / FFBP；底层子图用 1D NUFFT，比较 bilinear / bicubic 子图插值。未核验性能数字 | FFBP 多级结构搬到 GPU 不是空白；应比较质量受约束后的性能 |
| 2014，Craig Stringham, David G. Long，**GPU Processing for UAS-Based LFM-CW Stripmap SAR**，PERS 80(12), 1107–1115 | [作者 BYU 全文](https://www.scp.byu.edu/mers/long/papers/ASPRS2014_Stringham.pdf)；DOI 10.14358/PERS.80.12.1107；读到全文段落，未完整提取计时表 | UAS stripmap BP，考虑脉内运动与移动天线方向图，以 CASIE 2009 数据验证实时成像；GPU texture 插值精度也是讨论内容 | 利用天线有效照射支撑裁剪 pulse–pixel 对，以及 FMCW 运动补偿，都应做 prior-art 对照 |
| 2014，Othmar Frey, Charles L. Werner, Urs Wegmüller，**GPU-based parallelized time-domain back-projection processing for agile SAR platforms**，IGARSS，1132–1135 | [作者机构全文](https://www.gamma-rs.ch/files/instruments/literature/SAR-systems/pdf/freyWernerWegmullerIGARSS2014GPUbasedTDBP.pdf)；[ETH 书目](https://ifu-sar.ethz.ch/otfrey/SARbibliography/keyword/GPU-BASED-PARALLELIZED-TDBP.html)；DOI 10.1109/IGARSS.2014.6946629 | CUDA TDBP，用车载非线性轨迹数据；预处理维持常规 slant-range/azimuth SLC 几何同时做完整 3D motion compensation。未核验精确性能表 | 非规则轨迹本身并不自动构成 GPU 优化创新；复杂几何下性能与相位真实性共同验证才有价值 |
| 2017，M. Wielage 等，**High Performance and Low Power Architectures: GPU vs. FPGA for Fast Factorized Backprojection**，CANDAR，351–357 | [汉诺威大学正式记录](https://research.uni-hannover.de/de/publications/high-performance-and-low-power-architectures-gpu-vs-fpga-for-fast/)，DOI 10.1109/CANDAR.2017.101；本次读到元数据，未获完整性能表 | GPU/FPGA FFBP 性能与功耗比较。不能从后续论文转引填成自己的全文结果 | 能效和低精度 FFBP 也有较早先例；工程论文需公平 energy/image 对比 |
| 2021，**An Improved Back-Projection Algorithm for GNSS-R BSAR Imaging Based on CPU and GPU Platform**，Remote Sensing 13(11), 2107 | [出版社全文](https://www.mdpi.com/2072-4292/13/11/2107)，DOI 10.3390/rs13112107；读取方法和实验表 2 | GNSS-R 双基地 BP，CPU/GPU 流式处理。4 km×4 km 仿真：CPU 17456.88 s、GPU 106.468 s，163.96×；BP 阶段 164.17×。作者称卡为“GeForce GTX 3090”，命名疑有误，不宜默默当成完全核实的规格 | CPU→GPU 大 speedup 并不能证明超越先进 GPU BP；需把主张限定到本文基线 |
| 2021，Seokwon Lee 等，**Architecture Exploration of a Backprojection Algorithm for Real-Time Video SAR**，Sensors 21(24), 8258 | [出版社全文](https://www.mdpi.com/1424-8220/21/24/8258)，DOI 10.3390/s21248258；读取流水线与实验结论 | 全/子孔径共六种架构、重叠帧复用、多 stream、多 GPU。4096²，0.5 m cross-range：单 GPU 23.0 Hz，四 GPU 73.5 Hz。应补齐具体 GPU、overlap、脉冲数再复现 | 只做滑窗加减 / 子孔径缓存 / stream overlap / 多卡缩放，难声称首创 |
| 2022，Yihao Xu 等，**The Adaptive Streaming SAR Back-Projection Algorithm Based on Half-Precision in GPU**，Electronics 11(18),2807 | [出版社全文](https://www.mdpi.com/2079-9292/11/18/2807)，DOI 10.3390/electronics11182807；读取方法和表 6–8 | 自适应缩放、FP16 mixed precision、异步 streaming，扩展到 FBP。表 7 BP total：FP32 2295 ms→FP16 1274 ms（约1.80×），FBP 548→415 ms（约1.32×）。显存 BP 393.65→292.24 MB；FBP1125.49→663.87 MB。表6 BP PSNR 49.9150→44.8880 dB、MSSIM0.9986→0.9940 | 已有“低精度+动态缩放+异步流”的直接先例；其速度并非无损，需要报告相位/弱目标误差，而不能只给肉眼图 |
| online 2022 / 卷年 2023，Hao Gong 等，**Scene optimization of GPU-based back-projection algorithm**，Journal of Supercomputing 79,4192–4214 | [Springer 出版页](https://link.springer.com/article/10.1007/s11227-022-04785-w)，DOI 10.1007/s11227-022-04785-w；本次官方摘要与元数据，未读付费正文 | 针对服务器和微型 GPU、不同数据规模的 heuristic block 策略；unified/pinned memory；摘要宣称接近设备 peak performance，无已核验具体数字 | 数据/硬件自适应分块与内存策略已有人做；新调优器需超过简单启发式并提供可解释代价模型 |
| 2023，Marcel Hoffmann 等，**Implementation of Real-Time Automotive SAR Imaging**，EuRAD 2023 | [FAU 正式出版记录](https://cris.fau.de/publications/313307079/)，DOI 10.23919/EuRAD58043.2023.10289475；[开放作者稿](https://arxiv.org/html/2306.09784v1)，本次读取方法与表格 | GTX1080Ti 18.8 ms，Quadro RTX3000 Mobile 28.4 ms，30m×30m；六项措施含 GPU FP32、窗权、常量/索引、Doppler简化、PSF匹配网格、减少孔径/接收通道；摘要称合成8192 chirps | 这是算法/工作量联合改变，不是同一完整问题的纯 kernel 加速；“自适应网格”有直接近邻工作。原始基准笛卡尔网格1201²，绝不能把优化时延等同1201²完整BP |
| 2023，Ziya Li 等，**An Efficient BP Algorithm Based on TSU-ICSI Combined with GPU Parallel Computing**，Remote Sensing 15(23),5529 | [出版社全文](https://www.mdpi.com/2072-4292/15/23/5529)，DOI 10.3390/rs15235529；读取方法与表4/5 | 时间移位上采样+改进三次样条；Quadro RTX8000。6000² 图：机载输入10000×8000，TSU-ICSI 152 s，对16/32点 sinc477/848 s；星载输入10000×57600，223 s，对1126/2150 s。同TSU-ICSI新GPU映射152 s vs conventional273 s，约1.80× | 插值本身能有比线程微调更大的收益；应同图像质量比较，不应把8点/32点计算差异归因纯GPU优化 |
| 2024，Jiongming Chen 等，**A Fast Inverse Synthetic Aperture Radar Imaging Scheme Combining GPU-Accelerated Shooting and Bouncing Ray and Back Projection Algorithm under Wide Bandwidths and Angles**，Electronics 13(15),3062 | [出版社全文](https://www.mdpi.com/2079-9292/13/15/3062)，DOI10.3390/electronics13153062；读取SBR/BP方法章节，未核验完整性能表 | BVH 加速 SBR 电磁散射仿真 + CUDA BP / cuFFT，宽带宽角 ISAR。此项是 ISAR/模拟回波近邻，不是与实测 SAR 直接等价 | RT/BVH 对回波仿真的价值与 BP gather 核的价值应分开；不可见有 RTX 卡就推断 BP 用了 RT Core |
| 2025，Sergi Hernández-Burgos 等，**Accelerated Algorithm for Back-Projection Fully Focused SAR Altimetry**，IEEE TGRS63,5209816 | [UPC 官方开放存档](https://upcommons.upc.edu/handle/2117/429000)，[开放全文PDF](https://upcommons.upc.edu/server/api/core/bitstreams/03c53060-b955-45ab-9ce3-88e70e515104/content)，DOI10.1109/TGRS.2025.3556544；访问官方摘要及PDF检索段落 | CPU算法28×；同GPU accelerated BP相对classic BP 8–13×；CPU classic→GPU accelerated最高1570×。真实 transponder/open-ocean 数据。full processor含读写+CPU其他阶段时不能沿用单BP巨大倍率 | 很重要的TGRS路线示范：算法结构创新+同卡先进基线+真实科学应用+精度验证，比只报CUDA提速强。测高几何与二维SAR不同，复用需重新推导 |
| 2025，Masato Gocho, Masayoshi Tsuchida, Kazunori Ueda，**Mixed-Precision Backprojection Using FP32 and FP64 for Boosting Satellite SAR Imaging on Embedded GPUs**，IEEE MCSoC2025,614–621 | [早稻田大学官方记录](https://waseda.elsevierpure.com/en/publications/mixed-precision-backprojection-using-fp32-and-fp64-for-boosting-s/)，DOI10.1109/MCSoC67473.2025.00101；本次摘要，未获正文 | FP64算代表位置base距离，FP32算其余offset；AGX Orin FFBP 25.8s，比FP64 FFBP快10.3×；32768×24000输出，32768×33792输入 | “参考距离双精度+残差单精度”已有2025直接论文；局部坐标/残差精度工作必须明确越过它 |
| 2025，雷苏力等，**基于NVIDIA GPU的高轨SAR快速BP算法子孔径成像CUDA设计与实现**，空间电子技术22(3),54–59 | [期刊官方页](https://www.spacejournal.cn/cn/article/id/7430e60b-e38b-44df-b234-f3c3a47e1ed3)，DOI10.3969/j.issn.1674-7135.2025.03.008；本次摘要，官网提供PDF但未精读 | FFBP子孔径CUDA：stream隐藏分块传输、细粒度线程；相对CPU32线程120×，作者报设备执行效率>90%；未核验指标定义/硬件/完整规模 | CPU多线程比较优于串行但仍不足以支持顶级GPU论文；利用率不是性能充分条件 |
| 2026，Masato Gocho, Isao Matsushima, Ryuhei Takahashi，**Study of GPU-based parallel backprojection for 3D mmWave SAR imaging**，IEICE SANE2025-118，pp100–103，2026-02-20报告 | [IEICE 官方摘要](https://www.ieice.org/publications/ken/summary.php?contribution_id=141675&expandable=1&ken_id=AN&lang=en&presen_date=2026-02-20&schedule_id=8854&society_cd=CS&year=2026)，技术报告，勿写成顶刊；本次摘要 | 四个传感器、1281 hits、12×16 elements；64-core Xeon+4 H100，摘要显示“2563 pixels”（很可能上标丢失，未做256³无保留确认）；重建3695ms，经CPU预处理/GPU主处理流水线2411ms | 2026尚有大规模3D流水线工程工作；未达到检索穷尽程度，不能说这是2026全部最新成果 |

## 与“顶会”目标特别相关的邻近工作

**Park 等，Efficient Backprojection-based Synthetic Aperture Radar Computation with Many-core Processors**：SC2012，DOI10.1109/SC.2012.53；扩展版 Scientific Programming2013，DOI10.3233/SPR-130372。[作者大学PDF](https://web.eecs.umich.edu/~msmelyan/papers/efficient-2012.pdf)、[期刊正式页](https://onlinelibrary.wiley.com/doi/10.3233/SPR-130372)。approximate strength reduction、locality优化、宽向量多核；单节点 Xeon E5-2670 + Xeon Phi 超35GBP/s、3K²图约1秒。它是**Xeon Phi工作而非GPU论文**，但已证明BP题目可以进入SC：创新中心是可迁移计算方法与跨层性能分析。不能把它错误列为CUDA成果。

**A Review of Synthetic-Aperture Radar Image Formation Algorithms and Implementations: A Computational Perspective**，Remote Sensing2022，14(5),1258，[出版社全文](https://www.mdpi.com/2072-4292/14/5/1258)。适合跟踪经典实现与precision实验的入口，不能用综述中的二手倍率替代原论文。

**Synthetic Aperture Radar Processing Using Flexible and Seamless Factorized Back-Projection**，Remote Sensing2025，17(6),1046，[出版社全文](https://www.mdpi.com/2072-4292/17/6/1046)。按计算代价在FFBP合适级别停止factorization，低空平台变分辨率条件更有收益；这是算法工作，未确认GPU实现。提示“混合BP/FFBP调度”也不是完全空白。

**An interpolation-free FFBP algorithm for spotlight SAR processing**，IGARSS2015，DOI10.1109/IGARSS.2015.7326827，[IEEE摘要](https://doi.org/10.1109/IGARSS.2015.7326827)：子孔径融合用chirp-z与循环移位替代2D插值，受适用条件限制。需要与低秩/FFT加速主张比较。

排除项：2026 **Quantitative Assessment of the Computing Performance for the Parallel Implementation of a Time-Domain Airborne SAR Raw Data Focusing Procedure**，[出版社](https://www.mdpi.com/2072-4292/18/2/221)，DOI10.3390/rs18020221，当前是CPU/IDL多job研究，正文将GPU迁移列为未来工作，不能列作GPU加速已完成实例。

## 必须关注的官方代码现状（不是论文）

1. **NVIDIA MatX experimental::sar_bp**，[官方文档](https://nvidia.github.io/MatX/api/signalimage/radar/sar_bp.html)。当前页面已经有 Double / Mixed / FloatFloat / TaylorFast / Float；Mixed保留range高精度而插值累加FP32；FloatFloat用两个FP32承载更高精度；PhaseLUTOptimization预存参考相位并在range bin内计算小量增量相位；还支持z=0/固定平面特化。**把mixed precision、phase LUT、局部Taylor当全新贡献风险非常高。**本次文档已核验，尚未在本地编译或完整审计源码。
2. **tbensonatl/gotcha-back**，[仓库](https://github.com/tbensonatl/gotcha-back)。FP64、mixed、incremental phase、Newton-Raphson sqrt、shared-memory reference range、texture interpolation、FP32多个核；GBP/s和SER衡量性能质量。优化部分追溯至2013 GTC。可作为公开标准比较基线，不等于现代硬件绝对SOTA。
3. **OSUBP**，[仓库](https://github.com/bambang/osubp)。与2010 IEEE Radar/2009 Cluster工作相连的早期PyCUDA代码；依赖陈旧，需要移植，属于历史可追溯基线。

## 对科研定位的直接判断

- 这些单点已明显拥挤：直接CUDA化、图像tile、纹理线性插值、pinned memory/stream overlap、FP16/FP32混合、base+offset、相位增量/查表、子孔径复用、普通多GPU拆图、PSF网格减量。
- 可以形成新研究的问题应是**精度约束下的组合选择规律与可证明误差**：给定轨迹/载频/口径/距离/PSLR-ISLR/复相位误差门槛，联合选择分块、Taylor阶数、精度、插值与FP64回退，而非列多个开关。需对MatX、gotcha-back、TSU-ICSI、2025星载mixed precision逐项消融。
- Tensor Core BP：本轮“backprojection/SAR/tensor core(s)/TensorCore”检索没有得到可核验的直接SAR BP Tensor Core论文；**这只能说本轮未找到，不能说无人做过。**BP常见核是几何距离+非均匀gather+相位+复累加，不自然等于GEMM。若要用Tensor Core，必须由局部相位可分离/低秩展开获得足够复用，并把建表、打包、额外采样、精度恢复成本算全；可作为高风险验证项，不能先许诺收益。
- “全局误差控制下、按几何结构自动生成最优kernel”的科学性，可能高于“再优化10%”：需要证明误差上界随tile尺寸/载频/残差范围变化，并证明选择策略跨consumer/embedded/datacenter GPU适用。
- 发表选择不能由题目保证。TGRS路径须把真实雷达成像科学与复杂几何精度作为核心；SC/PPoPP/ICS路径则需一般性的非规则复振荡求和优化、扎实性能模型、跨场景/跨架构验证及可复现代码。SC2012先例给的是要求，不是录用承诺。

## 复现与比较时容易踩的坑

1. 将CPU串行→GPU千倍写成先进GPU方法千倍。
2. 将修改网格/抽脉冲带来的工作量下降计作相同问题加速。
3. 只比幅值PSNR/SSIM：复相位错误可能破坏InSAR、相干变化检测、弱目标，即使幅度图正常。
4. FFBP/直接BP用同一个“GBP/s”比较却未说明实际kernel工作量与原始等效工作量。
5. BP kernel时间与读盘、上采样、H2D、最终D2H总时间混用。
6. tensor-core/RT-core/FP16的理论峰值与BP实际有效吞吐直接画等号。

## 补查：与拟议 planner 主线的直接重叠

**18. Fast and Energy Efficient Image Formation of SAR Satellite Data using Embedded GPUs**，Michael Fritzenwallner、Andreas Stelzer、Reinhard Feger，IRS2025，DOI10.23919/IRS64527.2025.11046049。[IRS官方会议程序](https://www.dgon-irs.org/fileadmin/documents/IRS_2025_Programm_PDF_Web.pdf)确认标题、作者和报告；[Kernel Tuner软件记录](https://research-software-directory.org/software/kernel-tuner)列为mention，支持它与该工具存在引用关联，**不足以确认具体调参变量/收益**。本次IEEE访问无正文，也未找到开放作者稿，因此性能数字和方法细节留空。JST机器翻译摘要提到差分距离近似使单精度成为可能、DEM、AGX Orin速度和能耗，但这属于二级线索，正式论证应获取原文后核验。

**Spiral SAR Imaging with Fast Factorized Back-Projection: A Phase Error Analysis**，Juliana A. Góes等，Sensors2021，21(15),5099，DOI10.3390/s21155099。[PubMed元数据](https://pubmed.ncbi.nlm.nih.gov/34372349/)、[开放全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC8348341/)。真实P-band无人机螺旋轨迹，以初始几何参数预测FFBP相位误差标准差；约12°条件下3D最高21×、2D最高13×（相对direct BP，**不要当作GPU speedup**）。此前如果材料把它写成其他2021 Sensors题名，应改为本完整标题。

**Flexible & Seamless 的具体边界**：[正式全文](https://www.mdpi.com/2072-4292/17/6/1046)第3.2/3.3节已实现每一级在继续层次融合与直接全局Cartesian融合之间选代价更小者；其代价为complex operations，明确排除了插值kernel成本。它因此是cost-based FFBP-depth选择的**直接前作**，不是可一笔略过的普通FFBP引用。区分空间应落在：联合复相位/PSF误差预算、插值与mixed precision误差传播、实际GPU时延/能耗模型、跨轨迹与跨架构验证。比较基线需含“Flexible & Seamless结构 + 成熟GPU核 + Kernel Tuner”，从而判断贡献是否超过已有组件组合。
