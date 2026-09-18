# 算法资料与开源实现检索

检索日期：2026-09-10。只用原始论文、研究机构页面、官方文档和代码作者仓库作为技术依据。实测数据不上传到这些网站。

| 来源 | 内容与本项目的关系 | 采用方式 |
|---|---|---|
| Ulander, Hellsten, Stenström, **Synthetic-aperture radar processing using fast factorized back-projection**, IEEE TAES 39(3), 760–776 (2003), DOI: [10.1109/TAES.2003.1238734](https://doi.org/10.1109/TAES.2003.1238734)；[Chalmers 作者机构记录](https://research.chalmers.se/en/publication/6728) | FFBP 经典论文，孔径递归分解的理论来源 | 确认方法类别；机构页面并非论文全文，未声称已读付费全文 |
| Góes et al., **Spiral SAR Imaging with Fast Factorized Back-Projection: A Phase Error Analysis**, Sensors 21, 5099 (2021), [论文全文](https://www.mdpi.com/1424-8220/21/15/5099) | 真实飞行数据的 FFBP、相位误差与直接 BP 对比；[配套代码 DOI](https://doi.org/10.5281/zenodo.4883259) | 采用“直接 BP 作为复数精度参照”的验证方法；配套 Zenodo 页面未成功读取，未使用其代码 |
| [MathWorks：Stripmap Synthetic Aperture Radar Image Formation](https://www.mathworks.com/help/radar/ug/stripmap-synthetic-aperture-radar-sar-image-formation.html) | 脉冲 LFM 数据先做匹配滤波，再按传播距离反投影；示例自身说明 BP 是近似形式 | 参考处理顺序；本数据的相位符号由实测对照确定，不能照搬示例 |
| [Ttl/torchbp](https://github.com/Ttl/torchbp)；[FFBP 说明](https://ttl.github.io/torchbp/examples/ffbp.html)；[极坐标插值](https://ttl.github.io/torchbp/examples/polar_interp.html) | 局部极坐标子孔径图、相位中心变化、复数插值与二叉合并；CPU/CUDA 实现 | 参考其公开算法结构和相位参考机制；独立实现 NumPy/Numba CPU 版本，不依赖 torchbp 编译扩展 |
| [dm6718/RITSAR](https://github.com/dm6718/RITSAR)；[imgTools.py](https://github.com/dm6718/RITSAR/blob/master/ritsar/imgTools.py) | 包含数字聚束、FFBP 和多进程 FFBP；相位历史输入与本项目原始脉冲 IQ 格式不同 | 阅读代码比较算法路线；未直接调用，也未把 FMCW/频域输入约定用于脉冲原始数据 |
| [tbensonatl/gotcha-back](https://github.com/tbensonatl/gotcha-back) | GOTCHA 数据的 CPU/CUDA BP、性能和信号误差对照 | 比较 GPU 加速直接 BP 与真正孔径分解 FFBP；未宣称本程序使用该 CUDA 实现 |
| [likemoongg/SARbackprojection](https://github.com/likemoongg/SARbackprojection) | MATLAB BP 教学和点目标例程 | 辅助对照；最终验证用独立生成的点散射体信号 |
| [Sandia：A comparison of spotlight synthetic aperture radar image formation techniques](https://www.sandia.gov/research/publications/details/a-comparison-of-spotlight-synthetic-aperture-radar-image-formation-techniqu-1996-10-01/) | BP 与其他成像算法的图像质量、计算负担对比 | 背景阅读；没有借用文献速度作为本机实测速度 |
| [Sandia：The phase gradient autofocus algorithm](https://www.sandia.gov/research/publications/details/the-phase-gradient-autofocus-algorithm-an-optimal-estimator-of-the-phase-de-1989-09-01/)；[torchbp 作者对 GPGA 和轨迹估计的说明](https://hforsten.com/synthetic-aperture-radar-autofocus-and-calibration.html) | 从强散射点相位梯度估计残余相位，并用多方向信息拟合轨迹修正 | 独立实现带低通、权重限制和稳健最小二乘的二维修正；以模拟已知误差验证符号，再在实测中选择实际评估过的迭代 |

`references/` 保存实际读取的部分源文件快照、许可文件、来源 URL 和 SHA-256，见 `references/repositories.json`。GitHub 匿名 API 当时触发速率限制，因此不伪造提交号；快照按内容散列标识。上述仓库源码仅供研究对照，不参与本程序执行。`sar_io.py`、`ffbp.py` 等为本次独立实现。

## 本实现的关键处理

1. 从 `.dat` 按真实帧头验证并读取交错 int16 IQ；每个有效脉冲保留原始帧号和时间。
2. 对负调频 LFM 进行线性相关匹配滤波。FFT 长度覆盖线性卷积，明确消除匹配滤波器群延迟；输出 k 定位接收脉冲中心，再按原项目 `t0-Tp/2` 的发射时刻约定换算距离，即 `c*(sampling_delay-Tp/2+k/fs)/2`。
3. 用记录的 ENU 速度积分得到连续轨迹，避免将重复的 100 Hz 导航位置当成 1 kHz 连续位置。18 m 高度来自项目 `raw_cs.m`；其作为初始离地高度假设，保留垂向变化。
4. 在每个短子孔径的局部极坐标网格，按每一像素到每一雷达位置的三维斜距对距离压缩回波做复数插值和相位补偿。
5. 令 `k=phase_sign*4*pi*fc/c`。子图存储 `J_s(p)=I_s(p)*exp(-j*k*R_s(p))`，以去掉径向载频。父图通过 `sum(interp(J_s,p)*exp(j*k*(R_s(p)-R_parent(p))))` 相干合并。
6. 每级按孔径长度提高角度采样密度；所有子图含边界保护样本，先合并复数值，最后取幅度。
7. 地面图采用平地假设，并提供物理距离轴、复数 MAT 数据和未增强的 dB 灰度图。未做生成式补图、锐化或地理配准。

本实现选用四点 Lagrange 复数插值和明确的内部过采样；torchbp 当前例程使用 Knab 插值等优化，两者不应混称为相同实现。插值误差通过模拟点目标和实测直接 BP 对照量化。
