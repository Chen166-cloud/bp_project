"""Collect every evaluation into the single requested Chinese Markdown report."""
from pathlib import Path
import json, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'sar_quality/results'
def read(n):return json.loads((OUT/n).read_text(encoding='utf8'))
def link(path,label=None):
    p=(ROOT/path).resolve().as_posix()
    return f'[{label or Path(path).name}](<{p}>)'
def picture(path,label):return '!'+link(path,label)
def table(head,rows):
    return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])+'\n'
def fmt(x,n=3):return f'{x:.{n}f}'

pt=read('point_targets.json');rois=read('real_rois.json');legacy=read('legacy_outputs.json')
probe=read('legacy_filter_probe.json');dot=read('dot_probe.json');proto=read('protocol.json')
scatter=read('real_scatterers.json');enl=read('enl_proxy.json')
names={'bp_nav':'直接 BP，导航轨迹','ffbp_nav':'FFBP，导航轨迹',
       'bp_autofocus':'直接 BP，自聚焦轨迹','ffbp_autofocus':'FFBP，自聚焦轨迹'}

# Compact overview of the quantitative findings (no image enhancement).
fig,axs=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
r=np.array([v['range_m'] for v in pt])
for key,label,style in [('direct_bp','Direct BP','-o'),('ffbp','FFBP','--s')]:
    axs[0].plot(r,[v['metrics'][key]['ground_range']['irw_m'] for v in pt],style,label=label+' range')
    axs[0].plot(r,[v['metrics'][key]['along_track']['irw_m'] for v in pt],style,label=label+' azimuth',alpha=.7)
axs[0].set(xlabel='Point-target slant range (m)',ylabel='Half-power IRW (m)',title='Ideal raw-LFM point tests');axs[0].legend(fontsize=8)
idx=np.arange(1,7)
dh=[v['metrics']['bp_autofocus']['entropy_nats']-v['metrics']['bp_nav']['entropy_nats'] for v in rois]
di=[100*(v['metrics']['bp_autofocus']['intensity_contrast']/v['metrics']['bp_nav']['intensity_contrast']-1) for v in rois]
axs[1].bar(idx,dh,color='#4267a1');axs[1].set(xlabel='Fixed real-data ROI',ylabel='Entropy change (nats)',title='Autofocus: entropy change');axs[1].axhline(0,color='k',lw=.6)
axs[2].bar(idx,di,color=['#4267a1' if x>=0 else '#c1693b' for x in di]);axs[2].set(xlabel='Fixed real-data ROI',ylabel='Intensity contrast change (%)',title='Autofocus: contrast change');axs[2].axhline(0,color='k',lw=.6)
for ax in axs:ax.grid(alpha=.2)
fig.savefig(OUT/'evaluation_overview.png',dpi=170);plt.close(fig)

text=[]
def add(s):text.append(s.strip()+'\n')
add('''# 项目 SAR 成像算法质量与分辨率评价报告

评价执行：2026-09-10 至 2026-09-11；整理完成：2026-09-11。主要实测数据：`2018-08-14_18-54-08.dat`。

本报告汇总权威资料检索、项目全部成像实现核查、旧输出有效性检查、独立点目标测量、统一实测区域对照及复现方法。数值均来自本项目计算或明确标记的既有测试；论文中的示例值没有当作本项目结果。原始采集文件、旧 MATLAB 成像脚本及既有成像结果均未覆盖。

## 1. 主要结论

1. **当前能够进行可靠数值对照的是 `fast_bp/` 中的直接 BP 与 FFBP。** 在相同实测脉冲、轨迹、窗函数和物理坐标下，两者复数相关性高，但快速近似仍有幅度及旁瓣误差。
2. **FFBP 的理想点目标距离向半功率宽度约为 0.99 m；方位向在 800、1400、2200 m 斜距分别约为 0.569、0.990、1.563 m。** 这些是指定信号模型和完整孔径下的算法测试结果，不能直接认证实测场景的外场分辨率。
3. **FFBP 不是在所有质量指标上完全等同于直接 BP。** 本次点目标实验中距离主瓣展宽约 1%，方位 PSLR 约劣化 1 dB，方位 ISLR 约劣化 1.8–2.0 dB；强散射体附近的弱目标显示仍需注意旁瓣。
4. **自聚焦的改善随区域而变化。** 六个固定区域的图像熵均下降，五个区域强度对比度增加，一个区域略降；部分实测局部峰的方位宽度增加。不能由单一全图分数推导“所有目标聚焦更好”。
5. **旧 RD 输出与 `dot.m` 当前示例均为全零，不能报告有效分辨率。** 旧 BP 的物理采样及成像实现存在问题；旧 CSA 使用不同采集参数或另一份卫星数据，不能与本 DAT 的 FFBP 做直接质量排名。
6. 项目目前没有可核验的角反射器标定、地面真值、噪声标定和完整天线外参。因此不提供没有依据的外场 PSLR/ISLR、NESZ、定位精度、识别准确率或“综合质量百分制”。
''')
add(picture('sar_quality/results/evaluation_overview.png','评价结果概览：理想点目标分辨率与实测区域自聚焦变化'))

add('''## 2. 项目与数据范围

该项目研究 SAR 原始回波读取、距离压缩、运动补偿、距离徙动校正与二维成像，包含 RD、CSA、BP 和快速 BP 的不同实现及试验脚本。`a.m` 是 LFM 压缩示例，`read_rawdata_1.m` 是解码程序；`Madsen.m`、`DoppCentEsti_SLC*.m`、`DopplerCenterEstimation_Eny.m` 等是多普勒估计辅助函数，不是可以独立比较图像质量的完整成像算法。

本次统一实测对照直接使用该 DAT 解码后保存的线性复数距离压缩缓存，并采用全部有效脉冲；没有从 BMP/JPG 反推复数回波。旧 MATLAB 各版本和原输出另行审计，未将它们修改成新算法后仍冠以原实现的名称。
''')
add(table(['项目','核验值或约定'],[
['DAT 大小','1,073,741,824 字节'],['有效 / 无效完整帧槽位','12,757 / 631'],
['脉冲编号','26781–39537；有效部分连续无缺号'],['每脉冲距离采样','20,000 点'],
['载频 / 带宽 / 脉宽','1.4 GHz / 200 MHz / 1 μs'],['采样率 / PRF','400 MHz / 1000 Hz'],
['有效记录时长','12.756 s'],['速度积分轨迹','平均 17.0902 m/s；沿轨孔径 218.0026 m'],
['初始离地高度','18 m；来自项目 raw_cs.m，采用平地模型，非独立标定'],
['地面坐标','x 为运动方向左侧横轨坐标，y 为沿轨坐标；不是经纬度定位结果'],
['成像距离原点','c × (3 μs − 1 μs / 2) / 2 = 374.7406 m'],
['物理距离采样间隔','c/(2 Fs) = 0.374741 m'],
['名义带宽距离尺度','c/(2 B) = 0.749481 m；不是本报告的实测 −3 dB 宽度'],
['统一处理窗','距离匹配参考：Hamming；方位完整孔径：Hann'],
['旧 MAT 尺寸','data_5000_10000.mat：MATLAB 5000 × 10000；HDF5 轴序相反'],
['独立卫星输入','CDdata1.mat：1728 × 3200；RADARSAT.m 使用 5.3 GHz 等另一套参数']]))
add('''方位分辨率取决于有效角孔径和加窗；窄角、近正侧视、固定孔径下可用 λR/(2L) 理解它随距离增加的趋势。本次最终数值通过完整点目标响应测量，不用这一近似式替代。天线方向、硬件群延迟和地形的剩余不确定性会影响真实成像位置及聚焦。
''')

add('''## 3. 权威资料与本次指标定义

检索优先采用 ESA/Copernicus、DLR、Sandia 官方资料、论文原文和作者开源仓库。已经阅读 Sentinel-1 产品定义附录 A、Sandia IPR 报告相关章节、DLR EUSAR 2024 的质量指标部分，并核对关键 PDF 页面；同时查阅 FFBP、熵聚焦和 ENL 论文。经典定义不因出版较早而失效。
''')
sources=[
['S1','ESA：Introduction to a SAR System','https://www.esa.int/Enabling_Support/Space_Engineering_Technology/Onboard_Data_Processing/Introduction_to_a_SAR_System','说明用明亮人工点目标插值分析脉冲响应；区分成像、重采样和地理定位。'],
['S2','Sentinel-1 Product Definition，Issue 2/7，2016，附录 A','https://sentinels.copernicus.eu/documents/247904/1877131/Sentinel-1-Product-Definition.pdf','IRF、PSLR、ISLR、ENL、NESZ 等产品指标及其测量条件。'],
['S3','Doerry，Anatomy of a SAR Impulse Response，SAND2007-5042，2007','https://www.sandia.gov/research/publications/details/anatomy-of-a-sar-impulse-response-2007-08-01/','第 4 节及 4.2 节：半功率宽度、加窗、旁瓣、像素间距与分辨率的区别；DOI 10.2172/920461。'],
['S4','Sigurd Huber，Spaceborne SAR Systems with Digital Beamforming and Reflector Antenna，KIT 博士论文，2014，DLR 存档，附录 G','https://elib.dlr.de/76420/1/Dissertation_Huber.pdf','以主峰两侧首个局部极小值划分主瓣；积分旁瓣 / 主瓣能量。'],
['S5','Fischer、Scheiber、Horn、Kubanek，First Comparative Analysis of Simulated Low and High Resolution Hydroterra and Conventional SAR Data，EUSAR 2024','https://elib.dlr.de/201422/1/Fischer-EUSAR2024-Hydroterra-Postprint.pdf','第 3 节明确给出强度变异系数和复数相干度；强调按场景区域比较。'],
['S6','Refocusing Moving Ship Targets in SAR Images Based on Fast Minimum Entropy Phase Compensation，Sensors 2019，19，1154','https://pmc.ncbi.nlm.nih.gov/articles/PMC6427439/','式 (5)–(6)：归一化强度 Shannon 熵；DOI 10.3390/s19051154。其船目标应用不能直接推广为本场景的分辨率认证。'],
['S7','Góes 等，Spiral SAR Imaging with Fast Factorized Back-Projection: A Phase Error Analysis，Sensors 2021，21，5099','https://pmc.ncbi.nlm.nih.gov/articles/PMC8348341/','FFBP 与 BP 的复数相干度、相位误差及耗时对照；DOI 10.3390/s21155099。'],
['S8','Estimation of Equivalent Number of Looks in Time-Series Pol(In)SAR Data，Remote Sensing 2020，12，2715','https://www.mdpi.com/2072-4292/12/17/2715','ENL 需要均匀区域及散斑统计假设；DOI 10.3390/rs12172715。'],
['S9','Ulander、Hellsten、Stenström，Synthetic-aperture radar processing using fast factorized back-projection，IEEE TAES 2003，39(3)，760–776','https://research.chalmers.se/en/publication/6728','经典 FFBP 论文书目信息；DOI 10.1109/TAES.2003.1238734。本次未取得该论文完整原文，不将未读细节作为具体阈值依据。'],
['S10','tbensonatl/gotcha-back，作者开源 SAR BP 实现','https://github.com/tbensonatl/gotcha-back','补充数值实现对照采用的 SER；开源实现不是雷达产品认证标准。']]
add(table(['编号','来源','本次用途'],[[a,f'[{b}]({c})',d] for a,b,c,d in sources]))
add(r'''### 3.1 点目标分辨率与旁瓣

令复数成像结果为 z，功率/强度为 I = |z|²。IRW 是主瓣功率下降到峰值一半处的全宽，即幅度图上的 −3.0103 dB 宽度，单位为米。用 0.025 m 间隔取剖面，再在线性功率上插值定位交点。这符合 [Sandia IPR 报告](https://www.sandia.gov/research/publications/details/anatomy-of-a-sar-impulse-response-2007-08-01/) 所采用的半功率分辨率口径。

本报告统一采用负 dB 的“旁瓣 / 主瓣”方向：

```text
PSLR = 10 log10[ max(I_side) / max(I_main) ]
ISLR = 10 log10[ ∫side I(x) dx / ∫main I(x) dx ]
```

主瓣边界取主峰两侧首个局部极小值。PSLR 搜索整个保留剖面内最高旁瓣；ISLR 用梯形积分。距离剖面积分范围 ±20 m，方位剖面 ±40 m；具体边界随每条曲线保存。首极小值的 ISLR 口径依据 [DLR Huber 附录 G](https://elib.dlr.de/76420/1/Dissertation_Huber.pdf)，但本报告 PSLR 检查两侧所有保留旁瓣，不限于该论文举例使用的第一右旁瓣。

**这里计算的是一维首极小值口径，不是 Sentinel-1 的二维矩形区域 ISLR。** [Sentinel-1 附录 A](https://sentinels.copernicus.eu/documents/247904/1877131/Sentinel-1-Product-Definition.pdf) 对主区和外围积分区另有规定，其文字还采用主瓣与旁瓣的反向表述。本报告不直接套用该卫星产品的门限。扩大积分窗口通常会改变 ISLR；一维结果也不自动等于二维结果。

PSLR/ISLR 只有在主瓣与旁瓣可分辨的孤立点响应中才可靠。真实多目标、线状结构及杂波中的次峰不能直接当作系统旁瓣。PSLR、ISLR 越负表示按本报告定义测得的旁瓣越低；加窗可能以主瓣展宽为代价降低旁瓣，不能只追求单项最小。

### 3.2 实测图像聚焦质量

在完全相同的 ROI、像素坐标和支持范围上计算：

```text
p_i = I_i / Σ I_i
H = −Σ p_i ln(p_i)                    [nats]
H_normalized = H / ln(N)
IC_intensity = std(I) / mean(I)
IC_amplitude = std(|z|) / mean(|z|)
```

熵公式依据 [Sensors 2019，式 (5)–(6)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6427439/)，强度 IC 依据 [DLR/ESA EUSAR 2024，第 3 节](https://elib.dlr.de/201422/1/Fischer-EUSAR2024-Hydroterra-Postprint.pdf)。为避免文献间将“幅度”和“强度”混用，两个 IC 均保存，主结论使用强度 IC。方差使用总体形式，除以 N。

固定场景的 H 下降、IC 上升常提示能量更加集中，但不是充分的清晰度证明。强散射体、条纹、失真和场景裁剪也会改变这些分数。不同数据、尺寸、窗函数的图像即使 H 做了归一化也不能无条件排名。全零图没有可归一化的能量，明确判为无效，不把“熵为零”误判为最好。

### 3.3 快速算法数值保真度

同一轨迹上的直接 BP 作为数值参考 z_ref，FFBP 为 z_fast：

```text
γ = |Σ z_ref* conj(z_fast)| / sqrt(Σ|z_ref|² Σ|z_fast|²)
ε = ||z_fast − z_ref||₂ / ||z_ref||₂
SER = −20 log10(ε)                    [dB]
phase_error = arg(z_fast conj(z_ref))
```

式中第一行的 `z_ref*` 中星号仅表示乘法语境，准确实现为 `abs(vdot(z_ref, z_fast)) / (norm(z_ref)*norm(z_fast))`，即只对其中一个向量取共轭。

复数相干度及相位对照依据 [DLR/ESA 第 3 节](https://elib.dlr.de/201422/1/Fischer-EUSAR2024-Hydroterra-Postprint.pdf) 和 [Góes 等 FFBP 论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC8348341/)；SER 参照 [GOTCHA 开源 BP 项目](https://github.com/tbensonatl/gotcha-back) 的数值误差用法。ε 和 SER 不做最优增益拟合，保留幅度偏差。相位统计仅使用直接 BP 幅度高于该 ROI 峰值 −30 dB 的像素，扣除圆均值后统计残余相位标准差，阈值样本数单独保存。

高 γ 表示两个数值算法接近，不表示外场图像已经正确聚焦；二者可能共享轨迹、定时或地形误差。SER 是算法近似误差比，不是接收机 SNR。

### 3.4 散斑及其他指标

均匀强度区域的 ENL = mean(I)² / var(I)。[ENL 论文](https://www.mdpi.com/2072-4292/12/17/2715) 说明其估计依赖均匀性、纹理和空间相关性。本报告只对候选均匀区报告条件性 ENL，不把数值更大等同于空间分辨率更高。

没有理想实测参考图像，不将 PSNR/SSIM 冒充成像真值误差；没有辐射定标和噪声参考，不估计 NESZ 或绝对辐射精度；没有已知地面点，不估计绝对定位误差。相关指标及前提可见 [Sentinel-1 附录 A](https://sentinels.copernicus.eu/documents/247904/1877131/Sentinel-1-Product-Definition.pdf)。
''')

add('## 4. 各成像实现的评价')
add(table(['实现','核验结果','质量 / 分辨率结论'],[
['raw_RD_Liu.m','传统 RD，包含 SRC、RCMC 和方位压缩；D 参数的有效多普勒范围未限制，复共轭转置进入方位指数','存在可复现的指数溢出；当前 RD BMP 全零，IRW/PSLR/ISLR 无效。'],
['rd.m','同类 RD，但 RCMC 主循环被注释；仍存在上述滤波数值问题','缺少徙动校正，且当前结果无有效数值依据。'],
['rd2.m','RCMC 并行变体；核心 D/Haz 公式问题仍存在','并行化不自动提高分辨率，也不能消除现有滤波问题；未测得独立有效成像结果。'],
['raw_cs_Liu.m','CSA；设 V=60 m/s、Tr=10 μs，与本 DAT 的约 17.09 m/s、1 μs 不符','不能当作已正确适配本次数据的 CSA，无法给出可信外场分辨率。'],
['raw_cs.m','CSA+运动补偿/PGA 试验；Tp=10 μs；频率轴设为 Kr*tau、−Ka*eta；PGA 输入为量化后的 Img','信号匹配和 FFT 频率坐标需修正；幅度量化后无法恢复原始复数相位。未形成可比的有效复数输出。'],
['RADARSAT.m','使用 CDdata1.mat、5.3 GHz、7062 m/s 等卫星参数；现有 CSA BMP 为 1728×3200','该 BMP 很可能来自此独立卫星链路（尺寸及代码推断，缺运行日志）；不能与 2018 DAT 排名。'],
['raw_bp.m','没有距离压缩与逐像素延时取样；使用固定 data(k,i) 累加；lambda/(2V) 被当作长度','当前实现不满足标准 BP 的传播距离回采模型，参数命名不能证明物理分辨率。'],
['bp2.m','上述简化 BP 的并行变体，保留固定距离列取样问题','并行不会修复成像模型；未保存可核验的独立正确成像输出。'],
['dot.m','单个矩阵 delta 不是完整点目标回波；子图提前取绝对值，平均合并，树只合并到两个子图','实际 MATLAB 运行得到 100×100 全零图；不是有效 FFBP 精度基准。'],
['fast_bp/ffbp.py：直接 BP','对真实三维斜距做复数插值和相干累加，保留复数图','本次统一测试的数值参考；不充当地面真值。'],
['fast_bp/ffbp.py：FFBP','64 脉冲叶节点上限，256 叶、255 次合并，复数重采样及相位补偿','通过独立点目标与实测同坐标对照；有轻微主瓣/旁瓣误差。'],
['fast_bp/autofocus.py','相位梯度驱动的二维有效轨迹修正；继承已选第 1 次迭代','改善多数所测区域，但非所有局部峰；不能等同于真实导航误差标定。']]))
add('上述判断针对仓库的具体实现和当前输出，不是否定 RD、CSA 或 BP 算法本身。受限于无效输出和参数不一致，本次没有伪造所有旧算法在同一场景上的质量排行榜，也没有声称已完整重跑全部旧成像脚本。')
add('### 4.1 RD 数值问题的 MATLAB 核验')
add(f'''从 MAT 元数据读取 Na={probe['Na']}，按原脚本的 V=17 m/s、PRF=1000 Hz 和 D/Haz 表达式独立执行小型探针。物理多普勒截止 2V/λ={probe['cutoff_hz']:.6f} Hz，而原网格覆盖约 ±500 Hz。结果：

- D 出现非实数：{probe['complex_D_count']} / {probe['Na']} 个频率单元。
- Haz 出现非有限数：{probe['nonfinite_Haz_count']} / {probe['Na']} 个频率单元。
- MATLAB 验证零乘这些无穷值会生成 NaN。

原脚本使用 `D'`（共轭转置），在非物理频率区域会把相位项变成指数增长。该问题足以破坏数值成像，是可复现的实现缺陷；仅凭已存在的黑图，仍不能反推出当时每一步运行历史。具体探针：{link('sar_quality/legacy_probe.m')}；日志和计数：{link('sar_quality/results/legacy_filter_probe.json')}。''')
add('### 4.2 旧输出有效性')
add(table(['文件','行 × 列','最小 / 最大码值','不同码值数','零码值比例'],[
 [v['file'],f"{v['shape'][0]} × {v['shape'][1]}",f"{v['min']} / {v['max']}",v['unique_levels'],f"{v['zero_fraction']:.4%}"] for v in legacy['bitmap_audit']]))
b=legacy['legacy_bp'];bm=b['metrics']
add(f'''`SAR_RDA_2.bmp` 的 50,000,000 个像素全部为零。`SAR_CSA.bmp` 的零码值比例约 70.82%，仅表明显示裁剪/黑区占比较大，不能解释为“70.82% 的回波无效”。

`SAR_BP_raw_data.mat` 保留了 200×100 个复数样本，有限值比例 100%，非零比例 {bm['nonzero_fraction']:.0%}；保存的横轨间隔 **{b['dx_m']:.6f} m**，沿轨间隔 {b['dy_m']:.6f} m。其 H={bm['entropy_nats']:.6f} nats、H/lnN={bm['normalized_entropy']:.6f}、强度 IC={bm['intensity_contrast']:.6f}、幅度 IC={bm['amplitude_contrast']:.6f}、最亮 1% 像素占能量 {bm['brightest_1pct_energy_fraction']:.2%}。这些只是该文件自身的描述统计：缺生成脚本和脉冲区间记录，且网格太粗，不能和新图按熵高低排名，也不能证明亚米距离分辨率。

`dot.m` 原样执行的最终图非零像素数为 {dot['nonzero_pixels']}，最大值 {dot['maximum']}。辅助包装器只隐藏绘图并保存计数，没有修改原算法。''')
add(picture('sar_quality/results/legacy_bp_audit.png','旧 BP 复数结果及其保存坐标'))

add('''## 5. 独立点目标：分辨率、PSLR、ISLR

逐个模拟三个互不叠加的单位点散射体，斜距 800、1400、2200 m，位于平均孔径中心正侧视方向。按完整导航积分轨迹独立生成含分数延时的 1 μs、200 MHz 原始矩形 LFM 回波，再调用实际距离压缩及两种成像核。没有添加噪声、传播衰减、天线方向图或运动误差；因此这些测试只检验给定理想模型下的算法响应。

本表的“距离向”沿局部地面横轨 x，方位向沿 y；18 m 高度下这里地距与斜距比例非常接近 1，但仍明确使用地面剖面。FFBP 成像使用 0.25 m 距离网格、角度过采样系数 3、叶节点上限 64；剖面取样为 0.025 m。直接 BP 在同一组物理剖面点直接求和。
''')
for axis,title in [('ground_range','5.1 距离向'),('along_track','5.2 方位向')]:
    add('### '+title)
    rows=[]
    for v in pt:
        for alg in ['direct_bp','ffbp']:
            m=v['metrics'][alg][axis]
            rows.append([int(v['range_m']),alg,fmt(m['irw_m'],4),fmt(m['pslr_db'],2),fmt(m['islr_db'],2),
                         ' / '.join(fmt(q,3) for q in m['mainlobe_bounds_m'])])
    add(table(['斜距 m','算法','IRW m','PSLR dB','ISLR dB','主瓣边界 m'],rows))
add('### 5.3 快速近似带来的变化')
rows=[]
for v in pt:
    d=v['metrics']['direct_bp'];f=v['metrics']['ffbp']
    rows.append([int(v['range_m']),fmt(100*(f['ground_range']['irw_m']/d['ground_range']['irw_m']-1),2)+'%',
                 fmt(100*(f['along_track']['irw_m']/d['along_track']['irw_m']-1),2)+'%',
                 fmt(f['along_track']['pslr_db']-d['along_track']['pslr_db'],2),
                 fmt(f['along_track']['islr_db']-d['along_track']['islr_db'],2),
                 fmt(v['fidelity']['complex_coherence'],7),fmt(v['fidelity']['ser_db'],2)])
add(table(['斜距 m','距离 IRW 变化','方位 IRW 变化','方位 PSLR 劣化 dB','方位 ISLR 劣化 dB','剖面复相干度','剖面 SER dB'],rows))
add('''三个点的剖面峰值位置误差在 0.025 m 采样下均为 0；这只是模拟坐标中的网格测量结果。主瓣几乎重合，但快速插值在低幅旁瓣区引入了可见差异，不能用很高的整体相关系数掩盖这些差异。

0.7495 m 的 c/(2B) 是名义带宽尺度，加窗后的本实现 IRW 约 0.99 m。两者定义不同；旧脚本中把 `c/(2Fs)` 或 `λ/(2V)` 直接命名为“分辨率”也不能替代脉冲响应测量。
''')
add(picture('sar_quality/results/point_target_profiles.png','三个距离的理想点目标距离向与方位向剖面'))

add('''## 6. 实测数据：统一 ROI 聚焦质量

六个区域在计算分数前按固定物理坐标指定，跨越近、中、远距离及两侧沿轨位置。每区 60×60 m，包含端点，241×241 像素，采样 0.25 m；四种结果全部使用相同坐标、相同 12,757 个脉冲和相同窗函数。只改变直接/快速计算方式以及导航/自聚焦轨迹。

FFBP 在每个 ROI 周边重新构建子图，再采样到相同地面坐标；没有把原来采用不同距离原点的 `ffbp_full` 直接与 `ffbp_final` 按像素相减。对照使用继承的已选自聚焦轨迹，没有利用本次六区分数重新挑选轨迹。六区仍位于曾用于自聚焦的同一采集场景内，属于场内诊断，**不是独立测试数据集或盲测**。
''')
add(table(['ROI','中心 x,y m','横轨范围 m','沿轨范围 m'],[[v['roi'],str(v['center_m']),str(v['bounds_m'][:2]),str(v['bounds_m'][2:])] for v in rois]))
add('### 6.1 全部聚焦分数')
rows=[]
for v in rois:
    for key,name in names.items():
        m=v['metrics'][key]
        rows.append([v['roi'],name,fmt(m['entropy_nats'],4),fmt(m['normalized_entropy'],4),
                     fmt(m['intensity_contrast'],4),fmt(m['amplitude_contrast'],4),
                     f"{m['brightest_1pct_energy_fraction']:.2%}"])
add(table(['ROI','处理','H nats','H/lnN','强度 IC','幅度 IC','最亮 1% 能量占比'],rows))
add('最亮 1% 能量占比是补充描述统计，用来提示强散射体主导程度，不是外场认证指标。全部 24 个 ROI 图均通过有限性、非零能量检查；计算没有使用 dB 裁剪图或 8 位图片。')
add('### 6.2 自聚焦的区域变化，以直接 BP 排除快速插值影响')
add(table(['ROI','熵变化 nats','强度 IC 变化','解释'],[
 [v['roi'],fmt(h,4),fmt(ic,2)+'%', '两项均支持能量集中' if ic>0 else '熵微降，但对比度略降'] for v,h,ic in zip(rois,dh,di)]))
add('''近距离 R1、R2 的改善较明显；远距离 R6 的变化很小且两个指标方向不一致。这里没有统计置信区间，也没有预先规定最小有效改善门限，因此不将微小差异写成具有统计显著性的结论。

下图每一行的四张图使用共同峰值参考和 −45 至 0 dB 显示范围；不同 ROI 行分别设置参考，避免把跨区域亮度差异误当作算法改善。所有数值来自显示前复数结果。
''')
add(picture('sar_quality/results/real_roi_comparison.png','六个固定区域的四种复数成像处理对照'))

add('''## 7. 实测数据：FFBP 对直接 BP 的数值保真度

两组轨迹分别进行直接/快速对照。表中的相位标准差只统计参考图 −30 dB 阈值以上像素；不把噪声底附近的任意相位纳入主结论。
''')
rows=[]
for v in rois:
    for key,name in [('nav','导航'),('autofocus','自聚焦')]:
        m=v['fidelity'][key]
        rows.append([v['roi'],name,fmt(m['complex_coherence'],7),f"{m['relative_complex_error']:.3%}",
                     fmt(m['ser_db'],2),fmt(m['phase_std_deg_above_minus30db'],3),m['phase_mask_samples']])
add(table(['ROI','轨迹','γ','复数相对误差','SER dB','相位标准差 °','相位统计像素数'],rows))
add('''本次 12 组实测对照，γ 约 0.999549–0.999882，复数相对误差约 2.85%–3.92%，SER 约 28.14–30.90 dB。它们支持 FFBP 保持直接 BP 的主要复数成像内容，同时也表明二者不是完全相同。

### 7.1 耗时

既有严格同极坐标网格基准：全部脉冲、360×365 个输出像素、已预热的 16 线程 CPU 内核，FFBP 0.491 s，直接 BP 10.363 s，加速 21.10 倍；相对误差 3.286%，γ=0.9998335。这个比值只适用于该 ROI，不能直接推广为整幅图端到端加速比。完整 FFBP 极坐标成像核心约 56.48 s，输出 9200×7312；没有运行同尺寸完整直接 BP 作为整幅耗时对照。

上述核心计时不包括原始数据解码、距离压缩、JIT、自聚焦和导出；硬件为 i9-14900HX、CPU 16 线程。当前六区还保存了含地面采样的调用耗时，但它们是单次诊断、未做重复计时统计，不能用来评价旧 RD/CSA 的相对速度。
''')
add(table(['ROI','轨迹','BP 调用 s','FFBP + 地面采样调用 s'],[
 [v['roi'],key,fmt(v['fidelity'][key]['direct_seconds'],3),fmt(v['fidelity'][key]['ffbp_seconds'],3)]
 for v in rois for key in ['nav','autofocus']]))

add('''## 8. 实测强散射峰：可测的表观宽度与不能认证的指标

在 R3 自聚焦图中自动选取三个间隔约 10 m 以上的强峰，随后在导航、自聚焦两种直接 BP 结果中分别局部寻找峰值，再以 0.025 m 间隔取剖面。观察到这些候选位于线状/多散射结构附近，且峰位前后沿轨相差约 3.4–3.7 m，无法确认严格匹配同一个孤立物理散射体。

因此下表是**局部峰的表观半功率宽度诊断**。不将其认证为系统分辨率，不报告这些曲线的“系统 PSLR/ISLR”，也不把局部寻峰误差误称为绝对定位误差。
''')
rows=[]
for s in scatter:
    for key in ['nav','autofocus']:
        v=s[key]
        def width(axis):return fmt(v[axis]['irw_m'],3) if v[axis].get('valid') else '无可靠交点'
        rows.append([s['target'],key,','.join(fmt(q,2) for q in v['center_xyz_m'][:2]),width('ground_range'),width('along_track')])
add(table(['候选峰','轨迹','局部峰 x,y m','表观距离宽度 m','表观方位宽度 m'],rows))
add('候选 1、2 的方位宽度在自聚焦后增加，候选 3 略减；候选 2 的距离剖面受到其他强峰干扰，未获得可靠半功率交点。这个结果限制了“自聚焦全面提高分辨率”的说法，也说明应先用独立角反射器或已知孤立散射体完成外场标定。')
add(picture('sar_quality/results/real_scatterer_profiles.png','实测多散射区表观峰宽诊断，非标定点目标'))

add('''## 9. 候选均匀区域的 ENL

采用 R5，另分成 4×4 个 60×60 像素小块，检查块均值变化；末行末列只在块均值检查时不使用，整 ROI 的 ENL 使用全部像素。
''')
add(table(['处理','ENL 条件估计','16 块均值 CV','块均值最大 / 最小'],[
 [names[k],fmt(v['enl_proxy'],4),fmt(v['tile_mean_cv'],4),fmt(v['tile_mean_max_to_min'],3)] for k,v in enl.items()]))
add('''R5 看起来接近分布式散斑，但块均值仍变化约 11%–13%，且无地面纹理分类；这些数值只能作为均匀性假设下的代理估计。自聚焦后 ENL 从约 1.02 降至 0.92，不足以判断质量变差：较清晰的散斑和纹理也可能提高强度方差。没有实施多视或去斑，不用 ENL 为成像分辨率打分。

## 10. 既有成像结果与验证的衔接

此前 `fast_bp/` 的完整成像最终结果保留在原目录；本报告的 ROI 和点目标测量是新增诊断，不替换其复数输出。完整结果使用 12,757 个脉冲、256 个叶节点及 255 次复数合并；4000 个同坐标直接 BP 对照点的相对复误差 2.945%，γ=0.99994335。该抽样包括随机点和亮点，与本次六个均匀物理 ROI 的取样分布不同，所以指标不必相同。

此前自聚焦在同场景六次迭代图中选取了已成像的迭代 1，有效水平轨迹修正 RMS 约 0.099 m、0.129 m。旧记录中的“contrast”3979.1→8174.6 实际是 E(I²)/E(I)² = IC_intensity²+1，并非本报告直接列出的 IC；两者不能数值混列。选择轨迹使用该全场分数，具有偏重强散射区的可能，本次多区与表观峰宽诊断正用于检查其局限。

最终地面导出为 2601×2351、1 m 网格。该文件适于概览，但 1 m 网格对近距离约 0.57 m 的理想方位主瓣过粗，不能用它可靠测亚米点目标宽度；本报告改用更细剖面重算。高分辨率分析应保留极坐标复数结果或重新生成足够密的地面网格。
''')
add(picture('fast_bp/results/ffbp_final_preview.png','既有实测 FFBP 自聚焦全景预览，固定显示窗'))
add('### 10.1 验证范围')
oldtests=json.loads((ROOT/'fast_bp/results/synthetic_validation.json').read_text())
add(table(['既有独立点测试叶上限','复数相对误差','γ','最大网格峰位误差 m'],[
 [v['leaf_size'],f"{v['relative_complex_error']:.3%}",fmt(v['complex_coherence'],7),max(v['peak_position_errors_m'])] for v in oldtests['point_tests']]))
add('''既有分数延时脉冲测试：理论 1512.4 样本，4 倍插值压缩峰值 1512.5 样本，通过 0.25 样本容差。已知模拟运动误差的自聚焦测试 RMS 从 0.06683 m 降至 0.00152 m，说明该模拟条件下修正可恢复；这不是实际导航精度的测量。

本次指标实现另用解析 sinc 校验：IRW=0.885893、PSLR=−13.26146 dB；有限 ±40 区间内 ISLR=−9.79511 dB，并确认全零图被拒绝。无限积分的 sinc ISLR 与该有限窗口会有差异。MATLAB 探针及 `dot.m` 检查亦已实际执行。

## 11. 使用这些结论时的边界与改进顺序

1. 先修复旧 RD 的多普勒支持限制、复数转置和无穷值处理，恢复 `rd.m` 的必要徙动校正；修复完成后应重新用独立点目标验证，而不是仅看能否生成非黑图。
2. CSA 应统一为本 DAT 的 1 μs、导航速度、FFT 频率轴和时延定义；任何自聚焦均应在复数阶段进行。当前卫星 CSA 图不能充当该修改的验证结果。
3. 旧 BP 应改为距离压缩后按每脉冲三维传播距离插值并相干累加，建立有单位的输出坐标；并行版本应先验证与串行版本的数值一致性。
4. FFBP 的主瓣保真较好，但若关注强目标附近弱散射体，应进一步研究插值密度与旁瓣代价；本次已测得的约 1 dB 峰值旁瓣和约 2 dB 积分旁瓣差异需要保留在判断中。
5. 外场质量认证需要已知位置、尺寸和 RCS 的独立点目标，噪声记录、天线方向/高度标定及地形信息；自聚焦应同时检查多个区域并保留未参与估计的验证目标。

以上是基于本次评价得出的后续修复建议，本轮没有把原算法改写后重新命名为已经修复的版本。

## 12. 可复现文件与运行方式

本文件是统一结论入口。图片通过本机绝对路径嵌入；复制报告到其他电脑时需要同时携带对应图片并调整路径。所有核心数值、条件和结论已写入本文，不要求另读其他 Markdown 才能理解。
''')
files=[('sar_quality/evaluate.py','采集审计、点目标、六区实测和 ENL 计算'),('sar_quality/metrics.py','指标函数及口径'),
('sar_quality/build_report.py','由 JSON 汇总生成本报告和概览图'),('sar_quality/legacy_probe.m','旧 RD 数值探针'),
('sar_quality/dot_probe.m','原 dot.m 输出核验'),('sar_quality/results/protocol.json','全部共同参数及信号模型'),
('sar_quality/results/legacy_outputs.json','旧图片及复数 MAT 审计'),('sar_quality/results/point_targets.json','点目标全部测量'),
('sar_quality/results/point_target_profiles.npz','可重绘点目标复数剖面'),('sar_quality/results/real_rois.json','24 幅 ROI 的聚焦及保真度指标'),
('sar_quality/results/real_scatterers.json','表观峰宽和各次寻峰坐标'),('sar_quality/results/enl_proxy.json','候选均匀区统计'),
('sar_quality/results/metric_selfcheck.json','指标函数解析自检'),('fast_bp/results/benchmark.json','既有相同极坐标网格计时'),
('fast_bp/results/ffbp_final_metrics.json','既有完整图处理记录'),('fast_bp/results/ffbp_final_ground.mat','既有完整地面复数图'),
('fast_bp/results/ffbp_final_complex_demodulated.npy','既有完整极坐标复数图'),('fast_bp/results/ffbp_final_axes.npz','极坐标、相位中心和波数'),
('fast_bp/results/autofocus_position.npy','本次直接复用的自聚焦轨迹')]
add(table(['文件','用途'],[[link(p),d] for p,d in files]))
add('六个 ROI 的线性复数结果另存 `sar_quality/results/real_roi_1.npz` 至 `real_roi_6.npz`，键包括 x、y、bp_nav、ffbp_nav、bp_autofocus、ffbp_autofocus。未保存生成式图像或通过视觉修补掩盖旁瓣。')
add(r'''在项目根目录，使用已安装 NumPy、SciPy、Numba、Matplotlib、Pillow 的 Python 环境：

```powershell
python sar_quality/evaluate.py
& 'D:\Program Files\MATLAB\R2024a\bin\matlab.exe' -batch "run('sar_quality/legacy_probe.m'); run('sar_quality/dot_probe.m')"
python sar_quality/build_report.py
```

评价需要既有 `fast_bp/results/range_compressed.npy` 和 `autofocus_position.npy`。如从原始 DAT 开始，先运行：

```powershell
python fast_bp/run.py --name baseline_for_cache
python fast_bp/autofocus.py
python fast_bp/run.py --position-file fast_bp/results/autofocus_position.npy --name ffbp_final
```

完整成像及既有 benchmark 的精确记录以 JSON 为准；更换硬件、线程数、窗函数、轨迹或插值密度后须重新评价，不能直接沿用本报告的耗时和旁瓣数值。
''')

target=ROOT/'SAR成像算法质量与分辨率评价报告.md'
# Eliminate an ambiguous multiplication typography in the formula before export.
report='\n'.join(text).replace('γ = |Σ z_ref* conj(z_fast)|','γ = |Σ z_ref · conj(z_fast)|')
report=report.replace('式中第一行的 `z_ref*` 中星号仅表示乘法语境，准确实现为','复数相干度的准确实现为')
for name in ['raw_RD_Liu.m','rd.m','rd2.m','raw_cs_Liu.m','raw_cs.m','RADARSAT.m','raw_bp.m','bp2.m','dot.m']:
    report=report.replace('| '+name+' |','| '+link(name)+' |')
target.write_text(report,encoding='utf8')
code=list((ROOT/'sar_quality').glob('*.py'))+list((ROOT/'sar_quality').glob('*.m'))+list((ROOT/'fast_bp').glob('*.py'))
(OUT/'code_sha256.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in code},indent=2),encoding='utf8')
print(str(target),len(report),'characters')
