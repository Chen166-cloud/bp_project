# 本地 CDdata1 的 BP / FFBP 实验

本目录读取项目原有 `CDdata1.mat`，沿用 `RADARSAT.m` 中的 RADARSAT-1 等效模型。文件原始来源、裁剪记录及温哥华归属未认证；本次没有下载新的原始回波。文件时间记录见 `results/provenance.json`，不得将其当成下载时间。

成像使用全部 1728 脉冲，不采用 PGA、附加相位估计或已有自聚焦轨迹。原始文件和既有 BP/FFBP 核心未修改。

从项目根目录运行：

```powershell
python vancouver_bp/validate.py
python vancouver_bp/quality.py --selfcheck vancouver_bp/results/quality_selfcheck.json
python vancouver_bp/run.py overview
python vancouver_bp/targets.py scout --count 6
python vancouver_bp/targets.py scout --offset 6 --count 6
python vancouver_bp/finalize.py --final-only
python vancouver_bp/build_report.py
python vancouver_bp/check_no_pga.py
```

主结果为 `results/REPORT.md`、`quality_table.csv`、`target_metrics.json` 和原始复图 NPZ。汇报仅使用最终无PGA成像及指标。

结果中的距离轴是斜距，方位轴是目标处局部等斜距切向米制坐标。半功率 IRW 为未标定强散射点的表观响应宽度；PSLR/ISLR 使用明确的一维首极小主瓣和有限积分窗，不能直接当作角反射器系统标定结果。

汇报稿为 `results/REPORT.md`，PDF为项目 `output/pdf/BP与FFBP成像实验报告.pdf`。PDF使用配套工作区Python运行 `vancouver_bp/render_meeting_pdf.py` 生成。

`check_no_pga.py` 从本地MAT独立重建距离压缩，核对CPU/GPU零附加相位和固定轨迹，重算两个局部BP/FFBP与整场FFBP，并以独立CPU BP抽检整场256个像素。核查通过不认证该MAT生成前的未知处理历史。运行时若出现非零附加相位或改变轨迹，成像器会立即拒绝执行。
