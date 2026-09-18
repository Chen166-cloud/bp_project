"""CPU-only analysis of saved first resident ablation; never touches GPU APIs."""
from pathlib import Path
import csv
import json
import re
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "research_gpu_bp"
RESULTS = ROOT / "fast_bp/results/resident_20260917"
source = RESULTS / "ablation_small.json"
experiment = json.loads(source.read_text(encoding="utf-8"))
original = {r["name"]: r for r in experiment["records"]}
rows = []
for record in experiment["records"]:
    a = np.asarray(record["timing"]["samples_ms"], dtype=float)
    n = len(a) // 2
    first, last = float(np.median(a[:n])), float(np.median(a[-n:]))
    q1, q3 = np.percentile(a, [25, 75])
    row = dict(name=record["name"], median_ms=float(np.median(a)),
               q1_ms=float(q1), q3_ms=float(q3), iqr_ms=float(q3-q1),
               iqr_percent_of_median=float((q3-q1)/np.median(a)*100),
               cv_percent=float(np.std(a)/np.mean(a)*100),
               first_7_median_ms=first, last_7_median_ms=last,
               last_over_first_change_percent=float((last/first-1)*100),
               min_ms=float(a.min()), max_ms=float(a.max()),
               maximum_over_minimum=float(a.max()/a.min()),
               quality=record["quality_vs_direct_fp64"])
    # Screening flags only; not statistical hypothesis tests.
    row["screening_flags"] = []
    if abs(row["last_over_first_change_percent"]) > 5:
        row["screening_flags"].append("absolute_half_drift_gt_5_percent")
    if row["cv_percent"] > 10:
        row["screening_flags"].append("cv_gt_10_percent")
    if row["iqr_percent_of_median"] > 10:
        row["screening_flags"].append("iqr_gt_10_percent_of_median")
    rows.append(row)

def comparison(label, before, after, independent_factor):
    a = np.asarray(original[before]["timing"]["samples_ms"])
    b = np.asarray(original[after]["timing"]["samples_ms"])
    ratios = a / b
    return dict(label=label, before=before, after=after,
                independent_factor=independent_factor,
                ratio_of_medians=float(np.median(a)/np.median(b)),
                median_time_reduction_percent=float((1-np.median(b)/np.median(a))*100),
                median_round_paired_speedup=float(np.median(ratios)),
                paired_speedup_q1=float(np.percentile(ratios,25)),
                paired_speedup_q3=float(np.percentile(ratios,75)),
                faster_rounds=int(np.count_nonzero(b<a)), rounds=len(a),
                all_saved_quality_fields_exactly_equal=original[before]["quality_vs_direct_fp64"] == original[after]["quality_vs_direct_fp64"])

pairs=[]
for mode in ("fp32", "fp64", "mixed"):
    for label,a,b in (("Graph / original", "original", "original_graph"),
                      ("fusion / no Graph", "original", "fused"),
                      ("fusion / Graph", "original_graph", "fused_graph"),
                      ("Graph / fused", "fused", "fused_graph"),
                      ("fusion + Graph", "original", "fused_graph")):
        pairs.append(comparison(f"{mode}: {label}",f"ffbp_{mode}_{a}",f"ffbp_{mode}_{b}",label!="fusion + Graph"))
pairs.extend([
    comparison("Direct FP32 Graph, chunk64", "direct_fp32_c64", "direct_fp32_c64_graph",True),
    comparison("Direct FP32 chunk64->256, Graph", "direct_fp32_c64_graph", "direct_fp32_c256_graph",True),
    comparison("Direct FP32 chunk256->1024, Graph", "direct_fp32_c256_graph", "direct_fp32_c1024_graph",True),
    comparison("Direct mixed -> phase32 + chunk256 + Graph", "direct_mixed_c64", "direct_mixed_phase32_c256",False),
    comparison("Direct mixed phase32 -> wrapped", "direct_mixed_phase32_c256", "direct_mixed_phase32_wrapped_c256",True),
])

def numeric(s):
    m = re.match(r"\s*(-?\d+(?:\.\d+)?)",s)
    return float(m.group(1)) if m else None

telemetry=[]
for path in RESULTS.glob("telemetry_ablation*.csv"):
    records=[]
    with path.open(encoding="utf8") as f:
        for row in csv.DictReader(f, skipinitialspace=True):
            record={"timestamp":row["timestamp"],"pstate":row["pstate"]}
            for key,value in row.items():
                if key not in ("timestamp","pstate"):
                    record[key]=numeric(value)
            records.append(record)
    active=[r for r in records if r["pstate"]=="P0" and r["utilization.gpu [%]"] is not None and r["utilization.gpu [%]"]>=90]
    summary={"path":str(path),"samples":len(records),"active_definition":"P0 and GPU utilization >= 90%; includes warmup/reference/timing/quality executions", "active_samples":len(active),"start":records[0]["timestamp"],"end":records[-1]["timestamp"]}
    for key in ("temperature.gpu","power.draw [W]","clocks.current.sm [MHz]","clocks.current.memory [MHz]","memory.used [MiB]"):
        a=np.array([r[key] for r in active if r[key] is not None])
        n=len(a)//2
        summary[key] = dict(min=float(a.min()),median=float(np.median(a)),max=float(a.max()),first_half_median=float(np.median(a[:n])),last_half_median=float(np.median(a[-n:])))
    summary["samples_at_or_above_87C"] = sum(r["temperature.gpu"]>=87 for r in active)
    summary["active_samples_below_2000_MHz"] = [r for r in active if r["clocks.current.sm [MHz]"]<2000]
    summary["limitation"] = "No per-candidate wall timestamp; 1 s snapshots cannot align clocks with individual kernels. No throttle-reason counter was captured. Thermal or power limiting is not proved."
    telemetry.append(summary)

baseline_path = ROOT / "fast_bp/results/gpu_baseline/metrics.json"
baseline = json.loads(baseline_path.read_text(encoding="utf8"))
data = baseline["data"]
hashes=[]
for label,pathkey,hashkey in (("raw","raw_file","raw_sha256"),("range_compressed","range_compressed_file","range_compressed_sha256"),("position","position_file","position_sha256")):
    p=Path(data[pathkey])
    hashes.append(dict(label=label,path=str(p),sha256=data[hashkey],source=str(baseline_path),source_experiment_date=baseline["experiment"]["date"],current_size_bytes=p.stat().st_size if p.exists() else None,current_mtime_ns=p.stat().st_mtime_ns if p.exists() else None,rehashed=False,caveat="Historical recorded SHA256 reused; current content identity was not reverified by this analysis"))

quality_groups=[]
for mode in ("fp32","fp64","mixed"):
    names=[f"ffbp_{mode}_{v}" for v in ("original","original_graph","fused","fused_graph")]
    qa=original[names[0]]["quality_vs_direct_fp64"]
    differences={}
    for n in names[1:]:
        qb=original[n]["quality_vs_direct_fp64"]
        differences[n]={k:dict(before=qa[k],after=qb[k],delta=float(qb[k]-qa[k])) for k in qa if qa[k]!=qb[k] and isinstance(qa[k],(float,int))}
    quality_groups.append(dict(mode=mode,names=names,all_saved_metric_fields_equal=all(original[n]["quality_vs_direct_fp64"]==qa for n in names),numeric_field_differences_vs_original=differences,bitwise_image_equality_verified=False))

result=dict(source=str(source),analysis_scope="CPU-only saved-data analysis",repeats=experiment["repeats"],warmups=experiment["warmups"],
            half_drift_policy="First 7 versus last 7 trials, omit middle trial 8",screening_thresholds="abs drift >5%, CV>10%, or IQR/median>10%; descriptive flags, not proof of failure",candidates=rows,comparisons=pairs,telemetry=telemetry,quality_groups=quality_groups,input_hashes=hashes,
            conclusions=["FP32 FFBP Graph and fusion have a substantial observed timing effect, but the trial is not uniformly steady-state.","Small FP64/mixed timing deltas and FP32 direct chunk rankings are insufficiently stable to rank confidently.","FP32 FFBP stored metrics match exactly across variants; FP64/mixed fused metrics differ from original by very small amounts. Within each variant Graph and ordinary execution metrics match. Full image bitwise equality was not measured.","All current FFBP variants retain about 3.29-3.50% total complex error on this ROI; pure execution changes do not solve the quality gap.","The phase32 probe comparison changes precision, chunk size and Graph together; cannot isolate FP32 trigonometry without same-chunk same-Graph mixed control."])
(OUT/"resident_first_ablation_analysis.json").write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf8")

lines=["# 第一组驻留消融实验独立分析", "", "数据来源：`fast_bp/results/resident_20260917/ablation_small.json`；每配置5次预热、15轮随机交错计时。仅分析已有文件，未执行 GPU 工作。", "", "## 结论", "", "FP32 FFBP 的 Graph 与融合确实显示出较大的提速信号，但本轮并非稳定平台上的最终性能定论。GPU 监控中温度达到90°C，活跃状态频率多次低于2000 MHz；多个候选出现超过5%的前后半轮漂移，部分候选的 CV 超过20%。温度与频率变化是已观察事实，监控没有记录节流原因，也没有每次执行的墙钟时间戳，不能证明温度或功耗就是某次变慢的原因。", "", "FP32 FFBP 的复误差仍约3.497%，FP64/mixed约3.286%；Graph的已保存质量指标与普通执行相同；融合在FP64/mixed下产生极小指标差异，尚未改变整体质量判断。它们解决执行成本，尚未解决达到1%复误差要求的问题。", "", "## 稳定性", "", "前半/后半采用前7次与后7次的中位数，舍去第8次；漂移为后/前−1。CV按总体标准差/均值计算。以下全部数值来自15次事件计时，未删异常点。", "", "| 候选 | 中位数 ms | Q1–Q3 ms | CV % | 前7→后7 ms | 漂移 % |", "|---|---:|---:|---:|---:|---:|"]
for r in rows:
    lines.append(f"| {r['name']} | {r['median_ms']:.3f} | {r['q1_ms']:.3f}–{r['q3_ms']:.3f} | {r['cv_percent']:.2f} | {r['first_7_median_ms']:.3f}→{r['last_7_median_ms']:.3f} | {r['last_over_first_change_percent']:+.2f} |")
lines += ["", "最明显的波动包括 `ffbp_fp32_fused`：CV 22.88%、IQR/中位数39.96%；`ffbp_fp32_original` 前后半中位数下降15.70%。`direct_mixed_phase32_c256` 与 wrapped 版本后半分别慢10.26%和11.55%。因此不能把数个百分点的差距当成确定优化。", "", "## Graph 与融合的条件收益", "", "下表的速度比是前配置中位数/后配置中位数；同时给出同轮比值的中位数和后配置胜出的轮数。每轮顺序随机，成对比值只能减弱慢漂移影响，不能消除轮内频率变化。", "", "| 比较 | 中位数速度比 | 同轮比值中位数 | 后配置更快轮数 |", "|---|---:|---:|---:|"]
for p in pairs:
    lines.append(f"| {p['label']} | {p['ratio_of_medians']:.3f}× | {p['median_round_paired_speedup']:.3f}× | {p['faster_rounds']}/{p['rounds']} |")
lines += ["", "Graph在未融合和已融合程序上的效果应分开看；融合也分别在普通执行和Graph条件下比较。FP32原18.084 ms→原Graph11.015 ms→融合Graph9.513 ms，组合中位数速度比约1.90×；Graph条件下融合约1.16×。二者效果存在交互，不宜拆成固定可相加的时间收益。", "", "FP64与mixed的小收益没有表现出同等稳健性。直接BP的FP32 Graph chunk64、256、1024分别约34.573、34.687、35.752 ms，现有波动不足以证明256优于64，或1024必然更差。", "", "phase32探针相对原mixed同时改变chunk64→256并加入Graph；只能报告组合实现快约1.78×。应补原mixed、chunk256、Graph对照后，再解释三角函数精度的独立收益。wrapped对phase32保持chunk与Graph相同，当前数据表明其额外操作明显增加耗时，图像指标收益很小。", "", "## 正确性与监控边界", "", "FP32的original/original_graph/fused/fused_graph全部质量指标字段一致。FP64与mixed各自的普通执行和Graph指标一致，但融合相对原版的复误差分别变化−7.60×10⁻¹²、−4.91×10⁻¹¹，SER分别变化2.01×10⁻⁹、1.30×10⁻⁸ dB；属于极小但非零差异。这里仅有相对参考的汇总指标，未保存这些候选的完整复图，因此没有验证真实图像逐位一致。合成验证结论不能自动扩大到真实图。", ""]
for t in telemetry:
    sm=t['clocks.current.sm [MHz]'];temp=t['temperature.gpu']
    lines.append(f"监控共{t['samples']}条，P0且利用率≥90%的{t['active_samples']}条中，温度{temp['min']:.0f}–{temp['max']:.0f}°C；SM频率{sm['min']:.0f}–{sm['max']:.0f} MHz，前后半中位数{sm['first_half_median']:.0f}→{sm['last_half_median']:.0f} MHz；低于2000 MHz的活跃样本{len(t['active_samples_below_2000_MHz'])}个。监控覆盖初始化、预热、参考、测量和结果检查，不能精确定位到候选或计时轮。")
lines += ["", "后续应把短时精度核对与性能重复分开；重点配置做交错A/B复测，保存每次开始/结束时间戳与热状态。大规模FP64参考每ROI只需生成一次，不必混入所有性能轮长期加热GPU。新的计时不能与历史旧后端计时直接组成单因素加速比。", "", "## 复用的输入哈希", "", "来源为2026-09-16原基线`metrics.json`，本次没有重新哈希2 GB输入。它们是历史来源记录，不是本次重新验证的文件身份。", "", "| 输入 | 已记录 SHA256 |", "|---|---|"]
for h in hashes: lines.append(f"| {h['label']} | `{h['sha256']}` |")
lines += ["", "## 下一步最小 Nsight 采集", "", "先用Systems采一个稳定后的完整resident帧，标注direct、leaf、regrid/merge范围，比较ordinary/Graph的CPU提交间隔、CUDA队列空隙、kernel累计时间与整帧时间。不要把系统时间线里的所有间隔都称作启动成本；需要同时看CPU API与GPU队列。[Nsight Systems用户指南](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)。", "", "Compute只选代表性的direct_chunk、leaf BP、低/中/高层regrid和fused merge实例。首轮使用`SpeedOfLight`、`LaunchStats`、`Occupancy`；只有出现计算或访存线索后，再分别增加`ComputeWorkloadAnalysis`或`MemoryWorkloadAnalysis`。关注实际DRAM读写量、L1/L2命中、寄存器/本地内存、执行管线利用率；调度器不能持续发射时再采`SchedulerStats`/`WarpStateStats`。指标名称随设备与版本查询，不先固定整套SASS计数器。[Nsight Compute官方Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)。", "", "Profiler可能重放kernel、恢复内存并改变缓存和计时条件，因此采集用于解释机制，最终速度仍以独立、未插桩、同质量驻留测试为准。当前本机尚未发现Nsight工具，本文没有Nsight实测结论。", ""]
(OUT/"resident_first_ablation_analysis.md").write_text("\n".join(lines),encoding="utf8")
print(json.dumps({"outputs":[str(OUT/"resident_first_ablation_analysis.json"),str(OUT/"resident_first_ablation_analysis.md")],"quality_groups":quality_groups,"telemetry":telemetry},ensure_ascii=False,indent=2))
