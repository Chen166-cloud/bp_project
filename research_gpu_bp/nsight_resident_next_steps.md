# 本轮驻留 BP 实验的 Nsight 采集路径

核验日期：2026-09-17。后续安装补记：已将 **Nsight Systems 2026.5.1** 和 **Nsight Compute 2026.3.1** 安装到 `E:\NVIDIA`，并把两个 CLI 目录加入当前用户 `PATH`。由于原工作区路径含中文，另建 `E:\bp_workspace` 目录联接作为采集入口，原数据没有移动。

已完成两项最小验证：Systems 的 node-level CUDA Graph 报告位于 `E:\NsightReports\resident_systems_node_smoke.nsys-rep`，包含 1 个叶节点核、8 个合并核和 1 个裁剪核；Compute 的管理员采集报告位于 `E:\NsightReports\resident_compute_smoke_admin.ncu-repz`，成功读取 `batched_bp_polar_fp32` 的 Launch Statistics。普通权限运行 Compute 会返回 `ERR_NVGPUCTRPERM`，因此当前应以管理员方式采集，或以后在 NVIDIA App 中单独开启普通用户计数器权限。没有永久修改性能计数器权限或显卡模式。

入口检查补记：`resident_profile_target.py --variant batched --mode fp32 --scale small --graph --iterations 1 --warmups 1` 已在无 Nsight 的环境实际运行通过，profiler API/NVTX 范围正常结束；这不是 Nsight 采集验证。

**Nsight 对决定下一步优化很重要。** 已有随机交错 CUDA-event 实验可以回答“哪个版本更快”；Systems 用来解释 CPU 发射与 GPU 执行空隙，Compute 用来区分显存、缓存、指令和占用率限制。现有逐核 events 与编译器寄存器数不能代替硬件计数器，也不能单凭 Graph 加速就断言所有差值都是 launch overhead。

## 本机是否适用

现有 manifest 记录 Windows 11、RTX 4070 Laptop、计算能力 8.9、CuPy 13.6.0、Python 3.12.3。官方 Compute 支持 AD10x 架构及 Windows 10/11；Systems 支持 Windows x64 和 Turing 及之后架构，因此平台与 GPU 架构符合范围。仍应实际核对安装版本及驱动要求；manifest 的 `driver_version=13030` 是 CUDA driver API 版本，不能当作 Windows 显示驱动版本号。[Compute GPU 支持](https://docs.nvidia.com/nsight-compute/ReleaseNotes/topics/gpu-support.html)、[Compute 系统要求](https://docs.nvidia.com/nsight-compute/ReleaseNotes/topics/system-requirements.html)、[Systems 安装要求](https://docs.nvidia.com/nsight-systems/InstallationGuide/)

WDDM 本身不构成直接否定条件：官方 GPU performance-counter 权限说明明确覆盖 WDDM。遇到 `ERR_NVGPUCTRPERM` 时，优先辨别权限问题。官方给出的 Windows 路径是管理员启动采集；若要允许普通用户采集，可由管理员修改 NVIDIA App 的 Developer → Manage GPU Performance Counters，较旧版本使用 NVIDIA Control Panel。这里没有执行任何权限变更，也不需要为了先做采集而尝试切换 TCC。[NVIDIA 官方权限说明](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters)

## 已准备好的小入口

[resident_profile_target.py](../fast_bp/resident_profile_target.py) 每次只运行一种算法/精度/ROI。加载真实输入、上传、编译、预分配、Graph 捕获和三次预热均在采集范围外；`cudaProfilerStart/Stop` 只包住成像循环，另有 `resident_profile` NVTX 范围。每张图结束保留一次同步，便于对齐本轮完成一张图的计时边界。程序不写任何正式 benchmark 结果。

接口来自本机 CuPy 13.6 的 `cuda/profiler.py`：其 `start/stop` 是 `runtime.profilerStart/profilerStop` 的别名。NVTX 使用官方 `RangePush/RangePop`；若不可用，仍保留 CUDA profiler API 范围。[CuPy NVTX 文档](https://docs.cupy.dev/en/v13.6.0/reference/generated/cupy.cuda.nvtx.RangePush.html)、[CUDA 采集范围 API](https://docs.nvidia.com/cuda/profiler-users-guide/index.html)

以下为工具已安装后的 PowerShell 命令。`nsys` / `ncu` 应解析到实际安装的可执行文件；Windows Systems CLI 官方文档要求管理员运行。正式执行时给每次采集使用不同输出名。[Systems CLI](https://docs.nvidia.com/nsight-systems/UserGuide/)

```powershell
$bpRoot = 'D:\Users\Chen\Desktop\research\试验数据'
$bpPython = 'D:\develop\Anaconda\python.exe'
$bpTarget = Join-Path $bpRoot 'fast_bp\resident_profile_target.py'
$bpOut = Join-Path $bpRoot 'fast_bp\results\resident_nsight'
New-Item -ItemType Directory -Path $bpOut -Force | Out-Null
Get-Command nsys, ncu
nsys --version
ncu --version
ncu --list-sections
```

## 先看 Systems 时间线

```powershell
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop -o "$bpOut\original_fp32_small" $bpPython -u $bpTarget --variant original --mode fp32 --scale small --iterations 3
```

随后分别改成以下组合，并更换 `-o` 名字：`original --graph`、`fused --graph`、`batched --graph`；最后将 `--scale small` 改为 `large`。对比同一张图内的发射次数、GPU 空隙、核执行时长分布和同步等待。需要排查 Windows 调度时再加 WDDM trace，不在第一轮加入大量系统事件。

`--cuda-graph-trace=node` 能展开图内节点，但官方明确它可能明显增加开销；再用 `--cuda-graph-trace=graph` 复查整图行为。时间线中的加速比不能替代原来的无 profiler 随机计时。[CUDA Graph trace 说明](https://docs.nvidia.com/nsight-systems/UserGuide/#cuda-graph-trace)

## 再用 Compute 只收少量关键核

下例只收批处理叶 BP 的一个 launch，限定三个 section；不采全应用 `--set full`。入口已完成预热，因此 `--launch-skip 0` 指向采集范围内第一次匹配。

```powershell
ncu --profile-from-start off --kernel-name-base function --kernel-name batched_bp_polar_fp32 --launch-skip 0 --launch-count 1 --section LaunchStats --section Occupancy --section SpeedOfLight --clock-control none -o "$bpOut\batched_leaf_fp32_small" $bpPython -u $bpTarget --variant batched --mode fp32 --scale small --iterations 1
```

按已安装版本的 `--list-sections` 输出确认名称。接下来只改变所需变量：

| 要回答的问题 | 目标核与采集安排 |
|---|---|
| batch 后叶 BP 的硬件利用率是否改善 | 对照 `bp_polar_fp32` 与 `batched_bp_polar_fp32`；选代表节点，记录 grid、节点像素数和脉冲数，避免用任意单叶的时间与整批直接相比 |
| 哪层 merge 最贵 | `batched_merge_regrid_fp32`，先 `--launch-count 8`；完整 256 叶树的 8 个合并层各一次，再针对重层只收一个 |
| 是实际 DRAM 流量，还是缓存/访存指令限制 | 对选定核增收 `MemoryWorkloadAnalysis`，核对读写字节、缓存行为和吞吐；不使用输入分配大小替代流量 |
| FP64 几何/相位与寄存器负担 | 将 `--mode mixed` 和核后缀同步改为 `mixed`，按需增收 `InstructionStats`、`SchedulerStats`；结合代码与原无 profiler 时间验证解释 |
| 直接 BP 的主要限制 | `--algorithm direct --variant original --mode fp32`，目标 `direct_chunk_fp32`，限定一个代表性 chunk，并记录 `--chunk-pulses` |

这里的目标核名称已与当前 CUDA 源码核对；`launch-skip/count` 只对匹配的核计数。逐核分析先用 eager 版本；检查 Graph 内核时再加入口 `--graph` 与 ncu `--graph-profiling node`。[Compute CLI 过滤与 section 说明](https://docs.nvidia.com/nsight-compute/NsightComputeCli/)

## 如何避免把 profiler 行为当成算法收益

Compute 可能串行化 kernel，并为不同计数器多次 replay；默认 cache-control 会清空 GPU cache，默认 clock-control 也可能改变时钟。因此其核时间与原应用时间不能直接混用。上面的 `--clock-control none` 避免工具主动改时钟，但也没有消除温度/功耗波动。[官方可重复性说明](https://docs.nvidia.com/nsight-compute/ProfilingGuide/#reproducibility)

先用默认 cache-control 做可复现的隔离诊断。若研究的是实际热缓存复用，再单独做 `--replay-mode application --cache-control none --clock-control none` 的对照，保持其余设置一致，并保存所需 replay 次数。`cache-control none` 配合 kernel replay 不自动恢复真实应用的每次缓存状态。最终性能结论仍来自无 profiler 的完整驻留 CUDA-event 实验；Nsight 报告用于支持瓶颈机制和优化依据。[官方缓存与工作负载时长说明](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)

最小交付：小/大 ROI 四种调度时间线，各选一组叶 BP 和重层 merge 的计数器报告，并将工具版本、驱动、filter、replay/cache/clock 设置与原始报告一起保存。拿到这些证据后，才能决定下轮优先做减少浮点代价、改善采样局部性，还是继续优化树级调度。
