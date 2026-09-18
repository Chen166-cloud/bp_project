# 驻留实验环境与协议审计

日期：2026-09-17。仅进行了文件搜索与只读环境查询，没有运行 GPU kernel 或 profiler，没有安装软件、修改驱动或计数器权限。

## 工具可用性

- GPU：NVIDIA GeForce RTX 4070 Laptop GPU，计算能力 8.9，驱动 610.88；`nvidia-smi` 报告 8188 MiB。
- `nvidia-smi`：`C:/WINDOWS/system32/nvidia-smi.exe`。
- CUDA 编译器：`D:/Program Files/MATLAB/R2024a/sys/cuda/win64/cuda/bin/nvcc.exe`，实测版本 12.2.140；未加入 PATH。
- NVRTC：`D:/Program Files/MATLAB/R2024a/bin/win64/nvrtc64_120_0.dll` 及同目录的 `nvrtc-builtins64_122.dll` 已存在。当前 CuPy RawModule 后端采用 NVRTC，实验不依赖命令行 `nvcc` 或 MSVC。
- 在 PATH 与 C、D 盘 Program Files、Program Files (x86)、用户 Local/Programs、ProgramData/NVIDIA Corporation、研究目录中，未找到 `nsys.exe`、`ncu.exe`、旧版 Nsight Compute CLI、MSVC `cl.exe` 或 `vswhere.exe`。不是全盘缺失证明；GPU 性能计数器权限尚未测试。

因此可以立即做 CUDA events 计时、Graph、融合消融，但不能把结果描述成 Nsight 已证实的启动、带宽、缓存或 SFU 瓶颈。精确搜索范围和命令记录见同名 JSON。

## 同网格协议

`gpu_backend.build_ffbp_plan` 根据轨迹、ROI、内部采样密度生成每层极坐标网格。输出不只是数组形状，还由 `origin/r0/dr/u0/du/side/crop` 决定。

1. 固定轨迹、ROI、过采样时，改变 `leaf_size` 不改变 root 网格，可共用同一直接 BP 参考；必须核对完整几何属性。
2. 直接修改 `angular_oversample` 会改变 root `du/u0/nu`，单纯比较时间会混入输出像素数变化。
3. 本轮拟采用固定最终 root 网格、仅增加后代网格密度的设计，能测量内部插值质量与时间的交换关系。应分别记录 `output_oversample` 和 `internal_oversample`，将候选计划 root 的几何与 base root 对齐，同时复用 base crop 和输出属性。root origin、索引、脉冲区间必须一致；节点索引表仍用候选计划，不能引用 base 的整个 origin 数组。
4. ROI 变化会改变输出坐标，应为每个 ROI 建立单独直接 BP 参考。脉冲子集会改变 root origin、孔径与角采样，不可仅替换 shape 或共用旧参考。第一轮固定全脉冲扫 ROI 更容易解释。
5. `gpu_baseline.cpu_reference` 使用模块全局 `EXTENT/LEAF_SIZE/RANGE_SPACING/ANGULAR_OVERSAMPLE`，不能直接用来验证任意候选 plan；CPU direct 应直接调用 `bp_polar` 并传入 plan 的几何参数。

## Halo 审计建议

四点 Lagrange 二维插值的输入索引是 `floor(q)-1` 至 `floor(q)+2`。对固定 root、内部过采样变大的方案，应至少检查最终 crop 中每个 root 输出点对左右 child 的全部插值支持是否在界内。root 被裁掉的 guard 区越界本身不影响最终结果。

严格检查应从 root crop 出发，把每个所需输出格点对应的 4×4 源格点集合递归传播到后代，统计所有会影响最终 crop 的越界访问。对子计划采用相同检查，以区别原有 guard 不足与新设计引入的问题。不能仅检查最终图形状，也不能默认 8 格 guard 在所有深度与几何下都足够。

## 最小实验矩阵与验收

- 小 ROI 完成稳定性基线：直接 BP FP32，以及原 FFBP/预分配 FFBP/融合父节点/Graph/融合+Graph。所有方法同输入、同输出坐标、同权重、同相位校正和同插值；直接 BP 清零以及每帧必需处理应计入。
- 固定小 ROI 扫 `leaf_size`，先 64/128/256/512；过采样先 3/4.5/6，只在固定最终网格下改变内部密度。若质量仍过不了 1% 复误差阈值，先停止扩大精度组合，查算法误差。
- 从上步选出的较优配置扩到中、大 ROI；报告运行时间、相对直接 BP 复误差、同质量条件下的加速。使用每 ROI 自己的高精度直接 BP 参考。
- 正确性分两层：融合/Graph 相对同计划未融合 FFBP 的计算一致性；FFBP 相对直接 BP 的成像总误差。这两个误差不能替换。
- 至少保留中位数、分位数、全部重复计时和 warmup 轨迹；对选择后的配置做交错 A/B 验证，避免笔记本温度、功耗与顺序漂移造成虚假收益。出现不稳定应报告不稳定，不能只取最快一次。
- 目标优先级：找出足以支持或否定当前选题的质量—时间边界。工程融合获益是必要证据，但不足以单独证明新算法或顶刊贡献。

以上为执行前审计建议，实际完成项以实验输出 JSON 为准。

## 已完成的 CPU 支持域审计

已提供 `fast_bp/resident_plan_audit.py`：`audit_support(plan)` 不导入 CUDA，使用分块计算和逐节点布尔集合，从最终 crop 递归传播所有 4×4 源格点。它遍历完整格点集合，并非随机抽样。它有意保守包含零权重路径，且不声称精确复现 CUDA FMA 或 FP32 边界舍入；因此这里的失败计数不等于最终失真像素数或误差幅度。

当前 small ROI、leaf_size=64、输出过采样=3 的完整传播结果：

| 内部角过采样 | 需要但越界的插值模板数（按树边计数） | CPU 秒 |
|---|---:|---:|
| 3 | 10,231 | 0.608 |
| 4.5 | 10,903 | 0.674 |
| 6 | 11,502 | 0.783 |

首批失败位于约第五层以后的径向边缘。每层插值会向外需要更多格点，而原始每节点固定 8 个径向 guard 并没有覆盖所有递归依赖。由于路径权重可能很小，不能由这个结果解释已有 3.29% 总误差，仍须比较实际复图。

`extend_plan_radial_guard(plan, extra_cells)` 返回深拷贝，为所有节点径向两侧增加格点，保留原有格点坐标，平移 root crop 并保持最终输出网格不变，重算运算量统计，并在 `guard_extension` 属性记录元数据。CPU 自检确认输入 plan 未修改、原格点坐标保持、输出轴保持。small、leaf64、内部过采样3 的结果为：额外0格仍有10,231个模板越界；额外8格和16格均为0。见 `resident_radial_guard_audit.json`。是否改善成像质量，应由 GPU 复图实验回答。

## 驻留实验脚本审查

已只读审查 `fast_bp/resident_experiment.py` 和 `resident_backend.py`。当前实现将输出网格、图像清零、事件边界、随机交错计时以及计时后的图像回传明确分开；每个 ROI 建立自己的 FP64 直接 BP 参考。融合父节点保留两次重采样分别舍入到 complex64 再相加的算子语义。Graph 与普通执行共享同一 program 的输出，但读取指标前重新执行，因此没有读到其他候选结果的问题。

建议补充保存输出坐标、原始输入来源哈希、审计脚本哈希；在真实数据上额外记录 fused/original 与 Graph/execute 的同计划差异，并分析前后半轮中位数检查漂移。新脚本使用预分配基线，没有测旧递归分配后端，因此不能把新旧历史时间差单独归因于预分配、Graph 或融合。
