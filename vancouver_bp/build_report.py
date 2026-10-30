"""Render the measured Vancouver BP/FFBP comparison without re-imaging.

Inputs are the final target_metrics.json, target_T*.npz, and scene_pair.npz.
All amplitude displays share the corresponding BP reference. Numerical tables
are transcribed from the saved measurement record, never read from a PNG.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
ROOT = HERE.parent
plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False, "font.size": 16,
    "axes.titlesize": 18, "axes.labelsize": 16,
    "xtick.labelsize": 14, "ytick.labelsize": 14,
    "figure.facecolor": "white",
    "savefig.facecolor": "white", "axes.spines.top": False,
    "axes.spines.right": False,
})


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def link(path, label=None):
    p = Path(path).resolve()
    return f"[{label or p.name}](<{p.as_posix()}>)"


def picture(path, alt):
    return f"![{alt}](<{Path(path).resolve().as_posix()}>)"


def num(value, decimals=3):
    if value is None:
        return "—"
    try:
        return f"{float(value):.{decimals}f}" if np.isfinite(float(value)) else "—"
    except (ValueError, TypeError):
        return str(value)


def metric(target, algorithm):
    obj = target[algorithm]
    return obj.get("metrics", obj)


def target_path(out, target):
    return out / f"target_{target['id']}.npz"


def read_pair(path):
    with np.load(path, allow_pickle=False) as src:
        return {name: src[name] for name in src.files}


def db(amplitude, reference):
    return 20*np.log10(np.maximum(np.asarray(amplitude)/max(float(reference), 1e-30), 1e-8))


def peak_amplitude(target, image):
    peak = metric(target, "bp").get("peak", {})
    return float(peak.get("amplitude", np.max(np.abs(image))))


def plot_scene(record, out):
    pair = read_pair(out / "scene_pair.npz")
    r, u = pair["r"], pair["u"]
    reference = float(np.max(np.abs(pair["bp"])))
    uc = float((u[0]+u[-1])/2)
    rc = float((r[0]+r[-1])/2)
    # A display coordinate only: tangent-plane arc at the common reference.
    along = (u-uc)*rc/np.sqrt(1-uc*uc)/1000
    extent = [along[0], along[-1], r[0]/1000, r[-1]/1000]
    fig, axs = plt.subplots(1, 2, figsize=(13.5, 8), sharex=True, sharey=True,
                            constrained_layout=True)
    for ax, name in zip(axs, ["bp", "ffbp"]):
        im = ax.imshow(db(np.abs(pair[name]), reference), origin="lower",
                       extent=extent, cmap="gray", vmin=-60, vmax=-10,
                       aspect="auto", interpolation="nearest")
        for target in record["targets"]:
            center = target["center"]
            x = (float(center["u"])-uc)*rc/np.sqrt(1-uc*uc)/1000
            y = float(center["r_m"])/1000
            ax.plot(x, y, marker="o", ms=10, mfc="none", mec="#ffdf65", mew=1.3)
            ax.annotate(target["id"], (x, y), xytext=(8, 8),
                        textcoords="offset points", color="#ffdf65", weight="bold",
                        bbox=dict(boxstyle="round,pad=.15", fc="black", alpha=.65, lw=0))
        ax.set(title=name.upper(), xlabel="局部方位显示坐标 / km")
    axs[0].set_ylabel("斜距 / km")
    cb = fig.colorbar(im, ax=axs, shrink=.88, pad=.02)
    cb.set_label("相对 BP 场景峰值 / dB")
    fig.suptitle("本地 CDdata1 片段：BP 与 FFBP（RADARSAT 等效模型，无 PGA）", fontsize=17)
    fig.savefig(out/"scene_comparison.png", dpi=190)
    plt.close(fig)


def plot_targets(record, out):
    targets = record["targets"]
    fig, axs = plt.subplots(len(targets), 2, figsize=(12, 5.15*len(targets)),
                            squeeze=False, constrained_layout=True)
    for row, target in enumerate(targets):
        pair = read_pair(target_path(out, target))
        r, az = pair["r"], pair["azimuth_m"]
        p = metric(target, "bp")["peak"]
        r0, y0 = float(p["range_m"]), float(p["azimuth_m"])
        ref = peak_amplitude(target, pair["bp"])
        extent = [az[0]-y0, az[-1]-y0, r[0]-r0, r[-1]-r0]
        for col, algorithm in enumerate(["bp", "ffbp"]):
            ax = axs[row, col]
            im = ax.imshow(db(np.abs(pair[algorithm]), ref), origin="lower",
                extent=extent, cmap="gray", vmin=-40, vmax=0,
                interpolation="nearest", aspect="equal")
            ax.set(title=f"{target['id']} · {algorithm.upper()}",
                   xlabel="相对 BP 峰值的方位距离 / m",
                   ylabel="相对 BP 峰值的斜距 / m")
            ax.axhline(0, lw=.6, color="#75baff", alpha=.6)
            ax.axvline(0, lw=.6, color="#75baff", alpha=.6)
        cb = fig.colorbar(im, ax=list(axs[row]), shrink=.84, pad=.015)
        cb.set_label(f"相对 {target['id']} BP 峰值 / dB")
    fig.suptitle("强散射点局部对照；每行共用 BP 归一化与 −40～0 dB 显示窗", fontsize=17)
    fig.savefig(out/"target_comparison.png", dpi=190)
    plt.close(fig)


def cuts_from_image(pair, measured, algorithm):
    image = pair[algorithm]
    ir, ia = map(int, measured["peak"]["index"])
    return {"range": (pair["r"], image[:, ia]),
            "azimuth": (pair["azimuth_m"], image[ir, :])}


def plot_profiles(record, out):
    targets = record["targets"]
    fig, axs = plt.subplots(len(targets), 2, figsize=(14, 4.55*len(targets)),
                            squeeze=False, constrained_layout=True)
    colors = {"bp": "#1261a0", "ffbp": "#db6b16"}
    for row, target in enumerate(targets):
        pair = read_pair(target_path(out, target))
        basepeak = metric(target, "bp")["peak"]
        reference = peak_amplitude(target, pair["bp"])
        for col, direction in enumerate(["range", "azimuth"]):
            ax = axs[row, col]
            offset = float(basepeak["range_m" if direction == "range" else "azimuth_m"])
            requested_xlim = []
            for algorithm in ["bp", "ffbp"]:
                measured = metric(target, algorithm)
                x, z = cuts_from_image(pair, measured, algorithm)[direction]
                measure = measured[direction]
                label = (f"{algorithm.upper()}  IRW={num(measure.get('irw_m'),2)} m\n"
                         f"PSLR={num(measure.get('pslr_db'),2)} dB；"
                         f"ISLR={num(measure.get('islr_db'),2)} dB")
                ax.plot(x-offset, db(abs(z), reference), lw=1.6,
                        color=colors[algorithm], label=label,
                        linestyle="-" if algorithm=="bp" else "--")
                for bound in measure.get("mainlobe_bounds_m", []):
                    ax.axvline(float(bound)-offset, lw=.8, alpha=.65,
                        color=colors[algorithm], linestyle=":" if algorithm=="bp" else "--")
                requested_xlim.extend(float(q)-offset for q in measure.get("integration_bounds_m", []))
            ax.axhline(10*np.log10(.5), lw=.9, color="#555555", linestyle=":")
            ax.text(.985, .91, "−3.0103 dB", transform=ax.transAxes,
                    ha="right", va="top", fontsize=14, color="#444444")
            if requested_xlim:
                ax.set_xlim(min(requested_xlim), max(requested_xlim))
            ax.set_ylim(-55, 3)
            ax.set(title=f"{target['id']} · {'距离向' if direction=='range' else '方位向'}",
                   xlabel=f"相对 BP 峰值的{'斜距' if direction=='range' else '方位距离'} / m",
                   ylabel="相对该目标 BP 峰值 / dB")
            ax.grid(True, alpha=.22)
            ax.legend(loc="lower left", fontsize=14, framealpha=.95)
    fig.suptitle("原复图峰值剖面：BP / FFBP；竖线为各自测得的主瓣边界", fontsize=17)
    fig.savefig(out/"target_profiles.png", dpi=190)
    plt.close(fig)


def quality_rows(record):
    rows = []
    for target in record["targets"]:
        for algorithm in ["bp", "ffbp"]:
            measured = metric(target, algorithm)
            for direction in ["range", "azimuth"]:
                m = measured[direction]
                bounds = m.get("mainlobe_bounds_m", [None, None])
                window = m.get("integration_bounds_m", [None, None])
                rows.append(dict(target=target["id"], source_candidate=target.get("source_candidate"),
                    algorithm=algorithm.upper(), direction=direction,
                    irw_m=m.get("irw_m"), apparent_spatial_resolution_m=m.get("irw_m"),
                    pslr_db=m.get("pslr_db"), islr_db=m.get("islr_db"),
                    sidelobes_valid=m.get("sidelobes_valid", False),
                    mainlobe_left_m=bounds[0], mainlobe_right_m=bounds[1],
                    integration_left_m=window[0], integration_right_m=window[1],
                    peak_range_m=measured["peak"]["range_m"],
                    peak_azimuth_m=measured["peak"]["azimuth_m"],
                    peak_amplitude=measured["peak"].get("amplitude"),
                    warnings="; ".join(m.get("warnings", []))))
    return rows


def write_csv(record, out):
    rows = quality_rows(record)
    with (out/"quality_table.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def readable_kv(obj, prefix="", depth=0):
    """Compact, faithful supplementary details even if metadata gains keys."""
    rows = []
    if not isinstance(obj, dict):
        return [(prefix or "记录", obj)]
    for key, value in obj.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict) and depth < 2:
            rows.extend(readable_kv(value, path, depth+1))
        elif isinstance(value, (str, float, int, bool)) or value is None:
            rows.append((path, value))
        elif isinstance(value, (list, tuple)) and len(value) <= 8:
            rows.append((path, json.dumps(value, ensure_ascii=False)))
    return rows


def timing_value(timing, key):
    if not isinstance(timing, dict):
        return None
    value = timing.get(key)
    if value is None:
        return None
    if isinstance(value, dict):
        for k in ["median", "median_ms", "mean", "mean_ms"]:
            if k in value:
                return float(value[k])
        return None
    if isinstance(value, (list, tuple)):
        return float(np.median(value)) if len(value) else None
    return float(value)


def create_report(record, rows, out):
    meta = record.get("metadata", {})
    parameters = meta.get("parameters", {})
    targets = record["targets"]
    date = str(meta.get("experiment_date", "2026-10-31"))
    lines = ["# 本地 CDdata1 片段：按 RADARSAT 模型进行 BP / FFBP 成像与质量评价", "",
        f"实验整理日期：{date}。直接使用项目中已有的 `CDdata1.mat`，本次未从 UBC 下载原始数据；使用文件中的全部 1728 条回波，不使用 PGA 或已有自聚焦轨迹。", "",
        "**数据来源尚未核实。** 本次仅能确认该文件已存在于项目中，不能确认是谁下载、何时下载，也不能仅凭文件名和配套脚本认证其温哥华归属。文件系统创建时间为 2025-04-22 10:20:20，MAT 内部生成时间为 2019-03-26 16:31:34；两者均不是可验证的下载时间。当前结果作为该本地片段的处理结果保留，不能视为已完成来源认证的温哥华数据实验。", "",
        "## 数据与共同处理条件", "",
        "本实验在教材等效直线运动斜距平面中成像。两种算法使用同一距离压缩回波、轨迹、相位符号、脉冲权重和输出网格；BP 作为同模型数值参考。", "",
        "| 条件 | 本次设置 |", "| --- | --- |",
        "| 原始回波 | CDdata1.mat，1728 × 3200 复数 I/Q |",
        f"| 载频 / 距离采样率 | {num(parameters.get('carrier_hz',5.3e9)/1e9,3)} GHz / {num(parameters.get('sample_rate_hz',32.317e6)/1e6,3)} MHz |",
        f"| 调频率 / 脉宽 | {num(parameters.get('chirp_rate_hz_s',-.72135e12)/1e12,5)} ×10¹² Hz/s / {num(parameters.get('pulse_duration_s',41.74e-6)*1e6,2)} μs |",
        f"| PRF / 等效速度 | {num(parameters.get('prf_hz',1256.98),2)} Hz / {num(parameters.get('equivalent_speed_m_s',7062.0),1)} m/s |",
        f"| 距离压缩 / 过采样 | {meta.get('range_window','Hamming')} 匹配滤波；{meta.get('oversample',4)} 倍距离过采样 |",
        f"| 方位权重 | {meta.get('azimuth_window','Hamming across all 1728 recorded pulses')} |",
        f"| 总多普勒中心 | {num(meta.get('total_doppler_centroid_hz'),3)} Hz；参数取值详见结果 JSON |",
        f"| GPU | {meta.get('gpu',{}).get('gpu','见结果 JSON')} |",
        "| 相位和几何 | FP64 几何/相位；保留复数输出；无自聚焦 |", "",
        "原始裁剪的完整 CEOS/轨道/AGC 提取记录尚不齐全；未重复施加 AGC 补偿。本次结果用于该回波片段和给定等效模型下的成像比较，未作绝对地理配准或辐射定标。", "",
        "## 同条件场景成像", "", picture(out/"scene_comparison.png", "BP与FFBP同条件场景对照"), "",
        "**图 1.** BP 与 FFBP 场景幅度图，T1、T2 为选定强散射点。两图共用 BP 场景峰值归一化与 −60～−10 dB 显示窗；横轴为共同参考点的局部方位显示坐标，纵轴为斜距。", "",
        "## 两个强散射点的局部响应", "", picture(out/"target_comparison.png", "两个强散射点的BP与FFBP局部对照"), "",
        "**图 2.** 每行对应同一散射点，左右分别为 BP、FFBP。坐标相对该点 BP 峰值；每行共用 BP 峰值归一化及 −40～0 dB 显示窗。图像未作锐化、背景扣除或幅度域插值。", "",
        "| 目标 | 来源候选 | 参考斜距 / m | 参考 sin(方位角) |", "| --- | --- | ---: | ---: |"]
    for target in targets:
        lines.append(f"| {target['id']} | {target.get('source_candidate','—')} | {num(target['center']['r_m'],3)} | {num(target['center']['u'],8)} |")
    lines += ["", picture(out/"target_profiles.png", "距离向和方位向BP与FFBP响应剖面"), "",
        "**图 3.** 从原始复图的各算法峰值处提取距离、方位剖面。BP 与 FFBP 共用该目标 BP 峰值归一化；横向点线为 −3.0103 dB，竖线为各自测得的主瓣边界。每条剖面的 IRW 相对于该条剖面自身峰值测量。", "",
        "| 目标 | 算法 | 方向 | 表观空间分辨率＝−3 dB IRW / m | PSLR / dB | ISLR / dB |", "| --- | --- | --- | ---: | ---: | ---: |"]
    for row in rows:
        star = "*" if not row["sidelobes_valid"] else ""
        lines.append(f"| {row['target']} | {row['algorithm']} | {'斜距向' if row['direction']=='range' else '方位向'} | {num(row['irw_m'])} | {num(row['pslr_db'],2)}{star} | {num(row['islr_db'],2)}{star} |")
    lines += ["", "本表的“空间分辨率”采用半功率 IRW 的表观响应宽度，IRW 与分辨率不重复视为两个独立测量量。强散射点未证实为角反射器或理想点目标，因此表值不等于独立标定的系统分辨率。缺少可靠入射角，本次报告斜距分辨率，不换算地距分辨率。", "",
        f"按当前模型参数，名义带宽 B=|Kr|Tp≈{abs(parameters.get('chirp_rate_hz_s',-.72135e12))*parameters.get('pulse_duration_s',41.74e-6)/1e6:.3f} MHz，带宽公式 c/(2B)≈{299792458/(2*abs(parameters.get('chirp_rate_hz_s',-.72135e12))*parameters.get('pulse_duration_s',41.74e-6)):.3f} m。该名义值与实际加窗后的 −3 dB IRW 定义不同；未掌握真实天线照射孔径，暂不给出未经限定的理论方位分辨率。", "",
        "PSLR＝10log₁₀(最大旁瓣功率/主峰功率)，ISLR＝10log₁₀(旁瓣积分能量/主瓣积分能量)，均为负值约定的一维有限窗指标。主瓣取两侧半功率点以外的首个合格极小值；边界检测允许轻度平滑，宽度和能量仍由原功率采样计算。旁瓣分析窗和主瓣边界保存在 CSV/JSON，不能与其他主瓣区域定义或二维 ISLR 直接混用。", "",
        "T1 方位剖面在主峰两侧有较宽肩部，首个合格极小值界定的主瓣范围较宽；因此其约 −29 dB 的 ISLR 是本次主瓣定义及有限积分窗下的表观值，不代表理想点目标或系统标定旁瓣性能。", ""]
    invalid = [row for row in rows if not row["sidelobes_valid"]]
    if invalid:
        lines += ["带 * 的旁瓣结果受局部杂波、响应重叠或积分窗限制，保留数值用于同条件比较：", ""]
        for row in invalid:
            lines.append(f"- {row['target']} / {row['algorithm']} / {row['direction']}：{row['warnings'] or '旁瓣可信度受限，见完整测量记录'}。")
        lines.append("")
    lines += ["## 数值保真与计算耗时", "",
        "| 目标 | FFBP 对 BP 的复数相对误差 | 复数 SER / dB | 复相干度 | BP 核心耗时 / ms | FFBP 核心耗时 / ms |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for target in targets:
        f = target.get("fidelity", {})
        bt = target["bp"].get("timing", {})
        ft = target["ffbp"].get("timing", {})
        error = f.get("relative_complex_error")
        lines.append(f"| {target['id']} | {num(None if error is None else 100*error,4)}% | {num(f.get('ser_db'),2)} | {num(f.get('complex_coherence'),10)} | {num(timing_value(bt,'kernel_ms'),3)} | {num(timing_value(ft,'kernel_ms'),3)} |")
    scene = record.get("scene", {})
    if scene:
        f = scene.get("fidelity", {})
        error = f.get("relative_complex_error")
        lines.append(f"| 场景图 | {num(None if error is None else error*100,3)}% | {num(f.get('ser_db'),2)} | {num(f.get('complex_coherence'),8)} | {num(timing_value(scene.get('bp_timing',{}),'kernel_ms'),3)} | {num(timing_value(scene.get('ffbp_timing',{}),'kernel_ms'),3)} |")
    lines += ["", "计时采用同一硬件与相同输出网格；表中核心耗时是本次运行记录，不包含原始读取、距离压缩、绘图和导出。完整计时范围见结果 JSON。两种算法数值接近只说明 FFBP 对同模型 BP 的保真程度，不能排除二者共享的参数或传播模型误差。", "",
        "| 输出 | 距离网格 / m | 角向过采样倍数 | 叶节点最大脉冲数 |", "| --- | ---: | ---: | ---: |"]
    for target in targets:
        cfg=target.get("ffbp_parameters",{})
        lines.append(f"| {target['id']} 局部图 | {num(cfg.get('range_spacing_m'))} | {cfg.get('angular_oversample','—')} | {cfg.get('leaf_size','—')} |")
    if scene:
        lines.append(f"| 场景图 | {num(scene.get('range_spacing_m'))} | {scene.get('angular_oversample','—')} | {scene.get('leaf_size','—')} |")
    lines += ["", "场景图与点目标局部图使用各自列明的采样参数。整场复数误差与局部精细网格误差分别评价，不能把局部结果外推到整个场景。", "",
        "## 本次观察", ""]
    differences=[]
    for target in targets:
        bp, ff = metric(target,"bp"), metric(target,"ffbp")
        for direction in ["range","azimuth"]:
            a, b = bp[direction].get("irw_m"), ff[direction].get("irw_m")
            if a is not None and b is not None:
                differences.append(abs(float(b)-float(a)))
    if differences:
        if max(differences)<.001:
            lines.append("两个目标、两个方向的 BP 与 FFBP IRW 差异均小于 0.001 m；在本次显示精度下，表中结果相同。该结论描述同网格算法差异，不表示实测系统分辨率具有毫米级标定精度。")
        else:
            lines.append(f"两个目标、两个方向的 BP 与 FFBP IRW 最大差为 {num(max(differences))} m。")
    lines += ["", "本次仅从两个强散射点检验局部响应，结论适用于当前选点、孔径和加窗条件。后续若研究插值或分解优化，应固定这套输入与评价口径，并补充已知点目标仿真及更多距离/方位位置。", "",
        "## 检查记录与复现", ""]
    checks = [(out/"synthetic_validation.json", "已知点目标与相位/几何检查"),
              (out/"quality_selfcheck.json", "解析 sinc 指标自检")]
    for path, desc in checks:
        if path.exists():
            lines.append(f"- {desc}：{link(path)}。")
    convergence_rows=[]
    for target in targets:
        convergence = target.get("convergence")
        if convergence:
            for key, group in convergence.items():
                flat=readable_kv(group)
                values={suffix:[abs(float(v)) for name,v in flat if name.endswith(suffix)
                        and isinstance(v,(int,float))] for suffix in
                        ["irw_difference_m","pslr_difference_db","islr_difference_db"]}
                if any(values.values()):
                    convergence_rows.append((target['id'],key,*[
                        max(values[suffix]) if values[suffix] else None for suffix in
                        ["irw_difference_m","pslr_difference_db","islr_difference_db"]]))
    if convergence_rows:
        lines += ["", "| 目标 | 对照检查 | 最大 IRW 差 / m | 最大 PSLR 差 / dB | 最大 ISLR 差 / dB |",
            "| --- | --- | ---: | ---: | ---: |"]
        for name,key,width,pslr,islr in convergence_rows:
            label={"half_density_grid":"输出网格减半密度", "range_oversampling_4_to_8":"距离过采样 4→8"}.get(key,key)
            lines.append(f"| {name} | {label} | {num(width,4)} | {num(pslr,3)} | {num(islr,3)} |")
        lines += ["", "上表取各方向绝对差的最大值；完整方向结果及网格配置见结果 JSON。"]
    lines += ["", "```powershell", "python vancouver_bp/build_report.py", "```", "",
        f"输入与完整数值：{link(out/'target_metrics.json')}；{link(out/'quality_table.csv')}；"
        + "、".join(link(target_path(out,t)) for t in targets) + "。", "",
        f"实际成像与评价代码：{link(HERE/'imaging.py')}、{link(HERE/'quality.py')}；报告与成图：{link(HERE/'build_report.py')}。", "",
        "参数及数据格式核对参考 [UBC 教材配套原始回波说明](https://sar.ece.ubc.ca/SAR_Book/Extracting_SAR_Data_from_the_CD.pdf)；该链接不是本次本地文件的下载来源或完整来源认证。指标自检与本次主瓣/积分窗定义以本地 `quality.py` 为准。", ""]
    (out/"REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=OUT)
    args = parser.parse_args()
    out = args.results.resolve()
    record = load_json(out/"target_metrics.json")
    if not record.get("targets"):
        raise ValueError("target_metrics.json contains no measured targets")
    for target in record["targets"]:
        pair = read_pair(target_path(out,target))
        if pair["bp"].shape != pair["ffbp"].shape:
            raise ValueError(f"{target['id']}: mismatching BP/FFBP grids")
        if pair["bp"].shape != (len(pair["r"]),len(pair["azimuth_m"])):
            raise ValueError(f"{target['id']}: mismatching axes")
        if not np.iscomplexobj(pair["bp"]) or not np.iscomplexobj(pair["ffbp"]):
            raise ValueError("measurements require original complex images")
    plot_scene(record,out)
    plot_targets(record,out)
    plot_profiles(record,out)
    rows = write_csv(record,out)
    from meeting_report import content, write_markdown
    write_markdown(content(record))
    print(json.dumps({"report":str(out/"REPORT.md"),"quality_rows":len(rows),
        "figures":["scene_comparison.png","target_comparison.png","target_profiles.png"]},
        ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
