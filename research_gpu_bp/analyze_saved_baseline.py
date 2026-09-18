"""Analyze saved images and geometry only; no GPU experiment is run."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
BASE = ROOT / 'fast_bp/results/gpu_baseline'
m = json.loads((BASE / 'metrics.json').read_text(encoding='utf-8'))
reference = np.load(BASE / 'cpu_direct_reference.npy').astype(np.complex128)
rows = []
for algorithm in ['direct', 'ffbp']:
    for precision in ['fp64', 'fp32', 'mixed']:
        z = np.load(BASE / f'{algorithm}_{precision}.npy').astype(np.complex128)
        error = float(np.linalg.norm(z-reference)/np.linalg.norm(reference))
        run = m['gpu_runs'][precision][algorithm]
        rows.append(dict(algorithm=algorithm, precision=precision,
                         resident_ms=run['event_timing']['median_ms'],
                         total_relative_error_vs_cpu_direct=error,
                         total_ser_db_vs_cpu_direct=float(-20*np.log10(error)),
                         same_algorithm_relative_error=run['vs_cpu_same_algorithm']['relative_complex_error'],
                         host_ram_to_result_estimate_ms=run['host_ram_to_result_ms']))

# A conservative per-pulse distance window for the final direct-BP ROI.
# The axis-aligned bounding box contains every final pixel; extrema of
# Euclidean distance to the box bound all actual pixel-to-platform ranges.
p = m['plan']
nr, nu = p['shape']
r = p['r0_m'] + np.arange(nr)*p['dr_m']
u = p['u0'] + np.arange(nu)*p['du']
o = np.asarray(p['origin_m'])
ground = np.sqrt(np.maximum(0, r*r-o[2]**2))
x = o[0] + p['side']*ground[:, None]*np.sqrt(1-u[None, :]**2)
y = o[1] + ground[:, None]*u[None, :]
bmin = np.array([x.min(), y.min(), 0.])
bmax = np.array([x.max(), y.max(), 0.])
pos = np.load(ROOT / 'fast_bp/results/autofocus_position.npy')
delta_min = np.maximum(np.maximum(bmin-pos, pos-bmax), 0)
delta_max = np.maximum(abs(pos-bmin), abs(pos-bmax))
rmin = np.linalg.norm(delta_min, axis=1)
rmax = np.linalg.norm(delta_max, axis=1)
dr = m['data']['range_sample_spacing_m']
rstart = m['data']['range_start_m']
samples = m['data']['range_samples']
# Four-point Lagrange support i-1...i+2, plus one guard on each end.
lo = np.clip(np.floor((rmin-rstart)/dr).astype(np.int64)-2, 0, samples)
hi = np.clip(np.floor((rmax-rstart)/dr).astype(np.int64)+4, 0, samples)
length = hi-lo
original = len(pos)*samples*8
window = dict(scope='Final direct-BP ROI only; not FFBP intermediate grids',
              method='AABB conservative range bounds, 4-point interpolation support plus guards',
              x_bounds_m=[float(bmin[0]), float(bmax[0])],
              y_bounds_m=[float(bmin[1]), float(bmax[1])],
              bins_per_pulse_min=int(length.min()),
              bins_per_pulse_median=float(np.median(length)),
              bins_per_pulse_max=int(length.max()),
              full_input_bytes=original,
              packed_window_bytes=int(length.sum()*8),
              common_window_bytes=int((hi.max()-lo.min())*len(pos)*8),
              packing_reduction_factor=float(original/(length.sum()*8)),
              caveat='Geometric byte estimate, not measured speedup or a validated cropped implementation')
transfer = m['transfer']
run = m['gpu_runs']['mixed']['ffbp']
fixed = transfer['h2d_wall_ms'] + run['d2h_timing']['median_ms']
bound = dict(scope='Same serial upload/download path and same measured transfer durations',
             mixed_ffbp_compute_to_zero_host_stage_speedup_bound=run['host_ram_to_result_ms']/fixed,
             excludes='raw decoding, range compression, autofocus, plan construction, compilation')
result = dict(provenance='Existing 2026-09-16 saved GPU outputs; CPU analysis only on 2026-09-17',
              reference='cpu_direct_reference.npy; complex64 stored output, cast to complex128 for comparison',
              rows=rows, direct_roi_window_estimate=window, amdahl_bound=bound)
(OUT/'saved_baseline_analysis.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')

font = FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family': font.get_name(), 'axes.unicode_minus': False, 'font.size': 10})
fig, axs = plt.subplots(1, 2, figsize=(12.4, 4.8), layout='constrained')
colors = {'direct':'#1869a8', 'ffbp':'#d66b27'}
offsets = {( 'direct','fp64'):(-105,-15), ('direct','fp32'):(12,15),
           ('direct','mixed'):(-68,14), ('ffbp','fp64'):(30,-8),
           ('ffbp','fp32'):(-40,-36), ('ffbp','mixed'):(15,-23)}
for row in rows:
    a, q = row['algorithm'], row['precision']
    label = ('直接 BP' if a=='direct' else 'FFBP')+' '+q.upper()
    axs[0].scatter(row['resident_ms'], row['total_ser_db_vs_cpu_direct'], c=colors[a], s=65,
                   marker='o' if a=='direct' else 's', zorder=3)
    annotation = dict(xytext=offsets[(a,q)], textcoords='offset points')
    if a == 'ffbp':
        annotation = dict(xytext={'fp32':(25,8), 'mixed':(56,8), 'fp64':(145,26)}[q], textcoords='data')
    axs[0].annotate(label, (row['resident_ms'], row['total_ser_db_vs_cpu_direct']),
                    **annotation, fontsize=9,
                    arrowprops=dict(arrowstyle='-', color='#999999', lw=.6))
axs[0].set(xscale='log', xlim=(23,1800), ylim=(-2,212),
           xlabel='GPU 驻留计算中位数 / ms（对数轴）',
           ylabel='统一相对 CPU 直接 BP 的 SER / dB', title='同一参考下的速度—误差关系')
axs[0].grid(alpha=.2)
labels=[]
for i,row in enumerate(rows):
    a,q=row['algorithm'],row['precision']
    labels.append(('直接 BP' if a=='direct' else 'FFBP')+' '+q.upper())
    timing=m['gpu_runs'][q][a]['event_timing']
    axs[1].hlines(i,timing['min_ms'],timing['max_ms'],color=colors[a],lw=1)
    axs[1].scatter(timing['samples_ms'],np.full(5,i)+np.linspace(-.12,.12,5),s=21,color=colors[a],alpha=.55)
    axs[1].scatter(timing['median_ms'],i,s=120,marker='|',color='#17202a',zorder=4)
    axs[1].text(timing['max_ms']*1.12,i,f"{timing['median_ms']:.2f}",va='center',fontsize=9)
axs[1].set(yticks=np.arange(6),yticklabels=labels,xscale='log',xlim=(27,1950),
           xlabel='CUDA event 时间 / ms（对数轴）',
           title='原有 5 次驻留计时：散点与中位数')
axs[1].invert_yaxis()
axs[1].grid(axis='x',alpha=.2)
fig.suptitle('RTX 4070 Laptop · 12,757 脉冲 · 360 × 365 ROI · 原有输出重分析',fontsize=13)
fig.savefig(OUT/'saved_baseline_analysis.png',dpi=170)
print(json.dumps(result,ensure_ascii=False,indent=2))
