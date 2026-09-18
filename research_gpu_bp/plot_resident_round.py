"""Regenerate scientific plots and descriptive summaries from saved JSON only.

This module never imports CUDA or the GPU backend. Missing follow-up result
files are skipped, so rerunning after another suite completes is sufficient.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parent.parent
COLORS={"fp32":"#0072B2","mixed":"#D55E00","fp64":"#009E73"}
VARIANTS={"original_graph":("Original + Graph","#777777"),
          "fused_graph":("Fused + Graph","#0072B2"),
          "batched":("Batched","#E69F00"),
          "batched_graph":("Batched + Graph","#009E73")}
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.titlesize":11,
                     "axes.labelsize":10,"legend.fontsize":8,"axes.spines.top":False,
                     "axes.spines.right":False,"savefig.dpi":180,"figure.dpi":100})

def read(path):
    if not path.exists(): return None
    try: return json.loads(path.read_text(encoding="utf8"))
    except json.JSONDecodeError: return None  # A writer may still be finishing.

def timing(record):
    t=record.get("timing",{})
    samples=t.get("samples_ms")
    if samples is None: samples=[s["event_ms"] for s in record.get("samples",[])]
    a=np.asarray(samples,dtype=float)
    if len(a)==0: return None
    n=len(a)//2
    q1,q3=np.percentile(a,[25,75]);med=float(np.median(a))
    first=float(np.median(a[:n])) if n else med
    last=float(np.median(a[-n:])) if n else med
    return dict(n=len(a),median_ms=med,q1_ms=float(q1),q3_ms=float(q3),
                minimum_ms=float(a.min()),maximum_ms=float(a.max()),
                cv_percent=float(a.std()/a.mean()*100),iqr_percent=float((q3-q1)/med*100),
                first_half_median_ms=first,last_half_median_ms=last,
                half_drift_percent=float((last/first-1)*100),
                samples_ms=a.tolist())

def record_summary(record):
    q=record.get("quality_vs_direct_fp64",{})
    t=timing(record)
    result={k:record.get(k) for k in ("name","algorithm","mode","variant","graph","leaf","internal_os","extra_guard") if k in record}
    result.update(timing=t,ser_db=q.get("ser_db"),relative_complex_error=q.get("relative_complex_error"),
                  quality=q,launch_count=record.get("launch_count"))
    if t:
        result["stability_screen_passed"]=(abs(t["half_drift_percent"])<=5 and t["cv_percent"]<=10 and t["iqr_percent"]<=10)
    return result

def pair_summary(before,after):
    a=np.asarray(timing(before)["samples_ms"]);b=np.asarray(timing(after)["samples_ms"])
    out=dict(before=before["name"],after=after["name"],ratio_of_medians=float(np.median(a)/np.median(b)),
             scope="Same JSON experiment group; descriptive randomized-round comparison")
    if len(a)==len(b):
        ratios=a/b
        out.update(paired_median=float(np.median(ratios)),paired_q1=float(np.percentile(ratios,25)),
                   paired_q3=float(np.percentile(ratios,75)),after_faster_rounds=int(np.count_nonzero(b<a)),rounds=len(a))
    return out

def save(fig,path):
    fig.savefig(path,bbox_inches="tight",facecolor="white")
    plt.close(fig)
    return str(path)

def plot_ablation(data,out):
    records={r["name"]:r for r in data["records"]}
    fig,axes=plt.subplots(1,3,figsize=(12,4.3))
    variants=("original","original_graph","fused","fused_graph")
    labels=("Original","Graph","Fused","Fused\n+ Graph")
    for ax,mode in zip(axes,("fp32","mixed","fp64")):
        rows=[records[f"ffbp_{mode}_{v}"] for v in variants]
        ts=[timing(r) for r in rows];ys=np.array([t["median_ms"] for t in ts])
        err=np.array([[t["median_ms"]-t["q1_ms"] for t in ts],[t["q3_ms"]-t["median_ms"] for t in ts]])
        ax.bar(np.arange(4),ys,color=["#888888","#56B4E9","#E69F00",COLORS[mode]],yerr=err,capsize=3,width=.65)
        ax.set_xticks(np.arange(4),labels);ax.set_title(mode.upper());ax.set_ylabel("Resident time (ms)")
        ax.set_ylim(0,max(t["q3_ms"] for t in ts)*1.35);ax.grid(axis="y",alpha=.2);ax.set_axisbelow(True)
        for x,y in enumerate(ys):ax.text(x,ts[x]["q3_ms"]+max(ys)*.035,f"{y:.2f}",ha="center",fontsize=8)
        q=rows[0]["quality_vs_direct_fp64"]
        direct=timing(records[f"direct_{mode}_c64"])["median_ms"]
        ax.text(.02,.96,f"Total complex error: {100*q['relative_complex_error']:.3f}%\nDirect BP: {direct:.1f} ms",transform=ax.transAxes,va="top",fontsize=8)
    fig.suptitle("FFBP execution ablation: same grid and precision",y=1.02)
    fig.text(.5,-.01,"Bars: medians; whiskers: IQR; 15 randomized rounds. Allocation excluded uniformly; required initialization included.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.035,1,1))
    return save(fig,out/"01_execution_ablation.png")

def sweep_summary(data,results,thresholds):
    rows=[record_summary(r) for r in data["records"]]
    direct=[r for r in data["records"] if r.get("algorithm")=="direct"]
    all_direct=np.concatenate([np.asarray(timing(r)["samples_ms"]) for r in direct])
    pooled={"name":f"direct_fp32_pooled_{data['name']}","algorithm":"direct","mode":"fp32",
            "timing":{"samples_ms":all_direct.tolist()},"quality_vs_direct_fp64":direct[0]["quality_vs_direct_fp64"]}
    pooled_summary=record_summary(pooled)
    pooled_summary["scope"]="Pooled repeated controls across six sequential groups; not a separately timed candidate"
    group_comparisons=[]
    for group in data.get("groups",[]):
        g=read(results/(group+".json"))
        if not g:continue
        d=next((r for r in g["records"] if r.get("algorithm")=="direct"),None)
        if d:
            group_comparisons.extend(pair_summary(d,r) for r in g["records"] if r is not d)
    select={}
    ff=[r for r in rows if r.get("algorithm")=="ffbp"]
    for threshold in thresholds:
        eligible=[r for r in ff if r["ser_db"] is not None and r["ser_db"]>=threshold]
        incl=eligible+([pooled_summary] if pooled_summary["ser_db"]>=threshold else [])
        stable=[r for r in eligible if r["stability_screen_passed"]]
        select[str(threshold)]={"fastest_ffbp":min(eligible,key=lambda r:r["timing"]["median_ms"]) if eligible else None,
            "fastest_observed_including_pooled_direct":min(incl,key=lambda r:r["timing"]["median_ms"]) if incl else None,
            "fastest_ffbp_passing_descriptive_stability_screen":min(stable,key=lambda r:r["timing"]["median_ms"]) if stable else None,
            "ffbp_eligible_count":len(eligible)}
    meds=[timing(r)["median_ms"] for r in direct]
    return dict(name=data["name"],shape=data["shape"],records=rows,
                direct_control_pooled=pooled_summary,direct_group_median_range_ms=[min(meds),max(meds)],
                quality_thresholds_db=select,same_group_direct_comparisons=group_comparisons)

def plot_pareto(sweeps,summaries,out,thresholds):
    scales=[s for s in ("small","medium","large") if s in sweeps]
    fig,axes=plt.subplots(1,len(scales),figsize=(5*len(scales),4.8),squeeze=False)
    for ax,scale in zip(axes[0],scales):
        data=sweeps[scale];summary=summaries[scale]
        ff=[r for r in data["records"] if r.get("algorithm")=="ffbp"]
        allpoints=[]
        for r in ff:
            t=timing(r);x=t["median_ms"];y=r["quality_vs_direct_fp64"]["ser_db"]
            marker="o" if r.get("internal_os")==3 else "^"
            ax.errorbar(x,y,xerr=[[x-t["q1_ms"]],[t["q3_ms"]-x]],fmt=marker,color=COLORS[r["mode"]],markersize=6,capsize=2,elinewidth=.8)
            ax.annotate(f"L{r['leaf']}",(x,y),xytext=(4,3 if r['mode']=='fp32' else -10),textcoords="offset points",fontsize=7,color=COLORS[r["mode"]])
            allpoints.append((x,y))
        d=summary["direct_control_pooled"];t=d["timing"];x=t["median_ms"];y=d["ser_db"]
        ax.errorbar(x,y,xerr=[[x-t["q1_ms"]],[t["q3_ms"]-x]],fmt="D",color="#222222",markersize=6,capsize=3)
        ax.axvspan(*summary["direct_group_median_range_ms"],color="black",alpha=.055)
        ax.annotate("Direct FP32",(x,y),xytext=(0,9),textcoords="offset points",ha="center",fontsize=8)
        allpoints.append((x,y))
        front=[];best=-np.inf
        for x,y in sorted(allpoints):
            if y>best:front.append((x,y));best=y
        if len(front)>1:ax.plot(*np.array(front).T,ls=":",lw=.8,color="#666666",alpha=.6,zorder=0)
        for threshold in thresholds:ax.axhline(threshold,ls="--",lw=.7,color="#666666",alpha=.65)
        ax.set_xscale("log");ax.set_ylim(26,53);ax.set_xlabel("Resident time (ms; log scale)");ax.set_ylabel("SER vs FP64 direct BP (dB)")
        ax.set_title(f"{scale.capitalize()}  |  {np.prod(data['shape'])/1e6:.3f} M pixels")
        ax.grid(alpha=.15,which="both");ax.margins(x=.16)
    handles=[Line2D([],[],marker="o",ls="",color=COLORS[m],label=m.upper()) for m in ("fp32","mixed")]
    handles += [Line2D([],[],marker="o",ls="",color="#555555",label="Internal OS 3"),Line2D([],[],marker="^",ls="",color="#555555",label="Internal OS 6")]
    fig.legend(handles=handles,loc="upper center",ncol=4,bbox_to_anchor=(.5,1.02))
    fig.text(.5,-.01,"Fixed output grid; FFBP fused + Graph. Whiskers: IQR. Dashed 40/45 dB lines are screening choices, not field standards.\nDotted: observed median frontier. Direct control pools repeated groups; shaded band: range of group medians. No cross-run causal speedups.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.05,1,.94))
    return save(fig,out/"02_time_quality_pareto.png")

def batch_key(r):
    v=r.get("variant",r.get("program",{}).get("variant"))
    return v+("_graph" if r.get("graph") else "")

def plot_batch(groups,out):
    fig,axes=plt.subplots(1,2,figsize=(12,4.5));summary=[]
    names=[p.stem.removeprefix("batch_").replace("_","\n") for p,_ in groups]
    x=np.arange(len(groups));width=.22
    methods=("fused_graph","batched","batched_graph")
    for j,key in enumerate(methods):
        speed=[];err=[];speed_q1=[];speed_q3=[]
        for path,data in groups:
            rows={batch_key(r):r for r in data["records"]}
            a=rows.get("original_graph");b=rows.get(key)
            if not a or not b:
                speed.append(np.nan);err.append(np.nan);speed_q1.append(np.nan);speed_q3.append(np.nan);continue
            comparison=pair_summary(a,b);speed.append(comparison["paired_median"])
            speed_q1.append(comparison["paired_q1"]);speed_q3.append(comparison["paired_q3"])
            validation=b.get("vs_first_candidate",{})
            error=validation.get("quality",{}).get("relative_complex_error",np.nan)
            err.append(max(error,1e-16))
            summary.append(dict(source=str(path),candidate=record_summary(b),comparison=comparison,vs_first_candidate=validation))
        label,color=VARIANTS[key]
        xx=x+(j-1)*width
        axes[0].bar(xx,speed,width=width,color=color,label=label,
                    yerr=[np.asarray(speed)-speed_q1,np.asarray(speed_q3)-speed],capsize=2)
        axes[1].scatter(xx,err,color=color,s=28,label=label)
    axes[0].axhline(1,color="#444444",lw=.8);axes[0].set_ylabel("Same-round speed ratio vs original + Graph");axes[0].set_title("Same-run execution comparison")
    axes[1].set_yscale("log");axes[1].axhline(1e-6,color="#777777",ls="--",lw=.8);axes[1].set_ylim(3e-17,1e-5);axes[1].set_ylabel("Relative complex error vs original + Graph");axes[1].set_title("Same-algorithm numerical agreement")
    for ax in axes:ax.set_xticks(x,names);ax.grid(axis="y",alpha=.2);ax.set_axisbelow(True)
    axes[0].legend(loc="best")
    fig.text(.5,.005,"Bars: median same-round ratios; whiskers: IQR of ratios. Each comparison uses one experiment group.\nAgreement uses saved full-image validation; zeros plotted at 1e-16. Bitwise-equality fields are retained in the JSON summary.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.09,1,1))
    return save(fig,out/"03_batch_execution_and_agreement.png"),summary

def plot_guard(groups,out):
    fig,axes=plt.subplots(1,2,figsize=(12,4.7));summary=[]
    palette=plt.get_cmap("tab10")
    for i,(path,data) in enumerate(groups):
        color=palette(i)
        for mode in ("fp32","mixed"):
            rows=sorted([r for r in data["records"] if r.get("mode")==mode],key=lambda r:r["extra_guard"])
            if not rows:continue
            xs=[r["extra_guard"] for r in rows]
            err=[100*r["quality_vs_direct_fp64"]["relative_complex_error"] for r in rows]
            ts=[timing(r) for r in rows];ys=[t["median_ms"] for t in ts]
            marker,ls=("o","-") if mode=="fp32" else ("s","--")
            label=f"L{rows[0]['leaf']}, OS{rows[0]['internal_os']:g}, {mode.upper()}"
            axes[0].plot(xs,err,marker=marker,ls=ls,color=color,label=label)
            axes[1].errorbar(xs,ys,yerr=[[t["median_ms"]-t["q1_ms"] for t in ts],[t["q3_ms"]-t["median_ms"] for t in ts]],marker=marker,ls=ls,color=color,capsize=3)
            summary.extend(dict(source=str(path),candidate=record_summary(r),vs_guard0_same_mode=r.get("vs_guard0_same_mode"),vs_guard16_same_mode=r.get("vs_guard16_same_mode"),support_audit_passed=r.get("support_audit_passed")) for r in rows)
    axes[0].axhline(1,color="#777777",ls=":",lw=.8);axes[0].set_ylabel("Total relative complex error (%)");axes[0].set_title("Quality vs direct BP, fixed output grid")
    axes[1].set_ylabel("Resident time (ms)");axes[1].set_title("Guard cost within each configuration")
    for ax in axes:ax.set_yscale("log");ax.set_xlabel("Extra radial guard cells per side");ax.set_xticks([0,8,16]);ax.grid(alpha=.2,which="both")
    fig.legend(*axes[0].get_legend_handles_labels(),loc="upper center",bbox_to_anchor=(.5,1.05),ncol=4,fontsize=7)
    fig.text(.5,.005,"CPU support audit checks recursive regrid coverage; support failures alone do not quantify image error.\nFP32 circles; mixed squares. Whiskers: IQR. Timings are not divided across experiment groups.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.09,1,.9))
    return save(fig,out/"04_guard_quality_and_cost.png"),summary

def plot_precision(data,out):
    fig,axes=plt.subplots(1,2,figsize=(11,4.4));rows=data["records"]
    labels=[]
    for r in rows:
        labels.append(r["mode"].upper() if r.get("variant")=="original" else r["variant"].replace("_","\n"))
    ts=[timing(r) for r in rows];ys=[t["median_ms"] for t in ts];x=np.arange(len(rows))
    colors=[COLORS.get(r["mode"],"#777777") for r in rows]
    axes[0].bar(x,ys,color=colors,yerr=[[t["median_ms"]-t["q1_ms"] for t in ts],[t["q3_ms"]-t["median_ms"] for t in ts]],capsize=3)
    errors=[r["quality_vs_direct_fp64"]["relative_complex_error"] for r in rows]
    axes[1].scatter(x,errors,c=colors,s=45)
    for ax in axes:ax.set_xticks(x,labels);ax.grid(axis="y",alpha=.2);ax.set_axisbelow(True)
    axes[0].set_ylabel("Resident time (ms)");axes[0].set_title("Matched chunk size and Graph execution")
    axes[1].set_yscale("log");axes[1].set_ylabel("Relative complex error vs FP64 direct BP");axes[1].set_title("Observed precision trade-off")
    fig.text(.5,.005,"Small ROI; all configurations use chunk=256 + Graph. Medians and IQR. LUT preparation is outside steady-state timing.\nThe LUT implementation is MatX-inspired; this is not a benchmark of the full MatX implementation.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.11,1,1))
    return save(fig,out/"05_precision_followup.png")

def stage_summary(data):
    output=[]
    for group in data:
        if not all(key in group.get("records",{}) for key in ("full","leaf_only","merge_crop_only")):continue
        ts={key:timing(record) for key,record in group["records"].items()}
        full,leaf,merge=(ts[key]["median_ms"] for key in ("full","leaf_only","merge_crop_only"))
        fraction=merge/full
        output.append(dict(scale=group["scale"],mode=group["mode"],shape=group["shape"],
            leaf=group["leaf"],internal_os=group["internal_os"],timing=ts,
            independent_merge_crop_over_full_median=fraction,
            ideal_remove_merge_additive_model_speedup=1/(1-fraction) if 0<=fraction<1 else None,
            full_over_independent_leaf_only=full/leaf,
            relative_nonadditivity=(leaf+merge-full)/full,
            caveat="Separate graph subset diagnostics change cache and launch context. Amdahl value assumes additive timing and unchanged leaf cost; it is an approximate model, not a rigorous achievable speedup bound. Single-kernel-event stages_*.json are deliberately excluded."))
    return output

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--results",type=Path,default=ROOT/"fast_bp/results/resident_20260917")
    ap.add_argument("--output",type=Path,default=ROOT/"research_gpu_bp/resident_round_figures")
    ap.add_argument("--summary",type=Path,default=ROOT/"research_gpu_bp/resident_round_summary.json")
    args=ap.parse_args();args.output.mkdir(exist_ok=True,parents=True)
    summary={"scope":"CPU-only descriptive analysis of saved results; no cross-run causal speedup attribution", "thresholds_db":[40,45],
             "threshold_note":"Preset quality screens, not domain standards; relative complex error is 10^(-SER/20)",
             "stability_screen":"Abs first/last half median change <=5%, CV<=10%, IQR/median<=10%; descriptive, not a statistical guarantee. Omit middle sample for odd repeat counts.",
             "interval_note":"Q1-Q3 empirical timing interval, not a confidence interval", "figures":[],"missing_or_incomplete":[]}
    ablation=read(args.results/"ablation_small.json")
    if ablation:summary["figures"].append(plot_ablation(ablation,args.output))
    else:summary["missing_or_incomplete"].append("ablation_small.json")
    sweeps={s:d for s in ("small","medium","large") if (d:=read(args.results/f"sweep_{s}.json"))}
    summary["scales"]={s:sweep_summary(d,args.results,[40,45]) for s,d in sweeps.items()}
    if sweeps:summary["figures"].append(plot_pareto(sweeps,summary["scales"],args.output,[40,45]))
    batches=[]
    for scale in ("small","medium","large"):
        for mode in ("fp32","mixed"):
            p=args.results/f"batch_{scale}_{mode}.json";d=read(p)
            if d:batches.append((p,d))
            else:summary["missing_or_incomplete"].append(p.name)
    if batches:
        fig,batch=plot_batch(batches,args.output);summary["figures"].append(fig);summary["batch"]=batch
    guards=[(p,d) for p in sorted(args.results.glob("guard_l*_os*.json")) if (d:=read(p)) and "records" in d]
    guards.sort(key=lambda item:(item[1]["records"][0]["leaf"],item[1]["records"][0]["internal_os"]))
    if guards:
        fig,guard=plot_guard(guards,args.output);summary["figures"].append(fig);summary["guard"]=guard
    else:summary["missing_or_incomplete"].append("guard_l{leaf}_os{os}.json")
    precision=read(args.results/"precision_small.json")
    if precision:
        summary["figures"].append(plot_precision(precision,args.output));summary["precision"]=[record_summary(r) for r in precision["records"]]
        base=next((r for r in precision["records"] if r.get("mode")=="mixed" and r.get("variant")=="original"),None)
        if base:summary["precision_same_run_comparisons"]=[pair_summary(base,r) for r in precision["records"] if r is not base]
    else:summary["missing_or_incomplete"].append("precision_small.json")
    stages=read(args.results/"graph_stage_probes.json")
    if stages:summary["graph_stage_diagnostics"]=stage_summary(stages)
    identity=read(args.results/"verified_input_identity.json")
    if identity:summary["verified_input_identity_from_experiment_runner"]=identity
    args.summary.write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf8")
    print(json.dumps({"figures":summary["figures"],"summary":str(args.summary),"missing":summary["missing_or_incomplete"]},ensure_ascii=True))

if __name__=="__main__":main()
