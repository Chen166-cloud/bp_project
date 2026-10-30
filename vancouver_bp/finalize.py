"""Produce the final paired scene and two uncalibrated point responses."""
from __future__ import annotations
import argparse
import json
import time
import numpy as np
from imaging import Imager, OUT, axes, write_json, fidelity, prepare, C, PARAMETERS
from targets import local_axes, measure
from quality import analyze_target, profile_metrics
from ffbp import backproject_points

def final_targets(imager, selected_ids=("C1","C9"), refine=False):
    scouts=json.loads((OUT/"candidate_scout.json").read_text(encoding="utf-8"))
    scouts+=json.loads((OUT/"candidate_scout_6.json").read_text(encoding="utf-8"))
    selected=[next(c for c in scouts if c["id"]==id_) for id_ in selected_ids]
    rows=[]
    for scout in selected:
        i=1 if scout["id"]=="C1" else 2
        source=scout["candidate"]
        r0=source["r_m"];u0=source["u"]
        rc=scout["metrics"]["peak"]["range_m"]
        uc=u0+scout["metrics"]["peak"]["azimuth_m"]*np.sqrt(1-u0*u0)/r0
        du=70*np.sqrt(1-uc*uc)/rc
        spacing=.0625 if refine else .125
        angular_os=48 if refine else 24
        plan=imager.plan((rc-70,rc+70,uc-du,uc+du),spacing,angular_os,64)
        print("target",i,plan.shape,flush=True)
        bp,tbp=imager.image(plan,"bp")
        ff,tff=imager.image(plan,"ffbp")
        mb,cb=measure(plan,bp,rc,uc);mf,cf=measure(plan,ff,rc,uc)
        r,u=axes(plan);_,az=local_axes(plan,rc,uc)
        for m in (mb,mf):
            m["coordinates"]={"range":"root-centred slant-range metres",
                "azimuth":"local isorange tangent metres; du scaled by rc/sqrt(1-uc^2)"}
        np.savez_compressed(OUT/f"target_T{i}.npz",r=r,u=u,azimuth_m=az,bp=bp,ffbp=ff)
        # Check the metric's output-grid sampling at half the final density.
        coarse=analyze_target(r[::2],az[::2],bp[::2,::2],expected_irw=(6.5,8),
            peak_hint=(rc,0),search_radius_m=(18,18),integration_half_widths_m=(50,50))
        convergence=dict(half_density_grid={
            d:dict(irw_m=coarse[d].get("irw_m"),irw_difference_m=coarse[d].get("irw_m",0)-mb[d].get("irw_m",0),
                pslr_difference_db=coarse[d].get("pslr_db",0)-mb[d].get("pslr_db",0),
                islr_difference_db=coarse[d].get("islr_db",0)-mb[d].get("islr_db",0))
            for d in ("range","azimuth")})
        row=dict(id=f"T{i}",source_candidate=scout["id"],center=dict(r_m=rc,u=uc),
            source_relative_scene_db=source["relative_db"],bp=dict(metrics=mb,timing=tbp),
            ffbp=dict(metrics=mf,timing=tff),fidelity=fidelity(bp,ff),convergence=convergence,
            ffbp_parameters=dict(leaf_size=64,range_spacing_m=spacing,angular_oversample=angular_os),
            interpretation="uncalibrated dominant strong scatterer; apparent IRF only")
        rows.append(row)
        print(json.dumps(dict(id=row["id"],fidelity=row["fidelity"],
            bp={d:{k:mb[d].get(k) for k in ("irw_m","pslr_db","islr_db","sidelobes_valid")} for d in ("range","azimuth")},
            ffbp={d:{k:mf[d].get(k) for k in ("irw_m","pslr_db","islr_db","sidelobes_valid")} for d in ("range","azimuth")})),flush=True)
    return rows

def interpolation_convergence(imager, rows):
    data8,meta8=prepare(8)
    for row in rows:
        data=np.load(OUT/f"target_{row['id']}.npz")
        r,u,az=data["r"],data["u"],data["azimuth_m"]
        ir,iu=row["bp"]["metrics"]["peak"]["index"]
        for direction,rs,us,x in (("range",r,np.full_like(r,u[iu]),r),
                                 ("azimuth",np.full_like(u,r[ir]),u,az)):
            xyz=np.column_stack([rs*np.sqrt(1-us*us),rs*us,np.zeros(len(rs))])
            physical=backproject_points(data8,imager.position,xyz,meta8["range_start_m"],
                meta8["compressed_spacing_m"],imager.k,imager.weight,imager.phase)
            z8=physical*np.exp(-1j*imager.k*rs)
            z4=data["bp"][:,iu] if direction=="range" else data["bp"][ir,:]
            peak=ir if direction=="range" else iu
            m8=profile_metrics(x,z8,integration_half_width_m=50,peak_index=peak)
            m4=row["bp"]["metrics"][direction]
            row["convergence"].setdefault("range_oversampling_4_to_8",{})[direction]=dict(
                irw_difference_m=m8.get("irw_m",0)-m4.get("irw_m",0),
                pslr_difference_db=m8.get("pslr_db",0)-m4.get("pslr_db",0),
                islr_difference_db=m8.get("islr_db",0)-m4.get("islr_db",0),
                profile_relative_complex_error=fidelity(z4,z8)["relative_complex_error"])
        print("convergence",row["id"],json.dumps(row["convergence"]),flush=True)

def scene_pair(imager, spacing=2.0, angular_os=2):
    valid=imager.meta["valid_full_reference_slant_range_m"]
    uc=imager.meta["sin_squint"]
    extent=(valid[0]+180,valid[1]-180,uc-.0020,uc+.0020)
    plan=imager.plan(extent,spacing,angular_os,64)
    print("scene pair",plan.shape,flush=True)
    bp,tbp=imager.image(plan,"bp")
    ff,tff=imager.image(plan,"ffbp")
    r,u=axes(plan)
    az=(u-uc)*float(np.mean(r))/np.sqrt(1-uc*uc)
    np.savez_compressed(OUT/"scene_pair.npz",r=r,u=u,azimuth_m=az,bp=bp,ffbp=ff)
    return dict(bp_timing=tbp,ffbp_timing=tff,fidelity=fidelity(bp,ff),extent=list(extent),
        range_spacing_m=spacing,angular_oversample=angular_os,leaf_size=64,
        ambiguity_note="One selected Doppler ambiguity interval; initial wide scout intentionally not used as final scene")

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--skip-scene",action="store_true")
    p.add_argument("--skip-convergence",action="store_true")
    p.add_argument("--refine-existing",action="store_true")
    p.add_argument("--final-only",action="store_true")
    args=p.parse_args();start=time.perf_counter()
    imager=Imager()
    if args.final_only:
        rows=final_targets(imager,("C1",))+final_targets(imager,("C9",),refine=True)
        if not args.skip_convergence:interpolation_convergence(imager,rows)
        result=dict(metadata=imager.meta,targets=rows,scene=scene_pair(imager,1.0,4),
                    execution_seconds=time.perf_counter()-start)
        write_json(OUT/"target_metrics.json",result)
        print("final experiment completed",result["execution_seconds"],flush=True)
        raise SystemExit(0)
    if args.refine_existing:
        result=json.loads((OUT/"target_metrics.json").read_text(encoding="utf-8"))
        result["coarse_scene"]=result["scene"]
        source=OUT/"scene_pair.npz";destination=OUT/"scene_pair_coarse.npz"
        if source.exists() and not destination.exists():source.replace(destination)
        old=result["targets"][1]
        new=final_targets(imager,("C9",),refine=True)[0]
        new["convergence"]["previous_0125m_grid"]={
            "peak":old["bp"]["metrics"]["peak"],
            "irw_m":{d:old["bp"]["metrics"][d]["irw_m"] for d in ("range","azimuth")}}
        interpolation_convergence(imager,[new])
        result["targets"][1]=new
        write_json(OUT/"target_metrics.json",result)
        result["scene"]=scene_pair(imager,1.0,4)
        result["refinement_seconds"]=time.perf_counter()-start
        write_json(OUT/"target_metrics.json",result)
        print("refinement completed",result["refinement_seconds"],flush=True)
        raise SystemExit(0)
    rows=final_targets(imager)
    result=dict(metadata=imager.meta,targets=rows,scene={})
    write_json(OUT/"target_metrics.json",result)
    if not args.skip_convergence:interpolation_convergence(imager,rows)
    write_json(OUT/"target_metrics.json",result)
    if not args.skip_scene:result["scene"]=scene_pair(imager)
    result["execution_seconds"]=time.perf_counter()-start
    write_json(OUT/"target_metrics.json",result)
    print("completed",result["execution_seconds"],flush=True)
