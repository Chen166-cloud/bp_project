"""Locate and measure strong scatterers in oversampled BP/FFBP patches."""
from __future__ import annotations
import argparse
import json
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from imaging import Imager, OUT, axes, write_json, save_image, fidelity, PARAMETERS, C
from quality import analyze_target

def local_axes(plan,rc,uc):
    r,u=axes(plan)
    # First-order physical arc length at the reference pixel.  Error is <1e-5
    # over the small target patches; r remains slant range, never ground range.
    az=(u-uc)*rc/np.sqrt(1-uc*uc)
    return r,az

def measure(plan,z,rc,uc,half=50):
    r,az=local_axes(plan,rc,uc)
    return analyze_target(r,az,z,expected_irw=(6.5,8.0),peak_hint=(rc,0),
        search_radius_m=(18,18),integration_half_widths_m=(half,half),return_profiles=True)

def scout(args):
    overview=json.loads((OUT/"overview.json").read_text(encoding="utf-8"))
    # Restrict to one Doppler ambiguity interval about the selected beam branch.
    uc=overview["metadata"]["sin_squint"]
    cand=[c for c in overview["candidates"] if abs(c["u"]-uc)<.0021][args.offset:args.offset+args.count]
    imager=Imager()
    rows=[];fig,axs=plt.subplots(2,3,figsize=(14,9))
    for i,c in enumerate(cand):
        rc,u0=c["r_m"],c["u"]
        du=85*np.sqrt(1-u0*u0)/rc
        plan=imager.plan((rc-85,rc+85,u0-du,u0+du),.5,6,64)
        number=i+1+args.offset
        print("scout",number,plan.shape,flush=True)
        z,timing=imager.image(plan,"bp")
        m,cuts=measure(plan,z,rc,u0)
        r,az=local_axes(plan,rc,u0)
        save_image(OUT/f"candidate_{number:02d}.npz",z,plan,azimuth_m=az)
        item=dict(id=f"C{number}",candidate=c,metrics=m,timing=timing)
        rows.append(item)
        ax=axs.ravel()[i]
        db=20*np.log10(np.maximum(abs(z)/abs(z).max(),1e-6))
        ax.imshow(db,origin="lower",aspect="equal",cmap="gray",vmin=-40,vmax=0,
            extent=[az[0],az[-1],r[0]-rc,r[-1]-rc])
        ax.set(title=f"C{number}: range {m['range'].get('irw_m',float('nan')):.2f} m, az {m['azimuth'].get('irw_m',float('nan')):.2f} m",
               xlabel="Local azimuth (m)",ylabel="Slant-range offset (m)")
        print(json.dumps(dict(id=f"C{number}",irw_range=m["range"].get("irw_m"),
            irw_azimuth=m["azimuth"].get("irw_m"),valid_range=m["range"]["sidelobes_valid"],
            valid_azimuth=m["azimuth"]["sidelobes_valid"]),ensure_ascii=False),flush=True)
    for ax in axs.ravel()[len(cand):]:ax.set_visible(False)
    suffix="" if args.offset==0 else f"_{args.offset}"
    fig.tight_layout();fig.savefig(OUT/f"candidate_scout{suffix}.png",dpi=170);plt.close(fig)
    write_json(OUT/f"candidate_scout{suffix}.json",rows)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("stage",choices=["scout"])
    p.add_argument("--count",type=int,default=6)
    p.add_argument("--offset",type=int,default=0)
    scout(p.parse_args())
