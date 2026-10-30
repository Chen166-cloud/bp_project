"""Run the no-PGA Vancouver BP/FFBP experiment. See results/REPORT.md."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import maximum_filter
from imaging import Imager, OUT, axes, write_json, save_image, fidelity

def preview(r,u,z,path,title, ref=None):
    ref = float(abs(z).max()) if ref is None else ref
    db=20*np.log10(np.maximum(abs(z)/ref,1e-6))
    rm=float(np.mean(r)); um=float(np.mean(u))
    az=(u-um)*rm/np.sqrt(1-um*um)
    fig,ax=plt.subplots(figsize=(10,7))
    im=ax.imshow(db,origin="lower",aspect="equal",cmap="gray",vmin=-45,vmax=0,
        extent=[az[0],az[-1],(r[0]-r.mean()),(r[-1]-r.mean())])
    ax.set(xlabel="Local azimuth coordinate (m)",ylabel="Slant-range offset (m)",title=title)
    fig.colorbar(im,ax=ax,label="Amplitude (dB)",shrink=.8)
    fig.tight_layout();fig.savefig(path,dpi=170);plt.close(fig)

def overview(args):
    imager=Imager(args.oversample)
    valid=imager.meta["valid_full_reference_slant_range_m"]
    uc=imager.meta["sin_squint"]
    extent=(valid[0]+180, valid[1]-180, uc-.0042, uc+.0042)
    plan=imager.plan(extent,spacing=8,angular_oversample=.4,leaf_size=64)
    print("overview",plan.shape,flush=True)
    z,timing=imager.image(plan,"bp")
    save_image(OUT/"overview_bp.npz",z,plan)
    r,u=axes(plan)
    preview(r,u,z,OUT/"overview_bp.png","RADARSAT Vancouver: direct BP, no PGA")
    power=abs(z)**2
    peaks=(power==maximum_filter(power,size=(15,15),mode="constant"))
    # Exclude image edges before ranking; dense local reimaging follows.
    peaks[:25]=False;peaks[-25:]=False;peaks[:,:25]=False;peaks[:,-25:]=False
    ii,jj=np.nonzero(peaks)
    order=np.argsort(power[ii,jj])[::-1][:35]
    candidates=[]
    for q in order:
        i,j=int(ii[q]),int(jj[q])
        candidates.append(dict(i=i,j=j,r_m=float(r[i]),u=float(u[j]),
            y_m=float(r[i]*u[j]),amplitude=float(abs(z[i,j])),
            relative_db=float(10*np.log10(power[i,j]/power.max()))))
    write_json(OUT/"overview.json",dict(metadata=imager.meta,timing=timing,candidates=candidates))
    print(json.dumps(candidates[:12],indent=2),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("stage",choices=["overview"])
    p.add_argument("--oversample",type=int,default=4)
    args=p.parse_args();overview(args)
