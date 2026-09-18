"""Direct BP diagnostic, on a modest region before running the full FFBP."""
from pathlib import Path
import time,json
import numpy as np
from scipy.signal.windows import hann
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sar_io import Capture,trajectory,range_compress,C,FS,FC,write_json
from ffbp import backproject_points,polar_points

ROOT=Path(__file__).resolve().parent.parent;OUT=ROOT/'fast_bp/results'
cap=Capture(ROOT/'2018-08-14_18-54-08.dat');pos,_=trajectory(cap)
idx=np.arange(3000,9000,3) # Coarse sign/side diagnostic only; final imaging uses every valid pulse.
rc=range_compress(cap.read(idx));p=pos[idx];w=hann(len(idx));phase=np.zeros(len(idx))
origin=p.mean(axis=0);r=np.arange(550.,2500.,4.);u=np.linspace(-.8,.6,451)
fig,axs=plt.subplots(2,2,figsize=(15,10),constrained_layout=True)
metrics=[]
for ax,(side,sign) in zip(axs.ravel(),[(1,1),(-1,1),(1,-1),(-1,-1)]):
    xyz=polar_points(origin,r,u,side);t=time.perf_counter()
    im=backproject_points(rc,p,xyz,C*cap.delay/2,C/(2*FS),sign*4*np.pi*FC/C,w,phase).reshape(len(r),len(u))
    amp=abs(im);power=amp**2;contrast=float(np.mean(power**2)/np.mean(power)**2)
    pk=np.unravel_index(amp.argmax(),amp.shape)
    row=dict(side=side,phase_sign=sign,seconds=time.perf_counter()-t,contrast=contrast,peak_range=float(r[pk[0]]),peak_u=float(u[pk[1]]),peak=float(amp.max()))
    metrics.append(row);print(row,flush=True)
    db=20*np.log10(amp+1e-6);db-=db.max()
    ax.imshow(db.T,origin='lower',extent=[r[0],r[-1],u[0],u[-1]],aspect='auto',cmap='gray',vmin=-40,vmax=0)
    ax.set(title=f'Cross-track side {side:+d}, phase sign {sign:+d}, contrast {contrast:.1f}',xlabel='Slant range (m)',ylabel='sin(look angle)')
    np.savez(OUT/f'pilot_side{side}_sign{sign}.npz',image=im,r=r,u=u,origin=origin)
fig.savefig(OUT/'pilot_comparison.png',dpi=160);write_json(OUT/'pilot_comparison.json',metrics)
