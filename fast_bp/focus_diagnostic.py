"""Evaluate timing and trajectory alternatives on the same strong-return ROI."""
from pathlib import Path
import numpy as np
from scipy.signal.windows import hann
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sar_io import Capture,trajectory,C,FS,FC,write_json
from ffbp import form_ffbp

ROOT=Path(__file__).resolve().parent.parent;OUT=ROOT/'fast_bp/results'
cap=Capture(ROOT/'2018-08-14_18-54-08.dat');rc=np.load(OUT/'range_compressed.npy',mmap_mode='r')
fig,axs=plt.subplots(3,4,figsize=(16,11),constrained_layout=True)
rows=[]
for i,mode in enumerate(['velocity','position','straight']):
    pos,_=trajectory(cap,mode=mode)
    for j,shift in enumerate([-149.896229,-74.9481145,0.,74.9481145]):
        result,counts=form_ffbp(rc,pos,C*cap.delay/2+shift,C/(2*FS),-FC,
            (1420+shift,1560+shift,-.06,.06),range_spacing=.375,angular_oversample=3,
            leaf_size=64,weight=hann(cap.n),verbose=False)
        power=abs(result.image)**2
        pk=np.unravel_index(power.argmax(),power.shape)
        p=power/power.sum()
        rec=dict(mode=mode,range_shift=shift,contrast=float(np.mean(power**2)/np.mean(power)**2),
            entropy=float(-np.sum(p*np.log(p+1e-30))),peak=float(np.sqrt(power.max())),
            peak_r=float(result.r[pk[0]]),peak_u=float(result.u[pk[1]]),seconds=counts['elapsed_s'])
        rows.append(rec);print(rec,flush=True)
        db=10*np.log10(power+1e-20);db-=db.max()
        axs[i,j].imshow(db.T,origin='lower',aspect='auto',cmap='gray',vmin=-35,vmax=0,
            extent=[result.r[0],result.r[-1],result.u[0],result.u[-1]])
        axs[i,j].set(title=f'{mode}, offset {shift:.1f} m\ncontrast {rec["contrast"]:.1f}',xlabel='Slant range (m)',ylabel='sin(angle)')
fig.savefig(OUT/'focus_diagnostic.png',dpi=160);write_json(OUT/'focus_diagnostic.json',rows)
