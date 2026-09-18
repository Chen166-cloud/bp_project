"""Recover known simulated phase-centre errors without fitting image pixels."""
from pathlib import Path
import numpy as np
from scipy import signal
from sar_io import C,FC,FS,write_json
from autofocus import estimate_update

n=4096;t=np.linspace(0,4,n)
truth=np.column_stack([np.zeros(n),np.linspace(-70,70,n),np.full(n,18.)])
targets=np.array([[710.,-290.,0.],[720.,320.,0.],[810.,-100.,0.],[870.,280.,0.]])
k=-4*np.pi*FC/C
err=np.column_stack([.06*np.sin(2*np.pi*t),.12*np.cos(np.pi*t),np.zeros(n)])
err[:,:2]=signal.detrend(err[:,:2],axis=0)
r0=650.;dr=C/(2*FS);rr=r0+dr*np.arange(900);data=np.zeros((n,len(rr)),np.complex64)
for target in targets:
    d=np.linalg.norm(truth-target,axis=1)
    data+=(np.exp(-1j*k*d)[:,None]*np.sinc((rr[None,:]-d[:,None])/(C/(400e6)))).astype(np.complex64)
delta,stats=estimate_update(data,truth+err,targets,r0,dr,k,1024,15.)
before=float(np.sqrt(np.mean(err[:,:2]**2)));after=float(np.sqrt(np.mean((err+delta)[:,:2]**2)))
assert after<.35*before
stats.update(initial_coordinate_rmse_m=before,corrected_coordinate_rmse_m=after,passed=True,
             scope='synthetic known targets; does not establish true navigation accuracy on the real collection')
write_json(Path(__file__).resolve().parent/'results/autofocus_validation.json',stats)
print(stats,flush=True)
