"""Independent point-scatterer tests and numerical equivalence to direct BP."""
from pathlib import Path
import time
import numpy as np
from scipy.signal.windows import hann
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sar_io import C,FC,FS,range_compress,write_json
from ffbp import form_ffbp,backproject_points,evaluate_image

OUT=Path(__file__).resolve().parent/'results';OUT.mkdir(exist_ok=True)
k=4*np.pi*FC/C
# Independently generated analytic range-compressed point scatterers, with
# off-grid ranges, nonuniform track, cross-track and vertical motion.
n=512;t=np.linspace(-1,1,n)
pos=np.column_stack([.08*np.sin(5*t),50*t+.4*t*t,18+.15*np.sin(2*t)])
targets=np.array([[710.31,-145.27,0.],[987.21,31.72,0.],[1190.77,-380.11,0.]])
strength=np.array([1,complex(.7,.3),complex(.5,-.2)])
r_start=500.;ddr=C/(2*FS)
rr=r_start+ddr*np.arange(2300)
data=np.zeros((n,len(rr)),np.complex64)
for xyz,a in zip(targets,strength):
    distance=np.linalg.norm(pos-xyz,axis=1)
    data+=(a*np.exp(-1j*k*distance)[:,None]*np.sinc((rr[None,:]-distance[:,None])/(C/(2*200e6)))).astype(np.complex64)
w=hann(n);phase=np.zeros(n)
allxyz=[]
for xyz in targets:
    x=xyz[0]+np.arange(-3,3.01,.15);y=xyz[1]+np.arange(-8,8.01,.2)
    xx,yy=np.meshgrid(x,y);allxyz.append(np.column_stack([xx.ravel(),yy.ravel(),np.zeros(xx.size)]))
points=np.vstack(allxyz)
t0=time.perf_counter();direct=backproject_points(data,pos,points,r_start,ddr,k,w,phase)
direct_seconds=time.perf_counter()-t0
results=[];fig,axs=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
for ax,leaf in zip(axs,[128,64,32]):
    result,counts=form_ffbp(data,pos,r_start,ddr,FC,(650,1300,-.5,.18),leaf_size=leaf,
                              range_spacing=.25,angular_oversample=3.,weight=w,verbose=False)
    fast=evaluate_image(result.image,result.origin,result.r0,result.dr,result.u0,result.du,points,k)
    err=float(np.linalg.norm(fast-direct)/np.linalg.norm(direct))
    corr=float(abs(np.vdot(fast,direct))/(np.linalg.norm(fast)*np.linalg.norm(direct)))
    peak_errors=[]
    cuts=np.split(fast,3);dcuts=np.split(direct,3)
    for xyz,p,v in zip(targets,allxyz,cuts):peak_errors.append(float(np.linalg.norm(p[np.argmax(abs(v))]-xyz)))
    rec=dict(leaf_size=leaf,relative_complex_error=err,complex_coherence=corr,peak_position_errors_m=peak_errors,**counts)
    results.append(rec);print(rec,flush=True)
    assert err<.12,rec
    assert corr>.99,rec
    assert max(peak_errors)<.3,rec
    v=abs(cuts[0]).reshape(81,41)[:,20];d=abs(dcuts[0]).reshape(81,41)[:,20]
    ax.plot(np.arange(-8,8.01,.2),20*np.log10(d/d.max()+1e-9),label='Direct BP')
    ax.plot(np.arange(-8,8.01,.2),20*np.log10(v/d.max()+1e-9),'--',label='FFBP')
    ax.set(xlabel='Along-track offset (m)',ylabel='dB',ylim=(-45,2),title=f'{counts["leaves"]} leaves; error {err:.2%}');ax.legend()
fig.savefig(OUT/'synthetic_validation.png',dpi=160);plt.close(fig)
# Test the raw LFM correlation lag separately, using a fractional sample delay.
ns=4000;sample=1512.4;tr=(np.arange(ns)-sample)/FS
raw=(np.exp(-1j*np.pi*200e6/1e-6*tr**2)*(abs(tr)<.5e-6)).astype(np.complex64)[None,:]
compressed=range_compress(raw,oversample=4)
found=np.argmax(abs(compressed[0]))/4
assert abs(found-sample)<=.25,(found,sample)
write_json(OUT/'synthetic_validation.json',dict(point_tests=results,lfm_peak_sample=float(found),expected_sample=sample,passed=True))
print('All independent synthetic checks passed.',flush=True)
