"""Wall-clock FFBP vs direct BP on exactly the same real-data ROI/grid."""
from pathlib import Path
import time
import numpy as np
from scipy.signal.windows import hann
from sar_io import Capture,C,FS,FC,write_json
from ffbp import form_ffbp,bp_polar

root=Path(__file__).resolve().parent.parent;out=root/'fast_bp/results'
cap=Capture(root/'2018-08-14_18-54-08.dat')
data=np.load(out/'range_compressed.npy',mmap_mode='r')
pos=np.load(out/'autofocus_position.npy');w=hann(cap.n);ph=np.zeros(cap.n)
r0=C*(cap.delay-cap.tp/2)/2;dr=C/(2*FS);k=-4*np.pi*FC/C
# Warm both kernels; the benchmark excludes JIT and input range compression.
bp_polar(data[:2],pos[:2],pos[:2].mean(0),1400.,.25,3,-.02,.01,3,r0,dr,k,w[:2],ph[:2],1)
result,_=form_ffbp(data,pos,r0,dr,-FC,(1380,1470,-.03,.03),leaf_size=64,
                    range_spacing=.25,angular_oversample=3.,weight=w,verbose=False)
t=time.perf_counter()
fast,counts=form_ffbp(data,pos,r0,dr,-FC,(1380,1470,-.03,.03),leaf_size=64,
                    range_spacing=.25,angular_oversample=3.,weight=w,verbose=False)
fast_time=time.perf_counter()-t
t=time.perf_counter()
direct=bp_polar(data,pos,fast.origin,fast.r0,fast.dr,fast.image.shape[0],fast.u0,fast.du,
                fast.image.shape[1],r0,dr,k,w,ph,1)
direct_time=time.perf_counter()-t
error=float(np.linalg.norm(fast.image-direct)/np.linalg.norm(direct))
coherence=float(abs(np.vdot(fast.image,direct))/(np.linalg.norm(fast.image)*np.linalg.norm(direct)))
record=dict(pulses=cap.n,image_shape=list(direct.shape),ffbp_seconds=fast_time,
    direct_bp_seconds=direct_time,speedup=direct_time/fast_time,relative_complex_error=error,
    complex_coherence=coherence,threads=16,hardware='Intel Core i9-14900HX, CPU only',
    timing_scope='Same input pulses and output grid; warmed kernels; excludes range compression, JIT, disk export and autofocus')
write_json(out/'benchmark.json',record);print(record,flush=True)
assert error<.15 and coherence>.99
