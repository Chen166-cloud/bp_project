from pathlib import Path
import json
import numpy as np
from scipy import signal,fft
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sar_io import Capture,trajectory,range_compress,C,FS,write_json

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'fast_bp'/'results';OUT.mkdir(parents=True,exist_ok=True)
cap=Capture(ROOT/'2018-08-14_18-54-08.dat')
pos,geo=trajectory(cap)
cap.summary['trajectory']=geo
for k in ['heading','pitch','roll','alt','ve','vn','vu']:
    a=cap.nav[k];cap.summary[k]=dict(min=float(a.min()),median=float(np.median(a)),max=float(a.max()))
print(json.dumps(cap.summary,indent=2),flush=True)
write_json(OUT/'capture_audit.json',cap.summary)
np.savez(OUT/'navigation.npz',position=pos,time=cap.time,**cap.nav)
fig,axs=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
axs[0,0].plot(pos[:,1],pos[:,0]);axs[0,0].set(xlabel='Along-track (m)',ylabel='Cross-track left (m)',title='Integrated measured velocity')
for k in ['ve','vn','vu']:axs[0,1].plot(cap.time,cap.nav[k],label=k)
axs[0,1].legend();axs[0,1].set(xlabel='Time (s)',ylabel='m/s',title='Measured velocity')
axs[1,0].plot(cap.time,cap.nav['alt']-cap.nav['alt'][0],label='Recorded altitude change')
axs[1,0].plot(cap.time,pos[:,2]-18,label='Integrated vertical velocity');axs[1,0].legend();axs[1,0].set(xlabel='Time (s)',ylabel='m')
for k in ['heading','pitch','roll']:axs[1,1].plot(cap.time,cap.nav[k]-np.median(cap.nav[k]),label=k)
axs[1,1].legend();axs[1,1].set(xlabel='Time (s)',ylabel='Deviation from median (deg)',title='Attitude variation')
fig.savefig(OUT/'navigation.png',dpi=150);plt.close(fig)
idx=np.linspace(0,cap.n-1,512).astype(int)
raw=cap.read(idx)
r=C/2*(cap.delay+np.arange(cap.nr)/FS)
fig,axs=plt.subplots(3,1,figsize=(14,10),constrained_layout=True)
axs[0].plot(r,10*np.log10(np.mean(abs(raw)**2,axis=0)+1e-12));axs[0].set(title='Raw return power',ylabel='dB',xlabel='Nominal slant range (m)')
summ={}
for ax,sgn in zip(axs[1:],[-1,1]):
    rc=range_compress(raw,chirp_sign=sgn)
    db=20*np.log10(abs(rc)+1e-9); db-=np.max(db)
    ax.imshow(db,extent=[r[0],r[-1],cap.time[-1],0],aspect='auto',cmap='gray',vmin=-55,vmax=0)
    ax.set(title=f'LFM matched filter: chirp sign {sgn:+d}',ylabel='Slow time (s)',xlabel='Nominal slant range (m)')
    power=np.mean(abs(rc)**2,axis=0)
    peaks,_=signal.find_peaks(power,distance=200)
    peaks=peaks[np.argsort(power[peaks])[-15:]]
    summ[str(sgn)]=dict(peak=float(abs(rc).max()),fourth_moment=float(np.mean(abs(rc)**4)/np.mean(abs(rc)**2)**2),strong_ranges_m=r[peaks].tolist())
    np.save(OUT/f'rc_diagnostic_{sgn}.npy',rc)
fig.savefig(OUT/'range_compression_diagnostic.png',dpi=150);plt.close(fig)
write_json(OUT/'chirp_comparison.json',summ);print(summ,flush=True)
# A continuous segment is required for Doppler, not sparsely spaced pulses.
raw=cap.read(np.arange(4000,6048))
rc=range_compress(raw,chirp_sign=-1)
spec=np.mean(abs(fft.fftshift(fft.fft(rc*signal.windows.hann(len(rc))[:,None],axis=0,workers=8),axes=0))**2,axis=1)
f=fft.fftshift(fft.fftfreq(len(rc),1/cap.prf))
fig,ax=plt.subplots(figsize=(12,4),constrained_layout=True);ax.plot(f,10*np.log10(spec/spec.max()+1e-12));ax.set(xlabel='Doppler (Hz)',ylabel='Normalized power (dB)',ylim=(-80,2),title='Measured Doppler spectrum: 2048 contiguous pulses')
fig.savefig(OUT/'doppler.png',dpi=150)
print('Doppler peak',f[np.argmax(spec)],flush=True)
