"""Independent range-Doppler diagnostic for the local RADARSAT-1 excerpt.

No autofocus. Exact hyperbolic range-Doppler phase and RCMC in a 2-D
equivalent-velocity slant plane. This is a cross-check, not calibrated truth.
Raw complex data, rather than the old MATLAB display image, are processed.
"""
from pathlib import Path
import argparse
import json
import time
import numpy as np
from scipy import fft, signal
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent/'results'
C=299792458.; F0=5.3e9; FS=32.317e6; PRF=1256.98
KR=-.72135e12; TP=41.74e-6; V=7062.; T0=.0065956


def compress(raw, chirp_sign=-1, window=False):
    count=round(TP*FS)
    times=(np.arange(count)-count//2)/FS
    ref=np.exp(1j*chirp_sign*np.pi*abs(KR)*times**2).astype(np.complex64)
    if window:
        ref*=signal.windows.kaiser(count,2.5).astype(np.float32)
    z=signal.fftconvolve(raw,ref[::-1].conj()[None,:],mode='full',axes=1)
    start=count-1-count//2
    return z[:,start:start+raw.shape[1]].copy()


def range_doppler(rc, fdc=-7040.1, carrier_sign=-1, nfft=4096,
                  beta=2.5, range_start=None):
    na,nr=rc.shape
    t=(np.arange(na)-(na-1)/2)/PRF
    q=rc*np.exp(-2j*np.pi*fdc*t[:,None]).astype(np.complex64)
    first=(nfft-na)//2
    q=np.pad(q,((first,nfft-na-first),(0,0)))
    s=fft.fftshift(fft.fft(fft.ifftshift(q,axes=0),axis=0,workers=4),axes=0)
    fa=fdc+fft.fftshift(fft.fftfreq(nfft,1/PRF))
    d=np.sqrt(1-(C*fa/(2*V*F0))**2)
    dr=C/(2*FS)
    if range_start is None: range_start=C*T0/2
    # Reference-axis labels are closest-approach slant ranges. RCMC evaluates
    # each Doppler row at its exact stationary-phase range Rclosest / D(f).
    ranges=range_start+np.arange(nr)*dr
    dc=np.sqrt(1-(C*fdc/(2*V*F0))**2)
    # The large physical Doppler centroid corresponds to a beam-centre target
    # many km away along-track. Remove its known linear spectral phase to
    # avoid circular wrap of this coordinate in the azimuth IFFT.
    yref=-carrier_sign*(C*fdc/(2*V*F0))*ranges/dc
    out=np.empty_like(s)
    bins=np.arange(nr)
    wa=np.kaiser(nfft,beta) if beta else np.ones(nfft)
    for i in range(nfft):
        loc=(ranges/d[i]-range_start)/dr
        real=np.interp(loc,bins,s[i].real,left=0.,right=0.)
        imag=np.interp(loc,bins,s[i].imag,left=0.,right=0.)
        # carrier_sign is the sign of the received propagation phase.
        phase=np.exp(-carrier_sign*1j*4*np.pi*F0/C*ranges*d[i]
                     +2j*np.pi*(fa[i]-fdc)*yref/V)
        out[i]=(real+1j*imag)*phase*wa[i]
    focused=fft.fftshift(fft.ifft(fft.ifftshift(out,axes=0),axis=0,workers=4),axes=0)
    return focused[first:first+na], ranges, t*V


def quick_quality(z):
    p=(abs(z).astype(np.float64)**2).ravel()
    prob=p/p.sum()
    return {'contrast':float(p.std()/p.mean()),
            'entropy_nats':float(-np.sum(prob[prob>0]*np.log(prob[prob>0]))),
            'peak':float(p.max()),
            'peak_index':list(map(int,np.unravel_index(p.argmax(),z.shape)))}


def synthetic_coordinate_check(fdc=-7040.1):
    """Independent analytic raw LFM target, with fractional range migration."""
    na,nr=1728,3200
    x=C*T0/2+1781*C/(2*FS)
    local_y=(858-(na-1)/2)*V/PRF
    dc=np.sqrt(1-(C*fdc/(2*V*F0))**2)
    y=(C*fdc/(2*V*F0))*x/dc+local_y
    slow=(np.arange(na)-(na-1)/2)/PRF
    ranges=np.sqrt(x*x+(V*slow-y)**2)
    fast=T0+np.arange(nr)/FS
    delta=fast[None,:]-2*ranges[:,None]/C
    raw=(np.exp(1j*np.pi*KR*delta**2-4j*np.pi*F0/C*ranges[:,None])
         *(abs(delta)<=TP/2)).astype(np.complex64)
    z,r,az=range_doppler(compress(raw,-1,True),fdc,-1,4096,2.5)
    peak=np.unravel_index(abs(z).argmax(),z.shape)
    return {'expected_closest_slant_range_m':float(x),
            'expected_local_y_m':float(local_y),
            'measured_closest_slant_range_m':float(r[peak[1]]),
            'measured_local_y_m':float(az[peak[0]]),
            'peak_index':list(map(int,peak)),
            'range_error_m':float(r[peak[1]]-x),
            'along_error_m':float(az[peak[0]]-local_y)}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--fdc',type=float,default=-7040.1)
    ap.add_argument('--nfft',type=int,default=4096)
    ap.add_argument('--beta',type=float,default=2.5)
    ap.add_argument('--chirp-sign',type=int,default=-1)
    ap.add_argument('--validate',action='store_true')
    args=ap.parse_args()
    OUT.mkdir(exist_ok=True,parents=True)
    raw=loadmat(ROOT/'CDdata1.mat')['data'].astype(np.complex64)
    pair=np.angle(np.sum(raw[1:]*raw[:-1].conj()))*PRF/(2*np.pi)
    rc=compress(raw,args.chirp_sign,True)
    np.save(OUT/'reference_rc.npy',rc)
    record={'raw_shape':list(raw.shape),'baseband_pulsepair_hz':float(pair),
            'fdc_hz':args.fdc,'doppler_ambiguity_assumption':-6,
            'chirp_sign':args.chirp_sign,'no_autofocus':True,
            'geometry':'equivalent-velocity hyperbola; no absolute geolocation',
            'along_track_axis':'local y; physical y = local_y - carrier_sign*(lambda*fdc/(2V))*closest_slant_range/D(fdc)',
            'rc_time_origin':'output sample k is raw pulse center sample k',
            'parameters':{'c':C,'carrier_hz':F0,'fs_hz':FS,'prf_hz':PRF,
                          'kr_hz_s':KR,'tp_s':TP,'v_m_s':V,'t0_s':T0},
            'variants':{}}
    fig,axs=plt.subplots(1,2,figsize=(14,7))
    for ax,sgn in zip(axs,[-1,1]):
        ts=time.perf_counter()
        z,r,y=range_doppler(rc,args.fdc,sgn,args.nfft,args.beta)
        quality=quick_quality(z)
        quality['elapsed_s']=time.perf_counter()-ts
        record['variants'][str(sgn)]=quality
        np.savez_compressed(OUT/f'reference_rda_sign{sgn}.npz',image=z,
                            closest_slant_range_m=r,along_track_m=y,
                            along_track_local_m=y,
                            reference_y_m=-sgn*(C*args.fdc/(2*V*F0))*r/
                                np.sqrt(1-(C*args.fdc/(2*V*F0))**2),
                            carrier_sign=sgn,fdc_hz=args.fdc)
        db=20*np.log10(abs(z)/abs(z).max()+1e-9)
        ax.imshow(db,origin='lower',aspect='auto',cmap='gray',vmin=-45,vmax=0,
                  extent=(r[0]/1000,r[-1]/1000,y[0]/1000,y[-1]/1000))
        ax.set(xlabel='Closest slant range (km)',ylabel='Along track (km)',
               title=f'Received carrier sign {sgn}; C={quality["contrast"]:.2f}')
        print(sgn,quality,flush=True)
    fig.tight_layout()
    fig.savefig(OUT/'reference_rda_sign_comparison.png',dpi=160)
    plt.close(fig)
    if args.validate:
        record['synthetic_coordinate_check']=synthetic_coordinate_check(args.fdc)
    (OUT/'reference_rda_diagnostic.json').write_text(json.dumps(record,indent=2),encoding='utf-8')


if __name__=='__main__': main()
