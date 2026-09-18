"""Read the original acquisition; no MAT intermediates are used for imaging.

Binary offsets follow ../read_rawdata_1.m. All offsets are bytes from the
32-byte start marker. Header: 160 bytes; interleaved int16 IQ; trailer: 32.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy import fft, signal, integrate

C = 299792458.0
FC = 1.4e9
FS = 400e6
MAGIC = bytes([0, 52, 0, 52, 243, 250, 243, 250])


class Capture:
    def __init__(self, path):
        self.path = Path(path)
        with self.path.open('rb') as f:
            prefix = f.read(10_000_000)
        self.offset = prefix.find(MAGIC)
        if self.offset < 0:
            raise ValueError('Frame marker not found in first 10 MB')
        self.nr = round(int(np.frombuffer(prefix, '<i2', 1, self.offset+60)[0])*1e-8*FS)
        self.frame_size = 192 + self.nr*4
        count = (self.path.stat().st_size-self.offset)//self.frame_size
        fields = [
            ('magic','u1',(8,),0), ('number','<u4',(),36), ('bandwidth','<u4',(),40),
            ('tp','<i2',(),48), ('delay','<i2',(),50), ('prt','<u4',(),52),
            ('window','<i2',(),60), ('heading','<i2',(),71),
            ('pitch','<i2',(),73), ('roll','<i2',(),75),
            ('lat','<i8',(),101), ('lon','<i8',(),109), ('alt','<i4',(),117),
            ('ve','<i4',(),121), ('vn','<i4',(),125), ('vu','<i4',(),129),
            ('lat_gnss','<i8',(),133), ('lon_gnss','<i8',(),141),
            ('gps_ms','<u4',(),149), ('sv','u1',(),155),
            ('iq','<i2',(self.nr,2),160), ('trailer','u1',(32,),160+4*self.nr)]
        dt = np.dtype({'names':[x[0] for x in fields],
                       'formats':[(x[1],x[2]) if x[2] else x[1] for x in fields],
                       'offsets':[x[3] for x in fields], 'itemsize':self.frame_size})
        all_frames = np.memmap(path, dtype=dt, mode='r', offset=self.offset, shape=(count,))
        good = np.all(all_frames['magic']==np.frombuffer(MAGIC,'u1'),axis=1)
        self.valid_indices = np.flatnonzero(good)
        if not len(self.valid_indices):
            raise ValueError('No valid frames')
        # Retain valid frames in file order; gaps in pulse numbering are
        # preserved in slow time rather than squeezed out.
        self.frames = all_frames
        self.n = len(self.valid_indices)
        self.nav = {}
        scale = {'lat':1e-9,'lon':1e-9,'lat_gnss':1e-9,'lon_gnss':1e-9,
                 'alt':1e-3,'ve':.01,'vn':.01,'vu':.01,
                 'heading':.01,'pitch':.01,'roll':.01}
        for k in ['number','gps_ms','sv',*scale]:
            self.nav[k] = np.asarray(all_frames[k][good],dtype=np.float64)*scale.get(k,1)
        for k in ['tp','delay','prt','bandwidth','window']:
            v=np.asarray(all_frames[k][good])
            if len(np.unique(v))!=1:
                raise ValueError(f'Changing {k} requires per-pulse handling')
        f=all_frames[self.valid_indices[0]]
        self.tp=float(f['tp'])*1e-8
        self.delay=float(f['delay'])*1e-8-.2e-6
        self.prf=1/(float(f['prt'])*1e-8)
        self.bandwidth=float(f['bandwidth'])
        self.time=(self.nav['number']-self.nav['number'][0])/self.prf
        if np.any(np.diff(self.time)<=0):
            raise ValueError('Non-monotonic pulse numbering')
        self.summary=dict(file=str(self.path.resolve()),bytes=self.path.stat().st_size,
            first_frame_offset=self.offset,frame_bytes=self.frame_size,
            complete_frame_slots=count,valid_frames=self.n,invalid_slots=int((~good).sum()),
            first_pulse=int(self.nav['number'][0]),last_pulse=int(self.nav['number'][-1]),
            missing_pulses=int(np.sum(np.diff(self.nav['number'])-1)),
            range_samples=self.nr,sample_rate_hz=FS,carrier_hz=FC,
            bandwidth_hz=self.bandwidth,pulse_width_s=self.tp,
            sampling_delay_s=self.delay,prf_hz=self.prf,duration_s=float(self.time[-1]),
            discarded_partial_bytes=int((self.path.stat().st_size-self.offset)%self.frame_size))

    def read(self, indices):
        q=self.frames['iq'][self.valid_indices[np.asarray(indices)]]
        return q[:,:,0].astype(np.float32)+1j*q[:,:,1].astype(np.float32)


def trajectory(capture, height=18.0, mode='velocity'):
    """ENU -> cross-track left/along-track/up, with scene flat at z=0.

Velocity integration avoids 100 Hz sample-and-hold jumps in position records.
Absolute latitude/longitude are reserved for diagnostics, not phase correction.
The mean measured velocity defines along-track; +cross-track is left of travel.
"""
    n,t=capture.nav,capture.time
    velocity=np.column_stack([n['ve'],n['vn'],n['vu']])
    enu=integrate.cumulative_trapezoid(velocity,t,axis=0,initial=0)
    if mode=='position':
        # Position updates occur at 100 Hz and are repeated across ten pulses.
        # Interpolate at the update indices, preserving the full pulse timeline.
        idx=np.r_[0,np.flatnonzero(np.diff(n['gps_ms']))+1]
        lat0=np.deg2rad(n['lat'][0]); earth_a=6378137.; e2=6.69437999014e-3
        rn=earth_a/np.sqrt(1-e2*np.sin(lat0)**2)
        rm=earth_a*(1-e2)/(1-e2*np.sin(lat0)**2)**1.5
        ll=np.column_stack([np.deg2rad(n['lon']-n['lon'][0])*rn*np.cos(lat0),
                            np.deg2rad(n['lat']-n['lat'][0])*rm,n['alt']-n['alt'][0]])
        enu=np.column_stack([np.interp(t,t[idx],ll[idx,k]) for k in range(3)])
    elif mode=='straight':
        enu=t[:,None]*velocity.mean(axis=0)
    along=velocity[:,:2].mean(axis=0); along/=np.linalg.norm(along)
    left=np.array([-along[1],along[0]])
    xy=enu[:,:2]-enu[len(t)//2,:2]
    pos=np.column_stack([xy@left,xy@along,enu[:,2]+height])
    return pos,dict(mode=mode,height_reference_m=height,along_enu=along.tolist(),
                   left_enu=left.tolist(),mean_speed_m_s=float(np.linalg.norm(velocity[:,:2],axis=1).mean()),
                   mean_course_deg=float(np.rad2deg(np.arctan2(along[0],along[1]))%360),
                   extent_m=np.ptp(pos,axis=0).tolist())


def range_compress(raw, tp=1e-6, bandwidth=200e6, chirp_sign=-1, oversample=1):
    """Linear correlation with a centred LFM reference, including exact lag.

Reference samples at (n-floor(L/2))/fs; sample k of returned output corresponds
to received pulse CENTRE at raw sample k. Thus R[k]=c*(delay+k/fs)/2.
FFT padding prevents circular convolution. Frequency zero padding interpolates
the complex compressed range profile, without changing the physical range.
"""
    l=round(tp*FS)
    tr=(np.arange(l)-l//2)/FS
    ref=np.exp(1j*np.pi*chirp_sign*bandwidth/tp*tr**2).astype(np.complex64)
    # Hamming reduces range sidelobes; do not window the entire fast-time swath.
    ref*=signal.windows.hamming(l,sym=False).astype(np.float32)
    nfft=fft.next_fast_len(raw.shape[1]+l-1)
    h=np.conj(ref[::-1])
    spectrum=fft.fft(raw-raw.mean(axis=1,keepdims=True),nfft,axis=1,workers=8)
    spectrum*=fft.fft(h,nfft)[None,:]
    offset=l-1-l//2
    if oversample==1:
        return fft.ifft(spectrum,axis=1,workers=8)[:,offset:offset+raw.shape[1]].copy()
    # scipy.signal.resample is Fourier interpolation with correct Nyquist.
    corr=fft.ifft(spectrum,axis=1,workers=8)
    corr=signal.resample(corr,nfft*oversample,axis=1)
    return corr[:,offset*oversample:(offset+raw.shape[1])*oversample].copy()


def write_json(path, content):
    Path(path).write_text(json.dumps(content,indent=2,ensure_ascii=False),encoding='utf-8')
