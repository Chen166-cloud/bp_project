"""Verify no-autofocus execution from raw input through saved BP/FFBP results.

Rebuilds range compression using an independent convolution call, checks actual
CPU/GPU phase and positions, re-images both target patches, and checks the full
scene FFBP plus independent CPU samples of the scene BP. Does not infer any
unknown processing before the supplied MAT file was created.
"""
from __future__ import annotations
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
from scipy import fft,signal
from scipy.io import loadmat
from imaging import Imager, OUT, ROOT, PARAMETERS, geometry, fidelity, write_json, axes, cp
from ffbp import backproject_points
from targets import measure

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def array_error(a,b):
    return float(np.linalg.norm((a-b).ravel())/np.linalg.norm(b.ravel()))

def fresh_range_compression():
    raw=loadmat(ROOT/'CDdata1.mat')['data'].astype(np.complex64)
    p=PARAMETERS;length=round(p['sample_rate_hz']*p['pulse_duration_s'])
    t=(np.arange(length)-length//2)/p['sample_rate_hz']
    ref=np.exp(1j*np.pi*p['chirp_rate_hz_s']*t*t).astype(np.complex64)
    ref*=signal.windows.hamming(length,sym=False).astype(np.float32)
    linear=signal.fftconvolve(raw,np.conj(ref[::-1])[None,:],mode='full',axes=1)
    nf=fft.next_fast_len(raw.shape[1]+length-1)
    padded=np.zeros((len(raw),nf),np.complex64);padded[:,:linear.shape[1]]=linear
    up=signal.resample(padded,nf*4,axis=1)
    offset=(length-1-length//2)*4
    return np.ascontiguousarray(up[:,offset:offset+raw.shape[1]*4])

def main():
    start=time.perf_counter()
    previous=json.loads((OUT/'target_metrics.json').read_text(encoding='utf-8'))
    record=dict(scope='Processing from existing local CDdata1.mat; pre-MAT history remains unknown',
        input_file=str(ROOT/'CDdata1.mat'),input_sha256=digest(ROOT/'CDdata1.mat'),
        no_pga=True,no_other_autofocus=True,checks={},targets=[],scene={})
    assert record['input_sha256']==previous['metadata']['raw_sha256']
    rebuilt=fresh_range_compression()
    cached=np.load(OUT/'range_compressed_os4_hamming.npy',mmap_mode='r')
    error=array_error(rebuilt,cached)
    record['checks']['independent_raw_to_cache']=dict(relative_complex_error=error,
        passed=error<2e-6,method='linear fftconvolve, explicit zero pad and Fourier upsampling; no phase correction')
    assert error<2e-6, 'Cached compressed data differ from fresh non-autofocused compression'
    print('raw-to-cache relative error',error,flush=True)
    imager=Imager();imager.data=rebuilt
    imager.assert_no_autofocus()
    record['checks']['host']=dict(additional_phase_max_abs_rad=float(abs(imager.phase).max()),
        fixed_trajectory_exact=bool(np.array_equal(imager.position,geometry(len(rebuilt)))),
        input_pulses=len(rebuilt),pga_module_loaded=any('autofocus' in name.lower() or name.lower()=='pga' for name in sys.modules))
    assert record['checks']['host']['additional_phase_max_abs_rad']==0
    assert not record['checks']['host']['pga_module_loaded']
    # Demonstrate that a later accidental correction is rejected at execution.
    imager.phase[0]=.01
    try:
        imager.assert_no_autofocus()
        raise AssertionError('Nonzero-phase guard did not trigger')
    except ValueError:record['checks']['nonzero_phase_rejected']=True
    finally:imager.phase[0]=0
    imager.position[0,1]+=.01
    try:
        imager.assert_no_autofocus()
        raise AssertionError('Trajectory guard did not trigger')
    except ValueError:record['checks']['changed_trajectory_rejected']=True
    finally:imager.position=geometry(len(rebuilt))
    for target in previous['targets']:
        rc,uc=target['center']['r_m'],target['center']['u'];cfg=target['ffbp_parameters']
        du=70*np.sqrt(1-uc*uc)/rc
        plan=imager.plan((rc-70,rc+70,uc-du,uc+du),cfg['range_spacing_m'],cfg['angular_oversample'],cfg['leaf_size'])
        imager.upload(plan)
        phase=cp.asnumpy(imager.device.phase64);pos=cp.asnumpy(imager.device.pos64)
        assert np.all(phase==0) and np.array_equal(pos,geometry(len(rebuilt)))
        results={}
        with np.load(OUT/f"target_{target['id']}.npz") as old:
            for alg in ('bp','ffbp'):
                z,timing=imager.image(plan,alg)
                met,_=measure(plan,z,rc,uc)
                err=array_error(z,old[alg])
                delta={d:{key:float(met[d][key]-target[alg]['metrics'][d][key])
                    for key in ('irw_m','pslr_db','islr_db')} for d in ('range','azimuth')}
                results[alg]=dict(saved_image_relative_complex_error=err,metric_differences=delta,
                    metrics=met,timing=timing,passed=err<3e-6)
                assert err<3e-6,(target['id'],alg,err)
                del z
        record['targets'].append(dict(id=target['id'],device_phase_max_abs_rad=float(abs(phase).max()),
            fixed_device_trajectory_exact=True,results=results))
        print('target checks',target['id'],{a:results[a]['saved_image_relative_complex_error'] for a in results},flush=True)
    scene=previous['scene'];plan=imager.plan(scene['extent'],scene['range_spacing_m'],scene['angular_oversample'],scene['leaf_size'])
    z,_=imager.image(plan,'ffbp')
    with np.load(OUT/'scene_pair.npz') as old:
        stored_ff=old['ffbp'];err=array_error(z,stored_ff);del stored_ff,z
        assert err<3e-6,('scene_ffbp',err)
        record['scene']['ffbp_full_image_relative_error']=err
        r,u=old['r'],old['u'];rng=np.random.default_rng(20261031)
        ir=rng.integers(0,len(r),256);iu=rng.integers(0,len(u),256)
        rr,uu=r[ir],u[iu]
        xyz=np.column_stack([rr*np.sqrt(1-uu*uu),rr*uu,np.zeros(len(rr))])
        direct=backproject_points(rebuilt,imager.position,xyz,imager.meta['range_start_m'],
            imager.meta['compressed_spacing_m'],imager.k,imager.weight,imager.phase)
        direct=direct*np.exp(-1j*imager.k*rr)
        saved=old['bp'][ir,iu]
        error=array_error(direct,saved)
        assert error<3e-6,('scene_cpu_bp',error)
        record['scene'].update(bp_independent_cpu_samples=256,bp_sample_relative_error=error)
    record['source_sha256']={str(p.relative_to(ROOT)):digest(p) for p in [
        ROOT/'vancouver_bp/imaging.py',ROOT/'vancouver_bp/finalize.py',
        ROOT/'fast_bp/gpu_backend.py',ROOT/'fast_bp/gpu/kernels.cu',Path(__file__)]}
    record['passed']=True;record['elapsed_seconds']=time.perf_counter()-start
    write_json(OUT/'no_pga_check.json',record)
    print('All no-PGA checks passed',record['elapsed_seconds'],flush=True)

if __name__=='__main__':main()
