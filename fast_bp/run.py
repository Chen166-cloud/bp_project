"""Reproduce the real-data FFBP result from the .dat acquisition."""
from pathlib import Path
import argparse,time,json,hashlib
import numpy as np
from scipy.signal.windows import hann
from scipy.io import savemat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from sar_io import Capture,trajectory,range_compress,C,FS,FC,write_json
from ffbp import form_ffbp,backproject_points,evaluate_image,polar_points

ROOT=Path(__file__).resolve().parent.parent


def cached_compression(cap,out):
    path=out/'range_compressed.npy';meta=out/'range_compressed.json'
    expected=dict(source_bytes=cap.path.stat().st_size,source_mtime_ns=cap.path.stat().st_mtime_ns,
                  frames=cap.n,samples=cap.nr,chirp_sign=-1,sample_rate_hz=FS,
                  pulse_width_s=cap.tp,bandwidth_hz=cap.bandwidth,algorithm_version=1)
    if path.exists() and meta.exists() and json.loads(meta.read_text())==expected:
        return np.load(path,mmap_mode='r')
    temp=out/'range_compressed.partial.npy'
    dest=np.lib.format.open_memmap(temp,mode='w+',shape=(cap.n,cap.nr),dtype=np.complex64)
    t0=time.perf_counter()
    for a in range(0,cap.n,256):
        b=min(a+256,cap.n)
        dest[a:b]=range_compress(cap.read(np.arange(a,b)),tp=cap.tp,bandwidth=cap.bandwidth)
        if a%2048==0:print(f'Range compression {b}/{cap.n}, {time.perf_counter()-t0:.1f} s',flush=True)
    dest.flush();del dest
    temp.replace(path);write_json(meta,expected)
    return np.load(path,mmap_mode='r')


def export_figures(result,out,name,dyn=45):
    amp=np.abs(result.image);db=20*np.log10(amp+1e-12);db-=db.max()
    # Save unmodified logarithmic magnitude; no generative processing or sharpening.
    display=(255*np.clip((db+dyn)/dyn,0,1)).astype(np.uint8)
    Image.fromarray(display.T[::-1]).save(out/f'{name}_polar.png')
    fig,ax=plt.subplots(figsize=(15,9),constrained_layout=True)
    # The figure preview subsamples only for display; full data are saved separately.
    a=max(1,len(result.r)//2500);b=max(1,len(result.u)//1800)
    im=ax.imshow(db[::a,::b].T,origin='lower',aspect='auto',cmap='gray',vmin=-dyn,vmax=0,
                 extent=[result.r[0],result.r[-1],result.u[0],result.u[-1]])
    ax.set(xlabel='Slant range from aperture centre (m)',ylabel='sin(horizontal look angle)',
           title='Real-data fast factorized backprojection | 2018-08-14 18:54:08')
    fig.colorbar(im,ax=ax,label='Relative magnitude (dB)');fig.savefig(out/f'{name}_overview.png',dpi=180);plt.close(fig)
    del amp,db,display


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,default=ROOT/'2018-08-14_18-54-08.dat')
    p.add_argument('--output',type=Path,default=ROOT/'fast_bp/results')
    p.add_argument('--rmin',type=float,default=500.)
    p.add_argument('--rmax',type=float,default=2800.)
    p.add_argument('--umin',type=float,default=-.65)
    p.add_argument('--umax',type=float,default=.55)
    p.add_argument('--side',type=int,choices=[-1,1],default=1)
    p.add_argument('--phase-sign',type=int,choices=[-1,1],default=-1)
    p.add_argument('--height',type=float,default=18.)
    p.add_argument('--trajectory',choices=['velocity','position','straight'],default='velocity')
    p.add_argument('--leaf',type=int,default=64)
    p.add_argument('--dr',type=float,default=.25)
    p.add_argument('--angular-oversample',type=float,default=3.)
    p.add_argument('--range-shift',type=float,default=None,help='Extra metres in range origin; default -c*Tp/4 from existing Rref convention')
    p.add_argument('--position-file',type=Path,default=None,help='Effective phase-centre positions saved by autofocus.py')
    p.add_argument('--name',default='ffbp_final')
    args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    cap=Capture(args.input);pos,geom=trajectory(cap,args.height,args.trajectory)
    rc=cached_compression(cap,out)
    if args.position_file is not None:
        refined=np.load(args.position_file)
        if refined.shape!=pos.shape or not np.isfinite(refined).all():
            raise ValueError('Invalid autofocus position array')
        geom['effective_autofocus_correction_rms_m']=np.sqrt(np.mean((refined-pos)**2,axis=0)).tolist()
        geom['autofocus_position_file']=str(args.position_file.resolve())
        pos=refined
    if args.range_shift is None:args.range_shift=-C*cap.tp/4
    r_start=C*cap.delay/2+args.range_shift
    weight=hann(cap.n).astype(np.float64);phase=np.zeros(cap.n)
    signed_fc=args.phase_sign*FC;k=4*np.pi*signed_fc/C
    print('Forming FFBP from all valid pulses',cap.n,flush=True)
    result,counts=form_ffbp(rc,pos,r_start,C/(2*FS),signed_fc,
        (args.rmin,args.rmax,args.umin,args.umax),side=args.side,leaf_size=args.leaf,
        range_spacing=args.dr,angular_oversample=args.angular_oversample,weight=weight)
    print('FFBP complete',counts,flush=True)
    # Direct BP at random interior grid pixels and brightest grid pixels.
    rng=np.random.default_rng(20180814)
    ir=rng.integers(8,len(result.r)-8,3000);iu=rng.integers(8,len(result.u)-8,3000)
    # Find strong pixels without constructing another full-sized sorting array.
    rstep=max(1,len(result.r)//1800);ustep=max(1,len(result.u)//1500)
    thumb=abs(result.image[::rstep,::ustep])
    pp=np.argpartition(thumb.ravel(),-1000)[-1000:]
    ri,ui=np.unravel_index(pp,thumb.shape);ir=np.r_[ir,ri*rstep];iu=np.r_[iu,ui*ustep]
    rr=result.r[ir];uu=result.u[iu];gr=np.sqrt(rr**2-result.origin[2]**2)
    xyz=np.column_stack([result.origin[0]+args.side*gr*np.sqrt(1-uu*uu),result.origin[1]+gr*uu,np.zeros(len(rr))])
    t0=time.perf_counter();direct=backproject_points(rc,pos,xyz,r_start,C/(2*FS),k,weight,phase)
    direct_time=time.perf_counter()-t0
    fast=result.image[ir,iu]*np.exp(1j*k*rr)
    error=float(np.linalg.norm(fast-direct)/np.linalg.norm(direct))
    coherence=float(abs(np.vdot(fast,direct))/(np.linalg.norm(fast)*np.linalg.norm(direct)))
    validation=dict(points=len(rr),relative_complex_error=error,complex_coherence=coherence,
                    direct_bp_sample_seconds=direct_time,comparison='identical data, geometry, aperture weights and pixels',
                    passed=error<.15 and coherence>.99)
    print('Real-data direct-BP validation',validation,flush=True)
    np.savez(out/f'{args.name}_validation_samples.npz',direct=direct,ffbp=fast,xyz=xyz)
    write_json(out/f'{args.name}_metrics.json',dict(capture=cap.summary,geometry=geom,
        options={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        ffbp=counts,validation=validation))
    if not validation['passed']:
        raise RuntimeError('FFBP differs too much from direct BP; increase oversampling before delivery')
    np.save(out/f'{args.name}_complex_demodulated.npy',result.image)
    np.savez(out/f'{args.name}_axes.npz',r=result.r,u=result.u,origin=result.origin,k=k,side=args.side)
    export_figures(result,out,args.name)
    print('Saved complex polar image and figures',flush=True)
    # Cartesian ground plane, 1 m pixel pitch. Covers the useful illuminated
    # sector; a validity mask keeps pixels outside the processed sector blank.
    x=np.arange(450.,2800.01,1.);y=np.arange(-1300.,1300.01,1.)
    cart=np.zeros((len(y),len(x)),np.complex64)
    for a in range(0,len(y),128):
        yy,xx=np.meshgrid(y[a:a+128],args.side*x,indexing='ij')
        xyz=np.column_stack([xx.ravel(),yy.ravel(),np.zeros(xx.size)])
        cart[a:a+128]=evaluate_image(result.image,result.origin,result.r0,result.dr,result.u0,result.du,xyz,k).reshape(xx.shape)
    savemat(out/f'{args.name}_ground.mat',dict(image_complex=cart,cross_track_m=args.side*x,
        along_track_m=y,height_reference_m=args.height,carrier_hz=FC,phase_sign=args.phase_sign,
        valid_pulse_count=cap.n,phase_centre=result.origin),do_compression=True)
    db=20*np.log10(abs(cart)+1e-12);db-=db.max()
    Image.fromarray((255*np.clip((db+45)/45,0,1)).astype(np.uint8)[::-1]).save(out/f'{args.name}_ground.png')
    fig,ax=plt.subplots(figsize=(13,11),constrained_layout=True)
    im=ax.imshow(db,origin='lower',cmap='gray',vmin=-45,vmax=0,extent=[x[0],x[-1],y[0],y[-1]],aspect='equal')
    suffix=' | phase-gradient autofocus' if args.position_file else ''
    ax.set(xlabel='Cross-track distance, left of travel (m)' if args.side==1 else 'Cross-track distance, right of travel (m)',
           ylabel='Along-track coordinate (m)',title=f'Real-data FFBP | {cap.n:,} valid pulses | H = {args.height:g} m'+suffix)
    fig.colorbar(im,ax=ax,label='Relative magnitude (dB)',shrink=.8)
    fig.savefig(out/f'{args.name}_ground_overview.png',dpi=180);plt.close(fig)
    print('Saved MATLAB complex ground image and metre axes',flush=True)


if __name__=='__main__':main()
