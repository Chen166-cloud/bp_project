"""Reproducible audit, isolated raw-LFM point tests, and common-grid real ROIs.

Run from any directory: python sar_quality/evaluate.py
Existing imaging files are never modified. Output is JSON, NPZ and PNG.
"""
from pathlib import Path
import sys, time, json, hashlib
import numpy as np
from scipy.io import loadmat
from scipy.signal.windows import hann
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'fast_bp'))
from sar_io import Capture, trajectory, range_compress, C, FS, FC, write_json
from ffbp import form_ffbp, backproject_points, evaluate_image
from metrics import focus_metrics, fidelity, impulse_metrics

OUT = ROOT/'sar_quality/results'; OUT.mkdir(exist_ok=True, parents=True)
OLD = ROOT/'fast_bp/results'
K = -4*np.pi*FC/C


def audit():
    records = []
    for name in ['SAR_RDA_2.bmp','SAR_CSA.bmp','SAR_BP.bmp','SAR_BP_visualization.bmp']:
        a = np.asarray(Image.open(ROOT/name))
        records.append(dict(file=name, shape=list(a.shape), dtype=str(a.dtype),
            min=int(a.min()), max=int(a.max()), unique_levels=int(np.unique(a).size),
            zero_fraction=float(np.mean(a == 0)), maxcode_fraction=float(np.mean(a == 255)),
            status='invalid_zero_image' if not a.any() else 'display_only_no_metric_ranking'))
    m = loadmat(ROOT/'SAR_BP_raw_data.mat')
    z = m['img_bp']; xx=m['x_grid'].ravel(); yy=m['y_grid'].ravel()
    bp = dict(file='SAR_BP_raw_data.mat', shape=list(z.shape),
        x_bounds_m=[float(xx[0]),float(xx[-1])], y_bounds_m=[float(yy[0]),float(yy[-1])],
        dx_m=float(np.diff(xx).mean()),dy_m=float(np.diff(yy).mean()),
        metrics=focus_metrics(z),provenance='Generator and input pulse interval not recorded; not a common-scene comparison')
    write_json(OUT/'legacy_outputs.json', dict(bitmap_audit=records, legacy_bp=bp))
    plt.figure(figsize=(11,4))
    plt.imshow(20*np.log10(abs(z)/abs(z).max()+1e-12),extent=[xx[0],xx[-1],yy[-1],yy[0]],
               aspect='auto',cmap='gray',vmin=-50,vmax=0)
    plt.xlabel('Stored x coordinate (m); spacing = %.2f m'%np.diff(xx).mean())
    plt.ylabel('Stored y coordinate (m)');plt.title('Legacy complex BP output: coarse grid, provenance incomplete')
    plt.colorbar(label='dB');plt.tight_layout();plt.savefig(OUT/'legacy_bp_audit.png',dpi=150);plt.close()


def direct(data,pos,points,rstart,w):
    return backproject_points(data,pos,np.ascontiguousarray(points),rstart,C/(2*FS),K,w,np.zeros(len(pos)))


def fast_on(data,pos,points,rstart,w,dr=.25):
    origin=pos.mean(0); delta=points[:,:2]-origin[:2]
    rr=np.sqrt((delta*delta).sum(1)+origin[2]**2)
    uu=delta[:,1]/np.linalg.norm(delta,axis=1)
    result,counts=form_ffbp(data,pos,rstart,C/(2*FS),-FC,
        (float(rr.min()-2),float(rr.max()+2),float(uu.min()-.002),float(uu.max()+.002)),
        leaf_size=64,range_spacing=dr,angular_oversample=3.,weight=w,verbose=False)
    values=evaluate_image(result.image,result.origin,result.r0,result.dr,result.u0,result.du,points,K)
    return values,counts


def point_tests(cap,pos):
    w=hann(cap.n); rows=[]; profiles={}
    fig,axs=plt.subplots(3,2,figsize=(12,11),constrained_layout=True)
    for row,range_m in enumerate([800.,1400.,2200.]):
        # Broadside, flat-ground, isolated unit scatterer. Raw pulse generation
        # is independent of the imaging kernels and uses fractional delays.
        target=np.array([np.sqrt(range_m**2-pos[:,2].mean()**2),pos[:,1].mean(),0.])
        R=np.linalg.norm(pos-target,axis=1)
        rstart=float(R.min()-190)
        nr=int(np.ceil((R.max()-rstart+190)/(C/(2*FS))))
        tt=(rstart+C/(2*FS)*np.arange(nr))*2/C
        compressed=np.empty((cap.n,nr),np.complex64)
        for start in range(0,cap.n,256):
            dist=R[start:start+256]
            t=tt[None,:]-2*dist[:,None]/C
            raw=(np.exp(-1j*np.pi*cap.bandwidth/cap.tp*t*t)*
                 (abs(t)<cap.tp/2)*np.exp(-1j*K*dist[:,None])).astype(np.complex64)
            compressed[start:start+len(dist)]=range_compress(raw,cap.tp,cap.bandwidth)
        # Range cut +/-20 m; azimuth cut +/-40 m. 0.025 m measurement spacing.
        x=np.linspace(-20,20,1601);y=np.linspace(-40,40,3201)
        p1=target+np.column_stack([x,np.zeros_like(x),np.zeros_like(x)])
        p2=target+np.column_stack([np.zeros_like(y),y,np.zeros_like(y)])
        points=np.vstack([p1,p2]);t=time.perf_counter()
        d=direct(compressed,pos,points,rstart,w)
        f,cnt=fast_on(compressed,pos,points,rstart,w)
        rec=dict(range_m=range_m,target_xyz_m=target.tolist(),raw_samples_per_pulse=nr,
                 fidelity=fidelity(d,f),metrics={})
        for name,z,style in [('direct_bp',d,'-'),('ffbp',f,'--')]:
            r=impulse_metrics(x,z[:len(x)]);a=impulse_metrics(y,z[len(x):])
            rec['metrics'][name]=dict(ground_range=r,along_track=a)
            for ax,coord,v in zip(axs[row],[x,y],[z[:len(x)],z[len(x):]]):
                ax.plot(coord,20*np.log10(abs(v)/abs(v).max()+1e-12),style,label=name)
        for j,ax in enumerate(axs[row]):
            ax.set(xlabel=('Ground-range' if j==0 else 'Along-track')+' offset (m)',
                   ylabel='Normalized amplitude (dB)',ylim=(-65,2),xlim=((-5,5) if j==0 else (-12,12)),
                   title=f'Raw LFM point, slant range {range_m:.0f} m')
            ax.axhline(-3.0103,color='gray',lw=.6);ax.grid(alpha=.25);ax.legend()
        profiles[f'r{int(range_m)}_range_offset']=x;profiles[f'r{int(range_m)}_az_offset']=y
        profiles[f'r{int(range_m)}_direct']=d;profiles[f'r{int(range_m)}_ffbp']=f
        rows.append(rec);write_json(OUT/'point_targets.json',rows)
        print('POINT',range_m,rec['metrics'],'seconds',round(time.perf_counter()-t,2),flush=True)
    fig.savefig(OUT/'point_target_profiles.png',dpi=170);plt.close(fig)
    np.savez_compressed(OUT/'point_target_profiles.npz',**profiles)


def real_tests(cap,pos):
    data=np.load(OLD/'range_compressed.npy',mmap_mode='r')
    corrected=np.load(OLD/'autofocus_position.npy')
    rstart=C*(cap.delay-cap.tp/2)/2; w=hann(cap.n)
    # Fixed physical ROIs chosen before computing metrics. No peak selection.
    rois=[(900.,-150.),(900.,150.),(1425.,-25.),(1425.,150.),(2100.,-150.),(2100.,150.)]
    rows=[];fig,axs=plt.subplots(len(rois),4,figsize=(13,17),constrained_layout=True)
    labels=['BP / nav','FFBP / nav','BP / autofocus','FFBP / autofocus']
    for ri,(xc,yc) in enumerate(rois):
        x=xc+np.arange(-30,30.001,.25);y=yc+np.arange(-30,30.001,.25)
        xx,yy=np.meshgrid(x,y);points=np.column_stack([xx.ravel(),yy.ravel(),np.zeros(xx.size)])
        arrays={}; rec=dict(roi=ri+1,center_m=[xc,yc],bounds_m=[x[0],x[-1],y[0],y[-1]],
                           shape=list(xx.shape),pixel_spacing_m=.25,metrics={},fidelity={})
        for pn,pp in [('nav',pos),('autofocus',corrected)]:
            t=time.perf_counter();d=direct(data,pp,points,rstart,w).reshape(xx.shape)
            dt=time.perf_counter()-t
            t=time.perf_counter();f,cnt=fast_on(data,pp,points,rstart,w);f=f.reshape(xx.shape)
            ft=time.perf_counter()-t
            arrays['bp_'+pn]=d;arrays['ffbp_'+pn]=f
            rec['metrics']['bp_'+pn]=focus_metrics(d);rec['metrics']['ffbp_'+pn]=focus_metrics(f)
            rec['fidelity'][pn]=dict(**fidelity(d,f),direct_seconds=dt,ffbp_seconds=ft)
        scale=max(abs(z).max() for z in arrays.values())
        for ax,(name,z),title in zip(axs[ri],arrays.items(),labels):
            ax.imshow(20*np.log10(abs(z)/scale+1e-12),origin='lower',extent=rec['bounds_m'],
                      cmap='gray',vmin=-45,vmax=0,aspect='equal')
            m=rec['metrics'][name]
            ax.set(title=f'R{ri+1} {title}\nH={m["entropy_nats"]:.3f}, IC(I)={m["intensity_contrast"]:.2f}',
                   xlabel='Cross-track (m)',ylabel='Along-track (m)')
        np.savez_compressed(OUT/f'real_roi_{ri+1}.npz',x=x,y=y,**arrays)
        rows.append(rec);write_json(OUT/'real_rois.json',rows)
        print('REAL ROI',ri+1,{n: {k:m[k] for k in ['entropy_nats','intensity_contrast']} for n,m in rec['metrics'].items()},flush=True)
    fig.savefig(OUT/'real_roi_comparison.png',dpi=155);plt.close(fig)
    m=np.load(OUT/'real_roi_5.npz');enl={}
    for name in ['bp_nav','ffbp_nav','bp_autofocus','ffbp_autofocus']:
        intensity=abs(m[name]).astype(float)**2
        means=np.array([intensity[i:i+60,j:j+60].mean()
                        for i in range(0,240,60) for j in range(0,240,60)])
        enl[name]=dict(enl_proxy=float(intensity.mean()**2/intensity.var()),
            tile_mean_cv=float(means.std()/means.mean()),
            tile_mean_max_to_min=float(means.max()/means.min()),
            assumption='R5 candidate homogeneous ROI, uncalibrated and not independently validated; spatially correlated samples')
    write_json(OUT/'enl_proxy.json',enl)
    # Apparent widths of three strongest well-separated scatterers in fixed R3.
    # They are not known point reflectors: deliberately do NOT certify PSLR/ISLR.
    m=np.load(OUT/'real_roi_3.npz');a=abs(m['bp_autofocus']).copy();targets=[]
    for _ in range(3):
        iy,ix=np.unravel_index(np.argmax(a),a.shape)
        targets.append([m['x'][ix],m['y'][iy],0.])
        a[max(0,iy-40):iy+41,max(0,ix-40):ix+41]=0
    recs=[]; fig,axs=plt.subplots(3,2,figsize=(12,10),constrained_layout=True)
    for ti,target in enumerate(np.asarray(targets)):
        # Recenter each implementation on its local 2-D peak; width comparisons
        # otherwise confound autofocus-induced peak displacement and defocus.
        cuts={};record=dict(target=ti+1,selected_center_xyz_m=target.tolist(),quality='apparent_scatterer_width_only')
        for pn,pp in [('nav',pos),('autofocus',corrected)]:
            gx,gy=np.meshgrid(np.arange(-2,2.001,.1),np.arange(-4,4.001,.1))
            local=target+np.column_stack([gx.ravel(),gy.ravel(),np.zeros(gx.size)])
            val=direct(data,pp,local,rstart,w);center=local[np.argmax(abs(val))]
            x=np.linspace(-10,10,801);y=np.linspace(-20,20,1601)
            points=np.vstack([center+np.column_stack([x,0*x,0*x]),center+np.column_stack([0*y,y,0*y])])
            v=direct(data,pp,points,rstart,w)
            metrics=[]
            for ax,coord,z in zip(axs[ti],[x,y],[v[:len(x)],v[len(x):]]):
                result=impulse_metrics(coord,z)
                metrics.append({k:result[k] for k in ['valid','irw_m','peak_offset_m'] if k in result})
                ax.plot(coord,20*np.log10(abs(z)/abs(z).max()+1e-12),label=pn)
            record[pn]=dict(center_xyz_m=center.tolist(),ground_range=metrics[0],along_track=metrics[1])
            cuts[pn]=v
        for j,ax in enumerate(axs[ti]):
            ax.set(title=f'Real scatterer {ti+1}: apparent width, clutter present',
                   xlabel=('Cross-track' if j==0 else 'Along-track')+' offset (m)',ylabel='dB',ylim=(-45,2))
            ax.axhline(-3.0103,color='gray',lw=.6);ax.grid(alpha=.25);ax.legend()
        recs.append(record)
        np.savez_compressed(OUT/f'real_scatterer_{ti+1}.npz',range_offset=x,az_offset=y,**cuts)
    write_json(OUT/'real_scatterers.json',recs)
    fig.savefig(OUT/'real_scatterer_profiles.png',dpi=155);plt.close(fig)


def metric_selfcheck():
    x=np.linspace(-40,40,80001);r=impulse_metrics(x,np.sinc(x))
    assert abs(r['irw_m']-.885893)<.001,r
    assert abs(r['pslr_db']+13.26146)<.01,r
    assert abs(r['islr_db']+9.680)<.15,r # finite +/-40 rather than infinite integral
    assert not focus_metrics(np.zeros(100,complex))['valid']
    write_json(OUT/'metric_selfcheck.json',dict(sinc=r,zero_energy_rejected=True,passed=True))


if __name__=='__main__':
    audit();metric_selfcheck()
    cap=Capture(ROOT/'2018-08-14_18-54-08.dat');pos,geom=trajectory(cap)
    write_json(OUT/'protocol.json',dict(capture=cap.summary,geometry=geom,
        point_model='Independent fractional-delay raw rectangular LFM, constant scatterer strength, full measured trajectory; no noise, no propagation loss, no antenna pattern or motion error',
        range_window='Hamming matched reference',azimuth_window='Hann full aperture',
        pulse_count=cap.n,ffbp_leaf_size=64,ffbp_range_grid_m=.25,angular_oversample=3,
        metrics='1D first-minima negative-dB sidelobe convention; intensity Shannon entropy; both amplitude and intensity CV; no dB/8-bit focus metrics',
        nominal_c_over_2B_m=C/(2*cap.bandwidth),raw_range_sample_spacing_m=C/(2*FS)))
    point_tests(cap,pos);real_tests(cap,pos)
    sources=list((ROOT/'sar_quality').glob('*.py'))+list((ROOT/'fast_bp').glob('*.py'))
    write_json(OUT/'code_sha256.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    print('Evaluation finished.',flush=True)
