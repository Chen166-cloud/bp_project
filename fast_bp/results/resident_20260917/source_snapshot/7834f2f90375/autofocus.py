"""Data-driven 2D trajectory refinement from generalized phase gradients.

Only complex measured samples are used. The fitted correction is an effective
phase-centre correction, not a claim of independently measured platform motion.
Height is fixed because this low-grazing-angle collection poorly observes z.
"""
from pathlib import Path
import time
import numpy as np
from scipy import signal
from scipy.signal.windows import hann
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sar_io import Capture,trajectory,C,FS,FC,write_json
from ffbp import form_ffbp,target_contributions


def pick_targets(result,nr_tiles=12,nu_tiles=12):
    pts=[];powers=[]
    rs=np.array_split(np.arange(len(result.r)),nr_tiles)
    us=np.array_split(np.arange(len(result.u)),nu_tiles)
    for ri in rs:
        for ui in us:
            a=abs(result.image[ri[0]:ri[-1]+1,ui[0]:ui[-1]+1])
            ij=np.unravel_index(a.argmax(),a.shape)
            r=result.r[ri[ij[0]]];u=result.u[ui[ij[1]]]
            g=np.sqrt(r*r-result.origin[2]**2)
            pts.append([result.origin[0]+result.side*g*np.sqrt(1-u*u),result.origin[1]+g*u,0.])
            powers.append(float(a.max())**2)
    powers=np.array(powers);pts=np.array(pts)
    # Drop noise-only cells and cap domination by a single scatterer.
    good=powers>powers.max()*10**(-35/10)
    return pts[good],powers[good]


def estimate_update(data,pos,points,r_start,dr,k,prf,cutoff):
    xi=target_contributions(data,pos,points,r_start,dr,k)
    sos=signal.butter(4,cutoff,fs=prf,output='sos')
    smooth=signal.sosfiltfilt(sos,xi,axis=1)
    cross=smooth[:,1:]*np.conj(smooth[:,:-1])
    gradient=np.angle(cross)
    amp=abs(cross)
    # Relative per-target reliability, capped to retain angular diversity.
    target_power=np.mean(amp,axis=1)
    strength=np.sqrt(target_power/np.median(target_power))
    strength=np.minimum(strength,3.)
    coherence=abs(np.sum(cross,axis=1))/np.sum(amp,axis=1)
    w=(amp/(target_power[:,None]+1e-12))*strength[:,None]
    w*=np.clip(coherence[:,None],0,1)**2
    # Avoid noisy estimate at very dim pulse/target combinations.
    w=np.minimum(w,4.)
    vec=points[:,None,:]-pos[None,1:,:]
    dist=np.linalg.norm(vec,axis=2)
    ux=vec[:,:,0]/dist;uy=vec[:,:,1]/dist
    # phi_residual = k (R_nominal-R_true). A small trajectory update satisfies
    # unit_LOS dot d(position) = phi_residual/k. Fit the derivatives first.
    rhs=gradient/k
    dx=np.zeros(len(pos)-1);dy=dx.copy()
    for _ in range(3):
        a=np.sum(w*ux*ux,axis=0);b=np.sum(w*ux*uy,axis=0);d=np.sum(w*uy*uy,axis=0)
        ex=np.sum(w*ux*rhs,axis=0);ey=np.sum(w*uy*rhs,axis=0)
        reg=.002*(a+d)
        det=(a+reg)*(d+reg)-b*b
        dx=(ex*(d+reg)-ey*b)/np.maximum(det,1e-12)
        dy=(ey*(a+reg)-ex*b)/np.maximum(det,1e-12)
        residual=rhs-ux*dx-uy*dy
        scale=1.4826*np.median(abs(residual),axis=0)+1e-6
        w*=np.minimum(1,2.5*scale[None,:]/(abs(residual)+1e-12))
    delta=np.column_stack([np.r_[0,np.cumsum(dx)],np.r_[0,np.cumsum(dy)],np.zeros(len(pos))])
    # Global translation and linear motion ambiguities are left to the
    # navigation/scene coordinate convention, not inferred from autofocus.
    delta[:,:2]=signal.detrend(delta[:,:2],axis=0,type='linear')
    delta[:,:2]=signal.savgol_filter(delta[:,:2],min(501,len(pos)//2*2-1),3,axis=0)
    # A large update would violate the local linear approximation. Damp it.
    maxnorm=float(np.linalg.norm(delta[:,:2],axis=1).max())
    if maxnorm>1.0:delta*=1./maxnorm
    return delta,dict(targets=len(points),cutoff_hz=cutoff,median_gradient_coherence=float(np.median(coherence)),
                       delta_rms_m=np.sqrt(np.mean(delta**2,axis=0)).tolist(),delta_max_m=np.max(abs(delta),axis=0).tolist())


def main():
    root=Path(__file__).resolve().parent.parent;out=root/'fast_bp/results'
    cap=Capture(root/'2018-08-14_18-54-08.dat')
    data=np.load(out/'range_compressed.npy',mmap_mode='r')
    pos,geo=trajectory(cap);original=pos.copy();k=-4*np.pi*FC/C
    shift=-C*cap.tp/4;r_start=C*cap.delay/2+shift
    records=[];t0=time.perf_counter();best_score=-np.inf;best_pos=pos.copy();best_iteration=0
    fig,axs=plt.subplots(2,3,figsize=(16,10),constrained_layout=True)
    for iteration,cutoff in enumerate([12,8,5,3,2,1]):
        result,counts=form_ffbp(data,pos,r_start,C/(2*FS),-FC,(550,2600,-.6,.5),
                range_spacing=.5,angular_oversample=2.,leaf_size=64,weight=hann(cap.n),verbose=False)
        power=abs(result.image)**2
        contrast=float(np.mean(power**2)/np.mean(power)**2)
        if contrast>best_score:
            best_score=contrast;best_pos=pos.copy();best_iteration=iteration
        db=10*np.log10(power+1e-12);db-=db.max()
        axs.ravel()[iteration].imshow(db[::3,::3].T,origin='lower',aspect='auto',cmap='gray',vmin=-50,vmax=0,
                     extent=[result.r[0],result.r[-1],result.u[0],result.u[-1]])
        axs.ravel()[iteration].set(title=f'Iteration {iteration}, contrast {contrast:.1f}',xlabel='Range (m)',ylabel='sin(angle)')
        points,powers=pick_targets(result)
        delta,stats=estimate_update(data,pos,points,r_start,C/(2*FS),k,cap.prf,cutoff)
        stats.update(iteration=iteration,input_contrast=contrast,elapsed_s=time.perf_counter()-t0)
        records.append(stats);print(stats,flush=True)
        np.savez(out/f'autofocus_iteration_{iteration}.npz',position=pos,points=points,delta=delta)
        pos=pos+delta
    # Select only a position model whose image has actually been evaluated.
    # Later iterations can trade one region's focus for another; retain the
    # best measured contrast rather than blindly accepting the last update.
    pos=best_pos
    np.save(out/'autofocus_position.npy',pos)
    write_json(out/'autofocus_metrics.json',dict(iterations=records,range_shift_m=shift,
                   selected_iteration=best_iteration,selection='maximum measured global intensity contrast on the fixed diagnostic sector',
                   total_correction_rms_m=np.sqrt(np.mean((pos-original)**2,axis=0)).tolist()))
    fig.savefig(out/'autofocus_iterations.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(2,1,figsize=(12,6),constrained_layout=True)
    for a in range(2):axs[a].plot(cap.time,pos[:,a]-original[:,a]);axs[a].set(xlabel='Time (s)',ylabel='Effective correction (m)',title=['Cross-track','Along-track'][a])
    fig.savefig(out/'autofocus_correction.png',dpi=160)


if __name__=='__main__':main()
