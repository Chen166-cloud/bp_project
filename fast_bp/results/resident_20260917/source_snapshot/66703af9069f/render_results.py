"""Display-only dB windowing; no change to the reconstructed complex image."""
from pathlib import Path
import json
import numpy as np
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

out=Path(__file__).resolve().parent/'results'
s=loadmat(out/'ffbp_final_ground.mat');im=s['image_complex']
x=s['cross_track_m'].ravel();y=s['along_track_m'].ravel()
assert np.iscomplexobj(im) and np.isfinite(im).all() and np.max(abs(im))>0
db=20*np.log10(abs(im)+1e-12);db-=db.max()
lo,hi=-65.,-10.
fig,ax=plt.subplots(figsize=(14,9),constrained_layout=True)
v=ax.imshow(db,origin='lower',extent=[x[0],x[-1],y[0],y[-1]],cmap='gray',vmin=lo,vmax=hi,aspect='equal')
ax.set(xlim=(450,2650),ylim=(-800,800),xlabel='Cross-track distance, left of travel (m)',
       ylabel='Along-track coordinate (m)',title='2018-08-14 18:54:08 | FFBP with phase-gradient autofocus')
fig.colorbar(v,ax=ax,label='Magnitude relative to image peak (dB)',shrink=.86)
fig.savefig(out/'ffbp_final_preview.png',dpi=180);plt.close(fig)
yi=(y>=-800)&(y<=800);xi=(x>=450)&(x<=2650)
arr=db[np.ix_(yi,xi)]
Image.fromarray((255*np.clip((arr-lo)/(hi-lo),0,1)).astype(np.uint8)[::-1]).save(out/'ffbp_final_display.png')
fig,axs=plt.subplots(1,2,figsize=(14,6),constrained_layout=True)
for ax,(xa,xb,ya,yb) in zip(axs,[(1000,1550,-300,150),(1550,2250,0,550)]):
    ax.imshow(db,origin='lower',extent=[x[0],x[-1],y[0],y[-1]],cmap='gray',vmin=lo,vmax=hi,aspect='equal')
    ax.set(xlim=(xa,xb),ylim=(ya,yb),xlabel='Cross-track (m)',ylabel='Along-track (m)')
axs[0].set_title('Detail A: linear scattering structures');axs[1].set_title('Detail B: distributed scattering')
fig.savefig(out/'ffbp_final_details.png',dpi=180);plt.close(fig)
metrics=json.loads((out/'ffbp_final_metrics.json').read_text())
metrics['display']=dict(preview_db_limits=[lo,hi],preview_cross_track_m=[450,2650],preview_along_track_m=[-800,800],
                       operation='fixed logarithmic display window only; complex image unchanged',ground_shape=list(im.shape),
                       ground_finite=True,ground_nonzero_pixels=int(np.count_nonzero(im)))
(out/'ffbp_final_metrics.json').write_text(json.dumps(metrics,indent=2,ensure_ascii=False),encoding='utf-8')
print('Display and complex MAT verification complete.',flush=True)
