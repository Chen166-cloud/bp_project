"""CPU parallel, phase-preserving polar fast factorized backprojection.

Independent implementation of the subaperture/recursive polar merge method.
References and differences from open-source implementations: SOURCES.md.
Coordinates: x=cross-track, y=along-track, z=height above a flat scene.
Polar radius is slant range to each node's phase centre, u=sin(horizontal angle).
Stored node images are demodulated by exp(-j*k*radius) to avoid carrier aliasing.
"""
import math
import time
from dataclasses import dataclass
import numpy as np
from numba import njit,prange,set_num_threads

set_num_threads(16)

@njit(inline='always')
def cubic_weights(t):
    # Four point Lagrange interpolator at indices floor(x)-1 ... floor(x)+2.
    return (-t*(t-1)*(t-2)/6,(t+1)*(t-1)*(t-2)/2,
            -(t+1)*t*(t-2)/2,(t+1)*t*(t-1)/6)


@njit(inline='always')
def sample1(row, q):
    i=int(math.floor(q));t=q-i
    if i<1 or i+2>=len(row):return 0j
    w=cubic_weights(t)
    return w[0]*row[i-1]+w[1]*row[i]+w[2]*row[i+1]+w[3]*row[i+2]


@njit(inline='always')
def sample2(img, r, u):
    ir=int(math.floor(r));iu=int(math.floor(u))
    if ir<1 or iu<1 or ir+2>=img.shape[0] or iu+2>=img.shape[1]:return 0j
    wr=cubic_weights(r-ir);wu=cubic_weights(u-iu)
    z=0j
    for a in range(4):
        for b in range(4):z+=img[ir+a-1,iu+b-1]*wr[a]*wu[b]
    return z


@njit(parallel=True,cache=True)
def backproject_points(data,pos,xyz,r_start,dr,k,weight,phase):
    out=np.zeros(len(xyz),np.complex64)
    for p in prange(len(xyz)):
        total=0j
        for n in range(len(pos)):
            dx=xyz[p,0]-pos[n,0];dy=xyz[p,1]-pos[n,1];dz=xyz[p,2]-pos[n,2]
            r=math.sqrt(dx*dx+dy*dy+dz*dz)
            v=sample1(data[n],(r-r_start)/dr)
            a=k*r+phase[n]
            total+=weight[n]*v*complex(math.cos(a),math.sin(a))
        out[p]=total
    return out


@njit(parallel=True,cache=True)
def bp_polar(data,pos,origin,r0,dr,nr,u0,du,nu,r_start,data_dr,k,weight,phase,side):
    out=np.empty((nr,nu),np.complex64)
    for ir in prange(nr):
        r=r0+ir*dr
        ground=math.sqrt(max(0.,r*r-origin[2]*origin[2]))
        for iu in range(nu):
            u=u0+iu*du
            x=origin[0]+side*ground*math.sqrt(max(0.,1-u*u))
            y=origin[1]+ground*u
            total=0j
            for n in range(len(pos)):
                dx=x-pos[n,0];dy=y-pos[n,1]
                d=math.sqrt(dx*dx+dy*dy+pos[n,2]*pos[n,2])
                v=sample1(data[n],(d-r_start)/data_dr)
                a=k*(d-r)+phase[n]
                total+=weight[n]*v*complex(math.cos(a),math.sin(a))
            out[ir,iu]=total
    return out


@njit(parallel=True,cache=True)
def regrid(img,src_origin,sr0,sdr,su0,sdu,dst_origin,r0,dr,nr,u0,du,nu,k,side):
    out=np.empty((nr,nu),np.complex64)
    for ir in prange(nr):
        r=r0+ir*dr
        gr=math.sqrt(max(0.,r*r-dst_origin[2]*dst_origin[2]))
        for iu in range(nu):
            u=u0+iu*du
            x=dst_origin[0]+side*gr*math.sqrt(max(0.,1-u*u))
            y=dst_origin[1]+gr*u
            dx=x-src_origin[0];dy=y-src_origin[1]
            g=math.sqrt(dx*dx+dy*dy)
            rr=math.sqrt(g*g+src_origin[2]*src_origin[2]);uu=dy/g
            v=sample2(img,(rr-sr0)/sdr,(uu-su0)/sdu)
            a=k*(rr-r)
            out[ir,iu]=v*complex(math.cos(a),math.sin(a))
    return out


@dataclass
class PolarImage:
    image: np.ndarray
    origin: np.ndarray
    r0: float
    dr: float
    u0: float
    du: float
    side: int

    @property
    def r(self):return self.r0+self.dr*np.arange(self.image.shape[0])
    @property
    def u(self):return self.u0+self.du*np.arange(self.image.shape[1])


def form_ffbp(data,pos,r_start,data_dr,fc,extent,side=1,leaf_size=128,
              range_spacing=.25,angular_oversample=2.5,weight=None,phase=None,
              verbose=True):
    """Form a fixed world-sector using exact geometry at leaves and merges.

extent=(slant_min,slant_max,u_min,u_max) about the full aperture mean origin.
Each node covers that same sector plus a guard band. Angular sample density
grows with node aperture length. Complex images are never magnitude-detected
until AFTER the root is produced. No pulse decimation is performed here.
"""
    start=time.perf_counter();k=4*np.pi*fc/299792458.
    wavelength=299792458./abs(fc)
    pos=np.ascontiguousarray(pos,dtype=np.float64)
    if weight is None:weight=np.ones(len(pos))
    if phase is None:phase=np.zeros(len(pos))
    weight=np.ascontiguousarray(weight,dtype=np.float64)
    phase=np.ascontiguousarray(phase,dtype=np.float64)
    root_origin=pos.mean(axis=0)
    rmin,rmax,umin,umax=extent
    # Densely sampled boundary: tight conservative bounds under translations.
    us=np.linspace(umin,umax,129);rs=np.linspace(rmin,rmax,129)
    rb=np.r_[np.full_like(us,rmin),np.full_like(us,rmax),rs,rs]
    ub=np.r_[us,us,np.full_like(rs,umin),np.full_like(rs,umax)]
    g=np.sqrt(rb**2-root_origin[2]**2)
    bx=root_origin[0]+side*g*np.sqrt(1-ub**2);by=root_origin[1]+g*ub
    counts={'leaves':0,'merges':0,'leaf_pixel_pulse_products':0,'merged_pixels':0}

    def node(a,b):
        origin=pos[a:b].mean(axis=0)
        dx=bx-origin[0];dy=by-origin[1];gr=np.sqrt(dx*dx+dy*dy)
        rr=np.sqrt(gr*gr+origin[2]**2);uu=dy/gr
        aperture=np.linalg.norm(np.ptp(pos[a:b,:2],axis=0))
        du=min(.025,wavelength/(2*max(aperture,.1)*angular_oversample))
        # A parent's interpolation needs four child samples plus margin.
        u0=max(-.995,float(uu.min())-8*du)
        u1=min(.995,float(uu.max())+8*du)
        r0=float(rr.min())-8*range_spacing
        nr=int(np.ceil((rr.max()-r0)/range_spacing))+9
        nu=int(np.ceil((u1-u0)/du))+1
        if b-a<=leaf_size:
            im=bp_polar(data[a:b],pos[a:b],origin,r0,range_spacing,nr,u0,du,nu,
                        r_start,data_dr,k,weight[a:b],phase[a:b],side)
            counts['leaves']+=1;counts['leaf_pixel_pulse_products']+=nr*nu*(b-a)
            if verbose and counts['leaves']%8==0:
                print(f"FFBP: {counts['leaves']} leaves, {counts['merges']} merges, {time.perf_counter()-start:.1f} s",flush=True)
        else:
            mid=(a+b)//2
            left=node(a,mid)
            im=regrid(left.image,left.origin,left.r0,left.dr,left.u0,left.du,
                        origin,r0,range_spacing,nr,u0,du,nu,k,side)
            del left
            right=node(mid,b)
            im+=regrid(right.image,right.origin,right.r0,right.dr,right.u0,right.du,
                        origin,r0,range_spacing,nr,u0,du,nu,k,side)
            del right
            counts['merges']+=1;counts['merged_pixels']+=2*nr*nu
        return PolarImage(im,origin,r0,range_spacing,u0,du,side)

    result=node(0,len(pos))
    # Crop guard cells only, retaining exact grid coordinates and phases.
    ri=np.flatnonzero((result.r>=rmin)&(result.r<=rmax))
    ui=np.flatnonzero((result.u>=umin)&(result.u<=umax))
    result.r0+=ri[0]*result.dr;result.u0+=ui[0]*result.du
    result.image=result.image[ri[0]:ri[-1]+1,ui[0]:ui[-1]+1].copy()
    counts['elapsed_s']=time.perf_counter()-start
    counts['shape']=list(result.image.shape)
    return result,counts


def polar_points(origin,r,u,side=1):
    rr,uu=np.meshgrid(r,u,indexing='ij');gr=np.sqrt(rr**2-origin[2]**2)
    return np.column_stack([(origin[0]+side*gr*np.sqrt(1-uu**2)).ravel(),
                            (origin[1]+gr*uu).ravel(),np.zeros(rr.size)])


@njit(parallel=True,cache=True)
def evaluate_image(img,origin,r0,dr,u0,du,xyz,k):
    out=np.zeros(len(xyz),np.complex64)
    for p in prange(len(xyz)):
        dx=xyz[p,0]-origin[0];dy=xyz[p,1]-origin[1]
        g=math.sqrt(dx*dx+dy*dy);r=math.sqrt(g*g+origin[2]*origin[2])
        u=dy/g
        v=sample2(img,(r-r0)/dr,(u-u0)/du)
        out[p]=v*complex(math.cos(k*r),math.sin(k*r))
    return out


@njit(parallel=True,cache=True)
def target_contributions(data,pos,xyz,r_start,dr,k):
    out=np.empty((len(xyz),len(pos)),np.complex64)
    for p in prange(len(xyz)):
        for n in range(len(pos)):
            dx=xyz[p,0]-pos[n,0];dy=xyz[p,1]-pos[n,1];dz=xyz[p,2]-pos[n,2]
            r=math.sqrt(dx*dx+dy*dy+dz*dz)
            v=sample1(data[n],(r-r_start)/dr)
            a=k*r
            out[p,n]=v*complex(math.cos(a),math.sin(a))
    return out
