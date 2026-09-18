"""Decision experiments for GPU-resident SAR BP. No existing output is overwritten.

Run suites separately so every stage is recoverable. Input is uploaded once per
process; initialization required per image is inside each measured execution.
"""
from __future__ import annotations
import argparse
import copy
from dataclasses import replace
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from scipy.signal.windows import hann
from gpu_backend import CudaBaseline, build_ffbp_plan, cp, cuda_environment
from gpu_baseline import fidelity
from resident_backend import ResidentProgram

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / 'fast_bp/results/resident_20260917'
EXTENTS = {'small': (1380.,1470.,-.03,.03),
           'medium': (1300.,1480.,-.08,.08),
           'large': (1200.,1560.,-.16,.16)}
RSTART = 374.74057250000004
DATA_DR = .3747405725
FC = 1.4e9
K = -4*np.pi*FC/299792458.

def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')

def clean_json(value):
    if isinstance(value, dict): return {str(k):clean_json(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [clean_json(v) for v in value]
    if isinstance(value, np.generic): return clean_json(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    return value

def stats(a):
    a=np.asarray(a,dtype=float)
    return dict(samples_ms=a.tolist(),median_ms=float(np.median(a)),
                min_ms=float(a.min()),max_ms=float(a.max()),
                q1_ms=float(np.percentile(a,25)),q3_ms=float(np.percentile(a,75)),
                coefficient_of_variation=float(a.std()/a.mean()))

def make_plan(position, extent, leaf=64, internal_os=3., base=None):
    plan=build_ffbp_plan(position,-FC,extent,side=1,leaf_size=leaf,
                         range_spacing=.25,angular_oversample=internal_os)
    if base is not None:
        # All output pixel coordinates remain identical; only descendant
        # sampling is refined. Keep each candidate's actual child references.
        root=replace(base.root,left=plan.root.left,right=plan.root.right)
        counts=dict(plan.counts)
        old_nrnu=plan.root.nr*plan.root.nu
        new_nrnu=root.nr*root.nu
        if root.is_leaf:
            counts['leaf_pixel_pulse_products']+=(new_nrnu-old_nrnu)*(root.b-root.a)
        else:
            counts['merged_pixels']+=2*(new_nrnu-old_nrnu)
        counts['shape']=list(base.shape)
        counts['direct_pixel_pulse_products']=base.nr*base.nu*len(position)
        plan=replace(plan,root=root,crop_r=base.crop_r,crop_u=base.crop_u,
                     origin=base.origin.copy(),r0=base.r0,dr=base.dr,nr=base.nr,
                     u0=base.u0,du=base.du,nu=base.nu,counts=counts)
        for name in ('r0','dr','nr','u0','du','nu','side','crop_r','crop_u'):
            assert getattr(plan,name)==getattr(base,name), name
        assert np.array_equal(plan.origin,base.origin)
    return plan

def for_plan(device,plan):
    return replace(device,origins64=cp.asarray(plan.origins),
                   origins32=cp.asarray(plan.origins.astype(np.float32)))

def timed(call):
    start,end=cp.cuda.Event(),cp.cuda.Event()
    t=time.perf_counter()
    start.record()
    call()
    end.record()
    end.synchronize()
    return float(cp.cuda.get_elapsed_time(start,end)),1000*(time.perf_counter()-t)

def benchmark_group(candidates,reference,out,key,repeats=15,warmups=5,seed=20260917):
    rng=np.random.default_rng(seed)
    for c in candidates:
        c['record']['warmup_ms']=[]
        for _ in range(warmups):
            ms,_=timed(c['call'])
            c['record']['warmup_ms'].append(ms)
    orders=[]
    for rep in range(repeats):
        order=rng.permutation(len(candidates)).tolist()
        orders.append([candidates[i]['record']['name'] for i in order])
        for i in order:
            c=candidates[i]
            ms,wall=timed(c['call'])
            c['record'].setdefault('event_ms',[]).append(ms)
            c['record'].setdefault('wall_ms',[]).append(wall)
        if rep==0 or (rep+1)%5==0:
            print(f'{key}: measured round {rep+1}/{repeats}',flush=True)
    # Transfer and CPU image metrics occur only after timing is complete.
    for c in candidates:
        c['call']();cp.cuda.get_current_stream().synchronize()
        z=cp.asnumpy(c['program'].output)
        c['record']['quality_vs_direct_fp64']=fidelity(reference,z)
        c['record']['timing']=stats(c['record'].pop('event_ms'))
        c['record']['wall_timing']=stats(c['record'].pop('wall_ms'))
        c['record']['program']=clean_json(c['program'].metadata)
        c['record']['launch_count']=c['program'].launch_count
        print(f"{c['record']['name']}: {c['record']['timing']['median_ms']:.3f} ms; "
              f"SER {c['record']['quality_vs_direct_fp64']['ser_db']:.3f} dB",flush=True)
    result=dict(key=key,seed=seed,repeats=repeats,warmups=warmups,
                randomized_orders=orders,records=[c['record'] for c in candidates])
    save_json(out/f'{key}.json',clean_json(result))
    return result

def candidate(program,name,graph=False,**metadata):
    if graph:
        if not program.capture_graph():
            raise RuntimeError(f"graph capture failed: {program.metadata}")
        call=program.replay
    else: call=program.execute
    return dict(program=program,call=call,record=dict(name=name,graph=graph,**metadata))

def dispose(candidates):
    seen=set()
    for c in candidates:
        if id(c['program']) not in seen:
            c['program'].close();seen.add(id(c['program']))
    candidates.clear();gc.collect();cp.get_default_memory_pool().free_all_blocks()

def compile_probes(backend):
    src=(ROOT/'fast_bp/gpu/kernels.cu').read_text(encoding='utf-8')+'\n'+(ROOT/'fast_bp/gpu/precision_probe.cu').read_text(encoding='utf-8')
    module=cp.RawModule(code=src,options=('-std=c++11',))
    proxies={}
    for suffix in ('phase32','phase32_wrapped'):
        proxy=copy.copy(backend);proxy.kernels=dict(backend.kernels)
        proxy.kernels['direct_chunk_mixed']=module.get_function('direct_chunk_mixed_'+suffix)
        proxies[suffix]=proxy
    return module,proxies

def validation(backend,out):
    from ffbp import bp_polar
    rng=np.random.default_rng(20260917)
    records=[]
    for sign,side in [(-1,1),(-1,-1),(1,1),(1,-1)]:
        n,ns=31,512
        t=np.arange(n)
        pos=np.column_stack((.05*np.sin(.37*t),np.linspace(-15,15,n),18+.1*np.cos(.23*t)))
        data=(rng.normal(size=(n,ns))+1j*rng.normal(size=(n,ns))).astype(np.complex64)
        w=hann(n);ph=np.linspace(-.2,.3,n);k=sign*abs(K)
        plan=build_ffbp_plan(pos,sign*FC,(1080.,1120.,-.05,.05),side=side,
                             leaf_size=8,range_spacing=.5,angular_oversample=3.)
        dev=backend.upload(data,pos,w,ph,plan)
        cpu=bp_polar(data,pos,plan.origin,plan.r0,plan.dr,plan.nr,plan.u0,plan.du,plan.nu,900.,.5,k,w,ph,side)
        for mode in ('fp64','fp32','mixed'):
            legacy,_,_=backend.run_ffbp(plan,dev,mode,900.,.5,k)
            legacy_host=cp.asnumpy(legacy)
            for algo,var in [('direct','original'),('ffbp','original'),('ffbp','fused')]:
                p=ResidentProgram(backend,plan,dev,mode,900.,.5,k,algorithm=algo,variant=var,chunk_pulses=7)
                p.execute();cp.cuda.get_current_stream().synchronize();z=cp.asnumpy(p.output)
                q=fidelity(cpu if algo=='direct' else legacy_host,z)
                tol=1e-6 if algo=='ffbp' else {'fp64':1e-5,'mixed':1e-3,'fp32':.03}[mode]
                assert q['finite_fraction']==1. and q['relative_complex_error']<tol,(sign,side,mode,algo,var,q)
                graph_ok=p.capture_graph()
                if graph_ok:
                    p.replay();cp.cuda.get_current_stream().synchronize()
                    zg=cp.asnumpy(p.output)
                    assert np.array_equal(z,zg),(sign,side,mode,algo,var,'graph differs')
                records.append(dict(sign=sign,side=side,mode=mode,algorithm=algo,variant=var,
                                    quality=q,graph=graph_ok,metadata=clean_json(p.metadata)))
                p.close()
            del legacy,legacy_host
        del dev
        gc.collect();cp.get_default_memory_pool().free_all_blocks()
        print(f'Validation passed: phase sign {sign}, side {side}',flush=True)
    result=dict(passed=True,cases=records)
    save_json(out/'validation.json',clean_json(result))
    return result

def direct_reference(backend,device,plan,out,name):
    p=ResidentProgram(backend,plan,device,'fp64',RSTART,DATA_DR,K,algorithm='direct',chunk_pulses=64)
    ms,_=timed(p.execute)
    z=cp.asnumpy(p.output)
    np.save(out/f'{name}_direct_reference.npy',z)
    p.close()
    print(f'{name}: FP64 direct reference ready ({ms:.1f} ms; not a benchmark sample)',flush=True)
    return z

def ablation(backend,device,plan,reference,out,repeats,warmups):
    candidates=[]
    for mode in ('fp64','fp32','mixed'):
        p=ResidentProgram(backend,plan,device,mode,RSTART,DATA_DR,K,algorithm='direct',chunk_pulses=64)
        candidates.append(candidate(p,f'direct_{mode}_c64',algorithm='direct',mode=mode,variant='original',chunk=64))
        for var in ('original','fused'):
            p=ResidentProgram(backend,plan,device,mode,RSTART,DATA_DR,K,algorithm='ffbp',variant=var)
            candidates.append(candidate(p,f'ffbp_{mode}_{var}',algorithm='ffbp',mode=mode,variant=var,leaf=64,internal_os=3.))
            candidates.append(candidate(p,f'ffbp_{mode}_{var}_graph',graph=True,algorithm='ffbp',mode=mode,variant=var,leaf=64,internal_os=3.))
    for chunk in (64,256,1024):
        p=ResidentProgram(backend,plan,device,'fp32',RSTART,DATA_DR,K,algorithm='direct',chunk_pulses=chunk)
        candidates.append(candidate(p,f'direct_fp32_c{chunk}_graph',graph=True,algorithm='direct',mode='fp32',variant='original',chunk=chunk))
    module,proxies=compile_probes(backend)
    for label,proxy in proxies.items():
        p=ResidentProgram(proxy,plan,device,'mixed',RSTART,DATA_DR,K,algorithm='direct',chunk_pulses=256)
        candidates.append(candidate(p,f'direct_mixed_{label}_c256',graph=True,algorithm='direct',mode='mixed_'+label,variant='precision_probe',chunk=256))
    result=benchmark_group(candidates,reference,out,'ablation_small',repeats,warmups)
    dispose(candidates)
    return result

def sweep(backend,device,position,base,reference,out,name,repeats,warmups):
    # Process only one leaf configuration at a time to keep GPU memory bounded.
    allrecords=[];groups=[]
    direct=ResidentProgram(backend,base,device,'fp32',RSTART,DATA_DR,K,algorithm='direct',chunk_pulses=256)
    for leaf in (64,256,1024):
        for internal_os in (3.,6.):
            plan=make_plan(position,base.extent,leaf,internal_os,base=base)
            pd=for_plan(device,plan)
            candidates=[]
            candidates.append(candidate(direct,f'direct_control_{name}_l{leaf}_os{internal_os:g}',graph=True,
                              algorithm='direct',mode='fp32',variant='original',chunk=256,role='repeated_control'))
            for mode in ('fp32','mixed'):
                p=ResidentProgram(backend,plan,pd,mode,RSTART,DATA_DR,K,algorithm='ffbp',variant='fused')
                candidates.append(candidate(p,f'ffbp_{name}_{mode}_l{leaf}_os{internal_os:g}',graph=True,
                                            algorithm='ffbp',mode=mode,variant='fused',leaf=leaf,
                                            internal_os=internal_os,output_os=3.,shape=list(base.shape),counts=plan.counts))
            key=f'sweep_{name}_l{leaf}_os{internal_os:g}'
            result=benchmark_group(candidates,reference,out,key,repeats,warmups,seed=20260917+leaf)
            groups.append(key);allrecords.extend(result['records'])
            # direct is a repeated, identical GPU-resident control across groups.
            for c in candidates[1:]: c['program'].close()
            candidates.clear();del p,pd,plan;gc.collect();cp.get_default_memory_pool().free_all_blocks()
    direct.close()
    result=dict(name=name,shape=list(base.shape),extent=list(base.extent),groups=groups,records=allrecords)
    save_json(out/f'sweep_{name}.json',clean_json(result))
    return result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--suite',choices=['validation','ablation','sweep','all'],default='all')
    ap.add_argument('--scales',nargs='+',choices=list(EXTENTS),default=['small','medium','large'])
    ap.add_argument('--repeats',type=int,default=15)
    ap.add_argument('--warmups',type=int,default=5)
    ap.add_argument('--output',type=Path,default=DEFAULT_OUT)
    args=ap.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    telemetry=(out/f'telemetry_{args.suite}_{int(time.time())}.csv').open('w',encoding='utf-8')
    monitor=None
    try:
        monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,pstate,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,utilization.gpu,memory.used',
                                  '--format=csv','--loop-ms=1000'],stdout=telemetry,stderr=subprocess.STDOUT,
                                 creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        backend=CudaBaseline()
        manifest=dict(date=time.strftime('%Y-%m-%d %H:%M:%S %z'),suite=args.suite,status='running',
                      environment=cuda_environment(),arguments=vars(args)|{'output':str(out)},
                      timing_scope='CUDA events around required initialization and complete resident execution; output remains on GPU; allocation/compile/plan/upload/download excluded uniformly',
                      profiler='Nsight not found; event experiments do not prove hardware-counter bottlenecks',
                      source_sha256={})
        for path in [Path(__file__),ROOT/'fast_bp/resident_backend.py',ROOT/'fast_bp/gpu/resident_kernels.cu',ROOT/'fast_bp/gpu/kernels.cu',ROOT/'fast_bp/gpu/precision_probe.cu']:
            if path.exists(): manifest['source_sha256'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        save_json(out/f'manifest_{args.suite}.json',clean_json(manifest))
        if args.suite in ('validation','all'): validation(backend,out)
        if args.suite!='validation':
            data=np.load(ROOT/'fast_bp/results/range_compressed.npy',mmap_mode='r')
            pos=np.load(ROOT/'fast_bp/results/autofocus_position.npy')
            base=make_plan(pos,EXTENTS['small'])
            device=backend.upload(data,pos,hann(len(pos)),np.zeros(len(pos)),base)
            print(f'Input resident: {device.bytes_uploaded/2**30:.3f} GiB',flush=True)
            if args.suite in ('ablation','all'):
                ref=np.load(ROOT/'fast_bp/results/gpu_baseline/cpu_direct_reference.npy')
                check=direct_reference(backend,device,base,out,'small')
                q=fidelity(ref,check)
                assert q['relative_complex_error']<1e-5,q
                ablation(backend,device,base,ref,out,args.repeats,args.warmups)
            if args.suite in ('sweep','all'):
                for name in args.scales:
                    base=make_plan(pos,EXTENTS[name]);pd=for_plan(device,base)
                    ref=direct_reference(backend,pd,base,out,name)
                    sweep(backend,pd,pos,base,ref,out,name,args.repeats,args.warmups)
                    del pd,ref;gc.collect();cp.get_default_memory_pool().free_all_blocks()
        manifest['status']='completed';manifest['completed_at']=time.strftime('%Y-%m-%d %H:%M:%S %z')
        save_json(out/f'manifest_{args.suite}.json',clean_json(manifest))
    finally:
        if monitor is not None:
            monitor.terminate();monitor.wait(timeout=10)
        telemetry.close()

if __name__=='__main__':main()
