"""Follow-up falsification experiments; same resident timing contract as round one."""
from __future__ import annotations
import argparse
import gc
import hashlib
from pathlib import Path
import subprocess
import time
import numpy as np
from scipy.signal.windows import hann
from resident_experiment import (ROOT, DEFAULT_OUT, EXTENTS, RSTART, DATA_DR, K, FC,
    CudaBaseline, ResidentProgram, cp, cuda_environment, fidelity, make_plan, for_plan,
    clean_json, save_json, stats, timed, candidate, dispose, compile_probes)
from batched_resident_backend import BatchedResidentProgram
from resident_plan_audit import audit_support, extend_plan_radial_guard


def measure(cs, ref, out, key, repeats, warmups, save_images=False):
    rng=np.random.default_rng(20260918)
    for c in cs:
        c['record']['warmup_ms']=[timed(c['call'])[0] for _ in range(warmups)]
    orders=[]
    for rep in range(repeats):
        order=rng.permutation(len(cs)).tolist(); orders.append(order)
        for i in order:
            c=cs[i]; start=time.time(); ms,wall=timed(c['call'])
            c['record'].setdefault('samples',[]).append(dict(round=rep,start_unix=start,event_ms=ms,wall_ms=wall))
        if rep==0 or (rep+1)%5==0: print(f'{key}: {rep+1}/{repeats}',flush=True)
    images={}
    for c in cs:
        p=c['program'];r=c['record'];c['call']();cp.cuda.get_current_stream().synchronize()
        z=cp.asnumpy(p.output);images[r['name']]=z
        r['quality_vs_direct_fp64']=fidelity(ref,z)
        r['timing']=stats([s['event_ms'] for s in r['samples']])
        r['wall_timing']=stats([s['wall_ms'] for s in r['samples']])
        r['launch_count']=p.launch_count;r['program']=clean_json(p.metadata)
        if save_images: np.save(out/(r['name']+'.npy'),z)
        print(f"{r['name']}: {r['timing']['median_ms']:.3f} ms; SER {r['quality_vs_direct_fp64']['ser_db']:.3f} dB",flush=True)
    first=images[cs[0]['record']['name']]
    for c in cs:
        z=images[c['record']['name']]
        c['record']['vs_first_candidate']=dict(bitwise_equal=bool(np.array_equal(first,z)),quality=fidelity(first,z))
    result=dict(key=key,seed=20260918,repeats=repeats,warmups=warmups,orders=orders,records=[c['record'] for c in cs])
    save_json(out/(key+'.json'),clean_json(result))
    return result,images


def validate(backend,out):
    from resident_precision import make_lut_program
    rng=np.random.default_rng(20260918);rows=[]
    for sign,side in [(-1,1),(-1,-1),(1,1),(1,-1)]:
        n,ns=31,512;t=np.arange(n)
        pos=np.column_stack((.05*np.sin(.37*t),np.linspace(-15,15,n),18+.1*np.cos(.23*t)))
        data=(rng.normal(size=(n,ns))+1j*rng.normal(size=(n,ns))).astype(np.complex64)
        w=hann(n);ph=np.linspace(-.2,.3,n);k=sign*abs(K)
        plan=make_plan(pos,(1080.,1120.,-.05,.05),leaf=8)
        from gpu_backend import build_ffbp_plan
        plan=build_ffbp_plan(pos,sign*FC,(1080.,1120.,-.05,.05),side=side,leaf_size=8,range_spacing=.5,angular_oversample=3.)
        dev=backend.upload(data,pos,w,ph,plan)
        for mode in ('fp64','fp32','mixed'):
            a=ResidentProgram(backend,plan,dev,mode,900.,.5,k,variant='original')
            b=BatchedResidentProgram(backend,plan,dev,mode,900.,.5,k)
            a.execute();b.execute();cp.cuda.get_current_stream().synchronize()
            za,zb=cp.asnumpy(a.output),cp.asnumpy(b.output);q=fidelity(za,zb)
            assert q['finite_fraction']==1 and q['relative_complex_error']<1e-6,(sign,side,mode,q)
            assert b.capture_graph();b.replay();cp.cuda.get_current_stream().synchronize()
            assert np.array_equal(zb,cp.asnumpy(b.output))
            rows.append(dict(sign=sign,side=side,mode=mode,variant='batched',quality=q,bitwise_equal=bool(np.array_equal(za,zb)),graph_equal=True))
            a.close();b.close()
        a=ResidentProgram(backend,plan,dev,'fp64',900.,.5,k,algorithm='direct',chunk_pulses=7)
        b=make_lut_program(backend,plan,dev,900.,.5,k,chunk_pulses=7)
        a.execute();b.execute();cp.cuda.get_current_stream().synchronize()
        za,zb=cp.asnumpy(a.output),cp.asnumpy(b.output);q=fidelity(za,zb)
        assert q['finite_fraction']==1 and q['relative_complex_error']<1e-4,(sign,side,'LUT',q)
        assert b.capture_graph();b.replay();cp.cuda.get_current_stream().synchronize()
        assert np.array_equal(zb,cp.asnumpy(b.output))
        rows.append(dict(sign=sign,side=side,mode='mixed',variant='phase_lut',quality=q,graph_equal=True))
        a.close();b.close();del dev;gc.collect();cp.get_default_memory_pool().free_all_blocks()
        print(f'Followup validation passed sign={sign} side={side}',flush=True)
    save_json(out/'followup_validation.json',clean_json(dict(passed=True,records=rows)))


def stage_diagnostic(p,out,key):
    # Instrumented eager execution: diagnostic overhead, not the Graph benchmark.
    rows=[];start,end=cp.cuda.Event(),cp.cuda.Event();start.record()
    for name,kernel,grid,block,args in p._schedule:
        a,b=cp.cuda.Event(),cp.cuda.Event();a.record();kernel(grid,block,args);b.record()
        rows.append((name,a,b,kernel,grid,block))
    end.record();end.synchronize()
    records=[]
    for name,a,b,kernel,grid,block in rows:
        records.append(dict(name=name,elapsed_ms=float(cp.cuda.get_elapsed_time(a,b)),grid=grid,block=block,attributes=kernel.attributes))
    total=float(cp.cuda.get_elapsed_time(start,end))
    save_json(out/(key+'.json'),clean_json(dict(note='Instrumented eager execution; events add overhead. Kernel attributes are compiler resource counts, not achieved occupancy or hardware counters.',outer_ms=total,records=records)))


def run_batch(backend,device,pos,out,scales,repeats,warmups):
    for name in scales:
        base=make_plan(pos,EXTENTS[name]);dev=for_plan(device,base)
        ref=np.load(out/f'{name}_direct_reference.npy')
        save_axes(base,out,name)
        for mode in ('fp32','mixed'):
            cs=[]
            for var in ('original','fused'):
                p=ResidentProgram(backend,base,dev,mode,RSTART,DATA_DR,K,variant=var)
                cs.append(candidate(p,f'batch_{name}_{mode}_{var}_graph',graph=True,mode=mode,variant=var,leaf=64,internal_os=3.))
            p=BatchedResidentProgram(backend,base,dev,mode,RSTART,DATA_DR,K)
            cs.append(candidate(p,f'batch_{name}_{mode}_batched',mode=mode,variant='batched',leaf=64,internal_os=3.))
            cs.append(candidate(p,f'batch_{name}_{mode}_batched_graph',graph=True,mode=mode,variant='batched',leaf=64,internal_os=3.))
            result,_=measure(cs,ref,out,f'batch_{name}_{mode}',repeats,warmups,save_images=name=='small')
            for r in result['records']:
                assert r['vs_first_candidate']['quality']['relative_complex_error']<1e-6,r
            stage_diagnostic(p,out,f'stages_{name}_{mode}')
            dispose(cs)
        del dev,ref;gc.collect();cp.get_default_memory_pool().free_all_blocks()


def save_axes(plan,out,name):
    np.savez(out/f'{name}_axes.npz',r=plan.r0+plan.dr*np.arange(plan.nr),u=plan.u0+plan.du*np.arange(plan.nu),origin=plan.origin,side=plan.side)


def run_guard(backend,device,pos,out,repeats,warmups):
    base=make_plan(pos,EXTENTS['small']);ref=np.load(out/'small_direct_reference.npy')
    for leaf,os in ((64,3.),(64,6.),(1024,6.),(1024,12.)):
        plain=make_plan(pos,base.extent,leaf,os,base=base)
        cs=[];audits=[]
        for extra in (0,8,16):
            plan=extend_plan_radial_guard(plain,extra)
            audit=audit_support(plan);audits.append(dict(extra=extra,audit=audit))
            dev=for_plan(device,plan)
            for mode in ('fp32','mixed'):
                p=BatchedResidentProgram(backend,plan,dev,mode,RSTART,DATA_DR,K)
                cs.append(candidate(p,f'guard_l{leaf}_os{os:g}_g{extra}_{mode}',graph=True,mode=mode,variant='batched',leaf=leaf,internal_os=os,extra_guard=extra,support_audit_passed=audit['passed'],counts=plan.counts))
        result,images=measure(cs,ref,out,f'guard_l{leaf}_os{os:g}',repeats,warmups,save_images=True)
        # Compare guard extension within each precision, independent of reference.
        for r in result['records']:
            prefix=f'guard_l{leaf}_os{os:g}'
            mode=r['mode'];z=images[r['name']]
            r['vs_guard0_same_mode']=fidelity(images[f'{prefix}_g0_{mode}'],z)
            r['vs_guard16_same_mode']=fidelity(images[f'{prefix}_g16_{mode}'],z)
        result['support_audits']=audits;save_json(out/f'guard_l{leaf}_os{os:g}.json',clean_json(result))
        dispose(cs)


def run_precision(backend,device,pos,out,repeats,warmups):
    from resident_precision import make_lut_program
    base=make_plan(pos,EXTENTS['small']);ref=np.load(out/'small_direct_reference.npy');cs=[]
    for mode in ('fp32','mixed'):
        p=ResidentProgram(backend,base,device,mode,RSTART,DATA_DR,K,algorithm='direct',chunk_pulses=256)
        cs.append(candidate(p,f'precision_direct_{mode}_c256_graph',graph=True,mode=mode,variant='original',chunk=256))
    module,proxies=compile_probes(backend)
    for label,proxy in proxies.items():
        p=ResidentProgram(proxy,base,device,'mixed',RSTART,DATA_DR,K,algorithm='direct',chunk_pulses=256)
        cs.append(candidate(p,f'precision_direct_{label}_c256_graph',graph=True,mode='mixed',variant=label,chunk=256))
    p=make_lut_program(backend,base,device,RSTART,DATA_DR,K,chunk_pulses=256)
    cs.append(candidate(p,'precision_direct_phase_lut_c256_graph',graph=True,mode='mixed',variant='phase_lut',chunk=256))
    measure(cs,ref,out,'precision_small',repeats,warmups,save_images=True);dispose(cs)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--suite',choices=['validation','batch','guard','precision','all'],default='all')
    ap.add_argument('--scales',nargs='+',default=list(EXTENTS),choices=list(EXTENTS));ap.add_argument('--repeats',type=int,default=11)
    ap.add_argument('--warmups',type=int,default=3);ap.add_argument('--output',type=Path,default=DEFAULT_OUT);args=ap.parse_args()
    out=args.output.resolve();out.mkdir(exist_ok=True,parents=True);monitor=None;manifest={}
    telemetry=(out/f'telemetry_followup_{args.suite}_{int(time.time())}.csv').open('w',encoding='utf-8')
    mp=out/f'manifest_followup_{args.suite}.json'
    try:
        monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,pstate,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,utilization.gpu,memory.used','--format=csv','--loop-ms=1000'],stdout=telemetry,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        backend=CudaBaseline();manifest=dict(status='running',started_unix=time.time(),environment=cuda_environment(),arguments=vars(args)|{'output':str(out)},source_sha256={})
        for p in [Path(__file__),*list((ROOT/'fast_bp').glob('*resident*.py')),*list((ROOT/'fast_bp/gpu').glob('*.cu'))]:
            digest=hashlib.sha256(p.read_bytes()).hexdigest();manifest['source_sha256'][str(p)]=digest
            folder=out/'source_snapshot'/digest[:12];folder.mkdir(exist_ok=True,parents=True);(folder/p.name).write_bytes(p.read_bytes())
        save_json(mp,clean_json(manifest))
        if args.suite in ('all','validation'):validate(backend,out)
        if args.suite!='validation':
            data=np.load(ROOT/'fast_bp/results/range_compressed.npy',mmap_mode='r');pos=np.load(ROOT/'fast_bp/results/autofocus_position.npy')
            base=make_plan(pos,EXTENTS['small']);device=backend.upload(data,pos,hann(len(pos)),np.zeros(len(pos)),base)
            if args.suite in ('all','batch'):run_batch(backend,device,pos,out,args.scales,args.repeats,args.warmups)
            if args.suite in ('all','guard'):run_guard(backend,device,pos,out,args.repeats,args.warmups)
            if args.suite in ('all','precision'):run_precision(backend,device,pos,out,args.repeats,args.warmups)
        manifest.update(status='completed',completed_unix=time.time());save_json(mp,clean_json(manifest))
    except Exception as e:
        manifest.update(status='failed',error=f'{type(e).__name__}: {e}',failed_unix=time.time());save_json(mp,clean_json(manifest));raise
    finally:
        if monitor is not None:monitor.terminate();monitor.wait(timeout=10)
        telemetry.close()


if __name__=='__main__':main()
