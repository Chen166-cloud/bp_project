"""Graph-level stage ablation, with no per-kernel event instrumentation."""
import gc
import time
import numpy as np
from scipy.signal.windows import hann
from resident_experiment import *
from batched_resident_backend import BatchedResidentProgram


def capture_schedule(program, entries):
    stream=cp.cuda.Stream(non_blocking=True)
    cp.cuda.get_current_stream().synchronize()
    with stream:
        stream.begin_capture()
        for name,kernel,grid,block,args in entries:kernel(grid,block,args)
        graph=stream.end_capture()
    return graph,stream


def main():
    out=DEFAULT_OUT;backend=CudaBaseline()
    data=np.load(ROOT/'fast_bp/results/range_compressed.npy',mmap_mode='r')
    pos=np.load(ROOT/'fast_bp/results/autofocus_position.npy');base=make_plan(pos,EXTENTS['small'])
    dev=backend.upload(data,pos,hann(len(pos)),np.zeros(len(pos)),base)
    results=[]
    for scale in EXTENTS:
        plan=make_plan(pos,EXTENTS[scale]);pd=for_plan(dev,plan)
        for mode in ('fp32','mixed'):
            p=BatchedResidentProgram(backend,plan,pd,mode,RSTART,DATA_DR,K)
            p.execute();cp.cuda.get_current_stream().synchronize()
            full=p.capture_graph();assert full
            leaf,ls=capture_schedule(p,p._schedule[:1])
            merge,ms=capture_schedule(p,p._schedule[1:])
            calls={'full':p.replay,'leaf_only':lambda:leaf.launch(stream=cp.cuda.get_current_stream()),'merge_crop_only':lambda:merge.launch(stream=cp.cuda.get_current_stream())}
            records={k:[] for k in calls};rng=np.random.default_rng(20260919)
            for call in calls.values():
                for _ in range(5):timed(call)
            for rep in range(15):
                for key in rng.permutation(list(calls)):
                    start=time.time();v,w=timed(calls[key]);records[key].append(dict(ms=v,wall_ms=w,start_unix=start))
            result=dict(scale=scale,mode=mode,shape=plan.shape,leaf=64,internal_os=3.,program=p.metadata,
                        note='Separate CUDA Graphs for full schedule, leaf only, and merge/crop only. Input/intermediates already initialized. Subset runs are diagnostics and do not produce independent images.',
                        records={k:dict(timing=stats([s['ms'] for s in rows]),samples=rows) for k,rows in records.items()})
            results.append(result);save_json(out/'graph_stage_probes.json',clean_json(results))
            print(scale,mode,{k:round(v['timing']['median_ms'],4) for k,v in result['records'].items()},flush=True)
            p.close();del p,leaf,merge,ls,ms,calls;gc.collect();cp.get_default_memory_pool().free_all_blocks()
    save_json(out/'graph_stage_probe_source.json',dict(sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))


if __name__=='__main__':main()
