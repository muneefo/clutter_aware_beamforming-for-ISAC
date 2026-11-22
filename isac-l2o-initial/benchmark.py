"""Randomized interleaved batch-one timing; identical inputs and 3 repeats."""
import time
import numpy as np
from model import subset

def benchmark(test,functions):
 for fn in functions.values():fn(subset(test,slice(0,1)))
 samples={n:[] for n in functions};rng=np.random.default_rng(876)
 for rep in range(3):
  for i in range(32):
   dd=subset(test,slice(i,i+1))
   for name in rng.permutation(list(functions)):
    start=time.perf_counter();functions[name](dd);samples[name].append((time.perf_counter()-start)*1000)
 return {n:dict(median_ms=float(np.median(a)),p95_ms=float(np.percentile(a,95)),samples_ms=a) for n,a in samples.items()}

if __name__=='__main__':
 import json
 from pathlib import Path
 from model import *
 R=Path('results');meta=json.loads((R/'metadata.json').read_text());base=meta['base'];test=generate(128,3003)
 functions={'ZF initialization':lambda d:rollout(d,depth=0,init=base['init']),'Fixed PGA (8)':lambda d:rollout(d,depth=8,**base),'Backtracking PGA (80)':lambda d:rollout(d,depth=80,backtrack=True,**base)}
 for mode in ['before','inside','combined']:
  for seed in [17,23,31]:
   p=np.load(R/f'{mode}_{seed}.npz')['params']
   if mode=='inside':functions[f'{mode}_{seed}']=lambda d,p=p:rollout(d,depth=8,params=p,init=base['init'])
   elif mode=='combined':functions[f'{mode}_{seed}']=lambda d,p=p:rollout(d,depth=8,params=p,learn_init=True)
   else:
    def logit(x):return np.log(x/(1-x))
    steps=np.r_[np.full(8,logit((base['alpha']-.005)/.395)),np.full(8,logit((base['phase_step']-.005)/.795))]
    functions[f'{mode}_{seed}']=lambda d,p=p:rollout(d,depth=8,params=np.r_[steps,p],learn_init=True)
 # Confirm all numerical results remain unchanged after failure early-exit optimization.
 for name,fn in functions.items():
  old=np.load(R/('test_'+name.replace(' ','_').replace('(','').replace(')','')+'.npz'));new=fn(test)
  assert np.all(old['feasible']==new['feasible'])
  assert np.allclose(old['scnr'],new['scnr'],rtol=1e-10,atol=1e-12)
 times=benchmark(test,functions);(R/'timings.json').write_text(json.dumps(times,indent=2))
 meta['timing_protocol']='Randomized interleaved methods, same 32 geometries x 3 repetitions, batch 1; immediate return on failed initialization.'
 (R/'metadata.json').write_text(json.dumps(meta,indent=2));print('Numerical invariance and interleaved timing completed')
