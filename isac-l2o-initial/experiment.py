import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json,time,platform,csv
from pathlib import Path
from dataclasses import asdict
import numpy as np
from model import *
R=Path('results');R.mkdir(exist_ok=True)
DEPTH=8
train=generate(64,1001);val=generate(32,2002);test=generate(128,3003);ood=generate(128,4004,shift=True)
for name,d in [('train',train),('validation',val),('test',test),('clutter_shift',ood)]:
 np.savez_compressed(R/(name+'.npz'),**{k:v for k,v in d.items() if k!='cfg'})

def utility(out):
 # Failure has a deliberately strong, explicit training penalty.
 return float(np.mean(np.where(out['feasible'],np.clip(np.log(out['scnr']),-20,20),-30.)))

def logit(x):return np.log(x/(1-x))

print('Tuning fixed schedule on validation only',flush=True)
best=(-np.inf,None)
for init in ['zero','target','mixture']:
 for a in [.02,.06,.15]:
  for b in [.03,.15,.5]:
   score=utility(rollout(val,depth=DEPTH,init=init,alpha=a,phase_step=b))
   if score>best[0]:best=(score,dict(init=init,alpha=a,phase_step=b))
base=best[1];print('Fixed baseline',base,'validation utility',best[0],flush=True)
basevec=np.r_[np.full(DEPTH,logit((base['alpha']-.005)/.395)),np.full(DEPTH,logit((base['phase_step']-.005)/.795))]
# Stable physics-dictionary mixture for the starting initialization model.
ip=np.zeros((3,3));ip[0,{'target':0,'zero':0,'mixture':0}[base['init']]]=2 if base['init']!='mixture' else 0
initvec=ip.ravel()

def predict(d,mode,p):
 if mode=='inside':return rollout(d,depth=DEPTH,params=p,init=base['init'])
 if mode=='before':return rollout(d,depth=DEPTH,params=np.r_[basevec,p],learn_init=True)
 if mode=='combined':return rollout(d,depth=DEPTH,params=p,learn_init=True)
 raise ValueError(mode)

models={};logs=[];total=time.perf_counter()
for seed in [17,23,31]:
 for mode in ['inside','before','combined']:
  rng=np.random.default_rng(seed);mean={'inside':basevec,'before':initvec,'combined':np.r_[basevec,initvec]}[mode].copy()
  std=np.ones(len(mean))*.9;bestp=mean.copy();bestval=utility(predict(val,mode,mean));start=time.perf_counter()
  for gen in range(15):
   population=mean+rng.normal(size=(20,len(mean)))*std
   population[0]=mean
   scores=np.array([utility(predict(train,mode,p)) for p in population])
   elite=population[np.argsort(scores)[-5:]]
   mean=.3*mean+.7*elite.mean(0);std=np.maximum(.15,.3*std+.7*elite.std(0))
   champion=population[scores.argmax()];vs=utility(predict(val,mode,champion))
   if vs>bestval:bestval=vs;bestp=champion.copy()
   logs.append(dict(seed=seed,mode=mode,generation=gen,train_best=float(scores.max()),validation_candidate=vs,validation_best=bestval))
  elapsed=time.perf_counter()-start;name=f'{mode}_{seed}';models[name]=bestp
  np.savez(R/(name+'.npz'),params=bestp)
  print(name,'validation',round(bestval,4),'seconds',round(elapsed,2),flush=True)
training_s=time.perf_counter()-total
(R/'training_log.json').write_text(json.dumps(logs,indent=2))
rows=[];raw={};functions={
 'ZF initialization':lambda d:rollout(d,depth=0,init=base['init']),
 'Fixed PGA (8)':lambda d:rollout(d,depth=DEPTH,**base),
 'Backtracking PGA (80)':lambda d:rollout(d,depth=80,backtrack=True,**base)}
for mode in ['before','inside','combined']:
 for seed in [17,23,31]:
  name=f'{mode}_{seed}';functions[name]=lambda d,mode=mode,p=models[name]:predict(d,mode,p)
for dataset,d in [('test',test),('clutter_shift',ood)]:
 for name,fun in functions.items():
  out=fun(d);raw[(dataset,name)]=out
  feasible=out['feasible'];db=10*np.log10(out['scnr'])
  rows.append(dict(dataset=dataset,method=name,n=len(feasible),feasible=int(feasible.sum()),feasible_percent=100*float(feasible.mean()),scnr_db_mean_feasible=float(db[feasible].mean()),scnr_db_median_feasible=float(np.median(db[feasible])),minimum_sinr_feasible=float(out['sinr'][feasible].min()),power_max=float(out['power'].max()),utility=utility(out)))
  np.savez_compressed(R/(dataset+'_'+name.replace(' ','_').replace('(','').replace(')','')+'.npz'),**out)
 print('Evaluated',dataset,flush=True)
# Deployment-like batch size 1, three timed repetitions on the same 32 instances.
# Common channel/feature construction is excluded and timed separately.
from benchmark import benchmark
timings=benchmark(test,functions)
with (R/'summary.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(R/'timings.json').write_text(json.dumps(timings,indent=2))
# Shared-feasible paired comparisons, preserving all failures in coverage metrics.
paired={}
for dataset in ['test','clutter_shift']:
 ref=raw[(dataset,'Fixed PGA (8)')]
 for mode in ['before','inside','combined']:
  values=[];common_counts=[]
  for seed in [17,23,31]:
   out=raw[(dataset,f'{mode}_{seed}')];common=ref['feasible']&out['feasible'];common_counts.append(int(common.sum()))
   values.append(10*np.log10(out['scnr'][common]/ref['scnr'][common]))
  paired[dataset+'_'+mode]=dict(mean_gain_db_per_seed=[float(x.mean()) for x in values],common_feasible_counts=common_counts)
(R/'paired.json').write_text(json.dumps(paired,indent=2))
# Generation and feature construction costs are common to all methods here.
genms=[]
for i in range(96):
 t=time.perf_counter();generate(1,9000+i);genms.append((time.perf_counter()-t)*1000)
meta=dict(config=asdict(test['cfg']),depth=DEPTH,training_instances=64,validation_instances=32,test_instances=128,shift_instances=128,training_seeds=[17,23,31],split_seeds=[1001,2002,3003,4004],base=base,training_seconds=training_s,training_algorithm='Cross-entropy method: 15 generations x 20 candidates, elite 5; validation-selected checkpoint',python=platform.python_version(),platform=platform.platform(),numpy=np.__version__,cpu=platform.processor(),threads=os.environ.get('OPENBLAS_NUM_THREADS'),common_synthetic_input_preparation_median_ms=float(np.median(genms)))
(R/'metadata.json').write_text(json.dumps(meta,indent=2));print('DONE',json.dumps(meta),flush=True)
