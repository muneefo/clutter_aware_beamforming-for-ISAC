import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json,time
from pathlib import Path
import numpy as np
from model import *
R=Path('results');meta=json.loads((R/'metadata.json').read_text());base=meta['base']
p=np.load(R/'combined_17.npz')['params']
# Dimension transfer without retraining: keep K=2 and the same aperture.
rows=[]
for n,m in [(6,12),(12,24),(24,48)]:
 d=generate(32,5100+n,Config(N=n,M=m));functions={'fixed8':lambda dd:rollout(dd,depth=8,**base),'combined17':lambda dd:rollout(dd,depth=8,params=p,learn_init=True)}
 for name,fn in functions.items():
  out=fn(d);times=[]
  for rep in range(3):
   for i in range(16):
    dd=subset(d,slice(i,i+1));t=time.perf_counter();fn(dd);times.append((time.perf_counter()-t)*1000)
  rows.append(dict(N=n,M=m,method=name,n_instances=32,feasible=int(out['feasible'].sum()),median_ms=float(np.median(times)),p95_ms=float(np.percentile(times,95))))
(R/'dimension_diagnostic.json').write_text(json.dumps(rows,indent=2))
# Nominal covariance identity checked independently by random symbol/scatterer draws.
d=generate(1,606);W,t,g=initialize(d,'mixture');u=receiver(d,W,t);v=effective(d,t)[0];c=d['cfg'];rng=np.random.default_rng(607);n=100000
symbols=(rng.normal(size=(n,c.K))+1j*rng.normal(size=(n,c.K)))/np.sqrt(2)
x=symbols@W[0].T
alphas=(rng.normal(size=(n,c.L+1))+1j*rng.normal(size=(n,c.L+1)))/np.sqrt(2)*np.sqrt(d['beta'][0,c.K:])
ytarget=alphas[:,0,None]*(x@v[c.K])[:,None]*v[c.K]
yclutter=np.zeros_like(ytarget)
for j in range(c.L):yclutter+=alphas[:,j+1,None]*(x@v[c.K+j+1])[:,None]*v[c.K+j+1]
noise=(rng.normal(size=yclutter.shape)+1j*rng.normal(size=yclutter.shape))*np.sqrt(c.noise_sense/2)
emp=np.mean(abs(ytarget@u[0].conj())**2)/np.mean(abs((yclutter+noise)@u[0].conj())**2)
exact=evaluate(d,W,t)['scnr'][0];err=float(abs(emp-exact)/exact)
assert err<.04,err
(R/'covariance_check.json').write_text(json.dumps(dict(samples=n,analytic_scnr=float(exact),empirical_scnr=float(emp),relative_error=err),indent=2))
# Timing ingredients, warm-up then repeated batch-one evaluations. Not additive.
d=generate(1,701);W,t,g=initialize(d,'mixture');u=receiver(d,W,t)
functions={'effective_channels':lambda:effective(d,t),'receiver_including_channels':lambda:receiver(d,W,t),'loss_and_gradients_including_receiver':lambda:evaluate(d,W,t,gradient=True),'power_repair_including_channels':lambda:repair(d,W,t),'metric_check_including_receiver':lambda:evaluate(d,W,t)}
profile={}
for name,fn in functions.items():
 fn();a=[]
 for _ in range(300):
  start=time.perf_counter();fn();a.append((time.perf_counter()-start)*1000)
 profile[name]=float(np.median(a))
(R/'profile.json').write_text(json.dumps(profile,indent=2));print('Additional checks passed; covariance relative error',err)
