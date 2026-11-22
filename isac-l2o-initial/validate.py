import json
import numpy as np
from scipy.linalg import eigh
from model import *
def checks():
 d=generate(5,801);c=d['cfg'];rng=np.random.default_rng(9)
 W=(rng.normal(size=(5,c.N,c.K))+1j*rng.normal(size=(5,c.N,c.K)))/5;theta=rng.uniform(-2,2,(5,c.M));a=evaluate(d,W,theta,gradient=True)
 errors=[];h=1e-6
 for b in range(5):
  dd=subset(d,slice(b,b+1));ww=W[b:b+1].copy();tt=theta[b:b+1].copy();analytic=[];numeric=[]
  for n in range(c.N):
   for k in range(c.K):
    for imaginary in [False,True]:
     direction=1j if imaginary else 1;plus=ww.copy();minus=ww.copy();plus[0,n,k]+=h*direction;minus[0,n,k]-=h*direction
     numeric.append((evaluate(dd,plus,tt)['loss'][0]-evaluate(dd,minus,tt)['loss'][0])/(2*h))
     analytic.append(2*(a['gW'][b,n,k].imag if imaginary else a['gW'][b,n,k].real))
  for m in range(c.M):
   plus=tt.copy();minus=tt.copy();plus[0,m]+=h;minus[0,m]-=h
   numeric.append((evaluate(dd,ww,plus)['loss'][0]-evaluate(dd,ww,minus)['loss'][0])/(2*h));analytic.append(a['gtheta'][b,m])
  errors.append(float(np.linalg.norm(np.array(numeric)-analytic)/max(np.linalg.norm(numeric),1e-12)))
 assert max(errors)<1e-5,errors
 v=effective(d,theta);z=v@W;ev_errors=[]
 for b in range(5):
  R=c.noise_sense*np.eye(c.N,dtype=complex)
  for j in range(c.K+1,c.K+1+c.L):R+=d['beta'][b,j]*np.sum(abs(z[b,j])**2)*np.outer(v[b,j],v[b,j].conj())
  S=d['beta'][b,c.K]*np.sum(abs(z[b,c.K])**2)*np.outer(v[b,c.K],v[b,c.K].conj())
  lam=eigh(S,R,eigvals_only=True)[-1];ev_errors.append(float(abs(lam-a['scnr'][b])/lam))
 assert max(ev_errors)<1e-8
 dd=generate(128,123);zf,th,good=initialize(dd);out=evaluate(dd,zf,th)
 assert good.any() and np.min(out['sinr'][good])>=c.gamma*(1-1e-8)
 res=rollout(generate(16,333),depth=20,backtrack=True,history=True)
 assert np.all(np.diff(res['trace'],axis=0)>=-1e-10)
 assert np.min(res['sinr'][res['feasible']])>=c.gamma*(1-1e-8)
 return dict(gradient_relative_errors=errors,receiver_relative_errors=ev_errors,zf_feasible_count=int(good.sum()),zf_instances=128,minimum_accepted_zf_sinr=float(out['sinr'][good].min()),backtracking_monotonic=True,power_max=float(res['power'].max()),phase_modulus_max_error=float(np.max(abs(abs(np.exp(1j*res['theta']))-1))))
if __name__=='__main__':
 r=checks();print(json.dumps(r,indent=2));open('results/validation.json','w').write(json.dumps(r,indent=2))
