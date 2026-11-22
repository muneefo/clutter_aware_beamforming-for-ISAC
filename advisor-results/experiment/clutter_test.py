"""Paired cold-clutter ablation on the BLP case. Run with the authors' repo as argument."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from experiment import hybrid, set_scene
import reference_beams as ref

class ZeroMap:
    def apply(self, covariance, back=False):
        return np.zeros_like(covariance)

def cov(w):
    return w @ w.conj().transpose(0,2,1)

def precoder(x, name):
    # Minimal QoS powers for fixed unit communication beams. Place surplus power
    # into the target direction projected onto each tone's user null space.
    g=x.H/np.linalg.norm(x.H,axis=1,keepdims=True)
    gram=g.conj().transpose(0,2,1) @ g
    if name=="MRT":
        v=g
    else:
        reg=0.1 if name=="RZF" else 0.
        v=g @ np.linalg.inv(gram+reg*np.eye(x.K))
        v=v/np.linalg.norm(v,axis=1,keepdims=True)
    response=np.abs(x.H.conj().transpose(0,2,1) @ v)**2
    gain=np.diagonal(response,axis1=1,axis2=2).copy()
    user=np.arange(x.K)
    response[:,user,user]=0.
    coupling=response*(x.g*(1+2e-5)/gain)[:,:,None]
    if np.max(np.abs(np.linalg.eigvals(coupling)))>=1:
        return None
    powers=np.linalg.solve(np.eye(x.K)-coupling,(x.noise*x.g*(1+2e-5)/gain)[:,:,None])[:,:,0]
    if powers.min()<0 or powers.sum()>x.P:
        return None
    hstar=g.conj().transpose(0,2,1)
    free=x.a-(g @ np.linalg.solve(gram,hstar @ x.a[:,:,None]))[:,:,0]
    w=np.zeros((x.N,x.T,x.S),complex)
    w[:,:,:x.K]=v*np.sqrt(powers)[:,None,:]
    w[:,:,x.K]=free/np.linalg.norm(free,axis=1,keepdims=True)*np.sqrt((x.P-powers.sum())/x.N)
    return w

def benchmark(repo):
    x=ref.load_case(repo)
    est=x.est
    zero=ZeroMap()
    records=[]; patterns=[]
    for k in (4,8):
        for gamma in (0,8,16):
            for seed in range(47001,47013):
                set_scene(x,k,gamma,seed)
                methods={}
                for design,label in [(est,'Estimated'),(zero,'Blind')]:
                    x.est=design
                    w,u=hybrid(x,'exact_LP',{})
                    methods[label]=(w,u,x.truth)
                    if label=='Blind':
                        methods['Absent']=(w,u,zero)
                x.est=est
                for name in ('ZF','MRT','RZF'):
                    w=precoder(x,name)
                    u=None if w is None else ref.fixed_receiver(x,cov(w),est)[-1]
                    methods[name]=(w,u,x.truth)
                for name,(w,u,environment) in methods.items():
                    row=dict(K=k,gamma=gamma,seed=seed,method=name,feasible=False)
                    if w is not None:
                        margin,power=ref.audit(x,w)
                        assert margin>=-1e-4 and power<=x.P*(1+1e-7)
                        q=cov(w)
                        scnr=ref.true_score(x,q,u,environment)
                        c=np.einsum('ni,nij,nj->',u.conj(),environment.apply(q),u).real
                        noise=np.einsum('ni,nij,nj->',u.conj(),x.Reta,u).real
                        row.update(feasible=True,scnr=scnr,powerW=power,marginDB=margin,coldToAdditive=c/noise)
                        if k==4 and gamma==8 and name in ('Estimated','Blind','ZF'):
                            angle=np.arange(-90,91)
                            f=1+np.arange(x.N)*120e3/28e9
                            steering=np.exp(-1j*np.pi*f[:,None,None]*np.arange(x.T)[None,:,None]*np.sin(np.deg2rad(angle)))/np.sqrt(x.T)
                            pattern=np.einsum('nta,ntv,nva->a',steering.conj(),q,steering).real/x.P
                            patterns.extend(dict(seed=seed,method=name,angle=a,fraction=p) for a,p in zip(angle,pattern))
                    records.append(row)
            print('Completed',k,gamma,flush=True)
    pd.DataFrame(records).to_csv('trials.csv',index=False)
    pd.DataFrame(patterns).to_csv('patterns.csv',index=False)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('repo',type=Path)
    with threadpool_limits(1):
        benchmark(parser.parse_args().repo)
