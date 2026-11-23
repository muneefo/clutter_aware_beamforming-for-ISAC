"""Reproduce the fixed, model-based solver and MATLAB baseline replay.
Usage: python verify_fast_beams.py /path/to/Clutter-Aware-ISAC-Tutorial
Dependencies: numpy, scipy, threadpoolctl. Writes fast_results.csv in the working directory.
"""
import argparse
import csv
from pathlib import Path
from types import SimpleNamespace
from time import perf_counter
from threadpoolctl import threadpool_limits
from scipy.io import loadmat
from scipy.linalg import eigh
from scipy.optimize import linprog
from scipy import sparse
import numpy as np
from math import exp,log,sqrt,erfc,pi
R=3.6541528853610088
area=R*exp(-R*R/2)+sqrt(pi/2)*erfc(R/sqrt(2))
edges=np.zeros(257)
edges[255:]=[R,area/exp(-R*R/2)]
for j in range(254,0,-1):
    x=edges[j+1]; edges[j]=sqrt(-2*log(area/x+exp(-.5*x*x)))
pdf=np.exp(-.5*edges**2)
class MatlabTwister:
    def __init__(self,seed):
        self.mt=np.random.RandomState(seed or 5489); self.buffer=iter(())
    def word(self):
        try: return next(self.buffer)
        except StopIteration:
            self.buffer=iter(self.mt.randint(0,2**32,16384,dtype=np.uint32).tolist())
            return next(self.buffer)
    def uniform(self):
        return ((self.word()>>5)*67108864+(self.word()>>6))*2**-53
    def normal(self):
        while True:
            w=self.word(); z=self.word(); j=(z>>24)+1
            v=(((w>>3)*16777216+(z & 16777215))*2**-52-1)*edges[j]
            if abs(v)<=edges[j-1]: return v
            if j<256:
                if pdf[j]+self.uniform()*(pdf[j-1]-pdf[j])<exp(-.5*v*v): return v
            else:
                while True:
                    x=-log(self.uniform())/R
                    if -2*log(self.uniform())>x*x: return np.copysign(R+x,v)
    def complex(self,shape):
        m=int(np.prod(shape))
        def arr(): return np.fromiter((self.normal() for _ in range(m)),float,count=m).reshape(shape,order='F')
        return (arr()+1j*arr())/sqrt(2)

def herm(m): return (m+m.conj().T)/2

def unit(v): return v/np.linalg.norm(v)

class Kernels:
    def __init__(self,kernels):
        self.maps=np.stack(kernels)
        self.adjoint=self.maps.conj().swapaxes(1,2).copy()
    def apply(self,X,back=False):
        m=self.adjoint if back else self.maps
        y=(m@X.swapaxes(1,2).reshape(len(X),-1,1))[:,:,0].reshape(X.shape).swapaxes(1,2)
        return (y+y.conj().swapaxes(1,2))/2

def load_case(repo):
    f=loadmat(Path(repo)/'data/covMat_BLP.mat',simplify_cells=True,variable_names=['para','Vcc','Vcc_hat','R_eta'])
    p=f['para']; x=SimpleNamespace(T=p['Nt'],R=p['Nr'],S=p['Ns'],N=32,K=4,P=100.,noise=1e-19*p['deltaf'],g=np.full((32,4),10**.8))
    freq=p['fc']+np.arange(x.N)*p['deltaf']
    def a(sz): return np.exp(-1j*np.pi*np.sin(np.deg2rad(-10))*np.arange(sz)[None,:]*freq[:,None]/p['fc'])/np.sqrt(sz)
    x.a=a(x.T); x.b=a(x.R); x.sig=x.P*(p['c0']/freq)**2*1e4*10**.5/((4*np.pi)**3*41.8**4)
    x.Reta=f['R_eta'].transpose(2,0,1)
    x.est=Kernels(f['Vcc_hat']); x.truth=Kernels(f['Vcc'])
    return x
def scene(x,seed):
    rng=MatlabTwister(seed)
    h=rng.complex((x.T,x.N,x.K)).transpose(1,0,2)
    dist=np.array([50+int(351*rng.uniform()) for _ in range(x.K)])
    x.H=h*10**((-30-27*np.log10(dist))/20)
    return x

def fixed_receiver(x,Q,kernel,adjoint=True):
    cov=kernel.apply(Q)+x.Reta
    u=np.linalg.solve(cov,x.b[:,:,None])[:,:,0]
    u/=np.einsum('ni,ni->n',x.b.conj(),u)[:,None]
    uu=u[:,:,None]*u[:,None,:].conj()
    hot=np.einsum('nij,nji->',uu,x.Reta).real
    coef=x.sig*abs(np.einsum('ni,ni->n',u.conj(),x.b))**2
    A=coef[:,None,None]*x.a[:,:,None]*x.a[:,None,:].conj()
    C=kernel.apply(uu,True)
    v=np.einsum('nij,nji->',A,Q).real/np.einsum('nij,nji->',cov,uu).real
    return v,A,C,hot,u
def true_score(x,Q,u,kernel):
    target=np.einsum('ni,nij,nj->n',x.a.conj(),Q,x.a).real
    num=np.sum(x.sig*abs(np.sum(u.conj()*x.b,axis=1))**2*target)
    dist=kernel.apply(Q)+x.Reta
    return 10*np.log10(num/np.einsum('ni,nij,nj->',u.conj(),dist,u).real)

def recover(x,H,dirs,tags,powers):
    W=np.zeros((x.N,x.T,x.S),complex)
    for n,(V,ids,p) in enumerate(zip(dirs,tags,powers)):
        V=V*np.sqrt(np.maximum(p,0))
        Q=V@V.conj().T
        for k,h in enumerate(H[n].T):
            Vk=V[:,ids==k]
            z=Vk.conj().T@h
            if np.linalg.norm(z)==0: continue
            W[n,:,k]=Vk@(z/np.linalg.norm(z))
        remaining=herm(Q-W[n,:,:x.K]@W[n,:,:x.K].conj().T)
        d,U=eigh(remaining); L=x.S-x.K
        if L: W[n,:,x.K:]=U[:,-L:]*np.sqrt(np.maximum(d[-L:],0))
    return W

def audit(x,W):
    power=np.linalg.norm(W)**2
    y=abs(x.H.conj().transpose(0,2,1)@W)**2
    k=np.arange(x.K); desired=y[:,k,k].copy(); y[:,k,k]=0
    return 10*np.log10(np.min(desired/((y.sum(2)+x.noise)*x.g))),power

def seed_bank(x):
    H=x.H/np.linalg.norm(x.H,axis=1,keepdims=True)
    Z=np.linalg.pinv(H.conj().transpose(0,2,1))
    d=x.noise/np.linalg.norm(x.H,axis=1)**2/x.P
    bank=[]; tags=[]; W=np.zeros((x.N,x.T,x.S),complex)
    for n,(hn,z,a) in enumerate(zip(H,Z,x.a)):
        orth=a-z@(hn.conj().T@a)
        V=[unit(v) for v in z.T]+[unit(orth),a]; ids=list(range(x.K))+[x.K]*2
        for k in range(x.K):
            # All channel responses except the desired one are nulled.
            v=orth+z[:,k]*np.vdot(hn[:,k],a)
            V.extend([a,unit(v)]); ids.extend([k,k])
        bank.append(np.column_stack(V)); tags.append(np.array(ids))
        W[n,:,:x.K]=z*np.sqrt(x.g[n]*x.P*d[n]*(1+2e-5))
        W[n,:,x.K]=unit(orth)
    used=np.linalg.norm(W[:,:,:x.K])**2
    if used>=x.P: raise ValueError('ZF seed unavailable')
    W[:,:,x.K]*=np.sqrt((x.P-used)/x.N)
    return H,d,bank,tags,W

def master(x,H,d,V,owner,A,C,hot,profile=None,lp_audit=None,objective_scale=None):
    if profile is not None:
        started=perf_counter()
    # Unit transmit power and unit fixed receive-noise offset.
    P=x.P*(1-1e-8); d=d*x.P/P
    reward=[np.real((v.conj()*(a@v)).sum(0))*P/hot for a,v in zip(A,V)]
    penalty=[np.maximum(0,np.real((v.conj()*(c@v)).sum(0)))*P/hot for c,v in zip(C,V)]
    scale=objective_scale if objective_scale is not None else max(np.max(z) for z in reward)
    ar=np.concatenate(reward)/scale; cr=np.concatenate(penalty)
    g=x.g*(1+2e-5); rowScale=1/np.sqrt(d)
    blocks=[]
    for n,v in enumerate(V):
        gains=abs(H[n].conj().T@v)**2
        coeff=(1+1/g[n,:,None])*(np.arange(x.K)[:,None]==owner[n][None,:])-1
        blocks.append(-gains*coeff*rowScale[n,:,None])
    B=sparse.block_diag(blocks,format='csc')
    I=sparse.vstack([sparse.hstack([B,(d*rowScale).reshape(-1,1)]),np.r_[np.ones(len(ar)),-1][None,:]])
    objective=-np.r_[ar,0]; ub=np.zeros(x.N*x.K+1); eq=np.r_[cr,1][None,:]
    options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9}
    if profile is not None:
        options['threads']=1
        profile['build']+=perf_counter()-started
        started=perf_counter()
    res=linprog(objective,A_ub=I,b_ub=ub,A_eq=eq,b_eq=[1],
                bounds=(0,None),method='highs',options=options)
    if profile is not None:
        profile['solve']+=perf_counter()-started
    if not res.success: raise RuntimeError(res.message)
    power=np.split(res.x[:-1]/res.x[-1]*P,np.cumsum([v.shape[1] for v in V])[:-1])
    mu=-res.ineqlin.marginals[:-1].reshape(x.N,x.K)*rowScale
    price=-res.ineqlin.marginals[-1]; eta=-res.fun
    if lp_audit is not None:
        lp_audit.update(scale=scale, safe_power=P, normalized_noise=d, y_t=res.x,
                        target=ar, clutter=cr, inequalities=I,
                        psi_dual=float(-res.eqlin.marginals[0]),
                        eq_violation=float(np.max(np.abs(res.eqlin.residual))),
                        inequality_violation=float(max(0., -np.min(res.ineqlin.residual))),
                        variable_violation=float(max(0., -np.min(res.x))),
                        dual_column_violation=float(max(0., -np.min(res.lower.marginals))))
    gram=A*(P/hot/scale)-eta*C*(P/hot)
    base=gram-(H*mu[:,None,:])@H.conj().transpose(0,2,1)
    return power,base,mu,price,eta,g

def price_beams(x,H,mu,base,price,g,threshold=1e-7):
    all_dirs=[]; all_tag=[]; residual=0.
    for n in range(x.N):
        vlist=[]; ids=[]
        for j in range(x.K+1):
            Z=base[n]
            if j<x.K: Z=Z+mu[n,j]*(1+1/g[n,j])*np.outer(H[n,:,j],H[n,:,j].conj())
            val,V=eigh(herm(Z),subset_by_index=[x.T-1,x.T-1],check_finite=False)
            excess=val[0]-price
            residual=max(residual,excess)
            if excess>threshold:
                vlist.append(V[:,0]); ids.append(j)
        all_dirs.append(np.array(vlist,complex).reshape(-1,x.T).T)
        all_tag.append(np.array(ids,int))
    return all_dirs,all_tag,residual

def solve(x,outer=4,inner=2,tol=.002):
    H,d,V,owners,W=seed_bank(x); rankLog=[]
    Q=W@W.conj().transpose(0,2,1)
    eta,A,C,hot,U=fixed_receiver(x,Q,x.est)
    for j in range(outer):
        old=eta; best_fixed=eta
        for k in range(inner):
            power,base,mu,price,theta,g=master(x,H,d,V,owners,A,C,hot)
            Y=recover(x,H,V,owners,power)
            margin,p=audit(x,Y)
            feasible = margin >= -1e-4 and p <= x.P*(1+1e-7)
            E=Y@Y.conj().transpose(0,2,1)
            s=(A*E.swapaxes(1,2)).sum().real/((C*E.swapaxes(1,2)).sum().real+hot)
            if feasible and s>=best_fixed*(1-1e-10):
                Q=E; W=Y; best_fixed=s
            dv,tag,rho=price_beams(x,H,mu,base,price,g)
            for n in range(x.N):
                V[n]=np.column_stack((V[n],dv[n]))
                owners[n]=np.r_[owners[n],tag[n]]
            boundDB=10*np.log10(1+rho/max(theta,1e-16))
            rankLog.append(boundDB)
            if boundDB<=tol: break
        eta,A,C,hot,U=fixed_receiver(x,Q,x.est)
        if 10*np.log10(eta/old)<tol and boundDB<=tol: break
    return W,U,rankLog

def align(x,W):
    Y=W.copy(); K=x.K
    hy=x.H.conj().transpose(0,2,1)@W[:,:,:K]
    Y[:,:,:K]*=np.exp(-1j*np.angle(np.diagonal(hy,axis1=1,axis2=2)))[:,None,:]
    return Y

def mix(x,Y,anchor):
    Y=align(x,Y);Y/=max(1,np.linalg.norm(Y)/np.sqrt(x.P))
    if audit(x,Y)[0]>=0: return Y
    lo,hi=0.,1.
    for _ in range(23):
        t=(lo+hi)/2
        if audit(x,anchor*(1-t)+t*Y)[0]>=0:lo=t
        else:hi=t
    return anchor*(1-lo)+lo*Y

def angles(x,W,A,C,eta,hot,cold,j,D,weights=None,step=.6):
    N,T,K=x.N,x.T,x.K
    Y=W.copy()
    for n in range(N):
        H=x.H[n]; Z=W[n]; q=np.zeros(Z.shape,complex)
        hy=H.conj().T@Z; desired=np.diag(hy[:,:K]).copy()
        hy[np.arange(K),np.arange(K)]=0
        denom=np.sum(abs(hy)**2,axis=1)+x.noise; ratios=abs(desired)**2/(denom*x.g[n])
        for k in range(K):
            g=-H[:,k,None]*hy[k,:]/denom[k]
            g[:,k]=H[:,k]*desired[k]/abs(desired[k])**2
            q+=g/np.linalg.norm(g)/max(1,ratios[k])**2
        z=np.zeros(4)
        if weights is not None:
            f=[cold[n]/(cold[n]+hot[n]),np.tanh(np.log(min(ratios))),np.tanh(np.log(np.linalg.norm(Z)**2*N/x.P)),j/D]
            h=np.tanh(weights[:24].reshape((6,4),order='F')@f+weights[24:30])
            z=weights[30:54].reshape((4,6),order='F')@h+weights[54:]
        tar=A[n]@Z; cl=eta*C[n]@Z
        Y[n]+=np.sqrt(x.P/N)*step*np.exp(np.tanh(z[3]))*((np.exp(np.tanh(z[0]))*tar-np.exp(np.tanh(z[1]))*cl)/(np.linalg.norm(tar)+np.linalg.norm(cl))+.15*(1+np.tanh(z[2]))*q/K)
    return Y

def replay(x,seed):
    scene(x,seed); rng=MatlabTwister((17389+1664525*seed)%2**32)
    raw=rng.complex((x.T,x.S,x.N)).transpose(2,0,1);raw*=np.sqrt(x.P)/np.linalg.norm(raw)
    Z=np.linalg.pinv(x.H.conj().transpose(0,2,1))
    A=np.zeros_like(raw); A[:,:,:x.K]=Z*np.sqrt(x.noise*x.g[:,None,:])
    S=raw[:,:,x.K:]-Z@(x.H.conj().transpose(0,2,1)@raw[:,:,x.K:])
    need=np.linalg.norm(A)**2;A*=np.sqrt((x.P+need)/(2*need))
    A[:,:,x.K:]=S*(np.sqrt((x.P-need)/2)/np.linalg.norm(S))
    W=mix(x,raw,A)
    for j in range(1,4):
        Q=W@W.conj().transpose(0,2,1)
        eta,At,C,hot,u=fixed_receiver(x,Q,x.est)
        Y=angles(x,W,At,C,eta,None,None,j,3)
        Y=mix(x,Y,A); E=Y@Y.conj().transpose(0,2,1)
        rat=np.einsum('nij,nji->',At,E).real/(hot+np.einsum('nij,nji->',C,E).real)
        if rat>=eta: W=Y
    Q=W@W.conj().transpose(0,2,1)
    _,_,_,_,u=fixed_receiver(x,Q,x.est)
    return true_score(x,Q,u,x.truth)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('repo',type=Path)
    args=parser.parse_args()
    x=load_case(args.repo)
    ao=[13.540,13.521,13.506,13.485]  # rounded AO values supplied by the user
    anchor=[4.9818,1.3637,3.5008,2.7467]
    scene(x,310)
    solve(x,outer=1,inner=1)  # untimed, separate scene
    records=[]
    for i,seed in enumerate(range(111,115)):
        anchorDB=replay(x,seed)
        assert abs(anchorDB-anchor[i])<0.0001, (seed,anchorDB)
        durations=[]
        for _ in range(5):
            t=perf_counter()
            W,U,history=solve(x,outer=5,inner=3)
            durations.append(perf_counter()-t)
        Q=W@W.conj().transpose(0,2,1)
        db=true_score(x,Q,U,x.truth)
        design=true_score(x,Q,U,x.est)
        margin,power=audit(x,W)
        assert margin>=-1e-4 and power<=x.P*(1+1e-7)
        row=dict(scene=seed,AO_dB=ao[i],fast_dB=db,AO_minus_fast_dB=ao[i]-db,
                 median_seconds=np.median(durations),worst_SINR_margin_dB=margin,power_W=power,
                 estimated_SCNR_dB=design,LP_solves=len(history),anchor_replay_dB=anchorDB)
        records.append(row)
        print(seed, 'SCNR %.6f, gap %+.6f, time %.4f, margin %.8f, power %.8f' %
              (db,ao[i]-db,row['median_seconds'],margin,power),flush=True)
    with open('fast_results.csv','w',newline='') as out:
        w=csv.DictWriter(out,fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)
if __name__=='__main__':
    with threadpool_limits(limits=1):
        main()
