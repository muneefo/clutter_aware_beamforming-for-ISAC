"""Power allocation and eigenvector pricing for the tutorial's block-level precoder."""
import numpy as np
from scipy.linalg import eigh, null_space
from scipy.optimize import linprog
from scipy import sparse
from time import perf_counter
from types import SimpleNamespace

def herm(m): return (m+m.conj().T)/2

def unit(v): return v/np.linalg.norm(v)

def steer(deg,n):
    return np.exp(-1j*np.pi*np.sin(np.deg2rad(deg))*np.arange(n))/np.sqrt(n)

def fixture(seed,N=32,T=16,K=4,P=100.0,clutter=1., env=0, mismatch=0.12):
    rng=np.random.default_rng(seed); scene=np.random.default_rng(6600+env)
    def cn(shape): return (rng.normal(size=shape)+1j*rng.normal(size=shape))/np.sqrt(2)
    R=T
    a=np.array([steer(-10,T) for i in range(N)])
    b=np.array([steer(-10,R) for i in range(N)])
    sig=np.full(N,0.85)
    base=scene.uniform(-80,80,8)
    def matrices(shift,weights):
        u=np.stack([steer(t+shift,R) for t in base],axis=1)
        v=np.stack([steer(t+shift,T) for t in base],axis=1)
        return [np.einsum('rj,tj,j->jrt',u,v.conj(),np.sqrt(g)) for g in weights]
    p=clutter*scene.lognormal(-.1,.6,(N,len(base))); p[:,0]*=2.3
    truth=matrices(0,p); est=matrices(.35*mismatch/.12,p*np.exp(mismatch*scene.normal(size=p.shape)))
    hot=np.stack([np.eye(R)*.52+2.2*np.outer(steer(13,R),steer(13,R).conj()) for i in range(N)])
    distance=rng.integers(50,401,K)
    H=cn((N,T,K))*10**((-30-27*np.log10(distance))/20)
    return SimpleNamespace(N=N,T=T,R=R,K=K,S=T,P=P,a=a,b=b,sig=sig,Reta=hot,H=H,
        g=np.full((N,K),10**.8),noise=1.2e-14,est=est,truth=truth)

def receiver(x, Q, kernel, adjoint=True):
    Es=[]; C=[]; A=[]; us=[]; num=hot=cold=0.
    for n,F in enumerate(kernel):
        dist=herm((F@Q[n]@F.conj().transpose(0,2,1)).sum(axis=0)+x.Reta[n])
        u=np.linalg.solve(dist,x.b[n]); u/=np.vdot(x.b[n],u)
        h=np.einsum('jrt,r->jt',F.conj(),u)
        cn=herm(h.T@h.conj()); an=x.sig[n]*abs(np.vdot(u,x.b[n]))**2*np.outer(x.a[n],x.a[n].conj())
        num+=np.trace(an@Q[n]).real; cold+=np.trace(cn@Q[n]).real
        hot+=np.vdot(u,x.Reta[n]@u).real
        A.append(an); C.append(cn); us.append(u)
    return num/(cold+hot),np.array(A),np.array(C),hot,np.array(us)

def score(x,Q,U,kernel):
    num=den=0.
    for n,F in enumerate(kernel):
        u=U[n]; V=(F@Q[n]@F.conj().transpose(0,2,1)).sum(axis=0)
        num+=x.sig[n]*abs(np.vdot(u,x.b[n]))**2*np.vdot(x.a[n],Q[n]@x.a[n]).real
        den+=np.vdot(u,(x.Reta[n]+V)@u).real
    return 10*np.log10(num/den)

def recover(x,H,dirs,tags,powers):
    W=np.zeros((x.N,x.T,x.S),complex)
    for n,(V,ids,p) in enumerate(zip(dirs,tags,powers)):
        V=V*np.sqrt(np.maximum(p,0))
        Q=V@V.conj().T
        for k,h in enumerate(H[n].T):
            Vk=V[:,ids==k]
            z=Vk.conj().T@h
            if np.linalg.norm(z)==0: raise ValueError('empty communication beam')
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

def master(x,H,d,V,owner,A,C,hot):
    # Unit transmit power and unit fixed receive-noise offset.
    P=x.P*(1-1e-8); d=d*x.P/P
    reward=[np.real((v.conj()*(a@v)).sum(0))*P/hot for a,v in zip(A,V)]
    penalty=[np.maximum(0,np.real((v.conj()*(c@v)).sum(0)))*P/hot for c,v in zip(C,V)]
    scale=max(np.max(z) for z in reward)
    ar=np.concatenate(reward)/scale; cr=np.concatenate(penalty)
    g=x.g*(1+2e-5); rowScale=1/np.sqrt(d)
    blocks=[]
    for n,v in enumerate(V):
        gains=abs(H[n].conj().T@v)**2
        coeff=(1+1/g[n,:,None])*(np.arange(x.K)[:,None]==owner[n][None,:])-1
        blocks.append(-gains*coeff*rowScale[n,:,None])
    B=sparse.block_diag(blocks,format='csc')
    I=sparse.vstack([sparse.hstack([B,(d*rowScale).reshape(-1,1)]),np.r_[np.ones(len(ar)),-1][None,:]])
    res=linprog(-np.r_[ar,0],A_ub=I,b_ub=np.zeros(x.N*x.K+1),A_eq=np.r_[cr,1][None,:],b_eq=[1],
                bounds=(0,None),method='highs',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
    if not res.success: raise RuntimeError(res.message)
    power=np.split(res.x[:-1]/res.x[-1]*P,np.cumsum([v.shape[1] for v in V])[:-1])
    mu=-res.ineqlin.marginals[:-1].reshape(x.N,x.K)*rowScale
    price=-res.ineqlin.marginals[-1]; eta=-res.fun
    gram=A*(P/hot/scale)-eta*C*(P/hot)
    base=gram-(H*mu[:,None,:])@H.conj().transpose(0,2,1)
    return power,base,mu,price,eta,g

def price_beams(x,H,mu,base,price,g):
    all_dirs=[]; all_tag=[]; residual=0.
    for n in range(x.N):
        vlist=[]; ids=[]
        for j in range(x.K+1):
            Z=base[n]
            if j<x.K: Z=Z+mu[n,j]*(1+1/g[n,j])*np.outer(H[n,:,j],H[n,:,j].conj())
            val,V=eigh(herm(Z),subset_by_index=[x.T-1,x.T-1],check_finite=False)
            excess=val[0]-price
            residual=max(residual,excess)
            if excess>1e-7:
                vlist.append(V[:,0]); ids.append(j)
        all_dirs.append(np.array(vlist,complex).reshape(-1,x.T).T)
        all_tag.append(np.array(ids,int))
    return all_dirs,all_tag,residual

def solve(x,outer=4,inner=2,tol=.002):
    H,d,V,owners,W=seed_bank(x); rankLog=[]
    Q=W@W.conj().transpose(0,2,1)
    eta,A,C,hot,U=receiver(x,Q,x.est)
    for j in range(outer):
        old=eta
        for k in range(inner):
            power,base,mu,price,theta,g=master(x,H,d,V,owners,A,C,hot)
            Y=recover(x,H,V,owners,power)
            margin,p=audit(x,Y)
            if margin < -1e-4 or p>x.P*(1+1e-7): raise ValueError((margin,p))
            E=Y@Y.conj().transpose(0,2,1)
            s=(A*E.swapaxes(1,2)).sum().real/((C*E.swapaxes(1,2)).sum().real+hot)
            if s>=old*(1-1e-10):
                Q=E; W=Y
            dv,tag,rho=price_beams(x,H,mu,base,price,g)
            for n in range(x.N):
                V[n]=np.column_stack((V[n],dv[n]))
                owners[n]=np.r_[owners[n],tag[n]]
            boundDB=10*np.log10(1+rho/max(theta,1e-16))
            rankLog.append(boundDB)
            if boundDB<=tol: break
        eta,A,C,hot,U=receiver(x,Q,x.est)
        if 10*np.log10(eta/old)<tol and boundDB<=tol: break
    return W,U,rankLog

if __name__=='__main__':
    for n in [4,32]:
        for seed in [111,112]:
            x=fixture(seed,N=n)
            for outer,inner in [(1,1),(3,2),(8,5)]:
                t=perf_counter(); w,u,gap=solve(x,outer,inner); t=perf_counter()-t
                q=w@w.conj().transpose(0,2,1)
                print(n,seed,outer,inner,'true',round(score(x,q,u,x.truth),5),'est',round(score(x,q,u,x.est),5),'t',round(t,3),'viol',audit(x,w),'bound',gap[-1],flush=True)
