"""Model-based joint solver from the benchmark validation, with an optional outer-iteration observer."""
from pathlib import Path
from types import SimpleNamespace
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
    ratio = np.min(desired/((y.sum(2)+x.noise)*x.g))
    return (10*np.log10(ratio) if ratio > 0 else -np.inf),power

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

def solve(x,outer=4,inner=2,tol=.002,observe=None):
    H,d,V,owners,W=seed_bank(x); rankLog=[]
    Q=W@W.conj().transpose(0,2,1)
    eta,A,C,hot,U=fixed_receiver(x,Q,x.est)
    if observe is not None:
        observe(0,0,W,U,eta)
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
        if observe is not None:
            observe(j+1,len(rankLog),W,U,eta)
        if 10*np.log10(eta/old)<tol and boundDB<=tol: break
    return W,U,rankLog

