from pathlib import Path
from scipy.io import loadmat
import bench
from bench import np, SimpleNamespace,perf_counter,herm,unit
from matlab_rng import MatlabTwister

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

if __name__=='__main__':
    bench.receiver=fixed_receiver
    x=load_case(Path(__file__).parent/'authors')
    for seed in [111,112,113,114]:
        scene(x,seed)
        for outer,inner in [(1,1),(3,2),(5,3),(10,6)]:
            t=perf_counter()
            W,u,r=bench.solve(x,outer,inner)
            t=perf_counter()-t
            Q=W@W.conj().transpose(0,2,1)
            print(seed,outer,inner,len(r),round(true_score(x,Q,u,x.truth),6),round(true_score(x,Q,u,x.est),6),round(t,4),bench.audit(x,W),round(r[-1],4),flush=True)
