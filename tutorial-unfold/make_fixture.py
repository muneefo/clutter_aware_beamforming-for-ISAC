"""Independent numerical check for the MATLAB small deterministic fixture."""
from pathlib import Path
import numpy as np
from scipy.io import savemat
rng=np.random.default_rng(29)
def complex_array(*shape):
    return (rng.normal(size=shape)+1j*rng.normal(size=shape))/np.sqrt(2)
nt,nr,N,K=4,3,2,2
W=complex_array(nt,nt,N); W*=np.sqrt(2)/np.linalg.norm(W)
H=complex_array(nt,N,K)
a=complex_array(nt,N); a/=np.linalg.norm(a,axis=0)
b=complex_array(nr,N); b/=np.linalg.norm(b,axis=0)
S=complex_array(2,nr,nt,N)/np.sqrt(nt*nr)
R=np.stack([np.eye(nr)*.1,np.eye(nr)*.15],axis=2)
sigma=[.8,1.2]
def solve(W):
    sig=den=cold=0.
    U=[];V=[];B=[];A=[];sinr=[]
    for n in range(N):
        w=W[:,:,n]; q=w@w.conj().T; blocks=S[:,:,:,n]
        cov=sum(s@q@s.conj().T for s in blocks); dist=cov+R[:,:,n]
        v=np.linalg.solve(dist,b[:,n]); v/=np.vdot(b[:,n],v)
        gain=sigma[n]*abs(np.vdot(v,b[:,n]))**2
        reward=gain*np.outer(a[:,n],a[:,n].conj())
        penalty=sum(np.outer(s.conj().T@v,(s.conj().T@v).conj()) for s in blocks)
        B.append(penalty); A.append(reward); U.append(v)
        kernel=sum(np.kron(s.conj(),s) for s in blocks); V.append(kernel)
        np.testing.assert_allclose(kernel@q.ravel('F'),cov.ravel('F'),atol=1e-14)
        sig+=np.trace(reward@q).real; den+=np.vdot(v,dist@v).real; cold+=np.vdot(v,cov@v).real
        powr=abs(H[:,n,:].conj().T@w)**2; desired=np.diag(powr[:,:K])
        sinr.append(desired/(.01+powr.sum(1)-desired))
    ratio=sig/den
    g=np.stack([(a-ratio*b)@W[:,:,i] for i,(a,b) in enumerate(zip(A,B))],axis=2)
    return ratio,den,dict(eta=ratio,frac=cold/den,u=np.stack(U,1),V=np.stack(V,2),grad=g,sinr=np.array(sinr))
v,den,fields=solve(W)
direction=complex_array(*W.shape); direction/=np.linalg.norm(direction)
e=1e-6
fd=(solve(W+e*direction)[0]-solve(W-e*direction)[0])/(2*e)
np.testing.assert_allclose(fd,2*np.vdot(fields['grad'],direction).real/den,rtol=1e-6,atol=1e-8)
savemat(Path(__file__).with_name('beam_checks.mat'),dict(H=H,W=W,a=a,b=b,R=R,**fields))
print('Explicit scatterer kernels and finite-difference gradient passed.')
