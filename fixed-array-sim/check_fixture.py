"""Independent matrix-equation check of the saved MATLAB inputs and expected answers."""
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy.optimize import linprog, lsq_linear
from scipy.special import gammaincinv
from scipy.linalg import eigh

root=Path(__file__).resolve().parent
f=loadmat(root/'reference_fixture.mat')
def check(got, expected):
    np.testing.assert_allclose(got,expected,rtol=3e-8,atol=2e-9)

def response(n,x0,x):
    coords=(np.arange(n)-(n-1)/2)*.005+x0
    ds=np.sqrt((coords-x[0])**2+x[1]**2)
    return .01*np.exp(-2j*np.pi*ds/.01)/(4*np.pi*ds)

def check_scene(i):
    pos=f['pos'][:,:,i];truth=f['truth'][:,i]
    W=f['W'][:,:,i];alpha=f['alpha'][:,:,:,i];noise=f['noise'][:,:,i]
    t=[response(24,-.008,p) for p in pos]
    r=[response(16,.008,p) for p in pos[:3]]
    A=[np.outer(ri,ti) for ri,ti in zip(r,t[:3])]
    H=np.zeros((6,2));m=np.zeros(6)
    for b in range(6):
        probe=t[b%3].conj()/np.linalg.norm(t[b%3])
        v=r[(b+b//3)%3]/np.linalg.norm(r[(b+b//3)%3])
        amplitudes=np.array([v.conj()@a@probe for a in A[1:]])
        H[b]=abs(amplitudes)**2/2e-11
        echoes=alpha[:,b,:]@(amplitudes*np.sqrt(truth))+np.sqrt(2e-11)*noise[:,b]
        m[b]=np.vdot(echoes,echoes).real/(len(echoes)*2e-11)
    check(H,f['H'][:,:,i]);check(m,f['m'][:,i])
    qhat=lsq_linear(H,m-1,bounds=(0,np.inf)).x
    check(qhat,f['hat'][:,i])
    a=.05/12;hi=96*m/gammaincinv(96,a);lo=96*m/gammaincinv(96,1-a)
    upper=[]
    for e in np.eye(2):
        sol=linprog(-e,A_ub=np.r_[H,-H],b_ub=np.r_[hi-1,1-lo],bounds=(0,None))
        assert sol.success
        upper.append(-sol.fun)
    check(upper,f['upper'][:,i])
    Q=W@W.conj().T
    S=[a@Q@a.conj().T/2e-11 for a in A]
    R=np.eye(16)+sum(q*s for q,s in zip(upper,S[1:]))
    vals,U=eigh(S[0],R);u=U[:,-1]/np.linalg.norm(U[:,-1])
    Rtrue=np.eye(16)+sum(q*s for q,s in zip(truth,S[1:]))
    scnr=(u.conj()@S[0]@u/(u.conj()@Rtrue@u)).real
    user_sinr=[]
    for k,user in enumerate(t[3:]):
        user_sinr.append(abs(user@W[:,k])**2/(6e-7+abs(user@W[:,1-k])**2))
    check([vals[-1],scnr,*user_sinr,np.trace(Q).real],f['values'][:,i])
    check(f['metrics'][-1,:,i],f['values'][:,i])

if __name__=='__main__':
    for i in range(f['W'].shape[2]):check_scene(i)
    print('Independent dense-channel, NNLS, LP, eigenvalue and SINR checks passed on all fixtures.')
