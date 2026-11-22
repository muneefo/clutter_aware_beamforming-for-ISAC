import numpy as np
from bench import audit
from actual import load_case,scene,fixed_receiver,true_score,MatlabTwister
from pathlib import Path

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
if __name__=='__main__':
    x=load_case(Path(__file__).parent/'authors')
    for s in range(111,115):print(s, replay(x,s))
