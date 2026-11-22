"""Joint confidence set solver for the two-reflector fixed-array model."""
from itertools import combinations
from pathlib import Path
import json
import csv
import numpy as np
from scipy.io import savemat
from scipy.optimize import brentq, linprog
import reference as m

ROOT = Path(__file__).resolve().parent

def polygon(C, b):
    vertices = []
    for i, j in combinations(range(len(b)), 2):
        A = C[[i,j]]
        if np.linalg.cond(A) > 1e10:
            continue
        v = np.linalg.solve(A, b[[i,j]])
        if np.all(C@v <= b+1e-9):
            vertices.append(v)
    if not vertices:
        return np.empty((0,2))
    vertices = np.unique(np.array(vertices),axis=0)
    center = vertices.mean(axis=0)
    return vertices[np.argsort(np.arctan2(*(vertices-center).T[::-1]))]

def receiver(t,r,W,V):
    # Fixed-W dual: minimize b0^H R(q)^-1 b0 over the joint polygon.
    illum = np.sum(abs(t.T@W)**2,axis=1)
    def at(q):
        cov = np.eye(len(r))+(r[:,1:]*(q*illum[1:]/m.RX_NOISE))@r[:,1:].conj().T
        v = np.linalg.solve(cov,r[:,0])
        return v, abs(v.conj()@r[:,1:])**2*illum[1:]/m.RX_NOISE
    def certified(q):
        v,d = at(q)
        return np.max((V-q)@d) <= 1e-11*max(np.linalg.norm(v)**2,abs(q@d),1e-30)
    q = next((z for z in V if certified(z)), None)
    if q is None:
        for a,b in zip(V,np.roll(V,-1,axis=0)):
            edge=b-a
            def slope(s):
                return float(edge@at(a+s*edge)[1])
            if slope(0)>0 and slope(1)<0:
                candidate=a+brentq(slope,0,1,xtol=1e-14)*edge
                if certified(candidate):
                    q=candidate
                    break
    if q is None:
        raise ArithmeticError('No certified receiver saddle point')
    u=m.unit(at(q)[0])
    weights=abs(u.conj()@r)**2/m.RX_NOISE
    d=weights[1:]*illum[1:]
    den=1+np.max(V@d)
    val=m.BETA*weights[0]*illum[0]/den
    grad=(t.conj()*(np.r_[m.BETA*weights[0],-val*q*weights[1:]]/den))@(t.T@W)
    return val,u,grad

def project(raw,h):
    # Dykstra's nearest-point projection onto the power ball and user SOCs.
    # Each user affects only its normalized channel row in beam space.
    norms=np.linalg.norm(h,axis=1)
    channels=h/norms[:,None]
    noise=np.sqrt(m.USER_NOISE)/norms
    root=np.sqrt(m.GAMMA)
    def one_user(X,k):
        a=channels[k]
        y=a@X
        others=np.arange(len(y))!=k
        s=y[k].real
        mag=np.linalg.norm(y[others])
        target=root*np.hypot(mag,noise[k])
        desired=y.copy()
        if s>=target:
            desired[k]=s
        else:
            # Scaling of interfering streams from the SOC projection KKT equations.
            def residual(kappa):
                radius=np.hypot(kappa*mag,noise[k])
                return kappa*root*(root*radius-s)-(1-kappa)*radius
            kappa=brentq(residual,0.,1.,xtol=1e-14)
            desired[others]*=kappa
            desired[k]=root*np.hypot(kappa*mag,noise[k])
        return X+np.outer(a.conj(),desired-y)
    x=raw.copy()
    corrections=[np.zeros_like(x) for _ in range(len(h)+1)]
    for _ in range(5000):
        max_move=0.
        for i,p in enumerate(corrections):
            y=x+p
            new=y/max(1.,np.linalg.norm(y)/np.sqrt(m.POWER)) if i==len(h) else one_user(y,i)
            corrections[i]=y-new
            max_move=max(max_move,np.linalg.norm(new-x))
            x=new
        if max_move<2e-11 and m.feasible(h,x):
            return x
    raise ArithmeticError('Projection did not converge')

def phase(W,h):
    return W*np.exp(-1j*np.angle(np.diag(h@W)))

def initial(case,theta):
    t,r,h,V,hat=case
    f=np.clip(m.features(t,h,hat)@theta,0,.94)
    return project(phase(m.start(t,h,f),h),h)

def update(case,W,eta):
    t,r,h,V,_=case
    val,_,grad=receiver(t,r,W,V)
    if np.linalg.norm(grad)<1e-15:
        return W
    for j in range(9):
        cand=project(W+eta*2**(-j)*m.unit(grad),h)
        if receiver(t,r,cand,V)[0]>=val-1e-12:
            return cand
    return W

def score(case,W):
    t,r,_,V,_=case
    return receiver(t,r,W,V)[0]

def sample(seed):
    pos,truth,a,n=m.scene(seed)
    t,r,h=m.geometry(pos)
    _,_,hat,_,C,b=m.estimate(m.probes(t,r),truth,a,n)
    return (t,r,h,polygon(C,b),hat),truth,C,b

def fit():
    X,Y,cases=[],[],[]
    for seed in range(100,160):
        case,_,_,_=sample(seed)
        t,r,h,V,hat=case
        if not len(V):
            continue
        values=[score(case,m.start(t,h,f)) if m.feasible(h,m.start(t,h,f)) else -np.inf for f in m.FRACTIONS]
        if not np.isfinite(max(values)):
            continue
        X.append(m.features(t,h,hat))
        Y.append(m.FRACTIONS[np.argmax(values)])
        cases.append(case)
    X=np.array(X)
    theta=np.linalg.solve(X.T@X+.001*np.eye(6),X.T@Y)
    beams=[initial(case,theta) for case in cases]
    schedule=[]
    for depth in range(3):
        trials=[[update(case,W,eta) for case,W in zip(cases,beams)] for eta in m.STEPS]
        means=[np.mean([np.log(score(case,W)) for case,W in zip(cases,trial)]) for trial in trials]
        k=int(np.argmax(means))
        schedule.append(float(m.STEPS[k]))
        beams=trials[k]
    return theta,schedule,len(cases)

def run():
    theta,schedule,n=fit()
    model=dict(theta=theta.tolist(),steps=schedule,training_cases=n)
    (ROOT/'joint_model.json').write_text(json.dumps(model,indent=2))
    rows=[]
    fixtures=[]
    for seed in range(1000,1128):
        case,truth,C,b=sample(seed)
        t,r,h,V,hat=case
        if not len(V):
            rows.append([seed,'empty_set','','','','',''])
            continue
        covered=np.all(C@truth<=b+1e-9)
        for name,coef,steps in [('geometry_fixed',np.r_[1.,np.zeros(5)],[.12]*3),('geometry_matched',np.r_[1.,np.zeros(5)],schedule),('learned_init_fixed',theta,[.12]*3),('combined',theta,schedule)]:
            W=initial(case,coef)
            for eta in steps:
                W=update(case,W,eta)
            val,u,_=receiver(t,r,W,V)
            g=np.sum(abs(t.T@W)**2,axis=1)*abs(u.conj()@r)**2/m.RX_NOISE
            worst=linprog(-g[1:],A_ub=C,b_ub=b,bounds=(0,None))
            assert worst.success
            np.testing.assert_allclose(val,m.BETA*g[0]/(1-worst.fun),rtol=1e-7)
            true=m.BETA*g[0]/(1+truth@g[1:])
            assert m.feasible(h,W) and (not covered or true+1e-8>=val)
            rows.append([seed,name,int(covered),val,true,np.min(m.quality(h,W)),np.linalg.norm(W)**2])
            if name=='combined' and len(fixtures)<4:
                fixtures.append((t,r,h,C,b,hat,W,val))
    with (ROOT/'heldout.csv').open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['seed','method','covered','bound_linear','true_linear','min_sinr','power_W'])
        writer.writerows(rows)
    savemat(ROOT/'joint_fixture.mat',{key:np.stack(items,axis=-1) for key,items in zip(['t','r','h','C','b','hat','W','bound'],zip(*fixtures))})
    print(model,flush=True)

if __name__=='__main__':
    run()
