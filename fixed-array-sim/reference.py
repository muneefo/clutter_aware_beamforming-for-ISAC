"""Fixed-array near-field ISAC. Run python reference.py to regenerate the fixture and results."""
from pathlib import Path
from itertools import combinations
import csv
import json
import numpy as np
from scipy.io import savemat
from scipy.optimize import linprog, nnls
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parent
NT, NR, K, L, B, S = 24, 16, 2, 2, 6, 96
WAVELENGTH, POWER, GAMMA = .01, 1., 2.
USER_NOISE, RX_NOISE, DELTA, BETA = 6e-7, 2e-11, .05, 1.
DEPTH = 3
FRACTIONS = np.array([0., .2, .4, .6, .76, .82, .85, .87, .88, .89, .90, .94])
STEPS = np.array([.015, .045, .12, .32])

def cn(rng, shape):
    return (rng.normal(size=shape) + 1j*rng.normal(size=shape))/np.sqrt(2)

def scene(seed):
    rng = np.random.default_rng(seed)
    # Rows are target, clutter 1, clutter 2, user 1, user 2. Coordinates in metres.
    pos = np.array([[.035,.73],[.085,.70],[-.12,.82],[-.19,.66],[.20,.84]])
    pos += rng.uniform(-.022,.022,pos.shape)
    truth = np.exp(rng.uniform(-.8,1.0,L))
    return pos, truth, cn(rng,(S,B,L)), cn(rng,(S,B))

def steering(pos,n,offset):
    x = (np.arange(n)-(n-1)/2)*WAVELENGTH/2 + offset
    d = np.hypot(x[:,None]-pos[None,:,0],pos[None,:,1])
    return WAVELENGTH/(4*np.pi*d)*np.exp(-2j*np.pi*d/WAVELENGTH)

def unit(v):
    return v/np.linalg.norm(v)

def geometry(pos):
    t,r = steering(pos,NT,-.008),steering(pos,NR,.008)
    return t[:,:3],r[:,:3],t[:,3:].T

def probes(t,r):
    # Each of the three mapped focal directions is paired cyclically with itself
    # and its neighbor. Target position is known; the target is absent in calibration.
    return np.array([[np.vdot(unit(r[:,(b+b//3)%3]), r[:,j]) *
                      (t[:,j] @ unit(t[:,b%3].conj())) for j in (1,2)] for b in range(B)])

def estimate(G, powers, alpha, noise):
    z = np.sum(alpha * (G*np.sqrt(powers))[None,:,:],axis=2) + np.sqrt(RX_NOISE)*noise
    m = np.mean(abs(z)**2,axis=0)/RX_NOISE
    H = abs(G)**2/RX_NOISE
    qhat = nnls(H,m-1)[0]
    a = DELTA/(2*B)
    lo,hi = 2*S*m/chi2.ppf(1-a,2*S),2*S*m/chi2.ppf(a,2*S)
    C = np.vstack((H,-H,-np.eye(L)))
    d = np.r_[hi-1,1-lo,np.zeros(L)]
    vertices = []
    for ids in combinations(range(len(d)),L):
        ix = list(ids)
        if np.linalg.cond(C[ix]) > 1e10: continue
        v = np.linalg.solve(C[ix],d[ix])
        if np.all(C@v<=d+1e-9): vertices.append(v)
    upper = np.maximum(0,np.max(vertices,axis=0)) if vertices else None
    return H,m,qhat,upper,C,d

def spatial(t,r,W,q):
    # Normalize sensing covariance by receiver noise to condition the linear solve.
    illum = np.sum(abs(t.T@W)**2,axis=1)
    cov = np.eye(NR)+(r[:,1:]*(q*illum[1:]/RX_NOISE))@r[:,1:].conj().T
    u = unit(np.linalg.solve(cov,r[:,0]))
    weights = abs(u.conj()@r)**2/RX_NOISE
    den = 1 + np.dot(q*weights[1:],illum[1:])
    val = BETA*weights[0]*illum[0]/den
    return val,u,weights,den

def quality(h,W):
    p = abs(h@W)**2
    return np.diag(p)/(USER_NOISE+p.sum(axis=1)-np.diag(p))

def feasible(h,W):
    return np.linalg.norm(W)**2<=POWER+1e-9 and np.min(quality(h,W))>=GAMMA-1e-9

def basis(t,h):
    zf = h.conj().T @ np.linalg.solve(h@h.conj().T,np.eye(K))
    W0 = zf / np.linalg.norm(zf,axis=0) * np.sqrt(POWER/K)
    v = t[:,0].conj()-zf@(h@t[:,0].conj())
    phases = np.exp(1j*np.angle(t[:,0]@W0))
    V = unit(v)[:,None]*phases*np.sqrt(POWER/K)
    return W0,V

def start(t,h,f):
    W,V = basis(t,h)
    return np.sqrt(1-f)*W+np.sqrt(f)*V

def features(t,h,q):
    W,_=basis(t,h)
    return np.r_[1,np.log1p(q),np.log(quality(h,W)/GAMMA),np.linalg.norm(t[:,0])*1e2]

def step(t,r,h,W,q,eta):
    val,u,weights,den = spatial(t,r,W,q)
    coeff=np.r_[BETA*weights[0],-val*q*weights[1:]]/den
    grad = (t.conj()*coeff) @ (t.T@W)
    if np.linalg.norm(grad)<1e-15:return W
    for j in range(9):
        cand=np.sqrt(POWER)*unit(W+eta*2.**(-j)*unit(grad))
        if feasible(h,cand) and spatial(t,r,cand,q)[0]>=val-1e-12:return cand
    return W

def design(t,r,h,q,theta,schedule):
    f=float(np.clip(features(t,h,q)@theta,0,.94))
    # ZF/null-space mixture has SINR (1-f) times its ZF value.
    f=min(f,max(0.,.995*(1-GAMMA/min(quality(h,start(t,h,0))))))
    W=start(t,h,f)
    if not feasible(h,W):W=start(t,h,0)
    if not feasible(h,W):raise ValueError('Infeasible zero-forcing initialization')
    for eta in schedule:W=step(t,r,h,W,q,float(eta))
    return W

def evaluate(t,r,h,W,qdesign,truth):
    pred,u,_,_=spatial(t,r,W,qdesign)
    illum=np.sum(abs(t.T@W)**2,axis=1)
    w=abs(u.conj()@r)**2/RX_NOISE
    actual=BETA*w[0]*illum[0]/(1+sum(truth*w[1:]*illum[1:]))
    return np.array([pred,actual,*quality(h,W),np.linalg.norm(W)**2])

def train():
    cases=[];X=[];Y=[]
    for seed in range(100,160):
        pos,q,a,n=scene(seed);t,r,h=geometry(pos)
        _,_,_,upper,_,_=estimate(probes(t,r),q,a,n)
        if upper is None or not feasible(h,start(t,h,0)):continue
        scores=[spatial(t,r,start(t,h,f),upper)[0] if feasible(h,start(t,h,f)) else -np.inf for f in FRACTIONS]
        X.append(features(t,h,upper));Y.append(FRACTIONS[np.argmax(scores)])
        cases.append((t,r,h,upper))
    X=np.array(X);theta=np.linalg.solve(X.T@X+.001*np.eye(X.shape[1]),X.T@Y)
    # Train greedy per-layer step sizes on a fixed set of independent calibration scenes.
    ws=[design(t,r,h,q,theta,[]) for t,r,h,q in cases];schedule=[]
    for _ in range(DEPTH):
        candidates=[[step(t,r,h,W,q,eta) for (t,r,h,q),W in zip(cases,ws)] for eta in STEPS]
        scores=[np.mean([np.log(spatial(t,r,W,q)[0]) for (t,r,h,q),W in zip(cases,trial)]) for trial in candidates]
        k=int(np.argmax(scores));schedule.append(STEPS[k]);ws=candidates[k]
    return theta,np.array(schedule),len(cases)

def verify_scene(t,r,h,W,q,C,d,upper,H,mu,hat):
    # Separate full-matrix oracle for the receiver, SCNR and clutter covariance.
    from scipy.linalg import eigvalsh
    A=[np.outer(r[:,j],t[:,j]) for j in range(3)]
    blocks=[a@W@W.conj().T@a.conj().T/RX_NOISE for a in A]
    R=np.eye(NR)+sum(v*b for v,b in zip(q,blocks[1:]))
    val,u,_,_=spatial(t,r,W,q)
    np.testing.assert_allclose(val,eigvalsh(BETA*blocks[0],R)[-1],rtol=2e-9)
    assert feasible(h,W) and abs(np.linalg.norm(u)-1)<1e-12
    np.testing.assert_allclose(nnls(H,H@q)[0],q,rtol=1e-8)
    assert np.all(hat>=0)
    for ell in range(L):
        res=linprog(-np.eye(L)[ell],A_ub=C,b_ub=d,bounds=(0,None),method='highs')
        assert res.success
        np.testing.assert_allclose(-res.fun,upper[ell],rtol=2e-8)
    # Finite difference with the combiner optimized at each perturbation.
    D=unit(np.arange(W.size).reshape(W.shape)+1j)
    _,_,weights,den=spatial(t,r,W,q)
    g=(t.conj()*(np.r_[BETA*weights[0],-val*q*weights[1:]]/den))@(t.T@W)
    eps=1e-6
    fd=(spatial(t,r,W+eps*D,q)[0]-spatial(t,r,W-eps*D,q)[0])/(2*eps)
    np.testing.assert_allclose(fd,2*np.real(np.vdot(g,D)),rtol=2e-5,atol=1e-7)

def run():
    theta,schedule,ntrain=train()
    nulltheta=np.zeros_like(theta)
    results=[];empty=0;inside=0;bound_pass=0;fixtures=[];errors=[]
    for seed in range(1000,1128):
        pos,truth,alpha,noise=scene(seed)
        t,r,h=geometry(pos)
        H,m,hat,upper,C,d=estimate(probes(t,r),truth,alpha,noise)
        assert np.linalg.matrix_rank(H)==L
        if upper is None:
            empty+=1
            continue
        covered=bool(np.all(C@truth<=d+1e-9));inside+=covered
        configs=[('known_strengths',truth,theta,schedule),('plug_in',hat,theta,schedule),
                 ('box_classical',upper,nulltheta,[.12]*DEPTH),
                 ('box_geometry_classical',upper,np.r_[1.,np.zeros(5)],[.12]*DEPTH),('box_init_only',upper,theta,[]),
                 ('box_steps_only',upper,nulltheta,schedule),('box_combined',upper,theta,schedule)]
        scene_metrics=[]
        for label,qdesign,coef,steps in configs:
            W=design(t,r,h,qdesign,coef,steps)
            values=evaluate(t,r,h,W,qdesign,truth)
            assert feasible(h,W)
            scene_metrics.append(values)
            results.append([seed,label,int(covered),*values])
            if label=='box_combined':
                passes=bool(values[0]<=values[1]+1e-9)
                assert not covered or passes
                bound_pass+=passes
                # New operating covariance, compared with true cov at this same W.
                D=r[:,1:]*(np.sum(abs(t[:,1:].T@W)**2,axis=1))
                raw=(D*truth)@r[:,1:].conj().T
                estimated=(D*hat)@r[:,1:].conj().T
                errors.append(np.linalg.norm(estimated-raw)/np.linalg.norm(raw))
                if len(fixtures)<4:
                    verify_scene(t,r,h,W,truth,C,d,upper,H,m,hat)
                    fixtures.append(dict(pos=pos,truth=truth,alpha=alpha,noise=noise,
                                         H=H,m=m,hat=hat,upper=upper,W=W,values=values,
                                         metrics=np.stack(scene_metrics)))
    cols=['seed','method','set_covers_truth','design_scnr','true_scnr','user1_sinr','user2_sinr','power_W']
    with (ROOT/'test_results.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(cols);w.writerows(results)
    assert fixtures and not (empty==128)
    pack={k:np.stack([row[k] for row in fixtures],axis=-1) for k in fixtures[0]}
    savemat(ROOT/'reference_fixture.mat',{**pack,'theta':theta[:,None],'schedule':schedule[:,None]},do_compression=True)
    # Evaluate negative cases independent of a favorable scene.
    t,r,h=geometry(scene(999)[0]);W=start(t,h,0);q=np.array([1.,2.])
    clean=spatial(t,r,W,np.zeros(2))[0];dirty=spatial(t,r,W,q)[0];assert clean>=dirty-1e-12
    G=probes(t,r);assert np.linalg.matrix_rank(np.column_stack([G[:,0],G[:,0]]))==1
    _,_,_,u,_,_=estimate(G,q,np.zeros((S,B,L)),np.zeros((S,B)));assert u is None
    info=dict(train_scenes_used=ntrain,test_attempts=128,empty_sets=empty,sets_containing_truth=inside,
              scnr_bounds_satisfied=bound_pass,feasible_combined=len(errors),heldout_covariance_median_relative_error=float(np.median(errors)),
              learned_steps=schedule.tolist(),initializer_weights=theta.tolist(),
              median_true_scnr_db={k:float(10*np.log10(np.median([row[4] for row in results if row[1]==k]))) for k in dict.fromkeys(row[1] for row in results)})
    (ROOT/'summary.json').write_text(json.dumps(info,indent=2)+'\n')
    print(json.dumps(info,indent=2))
if __name__=='__main__':run()
