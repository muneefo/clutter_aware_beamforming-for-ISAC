"""Generate absolute SCNR curves on the tutorial covariance file, with paired channels."""
import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from beam_solver import load_case, MatlabTwister, fixed_receiver, solve, true_score, audit

def setup(repo):
    global case, estimated, target_per_watt, radar_cache, limits
    limits=threadpool_limits(limits=1)
    case=load_case(repo)
    estimated=case.est
    target_per_watt=case.sig / case.P
    radar_cache={}

def channels(x,seed,study):
    low,high,exponent=(100,500,3.1) if study=='power' else (50,400,2.7)
    rng=MatlabTwister(seed)
    h=rng.complex((x.T,x.N,x.K)).transpose(1,0,2)
    distances=np.array([low+int((high-low+1)*rng.uniform()) for _ in range(x.K)])
    return h*10**((-30-10*exponent*np.log10(distances))/20)

def conventional(x, method, kernel):
    # Equal per-stream and per-tone power, as in baseline_comm_heur.m.
    W=np.zeros((x.N,x.T,x.S),complex)
    W[:,:,:x.K]=x.H/np.linalg.norm(x.H,axis=1,keepdims=True)
    if method=='heuristic':
        W[:,:,x.K:]=x.a[:,:,None]
    W*=np.sqrt(x.P/x.N)/np.linalg.norm(W,axis=(1,2))[:,None,None]
    Q=W@W.conj().transpose(0,2,1)
    RI=kernel.apply(Q)+x.Reta
    ridge=1e-10*np.trace(RI,axis1=1,axis2=2).real/x.R
    RI=RI+ridge[:,None,None]*np.eye(x.R)
    u=np.linalg.solve(RI,x.b[:,:,None])[:,:,0]
    return Q,u/np.einsum('ni,ni->n',x.b.conj(),u)[:,None]

def radar(x,kernel):
    # Exact fixed-U transmit update for the total-power-only radar formulation.
    # For one tone the quotient uses C + (hot/P) I, and A is rank one.
    Q=x.P/x.N*x.a[:,:,None]*x.a[:,None,:].conj()
    for it in range(30):
        old,A,C,hot,u=fixed_receiver(x,Q,kernel)
        pencil=C+(hot/x.P)*np.eye(x.T)
        v=np.linalg.solve(pencil,x.a[:,:,None])[:,:,0]
        v=v/np.linalg.norm(v,axis=1,keepdims=True)
        num=np.einsum('ni,nij,nj->n',v.conj(),A,v).real
        den=np.einsum('ni,nij,nj->n',v.conj(),pencil,v).real
        n=np.argmax(num/den)
        Q=np.zeros_like(Q)
        Q[n]=x.P*np.outer(v[n],v[n].conj())
        new,_,_,_,u=fixed_receiver(x,Q,kernel)
        gain=10*np.log10(new/old)
        assert gain>=-1e-6
        if gain<.01:break
    return Q,u

def evaluate(task):
    study,dbm,K,gamma,seeds=task
    x=case; x.K=K; x.P=10**((dbm-30)/10); x.sig=target_per_watt*x.P
    x.g=np.full((x.N,K),10**(gamma/10))
    rows=[]; trace=[]
    for seed in seeds:
        x.H=channels(x,seed,study)
        kernels=[('estimated',estimated)] if study=='trace' else [('perfect',x.truth),('estimated',estimated)]
        for label,kernel in kernels:
            x.est=kernel
            methods=['joint']
            if study=='power':
                methods+=['radar','heuristic']
                if label=='perfect':methods+=['comm']
            for method in methods:
                item=dict(study=study,dbm=dbm,K=K,gamma=gamma,seed=seed,method=method,kernel=label,
                          trueDB=np.nan,designDB=np.nan,marginDB=np.nan,powerW=np.nan,status='ok')
                try:
                    if method=='joint':
                        def observe(it,steps,W,U,eta):
                            Q=W@W.conj().transpose(0,2,1)
                            margin,power=audit(x,W)
                            trace.append(dict(seed=seed,iteration=it,lpSolves=steps,
                                designDB=10*np.log10(eta),trueDB=true_score(x,Q,U,x.truth),marginDB=margin,powerW=power))
                        W,U,_=solve(x,5 if study=='trace' else 3,3 if study=='trace' else 2,
                                    observe=observe if study=='trace' else None)
                        Q=W@W.conj().transpose(0,2,1)
                        item['marginDB'],item['powerW']=audit(x,W)
                        if item['marginDB'] < -1e-4 or item['powerW'] > x.P*(1+1e-7):
                            item['status']='infeasible'
                    elif method=='radar':
                        cache_key=(dbm,label)
                        if cache_key not in radar_cache:
                            radar_cache[cache_key]=radar(x,kernel)
                        Q,U=radar_cache[cache_key]
                    else:
                        Q,U=conventional(x,method,kernel)
                    item['designDB']=true_score(x,Q,U,kernel)
                    item['trueDB']=true_score(x,Q,U,x.truth)
                    if not np.isfinite([item['designDB'],item['trueDB']]).all():
                        raise ValueError('Nonfinite score')
                except (ValueError,RuntimeError,np.linalg.LinAlgError) as e:
                    item['status']=str(e)
                rows.append(item)
    return rows,trace

def write(file,rows):
    with file.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('repo',type=Path);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--seeds',type=int,default=8);p.add_argument('--out',type=Path,default=Path(__file__).parent)
    args=p.parse_args()
    if args.seeds<1: p.error('--seeds must be positive')
    args.out.mkdir(parents=True,exist_ok=True)
    seeds=range(11001,11001+args.seeds)
    jobs=[('power',power,6,10,seeds) for power in range(30,51,2)]
    jobs += [('sinr',50,k,g,seeds) for g in range(0,21,4) for k in (4,8,12)]
    jobs += [('trace',50,4,8,range(a,a+4)) for a in range(10001,10025,4)]
    rows=[]; traces=[]
    with ProcessPoolExecutor(max_workers=args.workers,initializer=setup,initargs=(args.repo,)) as pool:
        for job,(r,t) in zip(jobs,pool.map(evaluate,jobs)):
            rows.extend(r); traces.extend(t)
            print(job[:4],sum(z['status']!='ok' for z in r),'unsuccessful',flush=True)
    write(args.out/'sweep_trials.csv',rows);write(args.out/'outer_trace.csv',traces)
