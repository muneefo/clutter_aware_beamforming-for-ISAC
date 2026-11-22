"""Train, validate and test the separate unfolded power-allocation experiment."""
import os
for var in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[var]="1"
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from threadpoolctl import threadpool_limits
import allocator as alg
import reference_beams as ref

DEPTH=8
METHODS=("exact_LP","fixed_PD","learned_PD")


def write_csv(path, rows):
    with path.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)


def set_scene(x, users, threshold, seed):
    x.K=users; x.g=np.full((x.N,users),10**(threshold/10))
    return ref.scene(x,seed)


def prepare_set(x,seeds):
    data=[]
    for k in (4,8,12):
        for gamma in (4,16):
            for seed in seeds:
                set_scene(x,k,gamma,seed)
                problem,_=alg.build(x,ref.seed_bank(x))
                data.append(problem)
    return data


def utility(problems,schedule):
    values=[]
    for prob in problems:
        p,_=alg.unroll(prob,schedule)
        values.append(10*np.log10(prob.scale*prob.ratio(p)))
    return np.mean(values)


def fit(train,valid):
    history=[]
    def trial(stage,schedule):
        score=utility(train,schedule)
        history.append(dict(stage=stage,trainDB=score,schedule=schedule.tolist()))
        return score
    best=-np.inf
    for a in (.25,1,4,16,64,256):
        for b in (.03125,.25,2,16,128):
            schedule=np.tile(np.array([a,b],dtype=float),(DEPTH,1))
            score=trial("constant grid",schedule)
            if score>best: best=score; fixed=schedule
    # Refine the tied control on the same training scenes.
    for factor in (2.0,1.4):
        for j in range(2):
            start=fixed.copy()
            for ratio in (1/factor,factor):
                s=start.copy(); s[:,j]*=ratio
                v=trial("constant refinement",s)
                if v>best: fixed=s; best=v
    # Four pairs of adjacent layers share parameters (8 positive learned scalars).
    # Coordinate search trains on the deployed, post-repair objective with no AO labels.
    schedule=fixed.copy(); chosen=schedule.copy()
    valid_history=[utility(valid,schedule)]
    start_train=best
    for factor in (4.,2.,1.4):
        for stage in range(4):
            for j in range(2):
                start=schedule.copy()
                for ratio in (1/factor,factor):
                    s=start.copy(); s[2*stage:2*stage+2,j]*=ratio
                    v=trial("untied refinement",s)
                    if v>best: schedule=s; best=v
        val=utility(valid,schedule)
        if val>max(valid_history)+1e-12: chosen=schedule.copy()
        valid_history.append(val)
    info=dict(depth=DEPTH,trainFixedDB=start_train,trainSelectedLearnedDB=utility(train,chosen),
              validationByPassDB=valid_history,fixed=fixed.tolist(),learned=chosen.tolist(),
              trainConditions=[4,16],users=[4,8,12],trainSeeds=list(range(31001,31009)),
              validationSeeds=list(range(32001,32005)),testSeeds=list(range(33001,33013)))
    return fixed,chosen,info,history


def validation_fixture(x):
    init=ref.seed_bank(x)
    prob,U=alg.build(x,init)
    H,d,V,owners,W=init
    _,A,C,hot,_=ref.fixed_receiver(x,W@W.conj().transpose(0,2,1),x.est)
    powers,*_=ref.master(x,H,d,V,owners,A,C,hot)
    p,y=alg.exact(prob)
    old_ratio=prob.ratio(np.array(powers)/(x.P*(1-1e-8)))
    assert abs(10*np.log10(prob.ratio(p)/old_ratio))<1e-5
    assert alg.gap_bound(prob,p,y)<2e-5
    # Check the fixed-receiver objective against the physical scorer too.
    beams=ref.recover(x,H,V,owners,p*x.P*(1-1e-8))
    q=beams@beams.conj().transpose(0,2,1)
    assert abs(ref.true_score(x,q,U,x.est)-10*np.log10(prob.scale*prob.ratio(p)))<.002


def block_test(x,fixed,learned,rng):
    rows=[]
    schedules={"fixed_PD":fixed,"learned_PD":learned}
    for k in (4,8,12):
        for gamma in range(0,21,4):
            for seed in range(33001,33013):
                set_scene(x,k,gamma,seed)
                init=ref.seed_bank(x)
                prob,U=alg.build(x,init)
                H,_,V,owner,_=init
                times=defaultdict(list); full=defaultdict(list); fallbacktimes=defaultdict(list); out={}
                for rep in range(3):
                    for name in rng.permutation(METHODS):
                        t=perf_counter()
                        if name=="exact_LP": p,y=alg.exact(prob)
                        else: p,y=alg.unroll(prob,schedules[name])
                        elapsed=perf_counter()-t; times[name].append(1000*elapsed)
                        W=ref.recover(x,H,V,owner,p*x.P*(1-1e-8))
                        m,powW=ref.audit(x,W)
                        feasible=(m>=-1e-4 and powW<=x.P*(1+1e-7))
                        full[name].append(1000*(perf_counter()-t))
                        bound=alg.gap_bound(prob,p,y)
                        # Through final audited output; trigger only on observable certificate/constraints.
                        did_fallback=name!="exact_LP" and (not feasible or bound>.1)
                        if did_fallback:
                            fb,_=alg.exact(prob)
                            fbW=ref.recover(x,H,V,owner,fb*x.P*(1-1e-8))
                            fm,fp=ref.audit(x,fbW)
                            assert fm>=-1e-4 and fp<=x.P*(1+1e-7)
                        fallbacktimes[name].append(1000*(perf_counter()-t))
                        if rep==0:
                            Q=W@W.conj().transpose(0,2,1)
                            out[name]=dict(K=k,gammaDB=gamma,seed=seed,method=name,
                                designDB=ref.true_score(x,Q,U,x.est),
                                trueDB=ref.true_score(x,Q,U,x.truth),
                                bankDB=10*np.log10(prob.scale*prob.ratio(p)),
                                boundDB=bound, powerW=powW,marginDB=m,feasible=int(feasible),
                                fallback=int(did_fallback))
                for name,r in out.items():
                    gap=out["exact_LP"]["bankDB"]-r["bankDB"]
                    assert -.0001<=gap<=r['boundDB']+1e-5
                    r.update(bankGapDB=gap,trueGapDB=out["exact_LP"]["trueDB"]-r["trueDB"],
                             qualityHit=int(r["feasible"] and gap<=.1),
                             solverMS=np.median(times[name]),returnedMS=np.median(full[name]),
                             certifiedMS=np.median(fallbacktimes[name]))
                    rows.append(r)
            print(f"Block K={k}, SINR={gamma} dB complete",flush=True)
    return rows


def first_prices(x,init,problem,A,C,hot,p,y):
    H,_,_,_,_=init; safe=x.P*(1-1e-8)
    # Undo row scaling for the legacy eigenvalue-pricing routine.
    V=np.asarray(init[2]); ids=np.array(init[3])
    gains=abs(H.conj().transpose(0,2,1)@V)**2
    g=x.g*(1+2e-5)
    coeff=(1+1/g[:,:,None])*(np.arange(x.K)[None,:,None]==ids[:,None,:])-1
    mu=y/np.max(abs(coeff*gains),axis=-1)
    eta=problem.ratio(p)
    price=max(0.,np.max(problem.a-eta*problem.c+problem.transmit(y)))
    base=A*(safe/hot/problem.scale)-eta*C*(safe/hot)-(H*mu[:,None,:])@H.conj().transpose(0,2,1)
    return base,mu,price,eta,g


def hybrid(x,name,schedules):
    """Three outer updates, two allocations each. Replace only the first exact LP."""
    init=ref.seed_bank(x); H,d,V,owners,W=init
    Q=W@W.conj().transpose(0,2,1)
    eta,A,C,hot,U=ref.fixed_receiver(x,Q,x.est)
    for j in range(3):
        if j==0:
            problem,U=alg.build(x,init,(eta,A,C,hot,U))  # also appends the common fixed proposals
        else:
            alg.add_fixed_proposals(x,W,A,C,eta,V,owners)
        best=eta
        for i in range(2):
            if (j,i)==(0,0):
                if name=="exact_LP": p,y=alg.exact(problem)
                else: p,y=alg.unroll(problem,schedules[name])
                powers=p*x.P*(1-1e-8)
                base,mu,price,theta,g=first_prices(x,init,problem,A,C,hot,p,y)
            else:
                powers,base,mu,price,theta,g=ref.master(x,H,d,V,owners,A,C,hot)
            Y=ref.recover(x,H,V,owners,powers)
            margin,power=ref.audit(x,Y)
            E=Y@Y.conj().transpose(0,2,1)
            score=(A*E.swapaxes(1,2)).sum().real/((C*E.swapaxes(1,2)).sum().real+hot)
            if margin>=-1e-4 and power<=x.P*(1+1e-7) and score>=best*(1-1e-10):
                Q,W,best=E,Y,score
            # Approximate multipliers suggest directions; no approximate stopping certificate.
            new,tags,_=ref.price_beams(x,H,mu,base,price,g)
            for n in range(x.N):
                V[n]=np.column_stack((V[n],new[n]))
                owners[n]=np.r_[owners[n],tags[n]]
        eta,A,C,hot,U=ref.fixed_receiver(x,Q,x.est)
    return W,U


def hybrid_test(x,fixed,learned,rng):
    schedules={"fixed_PD":fixed,"learned_PD":learned}; rows=[]
    for k in (4,8,12):
        for seed in range(33001,33009):
            set_scene(x,k,8,seed); times=defaultdict(list); scene={}
            for rep in range(2):
                for name in rng.permutation(METHODS):
                    start=perf_counter(); W,U=hybrid(x,name,schedules)
                    margin,power=ref.audit(x,W); times[name].append(1000*(perf_counter()-start))
                    if rep==0:
                        Q=W@W.conj().transpose(0,2,1)
                        scene[name]=dict(K=k,gammaDB=8,seed=seed,method=name,powerW=power,marginDB=margin,
                            feasible=int(margin>=-1e-4 and power<=x.P*(1+1e-7)),
                            designDB=ref.true_score(x,Q,U,x.est),trueDB=ref.true_score(x,Q,U,x.truth))
            for name,r in scene.items():
                r.update(trueGapDB=scene["exact_LP"]["trueDB"]-r["trueDB"],ms=np.median(times[name]))
                rows.append(r)
        print(f"Full solver K={k} complete",flush=True)
    return rows


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("repo",type=Path,help="Tutorial root containing data/covMat_BLP.mat")
    parser.add_argument("--out",type=Path,default=Path(__file__).parent/"results")
    args=parser.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(892); x=ref.load_case(args.repo)
    with threadpool_limits(1):
        for k in (4,8,12):
            validation_fixture(set_scene(x,k,8,30000))
        print("Original-LP and recovered-beam fixture checks passed",flush=True)
        train=prepare_set(x,range(31001,31009)); valid=prepare_set(x,range(32001,32005))
        tick=perf_counter(); fixed,learned,info,history=fit(train,valid)
        info['fitSeconds']=perf_counter()-tick
        (args.out/'model.json').write_text(json.dumps(info,indent=2))
        (args.out/'training.json').write_text(json.dumps(history))
        print("Validation (fixed, successive untying passes):",info['validationByPassDB'],flush=True)
        write_csv(args.out/'block_trials.csv',block_test(x,fixed,learned,rng))
        write_csv(args.out/'hybrid_trials.csv',hybrid_test(x,fixed,learned,rng))
