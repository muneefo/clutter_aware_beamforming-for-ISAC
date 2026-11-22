"""Paired budget sensitivity on fresh channels with one fixed tutorial clutter map."""
from pathlib import Path
import argparse, csv
from time import perf_counter
import numpy as np
from threadpoolctl import threadpool_limits
from verify_fast_beams import load_case, scene, solve, true_score, audit

def run(repo, destination):
    x = load_case(repo)
    configurations = [(3,2),(5,3)]
    scene(x,8700)
    for outer,inner in configurations:
        solve(x,outer=outer,inner=inner)
    fields = ['seed','budget','lpSolves','seconds','trueDB','designDB','marginDB','powerW','feasible']
    with destination.open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for seed in range(9001,9033):
            scene(x,seed)
            order = configurations if seed%2 else configurations[::-1]
            for outer,inner in order:
                start = perf_counter()
                W,U,history = solve(x,outer=outer,inner=inner)
                seconds = perf_counter()-start
                Q = W@W.conj().transpose(0,2,1)
                margin, power = audit(x,W)
                writer.writerow(dict(seed=seed,budget=outer*inner,lpSolves=len(history),seconds=seconds,
                    trueDB=true_score(x,Q,U,x.truth), designDB=true_score(x,Q,U,x.est),
                    marginDB=margin,powerW=power,feasible=int(margin>=-1e-4 and power<=x.P*(1+1e-7))))
            stream.flush()
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('repo',type=Path)
    parser.add_argument('--csv',type=Path,default=Path('heldout_budget.csv'))
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.repo,args.csv)
