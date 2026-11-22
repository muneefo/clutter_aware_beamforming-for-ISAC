"""Evaluate frozen prefixes on fresh seeds; no retraining for shorter depths."""
import argparse,json
from pathlib import Path
import pandas as pd
import numpy as np
from threadpoolctl import threadpool_limits
from experiment import set_scene
import reference_beams as ref
import allocator as alg

def run(repo):
    x=ref.load_case(repo); model=json.loads(Path('model.json').read_text()); rows=[]
    for k in (4,8):
        for gamma in (0,8,16):
            for seed in range(47001,47013):
                set_scene(x,k,gamma,seed)
                init=ref.seed_bank(x)
                problem,u=alg.build(x,init); h,_,dirs,tags,_=init
                opt,_=alg.exact(problem); best=problem.ratio(opt)
                for method in ('fixed','learned'):
                    for layers in (2,4,6,8):
                        p,_=alg.unroll(problem,model[method][:layers])
                        w=ref.recover(x,h,dirs,tags,p*(1-1e-8)*x.P)
                        m,pw=ref.audit(x,w); gap=10*np.log10(best/problem.ratio(p))
                        assert gap>=-1e-6 and m>=-1e-4 and pw<=x.P*(1+1e-7)
                        rows.append(dict(K=k,gamma=gamma,seed=seed,method=method,layers=layers,
                                         bankGapDB=gap,marginDB=m,powerW=pw))
    pd.DataFrame(rows).to_csv('depth.csv',index=False)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('repo',type=Path)
    with threadpool_limits(1):
        run(parser.parse_args().repo)
