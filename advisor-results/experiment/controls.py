"""Additional receiver-only and thermal-noise-only controls on the same channels."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy.io import loadmat
from threadpoolctl import threadpool_limits
from clutter_test import cov, ZeroMap
from experiment import hybrid, set_scene
import reference_beams as ref

def controls(repo):
    x=ref.load_case(repo)
    data=loadmat(repo/'data/covMat_BLP.mat',variable_names=['para','theta_cc_all','sigma2_cc_all'],simplify_cells=True)
    thermal=np.eye(x.R)[None,:,:]*np.ones((x.N,1,1))*data['para']['sigma2']
    angles=np.asarray(data['theta_cc_all']).ravel()
    strengths=np.asarray(data['sigma2_cc_all']).mean(axis=1)
    pd.DataFrame(dict(angle=angles,meanStrength=strengths)).to_csv('scatterers.csv',index=False)
    additive=x.Reta; estimated=x.est; zero=ZeroMap(); rows=[]
    for k in (4,8):
        for gamma in (0,8,16):
            for seed in range(47001,47013):
                set_scene(x,k,gamma,seed)
                for method in ('RX only','Thermal only'):
                    x.est=zero; x.Reta=additive if method=='RX only' else thermal
                    w,u=hybrid(x,'exact_LP',{})
                    q=cov(w)
                    if method=='RX only':
                        u=ref.fixed_receiver(x,q,estimated)[-1]
                    kernel=x.truth if method=='RX only' else zero
                    score=ref.true_score(x,q,u,kernel)
                    margin,power=ref.audit(x,w)
                    assert margin>=-1e-4 and power<=x.P*(1+1e-7)
                    rows.append(dict(K=k,gamma=gamma,seed=seed,method=method,scnr=score,
                                     feasible=True,powerW=power,marginDB=margin))
            print('Controls complete',k,gamma,flush=True)
    pd.DataFrame(rows).to_csv('controls.csv',index=False)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('repo',type=Path)
    with threadpool_limits(1):
        controls(parser.parse_args().repo)
