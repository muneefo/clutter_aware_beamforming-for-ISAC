from pathlib import Path
root=Path(__file__).parent
rng=(root/'matlab_rng.py').read_text().split("if __name__==")[0]
bench=(root/'bench.py').read_text()
actual=(root/'actual.py').read_text()
replay=(root/'replay.py').read_text()
s='''"""Reproduce the fixed, model-based solver and MATLAB baseline replay.
Usage: python verify_fast_beams.py /path/to/Clutter-Aware-ISAC-Tutorial
Dependencies: numpy, scipy, threadpoolctl. Writes fast_results.csv in the working directory.
"""
import argparse
import csv
from pathlib import Path
from types import SimpleNamespace
from time import perf_counter
from threadpoolctl import threadpool_limits
from scipy.io import loadmat
from scipy.linalg import eigh
from scipy.optimize import linprog
from scipy import sparse
'''
s+=rng[rng.index('import numpy'):]+'\n'
s+=bench[bench.index('def herm('):bench.index('def steer(')]
s+=actual[actual.index('class Kernels'):actual.index("if __name__==")]
s+=bench[bench.index('def recover('):bench.index("if __name__==")].replace('receiver(x,Q,x.est)','fixed_receiver(x,Q,x.est)').replace('if np.linalg.norm(z)==0: raise ValueError(\'empty communication beam\')','if np.linalg.norm(z)==0: continue')
s=s.replace('old=eta\n','old=eta; best_fixed=eta\n').replace('if s>=old*(1-1e-10):\n                Q=E; W=Y','if s>=best_fixed*(1-1e-10):\n                Q=E; W=Y; best_fixed=s').replace("if margin < -1e-4 or p>x.P*(1+1e-7): raise ValueError((margin,p))","feasible = margin >= -1e-4 and p <= x.P*(1+1e-7)").replace("if s>=best_fixed*(1-1e-10):","if feasible and s>=best_fixed*(1-1e-10):")
s+=replay[replay.index('def align('):replay.index("if __name__==")]
s+='''
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('repo',type=Path)
    args=parser.parse_args()
    x=load_case(args.repo)
    ao=[13.540,13.521,13.506,13.485]  # rounded AO values supplied by the user
    anchor=[4.9818,1.3637,3.5008,2.7467]
    scene(x,310)
    solve(x,outer=1,inner=1)  # untimed, separate scene
    records=[]
    for i,seed in enumerate(range(111,115)):
        anchorDB=replay(x,seed)
        assert abs(anchorDB-anchor[i])<0.0001, (seed,anchorDB)
        durations=[]
        for _ in range(5):
            t=perf_counter()
            W,U,history=solve(x,outer=5,inner=3)
            durations.append(perf_counter()-t)
        Q=W@W.conj().transpose(0,2,1)
        db=true_score(x,Q,U,x.truth)
        design=true_score(x,Q,U,x.est)
        margin,power=audit(x,W)
        assert margin>=-1e-4 and power<=x.P*(1+1e-7)
        row=dict(scene=seed,AO_dB=ao[i],fast_dB=db,AO_minus_fast_dB=ao[i]-db,
                 median_seconds=np.median(durations),worst_SINR_margin_dB=margin,power_W=power,
                 estimated_SCNR_dB=design,LP_solves=len(history),anchor_replay_dB=anchorDB)
        records.append(row)
        print(seed, 'SCNR %.6f, gap %+.6f, time %.4f, margin %.8f, power %.8f' %
              (db,ao[i]-db,row['median_seconds'],margin,power),flush=True)
    with open('fast_results.csv','w',newline='') as out:
        w=csv.DictWriter(out,fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)
if __name__=='__main__':
    with threadpool_limits(limits=1):
        main()
'''
(root/'verify_fast_beams.py').write_text(s)
