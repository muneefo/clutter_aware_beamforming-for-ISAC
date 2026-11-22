"""Cold-clutter strength calibration at known bearings; frozen 6-allocation solvers."""
from pathlib import Path
from argparse import ArgumentParser
import csv,json
import numpy as np
from scipy.io import loadmat
from threadpoolctl import threadpool_limits
from experiment import hybrid,set_scene
import reference_beams as ref

class AngleMap:
    # Equal transmit/receive apertures and steering make this operator self-adjoint.
    def __init__(self, steering, powers):
        self.a=steering
        self.powers=powers
    def apply(self,q,back=False):
        power=(self.a.conj()*(q@self.a)).sum(1).real*self.powers
        return (self.a*power[:,None,:])@self.a.conj().transpose(0,2,1)

def steering(x,angles):
    f=1+np.arange(x.N)*120e3/28e9
    return np.exp(-1j*np.pi*f[:,None,None]*np.arange(x.T)[None,:,None]*
                  np.sin(np.deg2rad(angles)))/np.sqrt(x.T)

def run(repo):
    args=loadmat(repo/'data/covMat_BLP.mat',simplify_cells=True,variable_names=['theta_cc_all','sigma2_cc_all'])
    v=args['sigma2_cc_all'].T
    x=ref.load_case(repo)
    # Verify the factorization on the authors' original ideal kernel.
    original=AngleMap(steering(x,args['theta_cc_all']),v)
    q=x.a[:,:,None]*x.a[:,None,:].conj()
    assert np.linalg.norm(original.apply(q)-x.truth.apply(q))/np.linalg.norm(x.truth.apply(q))<1e-9
    b=Path(__file__).parent
    model=json.loads((b/'model.json').read_text())
    schedules={'learned_PD':model['learned'],'fixed_PD':model['fixed']}
    with (b/'measurements.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['environment','channelSeed','snapshots','method','scnrDB','minMarginDB','powerW','feasible'])
        for env in range(16):
            rng=np.random.default_rng(61000+env)
            angles=rng.uniform(-80,80,24)
            fractions=rng.dirichlet(np.ones(24))
            powers=v.sum(1)[:,None]*fractions
            a=steering(x,angles);x.truth=AngleMap(a,powers)
            # A resolved zero-mean circular Gaussian echo has exponential intensity.
            samples=rng.exponential(size=(64,len(angles)))
            zero=AngleMap(a,powers*0)
            for channel in range(2):
                seed=62000+2*env+channel
                set_scene(x,4,8,seed)
                x.est=zero; blind,u0=hybrid(x,'exact_LP',{})
                def record(w,u,budget,name):
                    q=w@w.conj().transpose(0,2,1)
                    margin,power=ref.audit(x,w)
                    writer.writerow([env,seed,budget,name,ref.true_score(x,q,u,x.truth),margin,power,int(margin>=-1e-4 and power<=x.P*(1+1e-7))])
                record(blind,u0,'0','blind')
                for budget in (1,4,16,64,'known'):
                    x.est=x.truth if budget=='known' else AngleMap(a,powers*samples[:budget].mean(0))
                    ub=ref.fixed_receiver(x,blind@blind.conj().transpose(0,2,1),x.est)[-1]
                    record(blind,ub,budget,'rxOnly')
                    for method in (('exact_LP','fixed_PD','learned_PD') if budget==4 else ('exact_LP',)):
                        w,u=hybrid(x,method,schedules)
                        record(w,u,budget,method)
            f.flush();print('completed environment',env+1,flush=True)
if __name__=='__main__':
    p=ArgumentParser();p.add_argument('repo',type=Path)
    with threadpool_limits(1):run(p.parse_args().repo)
