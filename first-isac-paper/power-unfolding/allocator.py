"""One fixed-receiver, fixed-proposal power subproblem from the ISAC benchmark."""
from dataclasses import dataclass
import numpy as np
from scipy import sparse
from scipy.optimize import linprog
import reference_beams as ref


def add_fixed_proposals(x, W, A, C, eta, directions, owners):
    """The fixedProposal transmit rule: all four network logits equal zero."""
    for n, Y in enumerate(W):
        H = x.H[n]
        response = H.conj().T @ Y
        k = np.arange(x.K)
        desired = response[k, k].copy()
        response[k, k] = 0
        interference = (abs(response)**2).sum(axis=1) + x.noise
        ratios = abs(desired)**2 / (interference * x.g[n])
        correction = np.zeros_like(Y)
        for j in k:
            gradient = -H[:, j, None] * response[j] / interference[j]
            gradient[:, j] = H[:, j] * desired[j] / max(abs(desired[j])**2, 1e-300)
            correction += gradient / (max(np.linalg.norm(gradient), 1e-300) * max(1, ratios[j])**2)
        target, clutter = A[n] @ Y, eta * C[n] @ Y
        direction = (target-clutter) / max(np.linalg.norm(target)+np.linalg.norm(clutter), 1e-300)
        proposal = Y + .6 * np.sqrt(x.P/x.N) * (direction + .15*correction/x.K)
        V = proposal[:, :x.K]
        norms = np.linalg.norm(V, axis=0)
        assert np.all(norms > 0)
        directions[n] = np.column_stack([directions[n], V/norms])
        owners[n] = np.r_[owners[n], k]


@dataclass
class Problem:
    a: np.ndarray
    c: np.ndarray
    B: np.ndarray
    d: np.ndarray
    zf: np.ndarray
    start: np.ndarray
    scale: float

    def ratio(self, p):
        return np.sum(self.a*p)/(1+np.sum(self.c*p))

    def receive(self, p):
        return (self.B @ p[..., None])[..., 0]

    def transmit(self, y):
        return (self.B.swapaxes(-1,-2) @ y[..., None])[..., 0]

    def feasible_decode(self, raw):
        """Top up each user's zero-forcing column, then mix with the minimum-power anchor."""
        p = raw.copy()
        users = self.d.shape[1]
        p[:, :users] += np.maximum(self.d-self.receive(p), 0)/self.zf
        anchor_mass = np.sum(self.d/self.zf)
        if p.sum() > 1:
            fraction = (1-anchor_mass)/(p.sum()-anchor_mass)
            p *= fraction
            p[:, :users] += (1-fraction)*self.d/self.zf
        return p


def build(x, initial, receiver=None):
    H, noise, directions, owners, W = initial
    if receiver is None:
        Q = W @ W.conj().transpose(0,2,1)
        receiver = ref.fixed_receiver(x,Q,x.est)
    eta,A,C,hot,U = receiver
    add_fixed_proposals(x,W,A,C,eta,directions,owners)
    V, labels = np.asarray(directions), np.asarray(owners)
    P = x.P*(1-1e-8)
    a = np.einsum('ntv,ntv->nv',V.conj(),A@V).real*P/hot
    c = np.maximum(0,np.einsum('ntv,ntv->nv',V.conj(),C@V).real)*P/hot
    scale = a.max()
    gains = abs(H.conj().transpose(0,2,1) @ V)**2
    correct = np.arange(x.K)[None,:,None] == labels[:,None,:]
    B = gains*((1+1/(x.g[:,:,None]*(1+2e-5)))*correct-1)
    # Row equilibration affects the dual coordinates, not the QoS feasible set.
    row_scale = np.max(abs(B),axis=-1)
    B = B / row_scale[:,:,None]
    d = (noise*x.P/P) / row_scale
    zf = np.diagonal(B[:,:,:x.K],axis1=1,axis2=2).copy()
    anchor = np.zeros_like(a)
    anchor[:,:x.K] = d/zf
    assert anchor.sum()<1 and np.all(zf>0)
    off = B[:,:,:x.K].copy()
    k = np.arange(x.K); off[:,k,k]=0
    assert abs(off).max()<1e-12
    anchor[:,x.K] = (1-anchor.sum())/x.N
    return Problem(a/scale,c,B,d,zf,anchor,scale), U


def exact(prob):
    """Charnes–Cooper LP. Recover nonnegative QoS multipliers in our row units."""
    a,c,B,d=prob.a,prob.c,prob.B,prob.d
    row_scale=1/np.sqrt(d)
    blocks = sparse.block_diag(list(-B*row_scale[:,:,None]),format='csr')
    constraints=sparse.vstack([sparse.hstack([blocks,(d*row_scale).reshape(-1,1)]),
                               sparse.csr_matrix(np.r_[np.ones(a.size),-1][None,:])],format='csr')
    res=linprog(-np.r_[a.ravel(),0.],A_ub=constraints,b_ub=np.zeros(d.size+1),
                A_eq=sparse.csr_matrix(np.r_[c.ravel(),1.][None,:]), b_eq=[1.],
                bounds=(0,None),method='highs',
                options={'dual_feasibility_tolerance':1e-9,'primal_feasibility_tolerance':1e-9})
    if not res.success:
        raise RuntimeError(res.message)
    p=(np.maximum(0,res.x[:-1])/res.x[-1]).reshape(a.shape)
    return prob.feasible_decode(p), -res.ineqlin.marginals[:-1].reshape(d.shape)*row_scale


def project_simplex(v):
    """Euclidean projection onto nonnegative unit-sum fractions."""
    ordered = np.sort(v,axis=None)[::-1]
    shift = (np.cumsum(ordered)-1)/np.arange(1,ordered.size+1)
    active = np.flatnonzero(ordered>shift)  # active prefix
    return np.maximum(0,v-shift[active[-1]])


def unroll(prob, schedule):
    p = prob.start.copy()
    best = p.copy(); eta = prob.ratio(p)
    dual = np.zeros_like(prob.d)
    for primal_step,dual_step in schedule:
        # Gradient of the fractional objective's Dinkelbach residual at current best.
        direction = prob.a - eta*prob.c + prob.transmit(dual)
        candidate = project_simplex(p+primal_step*direction)
        dual = np.maximum(0,dual+dual_step*(prob.d-prob.receive(2*candidate-p)))
        p = candidate
        feasible = prob.feasible_decode(p)
        score = prob.ratio(feasible)
        if score>eta:
            eta=score; best=feasible
    return best,dual


def gap_bound(prob,p,y):
    """A posteriori fixed-bank upper bound, using any nonnegative QoS multipliers."""
    eta=prob.ratio(p)
    prices=prob.a-eta*prob.c+prob.transmit(y)
    residual=max(0.,max(0.,prices.max())-np.sum(prob.d*y)-eta)
    return 10*np.log10(1+residual/eta)
