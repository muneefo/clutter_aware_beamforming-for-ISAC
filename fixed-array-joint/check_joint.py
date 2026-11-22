"""Independent projection, receiver, gradient and joint support-function checks."""
import numpy as np
from scipy.optimize import minimize
from joint_solver import project, receiver, sample, polygon
import reference as m

def check_projection():
    # Independent SLSQP formulation in real coordinates on a small fixture.
    rng=np.random.default_rng(82)
    h=m.cn(rng,(2,4))*.003
    raw=m.cn(rng,(4,2))
    result=project(raw,h)
    n=raw.size
    def unpack(x):
        return (x[:n]+1j*x[n:]).reshape(raw.shape)
    def pack(w):
        return np.r_[w.real.ravel(),w.imag.ravel()]
    def inequalities(x):
        w=unpack(x)
        y=h@w/np.sqrt(m.USER_NOISE)
        slack=y.diagonal().real/np.sqrt(m.GAMMA)-np.sqrt(1+abs(y[[0,1],[1,0]])**2)
        return np.r_[1-np.linalg.norm(w)**2,slack]
    target=pack(raw)
    p=minimize(lambda x:.5*np.sum((x-target)**2),target,jac=lambda x:x-target,
               constraints=[dict(type='ineq',fun=inequalities),
                            dict(type='eq',fun=lambda x:np.diag(h@unpack(x)).imag/.001)],
               method='SLSQP',options={'ftol':1e-12,'maxiter':400})
    assert p.success,p.message
    np.testing.assert_allclose(pack(result),p.x,rtol=0,atol=3e-6)

def check_receiver():
    # Symmetric adversaries on an edge. The least-favourable point is interior
    # to the edge and the box corner (1,1) is outside the simplex.
    t=np.ones((1,3),complex)
    r=np.array([[1,1,0],[1,0,1]],complex)*np.sqrt(m.RX_NOISE)
    W=np.ones((1,1),complex)
    val,u,_=receiver(t,r,W,np.array([[0.,0.],[1.,0.],[0.,1.]]))
    np.testing.assert_allclose(val,4/3)
    np.testing.assert_allclose(abs(u),np.ones(2)/np.sqrt(2))
    case,_,_,_=sample(1002);t,r,h,V,_=case
    W=m.start(t,h,.44)
    val,u,g=receiver(t,r,W,V)
    direction=m.unit(m.cn(np.random.default_rng(23),W.shape))
    eps=1e-6
    fd=(receiver(t,r,W+eps*direction,V)[0]-receiver(t,r,W-eps*direction,V)[0])/(2*eps)
    np.testing.assert_allclose(fd,2*np.vdot(g,direction).real,rtol=1e-5)
    assert not len(polygon(np.array([[1,0],[-1,0],[0,1],[0,-1]]),np.array([-1,0,-1,0])))
if __name__=='__main__':
    check_projection()
    check_receiver()
    print('Projection oracle, interior-edge optimum, finite differences and empty-set checks passed.')
