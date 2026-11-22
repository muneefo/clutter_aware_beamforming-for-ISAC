from types import SimpleNamespace
import numpy as np
from parta_ao import ConeCoordinates, JointTransmitSDP, pair_sum
from parta_barrier import BarrierSDP
from parta_eigencut import EigenCutSDP
from parta_incremental import IncrementalCutSDP

h = np.array([1, 1j]) / np.sqrt(2)
g = np.array([1, -1j]) / np.sqrt(2)
A = g[None, :, None] * g[None, None, :].conj()
x = SimpleNamespace(N=1, K=1, T=2, P=1, H=h[None, :, None], noise=0.1, g=np.array([[2.]]))
for ca in (False, True):
    layout = ConeCoordinates(2, ca)
    expected = A if ca else A.real
    assert np.allclose(layout.unpack(layout.pack(A)), expected)
p = JointTransmitSDP(x)
s, u, info = p.solve(A, np.zeros_like(A), 0)
q = s + u.sum(axis=1)
print(info)
print(p.residuals(s, u, q))
assert abs(pair_sum(A, q) - .8) < 1e-5
s, u, info = BarrierSDP(p).solve(A, np.zeros_like(A), 0)
print(info)
assert info["status"] == "solved"
assert abs(pair_sum(A, s + u.sum(axis=1)) - .8) < 1e-5

for solver in (EigenCutSDP(p), IncrementalCutSDP(p)):
    s, u, info = solver.solve(A, np.zeros_like(A), 0)
    print(info)
    assert info["status"] == "solved"
    assert abs(pair_sum(A, s + u.sum(axis=1)) - .8) < 1e-5
