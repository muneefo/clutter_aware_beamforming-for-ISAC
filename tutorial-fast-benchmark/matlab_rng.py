"""MT19937 normal generator cross-checked against MATLAB default normal values."""
import numpy as np
from math import exp,log,sqrt,erfc,pi
R=3.6541528853610088
area=R*exp(-R*R/2)+sqrt(pi/2)*erfc(R/sqrt(2))
edges=np.zeros(257)
edges[255:]=[R,area/exp(-R*R/2)]
for j in range(254,0,-1):
    x=edges[j+1]; edges[j]=sqrt(-2*log(area/x+exp(-.5*x*x)))
pdf=np.exp(-.5*edges**2)
class MatlabTwister:
    def __init__(self,seed):
        self.mt=np.random.RandomState(seed or 5489); self.buffer=iter(())
    def word(self):
        try: return next(self.buffer)
        except StopIteration:
            self.buffer=iter(self.mt.randint(0,2**32,16384,dtype=np.uint32).tolist())
            return next(self.buffer)
    def uniform(self):
        return ((self.word()>>5)*67108864+(self.word()>>6))*2**-53
    def normal(self):
        while True:
            w=self.word(); z=self.word(); j=(z>>24)+1
            v=(((w>>3)*16777216+(z & 16777215))*2**-52-1)*edges[j]
            if abs(v)<=edges[j-1]: return v
            if j<256:
                if pdf[j]+self.uniform()*(pdf[j-1]-pdf[j])<exp(-.5*v*v): return v
            else:
                while True:
                    x=-log(self.uniform())/R
                    if -2*log(self.uniform())>x*x: return np.copysign(R+x,v)
    def complex(self,shape):
        m=int(np.prod(shape))
        def arr(): return np.fromiter((self.normal() for _ in range(m)),float,count=m).reshape(shape,order='F')
        return (arr()+1j*arr())/sqrt(2)
if __name__=='__main__':
    t=MatlabTwister(0);print([t.normal() for _ in range(10)])
