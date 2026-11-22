"""Nominal reciprocal RIS-ISAC prototype (complex128). No robust guarantee."""
from dataclasses import dataclass
import numpy as np
@dataclass
class Config:
 N:int=6
 M:int=12
 K:int=2
 L:int=3
 wavelength:float=.1
 aperture_bs:float=.35
 aperture_ris:float=.55
 power:float=1.
 noise_comm:float=1e-6
 noise_sense:float=1e-9
 gamma:float=2.
 blockage_amplitude:float=.35
 penalty:float=2.

def hop(a,b,lam):
 r=np.linalg.norm(a[...,None,:]-b,axis=-1)
 return lam/(4*np.pi*r)*np.exp(-2j*np.pi*r/lam)

def generate(count,seed,cfg=Config(),shift=False):
 rng=np.random.default_rng(seed)
 bs=np.stack([np.zeros(cfg.N),np.linspace(-cfg.aperture_bs/2,cfg.aperture_bs/2,cfg.N)],-1)
 ris=np.stack([np.full(cfg.M,.65),np.linspace(.15,.15+cfg.aperture_ris,cfg.M)],-1)
 G=hop(ris,bs,cfg.wavelength)
 users=np.stack([rng.uniform(1,1.9,(count,cfg.K)),rng.uniform(-.8,.15,(count,cfg.K))],-1)
 target=np.stack([rng.uniform(1,1.7,count),rng.uniform(.75,1.2,count)],-1)[:,None,:]
 clutter=target+rng.normal(0,.12 if shift else .22,(count,cfg.L,2));clutter[:,:,0]=np.maximum(clutter[:,:,0],.85)
 points=np.concatenate([users,target,clutter],1)
 direct=cfg.blockage_amplitude*hop(points,bs,cfg.wavelength);rislink=hop(points,ris,cfg.wavelength)
 rcs=np.concatenate([np.zeros((count,cfg.K)),np.full((count,1),.05),rng.uniform(.03,.15,(count,cfg.L))*(3 if shift else 1)],1)
 beta=4*np.pi*rcs/cfg.wavelength**2
 features=np.stack([np.ones(count),(target[:,0,0]-1.35)/.35,(target[:,0,1]-.975)/.225],1)
 beam=direct.conj()/np.linalg.norm(direct,axis=-1,keepdims=True)
 field=np.einsum('mn,bjn->bjm',G,beam)
 dictionary=np.exp(-1j*np.angle(rislink*field))[:,[cfg.K,0,1],:]
 return dict(direct=direct,rislink=rislink,G=G,beta=beta,features=features,dictionary=dictionary,points=points,cfg=cfg)

def subset(d,idx):
 return {k:(v[idx] if isinstance(v,np.ndarray) and k!='G' else v) for k,v in d.items()}

def effective(d,theta,derivatives=False):
 w=d['rislink']*np.exp(1j*theta[:,None,:]);v=d['direct']+np.einsum('bjm,mn->bjn',w,d['G'])
 return (v,1j*w[:,:,:,None]*d['G'][None,None,:,:]) if derivatives else v

def receiver(d,W,theta):
 c=d['cfg'];v=effective(d,theta);z=v@W;vc=v[:,c.K+1:]
 weights=d['beta'][:,c.K+1:]*np.sum(abs(z[:,c.K+1:])**2,-1)
 R=np.einsum('bl,bln,blm->bnm',weights,vc,vc.conj())+c.noise_sense*np.eye(c.N)[None]
 u=np.linalg.solve(R,v[:,c.K,:,None])[...,0]
 return u/np.linalg.norm(u,axis=-1,keepdims=True)

def evaluate(d,W,theta,u=None,gradient=False):
 c=d['cfg'];u=receiver(d,W,theta) if u is None else u
 if gradient:v,dv=effective(d,theta,True)
 else:v=effective(d,theta)
 z=v@W;q=abs(z)**2;r=np.einsum('bn,bjn->bj',u.conj(),v)
 p=d['beta']*abs(r)**2*np.sum(q,-1);S=p[:,c.K];D=np.sum(p[:,c.K+1:],-1)+c.noise_sense*np.sum(abs(u)**2,-1)
 signal=np.diagonal(q[:,:c.K,:],axis1=1,axis2=2);I=np.sum(q[:,:c.K,:],-1)-signal+c.noise_comm
 sinr=signal/I;delta=np.maximum(0,np.log(c.gamma)-np.log(np.maximum(sinr,1e-30)))
 loss=-np.log(np.maximum(S/D,1e-30))+c.penalty*np.sum(delta**2,-1)
 ans=dict(loss=loss,scnr=S/D,sinr=sinr,u=u,power=np.sum(abs(W)**2,axis=(1,2)))
 if not gradient:return ans
 # Wirtinger W* derivative: Cartesian gradient is [2 Re(g),2 Im(g)].
 BjW=d['beta'][:,:,None,None]*abs(r)[:,:,None,None]**2*v.conj()[:,:,:,None]*z[:,:,None,:]
 gW=-BjW[:,c.K]/S[:,None,None]+np.sum(BjW[:,c.K+1:],1)/D[:,None,None]
 HvW=v[:,:c.K].conj()[:,:,:,None]*z[:,:c.K,None,:]
 coefficients=-np.ones((len(W),c.K,c.K))/I[:,:,None]
 for k in range(c.K):coefficients[:,k,k]=1/signal[:,k]
 gW-=2*c.penalty*np.einsum('bk,bkp,bknp->bnp',delta,coefficients,HvW)
 dz=np.einsum('bjmn,bnk->bjmk',dv,W);dr=np.einsum('bn,bjmn->bjm',u.conj(),dv)
 dq=2*np.real(z.conj()[:,:,None,:]*dz)
 dp=d['beta'][:,:,None]*(2*np.real(r.conj()[:,:,None]*dr)*np.sum(q,-1)[:,:,None]+abs(r)[:,:,None]**2*np.sum(dq,-1))
 gt=-dp[:,c.K]/S[:,None]+np.sum(dp[:,c.K+1:],1)/D[:,None]
 for k in range(c.K):
  ds=dq[:,k,:,k];di=np.sum(dq[:,k],-1)-ds
  gt-=2*c.penalty*delta[:,k,None]*(ds/signal[:,k,None]-di/I[:,k,None])
 ans.update(gW=gW,gtheta=gt);return ans

def repair(d,W,theta):
 """Exact fixed-direction power allocation; reject infeasible directions."""
 c=d['cfg'];V=effective(d,theta)[:,:c.K];beams=W/np.maximum(np.linalg.norm(W,axis=1,keepdims=True),1e-30)
 gains=abs(V@beams)**2;diag=np.diagonal(gains,axis1=1,axis2=2)
 F=c.gamma*gains/np.maximum(diag[:,:,None],1e-30)
 for k in range(c.K):F[:,k,k]=0
 b=c.gamma*c.noise_comm/np.maximum(diag,1e-30)
 good=np.all(diag>1e-25,axis=1)&(np.max(abs(np.linalg.eigvals(F)),axis=-1)<1-1e-10)
 mat=np.eye(c.K)[None]-F;mat[~good]=np.eye(c.K)
 p=np.linalg.solve(mat,b[:,:,None])[...,0]
 good&=np.all(p>0,axis=1)&(np.sum(p,axis=1)<=c.power*(1+1e-10))
 p=np.maximum(p,1e-30);p*= (c.power/np.sum(p,axis=1))[:,None]
 result=beams*np.sqrt(p)[:,None,:]
 actual=abs(V@result)**2;s=np.diagonal(actual,axis1=1,axis2=2)
 good&=np.all(s/(np.sum(actual,-1)-s+c.noise_comm)>=c.gamma*(1-1e-8),axis=1)
 return result,good

def initialize(d,kind='zero',init_params=None):
 c=d['cfg'];B=len(d['direct'])
 if kind=='zero':theta=np.zeros((B,c.M))
 elif kind=='target':theta=np.angle(d['dictionary'][:,0])
 elif kind=='mixture':theta=np.angle(np.sum(d['dictionary'],axis=1))
 elif kind=='learned':
  logits=d['features']@init_params.reshape(3,3);logits-=np.max(logits,-1,keepdims=True)
  w=np.exp(logits);w/=np.sum(w,-1,keepdims=True)
  theta=np.angle(np.sum(w[:,:,None]*d['dictionary'],axis=1))
 else:raise ValueError(kind)
 V=effective(d,theta)[:,:c.K]
 W=V.conj().transpose(0,2,1)@np.linalg.inv(V@V.conj().transpose(0,2,1))
 W,good=repair(d,W,theta);return W,theta,good

def decode(params,depth):
 sig=1/(1+np.exp(-np.clip(params[:2*depth],-20,20)))
 return .005+.395*sig[:depth],.005+.795*sig[depth:2*depth]

def rollout(d,depth=8,params=None,init='zero',learn_init=False,alpha=.05,phase_step=.1,backtrack=False,history=False):
 c=d['cfg'];B=len(d['direct']);ip=params[2*depth:] if learn_init else None
 W,theta,good0=initialize(d,'learned' if learn_init else init,ip)
 best=evaluate(d,W,theta);bestW=W.copy();besttheta=theta.copy()
 if not np.any(good0):
  best.update(W=W,theta=theta,feasible=good0,initial_good=good0,rejected=np.zeros(B,int))
  if history:best['trace']=np.tile(10*np.log10(best['scnr']),(depth,1))
  return best
 aa,bb=(np.full(depth,alpha),np.full(depth,phase_step)) if params is None else decode(params,depth)
 trace=[];rejected=np.zeros(B,int)
 for t in range(depth):
  out=evaluate(d,W,theta,gradient=True)
  gw=out['gW']/np.maximum(np.linalg.norm(out['gW'],axis=(1,2),keepdims=True),1e-20)
  gt=out['gtheta']/np.maximum(np.sqrt(np.mean(out['gtheta']**2,-1,keepdims=True)),1e-20)
  oldW=W.copy();oldtheta=theta.copy();accepted=np.zeros(B,bool)
  for bt in range(7 if backtrack else 1):
   scale=.5**bt;trialW=oldW-aa[t]*scale*gw
   trialW/=np.maximum(1,np.linalg.norm(trialW,axis=(1,2),keepdims=True)/np.sqrt(c.power))
   tt=oldtheta-bb[t]*scale*gt;candidate,ok=repair(d,trialW,tt);ev=evaluate(d,candidate,tt)
   ok&=good0
   if backtrack:ok&=ev['scnr']>=out['scnr']*(1-1e-12)
   take=ok&~accepted;W[take]=candidate[take];theta[take]=tt[take]
   improve=take&(ev['scnr']>best['scnr']);bestW[improve]=candidate[improve];besttheta[improve]=tt[improve];best['scnr'][improve]=ev['scnr'][improve]
   accepted|=take
   if np.all(accepted|~good0):break
  rejected+=~accepted
  if history:trace.append(10*np.log10(best['scnr']))
 end=evaluate(d,bestW,besttheta)
 feasible=good0&np.all(end['sinr']>=c.gamma*(1-1e-8),axis=1)&(end['power']<=c.power*(1+1e-8))
 end.update(W=bestW,theta=besttheta,feasible=feasible,initial_good=good0,rejected=rejected)
 if history:end['trace']=np.array(trace)
 return end
