"""Independent FP64 scorer; never averages fold/target R2s."""
import numpy as np
W=np.array([.1,.1,.1,.2,.5],dtype=np.float64)

def global_r2(y,p):
 y=np.asarray(y,dtype=np.float64);p=np.asarray(p,dtype=np.float64)
 if y.shape!=p.shape or y.ndim!=2 or y.shape[1]!=5 or not np.isfinite(y).all() or not np.isfinite(p).all():raise ValueError('Finite alignedNx5 required.')
 mu=np.sum(y*W)/(len(y)*W.sum());den=np.sum((y-mu)**2*W)
 if den<=0:raise ValueError('Undefined constant global denominator.')
 return float(1-np.sum((y-p)**2*W)/den)

def per_target(y,p):
 y=np.asarray(y,dtype=np.float64);p=np.asarray(p,dtype=np.float64);den=((y-y.mean(0))**2).sum(0)
 return {'r2':np.divide(den-((y-p)**2).sum(0),den,out=np.full(5,np.nan),where=den>0).tolist(),'mae':np.abs(p-y).mean(0).tolist(),'rmse':np.sqrt(((p-y)**2).mean(0)).tolist(),'bias':(p-y).mean(0).tolist()}

def paired_bootstrap(y,control,treatment,groups,draws=10000,seed=42):
 groups=np.asarray(groups);unique=np.unique(groups);indices=[np.flatnonzero(groups==g) for g in unique];rng=np.random.default_rng(seed);deltas=[];invalid=0
 for _ in range(draws):
  idx=np.concatenate([indices[i] for i in rng.integers(len(unique),size=len(unique))])
  try:deltas.append(global_r2(y[idx],treatment[idx])-global_r2(y[idx],control[idx]))
  except ValueError:invalid+=1
 return {'difference':global_r2(y,treatment)-global_r2(y,control),'ci95':np.percentile(deltas,[2.5,97.5]).tolist(),'effective_draws':len(deltas),'invalid_draws':invalid,'date_groups':len(unique),'seed':seed,'conditional_reused_development_only':True}
