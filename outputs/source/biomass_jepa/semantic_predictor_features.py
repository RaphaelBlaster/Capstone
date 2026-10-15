"""Fixed four42-token semantic targets,88-token masked context. No half-mask path."""
import torch
from torch import nn
from .official_ijepa import validate_ssl_masks

def semantic_masks(batch,device):
 targets=[]
 for rows,cols in [(range(0,6),range(0,7)),(range(0,6),range(9,16)),(range(10,16),range(0,7)),(range(10,16),range(9,16))]:
  targets.append(torch.tensor([r*16+c for r in rows for c in cols],device=device).expand(batch,-1))
 union=set(torch.cat([x[0] for x in targets]).tolist())
 context=[torch.tensor(sorted(set(range(256))-union),device=device).expand(batch,-1)]
 validate_ssl_masks(context,targets,batch,256)
 return context,targets

def pool_predictions(predictions,batch,views):
 if predictions.ndim!=3 or predictions.shape[:2]!=(4*batch*views,42):raise ValueError('Wrong target-major predictor output.')
 return predictions.reshape(4,batch*views,42,-1).mean(dim=(0,2)).reshape(batch,views,-1)

class SemanticPredictorReadout(nn.Module):
 def __init__(self,encoder,predictor):
  super().__init__();self.encoder=encoder;self.predictor=predictor;self.requires_grad_(False);self.eval()
 @torch.no_grad()
 def forward(self,views):
  b,v,c,h,w=views.shape
  if v!=2 or (h,w)!=(224,224):raise ValueError('Two complete224views required.')
  images=views.reshape(b*v,c,h,w);context,target=semantic_masks(b*v,images.device)
  hidden=self.encoder(images,context)
  return pool_predictions(self.predictor(hidden,context,target),b,v)
