"""Idempotent FP8 conversion for final study; FP32 masters are never cast."""
import torch
from torch import nn

def configure(module,device):
 if device.type!='cuda' or 'rtx pro 6000' not in torch.cuda.get_device_name(device).lower() or torch.cuda.get_device_capability(device)[0]<10:raise RuntimeError('Actual RTXPRO6000 Blackwell required.')
 from torchao.float8 import Float8LinearConfig,convert_to_float8_training
 from torchao.float8.float8_linear import Float8Linear
 def eligible(layer,name):
  return isinstance(layer,nn.Linear) and 'teacher' not in name and min(layer.in_features,layer.out_features)>=128 and layer.in_features%16==0 and layer.out_features%16==0
 def missing(layer,name):return eligible(layer,name) and not isinstance(layer,Float8Linear)
 if any(missing(m,n) for n,m in module.named_modules()):convert_to_float8_training(module,module_filter_fn=missing,config=Float8LinearConfig(pad_inner_dim=True))
 count=sum(isinstance(m,Float8Linear) for m in module.modules())
 if count==0 or any(missing(m,n) for n,m in module.named_modules()):raise RuntimeError('Eligible FP8 conversion failed.')
 if any(p.is_floating_point() and p.dtype!=torch.float32 for p in module.parameters()):raise RuntimeError('FP32 master parameters required.')
 return {'requested':'fp8','active':'fp8_linear_bf16_other','float8_layers':count,'master_weights':'FP32','conversion':'idempotent'}
